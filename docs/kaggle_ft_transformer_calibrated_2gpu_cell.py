# Paste this whole cell into Kaggle, edit DATA_DIR, then run.
# It writes a standalone script and launches 2-GPU DDP training with torchrun.
#
# Experiment: ft_transformer_calibrated_focal_swa
# Hypothesis: a calibrated FT-Transformer with class-balanced focal loss,
# stronger weight decay, and late-epoch SWA can add cleaner residual
# probabilities to the logistic meta-stack than the earlier FT run.

from pathlib import Path
import subprocess
import sys

DATA_DIR = "/kaggle/input/YOUR_DATASET_SLUG"  # must contain train.csv, test.csv, sample_submission.csv
OUT_DIR = "/kaggle/working"
SCRIPT_PATH = Path(OUT_DIR) / "ft_transformer_calibrated_ddp_train.py"

SCRIPT = r'''
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


SEED = 42
N_FOLDS = 5
CLASS_LABELS = ["GALAXY", "QSO", "STAR"]
PROBA_COLUMNS = [f"proba_{label}" for label in CLASS_LABELS]
TARGET_COLUMN = "class"
ID_COLUMN = "id"
FOLD_COLUMN = "fold"
BASE_NUMERIC_FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_FEATURES = ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z", "u_z"]
MAGNITUDE_BANDS = ["u", "g", "r", "i", "z"]
MAGNITUDE_SUMMARY_FEATURES = ["mag_mean", "mag_std", "mag_min", "mag_max", "mag_range"]
SPATIAL_FEATURES = ["alpha_sin", "alpha_cos", "delta_sin", "delta_cos", "sky_x", "sky_y", "sky_z"]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + COLOR_FEATURES + MAGNITUDE_SUMMARY_FEATURES + SPATIAL_FEATURES
CATEGORICAL_FEATURES = ["spectral_type", "galaxy_population", "spectral_population"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class FTTransformer(nn.Module):
    """FT-Transformer-style model for mixed tabular multiclass classification."""

    def __init__(
        self,
        n_numeric: int,
        cardinalities: list[int],
        token_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, token_dim))
        self.numeric_bias = nn.Parameter(torch.empty(n_numeric, token_dim))
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, token_dim) for cardinality in cardinalities]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, len(CLASS_LABELS)),
        )
        nn.init.xavier_uniform_(self.numeric_weight)
        nn.init.zeros_(self.numeric_bias)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight.unsqueeze(0) + self.numeric_bias.unsqueeze(0)
        categorical_tokens = [
            embedding(categorical[:, index]).unsqueeze(1)
            for index, embedding in enumerate(self.categorical_embeddings)
        ]
        cls = self.cls_token.expand(numeric.shape[0], -1, -1)
        tokens = torch.cat([cls, numeric_tokens, *categorical_tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss using effective-number class weights."""

    def __init__(self, class_counts: np.ndarray, gamma: float, beta: float) -> None:
        super().__init__()
        counts = np.maximum(class_counts.astype(np.float32), 1.0)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.mean()
        self.register_buffer("weights", torch.from_numpy(weights.astype(np.float32)))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        row_index = torch.arange(target.shape[0], device=target.device)
        target_log_probs = log_probs[row_index, target]
        target_probs = probs[row_index, target]
        target_weights = self.weights[target]
        focal_factor = torch.pow(1.0 - target_probs, self.gamma)
        return (-target_weights * focal_factor * target_log_probs).mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--model-name", type=str, default="ft_transformer_calibrated_focal_swa")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--infer-batch-size", type=int, default=8192)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--effective-number-beta", type=float, default=0.9999)
    parser.add_argument("--swa-start-epoch", type=int, default=11)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    set_seed(SEED + rank)

    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    train = create_folds(make_features(train_raw))
    test = make_features(test_raw)
    y = encode_labels(train[TARGET_COLUMN].to_numpy())
    oof_proba = np.zeros((len(train), len(CLASS_LABELS)), dtype=np.float32) if rank == 0 else None
    test_proba = np.zeros((len(test), len(CLASS_LABELS)), dtype=np.float32) if rank == 0 else None
    fold_scores: list[float] = []
    fold_losses: list[float] = []

    if rank == 0:
        print(
            "Experiment settings: "
            f"loss=class_balanced_focal gamma={args.focal_gamma} beta={args.effective_number_beta}, "
            f"weight_decay={args.weight_decay}, swa_start_epoch={args.swa_start_epoch}",
            flush=True,
        )

    for fold in range(N_FOLDS):
        result = train_fold(args, fold, train, test, y, local_rank, rank, world_size)
        if rank == 0:
            valid_idx = (train[FOLD_COLUMN] == fold).to_numpy()
            oof_proba[valid_idx] = result["valid_proba"]
            test_proba += result["test_proba"] / N_FOLDS
            fold_scores.append(float(result["score"]))
            fold_losses.append(float(result["loss"]))
            print(
                f"{args.model_name} fold {fold}: "
                f"balanced_accuracy={result['score']:.6f}, log_loss={result['loss']:.6f}, "
                f"selected={result['selected_model']}",
                flush=True,
            )
        dist.barrier()

    if rank == 0:
        write_outputs(args, train, test, sample, oof_proba, test_proba, fold_scores, fold_losses)
    dist.destroy_process_group()


def train_fold(
    args: argparse.Namespace,
    fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    local_rank: int,
    rank: int,
    world_size: int,
) -> dict[str, Any] | None:
    train_idx = (train[FOLD_COLUMN] != fold).to_numpy()
    valid_idx = (train[FOLD_COLUMN] == fold).to_numpy()
    encoding = fit_encoding(train.loc[train_idx, FEATURE_COLUMNS])
    x_train_num, x_train_cat = transform_features(train.loc[train_idx, FEATURE_COLUMNS], encoding)
    x_valid_num, x_valid_cat = transform_features(train.loc[valid_idx, FEATURE_COLUMNS], encoding)
    x_test_num, x_test_cat = transform_features(test[FEATURE_COLUMNS], encoding)
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    train_dataset = TensorDataset(
        torch.from_numpy(x_train_num),
        torch.from_numpy(x_train_cat),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = FTTransformer(
        n_numeric=len(NUMERIC_FEATURES),
        cardinalities=encoding["cardinalities"],
        token_dim=args.token_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).cuda(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    swa_model = AveragedModel(model.module)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = ClassBalancedFocalLoss(
        class_counts=np.bincount(y_train, minlength=len(CLASS_LABELS)),
        gamma=args.focal_gamma,
        beta=args.effective_number_beta,
    ).cuda(local_rank)
    scaler = torch.amp.GradScaler("cuda")
    best_score = -np.inf
    best_state = None
    selected_model = "base"

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for numeric_batch, categorical_batch, target_batch in loader:
            numeric_batch = numeric_batch.cuda(local_rank, non_blocking=True)
            categorical_batch = categorical_batch.cuda(local_rank, non_blocking=True)
            target_batch = target_batch.cuda(local_rank, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(numeric_batch, categorical_batch), target_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch + 1 >= args.swa_start_epoch:
            swa_model.update_parameters(model.module)
        dist.barrier()

        if rank == 0:
            valid_proba = predict_proba(model.module, x_valid_num, x_valid_cat, args.infer_batch_size, local_rank)
            score = balanced_accuracy_score(y_valid, valid_proba.argmax(axis=1))
            epoch_label = "base"
            if score > best_score:
                best_score = float(score)
                best_state = {key: value.detach().cpu().clone() for key, value in model.module.state_dict().items()}
                selected_model = epoch_label

            if epoch + 1 >= args.swa_start_epoch:
                swa_proba = predict_proba(swa_model.module, x_valid_num, x_valid_cat, args.infer_batch_size, local_rank)
                swa_score = balanced_accuracy_score(y_valid, swa_proba.argmax(axis=1))
                if swa_score > best_score:
                    best_score = float(swa_score)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in swa_model.module.state_dict().items()
                    }
                    selected_model = "swa"
                print(
                    f"{args.model_name} fold {fold} epoch {epoch + 1}: "
                    f"base_balanced_accuracy={score:.6f}, swa_balanced_accuracy={swa_score:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"{args.model_name} fold {fold} epoch {epoch + 1}: "
                    f"base_balanced_accuracy={score:.6f}",
                    flush=True,
                )
        dist.barrier()

    if rank != 0:
        return None
    if best_state is not None:
        model.module.load_state_dict(best_state)
    valid_proba = predict_proba(model.module, x_valid_num, x_valid_cat, args.infer_batch_size, local_rank)
    test_proba = predict_proba(model.module, x_test_num, x_test_cat, args.infer_batch_size, local_rank)
    score = balanced_accuracy_score(y_valid, valid_proba.argmax(axis=1))
    loss_value = log_loss(y_valid, valid_proba, labels=np.arange(len(CLASS_LABELS)))
    model_dir = args.out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.module.state_dict(), "encoding": encoding, "args": vars(args)},
        model_dir / f"{args.model_name}_fold_{fold}.pt",
    )
    return {
        "valid_proba": valid_proba,
        "test_proba": test_proba,
        "score": score,
        "loss": loss_value,
        "selected_model": selected_model,
    }


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    featured = data.copy()
    featured["u_g"] = featured["u"] - featured["g"]
    featured["g_r"] = featured["g"] - featured["r"]
    featured["r_i"] = featured["r"] - featured["i"]
    featured["i_z"] = featured["i"] - featured["z"]
    featured["u_r"] = featured["u"] - featured["r"]
    featured["g_i"] = featured["g"] - featured["i"]
    featured["r_z"] = featured["r"] - featured["z"]
    featured["u_z"] = featured["u"] - featured["z"]
    magnitude_bands = featured[MAGNITUDE_BANDS]
    featured["mag_mean"] = magnitude_bands.mean(axis=1)
    featured["mag_std"] = magnitude_bands.std(axis=1)
    featured["mag_min"] = magnitude_bands.min(axis=1)
    featured["mag_max"] = magnitude_bands.max(axis=1)
    featured["mag_range"] = featured["mag_max"] - featured["mag_min"]
    featured["spectral_population"] = featured["spectral_type"].astype(str) + "_" + featured["galaxy_population"].astype(str)
    alpha_rad = np.deg2rad(featured["alpha"])
    delta_rad = np.deg2rad(featured["delta"])
    featured["alpha_sin"] = np.sin(alpha_rad)
    featured["alpha_cos"] = np.cos(alpha_rad)
    featured["delta_sin"] = np.sin(delta_rad)
    featured["delta_cos"] = np.cos(delta_rad)
    featured["sky_x"] = featured["delta_cos"] * featured["alpha_cos"]
    featured["sky_y"] = featured["delta_cos"] * featured["alpha_sin"]
    featured["sky_z"] = featured["delta_sin"]
    keep_columns = [ID_COLUMN, *FEATURE_COLUMNS]
    if TARGET_COLUMN in featured.columns:
        keep_columns.append(TARGET_COLUMN)
    return featured.loc[:, keep_columns].copy()


def create_folds(data: pd.DataFrame) -> pd.DataFrame:
    folded = data.copy()
    folded[FOLD_COLUMN] = -1
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_idx) in enumerate(splitter.split(folded[FEATURE_COLUMNS], folded[TARGET_COLUMN])):
        folded.loc[valid_idx, FOLD_COLUMN] = fold
    return folded


def fit_encoding(data: pd.DataFrame) -> dict[str, Any]:
    scaler = StandardScaler()
    scaler.fit(data[NUMERIC_FEATURES])
    category_maps = {}
    cardinalities = []
    for column in CATEGORICAL_FEATURES:
        values = sorted(data[column].astype(str).unique().tolist())
        mapping = {value: index + 1 for index, value in enumerate(values)}
        category_maps[column] = mapping
        cardinalities.append(len(mapping) + 1)
    return {"scaler": scaler, "category_maps": category_maps, "cardinalities": cardinalities}


def transform_features(data: pd.DataFrame, encoding: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    numeric = encoding["scaler"].transform(data[NUMERIC_FEATURES]).astype(np.float32)
    categorical_columns = []
    for column in CATEGORICAL_FEATURES:
        mapping = encoding["category_maps"][column]
        encoded = data[column].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
        categorical_columns.append(encoded)
    return numeric, np.column_stack(categorical_columns).astype(np.int64)


def predict_proba(
    model: FTTransformer,
    numeric: np.ndarray,
    categorical: np.ndarray,
    batch_size: int,
    local_rank: int,
) -> np.ndarray:
    model.eval()
    probabilities = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(numeric), torch.from_numpy(categorical)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    with torch.no_grad():
        for numeric_batch, categorical_batch in loader:
            numeric_batch = numeric_batch.cuda(local_rank, non_blocking=True)
            categorical_batch = categorical_batch.cuda(local_rank, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(numeric_batch, categorical_batch)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.vstack(probabilities).astype(np.float32)


def write_outputs(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    oof_proba: np.ndarray,
    test_proba: np.ndarray,
    fold_scores: list[float],
    fold_losses: list[float],
) -> None:
    ensemble_dir = args.out_dir / "data" / "ensemble"
    oof_dir = ensemble_dir / "oof"
    test_dir = ensemble_dir / "test_proba"
    submissions_dir = args.out_dir / "submissions"
    oof_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    oof = train[[ID_COLUMN, FOLD_COLUMN, TARGET_COLUMN]].rename(columns={ID_COLUMN: "id", FOLD_COLUMN: "fold", TARGET_COLUMN: "y_true"}).copy()
    test_output = test[[ID_COLUMN]].rename(columns={ID_COLUMN: "id"}).copy()
    for index, column in enumerate(PROBA_COLUMNS):
        oof[column] = oof_proba[:, index]
        test_output[column] = test_proba[:, index]
    oof_path = oof_dir / f"{args.model_name}_oof.csv"
    test_path = test_dir / f"{args.model_name}_test_proba.csv"
    oof.to_csv(oof_path, index=False)
    test_output.to_csv(test_path, index=False)

    labels = np.asarray(CLASS_LABELS, dtype=object)
    submission = sample.copy()
    submission[TARGET_COLUMN] = labels[test_proba.argmax(axis=1)]
    submission_path = submissions_dir / f"submission_{args.model_name}.csv"
    submission.to_csv(submission_path, index=False)

    score = balanced_accuracy_score(oof["y_true"], labels[oof_proba.argmax(axis=1)])
    print(f"Mean fold balanced accuracy: {np.mean(fold_scores):.8f}", flush=True)
    print(f"OOF balanced accuracy: {score:.8f}", flush=True)
    print(f"Mean fold log loss: {np.mean(fold_losses):.8f}", flush=True)
    print(f"OOF artifact: {oof_path}", flush=True)
    print(f"Test probability artifact: {test_path}", flush=True)
    print(f"Submission: {submission_path}", flush=True)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    return np.asarray([label_to_index[label] for label in labels], dtype=np.int64)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    main()
'''

SCRIPT_PATH.write_text(SCRIPT)
cmd = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nproc_per_node=2",
    str(SCRIPT_PATH),
    "--data-dir",
    DATA_DIR,
    "--out-dir",
    OUT_DIR,
    "--model-name",
    "ft_transformer_calibrated_focal_swa",
    "--epochs",
    "16",
    "--batch-size",
    "4096",
    "--infer-batch-size",
    "8192",
    "--token-dim",
    "64",
    "--n-heads",
    "8",
    "--n-layers",
    "2",
    "--dropout",
    "0.12",
    "--learning-rate",
    "0.0008",
    "--weight-decay",
    "0.0004",
    "--focal-gamma",
    "1.5",
    "--effective-number-beta",
    "0.9999",
    "--swa-start-epoch",
    "11",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
