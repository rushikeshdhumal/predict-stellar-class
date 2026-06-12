# Paste this whole cell into Kaggle, edit DATA_DIR, then run.
# It writes a standalone script and trains TabNet folds in parallel across 2 GPUs.

from pathlib import Path
import subprocess
import sys

DATA_DIR = "/kaggle/input/YOUR_DATASET_SLUG"  # must contain train.csv, test.csv, sample_submission.csv
OUT_DIR = "/kaggle/working"
MODEL_NAME = "tabnet_2gpu_diversity"
SCRIPT_PATH = Path(OUT_DIR) / "tabnet_2gpu_train.py"

try:
    import pytorch_tabnet  # noqa: F401
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-tabnet"], check=True)

SCRIPT = r'''
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["orchestrate", "worker"], default="orchestrate")
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--model-name", type=str, default="tabnet_2gpu_diversity")
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--virtual-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-d", type=int, default=16)
    parser.add_argument("--n-a", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--cat-emb-dim", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "worker":
        run_worker(args)
    else:
        run_orchestrator(args)


def run_orchestrator(args: argparse.Namespace) -> None:
    """Launch fold workers across available GPUs, then aggregate artifacts."""
    partial_dir = args.out_dir / "tabnet_partials" / args.model_name
    partial_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = list(range(max(1, args.parallel_workers)))
    running: list[tuple[int, int, subprocess.Popen[str]]] = []
    pending_folds = list(range(N_FOLDS))

    while pending_folds or running:
        while pending_folds and len(running) < args.parallel_workers:
            fold = pending_folds.pop(0)
            gpu_id = gpu_ids[len(running) % len(gpu_ids)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--fold",
                str(fold),
                "--data-dir",
                str(args.data_dir),
                "--out-dir",
                str(args.out_dir),
                "--model-name",
                args.model_name,
                "--max-epochs",
                str(args.max_epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--virtual-batch-size",
                str(args.virtual_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--n-d",
                str(args.n_d),
                "--n-a",
                str(args.n_a),
                "--n-steps",
                str(args.n_steps),
                "--gamma",
                str(args.gamma),
                "--cat-emb-dim",
                str(args.cat_emb_dim),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
            ]
            print(f"Launching fold {fold} on GPU {gpu_id}: {' '.join(cmd)}", flush=True)
            running.append((fold, gpu_id, subprocess.Popen(cmd, env=env, text=True)))

        still_running = []
        for fold, gpu_id, proc in running:
            return_code = proc.poll()
            if return_code is None:
                still_running.append((fold, gpu_id, proc))
            elif return_code != 0:
                raise RuntimeError(f"Fold {fold} on GPU {gpu_id} failed with exit code {return_code}.")
            else:
                print(f"Fold {fold} on GPU {gpu_id} finished.", flush=True)
        running = still_running
        if running:
            time.sleep(10)

    aggregate_outputs(args)


def run_worker(args: argparse.Namespace) -> None:
    """Train one TabNet fold and save partial predictions."""
    set_seed(SEED + args.fold)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    train = create_folds(make_features(train_raw))
    test = make_features(test_raw)
    y = encode_labels(train[TARGET_COLUMN].to_numpy())
    train_idx = (train[FOLD_COLUMN] != args.fold).to_numpy()
    valid_idx = (train[FOLD_COLUMN] == args.fold).to_numpy()

    encoding = fit_encoding(train.loc[train_idx, FEATURE_COLUMNS])
    x_train = transform_features(train.loc[train_idx, FEATURE_COLUMNS], encoding)
    x_valid = transform_features(train.loc[valid_idx, FEATURE_COLUMNS], encoding)
    x_test = transform_features(test[FEATURE_COLUMNS], encoding)
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    cat_idxs = list(range(len(NUMERIC_FEATURES), len(FEATURE_COLUMNS)))
    model = TabNetClassifier(
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
        cat_idxs=cat_idxs,
        cat_dims=encoding["cardinalities"],
        cat_emb_dim=args.cat_emb_dim,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        seed=SEED + args.fold,
        device_name="cuda" if torch.cuda.is_available() else "cpu",
        verbose=10,
    )
    model.fit(
        X_train=x_train,
        y_train=y_train,
        eval_set=[(x_valid, y_valid)],
        eval_name=["valid"],
        eval_metric=["logloss"],
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=args.virtual_batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        weights=1,
    )

    valid_proba = model.predict_proba(x_valid).astype(np.float32)
    test_proba = model.predict_proba(x_test).astype(np.float32)
    score = balanced_accuracy_score(y_valid, valid_proba.argmax(axis=1))
    loss_value = log_loss(y_valid, valid_proba, labels=np.arange(len(CLASS_LABELS)))
    partial_dir = args.out_dir / "tabnet_partials" / args.model_name
    partial_dir.mkdir(parents=True, exist_ok=True)
    np.save(partial_dir / f"fold_{args.fold}_valid_idx.npy", np.where(valid_idx)[0].astype(np.int64))
    np.save(partial_dir / f"fold_{args.fold}_valid_proba.npy", valid_proba)
    np.save(partial_dir / f"fold_{args.fold}_test_proba.npy", test_proba)
    (partial_dir / f"fold_{args.fold}_metrics.json").write_text(
        json.dumps({"balanced_accuracy": float(score), "log_loss": float(loss_value)}, indent=2),
        encoding="utf-8",
    )
    model.save_model(str(partial_dir / f"{args.model_name}_fold_{args.fold}"))
    print(f"fold {args.fold}: balanced_accuracy={score:.8f}, log_loss={loss_value:.8f}", flush=True)


def aggregate_outputs(args: argparse.Namespace) -> None:
    """Aggregate fold partials into OOF, test-proba, and submission artifacts."""
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    train = create_folds(make_features(train_raw))
    test = make_features(test_raw)
    partial_dir = args.out_dir / "tabnet_partials" / args.model_name
    oof_proba = np.zeros((len(train), len(CLASS_LABELS)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(CLASS_LABELS)), dtype=np.float32)
    fold_scores = []
    fold_losses = []

    for fold in range(N_FOLDS):
        valid_idx = np.load(partial_dir / f"fold_{fold}_valid_idx.npy")
        valid_proba = np.load(partial_dir / f"fold_{fold}_valid_proba.npy")
        fold_test_proba = np.load(partial_dir / f"fold_{fold}_test_proba.npy")
        metrics = json.loads((partial_dir / f"fold_{fold}_metrics.json").read_text(encoding="utf-8"))
        oof_proba[valid_idx] = valid_proba
        test_proba += fold_test_proba / N_FOLDS
        fold_scores.append(float(metrics["balanced_accuracy"]))
        fold_losses.append(float(metrics["log_loss"]))

    ensemble_dir = args.out_dir / "data" / "ensemble"
    oof_dir = ensemble_dir / "oof"
    test_dir = ensemble_dir / "test_proba"
    submissions_dir = args.out_dir / "submissions"
    oof_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    oof = train[[ID_COLUMN, FOLD_COLUMN, TARGET_COLUMN]].rename(
        columns={ID_COLUMN: "id", FOLD_COLUMN: "fold", TARGET_COLUMN: "y_true"}
    ).copy()
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

    oof_score = balanced_accuracy_score(oof["y_true"], labels[oof_proba.argmax(axis=1)])
    oof_loss = log_loss(oof["y_true"], oof_proba, labels=CLASS_LABELS)
    print(f"Mean fold balanced accuracy: {np.mean(fold_scores):.8f}", flush=True)
    print(f"OOF balanced accuracy: {oof_score:.8f}", flush=True)
    print(f"Mean fold log loss: {np.mean(fold_losses):.8f}", flush=True)
    print(f"OOF log loss: {oof_loss:.8f}", flush=True)
    print(f"OOF artifact: {oof_path}", flush=True)
    print(f"Test probability artifact: {test_path}", flush=True)
    print(f"Submission: {submission_path}", flush=True)


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


def transform_features(data: pd.DataFrame, encoding: dict[str, Any]) -> np.ndarray:
    numeric = encoding["scaler"].transform(data[NUMERIC_FEATURES]).astype(np.float32)
    categorical_columns = []
    for column in CATEGORICAL_FEATURES:
        mapping = encoding["category_maps"][column]
        encoded = data[column].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.float32)
        categorical_columns.append(encoded)
    categorical = np.column_stack(categorical_columns).astype(np.float32)
    return np.concatenate([numeric, categorical], axis=1).astype(np.float32)


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
    str(SCRIPT_PATH),
    "--mode",
    "orchestrate",
    "--data-dir",
    DATA_DIR,
    "--out-dir",
    OUT_DIR,
    "--model-name",
    MODEL_NAME,
    "--parallel-workers",
    "2",
    "--max-epochs",
    "80",
    "--patience",
    "12",
    "--batch-size",
    "8192",
    "--virtual-batch-size",
    "1024",
    "--n-d",
    "16",
    "--n-a",
    "16",
    "--n-steps",
    "4",
    "--gamma",
    "1.4",
    "--cat-emb-dim",
    "2",
    "--learning-rate",
    "0.02",
    "--weight-decay",
    "0.00001",
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)
