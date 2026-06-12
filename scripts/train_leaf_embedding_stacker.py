"""Train a restrained fold-safe GBDT leaf-embedding neural stacker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data import load_raw_data, make_features
from src.ensemble import load_probability_artifact
from src.utils import ensure_output_dirs, get_best_score


@dataclass(frozen=True)
class LeafStackerResult:
    """Result metadata for a leaf-embedding stacker run."""

    balanced_accuracy: float
    log_loss: float
    oof_path: Path
    test_proba_path: Path


class LeafEmbeddingStacker(nn.Module):
    """Small neural stacker over LightGBM leaf tokens and base probabilities."""

    def __init__(self, vocab_size: int, proba_dim: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + proba_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(config.CLASS_LABELS)),
        )

    def forward(self, leaves: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        """Predict logits from leaf token IDs and probability meta features."""
        leaf_embedding = self.embedding(leaves).mean(dim=1)
        return self.network(torch.cat([leaf_embedding, probabilities], dim=1))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lgbm-model-name",
        type=str,
        default="lightgbm_optuna_trial_84_n_estimators_200",
        help="Saved LightGBM fold model prefix.",
    )
    parser.add_argument("--proba-paths", nargs="+", type=Path, required=True, help="OOF probability artifact paths.")
    parser.add_argument(
        "--test-proba-paths",
        nargs="+",
        type=Path,
        required=True,
        help="Test probability artifact paths in the same order as OOF probability paths.",
    )
    parser.add_argument("--base-names", nargs="+", required=True, help="Human-readable probability artifact names.")
    parser.add_argument("--output-name", type=str, required=True, help="Output artifact name.")
    parser.add_argument("--embedding-dim", type=int, default=8, help="Leaf embedding dimension.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width.")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout probability.")
    parser.add_argument("--epochs", type=int, default=8, help="Maximum training epochs per fold.")
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience.")
    parser.add_argument("--batch-size", type=int, default=4096, help="CPU training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    return parser.parse_args()


def main() -> None:
    """Train the leaf-embedding stacker and save OOF/test-proba artifacts."""
    args = parse_args()
    _set_seed(config.SEED)
    ensure_output_dirs()
    if len(args.proba_paths) != len(args.base_names):
        raise ValueError("OOF probability path count must match base-name count.")
    if len(args.test_proba_paths) != len(args.base_names):
        raise ValueError("Test probability path count must match base-name count.")

    train_raw, test_raw, _ = load_raw_data()
    train = make_features(train_raw)
    test = make_features(test_raw)
    proba_frames = [load_probability_artifact(path) for path in args.proba_paths]
    test_proba_features = _test_proba_features(args.test_proba_paths)
    _validate_proba_artifacts(proba_frames)
    if not train["id"].equals(proba_frames[0]["id"]):
        raise ValueError("Training rows are not aligned with probability artifacts by id.")
    y_true = proba_frames[0]["y_true"].to_numpy()
    y_encoded = _encode_labels(y_true)
    folds = proba_frames[0]["fold"].to_numpy()
    proba_features = _stack_probabilities(proba_frames)

    print("Extracting fold-safe OOF LightGBM leaves...", flush=True)
    train_leaf_codes, n_trees, max_leaf_id = _extract_oof_leaf_codes(args.lgbm_model_name, train, folds)
    vocab_size = n_trees * (max_leaf_id + 1)
    print(f"Leaf token matrix: rows={train_leaf_codes.shape[0]}, trees={n_trees}, vocab_size={vocab_size}", flush=True)

    oof_proba = np.zeros((len(train), len(config.CLASS_LABELS)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(config.CLASS_LABELS)), dtype=np.float32)
    fold_scores: list[float] = []
    fold_losses: list[float] = []

    for fold in range(config.N_FOLDS):
        print(f"Training leaf stacker fold {fold}...", flush=True)
        result = _train_fold(
            fold=fold,
            train_leaf_codes=train_leaf_codes,
            train_proba_features=proba_features,
            y_encoded=y_encoded,
            y_true=y_true,
            folds=folds,
            test=test,
            test_proba_features=test_proba_features,
            args=args,
            vocab_size=vocab_size,
            max_leaf_id=max_leaf_id,
        )
        valid_idx = folds == fold
        oof_proba[valid_idx] = result["valid_proba"]
        test_proba += result["test_proba"] / config.N_FOLDS
        fold_scores.append(float(result["score"]))
        fold_losses.append(float(result["loss"]))
        print(f"Fold {fold}: balanced_accuracy={result['score']:.8f}, log_loss={result['loss']:.8f}", flush=True)

    score = float(balanced_accuracy_score(y_true, _labels_from_proba(oof_proba)))
    loss_value = float(log_loss(y_true, oof_proba, labels=config.CLASS_LABELS))
    oof_path = config.STACKING_DIR / f"{args.output_name}_oof.csv"
    test_path = config.STACKING_DIR / f"{args.output_name}_test_proba.csv"
    _write_oof(proba_frames[0], oof_proba, oof_path)
    _write_test(test, test_proba, test_path)
    _log_mlflow(args, score, loss_value, fold_scores, fold_losses, oof_path, test_path)

    print(f"Leaf stacker balanced accuracy: {score:.8f}")
    print(f"Leaf stacker log loss: {loss_value:.8f}")
    print(f"Mean fold balanced accuracy: {np.mean(fold_scores):.8f}")
    print(f"Mean fold log loss: {np.mean(fold_losses):.8f}")
    print(f"OOF artifact: {oof_path}")
    print(f"Test probability artifact: {test_path}")
    print(f"Current champion threshold: {get_best_score():.8f}")


def _train_fold(
    *,
    fold: int,
    train_leaf_codes: np.ndarray,
    train_proba_features: np.ndarray,
    y_encoded: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
    test: pd.DataFrame,
    test_proba_features: np.ndarray,
    args: argparse.Namespace,
    vocab_size: int,
    max_leaf_id: int,
) -> dict[str, Any]:
    """Train one fold of the leaf-embedding stacker."""
    train_idx = folds != fold
    valid_idx = folds == fold
    model = LeafEmbeddingStacker(
        vocab_size=vocab_size,
        proba_dim=train_proba_features.shape[1],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=_class_weight_tensor(y_encoded[train_idx]))
    train_loader = _make_loader(
        train_leaf_codes[train_idx],
        train_proba_features[train_idx],
        y_encoded[train_idx],
        args.batch_size,
        shuffle=True,
    )
    valid_leaves = torch.from_numpy(train_leaf_codes[valid_idx].astype(np.int64))
    valid_features = torch.from_numpy(train_proba_features[valid_idx].astype(np.float32))
    best_score = -np.inf
    best_state = None
    stale_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        for leaf_batch, proba_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(leaf_batch, proba_batch), target_batch)
            loss.backward()
            optimizer.step()
        valid_proba = _predict_proba(model, valid_leaves, valid_features, args.batch_size)
        score = balanced_accuracy_score(y_true[valid_idx], _labels_from_proba(valid_proba))
        print(f"Fold {fold} epoch {epoch + 1}: balanced_accuracy={score:.8f}", flush=True)
        if score > best_score:
            best_score = float(score)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_proba = _predict_proba(model, valid_leaves, valid_features, args.batch_size)
    test_leaf_codes = _extract_test_leaf_codes(args.lgbm_model_name, fold, test, max_leaf_id)
    test_leaves = torch.from_numpy(test_leaf_codes.astype(np.int64))
    test_features = torch.from_numpy(test_proba_features.astype(np.float32))
    test_proba = _predict_proba(model, test_leaves, test_features, args.batch_size)
    return {
        "valid_proba": valid_proba,
        "test_proba": test_proba,
        "score": balanced_accuracy_score(y_true[valid_idx], _labels_from_proba(valid_proba)),
        "loss": log_loss(y_true[valid_idx], valid_proba, labels=config.CLASS_LABELS),
    }


def _extract_oof_leaf_codes(model_name: str, train: pd.DataFrame, folds: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Extract OOF leaf token IDs from saved LightGBM fold pipelines."""
    leaf_codes = None
    n_trees = 0
    max_leaf_id = 0
    for fold in range(config.N_FOLDS):
        valid_idx = folds == fold
        pipeline = joblib.load(config.MODELS_DIR / f"{model_name}_fold_{fold}.joblib")
        leaves = _predict_leaf_indices(pipeline, train.loc[valid_idx, config.FEATURE_COLUMNS])
        if leaf_codes is None:
            n_trees = leaves.shape[1]
            max_leaf_id = int(max(255, leaves.max()))
            leaf_codes = np.zeros((len(train), n_trees), dtype=np.int32)
        max_leaf_id = int(max(max_leaf_id, leaves.max()))
        leaf_codes[valid_idx] = leaves
    if leaf_codes is None:
        raise ValueError("No leaf codes were extracted.")
    if max_leaf_id > 65535:
        raise ValueError(f"Unexpectedly large LightGBM leaf id: {max_leaf_id}")
    offsets = (np.arange(n_trees, dtype=np.int32) * (max_leaf_id + 1))[None, :]
    return (leaf_codes + offsets).astype(np.int32), n_trees, max_leaf_id


def _extract_test_leaf_codes(model_name: str, fold: int, test: pd.DataFrame, max_leaf_id: int) -> np.ndarray:
    """Extract test leaf token IDs from one saved LightGBM fold pipeline."""
    pipeline = joblib.load(config.MODELS_DIR / f"{model_name}_fold_{fold}.joblib")
    leaves = _predict_leaf_indices(pipeline, test[config.FEATURE_COLUMNS])
    offsets = (np.arange(leaves.shape[1], dtype=np.int32) * (max_leaf_id + 1))[None, :]
    return (leaves + offsets).astype(np.int32)


def _predict_leaf_indices(pipeline: Any, features: pd.DataFrame) -> np.ndarray:
    """Predict LightGBM leaf indices from a saved sklearn pipeline."""
    transformed = pipeline.named_steps["preprocessor"].transform(features)
    leaves = pipeline.named_steps["model"].predict(transformed, pred_leaf=True)
    return leaves.astype(np.int32)


def _make_loader(
    leaves: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    """Build a CPU DataLoader for stacker training."""
    dataset = TensorDataset(
        torch.from_numpy(leaves.astype(np.int64)),
        torch.from_numpy(probabilities.astype(np.float32)),
        torch.from_numpy(targets.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_proba(
    model: LeafEmbeddingStacker,
    leaves: torch.Tensor,
    probabilities: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    """Predict class probabilities in batches."""
    model.eval()
    outputs = []
    dataset = TensorDataset(leaves, probabilities)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for leaf_batch, proba_batch in loader:
            outputs.append(torch.softmax(model(leaf_batch, proba_batch), dim=1).numpy())
    return np.vstack(outputs).astype(np.float32)


def _stack_probabilities(frames: list[pd.DataFrame]) -> np.ndarray:
    """Stack clipped log probabilities from base artifacts."""
    matrices = []
    for frame in frames:
        probabilities = np.clip(frame[config.PROBA_COLUMNS].to_numpy(dtype=np.float32), 1e-8, 1.0)
        matrices.append(np.log(probabilities))
    return np.hstack(matrices).astype(np.float32)


def _test_proba_features(paths: list[Path]) -> np.ndarray:
    """Load and stack test probability features."""
    frames = [load_probability_artifact(path) for path in paths]
    reference = frames[0]["id"]
    for frame in frames[1:]:
        if not reference.equals(frame["id"]):
            raise ValueError("Test probability artifacts are not aligned on id.")
    return _stack_probabilities(frames)


def _validate_proba_artifacts(frames: list[pd.DataFrame]) -> None:
    """Validate OOF probability artifact alignment."""
    reference = frames[0]
    for frame in frames[1:]:
        for column in ["id", "fold", "y_true"]:
            if not reference[column].equals(frame[column]):
                raise ValueError(f"OOF probability artifacts are not aligned on {column}.")


def _class_weight_tensor(y_train: np.ndarray) -> torch.Tensor:
    """Build inverse-frequency class weights for cross-entropy."""
    counts = np.bincount(y_train, minlength=len(config.CLASS_LABELS)).astype(np.float32)
    weights = counts.sum() / (len(config.CLASS_LABELS) * np.maximum(counts, 1.0))
    return torch.from_numpy(weights.astype(np.float32))


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    """Encode configured class labels as integer class indices."""
    label_to_index = {label: index for index, label in enumerate(config.CLASS_LABELS)}
    return np.asarray([label_to_index[label] for label in labels], dtype=np.int64)


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities to configured class labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


def _write_oof(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write OOF probability artifact."""
    output = reference[["id", "fold", "y_true"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_test(test: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write test probability artifact."""
    output = test[["id"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _log_mlflow(
    args: argparse.Namespace,
    score: float,
    loss_value: float,
    fold_scores: list[float],
    fold_losses: list[float],
    oof_path: Path,
    test_path: Path,
) -> None:
    """Log stacker parameters and metrics to MLflow."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"leaf_embedding_stacker_{args.output_name}"):
        mlflow.log_param("lgbm_model_name", args.lgbm_model_name)
        mlflow.log_param("base_models", ",".join(args.base_names))
        mlflow.log_param("embedding_dim", args.embedding_dim)
        mlflow.log_param("hidden_dim", args.hidden_dim)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("patience", args.patience)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("weight_decay", args.weight_decay)
        mlflow.log_param("current_best_score", get_best_score())
        mlflow.log_metric("cv_balanced_accuracy", score)
        mlflow.log_metric("cv_log_loss", loss_value)
        for fold, fold_score in enumerate(fold_scores):
            mlflow.log_metric(f"fold_{fold}_balanced_accuracy", fold_score)
        for fold, fold_loss in enumerate(fold_losses):
            mlflow.log_metric(f"fold_{fold}_log_loss", fold_loss)
        mlflow.log_artifact(str(oof_path))
        mlflow.log_artifact(str(test_path))


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducible CPU training."""
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
