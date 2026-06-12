"""Train a restrained entity-embedding MLP and save OOF probability artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data import create_stratified_folds, load_raw_data, make_features
from src.utils import ensure_output_dirs, get_best_score


MODEL_NAME = "entity_embedding_mlp_small"


@dataclass(frozen=True)
class FoldEncoding:
    """Fold-local preprocessing objects for neural inputs.

    Attributes:
        scaler: Fitted numeric feature scaler.
        category_maps: Per-column categorical value to integer-index maps.
        cardinalities: Embedding cardinality for each categorical feature, including unknown index 0.
    """

    scaler: StandardScaler
    category_maps: dict[str, dict[str, int]]
    cardinalities: list[int]


class EntityEmbeddingMLP(nn.Module):
    """Small MLP over standardized numeric features and learned categorical embeddings."""

    def __init__(self, n_numeric: int, cardinalities: list[int], hidden_units: int, dropout: float) -> None:
        """Initialize the network.

        Args:
            n_numeric: Number of numeric input features.
            cardinalities: Categorical cardinalities including unknown index 0.
            hidden_units: Width of the first hidden layer.
            dropout: Dropout rate between hidden layers.
        """
        super().__init__()
        embedding_dims = [min(8, max(2, int(np.ceil(cardinality**0.25 * 2)))) for cardinality in cardinalities]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension) for cardinality, dimension in zip(cardinalities, embedding_dims)]
        )
        input_dim = n_numeric + int(sum(embedding_dims))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.BatchNorm1d(hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units // 2, len(config.CLASS_LABELS)),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            numeric: Standardized numeric tensor.
            categorical: Integer-encoded categorical tensor.

        Returns:
            Class logits tensor.
        """
        embedded = [embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)]
        features = torch.cat([numeric, *embedded], dim=1)
        return self.network(features)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Stable artifact/model name.")
    parser.add_argument("--epochs", type=int, default=8, help="Maximum epochs per fold.")
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience by validation score.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Training and inference batch size.")
    parser.add_argument("--hidden-units", type=int, default=96, help="First hidden-layer width.")
    parser.add_argument("--dropout", type=float, default=0.12, help="Dropout rate.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="AdamW weight decay.")
    parser.add_argument("--no-save-models", action="store_true", help="Skip saving fold model state dictionaries.")
    return parser.parse_args()


def main() -> None:
    """Train the neural OOF model and save probability artifacts."""
    args = parse_args()
    ensure_output_dirs()
    _set_reproducibility(config.SEED)
    best_score = get_best_score()
    train_raw, test_raw, _ = load_raw_data()
    train = create_stratified_folds(make_features(train_raw), n_folds=config.N_FOLDS)
    test = make_features(test_raw)
    y = _encode_labels(train[config.TARGET_COLUMN].to_numpy())
    oof_proba = np.zeros((len(train), len(config.CLASS_LABELS)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(config.CLASS_LABELS)), dtype=np.float32)
    fold_scores: list[float] = []
    fold_losses: list[float] = []

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"oof_{args.model_name}"):
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("model_family", "entity_embedding_mlp")
        mlflow.log_param("seed", config.SEED)
        mlflow.log_param("n_folds", config.N_FOLDS)
        mlflow.log_param("current_best_score", best_score)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("patience", args.patience)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("hidden_units", args.hidden_units)
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("weight_decay", args.weight_decay)

        for fold in range(config.N_FOLDS):
            fold_result = _train_fold(args, fold, train, test, y, save_models=not args.no_save_models)
            valid_idx = (train[config.FOLD_COLUMN] == fold).to_numpy()
            oof_proba[valid_idx] = fold_result["valid_proba"]
            test_proba += fold_result["test_proba"] / config.N_FOLDS
            fold_scores.append(float(fold_result["score"]))
            fold_losses.append(float(fold_result["loss"]))
            mlflow.log_metric(f"fold_{fold}_balanced_accuracy", float(fold_result["score"]))
            mlflow.log_metric(f"fold_{fold}_log_loss", float(fold_result["loss"]))
            print(
                f"{args.model_name} fold {fold}: "
                f"balanced_accuracy={fold_result['score']:.6f}, log_loss={fold_result['loss']:.6f}"
            )

        mean_score = float(np.mean(fold_scores))
        mean_loss = float(np.mean(fold_losses))
        oof_path = config.OOF_DIR / f"{args.model_name}_oof.csv"
        test_proba_path = config.TEST_PROBA_DIR / f"{args.model_name}_test_proba.csv"
        _write_oof_artifact(train, oof_proba, oof_path)
        _write_test_proba_artifact(test, test_proba, test_proba_path)
        mlflow.log_metric("cv_mean_balanced_accuracy", mean_score)
        mlflow.log_metric("cv_mean_log_loss", mean_loss)
        mlflow.log_artifact(str(oof_path))
        mlflow.log_artifact(str(test_proba_path))

    print(f"Model: {args.model_name}")
    print(f"Mean balanced accuracy: {mean_score:.8f}")
    print(f"Mean log loss: {mean_loss:.8f}")
    print(f"OOF artifact: {oof_path}")
    print(f"Test probability artifact: {test_proba_path}")
    print(f"Current champion threshold: {best_score:.8f}")


def _train_fold(
    args: argparse.Namespace,
    fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    *,
    save_models: bool,
) -> dict[str, Any]:
    """Train one fold and return validation/test probabilities."""
    train_idx = (train[config.FOLD_COLUMN] != fold).to_numpy()
    valid_idx = (train[config.FOLD_COLUMN] == fold).to_numpy()
    encoding = _fit_fold_encoding(train.loc[train_idx, config.FEATURE_COLUMNS])
    x_train_num, x_train_cat = _transform_features(train.loc[train_idx, config.FEATURE_COLUMNS], encoding)
    x_valid_num, x_valid_cat = _transform_features(train.loc[valid_idx, config.FEATURE_COLUMNS], encoding)
    x_test_num, x_test_cat = _transform_features(test[config.FEATURE_COLUMNS], encoding)
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    model = EntityEmbeddingMLP(
        n_numeric=len(config.NUMERIC_FEATURES),
        cardinalities=encoding.cardinalities,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=_class_weight_tensor(y_train))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train_num),
            torch.from_numpy(x_train_cat),
            torch.from_numpy(y_train.astype(np.int64)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        for numeric_batch, categorical_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(numeric_batch, categorical_batch), target_batch)
            loss.backward()
            optimizer.step()
        valid_proba = _predict_proba(model, x_valid_num, x_valid_cat, args.batch_size)
        score = balanced_accuracy_score(y_valid, valid_proba.argmax(axis=1))
        if score > best_score:
            best_score = float(score)
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(f"{args.model_name} fold {fold} epoch {epoch + 1}: balanced_accuracy={score:.6f}")
        if stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_proba = _predict_proba(model, x_valid_num, x_valid_cat, args.batch_size)
    test_proba = _predict_proba(model, x_test_num, x_test_cat, args.batch_size)
    score = balanced_accuracy_score(y_valid, valid_proba.argmax(axis=1))
    loss_value = log_loss(y_valid, valid_proba, labels=np.arange(len(config.CLASS_LABELS)))

    if save_models:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "category_maps": encoding.category_maps,
                "scaler_mean": encoding.scaler.mean_,
                "scaler_scale": encoding.scaler.scale_,
                "args": vars(args),
            },
            config.MODELS_DIR / f"{args.model_name}_fold_{fold}.pt",
        )
    return {"valid_proba": valid_proba, "test_proba": test_proba, "score": score, "loss": loss_value}


def _fit_fold_encoding(data: pd.DataFrame) -> FoldEncoding:
    """Fit numeric scaler and categorical maps on a training fold."""
    scaler = StandardScaler()
    scaler.fit(data[config.NUMERIC_FEATURES])
    category_maps: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    for column in config.CATEGORICAL_FEATURES:
        values = sorted(data[column].astype(str).unique().tolist())
        mapping = {value: index + 1 for index, value in enumerate(values)}
        category_maps[column] = mapping
        cardinalities.append(len(mapping) + 1)
    return FoldEncoding(scaler=scaler, category_maps=category_maps, cardinalities=cardinalities)


def _transform_features(data: pd.DataFrame, encoding: FoldEncoding) -> tuple[np.ndarray, np.ndarray]:
    """Transform features with fold-local preprocessing."""
    numeric = encoding.scaler.transform(data[config.NUMERIC_FEATURES]).astype(np.float32)
    categorical_columns = []
    for column in config.CATEGORICAL_FEATURES:
        mapping = encoding.category_maps[column]
        encoded = data[column].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
        categorical_columns.append(encoded)
    categorical = np.column_stack(categorical_columns).astype(np.int64)
    return numeric, categorical


def _predict_proba(
    model: EntityEmbeddingMLP,
    numeric: np.ndarray,
    categorical: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Predict probabilities for numpy feature arrays."""
    model.eval()
    probabilities = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(numeric), torch.from_numpy(categorical)),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for numeric_batch, categorical_batch in loader:
            logits = model(numeric_batch, categorical_batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probabilities).astype(np.float32)


def _class_weight_tensor(y_train: np.ndarray) -> torch.Tensor:
    """Build inverse-frequency class weights for cross-entropy."""
    counts = np.bincount(y_train, minlength=len(config.CLASS_LABELS)).astype(np.float32)
    weights = counts.sum() / (len(config.CLASS_LABELS) * np.maximum(counts, 1.0))
    return torch.from_numpy(weights.astype(np.float32))


def _set_reproducibility(seed: int) -> None:
    """Set deterministic-ish seeds for CPU training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    """Encode configured class labels as integer class indices."""
    label_to_index = {label: index for index, label in enumerate(config.CLASS_LABELS)}
    return np.asarray([label_to_index[label] for label in labels], dtype=np.int64)


def _write_oof_artifact(train: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write an OOF probability artifact."""
    _validate_probability_rows(proba, context="OOF probabilities")
    output = train[[config.ID_COLUMN, config.FOLD_COLUMN, config.TARGET_COLUMN]].rename(
        columns={config.ID_COLUMN: "id", config.FOLD_COLUMN: "fold", config.TARGET_COLUMN: "y_true"}
    )
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_test_proba_artifact(test: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write an averaged test probability artifact."""
    _validate_probability_rows(proba, context="test probabilities")
    output = test[[config.ID_COLUMN]].rename(columns={config.ID_COLUMN: "id"})
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _validate_probability_rows(proba: np.ndarray, *, context: str) -> None:
    """Validate probability values and row sums."""
    if np.isnan(proba).any():
        raise ValueError(f"{context} contains NaN values.")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{context} rows do not sum to 1 within tolerance.")


if __name__ == "__main__":
    main()
