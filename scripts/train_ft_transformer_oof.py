"""Train a small FT-Transformer-style tabular model and save OOF artifacts."""

from __future__ import annotations

import argparse
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_entity_embedding_mlp_oof import (
    _class_weight_tensor,
    _encode_labels,
    _fit_fold_encoding,
    _set_reproducibility,
    _transform_features,
    _write_oof_artifact,
    _write_test_proba_artifact,
)
from src import config
from src.data import create_stratified_folds, load_raw_data, make_features
from src.utils import ensure_output_dirs, get_best_score


MODEL_NAME = "ft_transformer_small"


class FTTransformerSmall(nn.Module):
    """Small FT-Transformer-style model for mixed tabular features."""

    def __init__(
        self,
        n_numeric: int,
        cardinalities: list[int],
        token_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the model.

        Args:
            n_numeric: Number of numeric features.
            cardinalities: Categorical cardinalities including unknown index 0.
            token_dim: Shared token embedding dimension.
            n_heads: Number of transformer attention heads.
            n_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.numeric_weight = nn.Parameter(torch.empty(n_numeric, token_dim))
        self.numeric_bias = nn.Parameter(torch.empty(n_numeric, token_dim))
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, token_dim) for cardinality in cardinalities]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, len(config.CLASS_LABELS)),
        )
        nn.init.xavier_uniform_(self.numeric_weight)
        nn.init.zeros_(self.numeric_bias)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """Compute class logits.

        Args:
            numeric: Standardized numeric tensor.
            categorical: Integer-encoded categorical tensor.

        Returns:
            Class logits.
        """
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight.unsqueeze(0) + self.numeric_bias.unsqueeze(0)
        categorical_tokens = [
            embedding(categorical[:, index]).unsqueeze(1)
            for index, embedding in enumerate(self.categorical_embeddings)
        ]
        cls = self.cls_token.expand(numeric.shape[0], -1, -1)
        tokens = torch.cat([cls, numeric_tokens, *categorical_tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Stable artifact/model name.")
    parser.add_argument("--epochs", type=int, default=6, help="Maximum epochs per fold.")
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience by validation score.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training and inference batch size.")
    parser.add_argument("--token-dim", type=int, default=24, help="Transformer token dimension.")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--n-layers", type=int, default=1, help="Number of transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout rate.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="AdamW weight decay.")
    parser.add_argument("--no-save-models", action="store_true", help="Skip saving fold model state dictionaries.")
    return parser.parse_args()


def main() -> None:
    """Train fixed-fold FT-Transformer OOF artifacts."""
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
        _log_params(args, best_score)
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


def _log_params(args: argparse.Namespace, best_score: float) -> None:
    """Log run parameters to MLflow."""
    mlflow.log_param("model_name", args.model_name)
    mlflow.log_param("model_family", "ft_transformer_small")
    mlflow.log_param("seed", config.SEED)
    mlflow.log_param("n_folds", config.N_FOLDS)
    mlflow.log_param("current_best_score", best_score)
    for key, value in vars(args).items():
        mlflow.log_param(key, value)


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

    model = FTTransformerSmall(
        n_numeric=len(config.NUMERIC_FEATURES),
        cardinalities=encoding.cardinalities,
        token_dim=args.token_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
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
            {"model_state_dict": model.state_dict(), "args": vars(args)},
            config.MODELS_DIR / f"{args.model_name}_fold_{fold}.pt",
        )
    return {"valid_proba": valid_proba, "test_proba": test_proba, "score": score, "loss": loss_value}


def _predict_proba(
    model: FTTransformerSmall,
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


if __name__ == "__main__":
    main()
