"""Train a fold-safe randomized-kernel classifier and save probability artifacts."""

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
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src import config
from src.data import create_stratified_folds, load_raw_data, make_features
from src.utils import ensure_output_dirs, get_best_score


@dataclass(frozen=True)
class KernelApproxResult:
    """Result metadata for a randomized-kernel run."""

    model_name: str
    mean_balanced_accuracy: float
    mean_log_loss: float
    fold_scores: list[float]
    oof_path: Path
    test_proba_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="kernel_rbf_sgd_c1024_g003", help="Stable artifact/model name.")
    parser.add_argument("--n-components", type=int, default=1024, help="Random Fourier feature count.")
    parser.add_argument("--gamma", type=float, default=0.03, help="RBF kernel gamma.")
    parser.add_argument("--alpha", type=float, default=1e-5, help="SGDClassifier regularization strength.")
    parser.add_argument("--max-iter", type=int, default=80, help="Maximum SGD epochs.")
    parser.add_argument("--no-save-models", action="store_true", help="Skip persisting fold model joblib artifacts.")
    return parser.parse_args()


def main() -> None:
    """Train one randomized-kernel base model with fixed fold artifacts."""
    args = parse_args()
    ensure_output_dirs()
    best_score = get_best_score()
    train_raw, test_raw, _ = load_raw_data()
    train = create_stratified_folds(make_features(train_raw), n_folds=config.N_FOLDS)
    test = make_features(test_raw)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"kernel_approx_{args.output_name}"):
        mlflow.log_param("model_name", args.output_name)
        mlflow.log_param("seed", config.SEED)
        mlflow.log_param("n_folds", config.N_FOLDS)
        mlflow.log_param("current_best_score", best_score)
        mlflow.log_param("features", ",".join(config.FEATURE_COLUMNS))
        mlflow.log_param("n_components", args.n_components)
        mlflow.log_param("gamma", args.gamma)
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("max_iter", args.max_iter)
        result = train_kernel_approx(
            train=train,
            test=test,
            output_name=args.output_name,
            n_components=args.n_components,
            gamma=args.gamma,
            alpha=args.alpha,
            max_iter=args.max_iter,
            save_models=not args.no_save_models,
        )
        mlflow.log_metric("cv_mean_balanced_accuracy", result.mean_balanced_accuracy)
        mlflow.log_metric("cv_mean_log_loss", result.mean_log_loss)
        for fold, score in enumerate(result.fold_scores):
            mlflow.log_metric(f"fold_{fold}_balanced_accuracy", score)
        mlflow.log_artifact(str(result.oof_path))
        mlflow.log_artifact(str(result.test_proba_path))

    print(f"Model: {result.model_name}")
    print(f"Mean balanced accuracy: {result.mean_balanced_accuracy:.8f}")
    print(f"Mean log loss: {result.mean_log_loss:.8f}")
    print(f"OOF artifact: {result.oof_path}")
    print(f"Test probability artifact: {result.test_proba_path}")
    print(f"Current champion threshold: {best_score:.8f}")


def train_kernel_approx(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_name: str,
    n_components: int,
    gamma: float,
    alpha: float,
    max_iter: int,
    save_models: bool,
) -> KernelApproxResult:
    """Train a randomized RBF feature model and save OOF/test probabilities.

    Args:
        train: Feature-engineered training dataframe with fold assignments.
        test: Feature-engineered test dataframe.
        output_name: Stable artifact/model name.
        n_components: Random Fourier feature count.
        gamma: RBF kernel gamma.
        alpha: SGD regularization strength.
        max_iter: Maximum SGD epochs.
        save_models: Whether to persist each fold model.

    Returns:
        Kernel approximation result metadata.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(train[config.TARGET_COLUMN])
    _validate_label_order(encoder)
    n_classes = len(config.CLASS_LABELS)
    oof_proba = np.zeros((len(train), n_classes), dtype=float)
    test_proba = np.zeros((len(test), n_classes), dtype=float)
    fold_scores: list[float] = []
    fold_losses: list[float] = []

    for fold in range(config.N_FOLDS):
        train_idx = train[config.FOLD_COLUMN] != fold
        valid_idx = train[config.FOLD_COLUMN] == fold
        x_train = train.loc[train_idx, config.FEATURE_COLUMNS]
        x_valid = train.loc[valid_idx, config.FEATURE_COLUMNS]
        y_train = y[train_idx.to_numpy()]
        y_valid = y[valid_idx.to_numpy()]

        estimator = _build_pipeline(n_components=n_components, gamma=gamma, alpha=alpha, max_iter=max_iter)
        estimator.fit(x_train, y_train)
        valid_proba = estimator.predict_proba(x_valid)
        fold_test_proba = estimator.predict_proba(test[config.FEATURE_COLUMNS])
        valid_pred = valid_proba.argmax(axis=1)
        fold_score = balanced_accuracy_score(y_valid, valid_pred)
        fold_loss = log_loss(y_valid, valid_proba, labels=np.arange(n_classes))

        oof_proba[valid_idx.to_numpy()] = valid_proba
        test_proba += fold_test_proba / config.N_FOLDS
        fold_scores.append(float(fold_score))
        fold_losses.append(float(fold_loss))

        if save_models:
            model_path = config.MODELS_DIR / f"{output_name}_fold_{fold}.joblib"
            joblib.dump(estimator, model_path)
        print(f"{output_name} fold {fold}: balanced_accuracy={fold_score:.6f}, log_loss={fold_loss:.6f}")

    oof_path = config.OOF_DIR / f"{output_name}_oof.csv"
    test_proba_path = config.TEST_PROBA_DIR / f"{output_name}_test_proba.csv"
    _write_oof_artifact(train, oof_proba, oof_path)
    _write_test_proba_artifact(test, test_proba, test_proba_path)
    return KernelApproxResult(
        model_name=output_name,
        mean_balanced_accuracy=float(np.mean(fold_scores)),
        mean_log_loss=float(np.mean(fold_losses)),
        fold_scores=fold_scores,
        oof_path=oof_path,
        test_proba_path=test_proba_path,
    )


def _build_pipeline(*, n_components: int, gamma: float, alpha: float, max_iter: int) -> Pipeline:
    """Build the fold-safe randomized-kernel classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )
    kernel = RBFSampler(gamma=gamma, n_components=n_components, random_state=config.SEED)
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-4,
        class_weight="balanced",
        random_state=config.SEED,
        average=True,
        n_jobs=config.N_JOBS,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("kernel", kernel), ("classifier", classifier)])


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


def _validate_label_order(encoder: LabelEncoder) -> None:
    """Validate encoded class order against project configuration."""
    classes = encoder.classes_.tolist()
    if classes != config.CLASS_LABELS:
        raise ValueError(f"Unexpected class order {classes}; expected {config.CLASS_LABELS}.")


if __name__ == "__main__":
    main()
