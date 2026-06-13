"""Train a fold-safe one-vs-one specialist and save probability artifacts."""

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
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import create_stratified_folds, load_raw_data, make_features
from src.utils import ensure_output_dirs, get_best_score


@dataclass(frozen=True)
class BinarySpecialistResult:
    """Result metadata for a binary specialist run."""

    model_name: str
    mean_pair_balanced_accuracy: float
    mean_pair_log_loss: float
    fold_pair_scores: list[float]
    oof_path: Path
    test_proba_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-a", required=True, choices=config.CLASS_LABELS, help="First class in the specialist pair.")
    parser.add_argument("--class-b", required=True, choices=config.CLASS_LABELS, help="Second class in the specialist pair.")
    parser.add_argument("--output-name", required=True, help="Stable artifact/model name.")
    parser.add_argument("--no-save-models", action="store_true", help="Skip persisting fold model joblib artifacts.")
    return parser.parse_args()


def main() -> None:
    """Train one binary specialist and save fold-safe OOF/test probabilities."""
    args = parse_args()
    if args.class_a == args.class_b:
        raise ValueError("--class-a and --class-b must be different.")

    ensure_output_dirs()
    best_score = get_best_score()
    train_raw, test_raw, _ = load_raw_data()
    train = create_stratified_folds(make_features(train_raw), n_folds=config.N_FOLDS)
    test = make_features(test_raw)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"binary_specialist_{args.output_name}"):
        mlflow.log_param("model_name", args.output_name)
        mlflow.log_param("class_a", args.class_a)
        mlflow.log_param("class_b", args.class_b)
        mlflow.log_param("seed", config.SEED)
        mlflow.log_param("n_folds", config.N_FOLDS)
        mlflow.log_param("current_best_score", best_score)
        mlflow.log_param("features", ",".join(config.FEATURE_COLUMNS))
        mlflow.log_params(_lgbm_binary_params())
        result = train_specialist(
            train=train,
            test=test,
            class_a=args.class_a,
            class_b=args.class_b,
            output_name=args.output_name,
            save_models=not args.no_save_models,
        )
        mlflow.log_metric("cv_mean_pair_balanced_accuracy", result.mean_pair_balanced_accuracy)
        mlflow.log_metric("cv_mean_pair_log_loss", result.mean_pair_log_loss)
        for fold, score in enumerate(result.fold_pair_scores):
            mlflow.log_metric(f"fold_{fold}_pair_balanced_accuracy", score)
        mlflow.log_artifact(str(result.oof_path))
        mlflow.log_artifact(str(result.test_proba_path))

    print(f"Model: {result.model_name}")
    print(f"Mean pair balanced accuracy: {result.mean_pair_balanced_accuracy:.8f}")
    print(f"Mean pair log loss: {result.mean_pair_log_loss:.8f}")
    print(f"OOF artifact: {result.oof_path}")
    print(f"Test probability artifact: {result.test_proba_path}")
    print(f"Current champion threshold: {best_score:.8f}")


def train_specialist(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    class_a: str,
    class_b: str,
    output_name: str,
    save_models: bool,
) -> BinarySpecialistResult:
    """Train one fold-safe binary specialist.

    The specialist is trained only on rows from ``class_a`` and ``class_b``, but
    predicts every validation and test row so the resulting artifact is aligned
    with the normal multiclass OOF schema.

    Args:
        train: Feature-engineered training dataframe with fold assignments.
        test: Feature-engineered test dataframe.
        class_a: First class in the pair, encoded as binary label 0.
        class_b: Second class in the pair, encoded as binary label 1.
        output_name: Stable artifact/model name.
        save_models: Whether to persist each fold model.

    Returns:
        Binary specialist result metadata.
    """
    oof_proba = np.zeros((len(train), len(config.CLASS_LABELS)), dtype=float)
    test_proba = np.zeros((len(test), len(config.CLASS_LABELS)), dtype=float)
    pair_scores: list[float] = []
    pair_losses: list[float] = []

    for fold in range(config.N_FOLDS):
        train_idx = (train[config.FOLD_COLUMN] != fold) & train[config.TARGET_COLUMN].isin([class_a, class_b])
        valid_idx = train[config.FOLD_COLUMN] == fold
        valid_pair_idx = valid_idx & train[config.TARGET_COLUMN].isin([class_a, class_b])
        x_train = train.loc[train_idx, config.FEATURE_COLUMNS]
        y_train = (train.loc[train_idx, config.TARGET_COLUMN] == class_b).astype(int).to_numpy()
        x_valid = train.loc[valid_idx, config.FEATURE_COLUMNS]
        x_valid_pair = train.loc[valid_pair_idx, config.FEATURE_COLUMNS]
        y_valid_pair = (train.loc[valid_pair_idx, config.TARGET_COLUMN] == class_b).astype(int).to_numpy()

        estimator = _build_pipeline()
        estimator.fit(x_train, y_train)
        valid_binary_proba = estimator.predict_proba(x_valid)
        test_binary_proba = estimator.predict_proba(test[config.FEATURE_COLUMNS])
        valid_pair_proba = estimator.predict_proba(x_valid_pair)
        pair_pred = valid_pair_proba.argmax(axis=1)
        pair_score = balanced_accuracy_score(y_valid_pair, pair_pred)
        pair_loss = log_loss(y_valid_pair, valid_pair_proba, labels=[0, 1])

        oof_proba[valid_idx.to_numpy()] = _binary_to_multiclass_proba(valid_binary_proba, class_a, class_b)
        test_proba += _binary_to_multiclass_proba(test_binary_proba, class_a, class_b) / config.N_FOLDS
        pair_scores.append(float(pair_score))
        pair_losses.append(float(pair_loss))

        if save_models:
            model_path = config.MODELS_DIR / f"{output_name}_fold_{fold}.joblib"
            joblib.dump(estimator, model_path)
        pair_accuracy = accuracy_score(y_valid_pair, pair_pred)
        print(
            f"{output_name} fold {fold}: "
            f"pair_balanced_accuracy={pair_score:.6f}, pair_accuracy={pair_accuracy:.6f}, "
            f"pair_log_loss={pair_loss:.6f}"
        )

    oof_path = config.OOF_DIR / f"{output_name}_oof.csv"
    test_proba_path = config.TEST_PROBA_DIR / f"{output_name}_test_proba.csv"
    _write_oof_artifact(train, oof_proba, oof_path)
    _write_test_proba_artifact(test, test_proba, test_proba_path)
    return BinarySpecialistResult(
        model_name=output_name,
        mean_pair_balanced_accuracy=float(np.mean(pair_scores)),
        mean_pair_log_loss=float(np.mean(pair_losses)),
        fold_pair_scores=pair_scores,
        oof_path=oof_path,
        test_proba_path=test_proba_path,
    )


def _build_pipeline() -> Pipeline:
    """Build the fold-safe binary specialist pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", LGBMClassifier(**_lgbm_binary_params()))])


def _lgbm_binary_params() -> dict[str, Any]:
    """Build LightGBM parameters for a one-vs-one specialist."""
    return {
        "objective": "binary",
        "class_weight": "balanced",
        "n_estimators": 420,
        "max_depth": -1,
        "num_leaves": 96,
        "min_child_samples": 45,
        "learning_rate": 0.04,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 2.0,
        "random_state": config.SEED,
        "n_jobs": config.N_JOBS,
        "verbosity": -1,
    }


def _binary_to_multiclass_proba(binary_proba: np.ndarray, class_a: str, class_b: str) -> np.ndarray:
    """Map binary pair probabilities into the standard three-class schema."""
    output = np.full((len(binary_proba), len(config.CLASS_LABELS)), 1e-6, dtype=float)
    class_a_index = config.CLASS_LABELS.index(class_a)
    class_b_index = config.CLASS_LABELS.index(class_b)
    output[:, class_a_index] = binary_proba[:, 0]
    output[:, class_b_index] = binary_proba[:, 1]
    output = output / output.sum(axis=1, keepdims=True)
    return output


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
