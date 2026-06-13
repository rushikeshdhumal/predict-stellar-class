"""Train a fold-safe one-vs-rest specialist and save scalar feature artifacts."""

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
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import create_stratified_folds, load_raw_data, make_features
from src.utils import append_experiment_log, ensure_output_dirs, get_best_score


@dataclass(frozen=True)
class OneVsRestResult:
    """Result metadata for a one-vs-rest specialist run."""

    model_name: str
    target_class: str
    mean_balanced_accuracy: float
    mean_log_loss: float
    oof_feature_path: Path
    test_feature_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-class", required=True, choices=config.CLASS_LABELS, help="Positive class.")
    parser.add_argument("--output-name", required=True, help="Stable artifact/model name.")
    parser.add_argument("--no-save-models", action="store_true", help="Skip persisting fold model joblib artifacts.")
    return parser.parse_args()


def main() -> None:
    """Train one fold-safe one-vs-rest specialist."""
    args = parse_args()
    ensure_output_dirs()
    best_score = get_best_score()
    train_raw, test_raw, _ = load_raw_data()
    train = create_stratified_folds(make_features(train_raw), n_folds=config.N_FOLDS)
    test = make_features(test_raw)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"one_vs_rest_{args.output_name}"):
        mlflow.log_param("model_name", args.output_name)
        mlflow.log_param("target_class", args.target_class)
        mlflow.log_param("seed", config.SEED)
        mlflow.log_param("n_folds", config.N_FOLDS)
        mlflow.log_param("current_best_score", best_score)
        mlflow.log_param("features", ",".join(config.FEATURE_COLUMNS))
        mlflow.log_params(_lgbm_binary_params())
        result = train_specialist(
            train=train,
            test=test,
            target_class=args.target_class,
            output_name=args.output_name,
            save_models=not args.no_save_models,
        )
        mlflow.log_metric("cv_mean_ovr_balanced_accuracy", result.mean_balanced_accuracy)
        mlflow.log_metric("cv_mean_ovr_log_loss", result.mean_log_loss)
        mlflow.log_artifact(str(result.oof_feature_path))
        mlflow.log_artifact(str(result.test_feature_path))

    append_experiment_log(
        f"train {args.target_class}-vs-rest LightGBM specialist for hierarchical gated ensemble",
        result.mean_balanced_accuracy,
        None,
    )
    print(f"Model: {result.model_name}")
    print(f"Target class: {result.target_class}")
    print(f"Mean one-vs-rest balanced accuracy: {result.mean_balanced_accuracy:.8f}")
    print(f"Mean one-vs-rest log loss: {result.mean_log_loss:.8f}")
    print(f"OOF scalar features: {result.oof_feature_path}")
    print(f"Test scalar features: {result.test_feature_path}")
    print(f"Current champion threshold: {best_score:.8f}")


def train_specialist(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_class: str,
    output_name: str,
    save_models: bool,
) -> OneVsRestResult:
    """Train one fold-safe one-vs-rest binary specialist."""
    oof_positive_proba = np.zeros(len(train), dtype=np.float64)
    test_positive_proba = np.zeros(len(test), dtype=np.float64)
    fold_scores: list[float] = []
    fold_losses: list[float] = []

    for fold in range(config.N_FOLDS):
        train_idx = train[config.FOLD_COLUMN] != fold
        valid_idx = train[config.FOLD_COLUMN] == fold
        x_train = train.loc[train_idx, config.FEATURE_COLUMNS]
        x_valid = train.loc[valid_idx, config.FEATURE_COLUMNS]
        y_train = (train.loc[train_idx, config.TARGET_COLUMN] == target_class).astype(int).to_numpy()
        y_valid = (train.loc[valid_idx, config.TARGET_COLUMN] == target_class).astype(int).to_numpy()

        estimator = _build_pipeline()
        estimator.fit(x_train, y_train)
        valid_proba = estimator.predict_proba(x_valid)
        fold_test_proba = estimator.predict_proba(test[config.FEATURE_COLUMNS])
        oof_positive_proba[valid_idx.to_numpy()] = valid_proba[:, 1]
        test_positive_proba += fold_test_proba[:, 1] / config.N_FOLDS
        fold_pred = valid_proba[:, 1] >= 0.5
        fold_score = balanced_accuracy_score(y_valid, fold_pred)
        fold_loss = log_loss(y_valid, valid_proba, labels=[0, 1])
        fold_scores.append(float(fold_score))
        fold_losses.append(float(fold_loss))

        if save_models:
            model_path = config.MODELS_DIR / f"{output_name}_fold_{fold}.joblib"
            joblib.dump(estimator, model_path)
        print(f"{output_name} fold {fold}: ovr_balanced_accuracy={fold_score:.6f}, ovr_log_loss={fold_loss:.6f}")

    output_dir = config.ENSEMBLE_DIR / "scalar_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / f"{output_name}_oof_features.csv"
    test_path = output_dir / f"{output_name}_test_features.csv"
    _write_features(train, oof_positive_proba, target_class, oof_path, include_target=True)
    _write_features(test, test_positive_proba, target_class, test_path, include_target=False)
    return OneVsRestResult(
        model_name=output_name,
        target_class=target_class,
        mean_balanced_accuracy=float(np.mean(fold_scores)),
        mean_log_loss=float(np.mean(fold_losses)),
        oof_feature_path=oof_path,
        test_feature_path=test_path,
    )


def _build_pipeline() -> Pipeline:
    """Build the one-vs-rest specialist pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", LGBMClassifier(**_lgbm_binary_params()))])


def _lgbm_binary_params() -> dict[str, Any]:
    """Build LightGBM parameters for a one-vs-rest specialist."""
    return {
        "objective": "binary",
        "class_weight": "balanced",
        "n_estimators": 520,
        "max_depth": -1,
        "num_leaves": 96,
        "min_child_samples": 60,
        "learning_rate": 0.035,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "reg_lambda": 7.0,
        "reg_alpha": 3.0,
        "random_state": config.SEED,
        "n_jobs": config.N_JOBS,
        "verbosity": -1,
    }


def _write_features(
    data: pd.DataFrame,
    positive_proba: np.ndarray,
    target_class: str,
    path: Path,
    *,
    include_target: bool,
) -> None:
    """Write scalar features for the target one-vs-rest specialist."""
    clipped = np.clip(positive_proba, 1e-8, 1.0 - 1e-8)
    prefix = f"ovr_{target_class.lower()}"
    columns = [config.ID_COLUMN]
    if include_target:
        columns.extend([config.FOLD_COLUMN, config.TARGET_COLUMN])
    output = data[columns].rename(
        columns={config.ID_COLUMN: "id", config.FOLD_COLUMN: "fold", config.TARGET_COLUMN: "y_true"}
    )
    output[f"{prefix}__prob"] = clipped
    output[f"{prefix}__logit_margin"] = np.log(clipped) - np.log1p(-clipped)
    output[f"{prefix}__confidence"] = np.abs(clipped - 0.5) * 2.0
    output.to_csv(path, index=False)


if __name__ == "__main__":
    main()
