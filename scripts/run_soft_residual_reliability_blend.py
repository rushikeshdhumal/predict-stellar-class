"""Run a soft residual reliability blend anchored on the current champion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import load_raw_data, make_features
from src.ensemble import load_probability_artifact
from src.meta_features import CONTEXT_CATEGORICAL_FEATURES, CONTEXT_NUMERIC_FEATURES
from src.utils import (
    append_decision_entry,
    append_experiment_log,
    ensure_output_dirs,
    get_best_score,
    update_best_score,
    update_best_submission,
)


DEFAULT_CHAMPION_OOF = config.STACKING_DIR / "meta_rich_lgbm_active_rescue_stronger_reg_oof.csv"
DEFAULT_CHAMPION_TEST = config.STACKING_DIR / "meta_rich_lgbm_active_rescue_stronger_reg_test_proba.csv"
DEFAULT_GQ_SCALAR_OOF = config.ENSEMBLE_DIR / "scalar_features" / "scalar_binary_galaxy_qso_lgbm_oof_features.csv"
DEFAULT_GQ_SCALAR_TEST = config.ENSEMBLE_DIR / "scalar_features" / "scalar_binary_galaxy_qso_lgbm_test_features.csv"
DEFAULT_STAR_OVR_OOF = config.ENSEMBLE_DIR / "scalar_features" / "one_vs_rest_star_lgbm_oof_features.csv"
DEFAULT_STAR_OVR_TEST = config.ENSEMBLE_DIR / "scalar_features" / "one_vs_rest_star_lgbm_test_features.csv"


@dataclass(frozen=True)
class ResidualFeatureSet:
    """Aligned feature matrices for the soft residual model.

    Attributes:
        train_features: Training features aligned to champion OOF rows.
        test_features: Test features aligned to champion test rows.
        numeric_columns: Numeric feature columns.
        categorical_columns: Categorical feature columns.
        y_true: Training labels aligned to ``train_features``.
        folds: Fold assignments aligned to ``train_features``.
        champion_oof: Champion OOF probability artifact.
        champion_test: Champion test probability artifact.
    """

    train_features: pd.DataFrame
    test_features: pd.DataFrame
    numeric_columns: list[str]
    categorical_columns: list[str]
    y_true: np.ndarray
    folds: np.ndarray
    champion_oof: pd.DataFrame
    champion_test: pd.DataFrame


@dataclass(frozen=True)
class BlendCandidate:
    """One residual blend candidate."""

    alpha: float
    balanced_accuracy: float
    global_balanced_accuracy: float
    log_loss: float
    fold_scores: list[float]
    fold_losses: list[float]
    oof_proba: np.ndarray
    test_proba: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-oof-path", type=Path, default=DEFAULT_CHAMPION_OOF)
    parser.add_argument("--champion-test-proba-path", type=Path, default=DEFAULT_CHAMPION_TEST)
    parser.add_argument("--gq-scalar-oof-path", type=Path, default=DEFAULT_GQ_SCALAR_OOF)
    parser.add_argument("--gq-scalar-test-path", type=Path, default=DEFAULT_GQ_SCALAR_TEST)
    parser.add_argument("--star-ovr-oof-path", type=Path, default=DEFAULT_STAR_OVR_OOF)
    parser.add_argument("--star-ovr-test-path", type=Path, default=DEFAULT_STAR_OVR_TEST)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.02, 0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--champion-error-weight", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    """Run the fold-safe soft residual reliability experiment."""
    args = parse_args()
    ensure_output_dirs()
    previous_best = get_best_score()
    features = _build_feature_set(args)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"soft_residual_reliability_{args.output_name}"):
        mlflow.log_param("champion_oof_path", str(args.champion_oof_path))
        mlflow.log_param("champion_test_proba_path", str(args.champion_test_proba_path))
        mlflow.log_param("gq_scalar_oof_path", str(args.gq_scalar_oof_path))
        mlflow.log_param("gq_scalar_test_path", str(args.gq_scalar_test_path))
        mlflow.log_param("star_ovr_oof_path", str(args.star_ovr_oof_path))
        mlflow.log_param("star_ovr_test_path", str(args.star_ovr_test_path))
        mlflow.log_param("alphas", ",".join(str(alpha) for alpha in args.alphas))
        mlflow.log_param("champion_error_weight", args.champion_error_weight)
        mlflow.log_param("current_best_score", previous_best)
        mlflow.log_params(_residual_lgbm_params())

        residual_oof, residual_test = _train_residual_model(features, args.champion_error_weight)
        candidates = _evaluate_blends(features, residual_oof, residual_test, args.alphas)
        result = _save_result(args.output_name, features, candidates, previous_best)

        for candidate in candidates:
            suffix = f"alpha_{candidate.alpha:g}".replace(".", "p")
            mlflow.log_metric(f"balanced_accuracy_{suffix}", candidate.balanced_accuracy)
            mlflow.log_metric(f"global_balanced_accuracy_{suffix}", candidate.global_balanced_accuracy)
            mlflow.log_metric(f"log_loss_{suffix}", candidate.log_loss)
        mlflow.log_metric("best_balanced_accuracy", result["best"].balanced_accuracy)
        mlflow.log_metric("best_global_balanced_accuracy", result["best"].global_balanced_accuracy)
        mlflow.log_metric("best_log_loss", result["best"].log_loss)
        mlflow.log_param("best_alpha", result["best"].alpha)
        mlflow.log_param("improved", result["improved"])
        mlflow.log_artifact(str(result["summary_path"]))
        mlflow.log_artifact(str(result["metrics_path"]))
        mlflow.log_artifact(str(result["oof_path"]))
        mlflow.log_artifact(str(result["test_proba_path"]))
        if result["submission_path"] is not None:
            mlflow.log_artifact(str(result["submission_path"]))

    best = result["best"]
    print(f"Best alpha: {best.alpha:g}")
    print(f"Mean fold balanced accuracy: {best.balanced_accuracy:.8f}")
    print(f"Global OOF balanced accuracy: {best.global_balanced_accuracy:.8f}")
    print(f"Mean fold log loss: {best.log_loss:.8f}")
    print(f"Previous champion threshold: {previous_best:.8f}")
    print(f"Decision: {'ACCEPTED' if result['improved'] else 'rejected'}")
    print(f"OOF artifact: {result['oof_path']}")
    print(f"Test probability artifact: {result['test_proba_path']}")
    print(f"Submission artifact: {result['submission_path']}")


def _build_feature_set(args: argparse.Namespace) -> ResidualFeatureSet:
    """Build aligned residual-model features from champion and specialist artifacts."""
    champion_oof = load_probability_artifact(args.champion_oof_path)
    champion_test = load_probability_artifact(args.champion_test_proba_path)
    gq_oof = pd.read_csv(args.gq_scalar_oof_path)
    gq_test = pd.read_csv(args.gq_scalar_test_path)
    star_oof = pd.read_csv(args.star_ovr_oof_path)
    star_test = pd.read_csv(args.star_ovr_test_path)
    _validate_alignment(champion_oof, champion_test, [gq_oof, star_oof], [gq_test, star_test])

    train_numeric = _probability_features(champion_oof, prefix="champion")
    test_numeric = _probability_features(champion_test, prefix="champion")
    train_numeric = pd.concat(
        [train_numeric, _scalar_features(gq_oof), _scalar_features(star_oof)],
        axis=1,
    )
    test_numeric = pd.concat(
        [test_numeric, _scalar_features(gq_test), _scalar_features(star_test)],
        axis=1,
    )
    train_context, test_context = _context_features(champion_oof, champion_test)
    train_numeric = pd.concat([train_numeric, train_context[CONTEXT_NUMERIC_FEATURES]], axis=1)
    test_numeric = pd.concat([test_numeric, test_context[CONTEXT_NUMERIC_FEATURES]], axis=1)
    train_categorical = train_context[CONTEXT_CATEGORICAL_FEATURES].reset_index(drop=True)
    test_categorical = test_context[CONTEXT_CATEGORICAL_FEATURES].reset_index(drop=True)
    train_features = pd.concat([train_numeric, train_categorical], axis=1)
    test_features = pd.concat([test_numeric, test_categorical], axis=1)
    numeric_columns = train_numeric.columns.tolist()
    categorical_columns = CONTEXT_CATEGORICAL_FEATURES.copy()
    _validate_no_identifier_features(numeric_columns, categorical_columns)
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Train and test residual feature columns are not aligned.")
    return ResidualFeatureSet(
        train_features=train_features,
        test_features=test_features,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        y_true=champion_oof["y_true"].to_numpy(),
        folds=champion_oof["fold"].to_numpy(),
        champion_oof=champion_oof,
        champion_test=champion_test,
    )


def _train_residual_model(features: ResidualFeatureSet, champion_error_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Train the weighted residual model fold-wise and return OOF/test probabilities."""
    residual_oof = np.zeros((len(features.train_features), len(config.CLASS_LABELS)), dtype=np.float64)
    residual_test = np.zeros((len(features.test_features), len(config.CLASS_LABELS)), dtype=np.float64)
    champion_pred = _labels_from_proba(features.champion_oof[config.PROBA_COLUMNS].to_numpy(dtype=np.float64))
    champion_wrong = champion_pred != features.y_true

    for fold in range(config.N_FOLDS):
        train_idx = features.folds != fold
        valid_idx = features.folds == fold
        sample_weight = np.ones(int(train_idx.sum()), dtype=np.float64)
        sample_weight += champion_error_weight * champion_wrong[train_idx]
        estimator = _build_pipeline(features)
        estimator.fit(
            features.train_features.loc[train_idx],
            features.y_true[train_idx],
            model__sample_weight=sample_weight,
        )
        _validate_estimator_classes(estimator)
        valid_proba = estimator.predict_proba(features.train_features.loc[valid_idx])
        fold_test_proba = estimator.predict_proba(features.test_features)
        residual_oof[valid_idx] = valid_proba
        residual_test += fold_test_proba / config.N_FOLDS
        valid_pred = _labels_from_proba(valid_proba)
        valid_score = balanced_accuracy_score(features.y_true[valid_idx], valid_pred)
        valid_loss = log_loss(features.y_true[valid_idx], valid_proba, labels=config.CLASS_LABELS)
        print(
            f"residual fold {fold}: balanced_accuracy={valid_score:.8f}, log_loss={valid_loss:.8f}",
            flush=True,
        )

    _validate_probability_rows(residual_oof, context="residual OOF probabilities")
    _validate_probability_rows(residual_test, context="residual test probabilities")
    return residual_oof, residual_test


def _evaluate_blends(
    features: ResidualFeatureSet,
    residual_oof: np.ndarray,
    residual_test: np.ndarray,
    alphas: list[float],
) -> list[BlendCandidate]:
    """Evaluate logit-space blends between champion and residual model."""
    champion_oof = features.champion_oof[config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
    champion_test = features.champion_test[config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
    candidates: list[BlendCandidate] = []
    for alpha in alphas:
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("Blend alpha values must be in the open interval (0, 1).")
        oof_proba = _logit_blend(champion_oof, residual_oof, alpha)
        test_proba = _logit_blend(champion_test, residual_test, alpha)
        fold_scores: list[float] = []
        fold_losses: list[float] = []
        for fold in range(config.N_FOLDS):
            valid_idx = features.folds == fold
            valid_pred = _labels_from_proba(oof_proba[valid_idx])
            fold_scores.append(float(balanced_accuracy_score(features.y_true[valid_idx], valid_pred)))
            fold_losses.append(float(log_loss(features.y_true[valid_idx], oof_proba[valid_idx], labels=config.CLASS_LABELS)))
        global_pred = _labels_from_proba(oof_proba)
        candidates.append(
            BlendCandidate(
                alpha=alpha,
                balanced_accuracy=float(np.mean(fold_scores)),
                global_balanced_accuracy=float(balanced_accuracy_score(features.y_true, global_pred)),
                log_loss=float(np.mean(fold_losses)),
                fold_scores=fold_scores,
                fold_losses=fold_losses,
                oof_proba=oof_proba,
                test_proba=test_proba,
            )
        )
    return candidates


def _build_pipeline(features: ResidualFeatureSet) -> Pipeline:
    """Build the residual reliability model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features.numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features.categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", LGBMClassifier(**_residual_lgbm_params()))])


def _residual_lgbm_params() -> dict[str, Any]:
    """Return conservative parameters for the residual reliability model."""
    return {
        "objective": "multiclass",
        "num_class": len(config.CLASS_LABELS),
        "class_weight": "balanced",
        "n_estimators": 220,
        "learning_rate": 0.025,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 1500,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_lambda": 45.0,
        "reg_alpha": 12.0,
        "random_state": config.SEED,
        "n_jobs": config.N_JOBS,
        "verbosity": -1,
    }


def _probability_features(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Create compact reliability features from a probability artifact."""
    proba = frame[config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
    clipped = np.clip(proba, 1e-8, 1.0)
    logits = np.log(clipped)
    sorted_proba = np.sort(proba, axis=1)
    features: dict[str, np.ndarray] = {
        f"{prefix}__entropy": -(clipped * np.log(clipped)).sum(axis=1),
        f"{prefix}__top_margin": sorted_proba[:, -1] - sorted_proba[:, -2],
        f"{prefix}__max_proba": sorted_proba[:, -1],
    }
    for class_index, label in enumerate(config.CLASS_LABELS):
        features[f"{prefix}__proba_{label}"] = proba[:, class_index]
        features[f"{prefix}__log_proba_{label}"] = logits[:, class_index]
    for left_index, left_label in enumerate(config.CLASS_LABELS):
        for right_index, right_label in enumerate(config.CLASS_LABELS):
            if left_index >= right_index:
                continue
            features[f"{prefix}__logit_gap_{left_label}_minus_{right_label}"] = (
                logits[:, left_index] - logits[:, right_index]
            )
    return pd.DataFrame(features)


def _scalar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return scalar specialist feature columns from an aligned artifact."""
    feature_columns = [column for column in frame.columns if column not in {"id", "fold", "y_true"}]
    if not feature_columns:
        raise ValueError("Scalar specialist artifact contains no feature columns.")
    return frame[feature_columns].reset_index(drop=True)


def _context_features(reference_oof: pd.DataFrame, reference_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and align raw-feature context by competition id."""
    train_raw, test_raw, _ = load_raw_data()
    train_features = make_features(train_raw).set_index(config.ID_COLUMN)
    test_features = make_features(test_raw).set_index(config.ID_COLUMN)
    columns = CONTEXT_NUMERIC_FEATURES + CONTEXT_CATEGORICAL_FEATURES
    train_context = train_features.loc[reference_oof["id"].to_numpy(), columns]
    test_context = test_features.loc[reference_test["id"].to_numpy(), columns]
    return train_context.reset_index(drop=True), test_context.reset_index(drop=True)


def _logit_blend(champion: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    """Blend champion and residual probabilities in logit space."""
    champion_log = np.log(np.clip(champion, 1e-8, 1.0))
    residual_log = np.log(np.clip(residual, 1e-8, 1.0))
    blended_log = (1.0 - alpha) * champion_log + alpha * residual_log
    blended_log -= blended_log.max(axis=1, keepdims=True)
    blended = np.exp(blended_log)
    blended /= blended.sum(axis=1, keepdims=True)
    _validate_probability_rows(blended, context="blended probabilities")
    return blended


def _save_result(
    output_name: str,
    features: ResidualFeatureSet,
    candidates: list[BlendCandidate],
    previous_best: float,
) -> dict[str, Any]:
    """Save the selected blend artifacts and update logs."""
    best = max(candidates, key=lambda candidate: candidate.balanced_accuracy)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    oof_path = config.STACKING_DIR / f"{output_name}_oof.csv"
    test_proba_path = config.STACKING_DIR / f"{output_name}_test_proba.csv"
    summary_path = config.STACKING_DIR / f"{output_name}_summary_{timestamp}.csv"
    metrics_path = config.STACKING_DIR / f"{output_name}_metrics_{timestamp}.csv"
    _write_oof(features.champion_oof, best.oof_proba, oof_path)
    _write_test(features.champion_test, best.test_proba, test_proba_path)
    _write_summary(candidates, summary_path)
    _write_metrics(features.y_true, best, metrics_path)
    improved = best.balanced_accuracy > previous_best
    submission_path = _write_submission(output_name, features.champion_test, best.test_proba) if improved else None
    change = f"run soft residual reliability logit blend anchored on rich LGBM champion; alpha={best.alpha:g}"
    append_experiment_log(change, best.balanced_accuracy, submission_path)
    append_decision_entry(change, best.balanced_accuracy, previous_best, improved)
    if improved and submission_path is not None:
        update_best_score(best.balanced_accuracy)
        update_best_submission(submission_path)
    return {
        "best": best,
        "oof_path": oof_path,
        "test_proba_path": test_proba_path,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "submission_path": submission_path,
        "improved": improved,
    }


def _write_summary(candidates: list[BlendCandidate], path: Path) -> None:
    """Write alpha-grid summary metrics."""
    rows = [
        {
            "alpha": candidate.alpha,
            "balanced_accuracy": candidate.balanced_accuracy,
            "global_balanced_accuracy": candidate.global_balanced_accuracy,
            "log_loss": candidate.log_loss,
            **{f"fold_{fold}_balanced_accuracy": score for fold, score in enumerate(candidate.fold_scores)},
            **{f"fold_{fold}_log_loss": loss for fold, loss in enumerate(candidate.fold_losses)},
        }
        for candidate in candidates
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_metrics(y_true: np.ndarray, best: BlendCandidate, path: Path) -> None:
    """Write recall and confusion metrics for the selected blend."""
    pred = _labels_from_proba(best.oof_proba)
    confusion = confusion_matrix(y_true, pred, labels=config.CLASS_LABELS)
    recalls = recall_score(y_true, pred, labels=config.CLASS_LABELS, average=None)
    rows = [
        {"metric": "balanced_accuracy", "label": "ALL", "value": best.balanced_accuracy},
        {"metric": "global_balanced_accuracy", "label": "ALL", "value": best.global_balanced_accuracy},
        {"metric": "log_loss", "label": "ALL", "value": best.log_loss},
    ]
    rows.extend({"metric": "recall", "label": label, "value": float(value)} for label, value in zip(config.CLASS_LABELS, recalls))
    for true_index, true_label in enumerate(config.CLASS_LABELS):
        for pred_index, pred_label in enumerate(config.CLASS_LABELS):
            rows.append(
                {
                    "metric": "confusion_count",
                    "label": f"true_{true_label}__pred_{pred_label}",
                    "value": int(confusion[true_index, pred_index]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_oof(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write an OOF probability artifact."""
    output = reference[["id", "fold", "y_true"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_test(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write an averaged test probability artifact."""
    output = reference[["id"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_submission(output_name: str, reference_test: pd.DataFrame, proba: np.ndarray) -> Path:
    """Write a competition submission from blended probabilities."""
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    if not sample[config.ID_COLUMN].equals(reference_test["id"]):
        raise ValueError("Sample submission ids are not aligned with test probabilities.")
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    submission = sample.copy()
    submission[config.TARGET_COLUMN] = labels[proba.argmax(axis=1)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.SUBMISSIONS_DIR / f"submission_{output_name}_{timestamp}.csv"
    submission.to_csv(path, index=False)
    return path


def _validate_alignment(
    champion_oof: pd.DataFrame,
    champion_test: pd.DataFrame,
    scalar_oof_frames: list[pd.DataFrame],
    scalar_test_frames: list[pd.DataFrame],
) -> None:
    """Validate all residual inputs are row-aligned."""
    for frame in scalar_oof_frames:
        for column in ["id", "fold", "y_true"]:
            if column not in frame.columns:
                raise ValueError(f"Scalar OOF artifact is missing `{column}`.")
            if not champion_oof[column].equals(frame[column]):
                raise ValueError(f"Scalar OOF artifact is not aligned on `{column}`.")
    for frame in scalar_test_frames:
        if "id" not in frame.columns:
            raise ValueError("Scalar test artifact is missing `id`.")
        if not champion_test["id"].equals(frame["id"]):
            raise ValueError("Scalar test artifact is not aligned on `id`.")


def _validate_estimator_classes(estimator: Pipeline) -> None:
    """Validate estimator class order matches probability column order."""
    classes = estimator.named_steps["model"].classes_.tolist()
    if classes != config.CLASS_LABELS:
        raise ValueError(f"Unexpected class order {classes}; expected {config.CLASS_LABELS}.")


def _validate_probability_rows(proba: np.ndarray, *, context: str) -> None:
    """Validate finite probability rows with unit sums."""
    if not np.isfinite(proba).all():
        raise ValueError(f"{context} contains non-finite values.")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{context} rows do not sum to 1.")


def _validate_no_identifier_features(numeric_columns: list[str], categorical_columns: list[str]) -> None:
    """Fail fast if an identifier column leaks into model features."""
    forbidden = {config.ID_COLUMN, "id", "obj_ID"}
    leaked = forbidden.intersection(numeric_columns).union(forbidden.intersection(categorical_columns))
    if leaked:
        raise ValueError(f"Identifier columns are not allowed as features: {sorted(leaked)}")


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert probabilities to configured class labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


if __name__ == "__main__":
    main()
