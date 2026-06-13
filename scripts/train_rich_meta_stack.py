"""Train rich-feature logistic or LightGBM meta-stackers from OOF artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Literal

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.meta_features import (
    RichMetaFeatures,
    build_rich_meta_features,
    default_rich_base_names,
    resolve_probability_paths,
)
from src.utils import (
    append_decision_entry,
    append_experiment_log,
    ensure_output_dirs,
    get_best_score,
    update_best_score,
    update_best_submission,
)

ModelType = Literal["logistic", "lightgbm"]
LgbmVariant = Literal["conservative", "more_trees", "stronger_reg"]


@dataclass(frozen=True)
class StackCandidate:
    """Evaluation result for one rich meta-stacker candidate."""

    model_type: ModelType
    candidate_name: str
    balanced_accuracy: float
    global_balanced_accuracy: float
    log_loss: float
    fold_scores: list[float]
    fold_losses: list[float]
    oof_proba: np.ndarray
    test_proba: np.ndarray


@dataclass(frozen=True)
class StackRunResult:
    """Saved artifact metadata for a rich meta-stacking run."""

    best: StackCandidate
    oof_path: Path
    test_proba_path: Path
    summary_path: Path
    metrics_path: Path
    submission_path: Path | None
    previous_best: float
    improved: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=["logistic", "lightgbm"], required=True, help="Meta-stacker family.")
    parser.add_argument("--output-name", required=True, help="Stable output artifact name.")
    parser.add_argument("--base-names", nargs="+", default=default_rich_base_names(), help="Base artifact names.")
    parser.add_argument("--oof-paths", nargs="+", type=Path, default=None, help="Explicit OOF artifact paths.")
    parser.add_argument(
        "--test-proba-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit test probability artifact paths.",
    )
    parser.add_argument(
        "--scalar-oof-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Optional aligned scalar OOF feature artifacts.",
    )
    parser.add_argument(
        "--scalar-test-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Optional aligned scalar test feature artifacts.",
    )
    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=[0.025, 0.05, 0.075, 0.10, 0.125],
        help="Logistic C grid. Ignored for LightGBM.",
    )
    parser.add_argument(
        "--lgbm-variant",
        choices=["conservative", "more_trees", "stronger_reg"],
        default="conservative",
        help="LightGBM meta-stacker parameter variant. Ignored for logistic.",
    )
    parser.add_argument("--no-context", action="store_true", help="Disable small raw-feature context columns.")
    return parser.parse_args()


def main() -> None:
    """Train and evaluate one rich meta-stacking experiment."""
    args = parse_args()
    ensure_output_dirs()
    oof_paths, test_proba_paths = _resolve_paths(args)
    previous_best = get_best_score()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"rich_meta_{args.model_type}_{args.output_name}"):
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("base_models", ",".join(args.base_names))
        mlflow.log_param("oof_paths", ",".join(str(path) for path in oof_paths))
        mlflow.log_param("test_proba_paths", ",".join(str(path) for path in test_proba_paths))
        mlflow.log_param("scalar_oof_paths", ",".join(str(path) for path in args.scalar_oof_paths or []))
        mlflow.log_param("scalar_test_paths", ",".join(str(path) for path in args.scalar_test_paths or []))
        mlflow.log_param("include_context", not args.no_context)
        mlflow.log_param("current_best_score", previous_best)
        if args.model_type == "logistic":
            mlflow.log_param("c_values", ",".join(str(value) for value in args.c_values))
        else:
            mlflow.log_param("lgbm_variant", args.lgbm_variant)
            mlflow.log_params(_lightgbm_params(args.lgbm_variant))

        meta = build_rich_meta_features(
            oof_paths,
            test_proba_paths,
            args.base_names,
            include_context=not args.no_context,
            scalar_oof_paths=args.scalar_oof_paths,
            scalar_test_paths=args.scalar_test_paths,
        )
        candidates = _evaluate_candidates(args.model_type, meta, args.c_values, args.lgbm_variant)
        result = _save_and_log_result(args, meta, candidates, previous_best)

        for candidate in candidates:
            metric_suffix = candidate.candidate_name.replace(".", "_")
            mlflow.log_metric(f"balanced_accuracy_{metric_suffix}", candidate.balanced_accuracy)
            mlflow.log_metric(f"global_balanced_accuracy_{metric_suffix}", candidate.global_balanced_accuracy)
            mlflow.log_metric(f"log_loss_{metric_suffix}", candidate.log_loss)
        mlflow.log_metric("best_balanced_accuracy", result.best.balanced_accuracy)
        mlflow.log_metric("best_global_balanced_accuracy", result.best.global_balanced_accuracy)
        mlflow.log_metric("best_log_loss", result.best.log_loss)
        mlflow.log_param("best_candidate", result.best.candidate_name)
        mlflow.log_param("improved", result.improved)
        mlflow.log_artifact(str(result.oof_path))
        mlflow.log_artifact(str(result.test_proba_path))
        mlflow.log_artifact(str(result.summary_path))
        mlflow.log_artifact(str(result.metrics_path))
        if result.submission_path is not None:
            mlflow.log_artifact(str(result.submission_path))

    _print_result(result)


def _resolve_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Resolve explicit or default probability artifact paths."""
    if args.oof_paths is None and args.test_proba_paths is None:
        return resolve_probability_paths(args.base_names)
    if args.oof_paths is None or args.test_proba_paths is None:
        raise ValueError("Provide both --oof-paths and --test-proba-paths, or neither.")
    if len(args.oof_paths) != len(args.test_proba_paths):
        raise ValueError("OOF and test artifact path counts must match.")
    if len(args.oof_paths) != len(args.base_names):
        raise ValueError("Base name count must match artifact path count.")
    return args.oof_paths, args.test_proba_paths


def _evaluate_candidates(
    model_type: ModelType,
    meta: RichMetaFeatures,
    c_values: list[float],
    lgbm_variant: LgbmVariant,
) -> list[StackCandidate]:
    """Evaluate all requested candidates."""
    if model_type == "logistic":
        return [
            _evaluate_one("logistic", f"c_{c_value:g}", meta, c_value=c_value, lgbm_variant=lgbm_variant)
            for c_value in c_values
        ]
    return [_evaluate_one("lightgbm", lgbm_variant, meta, c_value=None, lgbm_variant=lgbm_variant)]


def _evaluate_one(
    model_type: ModelType,
    candidate_name: str,
    meta: RichMetaFeatures,
    *,
    c_value: float | None,
    lgbm_variant: LgbmVariant,
) -> StackCandidate:
    """Evaluate one fold-wise rich meta-stacker candidate."""
    oof_proba = np.zeros((len(meta.train_features), len(config.CLASS_LABELS)), dtype=np.float64)
    test_proba = np.zeros((len(meta.test_features), len(config.CLASS_LABELS)), dtype=np.float64)
    fold_scores: list[float] = []
    fold_losses: list[float] = []

    for fold in range(config.N_FOLDS):
        train_idx = meta.folds != fold
        valid_idx = meta.folds == fold
        estimator = _build_pipeline(model_type, meta, c_value=c_value, lgbm_variant=lgbm_variant)
        estimator.fit(meta.train_features.loc[train_idx], meta.y_true[train_idx])
        _validate_estimator_classes(estimator)
        valid_proba = estimator.predict_proba(meta.train_features.loc[valid_idx])
        fold_test_proba = estimator.predict_proba(meta.test_features)
        oof_proba[valid_idx] = valid_proba
        test_proba += fold_test_proba / config.N_FOLDS
        valid_pred = _labels_from_proba(valid_proba)
        fold_scores.append(float(balanced_accuracy_score(meta.y_true[valid_idx], valid_pred)))
        fold_losses.append(float(log_loss(meta.y_true[valid_idx], valid_proba, labels=config.CLASS_LABELS)))
        print(
            f"{model_type} {candidate_name} fold {fold}: "
            f"balanced_accuracy={fold_scores[-1]:.8f}, log_loss={fold_losses[-1]:.8f}",
            flush=True,
        )

    _validate_probability_rows(oof_proba, context=f"{candidate_name} OOF probabilities")
    _validate_probability_rows(test_proba, context=f"{candidate_name} test probabilities")
    oof_pred = _labels_from_proba(oof_proba)
    return StackCandidate(
        model_type=model_type,
        candidate_name=candidate_name,
        balanced_accuracy=float(np.mean(fold_scores)),
        global_balanced_accuracy=float(balanced_accuracy_score(meta.y_true, oof_pred)),
        log_loss=float(np.mean(fold_losses)),
        fold_scores=fold_scores,
        fold_losses=fold_losses,
        oof_proba=oof_proba,
        test_proba=test_proba,
    )


def _build_pipeline(
    model_type: ModelType,
    meta: RichMetaFeatures,
    *,
    c_value: float | None,
    lgbm_variant: LgbmVariant,
) -> Pipeline:
    """Build a fold-local preprocessing and classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), meta.numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), meta.categorical_columns),
        ],
        remainder="drop",
    )
    if model_type == "logistic":
        if c_value is None:
            raise ValueError("Logistic candidates require a C value.")
        alpha = 1.0 / (c_value * len(meta.train_features))
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            n_jobs=config.N_JOBS,
            random_state=config.SEED,
        )
    else:
        model = LGBMClassifier(**_lightgbm_params(lgbm_variant))
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _lightgbm_params(variant: LgbmVariant) -> dict[str, Any]:
    """Return conservative LightGBM meta-stacker parameters."""
    params: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": len(config.CLASS_LABELS),
        "class_weight": "balanced",
        "n_estimators": 260,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 1200,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 20.0,
        "reg_alpha": 5.0,
        "random_state": config.SEED,
        "n_jobs": config.N_JOBS,
        "verbosity": -1,
    }
    if variant == "more_trees":
        params.update(
            {
                "n_estimators": 380,
                "learning_rate": 0.024,
            }
        )
    elif variant == "stronger_reg":
        params.update(
            {
                "n_estimators": 240,
                "max_depth": 3,
                "num_leaves": 7,
                "min_child_samples": 1800,
                "reg_lambda": 35.0,
                "reg_alpha": 10.0,
                "colsample_bytree": 0.75,
            }
        )
    return params


def _save_and_log_result(
    args: argparse.Namespace,
    meta: RichMetaFeatures,
    candidates: list[StackCandidate],
    previous_best: float,
) -> StackRunResult:
    """Save artifacts, update logs, and accept the run if it beats the threshold."""
    best = max(candidates, key=lambda candidate: candidate.balanced_accuracy)
    oof_path = config.STACKING_DIR / f"{args.output_name}_oof.csv"
    test_proba_path = config.STACKING_DIR / f"{args.output_name}_test_proba.csv"
    summary_path = _write_summary(args.output_name, candidates)
    metrics_path = _write_metrics(args.output_name, best, meta)
    _write_oof(meta.reference_oof, best.oof_proba, oof_path)
    _write_test(meta.reference_test, best.test_proba, test_proba_path)

    improved = best.balanced_accuracy > previous_best
    submission_path = _write_submission(args.output_name, meta.reference_test, best.test_proba) if improved else None
    change = _change_description(args, best)
    append_experiment_log(change, best.balanced_accuracy, submission_path)
    append_decision_entry(change, best.balanced_accuracy, previous_best, improved)
    if improved and submission_path is not None:
        update_best_score(best.balanced_accuracy)
        update_best_submission(submission_path)

    return StackRunResult(
        best=best,
        oof_path=oof_path,
        test_proba_path=test_proba_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        submission_path=submission_path,
        previous_best=previous_best,
        improved=improved,
    )


def _write_summary(output_name: str, candidates: list[StackCandidate]) -> Path:
    """Write candidate-level summary metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.STACKING_DIR / f"{output_name}_summary_{timestamp}.csv"
    rows = [
        {
            "model_type": candidate.model_type,
            "candidate_name": candidate.candidate_name,
            "balanced_accuracy": candidate.balanced_accuracy,
            "global_balanced_accuracy": candidate.global_balanced_accuracy,
            "log_loss": candidate.log_loss,
            **{f"fold_{fold}_balanced_accuracy": score for fold, score in enumerate(candidate.fold_scores)},
            **{f"fold_{fold}_log_loss": loss for fold, loss in enumerate(candidate.fold_losses)},
        }
        for candidate in candidates
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_metrics(output_name: str, best: StackCandidate, meta: RichMetaFeatures) -> Path:
    """Write detailed OOF diagnostics for the best candidate."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.STACKING_DIR / f"{output_name}_metrics_{timestamp}.csv"
    pred = _labels_from_proba(best.oof_proba)
    confusion = confusion_matrix(meta.y_true, pred, labels=config.CLASS_LABELS)
    per_class_recall = recall_score(meta.y_true, pred, labels=config.CLASS_LABELS, average=None)
    rows = [
        {"metric": "balanced_accuracy", "label": "ALL", "value": best.balanced_accuracy},
        {"metric": "global_balanced_accuracy", "label": "ALL", "value": best.global_balanced_accuracy},
        {"metric": "log_loss", "label": "ALL", "value": best.log_loss},
    ]
    rows.extend(
        {"metric": "recall", "label": label, "value": float(value)}
        for label, value in zip(config.CLASS_LABELS, per_class_recall)
    )
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
    return path


def _write_oof(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write OOF probability artifact."""
    output = reference[["id", "fold", "y_true"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_test(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write averaged test probability artifact."""
    output = reference[["id"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_submission(output_name: str, reference_test: pd.DataFrame, proba: np.ndarray) -> Path:
    """Write a competition submission from test probabilities."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    predictions = labels[proba.argmax(axis=1)]
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    submission = sample.copy()
    if not submission[config.ID_COLUMN].equals(reference_test["id"]):
        raise ValueError("Sample submission ids are not aligned with test probability ids.")
    submission[config.TARGET_COLUMN] = predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.SUBMISSIONS_DIR / f"submission_{output_name}_{timestamp}.csv"
    submission.to_csv(path, index=False)
    return path


def _change_description(args: argparse.Namespace, best: StackCandidate) -> str:
    """Build a concise human-readable experiment description."""
    context_text = "with domain context" if not args.no_context else "without domain context"
    variant_text = f"; lgbm_variant={args.lgbm_variant}" if args.model_type == "lightgbm" else ""
    scalar_text = "; scalar specialist features" if args.scalar_oof_paths else ""
    return (
        f"train rich-feature {args.model_type} meta-stacker {context_text} over active/high-rescue bases; "
        f"best={best.candidate_name}{variant_text}{scalar_text}"
    )


def _validate_estimator_classes(estimator: Pipeline) -> None:
    """Validate estimator class order matches project probability columns."""
    classes = estimator.named_steps["model"].classes_.tolist()
    if classes != config.CLASS_LABELS:
        raise ValueError(f"Unexpected class order {classes}; expected {config.CLASS_LABELS}.")


def _validate_probability_rows(proba: np.ndarray, *, context: str) -> None:
    """Validate probability row sums and finite values."""
    if not np.isfinite(proba).all():
        raise ValueError(f"{context} contains non-finite values.")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{context} rows do not sum to 1.")


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities to configured class labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


def _print_result(result: StackRunResult) -> None:
    """Print the important run result details."""
    status = "ACCEPTED" if result.improved else "rejected"
    print(f"Best candidate: {result.best.candidate_name}")
    print(f"Mean fold balanced accuracy: {result.best.balanced_accuracy:.8f}")
    print(f"Global OOF balanced accuracy: {result.best.global_balanced_accuracy:.8f}")
    print(f"Mean fold log loss: {result.best.log_loss:.8f}")
    print(f"Previous champion threshold: {result.previous_best:.8f}")
    print(f"Decision: {status}")
    print(f"OOF artifact: {result.oof_path}")
    print(f"Test probability artifact: {result.test_proba_path}")
    print(f"Summary artifact: {result.summary_path}")
    print(f"Metrics artifact: {result.metrics_path}")
    print(f"Submission artifact: {result.submission_path}")


if __name__ == "__main__":
    main()
