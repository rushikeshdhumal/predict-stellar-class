"""Tune logistic meta-stack C values from explicit probability artifact paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss

from src import config
from src.ensemble import load_probability_artifact
from src.utils import ensure_output_dirs, get_best_score


@dataclass(frozen=True)
class MetaCandidate:
    """Evaluation result for one logistic meta-stack regularization value."""

    c_value: float
    balanced_accuracy: float
    log_loss: float
    oof_proba: np.ndarray
    test_proba: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-paths", nargs="+", type=Path, required=True, help="Base OOF artifact paths.")
    parser.add_argument(
        "--test-proba-paths",
        nargs="+",
        type=Path,
        required=True,
        help="Base test probability artifact paths in the same order as OOF paths.",
    )
    parser.add_argument("--base-names", nargs="+", required=True, help="Human-readable base artifact names.")
    parser.add_argument("--output-name", type=str, required=True, help="Output artifact name for the best C value.")
    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.25, 0.40, 0.75, 1.0],
        help="Candidate logistic regression C values.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a local C-value sensitivity grid and save the best meta-stack artifacts."""
    args = parse_args()
    _validate_args(args)
    ensure_output_dirs()
    best_score = get_best_score()
    oof_frames = [load_probability_artifact(path) for path in args.oof_paths]
    test_frames = [load_probability_artifact(path) for path in args.test_proba_paths]
    _validate_base_artifacts(oof_frames, test_frames)

    x_meta = _stack_log_probabilities(oof_frames)
    x_test_meta = _stack_log_probabilities(test_frames)
    y_true = oof_frames[0]["y_true"].to_numpy()
    folds = oof_frames[0]["fold"].to_numpy()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"meta_stack_c_grid_{args.output_name}"):
        mlflow.log_param("base_models", ",".join(args.base_names))
        mlflow.log_param("c_values", ",".join(str(value) for value in args.c_values))
        mlflow.log_param("current_best_score", best_score)
        candidates = [
            _evaluate_c_value(c_value, x_meta, x_test_meta, y_true, folds)
            for c_value in args.c_values
        ]
        summary_path = _write_summary(args.output_name, candidates)
        best = max(candidates, key=lambda candidate: candidate.balanced_accuracy)
        oof_path = config.STACKING_DIR / f"{args.output_name}_oof.csv"
        test_proba_path = config.STACKING_DIR / f"{args.output_name}_test_proba.csv"
        _write_meta_oof(oof_frames[0], best.oof_proba, oof_path)
        _write_meta_test(test_frames[0], best.test_proba, test_proba_path)

        for candidate in candidates:
            mlflow.log_metric(f"balanced_accuracy_c_{candidate.c_value:g}", candidate.balanced_accuracy)
            mlflow.log_metric(f"log_loss_c_{candidate.c_value:g}", candidate.log_loss)
        mlflow.log_metric("best_balanced_accuracy", best.balanced_accuracy)
        mlflow.log_metric("best_log_loss", best.log_loss)
        mlflow.log_param("best_c_value", best.c_value)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(oof_path))
        mlflow.log_artifact(str(test_proba_path))

    for candidate in sorted(candidates, key=lambda item: item.c_value):
        print(
            f"C={candidate.c_value:g}: "
            f"balanced_accuracy={candidate.balanced_accuracy:.8f}, "
            f"log_loss={candidate.log_loss:.8f}",
            flush=True,
        )
    print(f"Best C value: {best.c_value:g}")
    print(f"Best balanced accuracy: {best.balanced_accuracy:.8f}")
    print(f"Best log loss: {best.log_loss:.8f}")
    print(f"Current champion threshold: {best_score:.8f}")
    print(f"Summary artifact: {summary_path}")
    print(f"OOF artifact: {oof_path}")
    print(f"Test probability artifact: {test_proba_path}")


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument cardinalities."""
    if len(args.oof_paths) != len(args.test_proba_paths):
        raise ValueError("OOF and test artifact path counts must match.")
    if len(args.oof_paths) != len(args.base_names):
        raise ValueError("Base name count must match artifact path count.")


def _evaluate_c_value(
    c_value: float,
    x_meta: np.ndarray,
    x_test_meta: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
) -> MetaCandidate:
    """Evaluate one logistic regression C value with fold-wise stacking."""
    oof_proba = np.zeros((len(y_true), len(config.CLASS_LABELS)), dtype=float)
    test_proba = np.zeros((len(x_test_meta), len(config.CLASS_LABELS)), dtype=float)
    fold_losses: list[float] = []

    for fold in range(config.N_FOLDS):
        train_idx = folds != fold
        valid_idx = folds == fold
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=1000,
            random_state=config.SEED,
        )
        model.fit(x_meta[train_idx], y_true[train_idx])
        valid_proba = model.predict_proba(x_meta[valid_idx])
        oof_proba[valid_idx] = valid_proba
        test_proba += model.predict_proba(x_test_meta) / config.N_FOLDS
        fold_losses.append(float(log_loss(y_true[valid_idx], valid_proba, labels=config.CLASS_LABELS)))

    balanced_accuracy = float(balanced_accuracy_score(y_true, _labels_from_proba(oof_proba)))
    loss_value = float(np.mean(fold_losses))
    return MetaCandidate(
        c_value=c_value,
        balanced_accuracy=balanced_accuracy,
        log_loss=loss_value,
        oof_proba=oof_proba,
        test_proba=test_proba,
    )


def _stack_log_probabilities(frames: list[pd.DataFrame]) -> np.ndarray:
    """Build a meta-feature matrix from clipped log probabilities."""
    matrices = []
    for frame in frames:
        probabilities = np.clip(frame[config.PROBA_COLUMNS].to_numpy(), 1e-8, 1.0)
        matrices.append(np.log(probabilities))
    return np.hstack(matrices)


def _validate_base_artifacts(oof_frames: list[pd.DataFrame], test_frames: list[pd.DataFrame]) -> None:
    """Validate row alignment across base model artifacts."""
    reference_oof = oof_frames[0]
    reference_test = test_frames[0]
    for frame in oof_frames[1:]:
        for column in ["id", "fold", "y_true"]:
            if not reference_oof[column].equals(frame[column]):
                raise ValueError(f"OOF artifacts are not aligned on `{column}`.")
    for frame in test_frames[1:]:
        if not reference_test["id"].equals(frame["id"]):
            raise ValueError("Test probability artifacts are not aligned on `id`.")


def _write_summary(output_name: str, candidates: list[MetaCandidate]) -> Path:
    """Write the C-grid summary CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.STACKING_DIR / f"{output_name}_summary_{timestamp}.csv"
    summary = pd.DataFrame(
        {
            "c_value": [candidate.c_value for candidate in candidates],
            "balanced_accuracy": [candidate.balanced_accuracy for candidate in candidates],
            "log_loss": [candidate.log_loss for candidate in candidates],
        }
    )
    summary.to_csv(path, index=False)
    return path


def _write_meta_oof(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write meta-learner OOF probabilities."""
    output = reference[["id", "fold", "y_true"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_meta_test(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write meta-learner test probabilities."""
    output = reference[["id"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities to configured class labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


if __name__ == "__main__":
    main()
