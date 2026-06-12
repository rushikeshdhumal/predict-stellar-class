"""Tune class-specific logit biases for one probability artifact pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from src import config
from src.blending import apply_class_biases
from src.ensemble import load_probability_artifact
from src.utils import ensure_output_dirs, get_best_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-path", type=Path, required=True, help="OOF probability artifact path.")
    parser.add_argument("--test-proba-path", type=Path, required=True, help="Test probability artifact path.")
    parser.add_argument("--output-name", type=str, required=True, help="Output artifact name.")
    parser.add_argument("--bias-range", type=float, default=0.04, help="Absolute QSO/STAR logit-bias range.")
    parser.add_argument("--bias-step", type=float, default=0.002, help="QSO/STAR logit-bias step.")
    return parser.parse_args()


def main() -> None:
    """Tune biases and write adjusted OOF/test probability artifacts."""
    args = parse_args()
    ensure_output_dirs()
    oof = load_probability_artifact(args.oof_path)
    test = load_probability_artifact(args.test_proba_path)
    y_true = oof["y_true"].to_numpy()
    base_oof_proba = oof[config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
    base_test_proba = test[config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
    values = np.arange(-args.bias_range, args.bias_range + args.bias_step / 2.0, args.bias_step, dtype=np.float64)
    bias_candidates = np.array([[0.0, qso, star] for qso in values for star in values], dtype=np.float32)
    encoded_y = _encode_labels(y_true)
    base_logits = np.log(np.clip(base_oof_proba, 1e-12, 1.0)).astype(np.float32)
    scores = _score_bias_candidates(base_logits, encoded_y, bias_candidates, batch_size=64)
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])
    best_biases = bias_candidates[best_index].astype(np.float64)
    adjusted_oof = apply_class_biases(base_oof_proba, best_biases)
    adjusted_test = apply_class_biases(base_test_proba, best_biases)
    best_log_loss = float(log_loss(y_true, adjusted_oof, labels=config.CLASS_LABELS))
    oof_path = config.STACKING_DIR / f"{args.output_name}_oof.csv"
    test_path = config.STACKING_DIR / f"{args.output_name}_test_proba.csv"
    summary_path = config.STACKING_DIR / f"{args.output_name}_bias_grid.csv"
    _write_oof(oof, adjusted_oof, oof_path)
    _write_test(test, adjusted_test, test_path)
    _write_summary(bias_candidates, scores, summary_path)

    current_best = get_best_score()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"class_bias_{args.output_name}"):
        mlflow.log_param("oof_path", str(args.oof_path))
        mlflow.log_param("test_proba_path", str(args.test_proba_path))
        mlflow.log_param("bias_range", args.bias_range)
        mlflow.log_param("bias_step", args.bias_step)
        mlflow.log_param("current_best_score", current_best)
        for label, bias in zip(config.CLASS_LABELS, best_biases):
            mlflow.log_param(f"bias_{label}", float(bias))
        mlflow.log_metric("balanced_accuracy", best_score)
        mlflow.log_metric("log_loss", best_log_loss)
        mlflow.log_artifact(str(oof_path))
        mlflow.log_artifact(str(test_path))
        mlflow.log_artifact(str(summary_path))

    print(f"Biased balanced accuracy: {best_score:.8f}")
    print(f"Biased log loss: {best_log_loss:.8f}")
    print(f"Class biases: {json.dumps({label: float(bias) for label, bias in zip(config.CLASS_LABELS, best_biases)}, sort_keys=True)}")
    print(f"OOF artifact: {oof_path}")
    print(f"Test probability artifact: {test_path}")
    print(f"Bias grid artifact: {summary_path}")
    print(f"Current champion threshold: {current_best:.8f}")


def _write_oof(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write adjusted OOF probabilities."""
    output = reference[["id", "fold", "y_true"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_test(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write adjusted test probabilities."""
    output = reference[["id"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _score_bias_candidates(
    base_logits: np.ndarray,
    y_true: np.ndarray,
    bias_candidates: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Score additive logit-bias candidates by balanced accuracy."""
    scores = np.zeros(len(bias_candidates), dtype=np.float64)
    for start in range(0, len(bias_candidates), batch_size):
        batch = bias_candidates[start : start + batch_size]
        predictions = (base_logits[None, :, :] + batch[:, None, :]).argmax(axis=2).astype(np.int8)
        scores[start : start + len(batch)] = _balanced_accuracy_batch(predictions, y_true)
    return scores


def _balanced_accuracy_batch(predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute balanced accuracy for a batch of prediction vectors."""
    recalls = []
    for class_index in range(len(config.CLASS_LABELS)):
        class_mask = y_true == class_index
        recalls.append((predictions[:, class_mask] == class_index).mean(axis=1))
    return np.mean(np.vstack(recalls), axis=0)


def _write_summary(bias_candidates: np.ndarray, scores: np.ndarray, path: Path) -> None:
    """Write bias-grid balanced accuracy summary."""
    summary = pd.DataFrame(
        {
            "bias_GALAXY": bias_candidates[:, 0],
            "bias_QSO": bias_candidates[:, 1],
            "bias_STAR": bias_candidates[:, 2],
            "balanced_accuracy": scores,
        }
    )
    summary.to_csv(path, index=False)


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    """Encode configured class labels as integer class indices."""
    label_to_index = {label: index for index, label in enumerate(config.CLASS_LABELS)}
    return np.asarray([label_to_index[label] for label in labels], dtype=np.int8)


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities to configured class labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


if __name__ == "__main__":
    main()
