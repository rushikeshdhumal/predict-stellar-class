"""Optimize a deterministic local blend from saved probability artifacts."""

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

from src import config
from src.ensemble import load_probability_artifact
from src.utils import ensure_output_dirs, get_best_score


DEFAULT_MODELS: list[str] = [
    "lightgbm_optuna_trial_84_n_estimators_200",
    "lightgbm_optuna_trial_49",
    "lightgbm_optuna_trial_65",
    "lightgbm_optuna_trial_81",
    "lightgbm_optuna_trial_31",
]
DEFAULT_CENTER_WEIGHTS: list[float] = [
    0.3245607830775459,
    0.16358813933220528,
    0.2290813753067371,
    0.20810929450685292,
    0.07466040777665885,
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Base model names to blend.")
    parser.add_argument(
        "--center-weights",
        nargs="+",
        type=float,
        default=DEFAULT_CENTER_WEIGHTS,
        help="Center weights for the deterministic local grid.",
    )
    parser.add_argument("--output-name", type=str, required=True, help="Output artifact name.")
    parser.add_argument("--coarse-step", type=float, default=0.02, help="Coarse local-grid step.")
    parser.add_argument("--coarse-radius", type=int, default=3, help="Coarse integer radius around center.")
    parser.add_argument("--fine-step", type=float, default=0.005, help="Fine local-grid step.")
    parser.add_argument("--fine-radius", type=int, default=4, help="Fine integer radius around coarse best.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of candidates per scoring batch.")
    parser.add_argument("--bias-range", type=float, default=0.06, help="Absolute QSO/STAR logit-bias range.")
    parser.add_argument("--bias-step", type=float, default=0.005, help="QSO/STAR logit-bias step.")
    parser.add_argument("--no-bias", action="store_true", help="Disable deterministic class-bias tuning.")
    return parser.parse_args()


def main() -> None:
    """Run deterministic blend optimization and save OOF/test artifacts."""
    args = parse_args()
    ensure_output_dirs()
    model_names = list(args.models)
    center_weights = np.asarray(args.center_weights, dtype=np.float32)
    if len(model_names) != len(center_weights):
        raise ValueError("The number of models must match the number of center weights.")
    center_weights = _normalize_weights(center_weights)

    oof_frames = [load_probability_artifact(config.OOF_DIR / f"{name}_oof.csv") for name in model_names]
    test_frames = [load_probability_artifact(config.TEST_PROBA_DIR / f"{name}_test_proba.csv") for name in model_names]
    _validate_alignment(oof_frames, test_frames)
    y_true = _encode_labels(oof_frames[0]["y_true"].to_numpy())
    oof_arrays = np.stack(
        [frame[config.PROBA_COLUMNS].to_numpy(dtype=np.float32) for frame in oof_frames],
        axis=0,
    )
    test_arrays = np.stack(
        [frame[config.PROBA_COLUMNS].to_numpy(dtype=np.float32) for frame in test_frames],
        axis=0,
    )

    coarse_candidates = _make_local_candidates(center_weights, args.coarse_step, args.coarse_radius)
    coarse_score, coarse_weights = _score_weight_candidates(
        oof_arrays,
        y_true,
        coarse_candidates,
        args.batch_size,
    )
    fine_candidates = _make_local_candidates(coarse_weights, args.fine_step, args.fine_radius)
    best_score, best_weights = _score_weight_candidates(
        oof_arrays,
        y_true,
        fine_candidates,
        args.batch_size,
    )
    blended_oof = _weighted_average(oof_arrays, best_weights)
    blended_test = _weighted_average(test_arrays, best_weights)
    best_biases = np.zeros(len(config.CLASS_LABELS), dtype=np.float32)
    if not args.no_bias:
        best_biases, best_score, blended_oof, blended_test = _tune_biases(
            blended_oof,
            blended_test,
            y_true,
            bias_range=args.bias_range,
            bias_step=args.bias_step,
            batch_size=args.batch_size,
        )

    oof_path = config.STACKING_DIR / f"{args.output_name}_oof.csv"
    test_proba_path = config.STACKING_DIR / f"{args.output_name}_test_proba.csv"
    _write_oof(oof_frames[0], blended_oof, oof_path)
    _write_test(test_frames[0], blended_test, test_proba_path)

    best_threshold = get_best_score()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"deterministic_blend_{args.output_name}"):
        mlflow.log_param("base_models", ",".join(model_names))
        mlflow.log_param("center_weights", json.dumps(center_weights.astype(float).tolist()))
        mlflow.log_param("coarse_step", args.coarse_step)
        mlflow.log_param("coarse_radius", args.coarse_radius)
        mlflow.log_param("fine_step", args.fine_step)
        mlflow.log_param("fine_radius", args.fine_radius)
        mlflow.log_param("tune_biases", not args.no_bias)
        mlflow.log_param("current_best_score", best_threshold)
        mlflow.log_metric("cv_balanced_accuracy", float(best_score))
        for name, weight in zip(model_names, best_weights):
            mlflow.log_param(f"weight_{name}", float(weight))
        for label, bias in zip(config.CLASS_LABELS, best_biases):
            mlflow.log_param(f"bias_{label}", float(bias))
        mlflow.log_artifact(str(oof_path))
        mlflow.log_artifact(str(test_proba_path))

    print(f"Blend balanced accuracy: {best_score:.8f}")
    print(f"Weights: {json.dumps({name: float(weight) for name, weight in zip(model_names, best_weights)}, sort_keys=True)}")
    print(f"Class biases: {json.dumps({label: float(bias) for label, bias in zip(config.CLASS_LABELS, best_biases)}, sort_keys=True)}")
    print(f"OOF artifact: {oof_path}")
    print(f"Test probability artifact: {test_proba_path}")
    print(f"Current champion threshold: {best_threshold:.8f}")


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize nonnegative weights to sum to one."""
    clipped = np.clip(weights, 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError("At least one blend weight must be positive.")
    return clipped / total


def _make_local_candidates(center: np.ndarray, step: float, radius: int) -> np.ndarray:
    """Create deterministic local candidates with the last weight as residual."""
    if center.ndim != 1 or center.size < 2:
        raise ValueError("Center weights must be a one-dimensional array with at least two values.")
    offsets = np.arange(-radius, radius + 1, dtype=np.float32) * np.float32(step)
    candidates: list[np.ndarray] = [center.astype(np.float32)]
    for delta_values in np.array(np.meshgrid(*([offsets] * (center.size - 1)), indexing="ij")).T.reshape(-1, center.size - 1):
        candidate = center.copy()
        candidate[:-1] += delta_values
        candidate[-1] = 1.0 - float(candidate[:-1].sum())
        if np.all(candidate >= 0.0) and np.all(candidate <= 1.0):
            candidates.append(candidate.astype(np.float32))
    unique = np.unique(np.vstack(candidates), axis=0)
    return unique.astype(np.float32)


def _score_weight_candidates(
    proba_arrays: np.ndarray,
    y_true: np.ndarray,
    candidates: np.ndarray,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    """Find the best weight candidate by OOF balanced accuracy."""
    best_score = -np.inf
    best_weights = candidates[0]
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        blended = np.tensordot(batch, proba_arrays, axes=(1, 0))
        predictions = blended.argmax(axis=2).astype(np.int8)
        scores = _balanced_accuracy_batch(predictions, y_true)
        best_index = int(scores.argmax())
        if float(scores[best_index]) > best_score:
            best_score = float(scores[best_index])
            best_weights = batch[best_index].copy()
    return best_score, best_weights


def _balanced_accuracy_batch(predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute balanced accuracy for a batch of prediction vectors."""
    recalls = []
    for class_index in range(len(config.CLASS_LABELS)):
        class_mask = y_true == class_index
        recalls.append((predictions[:, class_mask] == class_index).mean(axis=1))
    return np.mean(np.vstack(recalls), axis=0)


def _tune_biases(
    blended_oof: np.ndarray,
    blended_test: np.ndarray,
    y_true: np.ndarray,
    *,
    bias_range: float,
    bias_step: float,
    batch_size: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Tune QSO/STAR additive logit biases with GALAXY fixed at zero."""
    base_logits = np.log(np.clip(blended_oof, 1e-12, 1.0)).astype(np.float32)
    values = np.arange(-bias_range, bias_range + bias_step / 2.0, bias_step, dtype=np.float32)
    bias_candidates = np.array([[0.0, qso, star] for qso in values for star in values], dtype=np.float32)
    best_score = float(_balanced_accuracy_batch(blended_oof.argmax(axis=1)[None, :].astype(np.int8), y_true)[0])
    best_biases = np.zeros(len(config.CLASS_LABELS), dtype=np.float32)
    for start in range(0, len(bias_candidates), batch_size):
        batch = bias_candidates[start : start + batch_size]
        adjusted_logits = base_logits[None, :, :] + batch[:, None, :]
        predictions = adjusted_logits.argmax(axis=2).astype(np.int8)
        scores = _balanced_accuracy_batch(predictions, y_true)
        best_index = int(scores.argmax())
        if float(scores[best_index]) > best_score:
            best_score = float(scores[best_index])
            best_biases = batch[best_index].copy()
    return best_biases, best_score, _apply_biases(blended_oof, best_biases), _apply_biases(blended_test, best_biases)


def _apply_biases(proba: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """Apply additive logit biases and renormalize probabilities."""
    logits = np.log(np.clip(proba, 1e-12, 1.0)).astype(np.float32) + biases
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _weighted_average(proba_arrays: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute a weighted probability average."""
    return np.tensordot(weights.astype(np.float32), proba_arrays, axes=(0, 0))


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    """Encode configured class labels as integer class indices."""
    label_to_index = {label: index for index, label in enumerate(config.CLASS_LABELS)}
    return np.asarray([label_to_index[label] for label in labels], dtype=np.int8)


def _validate_alignment(oof_frames: list[pd.DataFrame], test_frames: list[pd.DataFrame]) -> None:
    """Validate row alignment for blend inputs."""
    reference_oof = oof_frames[0]
    reference_test = test_frames[0]
    for frame in oof_frames[1:]:
        for column in ["id", "fold", "y_true"]:
            if not reference_oof[column].equals(frame[column]):
                raise ValueError(f"OOF artifacts are not aligned on `{column}`.")
    for frame in test_frames[1:]:
        if not reference_test["id"].equals(frame["id"]):
            raise ValueError("Test probability artifacts are not aligned on `id`.")


def _write_oof(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write blended OOF probabilities."""
    output = reference[["id", "fold", "y_true"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_test(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write blended test probabilities."""
    output = reference[["id"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


if __name__ == "__main__":
    main()
