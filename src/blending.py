"""Weighted blending and class-bias utilities for probability artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src import config
from src.ensemble import load_probability_artifact


@dataclass(frozen=True)
class BlendResult:
    """Result metadata for a weighted blend.

    Attributes:
        balanced_accuracy: OOF balanced accuracy after blending and bias tuning.
        weights: Base model blend weights.
        class_biases: Additive logit biases by class label.
        oof_path: Saved blended OOF probability artifact.
        test_proba_path: Saved blended test probability artifact.
    """

    balanced_accuracy: float
    weights: dict[str, float]
    class_biases: dict[str, float]
    oof_path: Path
    test_proba_path: Path


def optimize_weighted_blend(
    model_names: list[str],
    *,
    output_name: str,
    n_trials: int = 2000,
    tune_biases: bool = True,
    seed: int = config.SEED,
) -> BlendResult:
    """Optimize a supervised weighted probability blend on OOF balanced accuracy.

    Args:
        model_names: Base model names with OOF and test probability artifacts.
        output_name: Stable output artifact name.
        n_trials: Number of random Dirichlet weight samples.
        tune_biases: Whether to tune class-specific logit biases after blending.
        seed: Random seed for reproducible weight search.

    Returns:
        Blend result metadata.
    """
    if len(model_names) < 2:
        raise ValueError("At least two base models are required for blending.")
    oof_frames = [load_probability_artifact(config.OOF_DIR / f"{name}_oof.csv") for name in model_names]
    test_frames = [load_probability_artifact(config.TEST_PROBA_DIR / f"{name}_test_proba.csv") for name in model_names]
    _validate_alignment(oof_frames, test_frames)

    y_true = oof_frames[0]["y_true"].to_numpy()
    oof_arrays = [frame[config.PROBA_COLUMNS].to_numpy() for frame in oof_frames]
    test_arrays = [frame[config.PROBA_COLUMNS].to_numpy() for frame in test_frames]
    best_score = float("-inf")
    best_weights = np.zeros(len(model_names), dtype=float)
    rng = np.random.default_rng(seed)

    candidates = [np.eye(len(model_names))[index] for index in range(len(model_names))]
    candidates.append(np.full(len(model_names), 1.0 / len(model_names)))
    candidates.extend(rng.dirichlet(np.ones(len(model_names)), size=n_trials))
    for weights in candidates:
        blended = _weighted_average(oof_arrays, weights)
        score = balanced_accuracy_score(y_true, _labels_from_proba(blended))
        if score > best_score:
            best_score = float(score)
            best_weights = np.asarray(weights, dtype=float)

    blended_oof = _weighted_average(oof_arrays, best_weights)
    blended_test = _weighted_average(test_arrays, best_weights)
    class_biases = np.zeros(len(config.CLASS_LABELS), dtype=float)
    if tune_biases:
        class_biases, best_score = optimize_class_biases(blended_oof, y_true)
        blended_oof = apply_class_biases(blended_oof, class_biases)
        blended_test = apply_class_biases(blended_test, class_biases)

    config.STACKING_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = config.STACKING_DIR / f"{output_name}_oof.csv"
    test_proba_path = config.STACKING_DIR / f"{output_name}_test_proba.csv"
    _write_oof(oof_frames[0], blended_oof, oof_path)
    _write_test(test_frames[0], blended_test, test_proba_path)
    return BlendResult(
        balanced_accuracy=float(best_score),
        weights={name: float(weight) for name, weight in zip(model_names, best_weights)},
        class_biases={label: float(bias) for label, bias in zip(config.CLASS_LABELS, class_biases)},
        oof_path=oof_path,
        test_proba_path=test_proba_path,
    )


def optimize_class_biases(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    bias_values: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Tune class-specific logit biases with GALAXY fixed at zero.

    Args:
        proba: OOF class probabilities.
        y_true: True class labels.
        bias_values: Candidate bias grid for QSO and STAR.

    Returns:
        Best bias vector and balanced accuracy score.
    """
    values = bias_values if bias_values is not None else np.arange(-0.2, 0.2001, 0.02)
    best_biases = np.zeros(len(config.CLASS_LABELS), dtype=float)
    best_score = balanced_accuracy_score(y_true, _labels_from_proba(proba))
    for qso_bias in values:
        for star_bias in values:
            biases = np.array([0.0, qso_bias, star_bias], dtype=float)
            adjusted = apply_class_biases(proba, biases)
            score = balanced_accuracy_score(y_true, _labels_from_proba(adjusted))
            if score > best_score:
                best_score = float(score)
                best_biases = biases
    return best_biases, float(best_score)


def apply_class_biases(proba: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """Apply additive logit biases and renormalize probabilities.

    Args:
        proba: Class probability matrix.
        biases: Additive class-bias vector.

    Returns:
        Adjusted probability matrix.
    """
    logits = np.log(np.clip(proba, 1e-12, 1.0)) + biases
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _weighted_average(arrays: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Compute weighted average probabilities."""
    blended = np.zeros_like(arrays[0], dtype=float)
    for array, weight in zip(arrays, weights):
        blended += array * weight
    return blended


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


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities to configured class labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]
