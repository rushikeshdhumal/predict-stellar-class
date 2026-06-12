"""Stacking and blending utilities for OOF probability artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss

from src import config
from src.ensemble import load_probability_artifact


@dataclass(frozen=True)
class MetaStackResult:
    """Result metadata for a logistic meta-stacking run.

    Attributes:
        mean_balanced_accuracy: Fold-wise OOF balanced accuracy.
        mean_log_loss: Fold-wise OOF log loss.
        oof_path: Saved meta-learner OOF probability artifact.
        test_proba_path: Saved averaged meta-learner test probability artifact.
    """

    mean_balanced_accuracy: float
    mean_log_loss: float
    oof_path: Path
    test_proba_path: Path


def train_logistic_meta_stack(
    oof_paths: list[Path],
    test_proba_paths: list[Path],
    *,
    output_name: str,
    c_value: float = 0.25,
) -> MetaStackResult:
    """Train a fold-wise logistic meta-learner from base OOF artifacts.

    Args:
        oof_paths: Base model OOF artifact paths.
        test_proba_paths: Matching base model test probability paths.
        output_name: Stable output artifact name.
        c_value: Inverse regularization strength for logistic regression.

    Returns:
        Meta-stacking result metadata.
    """
    if len(oof_paths) != len(test_proba_paths):
        raise ValueError("OOF and test artifact path counts must match.")
    oof_frames = [load_probability_artifact(path) for path in oof_paths]
    test_frames = [load_probability_artifact(path) for path in test_proba_paths]
    _validate_base_artifacts(oof_frames, test_frames)

    y_true = oof_frames[0]["y_true"].to_numpy()
    folds = oof_frames[0]["fold"].to_numpy()
    x_meta = _stack_log_probabilities(oof_frames)
    x_test_meta = _stack_log_probabilities(test_frames)
    oof_proba = np.zeros((len(oof_frames[0]), len(config.CLASS_LABELS)), dtype=float)
    test_proba = np.zeros((len(test_frames[0]), len(config.CLASS_LABELS)), dtype=float)
    fold_scores: list[float] = []
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
        fold_scores.append(float(balanced_accuracy_score(y_true[valid_idx], _labels_from_proba(valid_proba))))
        fold_losses.append(float(log_loss(y_true[valid_idx], valid_proba, labels=config.CLASS_LABELS)))

    config.STACKING_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = config.STACKING_DIR / f"{output_name}_oof.csv"
    test_proba_path = config.STACKING_DIR / f"{output_name}_test_proba.csv"
    _write_meta_oof(oof_frames[0], oof_proba, oof_path)
    _write_meta_test(test_frames[0], test_proba, test_proba_path)
    return MetaStackResult(
        mean_balanced_accuracy=float(np.mean(fold_scores)),
        mean_log_loss=float(np.mean(fold_losses)),
        oof_path=oof_path,
        test_proba_path=test_proba_path,
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
