"""Fine-search a two-way OOF probability blend weight."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src import config
from src.ensemble import load_probability_artifact
from src.utils import get_best_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-oof-path", type=Path, required=True, help="Left OOF probability artifact.")
    parser.add_argument("--right-oof-path", type=Path, required=True, help="Right OOF probability artifact.")
    parser.add_argument("--min-right-weight", type=float, default=0.35, help="Minimum right artifact weight.")
    parser.add_argument("--max-right-weight", type=float, default=0.55, help="Maximum right artifact weight.")
    parser.add_argument("--step", type=float, default=0.001, help="Weight search step.")
    return parser.parse_args()


def main() -> None:
    """Run the fine weight search and print the best result."""
    args = parse_args()
    left_oof = load_probability_artifact(args.left_oof_path)
    right_oof = load_probability_artifact(args.right_oof_path)
    _validate_alignment(left_oof, right_oof)
    y_true = left_oof["y_true"].to_numpy()
    left_proba = left_oof[config.PROBA_COLUMNS].to_numpy(dtype=np.float32)
    right_proba = right_oof[config.PROBA_COLUMNS].to_numpy(dtype=np.float32)
    weights = np.arange(
        args.min_right_weight,
        args.max_right_weight + args.step / 2.0,
        args.step,
        dtype=np.float32,
    )
    best_score = -np.inf
    best_weight = 0.0
    for weight in weights:
        blended = (1.0 - weight) * left_proba + weight * right_proba
        score = balanced_accuracy_score(y_true, _labels_from_proba(blended))
        if score > best_score:
            best_score = float(score)
            best_weight = float(weight)
    print(f"Best balanced accuracy: {best_score:.8f}")
    print(f"Best right weight: {best_weight:.6f}")
    print(f"Best left weight: {1.0 - best_weight:.6f}")
    print(f"Current champion threshold: {get_best_score():.8f}")


def _validate_alignment(left_oof, right_oof) -> None:
    """Validate row alignment for OOF blend inputs."""
    for column in ["id", "fold", "y_true"]:
        if not left_oof[column].equals(right_oof[column]):
            raise ValueError(f"OOF artifacts are not aligned on `{column}`.")


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities to configured class labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


if __name__ == "__main__":
    main()
