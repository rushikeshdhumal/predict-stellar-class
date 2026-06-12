"""Create a fixed weighted blend from explicit OOF and test probability paths."""

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
from src.utils import ensure_output_dirs, get_best_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-oof-path", type=Path, required=True, help="Left OOF probability artifact.")
    parser.add_argument("--left-test-path", type=Path, required=True, help="Left test probability artifact.")
    parser.add_argument("--right-oof-path", type=Path, required=True, help="Right OOF probability artifact.")
    parser.add_argument("--right-test-path", type=Path, required=True, help="Right test probability artifact.")
    parser.add_argument("--right-weight", type=float, required=True, help="Right artifact blend weight.")
    parser.add_argument("--output-name", type=str, required=True, help="Output artifact name.")
    return parser.parse_args()


def main() -> None:
    """Create fixed weighted OOF/test blend artifacts."""
    args = parse_args()
    ensure_output_dirs()
    left_weight = 1.0 - args.right_weight
    if not 0.0 <= args.right_weight <= 1.0:
        raise ValueError("--right-weight must be in [0, 1].")

    left_oof = load_probability_artifact(args.left_oof_path)
    right_oof = load_probability_artifact(args.right_oof_path)
    left_test = load_probability_artifact(args.left_test_path)
    right_test = load_probability_artifact(args.right_test_path)
    _validate_alignment(left_oof, right_oof, left_test, right_test)

    oof_proba = (
        left_weight * left_oof[config.PROBA_COLUMNS].to_numpy()
        + args.right_weight * right_oof[config.PROBA_COLUMNS].to_numpy()
    )
    test_proba = (
        left_weight * left_test[config.PROBA_COLUMNS].to_numpy()
        + args.right_weight * right_test[config.PROBA_COLUMNS].to_numpy()
    )
    score = balanced_accuracy_score(left_oof["y_true"], _labels_from_proba(oof_proba))
    oof_path = config.STACKING_DIR / f"{args.output_name}_oof.csv"
    test_path = config.STACKING_DIR / f"{args.output_name}_test_proba.csv"
    _write_oof(left_oof, oof_proba, oof_path)
    _write_test(left_test, test_proba, test_path)

    print(f"Blend balanced accuracy: {score:.8f}")
    print(f"Left weight: {left_weight:.8f}")
    print(f"Right weight: {args.right_weight:.8f}")
    print(f"OOF artifact: {oof_path}")
    print(f"Test probability artifact: {test_path}")
    print(f"Current champion threshold: {get_best_score():.8f}")


def _validate_alignment(left_oof, right_oof, left_test, right_test) -> None:
    """Validate row alignment for explicit blend inputs."""
    for column in ["id", "fold", "y_true"]:
        if not left_oof[column].equals(right_oof[column]):
            raise ValueError(f"OOF artifacts are not aligned on `{column}`.")
    if not left_test["id"].equals(right_test["id"]):
        raise ValueError("Test probability artifacts are not aligned on `id`.")


def _write_oof(reference, proba: np.ndarray, path: Path) -> None:
    """Write OOF probability artifact."""
    output = reference[["id", "fold", "y_true"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_test(reference, proba: np.ndarray, path: Path) -> None:
    """Write test probability artifact."""
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
