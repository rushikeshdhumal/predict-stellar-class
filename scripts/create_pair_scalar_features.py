"""Create scalar margin features from a fold-safe binary specialist artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src import config
from src.ensemble import load_probability_artifact
from src.utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-path", type=Path, required=True, help="Binary specialist OOF probability artifact.")
    parser.add_argument("--test-proba-path", type=Path, required=True, help="Binary specialist test probability artifact.")
    parser.add_argument("--class-a", choices=config.CLASS_LABELS, required=True, help="First specialist class.")
    parser.add_argument("--class-b", choices=config.CLASS_LABELS, required=True, help="Second specialist class.")
    parser.add_argument("--output-name", required=True, help="Stable scalar feature artifact name.")
    return parser.parse_args()


def main() -> None:
    """Create and save aligned scalar feature artifacts."""
    args = parse_args()
    ensure_output_dirs()
    output_dir = config.ENSEMBLE_DIR / "scalar_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = load_probability_artifact(args.oof_path)
    test = load_probability_artifact(args.test_proba_path)
    prefix = _feature_prefix(args.class_a, args.class_b)
    oof_features = _build_features(oof, args.class_a, args.class_b, prefix, include_fold=True)
    test_features = _build_features(test, args.class_a, args.class_b, prefix, include_fold=False)
    oof_path = output_dir / f"{args.output_name}_oof_features.csv"
    test_path = output_dir / f"{args.output_name}_test_features.csv"
    oof_features.to_csv(oof_path, index=False)
    test_features.to_csv(test_path, index=False)
    print(f"OOF scalar features: {oof_path}")
    print(f"Test scalar features: {test_path}")


def _build_features(
    artifact: pd.DataFrame,
    class_a: str,
    class_b: str,
    prefix: str,
    *,
    include_fold: bool,
) -> pd.DataFrame:
    """Build scalar features from two class probabilities."""
    proba_a = np.clip(artifact[f"proba_{class_a}"].to_numpy(dtype=np.float64), 1e-8, 1.0)
    proba_b = np.clip(artifact[f"proba_{class_b}"].to_numpy(dtype=np.float64), 1e-8, 1.0)
    pair_sum = np.clip(proba_a + proba_b, 1e-8, None)
    pair_a = proba_a / pair_sum
    pair_b = proba_b / pair_sum
    margin = np.log(pair_b) - np.log(pair_a)
    entropy = -(pair_a * np.log(pair_a) + pair_b * np.log(pair_b))
    output_columns = ["id"]
    if include_fold:
        output_columns.extend(["fold", "y_true"])
    output = artifact[output_columns].copy()
    output[f"{prefix}__prob_{class_b}"] = pair_b
    output[f"{prefix}__prob_{class_a}"] = pair_a
    output[f"{prefix}__logit_margin_{class_b}_minus_{class_a}"] = margin
    output[f"{prefix}__abs_logit_margin"] = np.abs(margin)
    output[f"{prefix}__entropy"] = entropy
    output[f"{prefix}__predicts_{class_b}"] = (pair_b > pair_a).astype(np.float64)
    return output


def _feature_prefix(class_a: str, class_b: str) -> str:
    """Build a stable feature prefix."""
    return f"pair_{class_a.lower()}_vs_{class_b.lower()}"


if __name__ == "__main__":
    main()
