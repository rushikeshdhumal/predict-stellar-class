"""Create a submission CSV from a saved test probability artifact."""

from __future__ import annotations

import argparse
from datetime import datetime
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
    parser.add_argument("--proba-path", type=Path, required=True, help="Test probability artifact path.")
    parser.add_argument("--submission-name", type=str, required=True, help="Stable submission name suffix.")
    return parser.parse_args()


def main() -> None:
    """Create and save a class-label submission from probabilities."""
    args = parse_args()
    ensure_output_dirs()
    probabilities = load_probability_artifact(args.proba_path)
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    predictions = labels[probabilities[config.PROBA_COLUMNS].to_numpy().argmax(axis=1)]
    submission = sample.copy()
    submission[config.TARGET_COLUMN] = predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.SUBMISSIONS_DIR / f"submission_{args.submission_name}_{timestamp}.csv"
    submission.to_csv(output_path, index=False)
    print(f"Submission written to: {output_path}")


if __name__ == "__main__":
    main()
