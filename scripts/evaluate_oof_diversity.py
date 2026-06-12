"""Evaluate OOF diversity between a champion and candidate base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.ensemble import evaluate_diversity, load_probability_artifact
from src.utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion-model",
        type=str,
        default="lightgbm_optuna_trial_84_n_estimators_200",
        help="Champion OOF artifact model name.",
    )
    parser.add_argument("--candidate-model", type=str, required=True, help="Candidate OOF artifact model name.")
    parser.add_argument(
        "--champion-path",
        type=Path,
        default=None,
        help="Optional explicit champion OOF artifact path.",
    )
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=None,
        help="Optional explicit candidate OOF artifact path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON output path for diversity metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate and print OOF diversity metrics."""
    args = parse_args()
    ensure_output_dirs()
    champion_path = args.champion_path or config.OOF_DIR / f"{args.champion_model}_oof.csv"
    candidate_path = args.candidate_path or config.OOF_DIR / f"{args.candidate_model}_oof.csv"
    champion_oof = load_probability_artifact(champion_path)
    candidate_oof = load_probability_artifact(candidate_path)
    metrics = evaluate_diversity(champion_oof, candidate_oof)
    output_path = args.output_path or config.STACKING_DIR / f"diversity_{args.champion_model}_vs_{args.candidate_model}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    main()
