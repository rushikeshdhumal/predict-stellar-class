"""Optimize a weighted probability blend from saved base-model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow

from src import config
from src.blending import optimize_weighted_blend
from src.utils import ensure_output_dirs, get_best_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Base model names to blend.")
    parser.add_argument("--output-name", type=str, required=True, help="Output artifact name.")
    parser.add_argument("--trials", type=int, default=2000, help="Random weight samples.")
    parser.add_argument("--no-bias", action="store_true", help="Disable class logit-bias tuning.")
    return parser.parse_args()


def main() -> None:
    """Optimize and save a weighted probability blend."""
    args = parse_args()
    ensure_output_dirs()
    best_score = get_best_score()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"blend_{args.output_name}"):
        mlflow.log_param("base_models", ",".join(args.models))
        mlflow.log_param("weight_trials", args.trials)
        mlflow.log_param("tune_biases", not args.no_bias)
        mlflow.log_param("current_best_score", best_score)
        result = optimize_weighted_blend(
            args.models,
            output_name=args.output_name,
            n_trials=args.trials,
            tune_biases=not args.no_bias,
        )
        mlflow.log_metric("cv_balanced_accuracy", result.balanced_accuracy)
        mlflow.log_params({f"weight_{name}": weight for name, weight in result.weights.items()})
        mlflow.log_params({f"bias_{label}": bias for label, bias in result.class_biases.items()})
        mlflow.log_artifact(str(result.oof_path))
        mlflow.log_artifact(str(result.test_proba_path))

    print(f"Blend balanced accuracy: {result.balanced_accuracy:.8f}")
    print(f"Weights: {json.dumps(result.weights, sort_keys=True)}")
    print(f"Class biases: {json.dumps(result.class_biases, sort_keys=True)}")
    print(f"OOF artifact: {result.oof_path}")
    print(f"Test probability artifact: {result.test_proba_path}")
    print(f"Current champion threshold: {best_score:.8f}")


if __name__ == "__main__":
    main()
