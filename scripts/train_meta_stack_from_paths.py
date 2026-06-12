"""Train a fold-wise logistic meta-stack from explicit probability artifact paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow

from src import config
from src.stacking import train_logistic_meta_stack
from src.utils import ensure_output_dirs, get_best_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-paths", nargs="+", type=Path, required=True, help="Base OOF artifact paths.")
    parser.add_argument(
        "--test-proba-paths",
        nargs="+",
        type=Path,
        required=True,
        help="Base test probability artifact paths in the same order as OOF paths.",
    )
    parser.add_argument("--base-names", nargs="+", required=True, help="Human-readable base artifact names.")
    parser.add_argument("--output-name", type=str, required=True, help="Output artifact name.")
    parser.add_argument("--c-value", type=float, default=0.25, help="Logistic regression C value.")
    return parser.parse_args()


def main() -> None:
    """Train and evaluate a logistic meta-stack from explicit artifact paths."""
    args = parse_args()
    if len(args.oof_paths) != len(args.test_proba_paths):
        raise ValueError("OOF and test artifact path counts must match.")
    if len(args.oof_paths) != len(args.base_names):
        raise ValueError("Base name count must match artifact path count.")

    ensure_output_dirs()
    best_score = get_best_score()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"meta_stack_{args.output_name}"):
        mlflow.log_param("base_models", ",".join(args.base_names))
        mlflow.log_param("oof_paths", ",".join(str(path) for path in args.oof_paths))
        mlflow.log_param("test_proba_paths", ",".join(str(path) for path in args.test_proba_paths))
        mlflow.log_param("c_value", args.c_value)
        mlflow.log_param("current_best_score", best_score)
        result = train_logistic_meta_stack(
            args.oof_paths,
            args.test_proba_paths,
            output_name=args.output_name,
            c_value=args.c_value,
        )
        mlflow.log_metric("cv_mean_balanced_accuracy", result.mean_balanced_accuracy)
        mlflow.log_metric("cv_mean_log_loss", result.mean_log_loss)
        mlflow.log_artifact(str(result.oof_path))
        mlflow.log_artifact(str(result.test_proba_path))

    print(f"Meta-stack balanced accuracy: {result.mean_balanced_accuracy:.8f}")
    print(f"Meta-stack log loss: {result.mean_log_loss:.8f}")
    print(f"OOF artifact: {result.oof_path}")
    print(f"Test probability artifact: {result.test_proba_path}")
    print(f"Current champion threshold: {best_score:.8f}")


if __name__ == "__main__":
    main()
