"""Train one supervised base model and save OOF/test probability artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow

from src import config
from src.data import create_stratified_folds, load_raw_data, make_features
from src.ensemble import default_model_specs, train_base_model_oof
from src.utils import ensure_output_dirs, get_best_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", type=str, help="Base model spec name to train.")
    parser.add_argument("--list-models", action="store_true", help="Print available base model names.")
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Skip persisting fold model joblib artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Train one base model with fixed 5-fold OOF artifacts."""
    args = parse_args()
    specs = default_model_specs()
    if args.list_models:
        for name in specs:
            print(name)
        return
    if args.model_name not in specs:
        available = ", ".join(specs)
        raise ValueError(f"Unknown --model-name `{args.model_name}`. Available: {available}")

    ensure_output_dirs()
    best_score = get_best_score()
    spec = specs[args.model_name]
    train_raw, test_raw, _ = load_raw_data()
    train = create_stratified_folds(make_features(train_raw), n_folds=config.N_FOLDS)
    test = make_features(test_raw)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"oof_{spec.name}"):
        mlflow.log_param("model_name", spec.name)
        mlflow.log_param("model_family", spec.family)
        mlflow.log_param("seed", config.SEED)
        mlflow.log_param("n_folds", config.N_FOLDS)
        mlflow.log_param("current_best_score", best_score)
        feature_columns = spec.feature_columns or config.FEATURE_COLUMNS
        mlflow.log_param("features", ",".join(feature_columns))
        mlflow.log_params(spec.params)
        result = train_base_model_oof(spec, train, test, save_models=not args.no_save_models)
        mlflow.log_metric("cv_mean_balanced_accuracy", result.mean_balanced_accuracy)
        mlflow.log_metric("cv_mean_log_loss", result.mean_log_loss)
        for fold, score in enumerate(result.fold_scores):
            mlflow.log_metric(f"fold_{fold}_balanced_accuracy", score)
        mlflow.log_artifact(str(result.oof_path))
        mlflow.log_artifact(str(result.test_proba_path))

    print(f"Model: {result.model_name}")
    print(f"Mean balanced accuracy: {result.mean_balanced_accuracy:.8f}")
    print(f"Mean log loss: {result.mean_log_loss:.8f}")
    print(f"OOF artifact: {result.oof_path}")
    print(f"Test probability artifact: {result.test_proba_path}")
    print(f"Current champion threshold: {best_score:.8f}")


if __name__ == "__main__":
    main()
