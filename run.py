"""Run the fold-safe LightGBM baseline pipeline.

This root entry point is intentionally a stable, easy-to-run baseline. The more
advanced ensemble, stacking, and diversity experiments live under ``scripts/``.
"""

import mlflow
import pandas as pd

from src import config
from src.data import create_stratified_folds, load_raw_data, make_features, write_eda_report
from src.predict import predict_test_ensemble
from src.train import train_model_cv
from src.utils import (
    append_decision_entry,
    append_experiment_log,
    create_submission,
    ensure_output_dirs,
    get_best_score,
    update_best_score,
    update_best_submission,
)


BASELINE_CHANGE = "rerun configured feature-engineered LightGBM baseline"
BASELINE_HYPOTHESIS = (
    "The configured LightGBM pipeline provides a reproducible fold-safe baseline "
    "for comparison against ensemble experiments."
)


def log_run_configuration() -> None:
    """Log baseline configuration parameters to MLflow."""
    mlflow.log_param("pipeline_role", "baseline_lightgbm")
    mlflow.log_param("hypothesis", BASELINE_HYPOTHESIS)
    mlflow.log_param("seed", config.SEED)
    mlflow.log_param("n_folds", config.N_FOLDS)
    mlflow.log_param("features", ",".join(config.FEATURE_COLUMNS))
    mlflow.log_param("color_features", ",".join(config.COLOR_FEATURES))
    mlflow.log_param(
        "magnitude_summary_features",
        ",".join(config.MAGNITUDE_SUMMARY_FEATURES),
    )
    mlflow.log_param(
        "categorical_cross_features",
        ",".join(config.CATEGORICAL_CROSS_FEATURES),
    )
    mlflow.log_param("spatial_features", ",".join(config.SPATIAL_FEATURES))
    mlflow.log_params(config.LGBM_PARAMS)


def load_featured_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data, write EDA, and create fold-safe feature matrices.

    Returns:
        Feature-engineered training data with fold assignments, feature-engineered
        test data, and the sample submission dataframe.
    """
    train, test, sample = load_raw_data()
    eda_path = write_eda_report(train, test)
    mlflow.log_artifact(str(eda_path))
    print(f"EDA report written to {eda_path}")

    train_features = make_features(train)
    test_features = make_features(test)
    train_features = create_stratified_folds(train_features, n_folds=config.N_FOLDS)
    return train_features, test_features, sample


def main() -> None:
    """Run EDA, five-fold CV training, prediction, and champion tracking."""
    ensure_output_dirs()
    best = get_best_score()

    print(f"Current best balanced accuracy: {best if best != float('-inf') else 'none'}")
    print(f"Pipeline: {BASELINE_CHANGE}")
    print(f"Hypothesis: {BASELINE_HYPOTHESIS}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"baseline_{config.MODEL_NAME}"):
        log_run_configuration()
        train_features, test_features, sample = load_featured_data()

        models, _, cv_scores, label_encoder, mean_loss = train_model_cv(train_features)
        mean_ba = float(cv_scores.mean())
        mlflow.log_metric("cv_mean_balanced_accuracy", mean_ba)
        mlflow.log_metric("cv_std_balanced_accuracy", float(cv_scores.std()))
        mlflow.log_metric("cv_mean_log_loss", mean_loss)

        improved = mean_ba > best
        submission_path = None
        if improved:
            test_preds = predict_test_ensemble(models, test_features, label_encoder)
            submission_path = create_submission(test_preds, sample)
            update_best_submission(submission_path)
            update_best_score(mean_ba)
            mlflow.log_artifact(str(submission_path))
            mlflow.log_artifact(str(config.BEST_SUBMISSION_PATH))
            print(f"New best! {mean_ba:.6f} > {best if best != float('-inf') else 'none'}")
        else:
            print(f"Not improved: {mean_ba:.6f} <= {best:.6f}")

        append_experiment_log(BASELINE_CHANGE, mean_ba, submission_path)
        append_decision_entry(BASELINE_CHANGE, mean_ba, best, improved)


if __name__ == "__main__":
    main()
