"""Run a small Optuna hyperparameter search for XGBoost base models."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data import create_stratified_folds, load_raw_data, make_features
from src.utils import ensure_output_dirs, get_best_score


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top completed trials to print.")
    parser.add_argument(
        "--study-name",
        type=str,
        default="xgboost_small_search",
        help="Optuna study name.",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=config.SUBMISSIONS_DIR / "optuna_xgboost_small_search.db",
        help="SQLite database path used to persist and resume the study.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=config.SUBMISSIONS_DIR / "optuna_xgboost_small_search_live.csv",
        help="CSV path updated after every trial.",
    )
    return parser.parse_args()


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    """Build a fold-safe one-hot XGBoost pipeline.

    Args:
        params: XGBoost hyperparameters for this trial.

    Returns:
        Scikit-learn pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", XGBClassifier(**params))])


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest XGBoost parameters for a stronger diverse base model.

    Args:
        trial: Optuna trial object.

    Returns:
        Complete XGBoost parameter dictionary for this trial.
    """
    return {
        "objective": "multi:softprob",
        "num_class": len(config.CLASS_LABELS),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 350, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.085),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "subsample": trial.suggest_float("subsample", 0.75, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95, step=0.05),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 4.0),
        "random_state": config.SEED,
        "n_jobs": config.N_JOBS,
        "verbosity": 0,
    }


def score_params(train: pd.DataFrame, params: dict[str, Any], trial: optuna.Trial) -> tuple[float, float]:
    """Score one XGBoost parameter set with fixed stratified folds.

    Args:
        train: Training dataframe with features, target, and fold column.
        params: XGBoost hyperparameters.
        trial: Optuna trial object.

    Returns:
        Mean balanced accuracy and mean log loss.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(train[config.TARGET_COLUMN])
    n_classes = len(encoder.classes_)
    fold_scores: list[float] = []
    fold_losses: list[float] = []
    for fold in range(config.N_FOLDS):
        train_idx = train[config.FOLD_COLUMN] != fold
        valid_idx = train[config.FOLD_COLUMN] == fold
        x_train = train.loc[train_idx, config.FEATURE_COLUMNS]
        x_valid = train.loc[valid_idx, config.FEATURE_COLUMNS]
        y_train = y[train_idx.to_numpy()]
        y_valid = y[valid_idx.to_numpy()]

        pipeline = build_pipeline(params)
        pipeline.fit(x_train, y_train)
        valid_proba = pipeline.predict_proba(x_valid)
        valid_pred = valid_proba.argmax(axis=1)
        fold_score = balanced_accuracy_score(y_valid, valid_pred)
        fold_loss = log_loss(y_valid, valid_proba, labels=np.arange(n_classes))
        fold_scores.append(float(fold_score))
        fold_losses.append(float(fold_loss))
        trial.set_user_attr(f"fold_{fold}_balanced_accuracy", float(fold_score))
        trial.set_user_attr(f"fold_{fold}_log_loss", float(fold_loss))
        trial.report(float(np.mean(fold_scores)), step=fold)
        if fold >= 1 and trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_scores)), float(np.mean(fold_losses))


def make_objective(train: pd.DataFrame, best_score: float) -> Callable[[optuna.Trial], float]:
    """Create the XGBoost Optuna objective.

    Args:
        train: Training dataframe with fixed folds.
        best_score: Current champion score for logging context.

    Returns:
        Objective function.
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        with mlflow.start_run(run_name=f"xgboost_optuna_trial_{trial.number:03d}", nested=True):
            mlflow.log_param("seed", config.SEED)
            mlflow.log_param("n_folds", config.N_FOLDS)
            mlflow.log_param("current_best_score", best_score)
            mlflow.log_param("features", ",".join(config.FEATURE_COLUMNS))
            mlflow.log_params(params)
            mean_score, mean_loss = score_params(train, params, trial)
            mlflow.log_metric("cv_mean_balanced_accuracy", mean_score)
            mlflow.log_metric("cv_mean_log_loss", mean_loss)
        return mean_score

    return objective


def write_results(study: optuna.Study, output_path: Path) -> Path:
    """Write Optuna study results to CSV.

    Args:
        study: Optuna study.
        output_path: CSV output path.

    Returns:
        Written CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_path, index=False)
    return output_path


def main() -> None:
    """Run the XGBoost Optuna search."""
    args = parse_args()
    ensure_output_dirs()
    best_score = get_best_score()
    train_raw, _, _ = load_raw_data()
    train = create_stratified_folds(make_features(train_raw), n_folds=config.N_FOLDS)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    sampler = optuna.samplers.TPESampler(seed=config.SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    storage_uri = f"sqlite:///{args.storage_path.as_posix()}"
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage_uri,
        load_if_exists=True,
    )

    with mlflow.start_run(run_name=f"{args.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("search_trials", args.trials)
        mlflow.log_param("search_sampler", sampler.__class__.__name__)
        mlflow.log_param("search_pruner", pruner.__class__.__name__)
        mlflow.log_param("search_storage_path", str(args.storage_path))
        mlflow.log_param("search_results_path", str(args.results_path))
        mlflow.log_param("current_best_score", best_score)
        study.optimize(
            make_objective(train, best_score),
            n_trials=args.trials,
            callbacks=[lambda current_study, _: write_results(current_study, args.results_path)],
            show_progress_bar=True,
        )
        results_path = write_results(study, args.results_path)
        mlflow.log_artifact(str(results_path))
        if study.best_trial.value is not None:
            mlflow.log_metric("optuna_best_balanced_accuracy", float(study.best_value))

    print(f"Current best balanced accuracy: {best_score:.8f}")
    print(f"Completed trials in study: {len(study.trials)}")
    print(f"XGBoost Optuna best balanced accuracy: {study.best_value:.8f}")
    print(f"Results written to: {results_path}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Top {args.top_n} completed trials:")
    completed = [
        trial
        for trial in study.trials
        if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE
    ]
    for trial in sorted(completed, key=lambda item: item.value or float("-inf"), reverse=True)[: args.top_n]:
        print(f"  trial={trial.number} score={trial.value:.8f} params={trial.params}")


if __name__ == "__main__":
    main()
