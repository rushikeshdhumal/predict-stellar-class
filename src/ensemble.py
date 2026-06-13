"""OOF artifact and base-model utilities for supervised ensembling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src import config

ModelFamily = Literal["lightgbm", "xgboost", "catboost"]


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one base model in the ensemble zoo.

    Attributes:
        name: Stable artifact/model name.
        family: Model implementation family.
        params: Constructor parameters for the model.
        native_categoricals: Whether the model consumes categorical columns directly.
        feature_columns: Optional feature subset for specialist models.
    """

    name: str
    family: ModelFamily
    params: dict[str, Any]
    native_categoricals: bool = False
    feature_columns: list[str] | None = None


@dataclass(frozen=True)
class BaseModelResult:
    """Result metadata for an OOF base-model run.

    Attributes:
        model_name: Stable base-model name.
        mean_balanced_accuracy: Mean validation balanced accuracy.
        mean_log_loss: Mean validation log loss.
        fold_scores: Per-fold balanced accuracy scores.
        oof_path: Saved OOF probability artifact path.
        test_proba_path: Saved averaged test probability artifact path.
    """

    model_name: str
    mean_balanced_accuracy: float
    mean_log_loss: float
    fold_scores: list[float]
    oof_path: Path
    test_proba_path: Path


def default_model_specs() -> dict[str, ModelSpec]:
    """Build the default supervised base-model registry.

    Returns:
        Mapping from model name to base-model specification.
    """
    champion_params = dict(config.LGBM_PARAMS)
    specs = [
        ModelSpec("lightgbm_optuna_trial_84_n_estimators_200", "lightgbm", champion_params),
        ModelSpec(
            "lightgbm_optuna_trial_49",
            "lightgbm",
            _lgbm_params(
                learning_rate=0.068001,
                n_estimators=280,
                num_leaves=208,
                min_child_samples=20,
                subsample=1.0,
                colsample_bytree=0.8,
                reg_lambda=2.75,
                reg_alpha=1.6,
            ),
        ),
        ModelSpec(
            "lightgbm_optuna_trial_65",
            "lightgbm",
            _lgbm_params(
                learning_rate=0.062685,
                n_estimators=260,
                num_leaves=304,
                min_child_samples=20,
                subsample=0.95,
                colsample_bytree=0.75,
                reg_lambda=4.25,
                reg_alpha=1.8,
            ),
        ),
        ModelSpec(
            "lightgbm_optuna_trial_81",
            "lightgbm",
            _lgbm_params(
                learning_rate=0.067218,
                n_estimators=260,
                num_leaves=288,
                min_child_samples=30,
                subsample=0.9,
                colsample_bytree=0.75,
                reg_lambda=3.25,
                reg_alpha=2.0,
            ),
        ),
        ModelSpec(
            "lightgbm_optuna_trial_31",
            "lightgbm",
            _lgbm_params(
                learning_rate=0.050351,
                n_estimators=260,
                num_leaves=240,
                min_child_samples=50,
                subsample=1.0,
                colsample_bytree=0.75,
                reg_lambda=2.0,
                reg_alpha=1.6,
            ),
        ),
        ModelSpec(
            "lightgbm_dart_conservative",
            "lightgbm",
            {
                "boosting_type": "dart",
                "objective": "multiclass",
                "num_class": len(config.CLASS_LABELS),
                "n_estimators": 420,
                "max_depth": -1,
                "num_leaves": 160,
                "min_child_samples": 40,
                "learning_rate": 0.045,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "reg_lambda": 5.0,
                "reg_alpha": 2.5,
                "drop_rate": 0.08,
                "skip_drop": 0.50,
                "max_drop": 50,
                "random_state": config.SEED,
                "n_jobs": config.N_JOBS,
                "verbosity": -1,
            },
        ),
        ModelSpec(
            "lightgbm_gbdt_balanced_low_lr",
            "lightgbm",
            {
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "num_class": len(config.CLASS_LABELS),
                "class_weight": "balanced",
                "n_estimators": 520,
                "max_depth": -1,
                "num_leaves": 96,
                "min_child_samples": 70,
                "learning_rate": 0.032,
                "subsample": 0.90,
                "colsample_bytree": 0.80,
                "reg_lambda": 7.0,
                "reg_alpha": 3.0,
                "random_state": config.SEED,
                "n_jobs": config.N_JOBS,
                "verbosity": -1,
            },
        ),
        ModelSpec(
            "lightgbm_photometry_redshift_specialist",
            "lightgbm",
            _lgbm_params(
                learning_rate=0.055,
                n_estimators=260,
                num_leaves=128,
                min_child_samples=50,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=4.0,
                reg_alpha=2.0,
            ),
            feature_columns=[
                "u",
                "g",
                "r",
                "i",
                "z",
                "redshift",
                *config.COLOR_FEATURES,
                *config.MAGNITUDE_SUMMARY_FEATURES,
            ],
        ),
        ModelSpec(
            "lightgbm_categorical_redshift_specialist",
            "lightgbm",
            _lgbm_params(
                learning_rate=0.06,
                n_estimators=220,
                num_leaves=64,
                min_child_samples=80,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=4.0,
                reg_alpha=2.0,
            ),
            feature_columns=[
                "redshift",
                "spectral_type",
                "galaxy_population",
                "spectral_population",
            ],
        ),
        ModelSpec("xgboost_hist_depth_8", "xgboost", _xgb_params(max_depth=8)),
        ModelSpec("xgboost_hist_depth_10", "xgboost", _xgb_params(max_depth=10)),
        ModelSpec(
            "xgboost_optuna_trial_0",
            "xgboost",
            {
                "objective": "multi:softprob",
                "num_class": len(config.CLASS_LABELS),
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "n_estimators": 550,
                "learning_rate": 0.08204285838459498,
                "max_depth": 7,
                "min_child_weight": 12.374511199743695,
                "subsample": 0.75,
                "colsample_bytree": 0.7000000000000001,
                "gamma": 0.17425083650459838,
                "reg_lambda": 8.795585311974417,
                "reg_alpha": 2.404460046972835,
                "random_state": config.SEED,
                "n_jobs": config.N_JOBS,
                "verbosity": 0,
            },
        ),
        ModelSpec(
            "catboost_native_depth_8",
            "catboost",
            {
                "iterations": 900,
                "learning_rate": 0.055,
                "depth": 8,
                "loss_function": "MultiClass",
                "eval_metric": "MultiClass",
                "random_seed": config.SEED,
                "allow_writing_files": False,
                "verbose": False,
            },
            native_categoricals=True,
        ),
        ModelSpec(
            "catboost_native_balanced_depth_7_od120",
            "catboost",
            {
                "iterations": 1800,
                "learning_rate": 0.035,
                "depth": 7,
                "l2_leaf_reg": 8.0,
                "random_strength": 1.25,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 0.4,
                "auto_class_weights": "Balanced",
                "loss_function": "MultiClass",
                "eval_metric": "MultiClass",
                "use_best_model": True,
                "od_type": "Iter",
                "od_wait": 120,
                "random_seed": config.SEED,
                "allow_writing_files": False,
                "verbose": False,
            },
            native_categoricals=True,
        ),
    ]
    return {spec.name: spec for spec in specs}


def train_base_model_oof(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    save_models: bool = True,
) -> BaseModelResult:
    """Train one base model and save OOF/test probability artifacts.

    Args:
        spec: Base-model specification.
        train: Featured training data with target and fold columns.
        test: Featured test data.
        save_models: Whether to persist each fold model to `models/`.

    Returns:
        Base-model result metadata.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    config.TEST_PROBA_DIR.mkdir(parents=True, exist_ok=True)

    encoder = LabelEncoder()
    y = encoder.fit_transform(train[config.TARGET_COLUMN])
    _validate_label_order(encoder)
    n_classes = len(config.CLASS_LABELS)
    oof_proba = np.zeros((len(train), n_classes), dtype=float)
    test_proba = np.zeros((len(test), n_classes), dtype=float)
    fold_scores: list[float] = []
    fold_losses: list[float] = []
    feature_columns = _feature_columns(spec)

    for fold in range(config.N_FOLDS):
        train_idx = train[config.FOLD_COLUMN] != fold
        valid_idx = train[config.FOLD_COLUMN] == fold
        x_train = train.loc[train_idx, feature_columns]
        x_valid = train.loc[valid_idx, feature_columns]
        y_train = y[train_idx.to_numpy()]
        y_valid = y[valid_idx.to_numpy()]

        estimator = _build_estimator(spec)
        _fit_estimator(estimator, spec, x_train, y_train, x_valid, y_valid)
        valid_proba = _predict_proba(estimator, x_valid)
        fold_test_proba = _predict_proba(estimator, test[feature_columns])
        valid_pred = valid_proba.argmax(axis=1)
        fold_score = balanced_accuracy_score(y_valid, valid_pred)
        fold_loss = log_loss(y_valid, valid_proba, labels=np.arange(n_classes))

        oof_proba[valid_idx.to_numpy()] = valid_proba
        test_proba += fold_test_proba / config.N_FOLDS
        fold_scores.append(float(fold_score))
        fold_losses.append(float(fold_loss))

        if save_models:
            model_path = config.MODELS_DIR / f"{spec.name}_fold_{fold}.joblib"
            joblib.dump(estimator, model_path)
        print(f"{spec.name} fold {fold}: balanced_accuracy={fold_score:.6f}, log_loss={fold_loss:.6f}")

    oof_path = config.OOF_DIR / f"{spec.name}_oof.csv"
    test_proba_path = config.TEST_PROBA_DIR / f"{spec.name}_test_proba.csv"
    _write_oof_artifact(train, oof_proba, oof_path)
    _write_test_proba_artifact(test, test_proba, test_proba_path)
    return BaseModelResult(
        model_name=spec.name,
        mean_balanced_accuracy=float(np.mean(fold_scores)),
        mean_log_loss=float(np.mean(fold_losses)),
        fold_scores=fold_scores,
        oof_path=oof_path,
        test_proba_path=test_proba_path,
    )


def load_probability_artifact(path: Path) -> pd.DataFrame:
    """Load and validate a probability artifact.

    Args:
        path: OOF or test probability CSV path.

    Returns:
        Probability artifact dataframe.
    """
    data = pd.read_csv(path)
    missing = [column for column in config.PROBA_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing probability columns in {path}: {missing}")
    _validate_probability_rows(data[config.PROBA_COLUMNS].to_numpy(), context=str(path))
    return data


def evaluate_diversity(
    champion_oof: pd.DataFrame,
    candidate_oof: pd.DataFrame,
    *,
    min_blend_weight: float = 0.05,
    max_blend_weight: float = 0.50,
    blend_step: float = 0.05,
    spearman_threshold: float = 0.985,
    max_score_gap: float = 0.003,
    min_blend_gain: float = 0.0001,
    min_rescue_rate: float = 0.001,
) -> dict[str, float | bool]:
    """Evaluate OOF complementarity between champion and candidate models.

    Args:
        champion_oof: Champion OOF probability artifact.
        candidate_oof: Candidate OOF probability artifact.
        min_blend_weight: Minimum candidate blend weight to search.
        max_blend_weight: Maximum candidate blend weight to search.
        blend_step: Candidate blend weight step size.
        spearman_threshold: Mean Spearman threshold above which probabilities are redundant.
        max_score_gap: Maximum allowed candidate score gap for usefulness.
        min_blend_gain: Required OOF blend gain for usefulness.
        min_rescue_rate: Rescue-rate threshold for meaningful complementarity.

    Returns:
        Diversity and usefulness metrics.
    """
    _validate_oof_alignment(champion_oof, candidate_oof)
    y_true = champion_oof["y_true"].to_numpy()
    champion_proba = champion_oof[config.PROBA_COLUMNS].to_numpy()
    candidate_proba = candidate_oof[config.PROBA_COLUMNS].to_numpy()
    champion_pred = _labels_from_proba(champion_proba)
    candidate_pred = _labels_from_proba(candidate_proba)
    champion_score = balanced_accuracy_score(y_true, champion_pred)
    candidate_score = balanced_accuracy_score(y_true, candidate_pred)

    champion_correct = champion_pred == y_true
    candidate_correct = candidate_pred == y_true
    rescue_rate = float((~champion_correct & candidate_correct).mean())
    new_error_rate = float((champion_correct & ~candidate_correct).mean())
    shared_error_rate = float((~champion_correct & ~candidate_correct).mean())
    disagreement_rate = float((champion_pred != candidate_pred).mean())
    spearman_by_class = [
        float(champion_oof[column].corr(candidate_oof[column], method="spearman"))
        for column in config.PROBA_COLUMNS
    ]
    mean_spearman = float(np.mean(spearman_by_class))

    weights = np.arange(min_blend_weight, max_blend_weight + blend_step / 2, blend_step)
    best_blend_score = champion_score
    best_blend_weight = 0.0
    for weight in weights:
        blended = (1.0 - weight) * champion_proba + weight * candidate_proba
        score = balanced_accuracy_score(y_true, _labels_from_proba(blended))
        if score > best_blend_score:
            best_blend_score = float(score)
            best_blend_weight = float(weight)

    blend_gain = best_blend_score - champion_score
    useful = bool(
        candidate_score > champion_score
        or (
            candidate_score >= champion_score - max_score_gap
            and (mean_spearman < spearman_threshold or rescue_rate >= min_rescue_rate)
            and blend_gain >= min_blend_gain
        )
    )
    return {
        "champion_score": float(champion_score),
        "candidate_score": float(candidate_score),
        "mean_spearman": mean_spearman,
        "spearman_GALAXY": spearman_by_class[0],
        "spearman_QSO": spearman_by_class[1],
        "spearman_STAR": spearman_by_class[2],
        "disagreement_rate": disagreement_rate,
        "rescue_rate": rescue_rate,
        "new_error_rate": new_error_rate,
        "shared_error_rate": shared_error_rate,
        "best_blend_score": best_blend_score,
        "best_blend_weight": best_blend_weight,
        "blend_gain": float(blend_gain),
        "is_useful_for_stacking": useful,
    }


def _lgbm_params(
    *,
    learning_rate: float,
    n_estimators: int,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    reg_alpha: float,
) -> dict[str, Any]:
    """Build a LightGBM multiclass parameter dictionary."""
    return {
        "objective": "multiclass",
        "num_class": len(config.CLASS_LABELS),
        "n_estimators": n_estimators,
        "max_depth": -1,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "random_state": config.SEED,
        "n_jobs": config.N_JOBS,
        "verbosity": -1,
    }


def _xgb_params(*, max_depth: int) -> dict[str, Any]:
    """Build an XGBoost multiclass parameter dictionary."""
    return {
        "objective": "multi:softprob",
        "num_class": len(config.CLASS_LABELS),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": 650,
        "learning_rate": 0.045,
        "max_depth": max_depth,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.5,
        "random_state": config.SEED,
        "n_jobs": config.N_JOBS,
        "verbosity": 0,
    }


def _build_estimator(spec: ModelSpec) -> Any:
    """Build an estimator or pipeline for a base-model spec."""
    if spec.family == "lightgbm":
        return Pipeline(steps=[("preprocessor", _one_hot_preprocessor(spec)), ("model", LGBMClassifier(**spec.params))])
    if spec.family == "xgboost":
        from xgboost import XGBClassifier

        return Pipeline(steps=[("preprocessor", _one_hot_preprocessor(spec)), ("model", XGBClassifier(**spec.params))])
    if spec.family == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("CatBoost is not installed. Install `catboost` before training CatBoost specs.") from exc

        return CatBoostClassifier(**spec.params)
    raise ValueError(f"Unsupported model family: {spec.family}")


def _one_hot_preprocessor(spec: ModelSpec) -> ColumnTransformer:
    """Build fold-safe preprocessing for one-hot tree models."""
    feature_columns = _feature_columns(spec)
    numeric_features = [column for column in config.NUMERIC_FEATURES if column in feature_columns]
    categorical_features = [column for column in config.CATEGORICAL_FEATURES if column in feature_columns]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features))
    return ColumnTransformer(
        transformers=transformers,
    )


def _feature_columns(spec: ModelSpec) -> list[str]:
    """Return the feature columns used by a model spec."""
    return spec.feature_columns or config.FEATURE_COLUMNS


def _fit_estimator(
    estimator: Any,
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> None:
    """Fit an estimator, handling native categorical models."""
    if spec.family == "catboost" and spec.native_categoricals:
        estimator.fit(
            x_train,
            y_train,
            cat_features=config.CATEGORICAL_FEATURES,
            eval_set=(x_valid, y_valid),
        )
        return
    estimator.fit(x_train, y_train)


def _predict_proba(estimator: Any, data: pd.DataFrame) -> np.ndarray:
    """Predict probabilities and validate their shape."""
    proba = np.asarray(estimator.predict_proba(data), dtype=float)
    if proba.shape[1] != len(config.CLASS_LABELS):
        raise ValueError(f"Expected {len(config.CLASS_LABELS)} probability columns, got {proba.shape[1]}")
    return proba


def _write_oof_artifact(train: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write an OOF probability artifact."""
    _validate_probability_rows(proba, context="OOF probabilities")
    output = train[[config.ID_COLUMN, config.FOLD_COLUMN, config.TARGET_COLUMN]].rename(
        columns={config.ID_COLUMN: "id", config.FOLD_COLUMN: "fold", config.TARGET_COLUMN: "y_true"}
    )
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    if output["fold"].lt(0).any():
        raise ValueError("OOF artifact contains unassigned folds.")
    output.to_csv(path, index=False)


def _write_test_proba_artifact(test: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write an averaged test probability artifact."""
    _validate_probability_rows(proba, context="test probabilities")
    output = test[[config.ID_COLUMN]].rename(columns={config.ID_COLUMN: "id"})
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _validate_probability_rows(proba: np.ndarray, *, context: str) -> None:
    """Validate probability values and row sums."""
    if np.isnan(proba).any():
        raise ValueError(f"{context} contains NaN values.")
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"{context} rows do not sum to 1 within tolerance.")


def _validate_label_order(encoder: LabelEncoder) -> None:
    """Validate encoded class order against project configuration."""
    classes = encoder.classes_.tolist()
    if classes != config.CLASS_LABELS:
        raise ValueError(f"Unexpected class order {classes}; expected {config.CLASS_LABELS}.")


def _validate_oof_alignment(champion_oof: pd.DataFrame, candidate_oof: pd.DataFrame) -> None:
    """Validate that two OOF artifacts are row-aligned."""
    columns = ["id", "fold", "y_true"]
    for column in columns:
        if not champion_oof[column].equals(candidate_oof[column]):
            raise ValueError(f"OOF artifacts are not aligned on `{column}`.")


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert class probabilities to configured class labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]
