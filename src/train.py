"""Model training utilities for stratified cross-validation."""

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src import config


def build_baseline_pipeline() -> Pipeline:
    """Build the raw-feature LightGBM baseline pipeline.

    Returns:
        A scikit-learn pipeline with fold-safe preprocessing and LightGBM.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = LGBMClassifier(**config.LGBM_PARAMS)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model_cv(
    train: pd.DataFrame,
) -> tuple[list[Pipeline], np.ndarray, np.ndarray, LabelEncoder, float]:
    """Train a 5-fold stratified LightGBM baseline.

    Args:
        train: Training dataframe with feature, target, and fold columns.

    Returns:
        Trained fold pipelines, OOF class probabilities, balanced accuracy scores,
        fitted label encoder, and mean log loss.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    encoder = LabelEncoder()
    y = encoder.fit_transform(train[config.TARGET_COLUMN])
    n_classes = len(encoder.classes_)
    oof_preds = np.zeros((len(train), n_classes), dtype=float)
    cv_scores: list[float] = []
    fold_losses: list[float] = []
    models: list[Pipeline] = []

    for fold in range(config.N_FOLDS):
        train_idx = train[config.FOLD_COLUMN] != fold
        valid_idx = train[config.FOLD_COLUMN] == fold
        x_train = train.loc[train_idx, config.FEATURE_COLUMNS]
        x_valid = train.loc[valid_idx, config.FEATURE_COLUMNS]
        y_train = y[train_idx.to_numpy()]
        y_valid = y[valid_idx.to_numpy()]

        pipeline = build_baseline_pipeline()
        pipeline.fit(x_train, y_train)

        valid_proba = pipeline.predict_proba(x_valid)
        valid_pred = valid_proba.argmax(axis=1)
        fold_ba = balanced_accuracy_score(y_valid, valid_pred)
        fold_loss = log_loss(y_valid, valid_proba, labels=np.arange(n_classes))
        oof_preds[valid_idx.to_numpy()] = valid_proba

        mlflow.log_metric(f"fold_{fold}_balanced_accuracy", fold_ba)
        mlflow.log_metric(f"fold_{fold}_log_loss", fold_loss)
        joblib.dump(pipeline, config.MODELS_DIR / f"{config.MODEL_NAME}_fold_{fold}.joblib")

        cv_scores.append(fold_ba)
        fold_losses.append(fold_loss)
        models.append(pipeline)
        print(f"Fold {fold}: balanced_accuracy={fold_ba:.6f}, log_loss={fold_loss:.6f}")

    mean_loss = float(np.mean(fold_losses))
    return models, oof_preds, np.array(cv_scores), encoder, mean_loss
