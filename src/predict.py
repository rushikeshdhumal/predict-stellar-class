"""Prediction helpers for fold ensembles."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src import config


def predict_test_ensemble(
    models: list[Pipeline],
    test: pd.DataFrame,
    label_encoder: LabelEncoder,
) -> np.ndarray:
    """Predict test labels by averaging fold probabilities.

    Args:
        models: Trained fold pipelines.
        test: Test dataframe with raw feature columns.
        label_encoder: Fitted target label encoder.

    Returns:
        Predicted class labels for the test rows.
    """
    probabilities = np.zeros((len(test), len(label_encoder.classes_)), dtype=float)
    for model in models:
        probabilities += model.predict_proba(test[config.FEATURE_COLUMNS])
    probabilities /= len(models)
    return label_encoder.inverse_transform(probabilities.argmax(axis=1))
