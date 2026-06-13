"""Rich meta-feature construction for supervised stacking experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data import load_raw_data, make_features
from src.ensemble import load_probability_artifact


CONTEXT_NUMERIC_FEATURES: list[str] = [
    "redshift",
    *config.COLOR_FEATURES,
    *config.MAGNITUDE_SUMMARY_FEATURES,
]
CONTEXT_CATEGORICAL_FEATURES: list[str] = [
    "spectral_type",
    "galaxy_population",
    "spectral_population",
]


@dataclass(frozen=True)
class RichMetaFeatures:
    """Fold-safe rich meta-feature matrices.

    Attributes:
        train_features: Training meta-feature dataframe aligned to OOF rows.
        test_features: Test meta-feature dataframe aligned to test probability rows.
        numeric_columns: Numeric meta-feature columns.
        categorical_columns: Categorical meta-feature columns.
        y_true: Training labels aligned to ``train_features``.
        folds: Fold assignments aligned to ``train_features``.
        reference_oof: First OOF artifact, used as the output reference.
        reference_test: First test artifact, used as the output reference.
    """

    train_features: pd.DataFrame
    test_features: pd.DataFrame
    numeric_columns: list[str]
    categorical_columns: list[str]
    y_true: np.ndarray
    folds: np.ndarray
    reference_oof: pd.DataFrame
    reference_test: pd.DataFrame


def default_rich_base_names() -> list[str]:
    """Return the first rich-stacking base set to evaluate.

    Returns:
        Base artifact names, ordered with the strongest blend as the anchor.
    """
    return [
        "blend_deterministic_lgbm_entity_mlp_e12_w0435001",
        "blend_lightgbm_deterministic_local",
        "entity_embedding_mlp_small_e12",
        "ft_transformer_ddp_t4",
        "catboost_native_depth_8",
        "lightgbm_dart_conservative",
        "xgboost_hist_depth_8",
        "xgboost_optuna_trial_0",
        "lightgbm_gbdt_balanced_low_lr",
    ]


def resolve_probability_paths(base_names: list[str]) -> tuple[list[Path], list[Path]]:
    """Resolve OOF and test probability paths for base artifact names.

    Args:
        base_names: Stable base artifact names.

    Returns:
        OOF paths and matching test probability paths.
    """
    oof_paths = [_resolve_one_path(name, suffix="_oof.csv") for name in base_names]
    test_paths = [_resolve_one_path(name, suffix="_test_proba.csv") for name in base_names]
    return oof_paths, test_paths


def build_rich_meta_features(
    oof_paths: list[Path],
    test_proba_paths: list[Path],
    base_names: list[str],
    *,
    include_context: bool = True,
    scalar_oof_paths: list[Path] | None = None,
    scalar_test_paths: list[Path] | None = None,
) -> RichMetaFeatures:
    """Build aligned train/test rich meta-feature matrices.

    Args:
        oof_paths: Base OOF probability artifact paths.
        test_proba_paths: Matching base test probability artifact paths.
        base_names: Human-readable base names in the same order.
        include_context: Whether to add small raw-feature context columns.
        scalar_oof_paths: Optional aligned OOF scalar-feature artifacts.
        scalar_test_paths: Optional aligned test scalar-feature artifacts.

    Returns:
        Rich meta-feature matrices and alignment metadata.
    """
    if len(oof_paths) != len(test_proba_paths):
        raise ValueError("OOF and test artifact path counts must match.")
    if len(oof_paths) != len(base_names):
        raise ValueError("Base name count must match artifact path count.")

    oof_frames = [load_probability_artifact(path) for path in oof_paths]
    test_frames = [load_probability_artifact(path) for path in test_proba_paths]
    _validate_base_artifacts(oof_frames, test_frames)

    train_numeric = _build_probability_meta_features(oof_frames, base_names)
    test_numeric = _build_probability_meta_features(test_frames, base_names)
    categorical_columns: list[str] = []

    if scalar_oof_paths or scalar_test_paths:
        train_scalar, test_scalar = _load_scalar_feature_artifacts(
            scalar_oof_paths or [],
            scalar_test_paths or [],
            oof_frames[0],
            test_frames[0],
        )
        train_numeric = pd.concat([train_numeric, train_scalar], axis=1)
        test_numeric = pd.concat([test_numeric, test_scalar], axis=1)

    if include_context:
        train_context, test_context = _build_context_features(oof_frames[0], test_frames[0])
        train_numeric = pd.concat([train_numeric, train_context[CONTEXT_NUMERIC_FEATURES]], axis=1)
        test_numeric = pd.concat([test_numeric, test_context[CONTEXT_NUMERIC_FEATURES]], axis=1)
        train_categorical = train_context[CONTEXT_CATEGORICAL_FEATURES].reset_index(drop=True)
        test_categorical = test_context[CONTEXT_CATEGORICAL_FEATURES].reset_index(drop=True)
        categorical_columns = CONTEXT_CATEGORICAL_FEATURES.copy()
        train_features = pd.concat([train_numeric, train_categorical], axis=1)
        test_features = pd.concat([test_numeric, test_categorical], axis=1)
    else:
        train_features = train_numeric
        test_features = test_numeric

    numeric_columns = train_numeric.columns.tolist()
    _validate_no_identifier_features(numeric_columns, categorical_columns)
    if train_features.columns.tolist() != test_features.columns.tolist():
        raise ValueError("Train/test meta-feature columns are not aligned.")

    return RichMetaFeatures(
        train_features=train_features,
        test_features=test_features,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        y_true=oof_frames[0]["y_true"].to_numpy(),
        folds=oof_frames[0]["fold"].to_numpy(),
        reference_oof=oof_frames[0],
        reference_test=test_frames[0],
    )


def _build_probability_meta_features(frames: list[pd.DataFrame], base_names: list[str]) -> pd.DataFrame:
    """Build probability-derived meta features for one aligned split."""
    probabilities = [frame[config.PROBA_COLUMNS].to_numpy(dtype=np.float64) for frame in frames]
    stacked = np.stack(probabilities, axis=1)
    features: dict[str, np.ndarray] = {}

    for base_index, (name, proba) in enumerate(zip(base_names, probabilities)):
        safe_name = _safe_feature_name(name)
        clipped = np.clip(proba, 1e-8, 1.0)
        logits = np.log(clipped)
        sorted_proba = np.sort(proba, axis=1)
        entropy = -(clipped * np.log(clipped)).sum(axis=1)
        features[f"{safe_name}__entropy"] = entropy
        features[f"{safe_name}__top_margin"] = sorted_proba[:, -1] - sorted_proba[:, -2]
        features[f"{safe_name}__max_proba"] = sorted_proba[:, -1]
        for class_index, label in enumerate(config.CLASS_LABELS):
            features[f"{safe_name}__log_proba_{label}"] = logits[:, class_index]
            features[f"{safe_name}__proba_{label}"] = proba[:, class_index]
        for left_index, left_label in enumerate(config.CLASS_LABELS):
            for right_index, right_label in enumerate(config.CLASS_LABELS):
                if left_index >= right_index:
                    continue
                features[f"{safe_name}__logit_gap_{left_label}_minus_{right_label}"] = (
                    logits[:, left_index] - logits[:, right_index]
                )
        if base_index > 0:
            anchor = probabilities[0]
            anchor_logits = np.log(np.clip(anchor, 1e-8, 1.0))
            for class_index, label in enumerate(config.CLASS_LABELS):
                features[f"{safe_name}__delta_anchor_proba_{label}"] = proba[:, class_index] - anchor[:, class_index]
                features[f"{safe_name}__delta_anchor_logit_{label}"] = logits[:, class_index] - anchor_logits[:, class_index]

    for class_index, label in enumerate(config.CLASS_LABELS):
        class_values = stacked[:, :, class_index]
        features[f"all_bases__mean_proba_{label}"] = class_values.mean(axis=1)
        features[f"all_bases__std_proba_{label}"] = class_values.std(axis=1)
        features[f"all_bases__min_proba_{label}"] = class_values.min(axis=1)
        features[f"all_bases__max_proba_{label}"] = class_values.max(axis=1)
        features[f"all_bases__vote_count_{label}"] = (stacked.argmax(axis=2) == class_index).sum(axis=1)

    all_sorted = np.sort(stacked.mean(axis=1), axis=1)
    features["all_bases__mean_top_margin"] = all_sorted[:, -1] - all_sorted[:, -2]
    features["all_bases__mean_entropy"] = -(
        np.clip(stacked.mean(axis=1), 1e-8, 1.0) * np.log(np.clip(stacked.mean(axis=1), 1e-8, 1.0))
    ).sum(axis=1)
    return pd.DataFrame(features)


def _build_context_features(reference_oof: pd.DataFrame, reference_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and align small raw-feature context frames by competition id."""
    train_raw, test_raw, _ = load_raw_data()
    train_features = make_features(train_raw).set_index(config.ID_COLUMN)
    test_features = make_features(test_raw).set_index(config.ID_COLUMN)
    train_context = train_features.loc[reference_oof["id"].to_numpy(), CONTEXT_NUMERIC_FEATURES + CONTEXT_CATEGORICAL_FEATURES]
    test_context = test_features.loc[reference_test["id"].to_numpy(), CONTEXT_NUMERIC_FEATURES + CONTEXT_CATEGORICAL_FEATURES]
    return train_context.reset_index(drop=True), test_context.reset_index(drop=True)


def _load_scalar_feature_artifacts(
    scalar_oof_paths: list[Path],
    scalar_test_paths: list[Path],
    reference_oof: pd.DataFrame,
    reference_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aligned numeric scalar meta-feature artifacts."""
    if len(scalar_oof_paths) != len(scalar_test_paths):
        raise ValueError("Scalar OOF and test artifact path counts must match.")
    train_frames = []
    test_frames = []
    used_columns: set[str] = set()
    for oof_path, test_path in zip(scalar_oof_paths, scalar_test_paths):
        oof = pd.read_csv(oof_path)
        test = pd.read_csv(test_path)
        _validate_scalar_alignment(oof, test, reference_oof, reference_test)
        feature_columns = [column for column in oof.columns if column not in {"id", "fold", "y_true"}]
        if not feature_columns:
            raise ValueError(f"Scalar artifact has no feature columns: {oof_path}")
        overlap = used_columns.intersection(feature_columns)
        if overlap:
            raise ValueError(f"Duplicate scalar feature columns: {sorted(overlap)}")
        used_columns.update(feature_columns)
        train_frames.append(oof[feature_columns].reset_index(drop=True))
        test_frames.append(test[feature_columns].reset_index(drop=True))
    if not train_frames:
        return pd.DataFrame(index=reference_oof.index), pd.DataFrame(index=reference_test.index)
    train_scalar = pd.concat(train_frames, axis=1)
    test_scalar = pd.concat(test_frames, axis=1)
    if train_scalar.columns.tolist() != test_scalar.columns.tolist():
        raise ValueError("Scalar train/test feature columns are not aligned.")
    if not np.isfinite(train_scalar.to_numpy(dtype=np.float64)).all():
        raise ValueError("Scalar OOF features contain non-finite values.")
    if not np.isfinite(test_scalar.to_numpy(dtype=np.float64)).all():
        raise ValueError("Scalar test features contain non-finite values.")
    return train_scalar, test_scalar


def _resolve_one_path(base_name: str, *, suffix: str) -> Path:
    """Resolve one artifact path from standard ensemble directories."""
    for directory in (config.STACKING_DIR, config.OOF_DIR, config.TEST_PROBA_DIR):
        path = directory / f"{base_name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find artifact for {base_name}{suffix}.")


def _validate_base_artifacts(oof_frames: list[pd.DataFrame], test_frames: list[pd.DataFrame]) -> None:
    """Validate row alignment across base model artifacts."""
    reference_oof = oof_frames[0]
    reference_test = test_frames[0]
    for frame in oof_frames[1:]:
        for column in ["id", "fold", "y_true"]:
            if not reference_oof[column].equals(frame[column]):
                raise ValueError(f"OOF artifacts are not aligned on `{column}`.")
    for frame in test_frames[1:]:
        if not reference_test["id"].equals(frame["id"]):
            raise ValueError("Test probability artifacts are not aligned on `id`.")
    if reference_oof["id"].duplicated().any():
        raise ValueError("Reference OOF artifact contains duplicate ids.")
    if reference_oof["fold"].lt(0).any():
        raise ValueError("Reference OOF artifact contains unassigned folds.")


def _validate_scalar_alignment(
    scalar_oof: pd.DataFrame,
    scalar_test: pd.DataFrame,
    reference_oof: pd.DataFrame,
    reference_test: pd.DataFrame,
) -> None:
    """Validate scalar artifacts align with probability artifacts."""
    for column in ["id", "fold", "y_true"]:
        if column not in scalar_oof.columns:
            raise ValueError(f"Scalar OOF artifact is missing `{column}`.")
        if not reference_oof[column].equals(scalar_oof[column]):
            raise ValueError(f"Scalar OOF artifact is not aligned on `{column}`.")
    if "id" not in scalar_test.columns:
        raise ValueError("Scalar test artifact is missing `id`.")
    if not reference_test["id"].equals(scalar_test["id"]):
        raise ValueError("Scalar test artifact is not aligned on `id`.")


def _safe_feature_name(name: str) -> str:
    """Convert an artifact name into a stable feature-name prefix."""
    return "".join(character if character.isalnum() else "_" for character in name)


def _validate_no_identifier_features(numeric_columns: list[str], categorical_columns: list[str]) -> None:
    """Fail fast if an identifier leaks into the meta-feature matrix."""
    forbidden = {config.ID_COLUMN, "id", "obj_ID"}
    leaked = forbidden.intersection(numeric_columns).union(forbidden.intersection(categorical_columns))
    if leaked:
        raise ValueError(f"Identifier columns are not allowed as features: {sorted(leaked)}")
