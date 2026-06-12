"""Data loading, baseline EDA, and fold creation utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw train, test, and sample submission files.

    Returns:
        A tuple containing training data, test data, and sample submission data.
    """
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    return train, test, sample


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create accepted features plus spatial angle encodings.

    Args:
        data: Input dataframe containing raw competition columns.

    Returns:
        Dataframe with configured feature columns plus any target column present.
    """
    featured = data.copy()
    featured["u_g"] = featured["u"] - featured["g"]
    featured["g_r"] = featured["g"] - featured["r"]
    featured["r_i"] = featured["r"] - featured["i"]
    featured["i_z"] = featured["i"] - featured["z"]
    featured["u_r"] = featured["u"] - featured["r"]
    featured["g_i"] = featured["g"] - featured["i"]
    featured["r_z"] = featured["r"] - featured["z"]
    featured["u_z"] = featured["u"] - featured["z"]
    magnitude_bands = featured[config.MAGNITUDE_BANDS]
    featured["mag_mean"] = magnitude_bands.mean(axis=1)
    featured["mag_std"] = magnitude_bands.std(axis=1)
    featured["mag_min"] = magnitude_bands.min(axis=1)
    featured["mag_max"] = magnitude_bands.max(axis=1)
    featured["mag_range"] = featured["mag_max"] - featured["mag_min"]
    featured["spectral_population"] = (
        featured["spectral_type"].astype(str) + "_" + featured["galaxy_population"].astype(str)
    )
    alpha_rad = np.deg2rad(featured["alpha"])
    delta_rad = np.deg2rad(featured["delta"])
    featured["alpha_sin"] = np.sin(alpha_rad)
    featured["alpha_cos"] = np.cos(alpha_rad)
    featured["delta_sin"] = np.sin(delta_rad)
    featured["delta_cos"] = np.cos(delta_rad)
    featured["sky_x"] = featured["delta_cos"] * featured["alpha_cos"]
    featured["sky_y"] = featured["delta_cos"] * featured["alpha_sin"]
    featured["sky_z"] = featured["delta_sin"]

    keep_columns = [config.ID_COLUMN, *config.FEATURE_COLUMNS]
    if config.TARGET_COLUMN in featured.columns:
        keep_columns.append(config.TARGET_COLUMN)
    return featured.loc[:, keep_columns].copy()


def create_stratified_folds(data: pd.DataFrame, n_folds: int = config.N_FOLDS) -> pd.DataFrame:
    """Assign stratified CV folds by target class.

    Args:
        data: Training dataframe with the target column.
        n_folds: Number of stratified folds to create.

    Returns:
        Training dataframe with a fold column.
    """
    folded = data.copy()
    folded[config.FOLD_COLUMN] = -1
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    for fold, (_, valid_idx) in enumerate(
        splitter.split(folded[config.FEATURE_COLUMNS], folded[config.TARGET_COLUMN])
    ):
        folded.loc[valid_idx, config.FOLD_COLUMN] = fold

    return folded


def write_eda_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_path: Path = config.EDA_REPORT_PATH,
) -> Path:
    """Write a concise EDA report for the raw baseline data.

    Args:
        train: Raw training dataframe.
        test: Raw test dataframe.
        output_path: Destination markdown report path.

    Returns:
        Path to the written report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_train = train.isna().sum()
    missing_test = test.isna().sum()
    numeric_summary = _dataframe_to_markdown(
        train[config.BASE_NUMERIC_FEATURES].describe().round(4)
    )
    train_features = make_features(train)
    test_features = make_features(test)
    shape_stats = _numeric_shape_stats(train_features, config.NUMERIC_FEATURES)
    numeric_drift = _numeric_drift_stats(
        train_features,
        test_features,
        config.NUMERIC_FEATURES,
    )
    categorical_drift = _categorical_drift_stats(
        train_features,
        test_features,
        config.CATEGORICAL_FEATURES,
    )

    lines = [
        "# Baseline EDA",
        "",
        f"- Train shape: {train.shape[0]} rows x {train.shape[1]} columns",
        f"- Test shape: {test.shape[0]} rows x {test.shape[1]} columns",
        f"- Target classes: {', '.join(train[config.TARGET_COLUMN].value_counts().index)}",
        f"- Train missing values: {int(missing_train.sum())}",
        f"- Test missing values: {int(missing_test.sum())}",
        "",
        "## Class Balance",
        "",
        _series_to_markdown(train[config.TARGET_COLUMN].value_counts()),
        "",
        "## Categorical Values",
        "",
        "### spectral_type",
        _series_to_markdown(train["spectral_type"].value_counts()),
        "",
        "### galaxy_population",
        _series_to_markdown(train["galaxy_population"].value_counts()),
        "",
        "## Numeric Summary",
        "",
        numeric_summary,
        "",
        "## Active Numeric Feature Shape",
        "",
        "Skewness and kurtosis are computed on the training data. Kurtosis uses pandas' Fisher definition, where a normal distribution has kurtosis near 0.",
        "",
        _dataframe_to_markdown(shape_stats.round(4)),
        "",
        "## Numeric Train/Test Drift",
        "",
        "`standardized_mean_diff` is `(test_mean - train_mean) / train_std`. Larger absolute values suggest stronger mean drift.",
        "",
        _dataframe_to_markdown(numeric_drift.round(6)),
        "",
        "## Categorical Train/Test Drift",
        "",
        "`max_abs_proportion_diff` is the largest train/test category-proportion difference within the feature.",
        "",
        _dataframe_to_markdown(categorical_drift.round(6)),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _numeric_shape_stats(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute skewness and kurtosis diagnostics for numeric features.

    Args:
        data: Dataframe containing numeric feature columns.
        columns: Numeric columns to summarize.

    Returns:
        Dataframe indexed by feature with skewness and kurtosis diagnostics.
    """
    stats = pd.DataFrame(index=columns)
    stats["mean"] = data[columns].mean()
    stats["std"] = data[columns].std()
    stats["skew"] = data[columns].skew()
    stats["kurtosis"] = data[columns].kurt()
    stats["min"] = data[columns].min()
    stats["max"] = data[columns].max()
    return stats


def _numeric_drift_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Compute train/test drift diagnostics for numeric features.

    Args:
        train: Training feature dataframe.
        test: Test feature dataframe.
        columns: Numeric columns to compare.

    Returns:
        Dataframe indexed by feature with mean, std, and quantile drift metrics.
    """
    train_mean = train[columns].mean()
    test_mean = test[columns].mean()
    train_std = train[columns].std().replace(0, np.nan)
    test_std = test[columns].std()
    train_median = train[columns].median()
    test_median = test[columns].median()
    drift = pd.DataFrame(index=columns)
    drift["train_mean"] = train_mean
    drift["test_mean"] = test_mean
    drift["standardized_mean_diff"] = (test_mean - train_mean) / train_std
    drift["train_std"] = train_std
    drift["test_std"] = test_std
    drift["std_ratio"] = test_std / train_std
    drift["train_median"] = train_median
    drift["test_median"] = test_median
    drift["median_diff"] = test_median - train_median
    return drift.fillna(0)


def _categorical_drift_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Compute train/test category proportion drift diagnostics.

    Args:
        train: Training feature dataframe.
        test: Test feature dataframe.
        columns: Categorical columns to compare.

    Returns:
        Dataframe indexed by feature with cardinality and proportion drift metrics.
    """
    rows: list[dict[str, object]] = []
    for column in columns:
        train_props = train[column].value_counts(normalize=True)
        test_props = test[column].value_counts(normalize=True)
        categories = train_props.index.union(test_props.index)
        diffs = (test_props.reindex(categories, fill_value=0) - train_props.reindex(categories, fill_value=0)).abs()
        rows.append(
            {
                "feature": column,
                "train_unique": int(train[column].nunique()),
                "test_unique": int(test[column].nunique()),
                "max_abs_proportion_diff": float(diffs.max()),
                "largest_drift_category": str(diffs.idxmax()),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def _series_to_markdown(series: pd.Series) -> str:
    """Format a series as a small markdown table without optional dependencies.

    Args:
        series: Series to format.

    Returns:
        Markdown table text.
    """
    rows = ["| value | count |", "|---|---:|"]
    rows.extend(f"| {index} | {value} |" for index, value in series.items())
    return "\n".join(rows)


def _dataframe_to_markdown(data: pd.DataFrame) -> str:
    """Format a dataframe as markdown without optional dependencies.

    Args:
        data: Dataframe to format.

    Returns:
        Markdown table text.
    """
    header = "| statistic | " + " | ".join(str(column) for column in data.columns) + " |"
    separator = "|---|" + "|".join("---:" for _ in data.columns) + "|"
    rows = [header, separator]
    for index, row in data.iterrows():
        values = " | ".join(str(value) for value in row.tolist())
        rows.append(f"| {index} | {values} |")
    return "\n".join(rows)
