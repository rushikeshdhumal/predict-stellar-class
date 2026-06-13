"""Write champion error diagnostics for targeted stacking work."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src import config
from src.data import load_raw_data, make_features
from src.ensemble import load_probability_artifact
from src.meta_features import (
    CONTEXT_CATEGORICAL_FEATURES,
    CONTEXT_NUMERIC_FEATURES,
    default_rich_base_names,
    resolve_probability_paths,
)
from src.utils import ensure_output_dirs


DEFAULT_CHAMPION_OOF = config.STACKING_DIR / "meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias_oof.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-oof-path", type=Path, default=DEFAULT_CHAMPION_OOF, help="Champion OOF artifact.")
    parser.add_argument("--base-names", nargs="+", default=default_rich_base_names(), help="Base artifacts to audit.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Markdown report path. Defaults to data/processed with a timestamp.",
    )
    return parser.parse_args()


def main() -> None:
    """Write a diagnostic markdown report for current champion errors."""
    args = parse_args()
    ensure_output_dirs()
    output_path = args.output_path or (
        config.PROCESSED_DATA_DIR / f"champion_error_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    champion = load_probability_artifact(args.champion_oof_path)
    oof_paths, _ = resolve_probability_paths(args.base_names)
    base_frames = [load_probability_artifact(path) for path in oof_paths]
    _validate_base_alignment(champion, base_frames)

    context = _load_context(champion)
    report = _build_report(champion, base_frames, args.base_names, context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Champion diagnostic report written to: {output_path}")


def _build_report(
    champion: pd.DataFrame,
    base_frames: list[pd.DataFrame],
    base_names: list[str],
    context: pd.DataFrame,
) -> str:
    """Build the markdown report content."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    proba = champion[config.PROBA_COLUMNS].to_numpy()
    y_true = champion["y_true"].to_numpy()
    champion_pred = labels[proba.argmax(axis=1)]
    champion_correct = champion_pred == y_true
    margins = np.sort(proba, axis=1)[:, -1] - np.sort(proba, axis=1)[:, -2]
    score = balanced_accuracy_score(y_true, champion_pred)
    confusion = pd.DataFrame(
        confusion_matrix(y_true, champion_pred, labels=config.CLASS_LABELS),
        index=[f"true_{label}" for label in config.CLASS_LABELS],
        columns=[f"pred_{label}" for label in config.CLASS_LABELS],
    )
    lines = [
        "# Champion Error Diagnostics",
        "",
        f"- Champion OOF balanced accuracy: `{score:.8f}`",
        f"- OOF rows: `{len(champion)}`",
        f"- Error rows: `{int((~champion_correct).sum())}`",
        "",
        "## Confusion Matrix",
        "",
        _dataframe_to_markdown(confusion),
        "",
        "## Error Pairs",
        "",
        _dataframe_to_markdown(_error_pair_table(y_true, champion_pred)),
        "",
        "## Margin Buckets",
        "",
        _dataframe_to_markdown(_margin_bucket_table(y_true, champion_pred, margins)),
        "",
        "## Feature Quantile Error Rates",
        "",
        _dataframe_to_markdown(_feature_quantile_table(context, champion_correct)),
        "",
        "## Categorical Error Rates",
        "",
        _dataframe_to_markdown(_categorical_table(context, champion_correct)),
        "",
        "## Base Rescue Sources",
        "",
        _dataframe_to_markdown(_base_rescue_table(champion, base_frames, base_names)),
        "",
    ]
    return "\n".join(lines)


def _error_pair_table(y_true: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Summarize champion mistakes by true/predicted class pair."""
    rows = []
    for true_label in config.CLASS_LABELS:
        for pred_label in config.CLASS_LABELS:
            if true_label == pred_label:
                continue
            mask = (y_true == true_label) & (predicted == pred_label)
            rows.append({"true": true_label, "predicted": pred_label, "errors": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("errors", ascending=False)


def _margin_bucket_table(y_true: np.ndarray, predicted: np.ndarray, margins: np.ndarray) -> pd.DataFrame:
    """Summarize error rate by champion confidence-margin bucket."""
    data = pd.DataFrame({"margin": margins, "correct": predicted == y_true})
    data["margin_bucket"] = pd.qcut(data["margin"], q=10, duplicates="drop")
    grouped = data.groupby("margin_bucket", observed=True)["correct"].agg(["count", "mean"]).reset_index()
    grouped["error_rate"] = 1.0 - grouped["mean"]
    return grouped.drop(columns=["mean"]).rename(columns={"margin_bucket": "bucket", "count": "rows"})


def _feature_quantile_table(context: pd.DataFrame, champion_correct: np.ndarray) -> pd.DataFrame:
    """Summarize champion error rate by context numeric quantile."""
    rows = []
    for column in CONTEXT_NUMERIC_FEATURES:
        buckets = pd.qcut(context[column], q=5, duplicates="drop")
        frame = pd.DataFrame({"bucket": buckets, "correct": champion_correct})
        grouped = frame.groupby("bucket", observed=True)["correct"].agg(["count", "mean"]).reset_index()
        grouped["feature"] = column
        grouped["error_rate"] = 1.0 - grouped["mean"]
        rows.append(grouped[["feature", "bucket", "count", "error_rate"]])
    return pd.concat(rows, ignore_index=True)


def _categorical_table(context: pd.DataFrame, champion_correct: np.ndarray) -> pd.DataFrame:
    """Summarize champion error rate by context categorical group."""
    rows = []
    for column in CONTEXT_CATEGORICAL_FEATURES:
        frame = pd.DataFrame({"value": context[column].astype(str), "correct": champion_correct})
        grouped = frame.groupby("value", observed=True)["correct"].agg(["count", "mean"]).reset_index()
        grouped["feature"] = column
        grouped["error_rate"] = 1.0 - grouped["mean"]
        rows.append(grouped[["feature", "value", "count", "error_rate"]])
    return pd.concat(rows, ignore_index=True).sort_values(["feature", "error_rate"], ascending=[True, False])


def _base_rescue_table(champion: pd.DataFrame, base_frames: list[pd.DataFrame], base_names: list[str]) -> pd.DataFrame:
    """Summarize which base models rescue champion mistakes."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    y_true = champion["y_true"].to_numpy()
    champion_pred = labels[champion[config.PROBA_COLUMNS].to_numpy().argmax(axis=1)]
    champion_correct = champion_pred == y_true
    rows = []
    for name, frame in zip(base_names, base_frames):
        pred = labels[frame[config.PROBA_COLUMNS].to_numpy().argmax(axis=1)]
        correct = pred == y_true
        rescues = (~champion_correct) & correct
        new_errors = champion_correct & (~correct)
        rows.append(
            {
                "base_name": name,
                "base_score": balanced_accuracy_score(y_true, pred),
                "rescue_rows": int(rescues.sum()),
                "new_error_rows": int(new_errors.sum()),
                "rescue_rate": float(rescues.mean()),
                "new_error_rate": float(new_errors.mean()),
                "galaxy_rescues": int((rescues & (y_true == "GALAXY")).sum()),
                "qso_rescues": int((rescues & (y_true == "QSO")).sum()),
                "star_rescues": int((rescues & (y_true == "STAR")).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["rescue_rows", "base_score"], ascending=False)


def _load_context(reference: pd.DataFrame) -> pd.DataFrame:
    """Load feature context aligned to a reference OOF artifact."""
    train_raw, _, _ = load_raw_data()
    train_features = make_features(train_raw).set_index(config.ID_COLUMN)
    columns = CONTEXT_NUMERIC_FEATURES + CONTEXT_CATEGORICAL_FEATURES
    return train_features.loc[reference["id"].to_numpy(), columns].reset_index(drop=True)


def _validate_base_alignment(reference: pd.DataFrame, frames: list[pd.DataFrame]) -> None:
    """Validate all audited OOF artifacts align to the champion."""
    for frame in frames:
        for column in ["id", "fold", "y_true"]:
            if not reference[column].equals(frame[column]):
                raise ValueError(f"Base OOF artifact is not aligned on `{column}`.")


def _dataframe_to_markdown(data: pd.DataFrame) -> str:
    """Format a dataframe as markdown without optional dependencies."""
    if data.empty:
        return "_No rows._"
    formatted = data.copy()
    for column in formatted.select_dtypes(include=["float"]).columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.8f}")
    header = "| " + " | ".join(str(column) for column in formatted.columns) + " |"
    separator = "|" + "|".join("---" for _ in formatted.columns) + "|"
    rows = [header, separator]
    for _, row in formatted.iterrows():
        rows.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
