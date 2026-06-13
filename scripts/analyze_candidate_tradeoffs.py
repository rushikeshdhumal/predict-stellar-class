"""Analyze tradeoffs between the current champion and near-miss candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from src.meta_features import CONTEXT_CATEGORICAL_FEATURES, CONTEXT_NUMERIC_FEATURES
from src.utils import ensure_output_dirs


DEFAULT_CHAMPION_OOF = config.STACKING_DIR / "meta_rich_lgbm_active_rescue_stronger_reg_oof.csv"
DEFAULT_CANDIDATE_PATHS = [
    config.STACKING_DIR / "meta_rich_lgbm_active_rescue_no_context_oof.csv",
    config.STACKING_DIR / "meta_soft_residual_reliability_champion_gq_star_oof.csv",
    config.STACKING_DIR / "meta_rich_lgbm_active_rescue_more_trees_oof.csv",
    config.STACKING_DIR / "meta_rich_lgbm_active_rescue_conservative_oof.csv",
]
DEFAULT_CANDIDATE_NAMES = [
    "rich_lgbm_no_context",
    "soft_residual_reliability",
    "rich_lgbm_more_trees",
    "rich_lgbm_conservative",
]


@dataclass(frozen=True)
class CandidateAudit:
    """Aligned prediction audit for one candidate.

    Attributes:
        name: Human-readable candidate name.
        frame: Candidate OOF probability artifact.
        predicted: Candidate predicted labels.
        correct: Whether the candidate is correct per row.
        score: Candidate OOF balanced accuracy.
    """

    name: str
    frame: pd.DataFrame
    predicted: np.ndarray
    correct: np.ndarray
    score: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-oof-path", type=Path, default=DEFAULT_CHAMPION_OOF)
    parser.add_argument("--candidate-oof-paths", nargs="+", type=Path, default=DEFAULT_CANDIDATE_PATHS)
    parser.add_argument("--candidate-names", nargs="+", default=DEFAULT_CANDIDATE_NAMES)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Markdown report path. Defaults to data/processed with a timestamp.",
    )
    return parser.parse_args()


def main() -> None:
    """Create a champion-versus-candidate tradeoff report."""
    args = parse_args()
    ensure_output_dirs()
    if len(args.candidate_oof_paths) != len(args.candidate_names):
        raise ValueError("Candidate path and name counts must match.")
    champion = load_probability_artifact(args.champion_oof_path)
    candidate_frames = [load_probability_artifact(path) for path in args.candidate_oof_paths]
    _validate_alignment(champion, candidate_frames)

    y_true = champion["y_true"].to_numpy()
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    champion_proba = champion[config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
    champion_pred = labels[champion_proba.argmax(axis=1)]
    champion_correct = champion_pred == y_true
    champion_score = balanced_accuracy_score(y_true, champion_pred)
    candidates = [
        _build_candidate_audit(name, frame, y_true, labels)
        for name, frame in zip(args.candidate_names, candidate_frames)
    ]
    context = _build_context(champion, champion_proba, champion_pred)
    report = _build_report(champion_score, y_true, champion_pred, champion_correct, context, candidates)
    output_path = args.output_path or (
        config.PROCESSED_DATA_DIR / f"candidate_tradeoff_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Candidate tradeoff report written to: {output_path}")


def _build_candidate_audit(
    name: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    labels: np.ndarray,
) -> CandidateAudit:
    """Build one candidate audit object."""
    predicted = labels[frame[config.PROBA_COLUMNS].to_numpy(dtype=np.float64).argmax(axis=1)]
    correct = predicted == y_true
    return CandidateAudit(
        name=name,
        frame=frame,
        predicted=predicted,
        correct=correct,
        score=float(balanced_accuracy_score(y_true, predicted)),
    )


def _build_report(
    champion_score: float,
    y_true: np.ndarray,
    champion_pred: np.ndarray,
    champion_correct: np.ndarray,
    context: pd.DataFrame,
    candidates: list[CandidateAudit],
) -> str:
    """Build the markdown report."""
    summary = _candidate_summary(champion_score, y_true, champion_correct, candidates)
    rescue_pairs = pd.concat(
        [_pair_tradeoff_table(y_true, champion_pred, champion_correct, candidate) for candidate in candidates],
        ignore_index=True,
    )
    deployable_pockets = pd.concat(
        [_deployable_pocket_table(y_true, champion_correct, context, candidate) for candidate in candidates],
        ignore_index=True,
    ).sort_values(["balanced_accuracy_delta", "net_rows"], ascending=False)
    error_pockets = pd.concat(
        [_champion_error_pocket_table(y_true, champion_pred, champion_correct, context, candidate) for candidate in candidates],
        ignore_index=True,
    ).sort_values(["rescues", "rescue_rate"], ascending=False)
    lines = [
        "# Candidate Tradeoff Diagnostics",
        "",
        f"- Champion OOF balanced accuracy: `{champion_score:.8f}`",
        f"- Champion OOF errors: `{int((~champion_correct).sum())}`",
        "- Positive `balanced_accuracy_delta` means replacing the champion with that candidate only inside the listed pocket would improve OOF balanced accuracy, using full-OOF class denominators.",
        "- Tables using `y_true` are diagnostic only; deployable pockets use champion prediction, champion confidence, and original non-ID features.",
        "",
        "## Candidate Summary",
        "",
        _dataframe_to_markdown(summary),
        "",
        "## Rescues And Breaks By True Class",
        "",
        _dataframe_to_markdown(rescue_pairs),
        "",
        "## Top Deployable Candidate Pockets",
        "",
        _dataframe_to_markdown(deployable_pockets.head(40)),
        "",
        "## Champion Error Rescue Pockets",
        "",
        _dataframe_to_markdown(error_pockets.head(40)),
        "",
    ]
    return "\n".join(lines)


def _candidate_summary(
    champion_score: float,
    y_true: np.ndarray,
    champion_correct: np.ndarray,
    candidates: list[CandidateAudit],
) -> pd.DataFrame:
    """Summarize whole-OOF candidate tradeoffs."""
    rows = []
    class_counts = {label: int((y_true == label).sum()) for label in config.CLASS_LABELS}
    for candidate in candidates:
        rescues = (~champion_correct) & candidate.correct
        breaks = champion_correct & (~candidate.correct)
        row: dict[str, object] = {
            "candidate": candidate.name,
            "candidate_score": candidate.score,
            "score_delta": candidate.score - champion_score,
            "rescues": int(rescues.sum()),
            "breaks": int(breaks.sum()),
            "net_rows": int(rescues.sum() - breaks.sum()),
        }
        for label in config.CLASS_LABELS:
            class_mask = y_true == label
            row[f"{label}_rescue"] = int((rescues & class_mask).sum())
            row[f"{label}_break"] = int((breaks & class_mask).sum())
            row[f"{label}_recall_delta"] = (
                int((rescues & class_mask).sum()) - int((breaks & class_mask).sum())
            ) / class_counts[label]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score_delta", ascending=False)


def _pair_tradeoff_table(
    y_true: np.ndarray,
    champion_pred: np.ndarray,
    champion_correct: np.ndarray,
    candidate: CandidateAudit,
) -> pd.DataFrame:
    """Summarize candidate rescue and break counts by true class."""
    rows = []
    rescues = (~champion_correct) & candidate.correct
    breaks = champion_correct & (~candidate.correct)
    for label in config.CLASS_LABELS:
        class_mask = y_true == label
        rows.append(
            {
                "candidate": candidate.name,
                "true_class": label,
                "rescues": int((rescues & class_mask).sum()),
                "breaks": int((breaks & class_mask).sum()),
                "net": int((rescues & class_mask).sum() - (breaks & class_mask).sum()),
            }
        )
    for true_label in config.CLASS_LABELS:
        for pred_label in config.CLASS_LABELS:
            if true_label == pred_label:
                continue
            mask = (y_true == true_label) & (champion_pred == pred_label)
            rows.append(
                {
                    "candidate": candidate.name,
                    "true_class": f"{true_label}_as_{pred_label}",
                    "rescues": int((rescues & mask).sum()),
                    "breaks": 0,
                    "net": int((rescues & mask).sum()),
                }
            )
    return pd.DataFrame(rows)


def _deployable_pocket_table(
    y_true: np.ndarray,
    champion_correct: np.ndarray,
    context: pd.DataFrame,
    candidate: CandidateAudit,
) -> pd.DataFrame:
    """Find deployable pockets where a candidate has positive local tradeoff."""
    class_counts = {label: int((y_true == label).sum()) for label in config.CLASS_LABELS}
    group_sets = [
        ["champion_pred", "champion_margin_bucket"],
        ["champion_pred", "champion_entropy_bucket"],
        ["champion_pred", "spectral_type"],
        ["champion_pred", "galaxy_population"],
        ["champion_pred", "spectral_population"],
        ["champion_pred", "redshift_bucket"],
        ["champion_pred", "g_r_bucket"],
        ["champion_pred", "u_g_bucket"],
    ]
    rows = []
    for group_columns in group_sets:
        rows.extend(_group_tradeoffs(y_true, champion_correct, context, candidate, group_columns, class_counts))
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table[(table["rows"] >= 1000) & (table["balanced_accuracy_delta"] > 0.0)]
    return table.sort_values(["balanced_accuracy_delta", "net_rows"], ascending=False)


def _champion_error_pocket_table(
    y_true: np.ndarray,
    champion_pred: np.ndarray,
    champion_correct: np.ndarray,
    context: pd.DataFrame,
    candidate: CandidateAudit,
) -> pd.DataFrame:
    """Summarize rescued champion errors by diagnostic error pocket."""
    error_mask = ~champion_correct
    data = context.loc[error_mask].copy()
    data["true_class"] = y_true[error_mask]
    data["champion_error_pair"] = [
        f"{true_label}_as_{pred_label}" for true_label, pred_label in zip(y_true[error_mask], champion_pred[error_mask])
    ]
    rescued = candidate.correct[error_mask]
    rows = []
    for group_columns in [
        ["champion_error_pair", "champion_margin_bucket"],
        ["champion_error_pair", "redshift_bucket"],
        ["champion_error_pair", "spectral_population"],
        ["true_class", "spectral_population"],
    ]:
        grouped = data.assign(rescued=rescued).groupby(group_columns, observed=True)["rescued"].agg(["count", "sum"])
        for key, row in grouped.reset_index().iterrows():
            del key
            if int(row["count"]) < 100:
                continue
            rows.append(
                {
                    "candidate": candidate.name,
                    "group": " + ".join(group_columns),
                    "pocket": " | ".join(str(row[column]) for column in group_columns),
                    "champion_errors": int(row["count"]),
                    "rescues": int(row["sum"]),
                    "rescue_rate": float(row["sum"] / row["count"]),
                }
            )
    return pd.DataFrame(rows)


def _group_tradeoffs(
    y_true: np.ndarray,
    champion_correct: np.ndarray,
    context: pd.DataFrame,
    candidate: CandidateAudit,
    group_columns: list[str],
    class_counts: dict[str, int],
) -> list[dict[str, object]]:
    """Compute local replacement tradeoffs for one grouping."""
    data = context[group_columns].copy()
    data["_row"] = np.arange(len(data))
    grouped = data.groupby(group_columns, observed=True)["_row"].apply(list).reset_index()
    rows: list[dict[str, object]] = []
    for _, group in grouped.iterrows():
        indices = np.asarray(group["_row"], dtype=int)
        rescues = (~champion_correct[indices]) & candidate.correct[indices]
        breaks = champion_correct[indices] & (~candidate.correct[indices])
        if not rescues.any() and not breaks.any():
            continue
        delta = _balanced_accuracy_delta(y_true[indices], rescues, breaks, class_counts)
        rows.append(
            {
                "candidate": candidate.name,
                "group": " + ".join(group_columns),
                "pocket": " | ".join(str(group[column]) for column in group_columns),
                "rows": int(len(indices)),
                "rescues": int(rescues.sum()),
                "breaks": int(breaks.sum()),
                "net_rows": int(rescues.sum() - breaks.sum()),
                "balanced_accuracy_delta": delta,
            }
        )
    return rows


def _balanced_accuracy_delta(
    y_true: np.ndarray,
    rescues: np.ndarray,
    breaks: np.ndarray,
    class_counts: dict[str, int],
) -> float:
    """Compute balanced-accuracy delta if a local pocket swaps to the candidate."""
    deltas = []
    for label in config.CLASS_LABELS:
        class_mask = y_true == label
        class_count = class_counts[label]
        if class_count == 0:
            continue
        deltas.append((int((rescues & class_mask).sum()) - int((breaks & class_mask).sum())) / class_count)
    return float(np.mean(deltas)) if deltas else 0.0


def _build_context(champion: pd.DataFrame, champion_proba: np.ndarray, champion_pred: np.ndarray) -> pd.DataFrame:
    """Build deployable context features aligned to the champion OOF artifact."""
    train_raw, _, _ = load_raw_data()
    train_features = make_features(train_raw).set_index(config.ID_COLUMN)
    columns = CONTEXT_NUMERIC_FEATURES + CONTEXT_CATEGORICAL_FEATURES
    context = train_features.loc[champion["id"].to_numpy(), columns].reset_index(drop=True)
    sorted_proba = np.sort(champion_proba, axis=1)
    clipped = np.clip(champion_proba, 1e-8, 1.0)
    context["champion_pred"] = champion_pred
    context["champion_margin"] = sorted_proba[:, -1] - sorted_proba[:, -2]
    context["champion_entropy"] = -(clipped * np.log(clipped)).sum(axis=1)
    context["champion_margin_bucket"] = pd.qcut(context["champion_margin"], q=10, duplicates="drop")
    context["champion_entropy_bucket"] = pd.qcut(context["champion_entropy"], q=10, duplicates="drop")
    for column in ["redshift", "g_r", "u_g"]:
        context[f"{column}_bucket"] = pd.qcut(context[column], q=10, duplicates="drop")
    return context


def _validate_alignment(reference: pd.DataFrame, frames: list[pd.DataFrame]) -> None:
    """Validate all candidate OOF artifacts align to the champion."""
    for frame in frames:
        for column in ["id", "fold", "y_true"]:
            if not reference[column].equals(frame[column]):
                raise ValueError(f"Candidate OOF artifact is not aligned on `{column}`.")


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
