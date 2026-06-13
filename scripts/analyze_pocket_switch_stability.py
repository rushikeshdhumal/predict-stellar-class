"""Audit fold stability for deployable pocket-switch rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from src import config
from src.data import load_raw_data, make_features
from src.ensemble import load_probability_artifact
from src.utils import ensure_output_dirs


DEFAULT_CHAMPION_OOF = config.STACKING_DIR / "meta_rich_lgbm_active_rescue_stronger_reg_oof.csv"
DEFAULT_CANDIDATE_PATHS = {
    "more_trees": config.STACKING_DIR / "meta_rich_lgbm_active_rescue_more_trees_oof.csv",
    "no_context": config.STACKING_DIR / "meta_rich_lgbm_active_rescue_no_context_oof.csv",
    "conservative": config.STACKING_DIR / "meta_rich_lgbm_active_rescue_conservative_oof.csv",
}


@dataclass(frozen=True)
class PocketRule:
    """One deployable pocket switch rule."""

    name: str
    candidate_name: str
    mask_fn: Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class SwitchPolicy:
    """One deployable switch policy."""

    name: str
    rules: list[PocketRule]


def main() -> None:
    """Write fold-stability diagnostics for the current pocket-switch rules."""
    ensure_output_dirs()
    champion = load_probability_artifact(DEFAULT_CHAMPION_OOF)
    candidates = {name: load_probability_artifact(path) for name, path in DEFAULT_CANDIDATE_PATHS.items()}
    _validate_alignment(champion, candidates)
    context = _build_context(champion)
    policies = _build_policies()
    summary, fold_details = _audit_policies(champion, candidates, context, policies)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = config.PROCESSED_DATA_DIR / f"pocket_switch_stability_summary_{timestamp}.csv"
    fold_path = config.PROCESSED_DATA_DIR / f"pocket_switch_stability_by_fold_{timestamp}.csv"
    report_path = config.PROCESSED_DATA_DIR / f"pocket_switch_stability_report_{timestamp}.md"
    summary.to_csv(summary_path, index=False)
    fold_details.to_csv(fold_path, index=False)
    report_path.write_text(_build_report(summary, fold_details, summary_path, fold_path), encoding="utf-8")
    print(f"Pocket-switch stability summary written to: {summary_path}")
    print(f"Pocket-switch stability by fold written to: {fold_path}")
    print(f"Pocket-switch stability report written to: {report_path}")


def _audit_policies(
    champion: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    context: pd.DataFrame,
    policies: list[SwitchPolicy],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute overall and fold-level stability diagnostics."""
    y_true = champion["y_true"].to_numpy()
    folds = champion["fold"].to_numpy()
    champion_proba = champion[config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
    champion_pred = _labels_from_proba(champion_proba)
    champion_score = balanced_accuracy_score(y_true, champion_pred)
    champion_recalls = recall_score(y_true, champion_pred, labels=config.CLASS_LABELS, average=None)
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for policy in policies:
        switched = np.zeros(len(champion), dtype=bool)
        proba = champion_proba.copy()
        for rule in policy.rules:
            mask = rule.mask_fn(context)
            proba[mask] = candidates[rule.candidate_name].loc[mask, config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
            switched |= mask
        pred = _labels_from_proba(proba)
        score = balanced_accuracy_score(y_true, pred)
        recalls = recall_score(y_true, pred, labels=config.CLASS_LABELS, average=None)
        fold_deltas = []
        for fold in range(config.N_FOLDS):
            fold_mask = folds == fold
            baseline_fold_score = balanced_accuracy_score(y_true[fold_mask], champion_pred[fold_mask])
            policy_fold_score = balanced_accuracy_score(y_true[fold_mask], pred[fold_mask])
            fold_delta = float(policy_fold_score - baseline_fold_score)
            fold_deltas.append(fold_delta)
            fold_recalls = recall_score(y_true[fold_mask], pred[fold_mask], labels=config.CLASS_LABELS, average=None)
            fold_baseline_recalls = recall_score(
                y_true[fold_mask],
                champion_pred[fold_mask],
                labels=config.CLASS_LABELS,
                average=None,
            )
            fold_rows.append(
                {
                    "policy_name": policy.name,
                    "fold": fold,
                    "balanced_accuracy_delta": fold_delta,
                    "switched_rows": int((switched & fold_mask).sum()),
                    **{
                        f"{label}_recall_delta": float(fold_recalls[index] - fold_baseline_recalls[index])
                        for index, label in enumerate(config.CLASS_LABELS)
                    },
                }
            )
        summary_rows.append(
            {
                "policy_name": policy.name,
                "balanced_accuracy": float(score),
                "balanced_accuracy_delta": float(score - champion_score),
                "switched_rows": int(switched.sum()),
                "positive_folds": int(sum(delta > 0.0 for delta in fold_deltas)),
                "min_fold_delta": float(min(fold_deltas)),
                "max_fold_delta": float(max(fold_deltas)),
                "mean_fold_delta": float(np.mean(fold_deltas)),
                **{
                    f"{label}_recall_delta": float(recalls[index] - champion_recalls[index])
                    for index, label in enumerate(config.CLASS_LABELS)
                },
            }
        )
    return (
        pd.DataFrame(summary_rows).sort_values(["positive_folds", "balanced_accuracy_delta"], ascending=False),
        pd.DataFrame(fold_rows),
    )


def _build_context(reference: pd.DataFrame) -> pd.DataFrame:
    """Build deployable context aligned to the reference OOF artifact."""
    train_raw, _, _ = load_raw_data()
    train_features = make_features(train_raw).set_index(config.ID_COLUMN)
    context = train_features.loc[reference["id"].to_numpy(), ["g_r", "u_g", "spectral_type"]].reset_index(drop=True)
    context["champion_pred"] = _labels_from_proba(reference[config.PROBA_COLUMNS].to_numpy(dtype=np.float64))
    return context


def _build_policies() -> list[SwitchPolicy]:
    """Build accepted v1 rules and reduced stability candidates."""
    rules = [
        PocketRule(
            "more_galaxy_gr_1741_1964",
            "more_trees",
            lambda data: _between(data, "g_r", 1.741, 1.964) & (data["champion_pred"].to_numpy() == "GALAXY"),
        ),
        PocketRule(
            "noctx_galaxy_gr_053_0781",
            "no_context",
            lambda data: _between(data, "g_r", 0.530, 0.781) & (data["champion_pred"].to_numpy() == "GALAXY"),
        ),
        PocketRule(
            "more_star_gr_0312_053",
            "more_trees",
            lambda data: _between(data, "g_r", 0.312, 0.530) & (data["champion_pred"].to_numpy() == "STAR"),
        ),
        PocketRule(
            "noctx_star_ug_2282_2948",
            "no_context",
            lambda data: _between(data, "u_g", 2.282, 2.948) & (data["champion_pred"].to_numpy() == "STAR"),
        ),
        PocketRule(
            "cons_star_spectral_af",
            "conservative",
            lambda data: (data["spectral_type"].astype(str).to_numpy() == "A/F")
            & (data["champion_pred"].to_numpy() == "STAR"),
        ),
    ]
    return [
        *(SwitchPolicy(rule.name, [rule]) for rule in rules),
        SwitchPolicy("cumulative_galaxy_gr", rules[:2]),
        SwitchPolicy("cumulative_star", rules[2:]),
        SwitchPolicy("cumulative_all", rules),
        SwitchPolicy("stable_drop_noctx_star_ug", [rules[0], rules[1], rules[2], rules[4]]),
        SwitchPolicy("stable_top3", [rules[0], rules[1], rules[2]]),
        SwitchPolicy("stable_star_gr_and_af", [rules[2], rules[4]]),
    ]


def _build_report(
    summary: pd.DataFrame,
    fold_details: pd.DataFrame,
    summary_path: Path,
    fold_path: Path,
) -> str:
    """Build a markdown stability report."""
    return "\n".join(
        [
            "# Pocket Switch Stability Audit",
            "",
            f"- Summary CSV: `{summary_path}`",
            f"- Fold CSV: `{fold_path}`",
            "- Baseline is `meta_rich_lgbm_active_rescue_stronger_reg`.",
            "",
            "## Overall Stability",
            "",
            _dataframe_to_markdown(summary),
            "",
            "## Fold Details",
            "",
            _dataframe_to_markdown(fold_details),
            "",
        ]
    )


def _between(data: pd.DataFrame, column: str, left: float, right: float) -> np.ndarray:
    """Return open-left/closed-right interval membership."""
    values = data[column].to_numpy(dtype=np.float64)
    return (values > left) & (values <= right)


def _validate_alignment(reference: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> None:
    """Validate candidate OOF artifacts are aligned."""
    for name, frame in candidates.items():
        for column in ["id", "fold", "y_true"]:
            if not reference[column].equals(frame[column]):
                raise ValueError(f"Candidate `{name}` is not aligned on `{column}`.")


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert probabilities to configured labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


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
