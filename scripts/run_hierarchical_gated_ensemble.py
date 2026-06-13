"""Evaluate a hard-gated STAR/rest then GALAXY/QSO hierarchical ensemble."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, recall_score

from src import config
from src.ensemble import load_probability_artifact
from src.utils import (
    append_decision_entry,
    append_experiment_log,
    ensure_output_dirs,
    get_best_score,
    update_best_score,
    update_best_submission,
)


DEFAULT_CHAMPION_OOF = config.STACKING_DIR / "meta_rich_lgbm_active_rescue_stronger_reg_oof.csv"
DEFAULT_CHAMPION_TEST = config.STACKING_DIR / "meta_rich_lgbm_active_rescue_stronger_reg_test_proba.csv"
DEFAULT_GQ_OOF = config.OOF_DIR / "binary_specialist_galaxy_qso_lgbm_oof.csv"
DEFAULT_GQ_TEST = config.TEST_PROBA_DIR / "binary_specialist_galaxy_qso_lgbm_test_proba.csv"
DEFAULT_STAR_OVR_OOF = config.ENSEMBLE_DIR / "scalar_features" / "one_vs_rest_star_lgbm_oof_features.csv"
DEFAULT_STAR_OVR_TEST = config.ENSEMBLE_DIR / "scalar_features" / "one_vs_rest_star_lgbm_test_features.csv"


@dataclass(frozen=True)
class GateCandidate:
    """One threshold candidate result."""

    star_threshold: float
    gq_threshold: float
    balanced_accuracy: float
    log_loss: float
    oof_proba: np.ndarray
    test_proba: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-oof-path", type=Path, default=DEFAULT_CHAMPION_OOF)
    parser.add_argument("--champion-test-proba-path", type=Path, default=DEFAULT_CHAMPION_TEST)
    parser.add_argument("--star-ovr-oof-path", type=Path, default=DEFAULT_STAR_OVR_OOF)
    parser.add_argument("--star-ovr-test-path", type=Path, default=DEFAULT_STAR_OVR_TEST)
    parser.add_argument("--gq-oof-path", type=Path, default=DEFAULT_GQ_OOF)
    parser.add_argument("--gq-test-proba-path", type=Path, default=DEFAULT_GQ_TEST)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--star-thresholds", nargs="+", type=float, default=[0.55, 0.65, 0.75])
    parser.add_argument("--gq-thresholds", nargs="+", type=float, default=[0.55, 0.65, 0.75])
    return parser.parse_args()


def main() -> None:
    """Run the hierarchical gated ensemble experiment."""
    args = parse_args()
    ensure_output_dirs()
    previous_best = get_best_score()
    champion_oof = load_probability_artifact(args.champion_oof_path)
    champion_test = load_probability_artifact(args.champion_test_proba_path)
    gq_oof = load_probability_artifact(args.gq_oof_path)
    gq_test = load_probability_artifact(args.gq_test_proba_path)
    star_oof = pd.read_csv(args.star_ovr_oof_path)
    star_test = pd.read_csv(args.star_ovr_test_path)
    _validate_alignment(champion_oof, champion_test, gq_oof, gq_test, star_oof, star_test)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"hierarchical_gate_{args.output_name}"):
        mlflow.log_param("champion_oof_path", str(args.champion_oof_path))
        mlflow.log_param("champion_test_proba_path", str(args.champion_test_proba_path))
        mlflow.log_param("star_ovr_oof_path", str(args.star_ovr_oof_path))
        mlflow.log_param("star_ovr_test_path", str(args.star_ovr_test_path))
        mlflow.log_param("gq_oof_path", str(args.gq_oof_path))
        mlflow.log_param("gq_test_proba_path", str(args.gq_test_proba_path))
        mlflow.log_param("star_thresholds", ",".join(str(value) for value in args.star_thresholds))
        mlflow.log_param("gq_thresholds", ",".join(str(value) for value in args.gq_thresholds))
        mlflow.log_param("current_best_score", previous_best)
        candidates = [
            _evaluate_candidate(
                champion_oof,
                champion_test,
                star_oof,
                star_test,
                gq_oof,
                gq_test,
                star_threshold=star_threshold,
                gq_threshold=gq_threshold,
            )
            for star_threshold in args.star_thresholds
            for gq_threshold in args.gq_thresholds
        ]
        best = max(candidates, key=lambda candidate: candidate.balanced_accuracy)
        result = _save_result(args.output_name, champion_oof, champion_test, candidates, best, previous_best)
        for candidate in candidates:
            suffix = f"star_{candidate.star_threshold:g}_gq_{candidate.gq_threshold:g}".replace(".", "p")
            mlflow.log_metric(f"balanced_accuracy_{suffix}", candidate.balanced_accuracy)
            mlflow.log_metric(f"log_loss_{suffix}", candidate.log_loss)
        mlflow.log_metric("best_balanced_accuracy", best.balanced_accuracy)
        mlflow.log_metric("best_log_loss", best.log_loss)
        mlflow.log_param("best_star_threshold", best.star_threshold)
        mlflow.log_param("best_gq_threshold", best.gq_threshold)
        mlflow.log_param("improved", result["improved"])
        mlflow.log_artifact(str(result["summary_path"]))
        mlflow.log_artifact(str(result["metrics_path"]))
        mlflow.log_artifact(str(result["oof_path"]))
        mlflow.log_artifact(str(result["test_proba_path"]))
        if result["submission_path"] is not None:
            mlflow.log_artifact(str(result["submission_path"]))

    print(f"Best STAR threshold: {best.star_threshold:g}")
    print(f"Best GALAXY/QSO threshold: {best.gq_threshold:g}")
    print(f"Best balanced accuracy: {best.balanced_accuracy:.8f}")
    print(f"Best log loss: {best.log_loss:.8f}")
    print(f"Previous champion threshold: {previous_best:.8f}")
    print(f"Decision: {'ACCEPTED' if result['improved'] else 'rejected'}")
    print(f"Submission artifact: {result['submission_path']}")


def _evaluate_candidate(
    champion_oof: pd.DataFrame,
    champion_test: pd.DataFrame,
    star_oof: pd.DataFrame,
    star_test: pd.DataFrame,
    gq_oof: pd.DataFrame,
    gq_test: pd.DataFrame,
    *,
    star_threshold: float,
    gq_threshold: float,
) -> GateCandidate:
    """Evaluate one threshold pair."""
    y_true = champion_oof["y_true"].to_numpy()
    oof_proba = _apply_policy(champion_oof, star_oof, gq_oof, star_threshold=star_threshold, gq_threshold=gq_threshold)
    test_proba = _apply_policy(champion_test, star_test, gq_test, star_threshold=star_threshold, gq_threshold=gq_threshold)
    pred = _labels_from_proba(oof_proba)
    return GateCandidate(
        star_threshold=star_threshold,
        gq_threshold=gq_threshold,
        balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
        log_loss=float(log_loss(y_true, oof_proba, labels=config.CLASS_LABELS)),
        oof_proba=oof_proba,
        test_proba=test_proba,
    )


def _apply_policy(
    champion: pd.DataFrame,
    star_features: pd.DataFrame,
    gq: pd.DataFrame,
    *,
    star_threshold: float,
    gq_threshold: float,
) -> np.ndarray:
    """Apply STAR gate, then confident GALAXY/QSO gate, else champion fallback."""
    champion_proba = champion[config.PROBA_COLUMNS].to_numpy(dtype=np.float64)
    output = champion_proba.copy()
    star_prob = star_features["ovr_star__prob"].to_numpy(dtype=np.float64)
    g_proba = gq["proba_GALAXY"].to_numpy(dtype=np.float64)
    q_proba = gq["proba_QSO"].to_numpy(dtype=np.float64)
    gq_sum = np.clip(g_proba + q_proba, 1e-8, None)
    g_pair = g_proba / gq_sum
    q_pair = q_proba / gq_sum
    gq_conf = np.maximum(g_pair, q_pair)
    champion_star = champion_proba[:, config.CLASS_LABELS.index("STAR")]

    pred_indices = output.argmax(axis=1)
    star_index = config.CLASS_LABELS.index("STAR")
    galaxy_index = config.CLASS_LABELS.index("GALAXY")
    qso_index = config.CLASS_LABELS.index("QSO")
    star_gate = star_prob >= star_threshold
    gq_gate = (~star_gate) & (gq_conf >= gq_threshold) & (champion_star < star_threshold)
    pred_indices[star_gate] = star_index
    pred_indices[gq_gate] = np.where(g_pair[gq_gate] >= q_pair[gq_gate], galaxy_index, qso_index)

    gated = np.full_like(output, 1e-6)
    gated[np.arange(len(gated)), pred_indices] = 1.0 - (len(config.CLASS_LABELS) - 1) * 1e-6
    return gated


def _save_result(
    output_name: str,
    reference_oof: pd.DataFrame,
    reference_test: pd.DataFrame,
    candidates: list[GateCandidate],
    best: GateCandidate,
    previous_best: float,
) -> dict[str, Path | bool | None]:
    """Save artifacts and update logs for the best candidate."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    oof_path = config.STACKING_DIR / f"{output_name}_oof.csv"
    test_proba_path = config.STACKING_DIR / f"{output_name}_test_proba.csv"
    summary_path = config.STACKING_DIR / f"{output_name}_summary_{timestamp}.csv"
    metrics_path = config.STACKING_DIR / f"{output_name}_metrics_{timestamp}.csv"
    _write_oof(reference_oof, best.oof_proba, oof_path)
    _write_test(reference_test, best.test_proba, test_proba_path)
    _write_summary(candidates, summary_path)
    _write_metrics(reference_oof["y_true"].to_numpy(), best.oof_proba, metrics_path)
    improved = best.balanced_accuracy > previous_best
    submission_path = _write_submission(output_name, reference_test, best.test_proba) if improved else None
    change = (
        "run hierarchical gated ensemble with STAR-vs-rest gate and GALAXY-vs-QSO specialist; "
        f"star_threshold={best.star_threshold:g}; gq_threshold={best.gq_threshold:g}"
    )
    append_experiment_log(change, best.balanced_accuracy, submission_path)
    append_decision_entry(change, best.balanced_accuracy, previous_best, improved)
    if improved and submission_path is not None:
        update_best_score(best.balanced_accuracy)
        update_best_submission(submission_path)
    return {
        "oof_path": oof_path,
        "test_proba_path": test_proba_path,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "submission_path": submission_path,
        "improved": improved,
    }


def _validate_alignment(
    champion_oof: pd.DataFrame,
    champion_test: pd.DataFrame,
    gq_oof: pd.DataFrame,
    gq_test: pd.DataFrame,
    star_oof: pd.DataFrame,
    star_test: pd.DataFrame,
) -> None:
    """Validate all artifacts are row-aligned."""
    for frame in [gq_oof, star_oof]:
        for column in ["id", "fold", "y_true"]:
            if not champion_oof[column].equals(frame[column]):
                raise ValueError(f"OOF artifact is not aligned on `{column}`.")
    for frame in [gq_test, star_test]:
        if not champion_test["id"].equals(frame["id"]):
            raise ValueError("Test artifact is not aligned on `id`.")


def _write_summary(candidates: list[GateCandidate], path: Path) -> None:
    """Write threshold-grid summary."""
    pd.DataFrame(
        {
            "star_threshold": [candidate.star_threshold for candidate in candidates],
            "gq_threshold": [candidate.gq_threshold for candidate in candidates],
            "balanced_accuracy": [candidate.balanced_accuracy for candidate in candidates],
            "log_loss": [candidate.log_loss for candidate in candidates],
        }
    ).to_csv(path, index=False)


def _write_metrics(y_true: np.ndarray, proba: np.ndarray, path: Path) -> None:
    """Write recall and confusion metrics for the selected policy."""
    pred = _labels_from_proba(proba)
    confusion = confusion_matrix(y_true, pred, labels=config.CLASS_LABELS)
    recalls = recall_score(y_true, pred, labels=config.CLASS_LABELS, average=None)
    rows = [
        {"metric": "balanced_accuracy", "label": "ALL", "value": balanced_accuracy_score(y_true, pred)},
        {"metric": "log_loss", "label": "ALL", "value": log_loss(y_true, proba, labels=config.CLASS_LABELS)},
    ]
    rows.extend({"metric": "recall", "label": label, "value": float(value)} for label, value in zip(config.CLASS_LABELS, recalls))
    for true_index, true_label in enumerate(config.CLASS_LABELS):
        for pred_index, pred_label in enumerate(config.CLASS_LABELS):
            rows.append(
                {
                    "metric": "confusion_count",
                    "label": f"true_{true_label}__pred_{pred_label}",
                    "value": int(confusion[true_index, pred_index]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_oof(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write OOF probability artifact."""
    output = reference[["id", "fold", "y_true"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_test(reference: pd.DataFrame, proba: np.ndarray, path: Path) -> None:
    """Write test probability artifact."""
    output = reference[["id"]].copy()
    for index, column in enumerate(config.PROBA_COLUMNS):
        output[column] = proba[:, index]
    output.to_csv(path, index=False)


def _write_submission(output_name: str, reference_test: pd.DataFrame, proba: np.ndarray) -> Path:
    """Write a competition submission."""
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    if not sample[config.ID_COLUMN].equals(reference_test["id"]):
        raise ValueError("Sample submission ids are not aligned with test probabilities.")
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    submission = sample.copy()
    submission[config.TARGET_COLUMN] = labels[proba.argmax(axis=1)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.SUBMISSIONS_DIR / f"submission_{output_name}_{timestamp}.csv"
    submission.to_csv(path, index=False)
    return path


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert probabilities to labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


if __name__ == "__main__":
    main()
