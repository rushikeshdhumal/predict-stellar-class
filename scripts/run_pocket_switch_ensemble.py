"""Evaluate tiny deployable pocket switches from champion to near-miss candidates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, recall_score

from src import config
from src.data import load_raw_data, make_features
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
DEFAULT_CANDIDATE_PATHS = {
    "more_trees": (
        config.STACKING_DIR / "meta_rich_lgbm_active_rescue_more_trees_oof.csv",
        config.STACKING_DIR / "meta_rich_lgbm_active_rescue_more_trees_test_proba.csv",
    ),
    "no_context": (
        config.STACKING_DIR / "meta_rich_lgbm_active_rescue_no_context_oof.csv",
        config.STACKING_DIR / "meta_rich_lgbm_active_rescue_no_context_test_proba.csv",
    ),
    "conservative": (
        config.STACKING_DIR / "meta_rich_lgbm_active_rescue_conservative_oof.csv",
        config.STACKING_DIR / "meta_rich_lgbm_active_rescue_conservative_test_proba.csv",
    ),
}


@dataclass(frozen=True)
class PocketRule:
    """One deployable pocket switch rule.

    Attributes:
        name: Stable rule name.
        candidate_name: Candidate probability artifact to use inside the pocket.
        mask_fn: Function producing a boolean mask from deployable context.
    """

    name: str
    candidate_name: str
    mask_fn: Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class SwitchPolicy:
    """One rule policy to evaluate."""

    name: str
    rules: list[PocketRule]


@dataclass(frozen=True)
class SwitchResult:
    """Evaluation result for one switch policy."""

    policy_name: str
    balanced_accuracy: float
    log_loss: float
    switched_rows: int
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
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def main() -> None:
    """Run the tiny pocket-switch experiment."""
    args = parse_args()
    ensure_output_dirs()
    previous_best = get_best_score()
    champion_oof = load_probability_artifact(args.champion_oof_path)
    champion_test = load_probability_artifact(args.champion_test_proba_path)
    candidate_oof, candidate_test = _load_candidates(champion_oof, champion_test)
    train_context, test_context = _build_context(champion_oof, champion_test)
    policies = _build_policies()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"pocket_switch_{args.output_name}"):
        mlflow.log_param("champion_oof_path", str(args.champion_oof_path))
        mlflow.log_param("champion_test_proba_path", str(args.champion_test_proba_path))
        mlflow.log_param("candidate_names", ",".join(candidate_oof.keys()))
        mlflow.log_param("policy_names", ",".join(policy.name for policy in policies))
        mlflow.log_param("current_best_score", previous_best)
        results = [
            _evaluate_policy(policy, champion_oof, champion_test, candidate_oof, candidate_test, train_context, test_context)
            for policy in policies
        ]
        saved = _save_result(args.output_name, champion_oof, champion_test, results, previous_best)
        for result in results:
            suffix = result.policy_name.replace(".", "_")
            mlflow.log_metric(f"balanced_accuracy_{suffix}", result.balanced_accuracy)
            mlflow.log_metric(f"log_loss_{suffix}", result.log_loss)
            mlflow.log_metric(f"switched_rows_{suffix}", result.switched_rows)
        mlflow.log_metric("best_balanced_accuracy", saved["best"].balanced_accuracy)
        mlflow.log_metric("best_log_loss", saved["best"].log_loss)
        mlflow.log_param("best_policy", saved["best"].policy_name)
        mlflow.log_param("improved", saved["improved"])
        mlflow.log_artifact(str(saved["summary_path"]))
        mlflow.log_artifact(str(saved["metrics_path"]))
        mlflow.log_artifact(str(saved["oof_path"]))
        mlflow.log_artifact(str(saved["test_proba_path"]))
        if saved["submission_path"] is not None:
            mlflow.log_artifact(str(saved["submission_path"]))

    best = saved["best"]
    print(f"Best policy: {best.policy_name}")
    print(f"Balanced accuracy: {best.balanced_accuracy:.8f}")
    print(f"Log loss: {best.log_loss:.8f}")
    print(f"Switched OOF rows: {best.switched_rows}")
    print(f"Previous champion threshold: {previous_best:.8f}")
    print(f"Decision: {'ACCEPTED' if saved['improved'] else 'rejected'}")
    print(f"Submission artifact: {saved['submission_path']}")


def _load_candidates(
    champion_oof: pd.DataFrame,
    champion_test: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load and validate candidate artifacts."""
    candidate_oof = {}
    candidate_test = {}
    for name, (oof_path, test_path) in DEFAULT_CANDIDATE_PATHS.items():
        oof = load_probability_artifact(oof_path)
        test = load_probability_artifact(test_path)
        for column in ["id", "fold", "y_true"]:
            if not champion_oof[column].equals(oof[column]):
                raise ValueError(f"Candidate `{name}` OOF is not aligned on `{column}`.")
        if not champion_test["id"].equals(test["id"]):
            raise ValueError(f"Candidate `{name}` test probabilities are not aligned on `id`.")
        candidate_oof[name] = oof
        candidate_test[name] = test
    return candidate_oof, candidate_test


def _build_context(reference_oof: pd.DataFrame, reference_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deployable context for train and test rows."""
    train_raw, test_raw, _ = load_raw_data()
    train_features = make_features(train_raw).set_index(config.ID_COLUMN)
    test_features = make_features(test_raw).set_index(config.ID_COLUMN)
    columns = ["g_r", "u_g", "spectral_type"]
    train_context = train_features.loc[reference_oof["id"].to_numpy(), columns].reset_index(drop=True)
    test_context = test_features.loc[reference_test["id"].to_numpy(), columns].reset_index(drop=True)
    train_context["champion_pred"] = _labels_from_proba(reference_oof[config.PROBA_COLUMNS].to_numpy(dtype=np.float64))
    test_context["champion_pred"] = _labels_from_proba(reference_test[config.PROBA_COLUMNS].to_numpy(dtype=np.float64))
    return train_context, test_context


def _build_policies() -> list[SwitchPolicy]:
    """Build fixed hand-picked policies from the tradeoff diagnostic."""
    rules = [
        PocketRule(
            name="more_galaxy_gr_1741_1964",
            candidate_name="more_trees",
            mask_fn=lambda data: _between(data, "g_r", 1.741, 1.964) & (data["champion_pred"].to_numpy() == "GALAXY"),
        ),
        PocketRule(
            name="noctx_galaxy_gr_053_0781",
            candidate_name="no_context",
            mask_fn=lambda data: _between(data, "g_r", 0.530, 0.781) & (data["champion_pred"].to_numpy() == "GALAXY"),
        ),
        PocketRule(
            name="more_star_gr_0312_053",
            candidate_name="more_trees",
            mask_fn=lambda data: _between(data, "g_r", 0.312, 0.530) & (data["champion_pred"].to_numpy() == "STAR"),
        ),
        PocketRule(
            name="noctx_star_ug_2282_2948",
            candidate_name="no_context",
            mask_fn=lambda data: _between(data, "u_g", 2.282, 2.948) & (data["champion_pred"].to_numpy() == "STAR"),
        ),
        PocketRule(
            name="cons_star_spectral_af",
            candidate_name="conservative",
            mask_fn=lambda data: (data["spectral_type"].astype(str).to_numpy() == "A/F")
            & (data["champion_pred"].to_numpy() == "STAR"),
        ),
    ]
    return [
        *(SwitchPolicy(rule.name, [rule]) for rule in rules),
        SwitchPolicy("cumulative_galaxy_gr", rules[:2]),
        SwitchPolicy("cumulative_star", rules[2:]),
        SwitchPolicy("cumulative_all", rules),
    ]


def _between(data: pd.DataFrame, column: str, left: float, right: float) -> np.ndarray:
    """Return qcut-style open-left/closed-right interval membership."""
    values = data[column].to_numpy(dtype=np.float64)
    return (values > left) & (values <= right)


def _evaluate_policy(
    policy: SwitchPolicy,
    champion_oof: pd.DataFrame,
    champion_test: pd.DataFrame,
    candidate_oof: dict[str, pd.DataFrame],
    candidate_test: dict[str, pd.DataFrame],
    train_context: pd.DataFrame,
    test_context: pd.DataFrame,
) -> SwitchResult:
    """Evaluate one pocket-switch policy."""
    oof_proba = champion_oof[config.PROBA_COLUMNS].to_numpy(dtype=np.float64).copy()
    test_proba = champion_test[config.PROBA_COLUMNS].to_numpy(dtype=np.float64).copy()
    switched_train = np.zeros(len(oof_proba), dtype=bool)
    switched_test = np.zeros(len(test_proba), dtype=bool)
    for rule in policy.rules:
        train_mask = rule.mask_fn(train_context)
        test_mask = rule.mask_fn(test_context)
        oof_proba[train_mask] = candidate_oof[rule.candidate_name].loc[train_mask, config.PROBA_COLUMNS].to_numpy(
            dtype=np.float64
        )
        test_proba[test_mask] = candidate_test[rule.candidate_name].loc[test_mask, config.PROBA_COLUMNS].to_numpy(
            dtype=np.float64
        )
        switched_train |= train_mask
        switched_test |= test_mask
    _validate_probability_rows(oof_proba, context=f"{policy.name} OOF probabilities")
    _validate_probability_rows(test_proba, context=f"{policy.name} test probabilities")
    y_true = champion_oof["y_true"].to_numpy()
    pred = _labels_from_proba(oof_proba)
    del switched_test
    return SwitchResult(
        policy_name=policy.name,
        balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
        log_loss=float(log_loss(y_true, oof_proba, labels=config.CLASS_LABELS)),
        switched_rows=int(switched_train.sum()),
        oof_proba=oof_proba,
        test_proba=test_proba,
    )


def _save_result(
    output_name: str,
    reference_oof: pd.DataFrame,
    reference_test: pd.DataFrame,
    results: list[SwitchResult],
    previous_best: float,
) -> dict[str, object]:
    """Save the best policy artifacts and update logs."""
    best = max(results, key=lambda result: result.balanced_accuracy)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    oof_path = config.STACKING_DIR / f"{output_name}_oof.csv"
    test_proba_path = config.STACKING_DIR / f"{output_name}_test_proba.csv"
    summary_path = config.STACKING_DIR / f"{output_name}_summary_{timestamp}.csv"
    metrics_path = config.STACKING_DIR / f"{output_name}_metrics_{timestamp}.csv"
    _write_oof(reference_oof, best.oof_proba, oof_path)
    _write_test(reference_test, best.test_proba, test_proba_path)
    _write_summary(results, summary_path)
    _write_metrics(reference_oof["y_true"].to_numpy(), best, metrics_path)
    improved = best.balanced_accuracy > previous_best
    submission_path = _write_submission(output_name, reference_test, best.test_proba) if improved else None
    change = f"run tiny deployable pocket switch ensemble; best_policy={best.policy_name}"
    append_experiment_log(change, best.balanced_accuracy, submission_path)
    append_decision_entry(change, best.balanced_accuracy, previous_best, improved)
    if improved and submission_path is not None:
        update_best_score(best.balanced_accuracy)
        update_best_submission(submission_path)
    return {
        "best": best,
        "oof_path": oof_path,
        "test_proba_path": test_proba_path,
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "submission_path": submission_path,
        "improved": improved,
    }


def _write_summary(results: list[SwitchResult], path: Path) -> None:
    """Write policy summary metrics."""
    pd.DataFrame(
        {
            "policy_name": [result.policy_name for result in results],
            "balanced_accuracy": [result.balanced_accuracy for result in results],
            "log_loss": [result.log_loss for result in results],
            "switched_rows": [result.switched_rows for result in results],
        }
    ).to_csv(path, index=False)


def _write_metrics(y_true: np.ndarray, best: SwitchResult, path: Path) -> None:
    """Write detailed metrics for the selected policy."""
    pred = _labels_from_proba(best.oof_proba)
    confusion = confusion_matrix(y_true, pred, labels=config.CLASS_LABELS)
    recalls = recall_score(y_true, pred, labels=config.CLASS_LABELS, average=None)
    rows = [
        {"metric": "balanced_accuracy", "label": "ALL", "value": best.balanced_accuracy},
        {"metric": "log_loss", "label": "ALL", "value": best.log_loss},
        {"metric": "switched_rows", "label": "ALL", "value": best.switched_rows},
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
    """Write a competition submission from probabilities."""
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


def _validate_probability_rows(proba: np.ndarray, *, context: str) -> None:
    """Validate probability rows."""
    if not np.isfinite(proba).all():
        raise ValueError(f"{context} contains non-finite values.")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{context} rows do not sum to 1.")


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert probability rows to labels."""
    labels = np.asarray(config.CLASS_LABELS, dtype=object)
    return labels[proba.argmax(axis=1)]


if __name__ == "__main__":
    main()
