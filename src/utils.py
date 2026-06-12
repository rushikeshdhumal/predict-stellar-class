"""Submission, scoring, and experiment logging utilities."""

from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src import config


def ensure_output_dirs() -> None:
    """Create output directories used by the baseline run."""
    config.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    config.TEST_PROBA_DIR.mkdir(parents=True, exist_ok=True)
    config.STACKING_DIR.mkdir(parents=True, exist_ok=True)


def get_best_score(path: Path = config.BEST_SCORE_PATH) -> float:
    """Read the current best validation balanced accuracy.

    Args:
        path: Best-score file path.

    Returns:
        Current best score, or negative infinity when no champion exists.
    """
    if not path.exists():
        return float("-inf")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return float("-inf")
    return float(content)


def update_best_score(score: float, path: Path = config.BEST_SCORE_PATH) -> None:
    """Write a new best validation balanced accuracy.

    Args:
        score: New champion score.
        path: Best-score file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{score:.8f}\n", encoding="utf-8")


def create_submission(
    predictions: np.ndarray,
    sample_submission: pd.DataFrame,
    output_dir: Path = config.SUBMISSIONS_DIR,
) -> Path:
    """Create a competition submission file.

    Args:
        predictions: Predicted class labels.
        sample_submission: Sample submission dataframe containing the ID column.
        output_dir: Directory where the submission will be written.

    Returns:
        Path to the created submission file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission = sample_submission.copy()
    submission[config.TARGET_COLUMN] = predictions
    output_path = output_dir / f"submission_{config.MODEL_NAME}_{timestamp}.csv"
    submission.to_csv(output_path, index=False)
    return output_path


def update_best_submission(submission_path: Path) -> None:
    """Copy the champion submission to the canonical best.csv path.

    Args:
        submission_path: Path to the newly created champion submission.
    """
    shutil.copyfile(submission_path, config.BEST_SUBMISSION_PATH)


def append_experiment_log(change: str, score: float, submission_path: Path | None) -> None:
    """Append a human-readable experiment log line.

    Args:
        change: Description of the experimental change.
        score: Mean validation balanced accuracy.
        submission_path: Submission path if a submission was produced.
    """
    config.EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    submission_text = str(submission_path) if submission_path is not None else "None"
    line = f"{timestamp}\t{change}\tscore={score:.8f}\tsubmission={submission_text}\n"
    with config.EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(line)


def append_decision_entry(change: str, score: float, best: float, improved: bool) -> None:
    """Append a decision entry for a meaningful experiment.

    Args:
        change: Description of the experimental change.
        score: Mean validation balanced accuracy.
        best: Previous best score.
        improved: Whether the experiment improved on the previous best.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    previous = "none" if best == float("-inf") else f"{best:.8f}"
    status = "accepted" if improved else "rejected"
    entry = (
        f"\n## {timestamp} - {status}: {change}\n"
        f"- Mean validation balanced accuracy: {score:.8f}\n"
        f"- Previous best: {previous}\n"
    )
    with config.DECISIONS_PATH.open("a", encoding="utf-8") as file:
        file.write(entry)
