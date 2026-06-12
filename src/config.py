"""Project configuration for stellar classification experiments."""

from pathlib import Path

SEED: int = 42
N_FOLDS: int = 5
N_JOBS: int = -1

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
DOCS_DIR: Path = ROOT_DIR / "docs"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SUBMISSIONS_DIR: Path = ROOT_DIR / "submissions"
MODELS_DIR: Path = ROOT_DIR / "models"
MLRUNS_DIR: Path = ROOT_DIR / "mlruns"
ENSEMBLE_DIR: Path = DATA_DIR / "ensemble"
OOF_DIR: Path = ENSEMBLE_DIR / "oof"
TEST_PROBA_DIR: Path = ENSEMBLE_DIR / "test_proba"
STACKING_DIR: Path = ENSEMBLE_DIR / "stacking"

TRAIN_PATH: Path = RAW_DATA_DIR / "train.csv"
TEST_PATH: Path = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH: Path = RAW_DATA_DIR / "sample_submission.csv"
BEST_SCORE_PATH: Path = SUBMISSIONS_DIR / "best_score.txt"
BEST_SUBMISSION_PATH: Path = SUBMISSIONS_DIR / "best.csv"
EXPERIMENT_LOG_PATH: Path = SUBMISSIONS_DIR / "experiment_log.txt"
EDA_REPORT_PATH: Path = PROCESSED_DATA_DIR / "eda_summary_baseline.md"
DECISIONS_PATH: Path = ROOT_DIR / "DECISIONS.md"

MLFLOW_TRACKING_URI: str = f"file:///{MLRUNS_DIR.as_posix()}"
MLFLOW_EXPERIMENT_NAME: str = "stellar_classification"

TARGET_COLUMN: str = "class"
ID_COLUMN: str = "id"
FOLD_COLUMN: str = "fold"
CLASS_LABELS: list[str] = ["GALAXY", "QSO", "STAR"]
PROBA_COLUMNS: list[str] = [f"proba_{label}" for label in CLASS_LABELS]

BASE_NUMERIC_FEATURES: list[str] = [
    "alpha",
    "delta",
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
]
COLOR_FEATURES: list[str] = ["u_g", "g_r", "r_i", "i_z", "u_r", "g_i", "r_z", "u_z"]
MAGNITUDE_BANDS: list[str] = ["u", "g", "r", "i", "z"]
MAGNITUDE_SUMMARY_FEATURES: list[str] = [
    "mag_mean",
    "mag_std",
    "mag_min",
    "mag_max",
    "mag_range",
]
SPATIAL_FEATURES: list[str] = [
    "alpha_sin",
    "alpha_cos",
    "delta_sin",
    "delta_cos",
    "sky_x",
    "sky_y",
    "sky_z",
]
NUMERIC_FEATURES: list[str] = (
    BASE_NUMERIC_FEATURES + COLOR_FEATURES + MAGNITUDE_SUMMARY_FEATURES + SPATIAL_FEATURES
)
CATEGORICAL_CROSS_FEATURES: list[str] = ["spectral_population"]
CATEGORICAL_FEATURES: list[str] = [
    "spectral_type",
    "galaxy_population",
    *CATEGORICAL_CROSS_FEATURES,
]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_NAME: str = "lightgbm_optuna_trial_84_n_estimators_200"

LGBM_PARAMS: dict[str, object] = {
    "objective": "multiclass",
    "num_class": len(CLASS_LABELS),
    "n_estimators": 200,
    "max_depth": -1,
    "num_leaves": 240,
    "min_child_samples": 30,
    "learning_rate": 0.06570351934757576,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "reg_lambda": 4.0,
    "reg_alpha": 2.0,
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbosity": -1,
}
