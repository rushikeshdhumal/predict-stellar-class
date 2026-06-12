# Codex Agent Instructions – Stellar Classification Competition

## Project Context
- **Goal**: Predict stellar class (GALAXY, STAR, QSO) with balanced accuracy.
- **Data**: 11 features (synthetic, but based on SDSS17). Training set in `data/raw/train.csv`.
- **Evaluation**: Balanced accuracy (scikit-learn). Use stratified 5-fold CV as the gold standard.
- **Metric Decision**: Only accept changes that improve **mean validation balanced accuracy** (not public LB).

## Iterative Development Loop (REQUIRED)
For every experiment, follow this exact cycle:
1. **Read current state** – Check `submissions/best_score.txt` for the best balanced accuracy so far.
2. **Propose one hypothesis** – Before proposing a new experiment, read DECISIONS.md to avoid repeating failed approaches (e.g., "add interaction feature", "tune learning rate", "switch to LightGBM").
3. **Run experiment** – Execute `python run.py` with the new change. Codex must ensure MLflow logs all parameters.
4. **Evaluate** – Compare new balanced accuracy (from MLflow or printed output) against `best_score.txt`.
5. **Log result** – MLflow automatically logs. Also append a human-readable line to `submissions/experiment_log.txt` (timestamp, change, score, submission file).
6. **Decision**:
   - If improved → Update `submissions/best_score.txt` and `submissions/best.csv`.
   - If not improved → Revert change and move to next hypothesis.
   - After running an experiment, append a decision entry if the change was meaningful (improvement or notable failure).
7. **Repeat** until user says stop or no improvement after 5 consecutive tries.

## Coding Standards (Codex MUST follow)
- **Type hints** on all functions.
- **Docstrings** (Google style) for every module and public function.
- **No hardcoded paths** – use `src/config.py`.
- **Reproducibility** – set random seeds everywhere (`config.SEED`).
- **Always stratify** – by target class in train/test splits and cross-validation.

## Model Training Rules
- Always use **5-fold stratified cross-validation**.
- Track every run with MLflow: log params, metrics (balanced_accuracy, log_loss), and the CV scores per fold.
- Save each fold's model inside `models/` (not tracked by git).
- For ensemble: average predicted probabilities from all folds.

## Feature Engineering Rules
- Create features only from the original 11 – no leakage from test set.
- Map synthetic column names to SDSS17 original names (see `data/raw/column_mapping.md` if exists).
- Standardize or normalize only after splitting folds (use `sklearn.pipeline`).
- Avoid using ID columns (`obj_ID`, etc.) as features unless explicitly for splitting.

## Codex Behavior
- **Do not** propose changes without a hypothesis.
- **Do not** run more than one experiment at a time.
- **Ask for confirmation** before running a long hyperparameter search (>10 trials).
- **When in doubt**, print the current best score and ask user.

## Files to Monitor
- `run.py` – main orchestrator.
- `src/data.py` – feature engineering.
- `src/train.py` – model training and CV.
- `submissions/best_score.txt` – current champion score.

## Commands (for Codex to execute)
- `python run.py` – full pipeline (load → process → train → predict → submit).
- `mlflow ui` – launch tracking UI.
- `pytest tests/` – run tests (if tests exist).

## Data Schema (from Kaggle competition)

**Target**: `class` (string) – 3 classes: `GALAXY`, `STAR`, `QSO`

**Features (11 columns, no nulls)**:

| Column | Type | Unique Values / Notes |
|--------|------|----------------------|
| `id` | integer | Unique identifier – **DO NOT use as feature** |
| `alpha` | float | Right Ascension (spatial) |
| `delta` | float | Declination (spatial) |
| `u` | float | Magnitude in u band |
| `g` | float | Magnitude in g band |
| `r` | float | Magnitude in r band |
| `i` | float | Magnitude in i band |
| `z` | float | Magnitude in z band |
| `redshift` | float | Redshift value |
| `spectral_type` | string | 4 unique values (e.g., "Galaxy", "Star", etc. – check exact values) |
| `galaxy_population` | string | 2 unique values (e.g., "High", "Low" or similar) |

**Important constraints**:
- No missing values in any column.
- `spectral_type` and `galaxy_population` are categorical – one-hot encode or label encode.
- Never use `id` as a feature (leakage / non‑generalizing).
- For spatial features `alpha` and `delta`, consider creating interaction features (e.g., density) but avoid overfitting.