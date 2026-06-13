# Stellar Classification: Fold-Safe Ensemble Learning

An end-to-end machine learning project for classifying synthetic SDSS17-style objects as `GALAXY`, `QSO`, or `STAR`. The project emphasizes validation discipline, no-leakage feature engineering, fold-safe OOF artifacts, and supervised stacking.

## Highlights

- Improved from raw XGBoost CV balanced accuracy `0.92682847` to current CV champion `0.96723940`.
- Best public/test score so far: `0.96832`.
- Built a reusable OOF/test-probability artifact layer for blending, stacking, diversity checks, diagnostics, and post-processing.
- Current strongest general architecture is rich LightGBM meta-stacking over base-model probabilities.
- Current CV champion adds a small deployable pocket-switch layer on top of the strongest rich stacker.
- Fixed stratified 5-fold CV remains the source of truth, even when public/test feedback disagrees.

## Problem

The task is multiclass stellar classification from 11 tabular competition columns:

- spatial coordinates: `alpha`, `delta`
- photometric bands: `u`, `g`, `r`, `i`, `z`
- redshift: `redshift`
- categoricals: `spectral_type`, `galaxy_population`
- identifier: `id`, explicitly excluded from modeling

The optimization metric is balanced accuracy, so per-class recall matters more than raw accuracy.

## Current Results

| Milestone | CV balanced accuracy | Public/test score |
|---|---:|---:|
| Raw XGBoost baseline | `0.92682847` | `0.92730` |
| Raw LightGBM baseline | `0.95072371` | `0.95012` |
| Feature-engineered LightGBM | `0.95361566` | `0.95434` |
| Tuned LightGBM | `0.95797470` | `0.95801` |
| Deterministic LightGBM blend | `0.95877923` | `0.95894` |
| Entity-embedding MLP + LightGBM blend | `0.96237557` | `0.96291` |
| Logistic meta-learner with LGBM, MLP, FT | `0.96657219` | `0.96803` |
| Logistic meta-learner + CatBoost residual signal | `0.96666589` | `0.96805` |
| Conservative DART meta-stack | `0.96699299` | `0.96804` |
| Rich LightGBM meta-stacker, stronger regularization | `0.96712409` | `0.96832` |
| Pocket-switch ensemble | `0.96723940` | `0.96826` |

Current winners:

- **CV champion**: `meta_pocket_switch_candidate_tradeoff_v1`
  - CV balanced accuracy: `0.96723940`
  - Public/test score: `0.96826`
  - Submission: `submissions/submission_meta_pocket_switch_candidate_tradeoff_v1_20260612_173536.csv`
- **Public/test best**: `meta_rich_lgbm_active_rescue_stronger_reg`
  - CV balanced accuracy: `0.96712409`
  - Public/test score: `0.96832`
  - Submission: `submissions/submission_meta_rich_lgbm_active_rescue_stronger_reg_20260612_124931.csv`

## Modeling Approach

### Feature Engineering

The mature feature set includes:

- color indices: `u_g`, `g_r`, `r_i`, `i_z`, `u_r`, `g_i`, `r_z`, `u_z`
- magnitude summaries: mean, standard deviation, min, max, range
- spatial encodings: sine/cosine angles and unit-sphere coordinates
- categorical interaction: `spectral_population`

All features are derived from the original competition columns. `id` is never used as a predictive feature.

### Base Models

The active rich stack uses fold-aligned probability artifacts from:

- deterministic LightGBM blends
- entity-embedding MLP
- FT-Transformer
- CatBoost native categorical model
- conservative LightGBM DART
- XGBoost diversity candidates
- class-balanced low-learning-rate LightGBM

Weak standalone models are not automatically rejected. A model can be useful if it provides calibrated, different OOF probabilities that improve a stack.

### Stacking And Post-Processing

The strongest general stacker is a shallow, strongly regularized LightGBM trained on rich probability-derived meta-features:

- raw and log probabilities,
- entropy and top-1/top-2 margin,
- pairwise logit gaps,
- cross-base mean/std/min/max probabilities,
- vote counts,
- champion-vs-base deltas,
- compact domain context.

The current CV champion applies a tiny deployable pocket switch on top of the strongest rich stacker. It replaces champion probabilities with near-miss candidate probabilities only in five fixed pockets discovered by diagnostics. This improved CV but did not beat the public/test best, so further rule expansion needs stronger evidence.

## Reproducibility

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the main baseline pipeline:

```powershell
python run.py
```

Train one fold-safe OOF base model:

```powershell
python scripts/train_oof_model.py --model-name <model_name>
```

Audit model diversity:

```powershell
python scripts/evaluate_oof_diversity.py --candidate-model <model_name>
```

Train the rich LightGBM meta-stacker:

```powershell
python scripts/train_rich_meta_stack.py --model-type lightgbm --output-name <stack_name> --lgbm-variant stronger_reg
```

Run the accepted pocket-switch policy family:

```powershell
python scripts/run_pocket_switch_ensemble.py --output-name <output_name>
```

Create diagnostics:

```powershell
python scripts/analyze_champion_errors.py
python scripts/analyze_candidate_tradeoffs.py
python scripts/analyze_pocket_switch_stability.py
```

## Project Structure

```text
.
├── data/
│   ├── raw/                 # Competition train/test data
│   ├── processed/           # EDA and machine-readable diagnostic outputs
│   └── ensemble/            # OOF, test probability, and stacking artifacts
├── docs/
│   ├── reports/             # Human-readable diagnostics
│   ├── ENSEMBLE_PLAN.md     # Future-facing ensemble strategy
│   └── *.py                 # Kaggle notebook/export helper scripts
├── scripts/                 # Experiment, stacking, tuning, and audit scripts
├── src/                     # Reusable data, training, ensemble, and meta-feature code
├── submissions/             # Submission files, best score, experiment log
├── DECISIONS.md             # Durable modeling decisions and lessons
└── run.py                   # Main baseline pipeline
```

## Documentation

- `docs/ENSEMBLE_PLAN.md`: current strategy, next-step criteria, and guardrails.
- `DECISIONS.md`: compact durable decision record.
- `submissions/experiment_log.txt`: chronological experiment history.
- `docs/reports/`: human-readable diagnostic reports.
- `data/processed/eda_summary_baseline.md`: baseline EDA summary.

## Engineering Discipline

- Fixed stratified 5-fold CV is the selection metric.
- Every accepted candidate must beat `submissions/best_score.txt`.
- OOF artifacts must cover every training row exactly once.
- Test probabilities are averaged across fold models.
- Probability rows are validated to sum to 1.
- Meaningful successes and failures are recorded in `DECISIONS.md` and `submissions/experiment_log.txt`.
- No pseudo-labeling, test-label leakage, or transductive test-feature tricks are used.

## Current Direction

The next useful work should be transfer-aware. The CV champion did not become the public/test best, so broad pocket-rule expansion is risky. Future experiments should either:

- show a stronger, stable CV lift with a credible transfer reason, or
- target a specific diagnostic confusion/calibration gap with a materially different model or representation.
