# Stellar Classification: Fold-Safe Ensemble Learning

An end-to-end machine learning project for classifying synthetic SDSS17-style astronomical objects as `GALAXY`, `QSO`, or `STAR`. The work emphasizes rigorous validation, reproducible experimentation, feature engineering, and supervised stacking under a strict no-leakage protocol.

## Highlights

- Improved from a raw XGBoost baseline of `0.92683` CV balanced accuracy to a stacked ensemble CV champion of `0.96701`.
- Best submitted public/test balanced accuracy so far: `0.96805`.
- Built a fold-safe OOF artifact layer for blending, logistic meta-learning, and model diversity analysis.
- Combined gradient boosting, neural tabular models, FT-Transformer probabilities, CatBoost residual signal, and class-bias calibration.
- Used fixed stratified 5-fold CV as the selection metric throughout, even when public/test feedback disagreed.

## Problem

The task is multiclass stellar classification from 11 tabular features:

- Spatial coordinates: `alpha`, `delta`
- Photometric bands: `u`, `g`, `r`, `i`, `z`
- Redshift: `redshift`
- Categoricals: `spectral_type`, `galaxy_population`
- Identifier: `id`, explicitly excluded from modeling

The optimization target is balanced accuracy, which makes per-class recall quality more important than raw accuracy.

## Current Results

| Milestone | CV balanced accuracy | Public/test score |
|---|---:|---:|
| Raw XGBoost baseline | `0.92682847` | `0.92730` |
| Raw LightGBM baseline | `0.95072371` | `0.95012` |
| Feature-engineered LightGBM | `0.95361566` | `0.95434` |
| Tuned LightGBM | `0.95797470` | `0.95801` |
| Deterministic LightGBM blend + bias tuning | `0.95877923` | `0.95894` |
| Entity-embedding MLP + LightGBM blend | `0.96237557` | `0.96291` |
| Logistic meta-learner with LGBM, MLP, FT | `0.96657219` | `0.96803` |
| Logistic meta-learner + CatBoost residual feature | `0.96666589` | `0.96805` |
| Logistic meta-learner + CatBoost + DART + bias | `0.96700608` | `0.96787` |

The project tracks two important winners:

- **CV champion**: `meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias`
  - CV balanced accuracy: `0.96700608`
  - Submission: `submissions/submission_meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias_20260611_154836.csv`
- **Best public/test submission**: `meta_logreg_champion_lgbm_mlp_ft_catboost_c010`
  - Public/test balanced accuracy: `0.96805`
  - Submission: `submissions/submission_meta_logreg_champion_lgbm_mlp_ft_catboost_c010_20260611_122755.csv`

## Modeling Approach

### Feature Engineering

The strongest single-model feature set includes:

- Color indices: `u_g`, `g_r`, `r_i`, `i_z`, `u_r`, `g_i`, `r_z`, `u_z`
- Magnitude summaries: mean, standard deviation, min, max, range
- Spatial encodings: sine/cosine angles and unit-sphere coordinates
- Categorical interaction: `spectral_population`

All features are derived only from the original training columns, and `id` is never used as a predictive feature.

### Base Models

The active ensemble uses fold-aligned probability artifacts from:

- Tuned LightGBM blends
- Entity-embedding MLP
- FT-Transformer trained on Kaggle 2xT4 GPUs
- CatBoost native categorical model
- Conservative LightGBM DART model

Weak standalone models are not automatically rejected. A candidate can be valuable if it makes different errors and improves the fold-safe meta-stack.

### Stacking

The strongest ensemble family is multinomial logistic regression over clipped log probabilities from base models. Candidate models are screened with:

- Mean 5-fold balanced accuracy
- OOF probability Spearman correlation
- Disagreement, rescue, new-error, and shared-error rates
- OOF blend/meta-stack lift
- Per-class recall and confusion matrix behavior

## Reproducibility

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the main supervised pipeline:

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

Train a logistic meta-stack from probability artifacts:

```powershell
python scripts/train_meta_stack_from_paths.py --output-name <stack_name> --oof-paths <oof_a.csv> <oof_b.csv> --test-proba-paths <test_a.csv> <test_b.csv> --base-names <model_a> <model_b>
```

Create a submission from averaged test probabilities:

```powershell
python scripts/create_submission_from_proba.py --proba-path <test_proba.csv> --submission-name <name>
```

## Project Structure

```text
.
├── data/
│   ├── raw/                 # Competition train/test data
│   ├── processed/           # EDA summaries and derived reports
│   └── ensemble/            # OOF and test probability artifacts
├── docs/
│   └── ENSEMBLE_PLAN.md     # Future-facing ensemble strategy
├── scripts/                 # Experiment, stacking, tuning, and audit scripts
├── src/                     # Reusable data, training, ensemble, and stacking code
├── submissions/             # Submission files, best score, experiment log
├── DECISIONS.md             # Durable modeling decisions and lessons
└── run.py                   # Main baseline pipeline
```

## Engineering Discipline

- Fixed stratified 5-fold CV is the source of truth.
- Every accepted candidate must beat the current CV champion in `submissions/best_score.txt`.
- OOF artifacts must cover every training row exactly once.
- Test probabilities are averaged across fold models.
- Probability rows are validated to sum to 1.
- Meaningful successes and failures are recorded in `DECISIONS.md` and `submissions/experiment_log.txt`.
- No pseudo-labeling, test-label leakage, or transductive test-feature tricks are used in the current phase.

## What This Demonstrates

This repository is intentionally more than a leaderboard chase. It demonstrates practical data science judgment: moving from baseline modeling to targeted feature engineering, using validation discipline over noisy public feedback, building reusable ensemble infrastructure, and rejecting plausible ideas when their OOF behavior does not justify the complexity.

Current next direction: targeted diversity that can improve the meta-stack, especially calibrated FT-Transformer variants, materially different LightGBM objectives, error-focused binary specialists, or kernel approximation models.
