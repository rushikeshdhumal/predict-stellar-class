# Supervised Ensemble And Stacking Plan

## Current State

- Goal: chase `0.972` balanced accuracy without pseudo-labeling or test-label leakage.
- Selection rule: fixed stratified 5-fold CV remains the source of truth.
- Current CV champion: `0.96700608`.
- Current public/test best: `0.96805` from the un-biased old-CatBoost meta-stack.
- Current `best.csv`: biased DART meta-stack, because it is the CV champion.
- Do not use `id` as a feature.

## Champion Artifacts

### CV Champion

- Model: `meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias`
- CV balanced accuracy: `0.96700608`
- Public/test balanced accuracy: `0.96787`
- Biases: `GALAXY=0.0`, `QSO=-0.030`, `STAR=0.014`
- OOF: `data\ensemble\stacking\meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias_oof.csv`
- Test probabilities: `data\ensemble\stacking\meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias_test_proba.csv`
- Submission: `submissions\submission_meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias_20260611_154836.csv`

### Public/Test Best

- Model: `meta_logreg_champion_lgbm_mlp_ft_catboost_c010`
- CV balanced accuracy: `0.96666589`
- Public/test balanced accuracy: `0.96805`
- OOF: `data\ensemble\stacking\meta_logreg_champion_lgbm_mlp_ft_catboost_c010_oof.csv`
- Test probabilities: `data\ensemble\stacking\meta_logreg_champion_lgbm_mlp_ft_catboost_c010_test_proba.csv`
- Submission: `submissions\submission_meta_logreg_champion_lgbm_mlp_ft_catboost_c010_20260611_122755.csv`

## Active Base Artifacts

Use these as the main meta-stack base set unless a new experiment has a specific reason to remove one:

- `blend_deterministic_lgbm_entity_mlp_e12_w0435001`
- `blend_lightgbm_deterministic_local`
- `entity_embedding_mlp_small_e12`
- `ft_transformer_ddp_t4`
- `catboost_native_depth_8`
- `lightgbm_dart_conservative`

The current strongest un-biased DART meta-stack over this set is:

- Model: `meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010`
- CV balanced accuracy: `0.96699299`
- Public/test balanced accuracy: `0.96804`

## Infrastructure Rules

- Save OOF probabilities and averaged test probabilities for every useful base candidate.
- Standard OOF schema: `id`, `fold`, `y_true`, `proba_GALAXY`, `proba_QSO`, `proba_STAR`.
- Keep all base models on the same fixed 5 folds.
- Probability rows must sum to 1.
- `ModelSpec.feature_columns` supports feature-subset specialist models while preserving the same OOF/test artifact format.
- For stacking, use probabilities or log probabilities only; avoid hard voting.

## Acceptance Rules

Accept a new ensemble only if mean 5-fold CV balanced accuracy beats `0.96700608`.

A base model is useful if it beats the champion alone, or if it satisfies all of:

- CV balanced accuracy is within `0.003` of the champion, or there is a clear prior reason to test it as a meta-feature.
- Mean Spearman correlation versus champion OOF probabilities is below `0.985`, or rescue-rate analysis shows complementarity.
- OOF blend or meta-stack improvement is at least `0.0001`.

For diversity checks, compute:

- Spearman correlation per class probability, averaged across classes.
- `disagreement_rate`
- `rescue_rate`
- `new_error_rate`
- `shared_error_rate`
- Small-weight OOF blend lift.

Spearman is only a redundancy filter. The decisive signal is OOF blend/meta improvement plus rescue/new-error behavior.

## Lessons To Preserve

- Logistic meta-learning on saved OOF log probabilities is the most important lift so far.
- The old weak CatBoost is useful as a residual meta-feature even though it is weak standalone.
- Stronger retrained CatBoost improved standalone CV to `0.96341367`, but did not improve the meta-stack.
- Conservative DART is weak standalone (`0.95323340`) but useful in the meta-stack; weak standalone models can still matter if calibrated and different.
- DART class-bias tuning produced the CV champion but transferred poorly to public/test. Avoid more tiny post-hoc bias sweeps unless the CV gain is much larger or stability diagnostics support it.
- Wider C-grid tuning on the expanded DART meta-stack did not beat the current CV champion; best was `C=0.05` at `0.96699861`.
- Removing FT from the logistic stack hurt CV; keep FT as a meta feature.
- GBDT leaf-embedding stacker was close but not useful enough; best blend gain was far below `0.0001`.
- Photometry/redshift-only and categorical/redshift-only specialists both hurt the meta-stack. Do not tune those exact specialist branches without a new feature hypothesis.
- TabNet was too weak (`0.94992297`) and should not be tuned further.

## Rejected Branches

- `catboost_native_balanced_depth_7_od120`: stronger standalone, not useful as replacement/additive meta feature.
- `lightgbm_photometry_redshift_specialist`: standalone `0.93679082`, meta-stack `0.96692962`.
- `lightgbm_categorical_redshift_specialist`: standalone `0.86026112`, meta-stack `0.96693867`.
- `leaf_embedding_lgbm84_meta_probs_small`: standalone `0.96661637`, blend gain only `0.00000677`.
- `tabnet_2gpu_diversity`: standalone `0.94992297`.
- Direct FT weighted blend: tiny lift only, below materiality; FT remains useful inside logistic stacking.

## Next Candidate Queue

1. Targeted FT-Transformer calibration/error-control variant.
   - Do not simply make the model larger.
   - Try class-balanced focal loss, stronger weight decay, stochastic weight averaging, or blend-aware diagnostics.
   - Accept only through OOF/meta-stack improvement.

2. Different LightGBM objective/config family beyond the accepted DART.
   - Examples: log-loss-first GBDT, class-weight variants, or another DART configuration with materially different dropout/regularization.
   - Evaluate as a meta-feature, not only standalone.

3. Error-focused binary specialists.
   - Target hardest class confusions with one-vs-one or one-vs-rest probability features.
   - Feed OOF probabilities into the meta-learner.

4. Kernel approximation model.
   - Fold-safe `Nystroem` or `RBFSampler` plus logistic regression.
   - Proceed only if runtime is reasonable and OOF errors differ materially.

## Test Checklist

- Verify each OOF artifact covers every training row exactly once.
- Verify test probabilities are averaged across all 5 fold models.
- Verify probability rows sum to 1.
- Verify `id` is never used as a feature.
- Compare balanced accuracy, log loss, per-class recall, and confusion matrix for every accepted candidate.
- Record every meaningful accept/reject in `DECISIONS.md` and `submissions/experiment_log.txt`.

## Assumptions

- Use aggressive compute budget when justified.
- Installing CatBoost is allowed.
- No pseudo-labeling or transductive test-feature techniques in this phase.
