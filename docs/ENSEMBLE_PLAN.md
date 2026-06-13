# Ensemble Plan

## Current State

- Goal: improve 3-class balanced accuracy without pseudo-labeling, test-label leakage, or transductive tricks.
- Selection rule: fixed stratified 5-fold CV is the primary acceptance metric.
- Current CV champion: `meta_pocket_switch_candidate_tradeoff_v1`
  - CV balanced accuracy: `0.96723940`
  - Public/test score: `0.96826`
  - Submission: `submissions\submission_meta_pocket_switch_candidate_tradeoff_v1_20260612_173536.csv`
  - OOF: `data\ensemble\stacking\meta_pocket_switch_candidate_tradeoff_v1_oof.csv`
  - Test probabilities: `data\ensemble\stacking\meta_pocket_switch_candidate_tradeoff_v1_test_proba.csv`
- Current public/test best: `meta_rich_lgbm_active_rescue_stronger_reg`
  - CV balanced accuracy: `0.96712409`
  - Public/test score: `0.96832`
  - Submission: `submissions\submission_meta_rich_lgbm_active_rescue_stronger_reg_20260612_124931.csv`
- Current acceptance threshold: mean 5-fold CV balanced accuracy must beat `0.96723940`.

## Current Champion Architecture

The CV champion is a small deployable post-processing layer on top of the strongest rich LightGBM stacker.

1. Start from `meta_rich_lgbm_active_rescue_stronger_reg`.
2. Use fixed, hand-picked rules from candidate tradeoff diagnostics.
3. In those narrow pockets, replace champion probabilities with probabilities from near-miss candidates:
   - `meta_rich_lgbm_active_rescue_more_trees`
   - `meta_rich_lgbm_active_rescue_no_context`
   - `meta_rich_lgbm_active_rescue_conservative`
4. The accepted policy is `cumulative_all`, five deployable rules using only champion prediction plus original non-ID features (`g_r`, `u_g`, `spectral_type`).

This improved OOF balanced accuracy and all three OOF recalls, but it did not beat the previous public/test best. Treat it as the CV champion, not yet as evidence that rule expansion transfers well.

## Strongest Base/Stacking Direction

The strongest durable architecture is rich meta-stacking over saved OOF/test probabilities.

Use these base artifacts as the default rich stack inputs unless a new experiment has a specific reason to change them:

- `blend_deterministic_lgbm_entity_mlp_e12_w0435001`
- `blend_lightgbm_deterministic_local`
- `entity_embedding_mlp_small_e12`
- `ft_transformer_ddp_t4`
- `catboost_native_depth_8`
- `lightgbm_dart_conservative`
- `xgboost_hist_depth_8`
- `xgboost_optuna_trial_0`
- `lightgbm_gbdt_balanced_low_lr`

The key rich meta-features are raw/log base probabilities, entropy, top-margin, pairwise logit gaps, cross-base probability aggregates, vote counts, champion deltas, and small domain context.

## Important Lessons

- Logistic stacking on saved OOF log probabilities produced the first major ensemble jump.
- Rich LightGBM meta-stacking with strong regularization is the best general architecture so far.
- The old weak CatBoost and conservative DART remain useful as diversity/meta features despite weak standalone CV.
- FT-Transformer is useful inside stacks, but the tested focal/SWA calibration variant was rejected.
- Removing domain context was close in CV (`0.96710695`) but lower publicly (`0.96812`), so keep context unless testing a specific transfer hypothesis.
- The accepted pocket switch improved CV (`0.96723940`) but public/test fell to `0.96826`, below the rich stacker’s `0.96832`; rule-based gains may overfit small OOF pockets.
- Pocket-switch stability audit showed `cumulative_all` was positive on all folds and better than reduced policies, so do not replace it with a reduced policy.

## Rejected Directions

Do not repeat these without a materially different hypothesis:

- Stronger CatBoost retrain as replacement/additive feature.
- Photometry/redshift-only or categorical/redshift-only specialists.
- TabNet diversity branch.
- Randomized RBF/kernel approximation branch in its tested form.
- Scalar binary specialist margin features.
- Hard-gated hierarchy using STAR-vs-rest then GALAXY-vs-QSO.
- Soft residual reliability blend in its tested form.
- Crude class-weight tilts for GALAXY preservation; they over-correct GALAXY and damage STAR.
- Broad expansion of pocket-switch rules based only on tiny OOF pockets.

## Reports

Human-readable diagnostics live in `docs\reports`.

- `docs\reports\champion_error_diagnostics_20260612_110056.md`
- `docs\reports\candidate_tradeoff_diagnostics_20260612_172857.md`
- `docs\reports\pocket_switch_stability_report_20260612_174346.md`

Machine-readable diagnostic CSVs remain in `data\processed`.

## Next Best Steps

1. Prefer transfer-aware validation before more rule work.
   - The CV champion did not become the public/test best.
   - Any new pocket-switch experiment should have a larger CV lift than `+0.0001` and a reason it should transfer better.

2. If continuing pocket switches, test only a tiny fixed set.
   - No broad rule search without confirmation.
   - Require positive fold stability and balanced recall impact.
   - Compare against both CV champion and public/test best behavior.

3. If returning to models, use a materially new error target.
   - Do not run another generic full-feature GBDT.
   - A candidate must target a specific confusion/calibration gap shown in diagnostics.

4. FT work is only worth revisiting with a new diagnosis.
   - Do not repeat the focal/SWA recipe.
   - Revisit only if diagnostics show FT uniquely rescues a stable pocket.

## Guardrails

- Never use `id` as a feature.
- Keep all base models and stackers on fixed stratified 5-fold CV.
- Save OOF probabilities and averaged test probabilities for any useful candidate.
- Probability rows must sum to 1.
- Use standard OOF schema: `id`, `fold`, `y_true`, `proba_GALAXY`, `proba_QSO`, `proba_STAR`.
- Log meaningful results in `submissions\experiment_log.txt` and `DECISIONS.md`.
