# Decision Log

This file keeps durable modeling decisions and lessons. Full chronological run history lives in `submissions\experiment_log.txt`.

## Current Winners

- CV champion: `meta_pocket_switch_candidate_tradeoff_v1`
  - CV balanced accuracy: `0.96723940`
  - Public/test score: `0.96826`
  - Submission: `submissions\submission_meta_pocket_switch_candidate_tradeoff_v1_20260612_173536.csv`

- Public/test best: `meta_rich_lgbm_active_rescue_stronger_reg`
  - CV balanced accuracy: `0.96712409`
  - Public/test score: `0.96832`
  - Submission: `submissions\submission_meta_rich_lgbm_active_rescue_stronger_reg_20260612_124931.csv`

- Current acceptance threshold: mean 5-fold CV balanced accuracy must beat `0.96723940`.

## Major Milestones

- Feature-engineered LightGBM baseline reached CV `0.95361566`.
- Tuned LightGBM reached CV `0.95797470`.
- Deterministic LightGBM blend reached CV `0.95877923`.
- Entity-embedding MLP plus LightGBM blend reached CV `0.96237557`.
- Logistic meta-learner over saved OOF probabilities reached CV `0.96657219`.
- Adding old CatBoost probabilities improved public/test to `0.96805`.
- Conservative DART in the meta-stack reached CV `0.96699299`.
- Rich LightGBM meta-stacker with stronger regularization reached CV `0.96712409` and current public/test best `0.96832`.
- Tiny deployable pocket-switch ensemble reached current best CV `0.96723940`, but public/test was lower at `0.96826`.

## Accepted Principles

- Fixed stratified 5-fold CV is the selection metric, even when public/test disagrees.
- Saved OOF/test probability artifacts are the main ensemble substrate.
- Rich meta-features over probabilities are more useful than another generic full-feature base model.
- Keep weak-but-diverse OOF sources when they improve the stack; weak standalone CV is not enough reason to drop them.
- Keep `ft_transformer_ddp_t4`, old `catboost_native_depth_8`, and conservative DART in the ensemble context unless a specific ablation says otherwise.
- Use public/test as transfer feedback, not as the primary acceptance metric.

## Durable Architecture Decisions

- Best general stacker so far: rich-feature LightGBM over base probabilities.
- Best transfer model so far: `meta_rich_lgbm_active_rescue_stronger_reg`.
- Best CV model so far: `meta_pocket_switch_candidate_tradeoff_v1`.
- Logistic stacking on clipped log probabilities was the key jump that made later stacking useful.
- Rich meta-features should include raw/log probabilities, entropy, top-margin, pairwise logit gaps, cross-base aggregates, vote counts, champion deltas, and compact domain context.
- Pocket-switching can improve CV, but its public/test miss means rule expansion needs stronger evidence than a tiny OOF pocket gain.

## Feature Decisions

- Accepted feature families:
  - color indices,
  - magnitude summaries,
  - `spectral_population` categorical cross,
  - spatial trigonometric/unit-sphere features.
- Rejected as standalone improvement:
  - redshift log/negative-flag transform.
- Current feature set is mature. New feature work should be tied to a specific error-analysis hypothesis.

## Rejected Directions

Do not repeat these without a materially different hypothesis:

- Stronger CatBoost retrain as replacement/additive feature; old weak CatBoost remained more useful in the stack.
- GBDT leaf-embedding neural stacker; blend gain was too small.
- TabNet; standalone CV was too weak.
- Direct FT weighted blend; keep FT as a meta feature instead.
- Calibrated FT focal/SWA recipe; it weakened standalone quality and did not improve the stack.
- Photometry/redshift-only and categorical/redshift-only specialists.
- Near-duplicate full-feature GBDT diversity trials.
- Class-balanced low-learning-rate LightGBM as additive meta feature; useful but did not beat champion or transfer better.
- One-vs-one binary specialist pseudo-probability artifacts.
- Scalar pair-specialist margin features.
- Hard-gated hierarchy using STAR-vs-rest then GALAXY-vs-QSO.
- Soft residual reliability blend in its tested form.
- Crude GALAXY class-weight guarding; it over-corrected GALAXY and collapsed STAR recall.
- Randomized RBF/kernel approximation in its tested form.

## Report References

Human-readable diagnostics now live in `docs\reports`.

- `docs\reports\champion_error_diagnostics_20260612_110056.md`
- `docs\reports\candidate_tradeoff_diagnostics_20260612_172857.md`
- `docs\reports\pocket_switch_stability_report_20260612_174346.md`

Machine-readable diagnostic CSVs remain in `data\processed`.

## Next Decision Criteria

- Accept only if mean 5-fold CV balanced accuracy beats `0.96723940`.
- For rule-based/pocket-switch work, require:
  - positive fold stability,
  - balanced recall impact,
  - larger CV lift than the current tiny rule gain,
  - a credible reason public/test transfer should improve.
- For new model work, require a specific confusion/calibration target from diagnostics.
- Do not run a large rule search or hyperparameter search without confirmation.

## Guardrails

- Never use `id` as a feature.
- Keep all models on fixed stratified 5-fold CV.
- Save OOF and averaged test probabilities for useful candidates.
- Probability rows must sum to 1.
- Log meaningful results in `submissions\experiment_log.txt` and this file.
