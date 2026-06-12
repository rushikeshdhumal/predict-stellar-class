# Decision Log

This file keeps durable modeling decisions and lessons. Full chronological run history lives in `submissions/experiment_log.txt`.

## Current Winners

- CV champion: `meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias`
  - CV balanced accuracy: `0.96700608`
  - Public/test balanced accuracy: `0.96787`
  - Submission: `submissions\submission_meta_logreg_champion_lgbm_mlp_ft_catboost_dart_c010_bias_20260611_154836.csv`

- Public/test best: `meta_logreg_champion_lgbm_mlp_ft_catboost_c010`
  - CV balanced accuracy: `0.96666589`
  - Public/test balanced accuracy: `0.96805`
  - Submission: `submissions\submission_meta_logreg_champion_lgbm_mlp_ft_catboost_c010_20260611_122755.csv`

- Current acceptance threshold: mean 5-fold CV balanced accuracy must beat `0.96700608`.

## Score Milestones

- XGBoost raw baseline: CV `0.92682847`, test `0.92730`.
- LightGBM raw baseline: CV `0.95072371`, test `0.95012`.
- Accepted feature-engineered LightGBM baseline after colors, magnitude summaries, categorical cross, and spatial features: CV `0.95361566`, test `0.95434`.
- Tuned LightGBM champion: CV `0.95797470`, test `0.95801`.
- Deterministic LightGBM blend with class-bias tuning: CV `0.95877923`, test `0.95894`.
- Entity-embedding MLP plus LightGBM blend: CV `0.96237557`, test `0.96291`.
- Logistic meta-learner with LightGBM blend, deterministic LGBM, MLP, and FT: CV `0.96657219`, test `0.96803`.
- Old CatBoost probability feature added to meta-learner: CV `0.96666589`, test `0.96805`.
- Conservative LightGBM DART probability feature added to meta-learner: CV `0.96699299`, test `0.96804`.
- DART meta-stack class-bias tuning: CV `0.96700608`, test `0.96787`.

## Accepted Strategy

- Use fixed stratified 5-fold CV as the selection metric, even when public/test disagrees.
- Use saved OOF/test probability artifacts as the ensemble substrate.
- Use logistic meta-learning on clipped log probabilities as the primary stacker.
- Keep these active base artifacts unless a future experiment has a specific reason to remove one:
  - `blend_deterministic_lgbm_entity_mlp_e12_w0435001`
  - `blend_lightgbm_deterministic_local`
  - `entity_embedding_mlp_small_e12`
  - `ft_transformer_ddp_t4`
  - `catboost_native_depth_8`
  - `lightgbm_dart_conservative`
- Keep `ft_transformer_ddp_t4` inside the meta-stack. It is weak standalone but useful as a meta feature.
- Keep old `catboost_native_depth_8` inside the meta-stack. Its standalone score is weak, but its residual signal transferred best to public/test.
- Treat weak standalone models as potentially useful only if they add calibrated, different OOF probabilities to the stack.

## Feature Engineering Decisions

- Accepted:
  - Color indices.
  - Magnitude summary features.
  - `spectral_population` categorical cross.
  - Spatial trigonometric/unit-sphere features.
- Rejected:
  - Redshift log/negative-flag transform as a standalone LightGBM feature group.
- Current feature set is mature; further feature work should be tied to a clear specialist or error-analysis hypothesis.

## Meta-Learner Decisions

- Logistic regression on base-model log probabilities is the best-performing meta-learner family so far.
- C-grid tuning on the expanded DART base set did not beat the current CV champion.
  - Best completed wide-grid score: `0.96699861` at `C=0.05`.
  - Current biased DART CV champion remains `0.96700608`.
- Class-bias tuning can improve CV slightly, but recent transfer was poor.
  - Biased DART stack: CV `0.96700608`, test `0.96787`.
  - Unbiased DART stack: CV `0.96699299`, test `0.96804`.
  - Old-CatBoost stack: CV `0.96666589`, test `0.96805`.
- Do not run more tiny class-bias sweeps unless the CV gain is materially larger or supported by stability diagnostics.

## Rejected Branches

- Stronger CatBoost retrain:
  - `catboost_native_balanced_depth_7_od120`
  - Standalone CV `0.96341367`
  - Replacement meta-stack CV `0.96662527`
  - Additive meta-stack CV `0.96663920`
  - Decision: reject; old weak CatBoost is a better residual meta feature.

- GBDT leaf-embedding neural stacker:
  - Standalone CV `0.96661637`
  - Best blend gain only `0.00000677`
  - Decision: reject; too little reliable lift.

- TabNet:
  - Standalone CV `0.94992297`
  - Decision: reject; too weak to tune further.

- Direct FT weighted blend:
  - Standalone CV `0.95316013`
  - Best tiny blend lift below materiality threshold.
  - Decision: reject direct blend, keep FT as a logistic meta feature.

- Calibrated FT-Transformer focal/SWA variant:
  - `ft_transformer_calibrated_focal_swa`
  - Standalone CV `0.94284491`
  - Expanded DART meta-stack CV `0.96696908`
  - Decision: reject this exact calibration recipe; class-balanced focal loss plus stronger weight decay reduced standalone quality and did not improve the meta-stack.

- Photometry/redshift-only specialist:
  - Standalone CV `0.93679082`
  - Meta-stack CV `0.96692962`
  - Decision: reject; do not tune this exact branch without a new hypothesis.

- Categorical/redshift-only specialist:
  - Standalone CV `0.86026112`
  - Meta-stack CV `0.96693867`
  - Decision: reject; old CatBoost benefit is not explained by simple categorical/redshift-only signal.

- XGBoost and extra LightGBM diversity trials:
  - Produced small or zero blend gains below the `0.0001` usefulness threshold.
  - Decision: do not repeat near-duplicate full-feature GBDT diversity runs without a materially different objective/config.

## Next Experiment Direction

Priority candidates:

1. Different LightGBM objective/config family beyond the accepted conservative DART.
   - Examples: log-loss-first GBDT, class-weight variants, or another DART configuration with materially different dropout/regularization.
   - Must be evaluated as a meta feature.

2. Error-focused binary specialists.
   - Target hardest one-vs-one or one-vs-rest confusions.
   - Feed fold-safe OOF probabilities into the meta-learner.

3. Kernel approximation model.
   - Fold-safe `Nystroem` or `RBFSampler` plus logistic regression.
   - Proceed only if runtime is reasonable and OOF errors differ materially.

4. Further FT-Transformer work only with a materially different hypothesis.
   - Do not repeat the focal/SWA recipe.
   - Consider only if diagnostics point to a specific calibration or class-confusion failure mode.

## Guardrails

- Never use `id` as a feature.
- Keep all experiments on the same fixed stratified 5 folds.
- Save OOF and averaged test probability artifacts for any base candidate worth auditing.
- Accept only if mean 5-fold CV balanced accuracy beats `0.96700608`.
- Log meaningful accept/reject decisions here and append the run line to `submissions/experiment_log.txt`.
