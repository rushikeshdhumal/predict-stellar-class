# Candidate Tradeoff Diagnostics

- Champion OOF balanced accuracy: `0.96712409`
- Champion OOF errors: `21976`
- Positive `balanced_accuracy_delta` means replacing the champion with that candidate only inside the listed pocket would improve OOF balanced accuracy, using full-OOF class denominators.
- Tables using `y_true` are diagnostic only; deployable pockets use champion prediction, champion confidence, and original non-ID features.

## Candidate Summary

| candidate | candidate_score | score_delta | rescues | breaks | net_rows | GALAXY_rescue | GALAXY_break | GALAXY_recall_delta | QSO_rescue | QSO_break | QSO_recall_delta | STAR_rescue | STAR_break | STAR_recall_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| soft_residual_reliability | 0.96711546 | -0.00000863 | 44 | 125 | -81 | 12 | 116 | -0.00027551 | 14 | 6 | 0.00006829 | 18 | 3 | 0.00018133 |
| rich_lgbm_no_context | 0.96710695 | -0.00001714 | 444 | 531 | -87 | 306 | 421 | -0.00030465 | 63 | 39 | 0.00020488 | 75 | 71 | 0.00004835 |
| rich_lgbm_more_trees | 0.96707575 | -0.00004834 | 504 | 581 | -77 | 353 | 437 | -0.00022253 | 60 | 58 | 0.00001707 | 91 | 86 | 0.00006044 |
| rich_lgbm_conservative | 0.96706995 | -0.00005414 | 431 | 543 | -112 | 295 | 428 | -0.00035234 | 61 | 43 | 0.00015366 | 75 | 72 | 0.00003627 |

## Rescues And Breaks By True Class

| candidate | true_class | rescues | breaks | net |
|---|---|---|---|---|
| rich_lgbm_no_context | GALAXY | 306 | 421 | -115 |
| rich_lgbm_no_context | QSO | 63 | 39 | 24 |
| rich_lgbm_no_context | STAR | 75 | 71 | 4 |
| rich_lgbm_no_context | GALAXY_as_QSO | 38 | 0 | 38 |
| rich_lgbm_no_context | GALAXY_as_STAR | 268 | 0 | 268 |
| rich_lgbm_no_context | QSO_as_GALAXY | 45 | 0 | 45 |
| rich_lgbm_no_context | QSO_as_STAR | 18 | 0 | 18 |
| rich_lgbm_no_context | STAR_as_GALAXY | 62 | 0 | 62 |
| rich_lgbm_no_context | STAR_as_QSO | 13 | 0 | 13 |
| soft_residual_reliability | GALAXY | 12 | 116 | -104 |
| soft_residual_reliability | QSO | 14 | 6 | 8 |
| soft_residual_reliability | STAR | 18 | 3 | 15 |
| soft_residual_reliability | GALAXY_as_QSO | 1 | 0 | 1 |
| soft_residual_reliability | GALAXY_as_STAR | 11 | 0 | 11 |
| soft_residual_reliability | QSO_as_GALAXY | 14 | 0 | 14 |
| soft_residual_reliability | QSO_as_STAR | 0 | 0 | 0 |
| soft_residual_reliability | STAR_as_GALAXY | 17 | 0 | 17 |
| soft_residual_reliability | STAR_as_QSO | 1 | 0 | 1 |
| rich_lgbm_more_trees | GALAXY | 353 | 437 | -84 |
| rich_lgbm_more_trees | QSO | 60 | 58 | 2 |
| rich_lgbm_more_trees | STAR | 91 | 86 | 5 |
| rich_lgbm_more_trees | GALAXY_as_QSO | 71 | 0 | 71 |
| rich_lgbm_more_trees | GALAXY_as_STAR | 282 | 0 | 282 |
| rich_lgbm_more_trees | QSO_as_GALAXY | 40 | 0 | 40 |
| rich_lgbm_more_trees | QSO_as_STAR | 20 | 0 | 20 |
| rich_lgbm_more_trees | STAR_as_GALAXY | 74 | 0 | 74 |
| rich_lgbm_more_trees | STAR_as_QSO | 17 | 0 | 17 |
| rich_lgbm_conservative | GALAXY | 295 | 428 | -133 |
| rich_lgbm_conservative | QSO | 61 | 43 | 18 |
| rich_lgbm_conservative | STAR | 75 | 72 | 3 |
| rich_lgbm_conservative | GALAXY_as_QSO | 49 | 0 | 49 |
| rich_lgbm_conservative | GALAXY_as_STAR | 246 | 0 | 246 |
| rich_lgbm_conservative | QSO_as_GALAXY | 39 | 0 | 39 |
| rich_lgbm_conservative | QSO_as_STAR | 22 | 0 | 22 |
| rich_lgbm_conservative | STAR_as_GALAXY | 60 | 0 | 60 |
| rich_lgbm_conservative | STAR_as_QSO | 15 | 0 | 15 |

## Top Deployable Candidate Pockets

| candidate | group | pocket | rows | rescues | breaks | net_rows | balanced_accuracy_delta |
|---|---|---|---|---|---|---|---|
| rich_lgbm_more_trees | champion_pred + g_r_bucket | GALAXY | (1.741, 1.964] | 56201 | 14 | 23 | -9 | 0.00003610 |
| rich_lgbm_no_context | champion_pred + g_r_bucket | GALAXY | (0.53, 0.781] | 25598 | 26 | 60 | -34 | 0.00003402 |
| rich_lgbm_more_trees | champion_pred + g_r_bucket | GALAXY | (0.53, 0.781] | 25598 | 25 | 59 | -34 | 0.00003206 |
| rich_lgbm_more_trees | champion_pred + redshift_bucket | GALAXY | (0.128, 0.244] | 42061 | 27 | 90 | -63 | 0.00002932 |
| rich_lgbm_conservative | champion_pred + g_r_bucket | GALAXY | (0.53, 0.781] | 25598 | 25 | 63 | -38 | 0.00002735 |
| rich_lgbm_more_trees | champion_pred + champion_margin_bucket | GALAXY | (-8.324000000000001e-05, 0.7986] | 33423 | 114 | 437 | -323 | 0.00002611 |
| rich_lgbm_more_trees | champion_pred + champion_entropy_bucket | GALAXY | (0.353, 1.099] | 33127 | 114 | 437 | -323 | 0.00002611 |
| rich_lgbm_conservative | champion_pred + spectral_type | STAR | A/F | 39176 | 78 | 17 | 61 | 0.00002589 |
| rich_lgbm_conservative | champion_pred + spectral_population | GALAXY | G/K_Blue_Cloud | 25430 | 25 | 64 | -39 | 0.00002528 |
| rich_lgbm_conservative | champion_pred + g_r_bucket | GALAXY | (1.741, 1.964] | 56201 | 11 | 22 | -11 | 0.00002490 |
| rich_lgbm_more_trees | champion_pred + spectral_type | GALAXY | G/K | 57660 | 39 | 126 | -87 | 0.00002457 |
| rich_lgbm_no_context | champion_pred + galaxy_population | GALAXY | Blue_Cloud | 80383 | 55 | 173 | -118 | 0.00002386 |
| rich_lgbm_no_context | champion_pred + g_r_bucket | GALAXY | (1.741, 1.964] | 56201 | 10 | 19 | -9 | 0.00002352 |
| rich_lgbm_more_trees | champion_pred + u_g_bucket | GALAXY | (1.924, 2.282] | 49691 | 17 | 47 | -30 | 0.00002345 |
| rich_lgbm_more_trees | champion_pred + g_r_bucket | STAR | (0.312, 0.53] | 19882 | 54 | 9 | 45 | 0.00002319 |
| rich_lgbm_more_trees | champion_pred + u_g_bucket | GALAXY | (2.948, 7.523] | 50335 | 15 | 40 | -25 | 0.00002275 |
| rich_lgbm_no_context | champion_pred + redshift_bucket | GALAXY | (0.498, 0.627] | 54554 | 10 | 7 | 3 | 0.00002227 |
| rich_lgbm_no_context | champion_pred + spectral_population | GALAXY | G/K_Blue_Cloud | 25430 | 24 | 65 | -41 | 0.00002155 |
| rich_lgbm_no_context | champion_pred + u_g_bucket | STAR | (2.282, 2.948] | 6503 | 46 | 5 | 41 | 0.00002047 |
| rich_lgbm_more_trees | champion_pred + spectral_population | GALAXY | G/K_Blue_Cloud | 25430 | 23 | 65 | -42 | 0.00001989 |
| rich_lgbm_no_context | champion_pred + spectral_type | GALAXY | G/K | 57660 | 39 | 131 | -92 | 0.00001897 |
| rich_lgbm_no_context | champion_pred + g_r_bucket | STAR | (0.312, 0.53] | 19882 | 49 | 8 | 41 | 0.00001888 |
| rich_lgbm_conservative | champion_pred + spectral_population | STAR | A/F_Red_Sequence | 3871 | 20 | 0 | 20 | 0.00001766 |
| rich_lgbm_no_context | champion_pred + g_r_bucket | STAR | (1.343, 1.552] | 3143 | 29 | 2 | 27 | 0.00001755 |
| rich_lgbm_conservative | champion_pred + u_g_bucket | STAR | (2.282, 2.948] | 6503 | 42 | 5 | 37 | 0.00001694 |
| rich_lgbm_conservative | champion_pred + g_r_bucket | STAR | (0.312, 0.53] | 19882 | 49 | 9 | 40 | 0.00001682 |
| rich_lgbm_no_context | champion_pred + spectral_population | STAR | A/F_Red_Sequence | 3871 | 19 | 0 | 19 | 0.00001678 |
| rich_lgbm_more_trees | champion_pred + galaxy_population | GALAXY | Red_Sequence | 284282 | 65 | 266 | -201 | 0.00001637 |
| rich_lgbm_no_context | champion_pred + u_g_bucket | STAR | (1.625, 1.924] | 8496 | 41 | 5 | 36 | 0.00001606 |
| rich_lgbm_no_context | champion_pred + spectral_type | STAR | A/F | 39176 | 71 | 16 | 55 | 0.00001589 |
| rich_lgbm_conservative | champion_pred + redshift_bucket | GALAXY | (0.498, 0.627] | 54554 | 8 | 8 | 0 | 0.00001570 |
| rich_lgbm_more_trees | champion_pred + u_g_bucket | STAR | (1.625, 1.924] | 8496 | 45 | 6 | 39 | 0.00001556 |
| rich_lgbm_no_context | champion_pred + galaxy_population | STAR | Red_Sequence | 26596 | 179 | 36 | 143 | 0.00001497 |
| rich_lgbm_no_context | champion_pred + u_g_bucket | GALAXY | (0.435, 0.758] | 24083 | 10 | 18 | -8 | 0.00001493 |
| rich_lgbm_no_context | champion_pred + spectral_type | STAR | M | 14608 | 131 | 26 | 105 | 0.00001484 |
| rich_lgbm_more_trees | champion_pred + spectral_population | STAR | A/F_Red_Sequence | 3871 | 19 | 1 | 18 | 0.00001471 |
| rich_lgbm_no_context | champion_pred + redshift_bucket | GALAXY | (0.128, 0.244] | 42061 | 23 | 87 | -64 | 0.00001467 |
| rich_lgbm_conservative | champion_pred + u_g_bucket | GALAXY | (1.333, 1.625] | 40168 | 13 | 42 | -29 | 0.00001411 |
| rich_lgbm_more_trees | champion_pred + g_r_bucket | STAR | (1.343, 1.552] | 3143 | 25 | 2 | 23 | 0.00001402 |
| rich_lgbm_more_trees | champion_pred + u_g_bucket | GALAXY | (1.333, 1.625] | 40168 | 14 | 47 | -33 | 0.00001373 |

## Champion Error Rescue Pockets

| candidate | group | pocket | champion_errors | rescues | rescue_rate |
|---|---|---|---|---|---|
| rich_lgbm_more_trees | champion_error_pair + champion_margin_bucket | GALAXY_as_STAR | (-8.324000000000001e-05, 0.7986] | 8245 | 282 | 0.03420255 |
| rich_lgbm_no_context | champion_error_pair + champion_margin_bucket | GALAXY_as_STAR | (-8.324000000000001e-05, 0.7986] | 8245 | 268 | 0.03250455 |
| rich_lgbm_conservative | champion_error_pair + champion_margin_bucket | GALAXY_as_STAR | (-8.324000000000001e-05, 0.7986] | 8245 | 246 | 0.02983626 |
| rich_lgbm_more_trees | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (0.0501, 0.128] | 4242 | 124 | 0.02923149 |
| rich_lgbm_no_context | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (0.0501, 0.128] | 4242 | 119 | 0.02805281 |
| rich_lgbm_no_context | true_class + spectral_population | GALAXY | M_Red_Sequence | 4096 | 114 | 0.02783203 |
| rich_lgbm_no_context | champion_error_pair + spectral_population | GALAXY_as_STAR | M_Red_Sequence | 4002 | 113 | 0.02823588 |
| rich_lgbm_conservative | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (0.0501, 0.128] | 4242 | 113 | 0.02663838 |
| rich_lgbm_more_trees | true_class + spectral_population | GALAXY | M_Red_Sequence | 4096 | 110 | 0.02685547 |
| rich_lgbm_more_trees | champion_error_pair + spectral_population | GALAXY_as_STAR | M_Red_Sequence | 4002 | 109 | 0.02723638 |
| rich_lgbm_conservative | true_class + spectral_population | GALAXY | M_Red_Sequence | 4096 | 92 | 0.02246094 |
| rich_lgbm_conservative | champion_error_pair + spectral_population | GALAXY_as_STAR | M_Red_Sequence | 4002 | 91 | 0.02273863 |
| rich_lgbm_more_trees | true_class + spectral_population | GALAXY | A/F_Blue_Cloud | 4993 | 82 | 0.01642299 |
| rich_lgbm_more_trees | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (-0.01097, 0.0501] | 3941 | 81 | 0.02055316 |
| rich_lgbm_no_context | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (-0.01097, 0.0501] | 3941 | 76 | 0.01928445 |
| rich_lgbm_more_trees | champion_error_pair + champion_margin_bucket | STAR_as_GALAXY | (-8.324000000000001e-05, 0.7986] | 1703 | 74 | 0.04345273 |
| rich_lgbm_more_trees | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (0.128, 0.244] | 2594 | 73 | 0.02814187 |
| rich_lgbm_more_trees | champion_error_pair + champion_margin_bucket | GALAXY_as_QSO | (-8.324000000000001e-05, 0.7986] | 3694 | 71 | 0.01922036 |
| rich_lgbm_no_context | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (0.128, 0.244] | 2594 | 68 | 0.02621434 |
| rich_lgbm_conservative | true_class + spectral_population | GALAXY | A/F_Blue_Cloud | 4993 | 68 | 0.01361907 |
| rich_lgbm_conservative | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (0.128, 0.244] | 2594 | 66 | 0.02544333 |
| rich_lgbm_conservative | champion_error_pair + redshift_bucket | GALAXY_as_STAR | (-0.01097, 0.0501] | 3941 | 63 | 0.01598579 |
| rich_lgbm_no_context | champion_error_pair + champion_margin_bucket | STAR_as_GALAXY | (-8.324000000000001e-05, 0.7986] | 1703 | 62 | 0.03640634 |
| rich_lgbm_no_context | true_class + spectral_population | GALAXY | A/F_Blue_Cloud | 4993 | 61 | 0.01221710 |
| rich_lgbm_conservative | champion_error_pair + champion_margin_bucket | STAR_as_GALAXY | (-8.324000000000001e-05, 0.7986] | 1703 | 60 | 0.03523194 |
| rich_lgbm_more_trees | true_class + spectral_population | GALAXY | G/K_Red_Sequence | 1745 | 55 | 0.03151862 |
| rich_lgbm_more_trees | champion_error_pair + spectral_population | GALAXY_as_STAR | G/K_Red_Sequence | 1619 | 53 | 0.03273626 |
| rich_lgbm_more_trees | true_class + spectral_population | GALAXY | G/K_Blue_Cloud | 3379 | 52 | 0.01538917 |
| rich_lgbm_more_trees | champion_error_pair + spectral_population | GALAXY_as_STAR | A/F_Blue_Cloud | 2323 | 49 | 0.02109341 |
| rich_lgbm_conservative | champion_error_pair + champion_margin_bucket | GALAXY_as_QSO | (-8.324000000000001e-05, 0.7986] | 3694 | 49 | 0.01326475 |
| rich_lgbm_no_context | true_class + spectral_population | GALAXY | G/K_Red_Sequence | 1745 | 47 | 0.02693410 |
| rich_lgbm_no_context | champion_error_pair + spectral_population | GALAXY_as_STAR | G/K_Red_Sequence | 1619 | 46 | 0.02841260 |
| rich_lgbm_no_context | champion_error_pair + champion_margin_bucket | QSO_as_GALAXY | (-8.324000000000001e-05, 0.7986] | 1069 | 45 | 0.04209542 |
| rich_lgbm_conservative | champion_error_pair + spectral_population | GALAXY_as_STAR | A/F_Blue_Cloud | 2323 | 45 | 0.01937150 |
| rich_lgbm_conservative | true_class + spectral_population | GALAXY | G/K_Blue_Cloud | 3379 | 45 | 0.01331755 |
| rich_lgbm_more_trees | champion_error_pair + spectral_population | STAR_as_GALAXY | M_Red_Sequence | 1172 | 44 | 0.03754266 |
| rich_lgbm_more_trees | true_class + spectral_population | STAR | M_Red_Sequence | 1175 | 44 | 0.03744681 |
| rich_lgbm_no_context | champion_error_pair + spectral_population | GALAXY_as_STAR | A/F_Blue_Cloud | 2323 | 43 | 0.01851055 |
| rich_lgbm_conservative | true_class + spectral_population | GALAXY | G/K_Red_Sequence | 1745 | 42 | 0.02406877 |
| rich_lgbm_conservative | champion_error_pair + spectral_population | GALAXY_as_STAR | G/K_Red_Sequence | 1619 | 41 | 0.02532427 |
