| dataset | reference_method | comparison_method | n_pairs | enough_pairs | mean_diff_reference_minus_candidate | ci95_low | ci95_high | t_statistic | t_p_value | wilcoxon_statistic | wilcoxon_p_value | raw_p_value | holm_adjusted_p_value | cohens_d | significant_at_alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wisconsin Breast Cancer | NSGA-II (DEAP) | PCA + RandomForest | 2.0 | True | 0.0022 | -0.1374 | 0.1417 | 0.1979 | 0.8756 | 1.0 | 1.0 | 1.0 | 1.0 | 0.1399 | False |
| Wisconsin Breast Cancer | NSGA-II (DEAP) | RFE + RandomForest | 2.0 | True | 0.0 | 0.0 | 0.0 | nan | nan | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | False |
| Wisconsin Breast Cancer | NSGA-II (DEAP) | RF Importance + RandomForest | 2.0 | True | -0.0044 | -0.0045 | -0.0043 | -455.0 | 0.0014 | 0.0 | 0.5 | 0.5 | 1.0 | -321.7336 | False |
| Wisconsin Breast Cancer | NSGA-II (DEAP) | Grid Search RF | 2.0 | True | -0.0044 | -0.0045 | -0.0043 | -455.0 | 0.0014 | 0.0 | 0.5 | 0.5 | 1.0 | -321.7336 | False |
| Wisconsin Breast Cancer | NSGA-II (DEAP) | Random Search RF | 2.0 | True | 0.0088 | -0.0473 | 0.0649 | 1.9934 | 0.296 | 0.0 | 0.5 | 0.5 | 1.0 | 1.4096 | False |