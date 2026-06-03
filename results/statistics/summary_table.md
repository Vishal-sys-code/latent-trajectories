# Statistical Summary

## Bootstrap Confidence Intervals

| metric                               | category       |       mean |        std |   ci_lower |   ci_upper |
|:-------------------------------------|:---------------|-----------:|-----------:|-----------:|-----------:|
| length_norm                          | animals        |   3.33812  |  0.211551  |   3.19985  |    3.46395 |
| length_norm                          | objects        |   3.21177  |  0.18908   |   3.08947  |    3.32934 |
| length_norm                          | reasoning      |   3.72874  |  0.093957  |   3.67086  |    3.78615 |
| curvature                            | animals        |   1.0893   |  0.271912  |   0.892298 |    1.23252 |
| curvature                            | objects        |   0.942446 |  0.318696  |   0.743653 |    1.13628 |
| curvature                            | reasoning      |   1.26927  |  0.0366674 |   1.24574  |    1.29157 |
| convergence_between_dist_final_layer | global_between | 104.757    | 71.4529    | 100.149    |  109.417   |
| convergence_within_dist_final_layer  | global_within  |  87.634    | 61.5763    |  80.7799   |   94.5042  |

## Pairwise Significance Tests

| metric      | comparison           |     p_value |   corrected_p_value |   effect_size |
|:------------|:---------------------|------------:|--------------------:|--------------:|
| length_norm | animals_vs_objects   | 0.241322    |         0.241322    |      0.597484 |
| length_norm | animals_vs_reasoning | 0.00058284  |         0.00116568  |     -2.26403  |
| length_norm | objects_vs_reasoning | 0.000182672 |         0.000548015 |     -3.28506  |