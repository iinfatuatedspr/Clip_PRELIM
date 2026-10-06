# Preliminary experiment report: `vitl14_openai`

- model: `ViT-L-14-quickgelu` / `openai`
- gpu: NVIDIA GeForce RTX 4060 Ti, torch 2.14.0+cu130, open_clip 3.3.0
- prompts.yaml sha256: `87e97227b686da6c`
- updated: 2026-09-29T12:02:43
- experiments: {"E1": {"split": "test", "n_images": 632, "conditions": ["label", "term_literal", "attr", "label_attr", "llm", "label_llm", "waffle"]}, "E2": {"n_images": 537, "sets": ["adjective", "width", "extent", "manual", "random_control"]}, "E3": {"n_images": 537, "conditions": ["label", "term_literal", "label_attr", "label_llm", "waffle"]}}

## E1 CODEBRIM: AUROC [95% bootstrap CI]

### normal prompts = `basic`

| condition | T1_anomaly | macro_T2 | crack | spallation | efflorescence | exposed_bars | corrosion_stain |
|---|---|---|---|---|---|---|---|
| attr | 0.760 [0.715, 0.801] | 0.661 [0.638, 0.683] | 0.720 [0.679, 0.762] | 0.386 [0.327, 0.446] | 0.814 [0.773, 0.853] | 0.896 [0.863, 0.926] | 0.488 [0.421, 0.550] |
| label | 0.815 [0.778, 0.849] | 0.705 [0.682, 0.727] | 0.781 [0.737, 0.824] | 0.511 [0.459, 0.562] | 0.787 [0.743, 0.830] | 0.671 [0.615, 0.728] | 0.775 [0.729, 0.817] |
| label_attr | 0.825 [0.786, 0.858] | 0.737 [0.713, 0.759] | 0.838 [0.800, 0.874] | 0.556 [0.500, 0.610] | 0.811 [0.770, 0.852] | 0.779 [0.728, 0.829] | 0.701 [0.645, 0.750] |
| label_llm | 0.839 [0.802, 0.871] | 0.732 [0.708, 0.755] | 0.815 [0.775, 0.853] | 0.522 [0.463, 0.578] | 0.824 [0.782, 0.862] | 0.798 [0.750, 0.846] | 0.700 [0.644, 0.747] |
| llm | 0.774 [0.729, 0.817] | 0.669 [0.646, 0.690] | 0.697 [0.654, 0.739] | 0.371 [0.309, 0.430] | 0.835 [0.796, 0.870] | 0.907 [0.877, 0.936] | 0.532 [0.463, 0.590] |
| term_literal | 0.780 [0.741, 0.816] | 0.642 [0.623, 0.661] | 0.811 [0.775, 0.843] | 0.264 [0.215, 0.315] | 0.616 [0.564, 0.666] | 0.826 [0.786, 0.867] | 0.695 [0.648, 0.741] |
| waffle | 0.841 [0.806, 0.873] | 0.692 [0.668, 0.716] | 0.829 [0.790, 0.865] | 0.487 [0.434, 0.542] | 0.750 [0.704, 0.796] | 0.670 [0.612, 0.731] | 0.725 [0.671, 0.771] |

### normal prompts = `extended`

| condition | T1_anomaly | macro_T2 | crack | spallation | efflorescence | exposed_bars | corrosion_stain |
|---|---|---|---|---|---|---|---|
| attr | 0.805 [0.769, 0.839] | 0.671 [0.648, 0.693] | 0.731 [0.689, 0.773] | 0.407 [0.348, 0.466] | 0.812 [0.770, 0.850] | 0.901 [0.869, 0.931] | 0.506 [0.439, 0.567] |
| label | 0.838 [0.805, 0.870] | 0.726 [0.704, 0.749] | 0.761 [0.714, 0.807] | 0.579 [0.527, 0.628] | 0.792 [0.746, 0.836] | 0.692 [0.636, 0.748] | 0.807 [0.764, 0.847] |
| label_attr | 0.862 [0.830, 0.890] | 0.757 [0.733, 0.779] | 0.815 [0.772, 0.857] | 0.617 [0.563, 0.670] | 0.815 [0.773, 0.855] | 0.799 [0.751, 0.846] | 0.740 [0.687, 0.787] |
| label_llm | 0.876 [0.847, 0.902] | 0.753 [0.729, 0.775] | 0.798 [0.753, 0.839] | 0.583 [0.524, 0.638] | 0.829 [0.788, 0.867] | 0.816 [0.768, 0.861] | 0.739 [0.686, 0.786] |
| llm | 0.818 [0.784, 0.852] | 0.677 [0.654, 0.698] | 0.707 [0.666, 0.751] | 0.387 [0.324, 0.445] | 0.835 [0.795, 0.870] | 0.911 [0.882, 0.939] | 0.545 [0.476, 0.603] |
| term_literal | 0.788 [0.749, 0.825] | 0.658 [0.639, 0.678] | 0.789 [0.750, 0.825] | 0.300 [0.246, 0.352] | 0.621 [0.570, 0.670] | 0.852 [0.812, 0.889] | 0.729 [0.682, 0.775] |
| waffle | 0.883 [0.853, 0.907] | 0.713 [0.689, 0.737] | 0.805 [0.764, 0.846] | 0.550 [0.496, 0.603] | 0.758 [0.710, 0.802] | 0.692 [0.636, 0.751] | 0.762 [0.713, 0.807] |

### E1 paired differences (A - B), main targets

| normal_set | A | B | question | target | diff [CI] | p_boot |
|---|---|---|---|---|---|---|
| basic | term_literal | label | 直译点検要领用语 vs 通用英语标签 | T1_anomaly | -0.035 [-0.056, -0.014] | 0.000 |
| basic | term_literal | label | 直译点検要领用语 vs 通用英语标签 | macro_T2 | -0.063 [-0.085, -0.040] | 0.000 |
| basic | attr | label | 只用视觉属性描述(无类别名) vs 标签 | T1_anomaly | -0.054 [-0.089, -0.021] | 0.000 |
| basic | attr | label | 只用视觉属性描述(无类别名) vs 标签 | macro_T2 | -0.044 [-0.069, -0.020] | 0.000 |
| basic | label_attr | label | 标签+视觉属性 vs 标签 | T1_anomaly | 0.010 [-0.007, 0.028] | 0.338 |
| basic | label_attr | label | 标签+视觉属性 vs 标签 | macro_T2 | 0.032 [0.021, 0.042] | 0.000 |
| basic | label_attr | waffle | 视觉属性的内容 vs 同数量随机词(集成效应对照) | T1_anomaly | -0.017 [-0.028, -0.006] | 0.000 |
| basic | label_attr | waffle | 视觉属性的内容 vs 同数量随机词(集成效应对照) | macro_T2 | 0.044 [0.035, 0.054] | 0.000 |
| basic | label_llm | label_attr | LLM言い換え扩展 vs 人手属性描述 | T1_anomaly | 0.014 [0.010, 0.019] | 0.000 |
| basic | label_llm | label_attr | LLM言い換え扩展 vs 人手属性描述 | macro_T2 | -0.005 [-0.010, -0.000] | 0.048 |
| basic | label_llm | waffle | LLM言い換え vs 随机词对照 | T1_anomaly | -0.003 [-0.013, 0.007] | 0.608 |
| basic | label_llm | waffle | LLM言い換え vs 随机词对照 | macro_T2 | 0.040 [0.028, 0.052] | 0.000 |
| basic | llm | attr | LLM言い換え vs 人手属性描述(均无类别名) | T1_anomaly | 0.013 [0.004, 0.023] | 0.010 |
| basic | llm | attr | LLM言い換え vs 人手属性描述(均无类别名) | macro_T2 | 0.008 [-0.000, 0.016] | 0.068 |
| extended | term_literal | label | 直译点検要领用语 vs 通用英语标签 | T1_anomaly | -0.050 [-0.075, -0.029] | 0.000 |
| extended | term_literal | label | 直译点検要领用语 vs 通用英语标签 | macro_T2 | -0.068 [-0.091, -0.045] | 0.000 |
| extended | attr | label | 只用视觉属性描述(无类别名) vs 标签 | T1_anomaly | -0.033 [-0.070, 0.006] | 0.102 |
| extended | attr | label | 只用视觉属性描述(无类别名) vs 标签 | macro_T2 | -0.055 [-0.081, -0.031] | 0.000 |
| extended | label_attr | label | 标签+视觉属性 vs 标签 | T1_anomaly | 0.024 [0.003, 0.044] | 0.020 |
| extended | label_attr | label | 标签+视觉属性 vs 标签 | macro_T2 | 0.031 [0.020, 0.042] | 0.000 |
| extended | label_attr | waffle | 视觉属性的内容 vs 同数量随机词(集成效应对照) | T1_anomaly | -0.020 [-0.034, -0.009] | 0.000 |
| extended | label_attr | waffle | 视觉属性的内容 vs 同数量随机词(集成效应对照) | macro_T2 | 0.044 [0.034, 0.054] | 0.000 |
| extended | label_llm | label_attr | LLM言い換え扩展 vs 人手属性描述 | T1_anomaly | 0.014 [0.010, 0.019] | 0.000 |
| extended | label_llm | label_attr | LLM言い換え扩展 vs 人手属性描述 | macro_T2 | -0.005 [-0.010, -0.000] | 0.050 |
| extended | label_llm | waffle | LLM言い換え vs 随机词对照 | T1_anomaly | -0.006 [-0.018, 0.005] | 0.292 |
| extended | label_llm | waffle | LLM言い換え vs 随机词对照 | macro_T2 | 0.039 [0.028, 0.052] | 0.000 |
| extended | llm | attr | LLM言い換え vs 人手属性描述(均无类别名) | T1_anomaly | 0.012 [-0.001, 0.025] | 0.060 |
| extended | llm | attr | LLM言い換え vs 人手属性描述(均无类别名) | macro_T2 | 0.006 [-0.002, 0.014] | 0.190 |

### E1 hard-negative normal prompts (extended - basic), T1 anomaly

| condition | auroc_basic | auroc_extended | diff | ci_lo | ci_hi | p_boot |
|---|---|---|---|---|---|---|
| label | 0.815 | 0.838 | 0.023 | 0.002 | 0.045 | 0.034 |
| term_literal | 0.780 | 0.788 | 0.008 | -0.015 | 0.034 | 0.484 |
| attr | 0.760 | 0.805 | 0.045 | 0.013 | 0.077 | 0.010 |
| label_attr | 0.825 | 0.862 | 0.037 | 0.018 | 0.060 | 0.000 |
| llm | 0.774 | 0.818 | 0.044 | 0.015 | 0.074 | 0.008 |
| label_llm | 0.839 | 0.876 | 0.037 | 0.018 | 0.059 | 0.000 |
| waffle | 0.841 | 0.883 | 0.041 | 0.021 | 0.063 | 0.000 |

### E1 confusion on single-label defect images (condition `label_attr`; rows=true, cols=top-1)

| true | crack | spallation | efflorescence | exposed_bars | corrosion_stain |
|---|---|---|---|---|---|
| crack | 104 | 10 | 7 | 1 | 5 |
| spallation | 13 | 1 | 2 | 0 | 1 |
| efflorescence | 27 | 1 | 72 | 3 | 16 |
| exposed_bars | 4 | 1 | 0 | 3 | 12 |
| corrosion_stain | 3 | 0 | 0 | 4 | 21 |

## E2 DeepCrack: ordinality of degree expressions

Text side (`t_moderate` in (0,1) means 'moderate' lies between 'slight' and 'severe'):

| set | t_moderate | ordered | cos_slight_moderate | cos_moderate_severe | cos_slight_severe |
|---|---|---|---|---|---|
| adjective | 0.404 | True | 0.989 | 0.986 | 0.985 |
| width | 0.906 | True | 0.963 | 0.989 | 0.969 |
| extent | 0.447 | True | 0.947 | 0.935 | 0.886 |
| manual | 0.551 | True | 0.971 | 0.976 | 0.952 |
| random_control | 0.396 | True | 0.959 | 0.948 | 0.946 |

Image side (Spearman between expected level and mask proxies):

| set | area_ratio | length | width_p90 |
|---|---|---|---|
| adjective | 0.216 [0.132, 0.303] | 0.342 [0.269, 0.418] | 0.104 [0.015, 0.189] |
| extent | 0.062 [-0.030, 0.148] | 0.331 [0.243, 0.414] | -0.123 [-0.214, -0.040] |
| manual | 0.176 [0.095, 0.255] | 0.435 [0.358, 0.509] | -0.013 [-0.095, 0.067] |
| random_control | 0.112 [0.032, 0.194] | 0.166 [0.081, 0.251] | 0.093 [0.005, 0.178] |
| width | -0.165 [-0.239, -0.084] | -0.100 [-0.183, -0.019] | -0.115 [-0.191, -0.031] |

## E3 DeepCrack: window-based localisation

| condition | pixel_auroc | ci_lo | ci_hi | pixel_ap | mean_image_auroc | n_images | pos_rate |
|---|---|---|---|---|---|---|---|
| label | 0.822 | 0.811 | 0.835 | 0.109 | 0.858 | 537 | 0.036 |
| term_literal | 0.825 | 0.815 | 0.839 | 0.114 | 0.856 | 537 | 0.036 |
| label_attr | 0.823 | 0.812 | 0.835 | 0.107 | 0.858 | 537 | 0.036 |
| label_llm | 0.829 | 0.818 | 0.841 | 0.110 | 0.859 | 537 | 0.036 |
| waffle | 0.833 | 0.823 | 0.845 | 0.110 | 0.862 | 537 | 0.036 |

| A | B | diff | ci_lo | ci_hi | p_boot |
|---|---|---|---|---|---|
| label_attr | label | 0.001 | -0.007 | 0.008 | 0.760 |
| label_attr | waffle | -0.010 | -0.013 | -0.007 | 0.000 |
| label_llm | waffle | -0.004 | -0.007 | -0.001 | 0.000 |
| term_literal | label | 0.003 | -0.004 | 0.010 | 0.500 |
| label_llm | label_attr | 0.006 | 0.004 | 0.008 | 0.000 |

## E0 term frequency in captions

| term | captions_with_term | per_million | corpus | n_captions |
|---|---|---|---|---|
| crack | 2894 | 232.948 | cc12m.tsv | 12423374 |
| peeling | 1670 | 134.424 | cc12m.tsv | 12423374 |
| cracking | 774 | 62.302 | cc12m.tsv | 12423374 |
| rebar | 283 | 22.780 | cc12m.tsv | 12423374 |
| water leakage | 37 | 2.978 | cc12m.tsv | 12423374 |
| hairline crack | 27 | 2.173 | cc12m.tsv | 12423374 |
| efflorescence | 11 | 0.885 | cc12m.tsv | 12423374 |
| spalling | 7 | 0.563 | cc12m.tsv | 12423374 |
| free lime | 6 | 0.483 | cc12m.tsv | 12423374 |
| delamination | 6 | 0.483 | cc12m.tsv | 12423374 |
| rust stain | 4 | 0.322 | cc12m.tsv | 12423374 |
| exposed rebar | 2 | 0.161 | cc12m.tsv | 12423374 |
| spallation | 2 | 0.161 | cc12m.tsv | 12423374 |
| corrosion stain | 0 | 0.000 | cc12m.tsv | 12423374 |
| exposed reinforcement | 0 | 0.000 | cc12m.tsv | 12423374 |
| exposed reinforcement bars | 0 | 0.000 | cc12m.tsv | 12423374 |
| rust juice | 0 | 0.000 | cc12m.tsv | 12423374 |
| iron bar exposure | 0 | 0.000 | cc12m.tsv | 12423374 |

## Figures
- figures/e1_false_negatives.png
- figures/e1_false_positives.png
- figures/e2_scatter.png
- figures/e3_heatmaps.png

## Observations
See `observations.md` (written by the experimenter).
