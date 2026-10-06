# Preliminary experiment report: `vitb16_openai`

- model: `ViT-B-16-quickgelu` / `openai`
- gpu: NVIDIA GeForce RTX 4060 Ti, torch 2.14.0+cu130, open_clip 3.3.0
- prompts.yaml sha256: `87e97227b686da6c`
- updated: 2026-09-29T12:06:36
- experiments: {"E1": {"split": "test", "n_images": 632, "conditions": ["label", "term_literal", "attr", "label_attr", "llm", "label_llm", "waffle"]}, "E2": {"n_images": 537, "sets": ["adjective", "width", "extent", "manual", "random_control"]}, "E3": {"n_images": 537, "conditions": ["label", "term_literal", "label_attr", "label_llm", "waffle"]}}

## E1 CODEBRIM: AUROC [95% bootstrap CI]

### normal prompts = `basic`

| condition | T1_anomaly | macro_T2 | crack | spallation | efflorescence | exposed_bars | corrosion_stain |
|---|---|---|---|---|---|---|---|
| attr | 0.681 [0.634, 0.726] | 0.691 [0.669, 0.713] | 0.632 [0.586, 0.679] | 0.598 [0.545, 0.653] | 0.779 [0.735, 0.821] | 0.866 [0.832, 0.899] | 0.579 [0.513, 0.635] |
| label | 0.864 [0.829, 0.897] | 0.685 [0.663, 0.708] | 0.720 [0.669, 0.762] | 0.510 [0.453, 0.572] | 0.718 [0.669, 0.765] | 0.767 [0.718, 0.814] | 0.711 [0.660, 0.762] |
| label_attr | 0.850 [0.811, 0.884] | 0.728 [0.706, 0.749] | 0.808 [0.767, 0.844] | 0.543 [0.489, 0.603] | 0.748 [0.701, 0.791] | 0.826 [0.786, 0.866] | 0.713 [0.663, 0.762] |
| label_llm | 0.856 [0.819, 0.889] | 0.723 [0.701, 0.745] | 0.765 [0.723, 0.806] | 0.549 [0.496, 0.609] | 0.755 [0.706, 0.797] | 0.828 [0.788, 0.867] | 0.719 [0.668, 0.769] |
| llm | 0.720 [0.675, 0.763] | 0.680 [0.657, 0.701] | 0.598 [0.552, 0.643] | 0.555 [0.497, 0.613] | 0.781 [0.735, 0.823] | 0.892 [0.862, 0.920] | 0.573 [0.506, 0.630] |
| term_literal | 0.848 [0.815, 0.879] | 0.663 [0.644, 0.681] | 0.855 [0.815, 0.889] | 0.188 [0.143, 0.234] | 0.671 [0.628, 0.713] | 0.848 [0.803, 0.889] | 0.753 [0.707, 0.794] |
| waffle | 0.859 [0.823, 0.891] | 0.657 [0.633, 0.679] | 0.768 [0.720, 0.808] | 0.396 [0.344, 0.452] | 0.705 [0.658, 0.751] | 0.733 [0.680, 0.784] | 0.681 [0.629, 0.733] |

### normal prompts = `extended`

| condition | T1_anomaly | macro_T2 | crack | spallation | efflorescence | exposed_bars | corrosion_stain |
|---|---|---|---|---|---|---|---|
| attr | 0.728 [0.689, 0.768] | 0.705 [0.682, 0.726] | 0.643 [0.597, 0.691] | 0.624 [0.573, 0.676] | 0.777 [0.733, 0.821] | 0.871 [0.838, 0.903] | 0.608 [0.543, 0.663] |
| label | 0.869 [0.837, 0.899] | 0.719 [0.696, 0.741] | 0.703 [0.651, 0.747] | 0.599 [0.545, 0.654] | 0.741 [0.692, 0.787] | 0.785 [0.739, 0.832] | 0.765 [0.716, 0.811] |
| label_attr | 0.889 [0.860, 0.915] | 0.757 [0.736, 0.777] | 0.812 [0.771, 0.850] | 0.612 [0.560, 0.666] | 0.758 [0.710, 0.802] | 0.843 [0.805, 0.880] | 0.759 [0.708, 0.806] |
| label_llm | 0.895 [0.867, 0.920] | 0.750 [0.728, 0.771] | 0.780 [0.737, 0.821] | 0.607 [0.559, 0.662] | 0.766 [0.717, 0.808] | 0.843 [0.806, 0.880] | 0.755 [0.704, 0.804] |
| llm | 0.780 [0.741, 0.817] | 0.692 [0.670, 0.713] | 0.610 [0.565, 0.656] | 0.580 [0.525, 0.634] | 0.780 [0.733, 0.822] | 0.896 [0.867, 0.923] | 0.595 [0.529, 0.652] |
| term_literal | 0.834 [0.800, 0.864] | 0.691 [0.673, 0.710] | 0.838 [0.797, 0.871] | 0.266 [0.215, 0.320] | 0.666 [0.623, 0.706] | 0.871 [0.831, 0.908] | 0.816 [0.774, 0.851] |
| waffle | 0.903 [0.876, 0.928] | 0.686 [0.663, 0.708] | 0.774 [0.725, 0.812] | 0.466 [0.411, 0.524] | 0.716 [0.668, 0.761] | 0.751 [0.700, 0.802] | 0.722 [0.672, 0.770] |

### E1 paired differences (A - B), main targets

| normal_set | A | B | question | target | diff [CI] | p_boot |
|---|---|---|---|---|---|---|
| basic | term_literal | label | 直译点検要领用语 vs 通用英语标签 | T1_anomaly | -0.016 [-0.037, 0.008] | 0.202 |
| basic | term_literal | label | 直译点検要领用语 vs 通用英语标签 | macro_T2 | -0.022 [-0.046, -0.000] | 0.048 |
| basic | attr | label | 只用视觉属性描述(无类别名) vs 标签 | T1_anomaly | -0.183 [-0.222, -0.138] | 0.000 |
| basic | attr | label | 只用视觉属性描述(无类别名) vs 标签 | macro_T2 | 0.006 [-0.019, 0.029] | 0.658 |
| basic | label_attr | label | 标签+视觉属性 vs 标签 | T1_anomaly | -0.014 [-0.033, 0.006] | 0.200 |
| basic | label_attr | label | 标签+视觉属性 vs 标签 | macro_T2 | 0.043 [0.035, 0.051] | 0.000 |
| basic | label_attr | waffle | 视觉属性的内容 vs 同数量随机词(集成效应对照) | T1_anomaly | -0.009 [-0.019, 0.000] | 0.054 |
| basic | label_attr | waffle | 视觉属性的内容 vs 同数量随机词(集成效应对照) | macro_T2 | 0.071 [0.061, 0.081] | 0.000 |
| basic | label_llm | label_attr | LLM言い換え扩展 vs 人手属性描述 | T1_anomaly | 0.006 [0.002, 0.010] | 0.000 |
| basic | label_llm | label_attr | LLM言い換え扩展 vs 人手属性描述 | macro_T2 | -0.004 [-0.008, -0.001] | 0.026 |
| basic | label_llm | waffle | LLM言い換え vs 随机词对照 | T1_anomaly | -0.003 [-0.012, 0.006] | 0.584 |
| basic | label_llm | waffle | LLM言い換え vs 随机词对照 | macro_T2 | 0.067 [0.054, 0.078] | 0.000 |
| basic | llm | attr | LLM言い換え vs 人手属性描述(均无类别名) | T1_anomaly | 0.039 [0.029, 0.049] | 0.000 |
| basic | llm | attr | LLM言い換え vs 人手属性描述(均无类别名) | macro_T2 | -0.011 [-0.017, -0.005] | 0.000 |
| extended | term_literal | label | 直译点検要领用语 vs 通用英语标签 | T1_anomaly | -0.036 [-0.057, -0.013] | 0.000 |
| extended | term_literal | label | 直译点検要领用语 vs 通用英语标签 | macro_T2 | -0.027 [-0.051, -0.006] | 0.008 |
| extended | attr | label | 只用视觉属性描述(无类别名) vs 标签 | T1_anomaly | -0.141 [-0.186, -0.097] | 0.000 |
| extended | attr | label | 只用视觉属性描述(无类别名) vs 标签 | macro_T2 | -0.014 [-0.039, 0.010] | 0.236 |
| extended | label_attr | label | 标签+视觉属性 vs 标签 | T1_anomaly | 0.020 [-0.003, 0.043] | 0.092 |
| extended | label_attr | label | 标签+视觉属性 vs 标签 | macro_T2 | 0.038 [0.030, 0.047] | 0.000 |
| extended | label_attr | waffle | 视觉属性的内容 vs 同数量随机词(集成效应对照) | T1_anomaly | -0.014 [-0.025, -0.004] | 0.006 |
| extended | label_attr | waffle | 视觉属性的内容 vs 同数量随机词(集成效应对照) | macro_T2 | 0.071 [0.061, 0.081] | 0.000 |
| extended | label_llm | label_attr | LLM言い換え扩展 vs 人手属性描述 | T1_anomaly | 0.006 [0.002, 0.010] | 0.002 |
| extended | label_llm | label_attr | LLM言い換え扩展 vs 人手属性描述 | macro_T2 | -0.006 [-0.010, -0.003] | 0.002 |
| extended | label_llm | waffle | LLM言い換え vs 随机词对照 | T1_anomaly | -0.008 [-0.019, 0.002] | 0.112 |
| extended | label_llm | waffle | LLM言い換え vs 随机词对照 | macro_T2 | 0.065 [0.053, 0.076] | 0.000 |
| extended | llm | attr | LLM言い換え vs 人手属性描述(均无类别名) | T1_anomaly | 0.052 [0.035, 0.067] | 0.000 |
| extended | llm | attr | LLM言い換え vs 人手属性描述(均无类别名) | macro_T2 | -0.012 [-0.019, -0.007] | 0.000 |

### E1 hard-negative normal prompts (extended - basic), T1 anomaly

| condition | auroc_basic | auroc_extended | diff | ci_lo | ci_hi | p_boot |
|---|---|---|---|---|---|---|
| label | 0.864 | 0.869 | 0.005 | -0.017 | 0.028 | 0.578 |
| term_literal | 0.848 | 0.834 | -0.015 | -0.037 | 0.007 | 0.194 |
| attr | 0.681 | 0.728 | 0.047 | 0.010 | 0.084 | 0.020 |
| label_attr | 0.850 | 0.889 | 0.039 | 0.021 | 0.057 | 0.000 |
| llm | 0.720 | 0.780 | 0.060 | 0.031 | 0.089 | 0.000 |
| label_llm | 0.856 | 0.895 | 0.039 | 0.023 | 0.056 | 0.000 |
| waffle | 0.859 | 0.903 | 0.044 | 0.026 | 0.063 | 0.000 |

### E1 confusion on single-label defect images (condition `label_attr`; rows=true, cols=top-1)

| true | crack | spallation | efflorescence | exposed_bars | corrosion_stain |
|---|---|---|---|---|---|
| crack | 70 | 36 | 2 | 0 | 19 |
| spallation | 6 | 7 | 4 | 0 | 0 |
| efflorescence | 16 | 24 | 44 | 0 | 35 |
| exposed_bars | 3 | 4 | 0 | 4 | 9 |
| corrosion_stain | 3 | 3 | 1 | 1 | 20 |

## E2 DeepCrack: ordinality of degree expressions

Text side (`t_moderate` in (0,1) means 'moderate' lies between 'slight' and 'severe'):

| set | t_moderate | ordered | cos_slight_moderate | cos_moderate_severe | cos_slight_severe |
|---|---|---|---|---|---|
| adjective | 0.146 | True | 0.993 | 0.984 | 0.987 |
| width | 0.929 | True | 0.983 | 0.995 | 0.987 |
| extent | 0.442 | True | 0.972 | 0.964 | 0.937 |
| manual | 0.653 | True | 0.976 | 0.987 | 0.965 |
| random_control | 0.322 | True | 0.973 | 0.944 | 0.920 |

Image side (Spearman between expected level and mask proxies):

| set | area_ratio | length | width_p90 |
|---|---|---|---|
| adjective | -0.098 [-0.181, -0.012] | 0.001 [-0.083, 0.089] | -0.037 [-0.120, 0.052] |
| extent | -0.078 [-0.166, 0.005] | 0.290 [0.205, 0.368] | -0.274 [-0.352, -0.196] |
| manual | 0.311 [0.227, 0.388] | 0.460 [0.392, 0.518] | 0.145 [0.056, 0.232] |
| random_control | -0.117 [-0.198, -0.037] | 0.076 [-0.011, 0.163] | -0.137 [-0.221, -0.051] |
| width | 0.172 [0.089, 0.257] | 0.174 [0.097, 0.250] | 0.133 [0.048, 0.220] |

## E3 DeepCrack: window-based localisation

| condition | pixel_auroc | ci_lo | ci_hi | pixel_ap | mean_image_auroc | n_images | pos_rate |
|---|---|---|---|---|---|---|---|
| label | 0.795 | 0.785 | 0.808 | 0.095 | 0.850 | 537 | 0.036 |
| term_literal | 0.767 | 0.754 | 0.782 | 0.087 | 0.843 | 537 | 0.036 |
| label_attr | 0.776 | 0.762 | 0.793 | 0.087 | 0.846 | 537 | 0.036 |
| label_llm | 0.781 | 0.766 | 0.796 | 0.089 | 0.847 | 537 | 0.036 |
| waffle | 0.789 | 0.774 | 0.805 | 0.090 | 0.849 | 537 | 0.036 |

| A | B | diff | ci_lo | ci_hi | p_boot |
|---|---|---|---|---|---|
| label_attr | label | -0.019 | -0.031 | -0.006 | 0.000 |
| label_attr | waffle | -0.012 | -0.018 | -0.007 | 0.000 |
| label_llm | waffle | -0.008 | -0.014 | -0.003 | 0.010 |
| term_literal | label | -0.028 | -0.037 | -0.018 | 0.000 |
| label_llm | label_attr | 0.004 | 0.002 | 0.006 | 0.000 |

## Figures
- figures/e1_false_negatives.png
- figures/e1_false_positives.png
- figures/e2_scatter.png
- figures/e3_heatmaps.png

## Observations
See `observations.md` (written by the experimenter).
