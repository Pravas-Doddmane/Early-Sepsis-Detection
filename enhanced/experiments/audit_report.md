# PhysioNet 2019 Sepsis Data Audit Report

## Dataset Overview
- **Total Patients**: 40,336
- **Total Hourly Records**: 1,552,210
- **Sepsis-Positive Patients**: 2,932 (7.27%)
- **Sepsis-Negative Patients**: 37,404 (92.73%)

## ICU Stay Length Distribution
| Statistic | Value (hours) |
|-----------|---------------|
| Count | 40,336 |
| Mean | 38.5 |
| Std | 22.8 |
| Min | 8 |
| 25% | 24 |
| 50% (Median) | 38 |
| 75% | 47 |
| Max | 336 |

## Patient-Level Missingness Distribution
| Statistic | Value (%) |
|-----------|-----------|
| Mean | 68.6 |
| Std | 5.2 |
| Min | 38.2 |
| 25% | 65.4 |
| 50% (Median) | 68.1 |
| 75% | 71.8 |
| Max | 87.8 |

## Sepsis Onset Timing (Sepsis-Positive Patients Only)
| Statistic | Value (hours) |
|-----------|---------------|
| Count | 2,932 |
| Mean | 50.9 |
| Std | 59.4 |
| Min | 1 |
| 25% | 7 |
| 50% (Median) | 29 |
| 75% | 73 |
| Max | 331 |

## Variable-Level Missingness (Top 20 Most Missing)
| variable         |   pct_missing |       mean |   median |
|:-----------------|--------------:|-----------:|---------:|
| Bilirubin_direct |       99.8074 |   1.83618  |    0.445 |
| Fibrinogen       |       99.3402 | 287.386    |  250     |
| TroponinI        |       99.0477 |   8.2901   |    0.3   |
| Bilirubin_total  |       98.5092 |   2.11406  |    0.9   |
| Alkalinephos     |       98.3932 | 102.484    |   74     |
| AST              |       98.3776 | 260.223    |   41     |
| Lactate          |       97.3299 |   2.64667  |    1.8   |
| PTT              |       97.0559 |  41.2312   |   32.4   |
| SaO2             |       96.5494 |  92.6542   |   97     |
| EtCO2            |       96.2868 |  32.9577   |   33     |
| Phosphate        |       95.9863 |   3.54424  |    3.3   |
| HCO3             |       95.8106 |  24.0755   |   24     |
| Chloride         |       95.4603 | 105.828    |  106     |
| BaseExcess       |       94.579  |  -0.689919 |    0     |
| PaCO2            |       94.4401 |  41.0219   |   40     |
| Calcium          |       94.1161 |   7.55753  |    8.3   |
| Platelets        |       94.0595 | 196.014    |  181     |
| Creatinine       |       93.9044 |   1.5107   |    0.94  |
| Magnesium        |       93.6896 |   2.05145  |    2     |
| WBC              |       93.5932 |  11.4464   |   10.3   |

## Variable-Level Missingness (Least Missing - Fully Observed)
| variable    |   pct_missing |        mean |   median |
|:------------|--------------:|------------:|---------:|
| Age         |   0           |  62.0095    |    64    |
| Gender      |   0           |   0.559269  |     1    |
| ICULOS      |   0           |  26.995     |    21    |
| SepsisLabel |   0           |   0.0179847 |     0    |
| HospAdmTime |   0.000515394 | -56.1251    |    -6.03 |
| HR          |   9.88262     |  84.5814    |    83.5  |
| MAP         |  12.4513      |  82.4001    |    80    |
| O2Sat       |  13.0611      |  97.194     |    98    |
| SBP         |  14.577       | 123.75      |   121    |
| Resp        |  15.3546      |  18.7265    |    18    |

## Key Clinical Variables Summary Statistics
| variable        |   pct_missing |       mean |        std |   median |   min |    max |
|:----------------|--------------:|-----------:|-----------:|---------:|------:|-------:|
| HR              |       9.88262 |  84.5814   |  17.3252   |    83.5  |  20   |  280   |
| O2Sat           |      13.0611  |  97.194    |   2.93692  |    98    |  20   |  100   |
| Temp            |      66.1627  |  36.9772   |   0.770014 |    37    |  20.9 |   50   |
| MAP             |      12.4513  |  82.4001   |  16.3418   |    80    |  20   |  300   |
| Resp            |      15.3546  |  18.7265   |   5.09819  |    18    |   1   |  100   |
| Creatinine      |      93.9044  |   1.5107   |   1.8056   |     0.94 |   0.1 |   46.6 |
| Glucose         |      82.8943  | 136.932    |  51.3107   |   127    |  10   |  988   |
| Lactate         |      97.3299  |   2.64667  |   2.52621  |     1.8  |   0.2 |   31   |
| Bilirubin_total |      98.5092  |   2.11406  |   4.31147  |     0.9  |   0.1 |   49.6 |
| WBC             |      93.5932  |  11.4464   |   7.73101  |    10.3  |   0.1 |  440   |
| Platelets       |      94.0595  | 196.014    | 103.635    |   181    |   1   | 2322   |
| Age             |       0       |  62.0095   |  16.3862   |    64    |  14   |  100   |
| Gender          |       0       |   0.559269 |   0.496475 |     1    |   0   |    1   |

## Hourly Sepsis Rate (Class Balance Over Time)
| ICULOS (hour) | Sepsis Rate |
|---------------|-------------|
|   1 | 0.0117 |
|   2 | 0.0129 |
|   3 | 0.0141 |
|   4 | 0.0151 |
|   5 | 0.0163 |
|   6 | 0.0177 |
|   7 | 0.0188 |
|   8 | 0.0202 |
|   9 | 0.0161 |
|  10 | 0.0141 |
|  11 | 0.0130 |
|  12 | 0.0123 |
|  13 | 0.0119 |
|  14 | 0.0120 |
|  15 | 0.0114 |
|  16 | 0.0108 |
|  17 | 0.0105 |
|  18 | 0.0101 |
|  19 | 0.0096 |
|  20 | 0.0100 |
|  21 | 0.0099 |
|  22 | 0.0095 |
|  23 | 0.0089 |
|  24 | 0.0088 |
|  25 | 0.0089 |
|  26 | 0.0091 |
|  27 | 0.0091 |
|  28 | 0.0089 |
|  29 | 0.0090 |
|  30 | 0.0085 |
| ... | ... |
| 332 | 0.2222 |
| 333 | 0.2222 |
| 334 | 0.2222 |
| 335 | 0.1765 |
| 336 | 0.1250 |


## Data Quality Observations
1. **High Missingness**: Most lab variables have >80% missing values
2. **Vitals More Complete**: HR, O2Sat, SBP, MAP, DBP, Resp have lower missingness
3. **Static Variables**: Age, Gender, Unit1, Unit2, HospAdmTime are constant per patient
4. **Class Imbalance**: Sepsis rate ~7.3% - highly imbalanced
5. **Temporal Pattern**: Sepsis rate increases with ICU hours (expected)
6. **ICU Stay**: Median stay ~38 hours, max 336 hours

## Recommendations for Preprocessing
1. **Imputation Strategy**: Use MICE/KNN/MissForest benchmark; vitals vs labs may need different approaches
2. **Outlier Handling**: IQR capping (1.5×IQR) fitted on training patients only
3. **Normalization**: RobustScaler for skewed lab distributions; StandardScaler for vitals
4. **Feature Engineering**: Focus on variables with <50% missingness for temporal features
7. **Patient-Level Split**: Ensure no patient appears in multiple splits

---
*Generated by enhanced/data/audit.py*
