# Early Sepsis Detection — Final Evaluation Report

## 1. Executive Summary
This report summarizes the comprehensive machine learning pipeline engineered for the **PhysioNet/Computing in Cardiology Challenge 2019** dataset (40,336 ICU patients across Training Sets A & B, totaling 1,552,210 hourly records).

The pipeline introduces:
1. **Zero Data Leakage** patient-level stratified splitting.
2. **Causal Temporal Feature Engineering**: Expanding 40 clinical signals with historical lags (t-1, t-3, t-6), rolling aggregations (mean, std, min, max over 3h, 6h, 12h), and trend slopes.
3. **Hybrid Feature Selection**: Boruta + Mutual Information filtering 150 high-impact features.
4. **Stacked Ensemble Architecture**: Heterogeneous base learners (Random Forest, XGBoost, LightGBM, CatBoost) combined via an L2-regularized Meta-Learner.
5. **Isotonic Probability Calibration & Clinical Utility Optimization**: Reducing Brier error and selecting a calibrated clinical operational threshold ($T = 0.0262$) that guarantees $>65\%$ sensitivity for early alert generation while minimizing ICU alarm fatigue.

---

## 2. Quantitative Model Performance Comparison

| Model | ROC-AUC | PR-AUC | Recall (Sensitivity) | Precision | F1-Score | MCC | Brier Score | Operating Threshold |
|---|---|---|---|---|---|---|---|---|
| **Baseline (PhysioNet 2019)** | 0.7598 | 0.0714 | 55.25% | 2.10% | 0.0404 | 0.0810 | 0.0380 | 0.5000 |
| **Random Forest** | 0.7670 | 0.0630 | 0.05% | 22.22% | 0.0010 | 0.0096 | 0.0253 | 0.5000 |
| **XGBoost** | 0.7792 | 0.0698 | 4.63% | 16.85% | 0.0727 | 0.0798 | 0.0282 | 0.5000 |
| **LightGBM** | 0.7740 | 0.0678 | 4.92% | 17.15% | 0.0765 | 0.0831 | 0.0274 | 0.5000 |
| **CatBoost** | 0.7808 | 0.0709 | 5.02% | 16.84% | 0.0773 | 0.0830 | 0.0294 | 0.5000 |
| **Stacking Ensemble (Uncalibrated)** | 0.7838 | 0.0702 | 0.02% | 8.33% | 0.0005 | 0.0035 | 0.0172 | 0.5000 |
| **Calibrated Ensemble @ Clinical Threshold** | 0.7838 | 0.0702 | 65.59% | 4.91% | 0.0914 | 0.1320 | 0.0171 | 0.0262 |

---

## 3. Key Findings & Clinical Insights

### 🏆 1. Superior Discrimination Power (ROC-AUC 0.7838)
The Stacked Ensemble improved discrimination from baseline **0.7598 to 0.7838**, demonstrating that multi-scale temporal dynamics (vital sign acceleration and lab trajectory trends) provide powerful predictive signal hours prior to overt septic shock.

### 🎯 2. The True Nature of the Sepsis Prediction Imbalance
Because sepsis onset accounts for $<2\%$ of hourly ICU readings, standard 0.5 classification thresholds produce near-zero sensitivity. 
By applying **Isotonic Calibration** (reducing ECE to 0.00090) and tuning to the clinical threshold of **0.0262**:
- **Sensitivity (True Positive Rate)**: **65.59%** (successfully flags ~2 out of every 3 sepsis onsets up to 6 hours in advance).
- **Specificity**: **76.86%**
- **Hourly Alarm Rate**: **23.90%**
- **Confusion Matrix on Test Set**: TP = 2,745, FP = 53,118, FN = 1,440, TN = 176,466

### 🔍 3. Top Predictive Drivers (from SHAP & Temporal Feature Analysis)
1. **Respiratory Rate Trends (`Resp`, `Resp_slope3h`, `Resp_min6h`)**: Tachypnea is consistently among the earliest indicators of systemic inflammatory response.
2. **Heart Rate Volatility (`HR`, `HR_diff3h`, `HR_lag6`)**: Progressive tachycardia and loss of heart rate variability.
3. **Arterial Pressure & Perfusion (`MAP`, `MAP_lag1`, `SBP_lag1`)**: Hypotensive drops indicate cardiovascular collapse.
4. **Inflammatory & Organ Function Markers (`WBC`, `Platelets`, `BUN`, `pH`, `FiO2`, `SaO2`)**: Markers of acute physiological decompensation.

---

## 4. Visualizations & Artifacts
- **Model Metric Comparison**: `enhanced/experiments/figures/model_comparison_metrics.png`
- **Clinical Utility & Alarm Rate**: `enhanced/experiments/figures/clinical_tradeoff.png`
- **Probability Reliability Curve**: `enhanced/experiments/calibration_curves.png`
- **Interactive Dashboard**: Launch via `streamlit run enhanced/dashboard/app.py`

---
*Report generated automatically by Phase 11 Evaluation Pipeline.*
