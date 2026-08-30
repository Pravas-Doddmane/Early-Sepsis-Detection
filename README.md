# 🏥 Enhanced Early Sepsis Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-PhysioNet%202019-green.svg)](https://physionet.org/content/challenge-2019/)

An end-to-end clinical machine learning system for **early prediction of sepsis onset (up to 6 hours in advance)** using hourly physiological data from the **PhysioNet / Computing in Cardiology Challenge 2019** dataset (40,336 ICU patients across 2 hospital systems, totaling >1.55M hourly records).

---

## 📌 Table of Contents
1. [Project Overview & Clinical Motivation](#-project-overview--clinical-motivation)
2. [Key Results & Benchmark Comparison](#-key-results--benchmark-comparison)
3. [System Architecture & 11-Phase Pipeline](#-system-architecture--11-phase-pipeline)
4. [Quick Start & Running the Project](#-quick-start--running-the-project)
5. [Interactive Clinical Dashboard (Phase 10)](#-interactive-clinical-dashboard-phase-10)
6. [Repository File Map](#-repository-file-map)
7. [Detailed Methodology](#-detailed-methodology)
8. [Clinical Interpretability & Explainability (SHAP & LIME)](#-clinical-interpretability--explainability)

---

## 🩺 Project Overview & Clinical Motivation

Sepsis is a life-threatening condition caused by the body's overwhelming and dysregulated immune response to infection, leading to tissue damage, organ failure, and death. In ICU settings:
- **Mortality increases by ~7.6% for every hour** of delayed antibiotic administration following septic shock onset.
- **The Challenge**: Clinically diagnosing sepsis early is difficult due to non-specific vital signs.
- **Goal**: Predict sepsis onset **6 hours before clinical criteria are met** ($t_{\text{sepsis}} - 6\text{h}$), allowing physicians critical time to initiate targeted bundles (blood cultures, fluid resuscitation, and broad-spectrum antibiotics).

---

## 📊 Key Results & Benchmark Comparison

Performance evaluated on the strict patient-stratified **Test Set (6,051 patients, 233,769 hourly records)**:

| Model / Pipeline Stage | ROC-AUC | PR-AUC | Sensitivity (Recall) | Precision | F1-Score | Brier Score | Operating Threshold | Key Characteristics |
|---|---|---|---|---|---|---|---|---|
| **Challenge Baseline** | 0.7598 | **0.0714** | 55.25% | 2.10% | 0.0404 | 0.0380 | 0.5000 | Static single-hour features |
| **Random Forest** | 0.7670 | 0.0630 | 0.05% | **22.22%** | 0.0010 | 0.0253 | 0.5000 | 500 trees, balanced weights |
| **XGBoost** | 0.7792 | 0.0698 | 4.63% | 16.85% | 0.0727 | 0.0282 | 0.5000 | Gradient boosted trees |
| **LightGBM** | 0.7740 | 0.0678 | 4.92% | 17.15% | 0.0765 | 0.0274 | 0.5000 | Fast histogram gradient boosting |
| **CatBoost** | 0.7808 | **0.0709** | 5.02% | 16.84% | 0.0773 | 0.0294 | 0.5000 | Top single base model |
| **Stacking Ensemble (Raw)** | **0.7838** | 0.0702 | 0.02% | 8.33% | 0.0005 | 0.0172 | 0.5000 | L2-Regularized Meta-Learner |
| **Calibrated Ensemble @ Clinical Threshold** | **0.7838** | 0.0702 | **65.59%** | **4.91%** | **0.0914** | **0.0171** | **0.0262** | **Isotonic Calibrated + $\ge 65\%$ Sensitivity Rule** |

### 💡 Clinical Highlights:
- **Discrimination Superiority**: Stacked Ensemble achieves **0.7838 ROC-AUC** (+0.0240 improvement over baseline).
- **High Operational Sensitivity**: Detects **65.6%** of all true sepsis onsets in advance.
- **Reliable Probability Calibration**: Isotonic calibration lowered Expected Calibration Error (ECE) to **0.00089**, providing reliable risk probabilities.

---

## 🏗️ System Architecture & 11-Phase Pipeline

```
Raw PhysioNet 2019 Data (40,336 Patients, 1.55M Hourly Records)
                      │
                      ▼
[Phase 1: Data Audit] ──► Missingness, cohort demographics, quality report
                      │
                      ▼
[Phase 2: Stratified Preprocessing] ──► Patient-level split (70/15/15), IQR capping, MICE imputation
                      │
                      ▼
[Phase 3: Temporal Feature Engineering] ──► 309 causal features (lags t-1,3,6; rolling 3,6,12h; slopes)
                      │
                      ▼
[Phase 4: Hybrid Feature Selection] ──► Boruta (RF) + Mutual Information ──► Top 150 features
                      │
                      ▼
[Phase 5: Four Base Models] ──► RandomForest + XGBoost + LightGBM + CatBoost
                      │
                      ▼
[Phase 6: Stacking Meta-Learner] ──► Out-of-fold val predictions ──► Logistic Regression meta-model
                      │
                      ▼
[Phase 7: Probability Calibration] ──► Isotonic Regression (Brier: 0.0171)
                      │
                      ▼
[Phase 8: Clinical Threshold Optimization] ──► Optimal threshold T = 0.0262 (Sensitivity >= 65%)
                      │
                      ▼
[Phase 9: Explainable AI] ──► SHAP TreeExplainer & LIME patient local explanations
                      │
                      ▼
[Phase 10: Interactive Dashboard] ──► Streamlit Clinical Surveillance & Risk Calculator
                      │
                      ▼
[Phase 11: Final Evaluation] ──► Comparison report, figures, and benchmark tables
```

---

## 🚀 Quick Start & Running the Project

### 1. Prerequisites & Setup
Ensure Python 3.10+ is installed:
```powershell
# Navigate to the project root
cd "c:\Users\prava\7th sem major project\Project 7\Early-Sepsis-Detection"

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard (Phase 10)
```powershell
streamlit run enhanced/dashboard/app.py
```
> The dashboard will automatically open in your browser at **`http://localhost:8501`**.

### 3. Run the Final Evaluation & Generate Benchmark Reports (Phase 11)
```powershell
python enhanced/experiments/final_eval.py
```
Generates:
- `enhanced/experiments/results_table.csv`
- `enhanced/experiments/final_report.md`
- `enhanced/experiments/figures/model_comparison_metrics.png`
- `enhanced/experiments/figures/clinical_tradeoff.png`

---

## 🖥️ Interactive Clinical Dashboard (Phase 10)

The Streamlit application (`enhanced/dashboard/app.py`) provides four interactive modes:

1. **🏥 Live Patient ICU Monitor**:
   - Select any patient ID (both septic and non-septic cohorts) from the PhysioNet dataset.
   - Adjust the **ICU Hour ($t$) slider** to observe how the patient's vitals evolve over time.
   - Real-time risk badge:
     - 🟢 **LOW RISK** ($P < 2.62\%$)
     - 🟡 **ELEVATED MONITORING ALERT** ($2.62\% \le P < 8.0\%$)
     - 🔴 **HIGH SEPSIS RISK** ($P \ge 8.0\%$)
   - Multi-signal interactive Plotly trajectory charts (Heart Rate, MAP, Respiration, Temperature, Oxygenation).

2. **🎛️ Manual Clinical Parameter Risk Calculator**:
   - Clinicians can enter current vitals (HR, BP, Temp, RR, O2Sat) and lab biomarkers (WBC, Platelets, BUN, Glucose, Creatinine).
   - Get instant calibrated predictions from individual base models and the stacking ensemble.

3. **📊 Model Performance & Benchmarks**:
   - Live metrics table comparing all models.
   - Reliability calibration curves and ROC-AUC charts.

4. **ℹ️ About the Project**:
   - Architectural summary and clinical utility guidelines.

---

## 📁 Repository File Map

```text
Early-Sepsis-Detection/
├── dataset/physionet_sepsis/training/    # Raw .psv challenge records (training_setA & B)
├── baseline/                             # Read-only challenge baseline
├── .streamlit/
│   └── config.toml                       # Streamlit UI configuration
├── enhanced/
│   ├── data/
│   │   ├── audit.py                      # Phase 1: Data audit & profiling script
│   │   ├── preprocessing_fast.py         # Phase 2: Stratified preprocessor & MICE imputation
│   │   └── make_report.py                # Audit report generator
│   ├── features/
│   │   ├── temporal.py                   # Phase 3: Causal temporal feature extraction
│   │   └── selection.py                  # Phase 4: Boruta + Mutual Information selection
│   ├── models/
│   │   ├── _utils.py                     # Shared evaluation metrics & data loader
│   │   ├── train_rf.py                   # Phase 5: RandomForest training
│   │   ├── train_xgb.py                  # Phase 5: XGBoost training
│   │   ├── train_lgbm.py                 # Phase 5: LightGBM training
│   │   ├── train_catboost.py             # Phase 5: CatBoost training
│   │   ├── rf_model.pkl                  # Saved Random Forest model
│   │   ├── xgb_model.pkl                 # Saved XGBoost model
│   │   ├── lgbm_model.pkl                # Saved LightGBM model
│   │   ├── catboost_model.cbm            # Saved CatBoost model
│   │   ├── meta_learner.pkl              # Phase 6: Saved Stacking Meta-Learner
│   │   ├── calibrator.pkl                # Phase 7: Saved Isotonic Calibrator
│   │   ├── optimal_threshold.json        # Phase 8: Saved Clinical Threshold configuration
│   │   └── *_metrics.json                # Performance JSON metrics for each model
│   ├── stacking/
│   │   └── stack.py                      # Phase 6: Stacking ensemble training pipeline
│   ├── calibration/
│   │   ├── calibrate.py                  # Phase 7: Probability calibration pipeline
│   │   └── threshold.py                  # Phase 8: Clinical decision threshold optimizer
│   ├── xai/
│   │   └── explain.py                    # Phase 9: SHAP TreeExplainer & LIME explainability
│   ├── dashboard/
│   │   └── app.py                        # Phase 10: Interactive Streamlit clinical application
│   └── experiments/
│       ├── final_eval.py                 # Phase 11: Final evaluation script
│       ├── results_table.csv             # Final comparative results table
│       ├── final_report.md               # Final comprehensive markdown report
│       ├── audit_report.md               # Data audit summary report
│       ├── selected_features.json        # Selected 150 temporal features
│       ├── calibration_curves.png        # Reliability curve visualization
│       └── figures/                      # Final evaluation charts
│           ├── model_comparison_metrics.png
│           └── clinical_tradeoff.png
├── PROJECT_GUIDE.md                      # Comprehensive developer reference guide
├── PROJECT_CONTEXT.md                    # Quick session context
├── requirements.txt                      # Project dependencies
└── README.md                             # Project overview & documentation
```

---

## 🔬 Detailed Methodology

### 1. Strict Patient-Level Partitioning
To prevent data leakage, all splits (70% Train, 15% Validation, 15% Test) are partitioned strictly by `patient_id` with stratification on sepsis status. No patient record in the test set is ever seen during imputation, scaling, feature selection, or model training.

### 2. Causal Temporal Dynamics
Sepsis onset is rarely an instantaneous static event; it manifests as subtle physiological acceleration:
- **Lags**: $t-1\text{h}, t-3\text{h}, t-6\text{h}$
- **Short-term deltas**: $\Delta_{1\text{h}}, \Delta_{3\text{h}}$
- **Rolling aggregations**: Mean, standard deviation, minimum, and maximum over causal windows ($3\text{h}, 6\text{h}, 12\text{h}$).
- **Linear trajectory slopes**: Linear slope over $3\text{h}$ and $6\text{h}$ windows (e.g. rising respiration rate, dropping arterial pressure).

### 3. Probability Calibration & Clinical Decision Thresholds
Because sepsis onset is rare ($<2\%$ of hourly ICU measurements), raw model probabilities require calibration:
- **Isotonic Regression** maps raw ensemble scores to true posterior probabilities, minimizing Brier error.
- **Threshold Optimization**: Operating at the optimal threshold ($T = 0.0262$) guarantees $>65\%$ sensitivity for timely clinical intervention.

---

## 🔍 Clinical Interpretability & Explainability

Global and patient-level explanations are provided using **SHAP (SHapley Additive exPlanations)** and **LIME**:

1. **Respiratory Rate Acceleration (`Resp`, `Resp_slope3h`, `Resp_min6h`)**: Tachypnea is one of the earliest compensatory responses to systemic inflammatory response syndrome (SIRS).
2. **Heart Rate Dynamics (`HR`, `HR_diff3h`, `HR_lag6`)**: Persistent tachycardia and loss of heart rate variability.
3. **Arterial Blood Pressure (`MAP`, `MAP_lag1`, `SBP_lag1`)**: Hypotensive drops indicate impending cardiovascular instability.
4. **Biochemical Markers (`WBC`, `Platelets`, `BUN`, `FiO2`, `SaO2`)**: Indication of leukocytosis/leukopenia, thrombocytopenia, and worsening oxygen exchange.

---

*Developed for 7th Semester Major Project — Early Sepsis Detection System.*
