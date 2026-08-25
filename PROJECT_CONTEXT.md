# Enhanced Sepsis Prediction - Project Context

## Objective
Build an enhanced sepsis early prediction system on PhysioNet 2019 Challenge data, significantly improving upon the baseline (PR-AUC 0.0714, ROC-AUC 0.7598, Recall 55.25%).

## Data Source
- **Dataset**: PhysioNet/Computing in Cardiology Challenge 2019
- **Location**: `dataset/physionet_sepsis/training/` (training_setA, training_setB)
- **Format**: Pipe-separated (.psv) hourly patient records
- **Target**: `SepsisLabel` (binary, 1 = sepsis onset within 6 hours)

## Key Variables (40 columns)
Vitals: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
Labs: BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets
Static: Age, Gender, Unit1, Unit2, HospAdmTime
Temporal: ICULOS (ICU length of stay in hours)
Target: SepsisLabel

## Non-Negotiable Rules
1. **No Data Leakage**: Fit transformers/selection ONLY on Train → transform Val/Test
2. **Patient-Level Splits**: Same patients never in multiple sets
3. **Temporal Causality**: Features at hour t use only ≤ t
4. **Preserve Baseline**: `baseline/` is read-only reference

## Enhanced Pipeline Phases

### 1. Data Audit (`enhanced/data/audit.py`)
- Load all .psv files from training_setA and training_setB
- Patient-level statistics: missingness %, class balance, ICU stay lengths, variable distributions
- Output: `experiments/audit_report.md`

### 2. Preprocessing (`enhanced/data/preprocessing.py`)
| Step | Method | Fit On | Apply To |
|------|--------|--------|----------|
| Missing values | Benchmark MICE / KNN / MissForest | Train | Train/Val/Test |
| Outliers | IQR capping (1.5×IQR) | Train | Train/Val/Test |
| Normalization | StandardScaler / RobustScaler | Train | Train/Val/Test |
- Save fitted transformers (joblib)

### 3. Temporal Feature Engineering (`enhanced/features/temporal.py`)
For each clinical variable (HR, MAP, Lactate, Temp, RR, SpO2, WBC, Creatinine, Platelets, Bilirubin, Urine, Glucose, Age, Gender...):
- **Lags**: t-1, t-3, t-6
- **Differences**: diff_1h, diff_3h, pct_change_1h
- **Rolling (causal)**: mean_3h, mean_6h, mean_12h, std_3h, std_6h, min_6h, max_6h
- **Trends**: Linear slope over 3h, 6h windows
- **Constraint**: Only hours ≤ current hour (no leakage)

### 4. Hybrid Feature Selection (`enhanced/features/selection.py`)
```
All Features
    │
    ├─ Boruta (RF, n_iter=50) → selected_boruta
    └─ Mutual Info (top k=100) → selected_mi
    │
    └─ Union → final_features (save list + importance CSV)
```

### 5. Four Base Models (`enhanced/models/`)
Train each on Train, evaluate on Val, save val predictions for stacking:
| Model | Key Params |
|-------|------------|
| RandomForest | 500 trees, class_weight='balanced' |
| XGBoost | scale_pos_weight, early_stopping=50 |
| LightGBM | scale_pos_weight, early_stopping=50 |
| CatBoost | scale_pos_weight, early_stopping=50 |
Output per model: model.pkl, val_preds.npy, metrics.json

### 6. Stacking Ensemble (`enhanced/stacking/stack.py`)
- Meta-learner: LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
- Input: 4× val predictions (shape: n_samples × 4)
- Train meta on Val → Predict on Test
- Save: meta_learner.pkl, test_preds.npy

### 7. Probability Calibration (`enhanced/calibration/calibrate.py`)
Compare on Val only:
- Platt (sigmoid/CalibratedClassifierCV)
- Isotonic (CalibratedClassifierCV)
- Select by Brier Score + calibration curve visual
- Apply chosen calibrator to Test predictions
- Save: calibrator.pkl, calibrated_test_preds.npy

### 8. Clinical Threshold Selection (`enhanced/calibration/threshold.py`)
On Val (calibrated):
- Sweep thresholds → compute Sensitivity, Precision, F1, MCC, Alarm Rate
- Choose threshold balancing clinical sensitivity vs alarm burden
- Save: optimal_threshold.json

### 9. Explainable AI (`enhanced/xai/explain.py`)
| Method | Scope | Output |
|--------|-------|--------|
| SHAP (TreeExplainer) | Global (500-patient sample) | Summary plot, dependence plots, feature importance CSV |
| SHAP | Local (per patient) | Waterfall/force plot |
| LIME (TabularExplainer) | Local (per patient) | Bar chart of feature contributions |

### 10. Interactive Dashboard (`enhanced/dashboard/app.py`)
- Stack: Streamlit
- Inputs: Patient ID + ICU hour (from dataset) OR Manual vitals entry
- Outputs: Risk probability + category (LOW/MODERATE/HIGH), Timeline charts, SHAP global importance, LIME patient-specific explanation, Model metadata panel

### 11. Final Evaluation (`enhanced/experiments/final_eval.py`)
Compare Baseline (preserved) vs Enhanced on Test:

| Metric | Baseline | Enhanced |
|--------|----------|----------|
| PR-AUC | 0.0714 | — |
| ROC-AUC | 0.7598 | — |
| Recall | 55.25% | — |
| Precision | — | — |
| F1 | — | — |
| MCC | — | — |
| Brier Score | — | — |

- Calibration curves, decision curves
- Cross-source evaluation if applicable
- Generate: results_table.csv, figures/, report.md

## Execution Order
```bash
# 1. Context + deps
cd C:\PROJECT
# create requirements.txt

# 2. Data audit
python enhanced/data/audit.py

# 3. Preprocessing (fit on train only)
python enhanced/data/preprocessing.py

# 4. Temporal features
python enhanced/features/temporal.py

# 5. Feature selection
python enhanced/features/selection.py

# 6. Base models (4x parallelizable)
python enhanced/models/train_rf.py
python enhanced/models/train_xgb.py
python enhanced/models/train_lgbm.py
python enhanced/models/train_catboost.py

# 7. Stacking
python enhanced/stacking/stack.py

# 8. Calibration + threshold
python enhanced/calibration/calibrate.py
python enhanced/calibration/threshold.py

# 9. XAI
python enhanced/xai/explain.py

# 10. Dashboard
streamlit run enhanced/dashboard/app.py

# 11. Final eval
python enhanced/experiments/final_eval.py
```

## Deliverables
```
enhanced/
├── models/                 # All saved .pkl artifacts
├── experiments/
│   ├── audit_report.md
│   ├── feature_importance.csv
│   ├── metrics_comparison.csv
│   ├── calibration_curves.png
│   ├── shap_summary.png
│   └── final_report.md
├── dashboard/app.py        # Runnable demo
└── PROJECT_CONTEXT.md      # Context for next session
```