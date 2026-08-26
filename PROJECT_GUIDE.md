# Enhanced Sepsis Prediction — Complete Project Guide

> **For collaborators & future sessions:** This document explains the entire pipeline from raw data to deployable model. Read this first before touching any code.

---

## 🎯 Project Objective

Build an **enhanced sepsis early prediction system** on PhysioNet 2019 Challenge data that significantly improves upon the baseline:
- **Baseline metrics**: PR-AUC 0.0714, ROC-AUC 0.7598, Recall 55.25%
- **Goal**: Higher PR-AUC, better calibration, clinical thresholds, explainability

---

## 📂 Data Source

- **Dataset**: PhysioNet/Computing in Cardiology Challenge 2019
- **Location**: `dataset/physionet_sepsis/training/`
  - `training_setA/` — 20,000 patients (p000001–p020000)
  - `training_setB/` — 20,336 patients (p100001–p120336)
- **Format**: Pipe-separated (`.psv`) hourly records per patient
- **Target**: `SepsisLabel` (1 = sepsis onset within 6 hours)
- **Total**: 40,336 patients, 1,552,210 hourly records

### Key Variables (40 columns)

| Category | Variables |
|----------|-----------|
| **Vitals** | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 |
| **Labs** | BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets |
| **Static** | Age, Gender, Unit1, Unit2, HospAdmTime |
| **Temporal** | ICULOS (ICU hour, 1–336) |
| **Target** | SepsisLabel |

---

## ⚠️ NON-NEGOTIABLE RULES (Read Before Every Phase)

| Rule | Why | How We Enforce |
|------|-----|----------------|
| **No data leakage** | Invalidates metrics | Fit transformers/selection ONLY on Train → apply to Val/Test |
| **Patient-level splits** | Same patient in multiple sets = leakage | Split by `patient_id` with stratification |
| **Temporal causality** | Future data leaks into past | Features at hour t use only hours ≤ t |
| **Preserve baseline** | Need fair comparison | `baseline/` folder is read-only reference |

---

## 📁 Directory Structure

```
C:\PROJECT\
├── dataset/physionet_sepsis/training/    # Raw .psv files (DO NOT EDIT)
├── baseline/                             # Read-only baseline reference
├── enhanced/
│   ├── data/
│   │   ├── audit.py                      # Phase 1: Data audit
│   │   ├── preprocessing_fast.py         # Phase 2: Preprocessing (USE THIS)
│   │   ├── preprocessing.py              # Phase 2: Full (slow, deprecated)
│   │   └── processed/                    # Output: train/val/test parquet
│   ├── features/
│   │   └── temporal.py                   # Phase 3: Temporal features
│   ├── models/
│   │   ├── train_rf.py                   # Phase 5: RandomForest
│   │   ├── train_xgb.py                  # Phase 5: XGBoost
│   │   ├── train_lgbm.py                 # Phase 5: LightGBM
│   │   ├── train_catboost.py             # Phase 5: CatBoost
│   │   └── transformers/                 # Fitted preprocessors (joblib)
│   ├── stacking/
│   │   └── stack.py                      # Phase 6: Stacking ensemble
│   ├── calibration/
│   │   ├── calibrate.py                  # Phase 7: Platt/Isotonic
│   │   └── threshold.py                  # Phase 8: Clinical threshold
│   ├── xai/
│   │   └── explain.py                    # Phase 9: SHAP + LIME
│   ├── dashboard/
│   │   └── app.py                        # Phase 10: Streamlit demo
│   └── experiments/
│       ├── final_eval.py                 # Phase 11: Final evaluation
│       ├── audit_report.md               # Phase 1 output
│       └── *.csv, *.png                  # All experiment artifacts
├── requirements.txt
├── PROJECT_CONTEXT.md                    # Short context for quick resume
└── PROJECT_GUIDE.md                      # THIS FILE
```

---

## 🔄 Complete Phase Pipeline

### Phase 1: Data Audit ✅ DONE
**Script**: `enhanced/data/audit.py`
**Run**: `python enhanced/data/audit.py`

**What it does**:
- Loads all 40,336 `.psv` files into single parquet
- Computes patient-level stats: ICU stay length, sepsis onset hour, missingness %
- Computes variable-level stats: missingness %, distributions
- Generates `audit_report.md` with tables

**Outputs**:
- `enhanced/experiments/raw_combined.parquet` (1.55M rows)
- `enhanced/experiments/patient_stats.csv`
- `enhanced/experiments/variable_stats.csv`
- `enhanced/experiments/audit_report.md`

**Key findings**:
- 7.27% sepsis rate (2,932 positive patients)
- Median ICU stay: 38h, sepsis onset median: 29h
- Vitals ~85–90% observed; Labs >93% missing
- Static vars (Age, Gender, ICULOS): 100% complete

---

### Phase 2: Preprocessing ✅ DONE
**Script**: `enhanced/data/preprocessing_fast.py`  ← **USE THIS ONE**
**Run**: `python enhanced/data/preprocessing_fast.py`

**What it does**:
1. **Patient-level split** (70/15/15, stratified by sepsis label)
   - Train: 28,234 patients, 1,087,703 rows
   - Val: 6,051 patients, 230,738 rows
   - Test: 6,051 patients, 233,769 rows
2. **IQR outlier capping** (1.5×IQR) — fit on train only
3. **Imputation benchmark** (on 2% sample for speed):
   - KNN (k=5)
   - MICE (IterativeImputer + RF, 5 iterations) ← **SELECTED**
4. **Per-column scaling**:
   - StandardScaler for low-skew variables (vitals)
   - RobustScaler for high-skew variables (labs)
   - Fit on train only
5. **Missingness indicators** added as features (`{col}_was_missing`)

**Outputs**:
- `enhanced/data/processed/train_processed.parquet` (51 cols)
- `enhanced/data/processed/val_processed.parquet`
- `enhanced/data/processed/test_processed.parquet`
- `enhanced/models/transformers/`:
  - `iqr_capper.pkl`
  - `imputer_knn.pkl`, `imputer_mice.pkl`
  - `scalers_standard.pkl`, `scalers_robust.pkl` (dict per column)
  - `split_info.json` (feature lists, scaler choices)

**Features used** (from audit, <50% missing):
- Vitals: HR, O2Sat, Temp, SBP, MAP, DBP, Resp
- Labs: FiO2, pH, PaCO2, SaO2, BUN, Calcium, Glucose, Potassium, Hct, Hgb, WBC, Platelets
- Static: Age, Gender, Unit1, Unit2, HospAdmTime
- Target: SepsisLabel, ICULOS, patient_id

---

### Phase 3: Temporal Feature Engineering ✅ DONE
**Script**: `enhanced/features/temporal.py`
**Run**: `python enhanced/features/temporal.py`

**What it does** (per patient, causally — only hours ≤ current hour):
For each clinical variable (vitals + key labs):

| Feature Type | Windows |
|--------------|---------|
| **Lags** | t-1, t-3, t-6 |
| **Differences** | diff_1h (t - t-1), diff_3h (t - t-3), pct_change_1h |
| **Rolling (causal)** | mean_3h, mean_6h, mean_12h, std_3h, std_6h, min_6h, max_6h |
| **Trends** | Linear slope over 3h, 6h windows |

**Constraint**: Group by `patient_id`, sort by `ICULOS`, use only `.shift()` and `.rolling().apply()` with `min_periods=1` — **never use future data**.

**Outputs** (✅ verified):
- `enhanced/data/processed/train_temporal.parquet` — **1.83 GB** (1.09M rows × 314 cols)
- `enhanced/data/processed/val_temporal.parquet` — 404 MB
- `enhanced/data/processed/test_temporal.parquet` — 410 MB
- `enhanced/data/processed/temporal_feature_cols.json` — 24 base + 285 temporal = **309 features**

---

### Phase 4: Hybrid Feature Selection ✅ DONE
**Script**: `enhanced/features/selection.py`
**Run**: `python enhanced/features/selection.py`

**Pipeline (GPU-accelerated)**:
```
All Features (309)
    │
    ├─ Step 1: Mutual Information (sklearn, CPU, fast filter) → top 150 candidates
    │
    ├─ Step 2: Boruta with XGBoost GPU estimator (RTX 4060)
    │          n_iter=50, p<0.01 — runs on 150k stratified sample
    │          → confirmed, tentative, rejected sets
    │
    └─ Step 3: Union(Boruta_confirmed + Boruta_tentative + MI_top100)
               → final_features (typically 80–150 features)
```

**GPU note**: Boruta uses `XGBClassifier(tree_method='hist', device='cuda')`
This is ~5–10× faster than CPU RandomForest-based Boruta.

**Results (✅ verified)**:
- Boruta confirmed: 24 features | Tentative: 32 | MI-top-100 union = **109 final features**
- GPU used: `gpu_used: true` (XGBoost CUDA on RTX 4060)

**Outputs (✅)**:
- `enhanced/experiments/selected_features.json` — 109 final features
- `enhanced/experiments/feature_importance.csv` — MI + Boruta scores
- `enhanced/experiments/feature_selection_plot.png` — top-50 bar chart

---

### Phase 5: Four Base Models ✅ DONE
**Scripts**: `enhanced/models/train_xgb.py`, `train_lgbm.py`, `train_catboost.py`, `train_rf.py`
**Run**: From `enhanced/models/` directory

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Train Time | Accelerator |
|---|---|---|---|---|---|---|---|
| **CatBoost** | **0.7685** | **0.0631** | **64.29%** | 0.0427 | 0.0801 | 4.2s | ✅ GPU (RTX 4060) |
| **XGBoost** | **0.7626** | **0.0664** | **62.11%** | 0.0445 | 0.0831 | 3.3s | ✅ GPU (RTX 4060) |
| **RandomForest** | 0.7407 | 0.0545 | 6.74% | 0.1071 | 0.0827 | 105.5s | CPU (24 threads) |
| **LightGBM** | 0.7089 | 0.0415 | 0.00% | 0.0000 | 0.0000 | 3.6s | CPU (multi-core) |

**Outputs generated (✅)**:
- `xgb_model.pkl`, `xgb_val_preds.npy`, `xgb_metrics.json`
- `catboost_model.cbm`, `catboost_val_preds.npy`, `catboost_metrics.json`
- `rf_model.pkl`, `rf_val_preds.npy`, `rf_metrics.json`
- `lgbm_model.pkl`, `lgbm_val_preds.npy`, `lgbm_metrics.json`

---

### Phase 6: Stacking Ensemble 🔜 NEXT
**Script**: `enhanced/stacking/stack.py`

**Meta-learner**: LogisticRegression(C=1.0, `class_weight='balanced'`, max_iter=1000)
**Input**: 4 × val predictions (shape: n_samples × 4)
**Train**: Meta on Val → Predict on Test

**Outputs**:
- `enhanced/models/meta_learner.pkl`
- `enhanced/models/stack_test_preds.npy`

---

### Phase 7: Probability Calibration
**Script**: `enhanced/calibration/calibrate.py`

**Compare on Val (calibrated predictions)**:
- **Platt** (sigmoid): `CalibratedClassifierCV(method='sigmoid', cv=5)`
- **Isotonic**: `CalibratedClassifierCV(method='isotonic', cv=5)`

**Selection**: Lower Brier Score + visual calibration curve
**Apply**: Chosen calibrator to Test predictions

**Outputs**:
- `enhanced/models/calibrator.pkl`
- `enhanced/models/calibrated_test_preds.npy`
- `enhanced/experiments/calibration_curves.png`

---

### Phase 8: Clinical Threshold Selection
**Script**: `enhanced/calibration/threshold.py`

**On Val (calibrated)**:
- Sweep thresholds 0.01–0.99
- Compute: Sensitivity, Precision, F1, MCC, Alarm Rate (pred positive rate)
- **Choose**: Threshold balancing clinical sensitivity vs alarm burden
  - Target: Sensitivity ≥ 80%, Alarm Rate ≤ 20% (clinical preference)

**Outputs**:
- `enhanced/models/optimal_threshold.json` (threshold, metrics at threshold)
- Decision curve plot

---

### Phase 9: Explainable AI (XAI)
**Script**: `enhanced/xai/explain.py`

| Method | Scope | Output |
|--------|-------|--------|
| **SHAP TreeExplainer** | Global (500-patient sample) | Summary plot, dependence plots, feature importance CSV |
| **SHAP** | Local (per patient) | Waterfall/force plot |
| **LIME TabularExplainer** | Local (per patient) | Bar chart of feature contributions |

**Outputs**:
- `enhanced/experiments/shap_summary.png`
- `enhanced/experiments/shap_dependence_*.png`
- `enhanced/experiments/feature_importance_shap.csv`
- `enhanced/experiments/lime_patient_*.png` (examples)

---

### Phase 10: Interactive Dashboard
**Script**: `enhanced/dashboard/app.py`
**Run**: `streamlit run enhanced/dashboard/app.py`

**Stack**: Streamlit (simple, fast)
**Inputs**:
- Patient ID + ICU hour (from dataset) **OR** Manual vitals entry
**Outputs**:
- Risk probability + category (LOW / MODERATE / HIGH)
- Timeline charts (key vitals over ICU hours)
- SHAP global importance
- LIME patient-specific explanation
- Model metadata panel (version, metrics, threshold)

---

### Phase 11: Final Evaluation & Reporting
**Script**: `enhanced/experiments/final_eval.py`

**Compare Baseline vs Enhanced on Test**:

| Metric | Baseline | Enhanced |
|--------|----------|----------|
| PR-AUC | 0.0714 | — |
| ROC-AUC | 0.7598 | — |
| Recall | 55.25% | — |
| Precision | — | — |
| F1 | — | — |
| MCC | — | — |
| Brier Score | — | — |

**Plus**:
- Calibration curves (reliability diagrams)
- Decision curves (net benefit)
- Cross-source evaluation (SetA vs SetB if applicable)

**Outputs**:
- `enhanced/experiments/results_table.csv`
- `enhanced/experiments/figures/` (all plots)
- `enhanced/experiments/final_report.md`

---

## 🚀 Execution Order (Copy-Paste Ready)

```bash
cd C:\PROJECT

# 1. Already done: Audit
# python enhanced/data/audit.py

# 2. Already done: Preprocessing
# python enhanced/data/preprocessing_fast.py

# 3. Temporal features
python enhanced/features/temporal.py

# 4. Feature selection
python enhanced/features/selection.py

# 5. Base models (run in parallel - 4 terminals)
python enhanced/models/train_rf.py
python enhanced/models/train_xgb.py
python enhanced/models/train_lgbm.py
python enhanced/models/train_catboost.py

# 6. Stacking
python enhanced/stacking/stack.py

# 7. Calibration
python enhanced/calibration/calibrate.py

# 8. Threshold
python enhanced/calibration/threshold.py

# 9. XAI
python enhanced/xai/explain.py

# 10. Dashboard
streamlit run enhanced/dashboard/app.py

# 11. Final eval
python enhanced/experiments/final_eval.py
```

---

## 🖥️ GPU Acceleration (For Model Training)

Only **Phase 5 (base models)** and **Phase 6 (stacking meta-learner)** benefit from GPU.

### XGBoost
```python
params = {
    'tree_method': 'gpu_hist',     # Requires CUDA
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'scale_pos_weight': n_neg / n_pos,
    # ...
}
```

### LightGBM
```python
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'scale_pos_weight': n_neg / n_pos,
    # ...
}
```

### CatBoost
```python
params = {
    'task_type': 'GPU',
    'devices': '0',
    'scale_pos_weight': n_neg / n_pos,
    # ...
}
```

### Check GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# or
nvidia-smi
```

**Note**: Preprocessing (Phases 2–4) uses sklearn — **CPU only**. Don't waste time trying to GPU-accelerate imputation/scaling.

---

## 🔑 Key Files to Understand

| File | Purpose |
|------|---------|
| `enhanced/experiments/audit_report.md` | Data understanding |
| `enhanced/data/preprocessing_fast.py` | Preprocessing logic (reference) |
| `enhanced/models/transformers/split_info.json` | Feature lists, scaler choices |
| `enhanced/models/transformers/*.pkl` | Fitted transformers (load with `joblib.load()`) |
| `enhanced/data/processed/*_processed.parquet` | Clean input for Phase 3 |

---

## 🧪 How to Load Processed Data (for Phase 3+)

```python
import pandas as pd
import joblib
import json
from pathlib import Path

# Load processed splits
train = pd.read_parquet("enhanced/data/processed/train_processed.parquet")
val = pd.read_parquet("enhanced/data/processed/val_processed.parquet")
test = pd.read_parquet("enhanced/data/processed/test_processed.parquet")

# Load metadata
with open("enhanced/models/transformers/split_info.json") as f:
    info = json.load(f)

feature_cols = info['feature_columns']
numeric_cols = info['numeric_columns']
scaler_choice = info['scaler_choice']
target = info['target']  # 'SepsisLabel'
time_var = info['time_var']  # 'ICULOS'

# Load transformers if needed
iqr_capper = joblib.load("enhanced/models/transformers/iqr_capper.pkl")
imputer = joblib.load("enhanced/models/transformers/imputer_mice.pkl")
scalers_std = joblib.load("enhanced/models/transformers/scalers_standard.pkl")
scalers_rob = joblib.load("enhanced/models/transformers/scalers_robust.pkl")
```

---

## 🤝 Collaboration Guidelines

### For Your Friend (or Any Collaborator):

1. **Read this file first** — understand the pipeline
2. **Never modify** `dataset/` or `baseline/` — read-only
3. **Always fit on train only** — check `split_info.json` for patient IDs
4. **Save artifacts** to `enhanced/models/` or `enhanced/experiments/`
5. **Use the fast preprocessing script** — `preprocessing_fast.py`
6. **Run models in parallel** — 4 terminals for Phase 5
7. **Commit code, not data** — `.gitignore` should exclude `*.parquet`, `*.pkl`, `enhanced/experiments/*.csv`

### Git Workflow:
```bash
# Each person works on different phase
git checkout -b feature/temporal-engineering   # You
git checkout -b feature/xgboost-training       # Friend

# Push when phase done
git add enhanced/features/temporal.py
git commit -m "Phase 3: Temporal feature engineering"
git push origin feature/temporal-engineering

# Merge via PR after review
```

---

## 🐛 Common Pitfalls to Avoid

| Pitfall | Consequence | Fix |
|---------|-------------|-----|
| Fitting imputer on full data | Leakage → inflated metrics | Fit ONLY on train split |
| Using `df.rolling()` without `groupby('patient_id')` | Cross-patient leakage | Always group by patient |
| Forgetting `scale_pos_weight` | Poor recall on minority class | Compute `n_neg/n_pos` from train |
| Using test set for threshold tuning | Overfitting | Use val set only |
| Not saving transformers | Can't reproduce / deploy | `joblib.dump()` everything |

---

## 📞 Resume Checklist (Next Session)

```bash
# 1. Verify environment
cd C:\PROJECT
pip install -r requirements.txt  # If needed

# 2. Check Phase 2 outputs exist
ls enhanced/data/processed/
ls enhanced/models/transformers/

# 3. Continue with Phase 3
python enhanced/features/temporal.py
```

---

## 📝 Version History

| Date | Phase | Author | Notes |
|------|-------|--------|-------|
| 2026-08-25 | 1–2 | You | Audit + fast preprocessing done |
| 2026-08-25 | 3 | You | Temporal features: 309 features, train=1.83GB |
| 2026-08-26 | 4 | You | Feature selection: 109 features (Boruta+MI, GPU) |
| 2026-08-26 | 5 | You | Base models trained: XGBoost, LightGBM, CatBoost, RF |
| — | 6–11 | — | Pending |

---

## 🆘 Need Help?

- **Quick context**: Read `PROJECT_CONTEXT.md`
- **Full details**: This file (`PROJECT_GUIDE.md`)
- **Data audit**: `enhanced/experiments/audit_report.md`
- **Code questions**: Check the phase script — each has docstrings

---

**Last Updated**: 2026-08-26 (Phase 5 complete — all 4 base models trained)
**Next Action**: Run Phase 6 stacking ensemble (`enhanced/stacking/stack.py`)