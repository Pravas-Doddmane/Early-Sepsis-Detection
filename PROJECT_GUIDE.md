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




# Enhanced Sepsis Prediction — Project README

> **Group project handoff doc.** Read this before touching any code. Full pipeline
> details live in `PROJECT_GUIDE.md` — this file is the fast-resume status page.

---

## Current Status: Phases 1–8 Complete ✅ | Next: Phase 9 (XAI)

```
[✅] 1. Data Audit
[✅] 2. Preprocessing
[✅] 3. Temporal Feature Engineering
[✅] 4. Hybrid Feature Selection
[✅] 5. Base Models (RF, XGBoost, LightGBM, CatBoost)
[✅] 6. Stacking Ensemble
[✅] 7. Probability Calibration
[✅] 8. Clinical Threshold Selection
[  ] 9. Explainable AI (SHAP + LIME)      <-- START HERE
[  ] 10. Interactive Dashboard
[  ] 11. Final Evaluation & Reporting
```

---

## Quick Resume

```bash
cd C:\PROJECT
python enhanced/xai/explain.py       # Phase 9 — not yet written, needs script
```

If `enhanced/xai/explain.py` doesn't exist yet, that's expected — it hasn't been
generated. Ask for it (see "How We've Been Working" below) or write it following
the Phase 9 spec in `PROJECT_GUIDE.md`.

---

## What's Done, In Detail

### Phase 1 — Data Audit
- 40,336 patients, 1,552,210 hourly records (PhysioNet 2019, setA + setB)
- Sepsis rate: 7.27% patient-level
- Output: `enhanced/experiments/audit_report.md`, `patient_stats.csv`, `variable_stats.csv`

### Phase 2 — Preprocessing
- Patient-level split 70/15/15 — **28,234 / 6,051 / 6,051 patients**, no leakage
- IQR capping (1.5×IQR), fit on train only
- Imputation: KNN vs MICE benchmarked → **MICE selected**
- Per-column StandardScaler / RobustScaler, fit on train only
- Output: `enhanced/data/processed/{train,val,test}_processed.parquet`
- Transformers: `enhanced/models/transformers/`

### Phase 3 — Temporal Feature Engineering
- Lags (t-1/3/6), diffs, rolling stats (mean/std/min/max @ 3h/6h/12h), trends
- Causal only (hour ≤ t, no leakage)
- **347 features** created from 19 base temporal columns
- Output: `enhanced/data/processed/{train,val,test}_temporal.parquet`,
  `temporal_feature_cols.json`
- ⚠️ **Known slow point**: took ~70 min total (per-patient Python loop). If
  re-running, consider vectorizing with `groupby().rolling()` — not blocking,
  just noting for anyone touching this script again.

### Phase 4 — Feature Selection
- Mutual Information → top 150
- Boruta (XGBoost GPU, 50 iterations) → 110 confirmed
- Union → **150 final features**
- Output: `enhanced/experiments/selected_features.json`, `mi_scores.csv`,
  `boruta_hits.csv`

### Phase 5 — Base Models (retrained on 150 selected temporal features)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | MCC | Notes |
|---|---|---|---|---|---|---|---|
| CatBoost | 0.7781 | 0.0676 | 65.8% | 4.6% | 0.086 | 0.124 | GPU, best overall |
| XGBoost | 0.7759 | 0.0672 | 64.0% | 4.8% | 0.089 | 0.126 | GPU |
| RandomForest | 0.7581 | 0.0592 | 31.9% | 6.5% | 0.108 | 0.109 | CPU |
| LightGBM | 0.7369 | 0.0556 | 25.8% | 6.7% | 0.106 | 0.100 | CPU (pip build has no GPU support) |

- ⚠️ LightGBM's pip build **does not support GPU** — trained on CPU with fixed
  500 iterations (early stopping on logloss was unreliable given 1.8% train
  positive rate — it kept stopping at iteration 1). If you want LightGBM on
  GPU, you'd need to build from source with `-DUSE_GPU=1`.

### Phase 6 — Stacking Ensemble
- Meta-learner: `LogisticRegression` on 4 base model val predictions
- Meta weights: CatBoost (3.26) > XGBoost (2.28) > LightGBM (0.28) > RandomForest (−0.51)
- **Test**: ROC-AUC 0.778, PR-AUC 0.070, Recall 69.4% (baseline recall: 55.25%)
- Output: `enhanced/models/meta_learner.pkl`, `stack_val_preds.npy`, `stack_test_preds.npy`

### Phase 7 — Probability Calibration
- Platt vs Isotonic compared on val
- Platt was essentially a no-op (meta-learner's own sigmoid was already close
  to optimal) — **Isotonic selected**
- Val: Brier 0.0173, ECE ~0.0000 (genuine — verified against test, not an
  artifact; see conversation history if the "why is ECE zero" question comes
  up again)
- Test: Brier 0.0171, ECE 0.0003 — generalizes well
- Output: `enhanced/models/calibrator.pkl`, `calibrated_test_preds.npy`,
  `calibration_info.json`

### Phase 8 — Clinical Threshold Selection
- Swept ~200 thresholds on calibrated val predictions
- Selection rule: minimum alarm rate subject to sensitivity ≥ 60%
  (this floor is a **placeholder constant**, `MIN_SENSITIVITY` in
  `threshold.py` — adjust and re-justify if your report needs a different
  clinical tradeoff)
- **Selected threshold: 0.0262**
- Test: Sensitivity 66.2%, Specificity 75.6%, Precision 4.7%, Alarm rate 25.2%
- ⚠️ **Important for the final report**: ~26% hourly alarm rate with ~4.7%
  precision means ~20 false alarms per true alarm. This is the expected
  consequence of a 1.8% positive base rate + a 60% recall floor, not a bug.
  Worth reporting alongside sensitivity, and worth computing a **patient-level**
  (not just hourly) alarm rate for the dashboard/report — it'll look more
  clinically meaningful than the raw hourly number.
- Output: `enhanced/models/optimal_threshold.json`, `enhanced/experiments/threshold_sweep.csv`

---

## Non-Negotiable Rules (still apply to Phases 9–11)

1. **No leakage** — anything fit on data must be fit on Train only
2. **Patient-level integrity** — the 28,234/6,051/6,051 patient split is final; don't resplit
3. **Temporal causality** — already respected in `temporal.py`; don't undo it
4. **Baseline preserved** — `baseline/` is read-only; compare against it in Phase 11
   (Baseline: PR-AUC 0.0714, ROC-AUC 0.7598, Recall 55.25%)

---

## What's Next

### Phase 9 — Explainable AI
```bash
python enhanced/xai/explain.py   # needs to be written
```
Spec: SHAP TreeExplainer (global, 500-patient sample + per-patient
waterfall/force plots) and LIME (per-patient bar charts). Use whichever base
model has the SHAP-compatible tree structure (CatBoost/XGBoost recommended)
or explain the stacked prediction via the meta-learner + base model SHAP
values combined — decide this before writing the script, it changes the
implementation.

### Phase 10 — Dashboard
```bash
streamlit run enhanced/dashboard/app.py   # needs to be written
```
Inputs: patient ID + ICU hour, or manual vitals entry.
Outputs: risk probability + LOW/MODERATE/HIGH category, vitals timeline, SHAP
global importance, LIME local explanation, model metadata.

### Phase 11 — Final Evaluation
```bash
python enhanced/experiments/final_eval.py   # needs to be written
```
Compare baseline vs enhanced on test set, generate `results_table.csv`,
figures, `final_report.md`. This is where the alarm-rate/precision tradeoff
from Phase 8 needs honest discussion, not just headline sensitivity.

---

## File Map (as of Phase 8)

```
enhanced/
├── data/
│   ├── audit.py, preprocessing_fast.py
│   └── processed/
│       ├── {train,val,test}_processed.parquet
│       ├── {train,val,test}_temporal.parquet
│       ├── temporal_feature_cols.json
│       └── split_info.json
├── features/
│   ├── temporal.py
│   └── selection.py
├── models/
│   ├── train_rf.py, train_xgb.py, train_lgbm.py, train_catboost.py
│   ├── _utils.py
│   ├── {rf,xgb,lgbm,catboost}_model.{pkl,cbm}
│   ├── {rf,xgb,lgbm,catboost}_val_preds.npy
│   ├── {rf,xgb,lgbm,catboost}_metrics.json
│   ├── meta_learner.pkl, stack_val_preds.npy, stack_test_preds.npy
│   ├── stack_{val,test}_metrics.json
│   ├── calibrator.pkl, calibrated_test_preds.npy, calibration_info.json
│   ├── optimal_threshold.json
│   └── transformers/
├── stacking/stack.py
├── calibration/calibrate.py, threshold.py
├── xai/                    # empty — Phase 9
├── dashboard/               # empty — Phase 10
└── experiments/
    ├── audit_report.md
    ├── selected_features.json, mi_scores.csv, boruta_hits.csv
    ├── calibration_curves.png
    ├── threshold_sweep.csv
    └── final_report.md      # empty — Phase 11
```

---

## Known Issues / Things to Watch

| Issue | Where | Status |
|---|---|---|
| LightGBM has no GPU in pip build | Phase 5 | Running on CPU, works fine, just slower |
| RandomForest recall much lower than others (32% vs 65%+) | Phase 5 | Not blocking — meta-learner already down-weights it (coef −0.51) |
| Temporal feature script is slow (~70 min) | Phase 3 | Works, just not optimized — leave as-is unless re-running from scratch |
| ECE bin-clipping edge case | `calibrate.py` | Fixed (clips `bin_indices` to valid range) — was a latent bug, didn't change results on this dataset but keep the fix in |
| High hourly alarm rate (~26%) at chosen threshold | Phase 8 | Expected given class imbalance — report honestly, compute patient-level rate too |

---

*Last updated after Phase 8. Update the checklist at the top whenever you
finish a phase, so whoever picks this up next (including a fresh AI session)
knows exactly where to resume without re-reading the whole history.*

---

## 🎯 Model Improvement Plan (Post-Phase 8)

**Goal**: Beat baseline PR-AUC 0.0714, ROC-AUC 0.7598, Recall 55.25%
**Current best (Phase 6-8)**: ROC-AUC 0.778, PR-AUC 0.070, Recall 66.2%

### Root Cause Identified
**Critical bug in `_utils.py:load_data()`**: Training loads `_temporal.parquet` (has NaNs) and uses `.fillna(0)` instead of the properly fitted MICE imputer from Phase 2. Labs with >90% missingness get filled with 0 (clinically wrong — Lactate=0 means normal, not missing).

---

### Implementation Order (Do Sequentially, Verify Each Step)

#### Step 1: Fix Imputation Bug (Highest Impact — ~15 min)
**File**: `enhanced/models/_utils.py`
**Change**: `load_data()` to load `_processed.parquet` (already imputed + scaled) instead of `_temporal.parquet`
```python
# BEFORE (buggy):
train = pd.read_parquet(PROCESSED / "train_temporal.parquet")
X_train = train[available].fillna(0).values.astype(np.float32)

# AFTER (fixed):
train = pd.read_parquet(PROCESSED / "train_processed.parquet")
X_train = train[available].values.astype(np.float32)  # No fillna(0)!
```
**Then retrain**: All 4 base models → re-stack → re-calibrate → re-threshold
**Expected**: PR-AUC +0.01-0.02, ROC-AUC +0.01, Recall +3%

---

#### Step 2: Better Class Imbalance Handling
**Files**: `enhanced/models/train_xgb.py`, `train_catboost.py`, `train_lgbm.py`

**XGBoost** — Add focal loss custom objective (gamma=2.0, alpha=0.25)
**CatBoost** — Use `auto_class_weights='Balanced'` + sampling
**LightGBM** — Enable early stopping on `average_precision` (PR-AUC) metric
```python
# LightGBM fix:
params = {
    "metric": "average_precision",  # PR-AUC for early stopping
    "is_unbalance": True,           # Native imbalance handling
    "early_stopping_rounds": 50,
    # ...
}
```

---

#### Step 3: Reduce Feature Count (150 → 80)
**File**: `enhanced/features/selection.py`
**Change**: `top_k=150` → `top_k=80` in `step1_mutual_info()`
**Rationale**: Remove noisy features; Boruta already confirmed only ~24

---

#### Step 4: Improve Stacking
**File**: `enhanced/stacking/stack.py`

A. **Calibrate base model predictions before stacking**:
```python
from sklearn.calibration import CalibratedClassifierCV
for name in ['rf', 'xgb', 'lgbm', 'catboost']:
    cal = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    cal.fit(X_val, y_val)
    val_probs_cal[name] = cal.predict_proba(X_val)[:, 1]

X_meta = np.column_stack([val_probs_cal[n] for n in MODEL_NAMES])
```

B. **Better meta-learner** (replace LogisticRegression):
```python
# Option: GradientBoostingClassifier
meta = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05)
# Or: XGBoost meta-learner
```

---

#### Step 5: Add LSTM as 5th Base Model
**New file**: `enhanced/models/train_lstm.py`
- Patient sequences (pad to 48h)
- LSTM captures true temporal dynamics tree models miss
- Add to stacking ensemble

---

### Verification Checklist Per Step

| Step | Verify On Val | Target Improvement |
|------|---------------|-------------------|
| 1. Fix imputation | PR-AUC, ROC-AUC, Recall | PR-AUC > 0.08 |
| 2. Imbalance handling | PR-AUC, Recall | PR-AUC > 0.085 |
| 3. Feature reduction | PR-AUC, Precision | Precision > 5% |
| 4. Better stacking | PR-AUC, ROC-AUC | PR-AUC > 0.09 |
| 5. LSTM | All metrics | PR-AUC > 0.095 |

---

### Baseline Comparison Target

| Metric | Baseline | Current Best | Target After Fixes |
|--------|----------|--------------|-------------------|
| **PR-AUC** | **0.0714** | 0.070 | **> 0.095** |
| **ROC-AUC** | **0.7598** | 0.778 | **> 0.81** |
| **Recall** | **55.25%** | 66.2% | **> 75%** |
| **Precision** | — | 4.7% | **> 7%** |
| **F1** | — | 0.087 | **> 0.12** |

---

### Execution Commands (After Each Step)

```bash
cd C:\PROJECT

# Step 1: Fix _utils.py, then retrain all 4 models
python enhanced/models/train_rf.py
python enhanced/models/train_xgb.py
python enhanced/models/train_lgbm.py
python enhanced/models/train_catboost.py

# Re-stack
python enhanced/stacking/stack.py

# Re-calibrate
python enhanced/calibration/calibrate.py

# Re-threshold
python enhanced/calibration/threshold.py
```


# Sepsis Early-Warning Model — Project Brief

**Base paper being differentiated from:** Santos et al., "Interpretable Machine
Learning Model Based on SOFA Score for ICU Sepsis Mortality Prediction with
Multicenter Validation" (IEEE Latin America Transactions, Dec 2025) —
retrospective **mortality** prediction on MIMIC-IV/eICU using static,
stay-level-aggregated SOFA variables, CatBoost, OR-based feature selection,
global SHAP.

**Our project:** real-time, hourly **sepsis onset** prediction (PhysioNet/CinC
2019 Challenge dataset), using causal temporal feature engineering — a
structurally different and arguably more clinically actionable task than the
base paper's one-shot mortality prediction.

---

## 1. Dataset & Pipeline (Phases 1–8) — DONE

### Phase 1 — Data Audit
- PhysioNet 2019 Challenge (setA + setB combined): 40,336 patients, 1,552,210
  hourly records
- Sepsis rate: 7.27% patient-level (1.8% row/hour-level — this is the
  imbalance the whole pipeline has to handle)

### Phase 2 — Preprocessing
- Patient-level split 70/15/15 → 28,234 / 6,051 / 6,051 patients, no leakage
- IQR capping (1.5×IQR), fit on train only
- Imputation: KNN vs MICE benchmarked → MICE selected (fit on train only)
- Per-column StandardScaler/RobustScaler, fit on train only
- Missingness indicators added as features

### Phase 3 — Temporal Feature Engineering
- Lags (t-1/3/6), diffs, rolling stats (mean/std/min/max @ 3h/6h/12h), trends
- Causal only (hour ≤ t, no leakage)
- 347 features from 19 base columns (HR, O2Sat, Temp, SBP, MAP, DBP, Resp,
  FiO2, pH, PaCO2, SaO2, BUN, Calcium, Glucose, Potassium, Hct, Hgb, WBC,
  Platelets + Age, Gender, Unit1, Unit2, HospAdmTime)
- Structural NaNs (insufficient history for rolling windows) are
  **intentionally filled with 0** ("no history yet") — consistent choice
  across train/val/test, confirmed correct after today's debugging

### Phase 4 — Feature Selection
- Mutual Information → top 150, Boruta (XGBoost GPU) → 110 confirmed
- Union → 150 final features used for all base models

### Phase 5 — Base Models + Class Imbalance Ablation ⭐ (strongest methodology result)
Four models trained (XGBoost, LightGBM, CatBoost, RandomForest), each
originally using their library's **native full-strength auto-balancing**
(`scale_pos_weight` at full ratio, `is_unbalance=True`,
`auto_class_weights='Balanced'`, `class_weight='balanced'`). All four were
found to **over-correct** — pushing recall up and precision/PR-AUC down.

**Fix applied uniformly:** explicit mild class weighting at
`√(n_neg/n_pos)` ≈ 7.39 instead of the full ratio (54.65), for all four
models. This consistently outperformed native auto-balancing:

| Model | PR-AUC (native auto-balance) | PR-AUC (mild √ratio weight) |
|---|---|---|
| XGBoost | 0.0672 | **0.0698** |
| LightGBM | 0.0556 → 0.0610 (is_unbalance) | **0.0678** |
| CatBoost | 0.0676 → 0.0677–0.0682 (Balanced) | **0.0709** |
| RandomForest | 0.0592 | **0.0630** |

This is a clean, generalizable, four-model-consistent ablation — worth its
own subsection in the paper.

Also fixed along the way: an early attempt at focal loss for XGBoost was
abandoned (custom gradient/hessian was numerically unstable in float32 —
finite-difference step size too small, produced garbage gradients); LightGBM
early-stopping was silently tracking the wrong metric (first metric in a
multi-metric list, not the best one) until narrowed to a single metric
(`average_precision`).

### Phase 6 — Stacking Ensemble
- Meta-learner: LogisticRegression on 4 base models' predictions
- **Fixed a real leakage bug:** original version fit the meta-learner on val
  predictions and then scored it on the *same* val predictions (not a valid
  validation). Rebuilt with proper 5-fold **out-of-fold** meta-training.
- Also fixed a `fillna(0)` vs MICE-imputer confusion for test-time NaN
  handling — resolved by matching train/val's existing `fillna(0)`
  convention for structural NaNs, not introducing a mismatched treatment.

**Final honest results:**
- Out-of-fold val PR-AUC: **0.0736** (above baseline 0.0714)
- Test PR-AUC: **0.0702** (essentially flat vs. baseline — reported honestly,
  not oversold)
- Test ROC-AUC: **0.7838** (baseline 0.7598) ✅ clear improvement
- Test Recall: **65.6%** (baseline 55.25%) ✅ clear improvement

### Phase 7 — Probability Calibration
- Platt vs Isotonic compared → Isotonic selected
- Test: Brier 0.0171, ECE 0.0007–0.0009 (well-calibrated)
- The base paper does **no calibration at all** — a genuine gap we close

### Phase 8 — Clinical Threshold Selection
- Rule: minimize alarm rate subject to sensitivity ≥ 60%
- Selected threshold: 0.0262
- Test: Sensitivity 65.6%, Specificity 76.9%, Precision 4.9%, Alarm rate
  23.9% (~1 true alert per ~20 false ones — expected consequence of 1.8%
  base rate + 60% recall floor, reported honestly, not hidden)

**A full clean end-to-end rerun (Phases 5→8) was done today after finding
several stale-file bugs during debugging — confirmed reproducible, numbers
above are the final, trustworthy ones.**

---

## 2. Debugging Notes Worth Remembering
- Several rounds of edits not landing / files getting cross-contaminated
  between scripts during manual editing — always verify a printed
  config/header line matches the intended change before trusting a run's
  output.
- Stale intermediate files (e.g. old `stack_val_preds.npy` after switching
  to `stack_oof_val_preds.npy`) caused a serious silent bug (val/test
  threshold mismatch: 75% vs 0.2% sensitivity) — resolved, and a full
  clean-slate rerun was done to confirm no other stale files remain.

---

## 3. Final Baseline vs Enhanced Comparison (for the paper)

| Metric | Baseline | Enhanced (Test) | Change |
|---|---|---|---|
| ROC-AUC | 0.7598 | **0.7838** | ✅ +0.024 |
| PR-AUC | 0.0714 | **0.0702** (test) / 0.0736 (val OOF) | ≈ flat |
| Sensitivity | 55.25% | **65.6%** | ✅ +10.3pp |
| Precision | — | 4.91% | — |
| Alarm Rate | — | 23.9% | — |

**Honest framing for the paper:** ROC-AUC and recall clearly improve;
PR-AUC is essentially matched rather than clearly beaten — report both val
and test numbers, don't cherry-pick. Combined with calibration and
interpretability (once done) that the base paper lacks, this is a
defensible result for a mid-tier venue submission.

---

## 4. SOFA / Clinical Score Decision
The base paper's SOFA score needs 6 components: respiratory (PaO2/FiO2),
coagulation (platelets), liver (bilirubin), cardiovascular (MAP +
vasopressor dose), CNS (GCS), renal (creatinine + urine output). Our
19-column feature set is **missing bilirubin, GCS, creatinine, urine
output, and vasopressor data** — a full/faithful SOFA replication is not
possible and was ruled out (would misrepresent what's being computed).

**Decided direction:** use **SIRS** (fully computable — Temp, HR, Resp, WBC
all present) as the primary composite clinical score, plus a **2-component
modified qSOFA** (Resp≥22, SBP≤100 — missing the GCS-based altered-mental-
status component, to be explicitly disclosed as a limitation). Not yet
implemented in code — deprioritized below Phase 9/external validation per
current plan, but the reasoning and choice are locked in.

---

## 5. What We're Doing Next (in order)

1. **External validation split (setA vs setB)** ← starting now
   - PhysioNet 2019 data combines two hospital systems (setA and setB).
     Currently pooled into one 70/15/15 patient-level split. Plan: check
     whether setA/setB origin survives in the data, and if so, re-split so
     setA trains and setB tests (or report both), directly mirroring the
     base paper's MIMIC-IV→eICU external validation — their headline
     methodological strength. Low additional modeling cost, high value for
     rigor.

2. **Phase 9 — Explainability (SHAP)**
   - Global SHAP (TreeExplainer) on best base model (CatBoost, PR-AUC
     0.0709) — directly comparable to the base paper's beeswarm plot
   - Stretch goal: SHAP importance at different points in ICU-LOS (e.g.
     hour t-12 vs t-6 vs t-1 before onset) — shows feature importance
     shifting as a patient approaches onset, something the base paper's
     static model structurally cannot do. This is our strongest
     interpretability novelty claim.

3. **Frontend + Backend (after Phase 9)**
   - Open stack choice (not mandated by course) — leaning FastAPI backend
     (wraps existing saved models/transformers, replicates the Phase
     2→8 inference pipeline for new uploads) + React frontend
   - Scope: CSV/vitals upload → live risk prediction + SHAP explanation +
     risk category, not just a static dashboard
   - Backend is the harder/more important half — must replicate training
     preprocessing exactly to avoid train/serve skew

4. **LSTM as a 5th base model** (if time permits)
   - Best remaining lever to genuinely move PR-AUC past baseline rather
     than just matching it (feature/threshold tuning alone is unlikely to
     get there)
   - Real scope: patient-level sequence construction (padded/masked),
     raw (not pre-engineered) features per timestep, causal-safe, mild
     class weighting (~7.39, consistent with the rest of the ablation),
     integrate its predictions as a 5th column into the stack
   - Honest expectation: plausible meaningful PR-AUC gain, not guaranteed

5. **qSOFA/SIRS features + feature reduction (150→80)** (if time permits)
   - Lower expected impact than the above, useful for methodology-section
     depth and one more ablation table, but shouldn't block writing

6. **Write the paper**
   - Structure: Data & Task → Preprocessing → Temporal Feature Engineering
     → Feature Selection → Base Models & Imbalance Ablation → Stacking
     (leak-safe) → Calibration → Clinical Threshold Selection → (SIRS/qSOFA
     if done) → Interpretability (SHAP) → Results → Discussion (honest
     PR-AUC framing) → external validation results
   - Explicit framing throughout: this extends the base paper's static,
     retrospective mortality-prediction paradigm to real-time, temporal,
     causal sepsis-onset prediction — a harder, more clinically actionable
     task, at a harder class-imbalance level (1.8% vs their ~12%)

---

*Last updated: end of today's debugging + planning session. Next action:
check setA/setB tracking in Phase 1/2 code before re-spli


## Phase 9 — Explainable AI (SHAP + LIME)

**Goal**: explain *why* the model flags a patient as high-risk for sepsis,
not just *that* it does — this is required for the paper's interpretability
section (directly compared against the base paper's SHAP analysis).

### What this phase does
1. **Global SHAP beeswarm** — which features matter most across the whole
   test set, on the CatBoost model (our best single base model).
2. **Temporal SHAP** (this is our own addition, not in the base paper) —
   shows whether feature importance changes between early ICU hours,
   mid-stay, and late-stay. This is the key novelty argument: the base
   paper's SOFA model only ever sees one static snapshot per patient, so
   it structurally cannot show this. We can, because our pipeline is
   temporal from the ground up.
3. **Per-patient explanations** — pick one correctly-caught sepsis case
   and one correctly-cleared non-sepsis case, explain both with SHAP
   *and* LIME (two independent explanation methods) — useful for the
   paper's "case study" figure and for sanity-checking that the model's
   reasoning makes clinical sense.

### How to run it
```bash
pip install shap lime --break-system-packages
python enhanced/xai/explain.py
```

Takes a few minutes (SHAP on 500 patients + temporal binning + LIME).
All outputs land in `enhanced/experiments/xai/`.

### What to check after running
- Open `global_shap_beeswarm.png` — sanity check: do the top features
  make clinical sense (vitals like Resp/MAP/HR should dominate, similar
  to the base paper's urine output/respiration rate findings)?
- Open `temporal_shap_importance.png` — does anything meaningfully shift
  between early/mid/late ICU stay? If yes, that's a good figure for the
  paper's discussion section.
- Open the two `patient_waterfall_*.png` and `patient_lime_*.png` pairs —
  do SHAP and LIME roughly agree on what drove each prediction? If they
  disagree a lot, flag it — worth discussing rather than hiding.

### Known things to double check while running
- If `explainer.expected_value` throws an error or looks like a list
  instead of a single number, print it first (`print(explainer.expected_value)`)
  — CatBoost's SHAP output format has changed across versions, may need
  `expected_value[0]` instead of `expected_value` depending on your
  installed `shap`/`catboost` versions.
- If the run is slow, reduce `N_GLOBAL_SAMPLE` at the top of the script
  (500 → 200) — global SHAP scales with sample size, temporal SHAP is
  already capped per bin.

### Don't touch
- The `fillna(0)` calls in this script match the exact same convention
  used everywhere else in the pipeline (structural NaN = "no history
  yet") — do not swap this for any other imputation here, it needs to
  stay consistent with how the models were trained.