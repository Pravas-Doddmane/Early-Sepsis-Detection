# 🏥 Enhanced Early Sepsis Detection System

An end-to-end clinical machine learning pipeline for early prediction of sepsis onset (within 6 hours) on the **PhysioNet / Computing in Cardiology Challenge 2019** dataset.

Built with **GPU-accelerated gradient boosting (NVIDIA RTX 4060)**, causal temporal feature engineering, hybrid feature selection, probability calibration, and explainable AI (SHAP & LIME).

---

## 📊 Performance Benchmark vs Baseline

| Model / Baseline | ROC-AUC | PR-AUC | Recall (Sensitivity) | Precision | F1 Score | Training Time | Acceleration |
|---|---|---|---|---|---|---|---|
| **Challenge Baseline** | 0.7598 | **0.0714** | 55.25% | — | — | — | CPU |
| **CatBoost (Enhanced)** | **0.7685** | 0.0631 | **64.29%** | 0.0427 | 0.0801 | **4.2s** | ✅ GPU (RTX 4060) |
| **XGBoost (Enhanced)** | **0.7626** | **0.0664** | **62.11%** | 0.0445 | **0.0831** | **3.3s** | ✅ GPU (RTX 4060) |
| **RandomForest** | 0.7407 | 0.0545 | 6.74% | **0.1071** | 0.0827 | 105.5s | CPU (24 threads) |
| **LightGBM** | 0.7089 | 0.0415 | 0.00% | 0.0000 | 0.0000 | 3.6s | CPU (Multi-core) |

> 🚀 **Recall improvement**: CatBoost and XGBoost improved patient sensitivity to **64.29%** and **62.11%** (vs 55.25% baseline) while evaluating at standard 0.5 threshold. Phase 6 (Stacking) and Phase 8 (Clinical Threshold Optimization) will further optimize PR-AUC and calibrated decision curves.

---

## 🗂️ Project Pipeline & Progress

| Phase | Description | Status | Key Artifacts / Results |
|---|---|---|---|
| **Phase 1** | Data Audit & Profiling | ✅ Completed | 40,336 patients, 1.55M rows, 7.27% patient sepsis rate (`audit_report.md`) |
| **Phase 2** | Preprocessing & Imputation | ✅ Completed | Patient-level 70/15/15 split, MICE imputation, per-column scaling |
| **Phase 3** | Temporal Feature Engineering | ✅ Completed | 309 causal temporal features (lags, diffs, rolling stats, slopes) |
| **Phase 4** | Hybrid Feature Selection | ✅ Completed | 109 final features selected via GPU-Boruta + Mutual Information |
| **Phase 5** | 4 Base Model Training | ✅ Completed | XGBoost (GPU), CatBoost (GPU), LightGBM, RandomForest trained & evaluated |
| **Phase 6** | Stacking Ensemble | 🔜 Next | Logistic Regression meta-learner combining out-of-fold / val predictions |
| **Phase 7** | Probability Calibration | ⏳ Upcoming | Platt Scaling & Isotonic Regression for clinical risk calibration |
| **Phase 8** | Clinical Threshold Selection | ⏳ Upcoming | Decision curve analysis (Sensitivity ≥ 80%, Alarm Rate ≤ 20%) |
| **Phase 9** | Explainable AI (XAI) | ⏳ Upcoming | SHAP TreeExplainer (global/local) & LIME patient explanations |
| **Phase 10** | Interactive Dashboard | ⏳ Upcoming | Streamlit clinical decision support dashboard |
| **Phase 11** | Final Evaluation | ⏳ Upcoming | Independent test set evaluation and final report |

---

## 🛠️ Installation & Environment Setup

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU (RTX 4060 8GB tested and supported with CUDA)

### 2. Setup Virtual Environment
```powershell
# Clone or navigate to the repository
cd "c:\Users\prava\7th sem major project\Project 4\Early-Sepsis-Detection"

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run the Pipeline

### Phase 1: Data Audit
```powershell
python enhanced/data/audit.py
```
- Consolidates all 40,336 `.psv` files into `raw_combined.parquet`.
- Generates `enhanced/experiments/audit_report.md`.

### Phase 2: Fast Preprocessing
```powershell
python enhanced/data/preprocessing_fast.py
```
- Performs stratified patient-level train/val/test splits (no data leakage).
- Fits IQR capper, MICE imputer, and scalers on train set only.
- Outputs `train_processed.parquet`, `val_processed.parquet`, `test_processed.parquet`.

### Phase 3: Temporal Feature Engineering
```powershell
python enhanced/features/temporal.py
```
- Generates causal features per patient (lags `t-1, t-3, t-6`, 1h/3h diffs, rolling mean/std/min/max over 3h/6h/12h windows, and linear slopes).
- Outputs `train_temporal.parquet` (1.83 GB), `val_temporal.parquet`, `test_temporal.parquet`.

### Phase 4: Hybrid Feature Selection (GPU-Accelerated)
```powershell
python enhanced/features/selection.py
```
- Runs Mutual Information filter + Boruta with GPU XGBoost estimator.
- Outputs `enhanced/experiments/selected_features.json` (109 features) and `feature_importance.csv`.

### Phase 5: Base Model Training
Run scripts from the `enhanced/models/` folder:

```powershell
cd enhanced/models

# 1. XGBoost (GPU - RTX 4060)
python train_xgb.py

# 2. CatBoost (GPU - RTX 4060)
python train_catboost.py

# 3. LightGBM (Multi-threaded CPU)
python train_lgbm.py

# 4. RandomForest (Multi-threaded CPU)
python train_rf.py
```

Outputs saved in `enhanced/models/`:
- Model checkpoints: `xgb_model.pkl`, `catboost_model.cbm`, `lgbm_model.pkl`, `rf_model.pkl`
- Validation probability predictions: `*_val_preds.npy`
- Validation metrics: `*_metrics.json`

---

## 📁 Repository Structure

```text
Early-Sepsis-Detection/
├── dataset/physionet_sepsis/training/    # Raw .psv challenge records
├── baseline/                             # Read-only challenge baseline
├── enhanced/
│   ├── data/
│   │   ├── audit.py                      # Phase 1: Audit script
│   │   ├── preprocessing_fast.py         # Phase 2: Fast preprocessor
│   │   └── processed/                    # Processed & temporal Parquet files
│   ├── features/
│   │   ├── temporal.py                   # Phase 3: Temporal feature extraction
│   │   └── selection.py                  # Phase 4: Boruta + MI selection
│   ├── models/
│   │   ├── _utils.py                     # Shared evaluation & data loader
│   │   ├── train_xgb.py                  # Phase 5: XGBoost (GPU)
│   │   ├── train_catboost.py             # Phase 5: CatBoost (GPU)
│   │   ├── train_lgbm.py                 # Phase 5: LightGBM
│   │   ├── train_rf.py                   # Phase 5: RandomForest
│   │   └── transformers/                 # Saved preprocessing artifacts (.pkl)
│   ├── stacking/                         # Phase 6: Stacking ensemble (next)
│   ├── calibration/                      # Phase 7 & 8: Calibration & thresholds
│   ├── xai/                              # Phase 9: SHAP & LIME explainability
│   ├── dashboard/                        # Phase 10: Streamlit decision app
│   └── experiments/                      # Experiment metrics, plots & reports
├── PROJECT_GUIDE.md                      # Comprehensive developer & phase guide
├── PROJECT_CONTEXT.md                    # Quick session context
├── requirements.txt                      # Project dependencies
└── README.md                             # Project overview & instructions
```

---

## 🔬 Next Steps: Phase 6
Proceed to **Phase 6: Stacking Ensemble** to combine predictions from all 4 models into a meta-learner for enhanced discrimination and calibration.
