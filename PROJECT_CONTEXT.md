# Enhanced Sepsis Prediction — Project Context

## Objective
Build an enhanced early sepsis prediction system on the PhysioNet / Computing in Cardiology Challenge 2019 dataset, significantly improving discrimination, calibration, and clinical early warning capability over the baseline (PR-AUC 0.0714, ROC-AUC 0.7598, Recall 55.25%).

---

## Final Project Status: 100% COMPLETE (Phases 1–11)

| Phase | Description | Key Artifacts | Status |
|---|---|---|---|
| **Phase 1: Data Audit** | Audited 40,336 patients, 1.55M rows | `raw_combined.parquet`, `audit_report.md` | ✅ Complete |
| **Phase 2: Preprocessing** | Stratified 70/15/15 split, IQR capping, MICE | `train_processed.parquet`, transformers | ✅ Complete |
| **Phase 3: Temporal Features** | 309 causal features (lags, rolling stats, slopes) | `train_temporal.parquet`, `val_temporal.parquet` | ✅ Complete |
| **Phase 4: Feature Selection** | Boruta + Mutual Information | `selected_features.json` (150 features) | ✅ Complete |
| **Phase 5: Base Models** | Trained RF, XGBoost, LightGBM, CatBoost | `*_model.pkl`, `*_model.cbm`, `*_metrics.json` | ✅ Complete |
| **Phase 6: Stacking Ensemble** | LogisticRegression meta-learner (ROC-AUC 0.7838) | `meta_learner.pkl`, `stack_test_preds.npy` | ✅ Complete |
| **Phase 7: Probability Calibration** | Isotonic Calibration (Brier: 0.0171, ECE: 0.00089) | `calibrator.pkl`, `calibration_info.json` | ✅ Complete |
| **Phase 8: Threshold Optimization** | Optimal threshold $T = 0.0262$ (Sensitivity 65.6%) | `optimal_threshold.json` | ✅ Complete |
| **Phase 9: Explainable AI** | SHAP TreeExplainer & LIME explanations | `enhanced/xai/explain.py` | ✅ Complete |
| **Phase 10: Interactive Dashboard** | Streamlit Clinical Surveillance & Risk App | `enhanced/dashboard/app.py` | ✅ Complete |
| **Phase 11: Final Evaluation** | Benchmark report, results table & figures | `results_table.csv`, `final_report.md`, figures | ✅ Complete |

---

## Benchmark Results Comparison (Test Set)

| Model | ROC-AUC | PR-AUC | Recall (Sensitivity) | Precision | F1-Score | Brier Score | Operating Threshold |
|---|---|---|---|---|---|---|---|
| **Challenge Baseline** | 0.7598 | **0.0714** | 55.25% | 2.10% | 0.0404 | 0.0380 | 0.5000 |
| **Random Forest** | 0.7670 | 0.0630 | 0.05% | 22.22% | 0.0010 | 0.0253 | 0.5000 |
| **XGBoost** | 0.7792 | 0.0698 | 4.63% | 16.85% | 0.0727 | 0.0282 | 0.5000 |
| **LightGBM** | 0.7740 | 0.0678 | 4.92% | 17.15% | 0.0765 | 0.0274 | 0.5000 |
| **CatBoost** | 0.7808 | 0.0709 | 5.02% | 16.84% | 0.0773 | 0.0294 | 0.5000 |
| **Stacked Ensemble (Raw)** | **0.7838** | 0.0702 | 0.02% | 8.33% | 0.0005 | 0.0172 | 0.5000 |
| **Calibrated Ensemble @ Clinical Threshold** | **0.7838** | 0.0702 | **65.59%** | **4.91%** | **0.0914** | **0.0171** | **0.0262** |

---

## How to Run

### 1. Launch the Clinical Dashboard
```powershell
streamlit run enhanced/dashboard/app.py
```

### 2. Generate Final Evaluation Reports
```powershell
python enhanced/experiments/final_eval.py
```