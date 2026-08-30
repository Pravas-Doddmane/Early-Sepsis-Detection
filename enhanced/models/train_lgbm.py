"""
Phase 5 — Model 2: LightGBM (CPU) with PR-AUC Early Stopping
=============================================================
CPU: n_jobs=-1 (LightGBM pip build lacks GPU)
Class imbalance: mild scale_pos_weight (sqrt of full ratio) — matches
  the fix that worked for XGBoost; is_unbalance=True over-corrected
  (PR-AUC 0.0610 vs 0.0669 with mild scale_pos_weight).
Early stopping: 50 rounds on val average_precision (PR-AUC), single
  metric only — do not add binary_logloss back into the metric list,
  LightGBM early-stops on the FIRST listed metric, not the best one.
"""
import json, time, joblib
import numpy as np
import lightgbm as lgb
from pathlib import Path
from _utils import load_data, evaluate, print_metrics, MODELS_DIR

print("=" * 60)
print("Phase 5 — LightGBM (CPU: all cores) + PR-AUC Early Stopping")
print("=" * 60)

X_train, y_train, X_val, y_val, features, spw = load_data()

spw_mild = float(np.sqrt(spw))  # e.g. 54.65 -> ~7.39

params = {
    "objective"         : "binary",
    "metric"            : "average_precision",   # single metric — no ambiguity
    "n_estimators"      : 2000,
    "learning_rate"     : 0.05,
    "num_leaves"        : 127,
    "max_depth"         : 8,
    "min_child_samples" : 50,
    "subsample"         : 0.8,
    "subsample_freq"    : 1,
    "colsample_bytree"  : 0.8,
    "reg_alpha"         : 0.1,
    "reg_lambda"        : 1.0,
    "scale_pos_weight"  : spw_mild,   # mild — replaces is_unbalance
    "random_state"      : 42,
    "n_jobs"            : -1,
    "verbose"           : -1,
    "force_col_wise"    : True,
}

print(f"\nTraining LightGBM | scale_pos_weight={spw_mild:.2f} (sqrt of full "
      f"ratio {spw:.2f}) | NO is_unbalance | CPU | Early stopping on PR-AUC")
t0 = time.time()

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(50),
    ],
)

elapsed = time.time() - t0
best_iter = model.best_iteration_
print(f"\nTraining done in {elapsed:.1f}s  |  Best iter: {best_iter}")

val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("LightGBM", metrics)

joblib.dump(model, MODELS_DIR / "lgbm_model.pkl")
np.save(MODELS_DIR / "lgbm_val_preds.npy", val_probs)
metrics["best_iteration"] = int(best_iter)
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
metrics["scale_pos_weight_used"] = spw_mild
with open(MODELS_DIR / "lgbm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODELS_DIR}/lgbm_model.pkl")
print(f"  {MODELS_DIR}/lgbm_val_preds.npy")
print(f"  {MODELS_DIR}/lgbm_metrics.json")
print("\nLightGBM DONE")