"""
Phase 5 — Model 2: LightGBM (GPU)
====================================
GPU: device='gpu'  (RTX 4060 via OpenCL)
Class imbalance: scale_pos_weight = n_neg / n_pos
Early stopping: 50 rounds on val PR-AUC
"""
import json, time, joblib
import numpy as np
import lightgbm as lgb
from pathlib import Path
from _utils import load_data, evaluate, print_metrics, MODELS_DIR

print("=" * 60)
print("Phase 5 — LightGBM (GPU: RTX 4060)")
print("=" * 60)

X_train, y_train, X_val, y_val, features, spw = load_data()

params = {
    "objective"         : "binary",
    "metric"            : ["binary_logloss", "average_precision"],
    "n_estimators"      : 2000,
    "learning_rate"     : 0.05,
    "num_leaves"        : 63,
    "max_depth"         : -1,
    "min_child_samples" : 50,
    "subsample"         : 0.8,
    "subsample_freq"    : 1,
    "colsample_bytree"  : 0.8,
    "reg_alpha"         : 0.1,
    "reg_lambda"        : 1.0,
    "scale_pos_weight"  : spw,
    "random_state"      : 42,
    "n_jobs"            : -1,
    "verbose"           : -1,
}

print(f"\nTraining LightGBM | scale_pos_weight={spw:.2f} | Multi-threaded CPU (n_jobs=-1)")
t0 = time.time()

callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)]

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks,
)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s  |  Best iter: {model.best_iteration_}")

# Evaluate
val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("LightGBM", metrics)

# Save
joblib.dump(model, MODELS_DIR / "lgbm_model.pkl")
np.save(MODELS_DIR / "lgbm_val_preds.npy", val_probs)
metrics["best_iteration"] = int(model.best_iteration_)
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
with open(MODELS_DIR / "lgbm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODELS_DIR}/lgbm_model.pkl")
print(f"  {MODELS_DIR}/lgbm_val_preds.npy")
print(f"  {MODELS_DIR}/lgbm_metrics.json")
print("\nLightGBM DONE")
