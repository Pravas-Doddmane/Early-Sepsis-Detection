"""
Phase 5 — Model 2: LightGBM (CPU)
====================================
CPU: n_jobs=-1 (LightGBM pip build lacks GPU)
Class imbalance: scale_pos_weight = n_neg / n_pos
Train fixed iterations (early stopping on logloss fails for extreme imbalance)
"""
import json, time, joblib
import numpy as np
import lightgbm as lgb
from pathlib import Path
from _utils import load_data, evaluate, print_metrics, MODELS_DIR

print("=" * 60)
print("Phase 5 — LightGBM (CPU: all cores)")
print("=" * 60)

X_train, y_train, X_val, y_val, features, spw = load_data()

params = {
    "objective"         : "binary",
    "metric"            : ["binary_logloss", "average_precision"],
    "n_estimators"      : 500,
    "learning_rate"     : 0.05,
    "num_leaves"        : 127,
    "max_depth"         : 8,
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
    "force_col_wise"    : True,
}

print(f"\nTraining LightGBM | scale_pos_weight={spw:.2f} | CPU (n_jobs=-1) | 500 fixed iter")
t0 = time.time()

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_X=X_val,
    eval_y=y_val,
    callbacks=[lgb.log_evaluation(50)],
)

elapsed = time.time() - t0
best_iter = 500
print(f"\nTraining done in {elapsed:.1f}s  |  Iterations: {best_iter}")

# Evaluate
val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("LightGBM", metrics)

# Save
joblib.dump(model, MODELS_DIR / "lgbm_model.pkl")
np.save(MODELS_DIR / "lgbm_val_preds.npy", val_probs)
metrics["best_iteration"] = best_iter
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
with open(MODELS_DIR / "lgbm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODELS_DIR}/lgbm_model.pkl")
print(f"  {MODELS_DIR}/lgbm_val_preds.npy")
print(f"  {MODELS_DIR}/lgbm_metrics.json")

print("\nLightGBM DONE")