"""
Phase 5 — Model 1: XGBoost (GPU)
==================================
GPU: tree_method='hist', device='cuda'  (RTX 4060)
Class imbalance: scale_pos_weight = n_neg / n_pos
Early stopping: 50 rounds on val PR-AUC
"""
import json, time, joblib
import numpy as np
import xgboost as xgb
from pathlib import Path
from _utils import load_data, evaluate, print_metrics, MODELS_DIR, EXPERIMENTS

print("=" * 60)
print("Phase 5 — XGBoost (GPU: RTX 4060)")
print("=" * 60)

X_train, y_train, X_val, y_val, features, spw = load_data()

params = {
    "objective"        : "binary:logistic",
    "eval_metric"      : ["logloss", "aucpr"],
    "tree_method"      : "hist",
    "device"           : "cuda",          # RTX 4060
    "n_estimators"     : 2000,
    "learning_rate"    : 0.05,
    "max_depth"        : 6,
    "min_child_weight" : 10,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "gamma"            : 1,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "scale_pos_weight" : spw,
    "random_state"     : 42,
    "verbosity"        : 1,
    "early_stopping_rounds": 50,
}

print(f"\nTraining XGBoost | scale_pos_weight={spw:.2f} | GPU=cuda")
t0 = time.time()

model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s  |  Best iter: {model.best_iteration}")

# Evaluate
val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("XGBoost", metrics)

# Save
joblib.dump(model, MODELS_DIR / "xgb_model.pkl")
np.save(MODELS_DIR / "xgb_val_preds.npy", val_probs)
metrics["best_iteration"] = int(model.best_iteration)
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
with open(MODELS_DIR / "xgb_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODELS_DIR}/xgb_model.pkl")
print(f"  {MODELS_DIR}/xgb_val_preds.npy")
print(f"  {MODELS_DIR}/xgb_metrics.json")
print("\nXGBoost DONE")
