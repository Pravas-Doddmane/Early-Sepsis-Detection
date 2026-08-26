"""
Phase 5 — Model 3: CatBoost (GPU)
====================================
GPU: task_type='GPU', devices='0'  (RTX 4060)
Class imbalance: scale_pos_weight = n_neg / n_pos
Early stopping: 50 rounds on val Logloss
"""
import json, time, joblib
import numpy as np
from catboost import CatBoostClassifier, Pool
from _utils import load_data, evaluate, print_metrics, MODELS_DIR

print("=" * 60)
print("Phase 5 — CatBoost (GPU: RTX 4060)")
print("=" * 60)

X_train, y_train, X_val, y_val, features, spw = load_data()

train_pool = Pool(X_train, label=y_train)
val_pool   = Pool(X_val,   label=y_val)

params = {
    "iterations"        : 2000,
    "learning_rate"     : 0.05,
    "depth"             : 6,
    "l2_leaf_reg"       : 3.0,
    "bootstrap_type"    : "Bernoulli",
    "subsample"         : 0.8,
    "scale_pos_weight"  : spw,
    "eval_metric"       : "Logloss",
    "early_stopping_rounds": 50,
    "task_type"         : "GPU",
    "devices"           : "0",
    "random_seed"       : 42,
    "verbose"           : 100,
}

print(f"\nTraining CatBoost | scale_pos_weight={spw:.2f} | GPU=CUDA")
t0 = time.time()

model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s  |  Best iter: {model.best_iteration_}")

# Evaluate
val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("CatBoost", metrics)

# Save
model.save_model(str(MODELS_DIR / "catboost_model.cbm"))
np.save(MODELS_DIR / "catboost_val_preds.npy", val_probs)
metrics["best_iteration"] = int(model.best_iteration_)
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
with open(MODELS_DIR / "catboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODELS_DIR}/catboost_model.cbm")
print(f"  {MODELS_DIR}/catboost_val_preds.npy")
print(f"  {MODELS_DIR}/catboost_metrics.json")
print("\nCatBoost DONE")
