"""
Phase 5 — Model 3: CatBoost (GPU) with Mild Class Weights
===============================================================
GPU: task_type='GPU', devices='0'  (RTX 4060)
Class imbalance: explicit mild class_weights (sqrt of full ratio) —
  auto_class_weights='Balanced' applies close to the full imbalance
  ratio internally and over-corrected (PR-AUC 0.0677-0.0682, barely
  moved from original 0.0676). Matches the fix that worked for
  XGBoost (0.0672->0.0698) and LightGBM (0.0556->0.0669).
Early stopping: 50 rounds on val PR-AUC (was Logloss originally —
  wrong metric to early-stop on at this imbalance level)
"""
import json, time, joblib
import numpy as np
from catboost import CatBoostClassifier, Pool, __version__ as cb_version
from _utils import load_data, evaluate, print_metrics, MODELS_DIR

print("=" * 60)
print("Phase 5 — CatBoost (GPU: RTX 4060) + Mild Class Weights")
print("=" * 60)
print(f"CatBoost version: {cb_version}")

X_train, y_train, X_val, y_val, features, spw = load_data()

spw_mild = float(np.sqrt(spw))  # e.g. 54.65 -> ~7.39, matches XGB/LGBM fix

train_pool = Pool(X_train, label=y_train)
val_pool   = Pool(X_val,   label=y_val)

# PRAUC eval_metric requires a reasonably recent CatBoost version.
EVAL_METRIC = "PRAUC"
try:
    _test = CatBoostClassifier(iterations=1, eval_metric=EVAL_METRIC, verbose=False)
except Exception as e:
    print(f"[WARNING] '{EVAL_METRIC}' not supported in this CatBoost "
          f"version ({e}); falling back to Logloss.")
    EVAL_METRIC = "Logloss"

params = {
    "iterations"           : 2000,
    "learning_rate"        : 0.05,
    "depth"                : 6,
    "l2_leaf_reg"          : 3.0,
    "bootstrap_type"       : "Bernoulli",
    "subsample"            : 0.8,
    "class_weights"        : [1.0, spw_mild],  # [negative, positive] — NOT auto_class_weights
    "sampling_frequency"   : "PerTree",
    "sampling_unit"        : "Object",
    "eval_metric"          : EVAL_METRIC,
    "early_stopping_rounds": 50,
    "task_type"            : "GPU",
    "devices"              : "0",
    "random_seed"          : 42,
    "verbose"              : 100,
}

print(f"\nTraining CatBoost | class_weights=[1.0, {spw_mild:.2f}] "
      f"(sqrt of full ratio {spw:.2f}) | NO auto_class_weights | GPU=CUDA | "
      f"Bernoulli sampling | early stop on {EVAL_METRIC}")
t0 = time.time()

model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s  |  Best iter: {model.best_iteration_}")

val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("CatBoost", metrics)

model.save_model(str(MODELS_DIR / "catboost_model.cbm"))
np.save(MODELS_DIR / "catboost_val_preds.npy", val_probs)
metrics["best_iteration"] = int(model.best_iteration_)
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
metrics["eval_metric_used"] = EVAL_METRIC
metrics["class_weights_used"] = [1.0, spw_mild]
with open(MODELS_DIR / "catboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODELS_DIR}/catboost_model.cbm")
print(f"  {MODELS_DIR}/catboost_val_preds.npy")
print(f"  {MODELS_DIR}/catboost_metrics.json")
print("\nCatBoost DONE")