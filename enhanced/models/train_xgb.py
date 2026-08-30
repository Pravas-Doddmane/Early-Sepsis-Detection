"""
Phase 5 — Model 1: XGBoost (GPU)
=================================
Built-in logistic objective (focal loss abandoned — custom gradient/hessian
proved numerically unstable in float32, see conversation history).
Class imbalance: reduced scale_pos_weight (sqrt of full ratio, not full
  ratio) — full ratio (54.65) likely over-corrects given current model
  already over-predicts positive (66-70% recall, 4.7% precision pre-fix).
Early stopping: 50 rounds on val PR-AUC (built-in 'aucpr' metric)
"""
import json, time, joblib
import numpy as np
import xgboost as xgb
from _utils import load_data, evaluate, print_metrics, MODELS_DIR, EXPERIMENTS

print("=" * 60)
print("Phase 5 — XGBoost (GPU: RTX 4060)")
print("=" * 60)

X_train, y_train, X_val, y_val, features, spw = load_data()

spw_mild = float(np.sqrt(spw))  # 54.65 -> ~7.4, much less aggressive

params = {
    "objective"        : "binary:logistic",
    "eval_metric"      : "aucpr",
    "tree_method"      : "hist",
    "device"           : "cuda",
    "n_estimators"     : 2000,
    "learning_rate"    : 0.05,
    "max_depth"        : 6,
    "min_child_weight" : 10,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "gamma"            : 1,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "scale_pos_weight" : spw_mild,
    "random_state"     : 42,
    "verbosity"        : 1,
    "early_stopping_rounds": 50,
}

print(f"\nTraining XGBoost | scale_pos_weight={spw_mild:.2f} (sqrt of full "
      f"ratio {spw:.2f}) | GPU=cuda | early stop on aucpr")
t0 = time.time()

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s  |  Best iter: {model.best_iteration}")

val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("XGBoost", metrics)

joblib.dump(model, MODELS_DIR / "xgb_model.pkl")
np.save(MODELS_DIR / "xgb_val_preds.npy", val_probs)
metrics["best_iteration"] = int(model.best_iteration)
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
metrics["scale_pos_weight_used"] = spw_mild
with open(MODELS_DIR / "xgb_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nXGBoost DONE")