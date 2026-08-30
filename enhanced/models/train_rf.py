"""
Phase 5 — Model 4: RandomForest (CPU, n_jobs=-1)
=================================================
RandomForest does NOT support GPU in sklearn.
Uses all CPU cores (n_jobs=-1) for parallelism.
Class imbalance: explicit mild class_weight (sqrt of full ratio) —
  class_weight='balanced' applies the full imbalance ratio internally,
  same over-correction pattern seen in XGBoost/LightGBM/CatBoost's
  native auto-balancing (all three improved when switched to mild
  weighting instead). RF was also the model with the highest meta-
  learner coefficient (3.31) despite unchanged imbalance handling,
  and its prediction scale (mean=0.30) was ~3x the other models'
  (mean=0.08-0.10) — consistent with over-correction.
"""
import json, time, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from _utils import load_data, evaluate, print_metrics, MODELS_DIR

print("=" * 60)
print("Phase 5 — RandomForest (CPU, all cores)")
print("=" * 60)

X_train, y_train, X_val, y_val, features, spw = load_data()

spw_mild = float(np.sqrt(spw))  # e.g. 54.65 -> ~7.39, matches other 3 models
class_weight_mild = {0: 1.0, 1: spw_mild}

params = {
    "n_estimators": 200,
    "max_depth": 15,
    "max_samples": 0.4,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "class_weight": class_weight_mild,  # was 'balanced' (full ratio)
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 1,
}

print(f"\nTraining RandomForest | class_weight={{0: 1.0, 1: {spw_mild:.2f}}} "
      f"(sqrt of full ratio {spw:.2f}) | NO 'balanced' | "
      f"200 trees, max_depth=15, max_samples=0.4")
t0 = time.time()

model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s")

val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("RandomForest", metrics)

joblib.dump(model, MODELS_DIR / "rf_model.pkl")
np.save(MODELS_DIR / "rf_val_preds.npy", val_probs)
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
metrics["class_weight_used"] = class_weight_mild
with open(MODELS_DIR / "rf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODELS_DIR}/rf_model.pkl")
print(f"  {MODELS_DIR}/rf_val_preds.npy")
print(f"  {MODELS_DIR}/rf_metrics.json")
print("\nRandomForest DONE")