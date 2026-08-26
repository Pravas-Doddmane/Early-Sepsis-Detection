"""
Phase 5 — Model 4: RandomForest (CPU, n_jobs=-1)
=================================================
RandomForest does NOT support GPU in sklearn.
Uses all CPU cores (n_jobs=-1) for parallelism.
Class imbalance: class_weight='balanced'
"""
import json, time, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from _utils import load_data, evaluate, print_metrics, MODELS_DIR

print("=" * 60)
print("Phase 5 — RandomForest (CPU, all cores)")
print("=" * 60)

X_train, y_train, X_val, y_val, features, spw = load_data()

params = {
    "n_estimators": 200,
    "max_depth": 15,
    "max_samples": 0.4,       # Subsample 40% per tree for massive speedup + diversity
    "min_samples_leaf": 20,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 1,
}

print(f"\nTraining RandomForest (Optimized for speed: 200 trees, max_depth=15, max_samples=0.4)")
t0 = time.time()

model = RandomForestClassifier(**params)
model.fit(X_train, y_train)

elapsed = time.time() - t0
print(f"\nTraining done in {elapsed:.1f}s")

# Evaluate
val_probs = model.predict_proba(X_val)[:, 1]
metrics   = evaluate(y_val, val_probs)
print_metrics("RandomForest", metrics)

# Save
joblib.dump(model, MODELS_DIR / "rf_model.pkl")
np.save(MODELS_DIR / "rf_val_preds.npy", val_probs)
metrics["train_time_sec"] = round(elapsed, 1)
metrics["n_features"] = len(features)
with open(MODELS_DIR / "rf_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved:")
print(f"  {MODELS_DIR}/rf_model.pkl")
print(f"  {MODELS_DIR}/rf_val_preds.npy")
print(f"  {MODELS_DIR}/rf_metrics.json")
print("\nRandomForest DONE")
