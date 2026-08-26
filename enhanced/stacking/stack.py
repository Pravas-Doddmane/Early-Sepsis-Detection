#!/usr/bin/env python
"""
Phase 6: Stacking Ensemble
Meta-learner: LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
Input: 4× val predictions (shape: n_samples × 4)
Train meta on Val → Predict on Test
"""
import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    matthews_corrcoef, brier_score_loss
)

MODELS_DIR = Path("enhanced/models")
EXPERIMENTS = Path("enhanced/experiments")
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ['rf', 'xgb', 'lgbm', 'catboost']


def load_val_predictions():
    """Load validation predictions from all 4 base models."""
    val_preds = []
    for name in MODEL_NAMES:
        pred_path = MODELS_DIR / f"{name}_val_preds.npy"
        preds = np.load(pred_path)
        val_preds.append(preds)
        print(f"Loaded {name}_val_preds.npy: {preds.shape}")

    X_meta = np.column_stack(val_preds)
    print(f"Meta-features shape: {X_meta.shape}")
    return X_meta


def load_val_labels():
    """Load validation labels from processed data."""
    import pandas as pd
    val = pd.read_parquet("enhanced/data/processed/val_temporal.parquet")
    return val['SepsisLabel'].values.astype(int)


def load_test_predictions():
    """Load test predictions from all 4 base models (for final prediction)."""
    test_preds = []
    for name in MODEL_NAMES:
        # We need to generate test predictions from saved models
        pass  # Will generate below
    return test_preds


def train_meta_learner(X_meta, y_val):
    """Train logistic regression meta-learner."""
    print("\nTraining meta-learner (LogisticRegression)...")

    meta = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    meta.fit(X_meta, y_val)

    # Meta-learner coefficients
    print(f"Meta coefficients: {dict(zip(MODEL_NAMES, meta.coef_[0]))}")
    print(f"Meta intercept: {meta.intercept_[0]:.4f}")

    return meta


def evaluate_meta(meta, X_meta, y_val):
    """Evaluate meta-learner on validation set."""
    val_probs = meta.predict_proba(X_meta)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, val_probs)),
        "pr_auc": float(average_precision_score(y_val, val_probs)),
        "recall": float(recall_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)),
        "precision": float(precision_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)),
        "f1": float(f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)),
        "mcc": float(matthews_corrcoef(y_val, (val_probs >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y_val, val_probs)),
        "threshold": 0.5
    }

    print(f"\n[Stacking Ensemble] Validation Metrics:")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}   (baseline: 0.7598)")
    print(f"  PR-AUC   : {metrics['pr_auc']:.4f}   (baseline: 0.0714)")
    print(f"  Recall   : {metrics['recall']:.4f}   (baseline: 0.5525)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  MCC      : {metrics['mcc']:.4f}")
    print(f"  Brier    : {metrics['brier']:.4f}")

    return val_probs, metrics


def generate_test_predictions(meta):
    """Generate test predictions by loading models and predicting on test_temporal.parquet."""
    import pandas as pd

    print("\nGenerating test predictions...")

    # Load selected features
    with open(EXPERIMENTS / "selected_features.json") as f:
        sel = json.load(f)
    features = sel if isinstance(sel, list) else sel.get("final_features", [])
    target = "SepsisLabel"

    # Load test data
    test = pd.read_parquet("enhanced/data/processed/test_temporal.parquet")
    available = [f for f in features if f in test.columns]
    X_test = test[available].fillna(0).values.astype(np.float32)

    # Get predictions from each base model
    test_preds = []
    for name in MODEL_NAMES:
        if name == 'catboost':
            from catboost import CatBoostClassifier, Pool
            model = CatBoostClassifier()
            model.load_model(str(MODELS_DIR / f"{name}_model.cbm"))
            preds = model.predict_proba(Pool(X_test))[:, 1]
        else:
            model = joblib.load(MODELS_DIR / f"{name}_model.pkl")
            preds = model.predict_proba(X_test)[:, 1]
        test_preds.append(preds)
        print(f"  {name}: mean={preds.mean():.4f}, std={preds.std():.4f}")

    X_meta_test = np.column_stack(test_preds)
    test_probs = meta.predict_proba(X_meta_test)[:, 1]

    return test_probs, test['SepsisLabel'].values.astype(int)


def evaluate_test(test_probs, y_test):
    """Evaluate on test set."""
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "pr_auc": float(average_precision_score(y_test, test_probs)),
        "recall": float(recall_score(y_test, (test_probs >= 0.5).astype(int), zero_division=0)),
        "precision": float(precision_score(y_test, (test_probs >= 0.5).astype(int), zero_division=0)),
        "f1": float(f1_score(y_test, (test_probs >= 0.5).astype(int), zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, (test_probs >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y_test, test_probs)),
        "threshold": 0.5
    }

    print(f"\n[Stacking Ensemble] TEST Metrics:")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC   : {metrics['pr_auc']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  MCC      : {metrics['mcc']:.4f}")
    print(f"  Brier    : {metrics['brier']:.4f}")

    return metrics


def main():
    print("=" * 60)
    print("Phase 6: Stacking Ensemble")
    print("=" * 60)

    # 1. Load validation predictions
    X_meta = load_val_predictions()
    y_val = load_val_labels()

    # 2. Train meta-learner
    meta = train_meta_learner(X_meta, y_val)

    # 3. Evaluate on validation
    val_probs, val_metrics = evaluate_meta(meta, X_meta, y_val)

    # 4. Generate test predictions
    test_probs, y_test = generate_test_predictions(meta)

    # 5. Evaluate on test
    test_metrics = evaluate_test(test_probs, y_test)

    # 6. Save meta-learner and predictions
    joblib.dump(meta, MODELS_DIR / "meta_learner.pkl")
    np.save(MODELS_DIR / "stack_val_preds.npy", val_probs)
    np.save(MODELS_DIR / "stack_test_preds.npy", test_probs)

    # Save metrics
    val_metrics["meta_coefficients"] = dict(zip(MODEL_NAMES, meta.coef_[0].tolist()))
    val_metrics["meta_intercept"] = float(meta.intercept_[0])
    with open(MODELS_DIR / "stack_val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    test_metrics["meta_coefficients"] = dict(zip(MODEL_NAMES, meta.coef_[0].tolist()))
    test_metrics["meta_intercept"] = float(meta.intercept_[0])
    with open(MODELS_DIR / "stack_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 6 Complete!")
    print("=" * 60)
    print(f"Saved: {MODELS_DIR}/meta_learner.pkl")
    print(f"Saved: {MODELS_DIR}/stack_val_preds.npy")
    print(f"Saved: {MODELS_DIR}/stack_test_preds.npy")
    print(f"Saved: {MODELS_DIR}/stack_val_metrics.json")
    print(f"Saved: {MODELS_DIR}/stack_test_metrics.json")
    print("\nNext: python enhanced/calibration/calibrate.py")


if __name__ == "__main__":
    main()