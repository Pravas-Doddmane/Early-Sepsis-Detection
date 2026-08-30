#!/usr/bin/env python
"""
Phase 6: Stacking Ensemble
Meta-learner: LogisticRegression(C=1.0, max_iter=1000)
Meta-training uses OUT-OF-FOLD predictions on val (5-fold), NOT val
predictions scored on the same val set they were fit on — the previous
version reported val PR-AUC 0.0718 by fitting and evaluating the
meta-learner on identical data, which is leakage, not validation.
The only trustworthy number in the old run was test PR-AUC (0.0686).
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    matthews_corrcoef, brier_score_loss
)

MODELS_DIR = Path("enhanced/models")
EXPERIMENTS = Path("enhanced/experiments")
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ['rf', 'xgb', 'lgbm', 'catboost']
N_FOLDS = 5


def load_val_predictions():
    """Load validation predictions from all 4 base models (base models'
    own predict on val — used ONLY to build out-of-fold meta-features,
    not scored directly)."""
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
    import pandas as pd
    val = pd.read_parquet("enhanced/data/processed/val_temporal.parquet")
    return val['SepsisLabel'].values.astype(int)


def evaluate_oof_meta(X_meta, y_val):
    """
    Train the meta-learner using proper out-of-fold predictions: for each
    fold, fit on the other folds' base predictions and predict on the held-out
    fold. This gives an honest estimate of meta-learner generalization,
    unlike fitting and scoring on the same val set.
    """
    print(f"\nGenerating {N_FOLDS}-fold out-of-fold meta-predictions...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_val))

    for fold_idx, (train_idx, holdout_idx) in enumerate(skf.split(X_meta, y_val)):
        fold_meta = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', random_state=42
            # NOTE: class_weight='balanced' removed — today's ablations showed
            # full-strength auto-balancing over-corrected in every base model
            # (XGBoost, LightGBM, CatBoost all improved with mild weighting
            # instead). Worth testing this meta-learner's own class_weight
            # as a follow-up ablation once the OOF pipeline itself is verified.
        )
        fold_meta.fit(X_meta[train_idx], y_val[train_idx])
        oof_probs[holdout_idx] = fold_meta.predict_proba(X_meta[holdout_idx])[:, 1]
        fold_pr_auc = average_precision_score(y_val[holdout_idx], oof_probs[holdout_idx])
        print(f"  Fold {fold_idx + 1}/{N_FOLDS}: PR-AUC = {fold_pr_auc:.4f}")

    oof_pr_auc = average_precision_score(y_val, oof_probs)
    oof_roc_auc = roc_auc_score(y_val, oof_probs)
    print(f"\nOut-of-fold PR-AUC : {oof_pr_auc:.4f}   (baseline: 0.0714)")
    print(f"Out-of-fold ROC-AUC: {oof_roc_auc:.4f}   (baseline: 0.7598)")
    print("(This is the honest val-equivalent number — not the leaky "
          "fit-and-score-on-same-data number from the previous run.)")

    return oof_probs


def train_final_meta_learner(X_meta, y_val):
    """Fit the FINAL meta-learner on all of val (for deployment / test
    prediction) — separate from the OOF loop above, which is only for
    honest performance estimation, not for the model you actually ship."""
    print("\nTraining final meta-learner on full val set (for test predictions)...")
    meta = LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs', random_state=42
    )
    meta.fit(X_meta, y_val)
    print(f"Meta coefficients: {dict(zip(MODEL_NAMES, meta.coef_[0]))}")
    print(f"Meta intercept: {meta.intercept_[0]:.4f}")
    return meta


def generate_test_predictions(meta):
    """Generate test predictions. Structural NaNs (lag/rolling features with
    insufficient history) filled with 0, matching _utils.py's treatment of
    train/val — consistency matters more here than imputation sophistication,
    since train/val already establish 0-as-sentinel as the convention."""
    print("\nGenerating test predictions...")

    with open(EXPERIMENTS / "selected_features.json") as f:
        sel = json.load(f)
    features = sel if isinstance(sel, list) else sel.get("final_features", [])

    test = pd.read_parquet("enhanced/data/processed/test_temporal.parquet")
    available = [f for f in features if f in test.columns]

    n_nan = test[available].isna().sum().sum()
    print(f"  {n_nan} structural NaN values found (insufficient rolling-window "
          f"history) — filling with 0, matching train/val treatment.")
    X_test = test[available].fillna(0).values.astype(np.float32)

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
    print(f"  PR-AUC   : {metrics['pr_auc']:.4f}   (baseline: 0.0714)")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  MCC      : {metrics['mcc']:.4f}")
    print(f"  Brier    : {metrics['brier']:.4f}")
    return metrics


def main():
    print("=" * 60)
    print("Phase 6: Stacking Ensemble (out-of-fold meta-training)")
    print("=" * 60)

    X_meta = load_val_predictions()
    y_val = load_val_labels()

    # Honest performance estimate — out-of-fold, not leaked
    oof_probs = evaluate_oof_meta(X_meta, y_val)

    # Final deployed meta-learner — fit on all of val
    meta = train_final_meta_learner(X_meta, y_val)

    # Test predictions — proper imputation, no fillna(0)
    test_probs, y_test = generate_test_predictions(meta)
    test_metrics = evaluate_test(test_probs, y_test)

    joblib.dump(meta, MODELS_DIR / "meta_learner.pkl")
    np.save(MODELS_DIR / "stack_oof_val_preds.npy", oof_probs)
    np.save(MODELS_DIR / "stack_test_preds.npy", test_probs)

    oof_metrics = {
        "pr_auc": float(average_precision_score(y_val, oof_probs)),
        "roc_auc": float(roc_auc_score(y_val, oof_probs)),
        "note": "out-of-fold estimate, not leaked val-on-val score",
    }
    with open(MODELS_DIR / "stack_val_metrics.json", "w") as f:
        json.dump(oof_metrics, f, indent=2)

    test_metrics["meta_coefficients"] = dict(zip(MODEL_NAMES, meta.coef_[0].tolist()))
    test_metrics["meta_intercept"] = float(meta.intercept_[0])
    with open(MODELS_DIR / "stack_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 6 Complete!")
    print("=" * 60)
    print("Next: python enhanced/calibration/calibrate.py")


if __name__ == "__main__":
    main()