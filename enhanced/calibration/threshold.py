#!/usr/bin/env python
"""
Phase 8: Clinical Threshold Selection
On Val (calibrated):
- Sweep thresholds -> compute Sensitivity, Precision, F1, MCC, Alarm Rate
- Choose threshold balancing clinical sensitivity vs alarm burden
Apply chosen threshold to Test (calibrated) for reporting.
Save: optimal_threshold.json
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix
)

MODELS_DIR = Path("enhanced/models")
EXPERIMENTS = Path("enhanced/experiments")
DATA_DIR = Path("enhanced/data/processed")
EXPERIMENTS.mkdir(parents=True, exist_ok=True)

# Minimum sensitivity (recall) the chosen threshold must achieve.
# Sepsis is a "don't miss it" condition, so we bias toward high recall
# and accept a higher false-alarm rate. Adjust to fit your clinical constraint.
MIN_SENSITIVITY = 0.60


def load_val():
    """Load calibrated validation probabilities and labels."""
    val_probs_raw = np.load(MODELS_DIR / "stack_oof_val_preds.npy")  # was stack_val_preds.npy (stale)
    calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")

    calib_info = json.load(open(MODELS_DIR / "calibration_info.json"))
    calibrator_type = calib_info["type"]

    if calibrator_type == "platt":
        logits = np.log(val_probs_raw / (1 - val_probs_raw + 1e-10) + 1e-10).reshape(-1, 1)
        val_probs = calibrator.predict_proba(logits)[:, 1]
    else:  # isotonic
        val_probs = calibrator.predict(val_probs_raw)

    val = pd.read_parquet(DATA_DIR / "val_temporal.parquet")
    y_val = val["SepsisLabel"].values.astype(int)
    return val_probs, y_val, calibrator_type, calibrator


def load_test():
    """Load pre-computed calibrated test predictions and labels."""
    test_probs = np.load(MODELS_DIR / "calibrated_test_preds.npy")
    test = pd.read_parquet(DATA_DIR / "test_temporal.parquet")
    y_test = test["SepsisLabel"].values.astype(int)
    return test_probs, y_test


def compute_metrics(y_true, probs, threshold):
    """Compute clinical metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    sensitivity = recall_score(y_true, preds, zero_division=0)
    precision = precision_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    mcc = matthews_corrcoef(y_true, preds) if (tp + fp) > 0 and (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    alarm_rate = preds.mean()  # fraction of all hourly records flagged as alarms

    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "mcc": float(mcc),
        "alarm_rate": float(alarm_rate),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def sweep_thresholds(y_true, probs, n_steps=199):
    """Sweep thresholds from ~0 to ~1 and compute metrics at each."""
    thresholds = np.linspace(0.001, 0.999, n_steps)
    results = [compute_metrics(y_true, probs, t) for t in thresholds]
    return pd.DataFrame(results)


def select_threshold(sweep_df, min_sensitivity=MIN_SENSITIVITY):
    """
    Clinical selection rule:
    Among thresholds achieving at least `min_sensitivity` recall,
    pick the one with the lowest alarm_rate (least alert fatigue).
    If none meet the sensitivity floor, fall back to the threshold
    that maximizes F1.
    """
    candidates = sweep_df[sweep_df["sensitivity"] >= min_sensitivity]

    if len(candidates) > 0:
        # Among qualifying thresholds, minimize alarm burden.
        best = candidates.loc[candidates["alarm_rate"].idxmin()]
        rule = f"min_alarm_rate_at_sensitivity>={min_sensitivity}"
    else:
        best = sweep_df.loc[sweep_df["f1"].idxmax()]
        rule = "fallback_max_f1_sensitivity_floor_unreachable"

    return best.to_dict(), rule


def main():
    print("=" * 60)
    print("Phase 8: Clinical Threshold Selection")
    print("=" * 60)

    # --- Validation sweep ---
    val_probs, y_val, calibrator_type, calibrator = load_val()
    print(f"Val (calibrated) probs: mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")
    print(f"Val labels: pos={y_val.sum()}, neg={len(y_val) - y_val.sum()}")
    print(f"Calibrator: {calibrator_type}")

    print("\nSweeping thresholds on validation set...")
    sweep_df = sweep_thresholds(y_val, val_probs)

    print(f"Selecting threshold (min sensitivity target: {MIN_SENSITIVITY:.0%})...")
    best, rule = select_threshold(sweep_df)

    print(f"\nSelected threshold: {best['threshold']:.4f}  (rule: {rule})")
    print("Validation metrics at selected threshold:")
    print(f"  Sensitivity : {best['sensitivity']:.4f}")
    print(f"  Specificity : {best['specificity']:.4f}")
    print(f"  Precision   : {best['precision']:.4f}")
    print(f"  F1          : {best['f1']:.4f}")
    print(f"  MCC         : {best['mcc']:.4f}")
    print(f"  Alarm Rate  : {best['alarm_rate']:.4f}")
    print(f"  Confusion   : TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']}")

    # --- Apply to test ---
    test_probs, y_test = load_test()
    test_metrics = compute_metrics(y_test, test_probs, best["threshold"])

    print("\nTest metrics at selected threshold:")
    print(f"  Sensitivity : {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity : {test_metrics['specificity']:.4f}")
    print(f"  Precision   : {test_metrics['precision']:.4f}")
    print(f"  F1          : {test_metrics['f1']:.4f}")
    print(f"  MCC         : {test_metrics['mcc']:.4f}")
    print(f"  Alarm Rate  : {test_metrics['alarm_rate']:.4f}")

    # --- Save sweep + selected threshold ---
    sweep_df.to_csv(EXPERIMENTS / "threshold_sweep.csv", index=False)

    output = {
        "selection_rule": rule,
        "min_sensitivity_target": MIN_SENSITIVITY,
        "calibrator_type": calibrator_type,
        "optimal_threshold": best["threshold"],
        "val_metrics": best,
        "test_metrics": test_metrics,
    }
    with open(MODELS_DIR / "optimal_threshold.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 8 Complete!")
    print("=" * 60)
    print(f"Saved: {MODELS_DIR}/optimal_threshold.json")
    print(f"Saved: {EXPERIMENTS}/threshold_sweep.csv")
    print("\nNext: python enhanced/xai/explain.py")


if __name__ == "__main__":
    main()