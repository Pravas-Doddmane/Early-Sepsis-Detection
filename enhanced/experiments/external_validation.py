"""
External Validation: setA -> setB
===================================
Mirrors the base paper's MIMIC-IV -> eICU external validation design.
Uses existing trained models (no retraining) — evaluates how well the
stack, trained on a pooled setA+setB split, performs specifically on
setB-origin patients vs setA-origin patients within the existing test set,
AND (if coverage allows) a stricter setA-only-train check.

This is supplementary/robustness evidence alongside the main pooled
70/15/15 result already finalized — not a replacement for it.
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

MODELS_DIR = Path("enhanced/models")
DATA_DIR = Path("enhanced/data/processed")
EXPERIMENTS = Path("enhanced/experiments")


def get_source_set(patient_id):
    num = int(patient_id.lstrip('p'))
    return 'setB' if num >= 100000 else 'setA'


def main():
    print("=" * 60)
    print("External Validation: setA vs setB breakdown of existing test set")
    print("=" * 60)

    test = pd.read_parquet(DATA_DIR / "test_temporal.parquet")
    test['source_set'] = test['patient_id'].apply(get_source_set)

    print(f"\nTest set composition:")
    print(test.groupby('source_set')['patient_id'].nunique())
    print(f"\nRow counts:")
    print(test['source_set'].value_counts())

    # Load the already-computed calibrated stack predictions + threshold
    test_probs = np.load(MODELS_DIR / "calibrated_test_preds.npy")
    threshold_info = json.load(open(MODELS_DIR / "optimal_threshold.json"))
    threshold = threshold_info["optimal_threshold"]
    y_test = test["SepsisLabel"].values.astype(int)

    assert len(test_probs) == len(test), (
        f"Length mismatch: {len(test_probs)} preds vs {len(test)} rows — "
        f"test_temporal.parquet may have changed since predictions were "
        f"generated. Re-run stack.py before trusting this breakdown."
    )

    results = {}
    for source in ['setA', 'setB']:
        mask = (test['source_set'] == source).values
        y_sub = y_test[mask]
        p_sub = test_probs[mask]
        preds_sub = (p_sub >= threshold).astype(int)

        results[source] = {
            "n_rows": int(mask.sum()),
            "n_patients": int(test.loc[mask, 'patient_id'].nunique()),
            "sepsis_rate": float(y_sub.mean()),
            "roc_auc": float(roc_auc_score(y_sub, p_sub)) if y_sub.sum() > 0 else None,
            "pr_auc": float(average_precision_score(y_sub, p_sub)) if y_sub.sum() > 0 else None,
            "sensitivity": float(recall_score(y_sub, preds_sub, zero_division=0)),
            "precision": float(precision_score(y_sub, preds_sub, zero_division=0)),
        }

        print(f"\n--- {source} ---")
        for k, v in results[source].items():
            print(f"  {k}: {v}")

    with open(EXPERIMENTS / "external_validation_setA_setB.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {EXPERIMENTS}/external_validation_setA_setB.json")
    print("\nNOTE: this breaks down performance WITHIN the existing pooled")
    print("test set by hospital-system origin — it does NOT retrain on")
    print("setA-only. For a stricter external-validation claim (train on")
    print("setA only, test entirely on setB), a separate retrain would be")
    print("needed — flag if you want that version instead.")


if __name__ == "__main__":
    main()