#!/usr/bin/env python
"""
Phase 7: Probability Calibration
Compare on Val only:
- Platt (sigmoid via logistic regression on logits)
- Isotonic
Select by Brier Score + calibration curve visual
Apply chosen calibrator to Test predictions
"""
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

MODELS_DIR = Path("enhanced/models")
EXPERIMENTS = Path("enhanced/experiments")
EXPERIMENTS.mkdir(parents=True, exist_ok=True)


def load_val_data():
    """Load validation probabilities and labels."""
    val_probs = np.load(MODELS_DIR / "stack_val_preds.npy")
    import pandas as pd
    val = pd.read_parquet("enhanced/data/processed/val_temporal.parquet")
    y_val = val['SepsisLabel'].values.astype(int)
    return val_probs, y_val


def load_test_data():
    """Load test probabilities and labels."""
    test_probs = np.load(MODELS_DIR / "stack_test_preds.npy")
    import pandas as pd
    test = pd.read_parquet("enhanced/data/processed/test_temporal.parquet")
    y_test = test['SepsisLabel'].values.astype(int)
    return test_probs, y_test


def fit_calibrators(val_probs, y_val):
    """Fit Platt and Isotonic calibrators on validation set."""
    print("Fitting calibrators on validation set...")

    # Platt: fit logistic regression on logits
    print("  Platt (sigmoid)...")
    logits = np.log(val_probs / (1 - val_probs + 1e-10) + 1e-10)
    logits = logits.reshape(-1, 1)

    platt_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs')
    platt_lr.fit(logits, y_val)

    # Isotonic
    print("  Isotonic...")
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    iso.fit(val_probs, y_val)

    return platt_lr, iso, logits


def apply_calibrator(calibrator, probs, calibrator_type):
    """Apply calibrator to probabilities."""
    if calibrator_type == 'platt':
        logits = np.log(probs / (1 - probs + 1e-10) + 1e-10).reshape(-1, 1)
        return calibrator.predict_proba(logits)[:, 1]
    else:  # isotonic
        return calibrator.predict(probs)


def evaluate_calibration(y_true, probs, name):
    """Evaluate calibration quality."""
    brier = brier_score_loss(y_true, probs)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy='quantile')

        # ECE (Expected Calibration Error)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)   # keep prob==1.0 in last bin instead of dropping it
    ece = 0
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_conf = probs[mask].mean()
            bin_acc = y_true[mask].mean()
            ece += mask.sum() * abs(bin_conf - bin_acc)
    ece /= len(probs)

    return { 
        'name': name,
        'brier': float(brier),
        'ece': float(ece),
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist()
    }


def plot_calibration_curves(results, val_probs, y_val, platt_lr, iso):
    """Plot calibration curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax = axes[0]
    # Uncalibrated
    prob_true, prob_pred = calibration_curve(y_val, val_probs, n_bins=10, strategy='quantile')
    ax.plot(prob_pred, prob_true, 'o-', label='Uncalibrated', alpha=0.7)

    for res in results:
        if res['name'] == 'Uncalibrated':
            continue
        ax.plot(res['prob_pred'], res['prob_true'], 'o-', label=f"{res['name']} (Brier={res['brier']:.4f}, ECE={res['ece']:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves (Reliability Diagram)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram of probabilities
    ax = axes[1]
    ax.hist(val_probs, bins=50, alpha=0.5, label='Uncalibrated', density=True)
    for res in results:
        if res['name'] == 'Platt':
            cal_probs = apply_calibrator(platt_lr, val_probs, 'platt')
            ax.hist(cal_probs, bins=50, alpha=0.5, label='Platt', density=True)
        elif res['name'] == 'Isotonic':
            cal_probs = apply_calibrator(iso, val_probs, 'isotonic')
            ax.hist(cal_probs, bins=50, alpha=0.5, label='Isotonic', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Probability Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(EXPERIMENTS / "calibration_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration curves: {EXPERIMENTS}/calibration_curves.png")


def main():
    print("=" * 60)
    print("Phase 7: Probability Calibration")
    print("=" * 60)

    # Load validation data
    val_probs, y_val = load_val_data()
    print(f"Val probs: mean={val_probs.mean():.4f}, std={val_probs.std():.4f}")
    print(f"Val labels: pos={y_val.sum()}, neg={len(y_val)-y_val.sum()}")

    # Fit calibrators
    platt_lr, iso, logits = fit_calibrators(val_probs, y_val)

    # Apply to validation
    val_platt = apply_calibrator(platt_lr, val_probs, 'platt')
    val_iso = apply_calibrator(iso, val_probs, 'isotonic')

    # Evaluate on validation
    print("\nValidation Calibration Metrics:")
    results = []
    uncal = evaluate_calibration(y_val, val_probs, 'Uncalibrated')
    print(f"  Uncalibrated: Brier={uncal['brier']:.4f}, ECE={uncal['ece']:.4f}")
    results.append(uncal)

    platt_res = evaluate_calibration(y_val, val_platt, 'Platt')
    print(f"  Platt:        Brier={platt_res['brier']:.4f}, ECE={platt_res['ece']:.4f}")
    results.append(platt_res)

    iso_res = evaluate_calibration(y_val, val_iso, 'Isotonic')
    print(f"  Isotonic:     Brier={iso_res['brier']:.4f}, ECE={iso_res['ece']:.4f}")
    results.append(iso_res)

    # Select best by Brier score (only among actual calibrators)
    candidates = [r for r in results if r['name'] != 'Uncalibrated']
    best = min(candidates, key=lambda x: x['brier'])
    print(f"\nSelected: {best['name']} (Brier={best['brier']:.4f})")

    # Plot
    plot_calibration_curves(results, val_probs, y_val, platt_lr, iso)

    # Apply best to test set
    test_probs, y_test = load_test_data()
    if best['name'] == 'Platt':
        test_cal = apply_calibrator(platt_lr, test_probs, 'platt')
    elif best['name'] == 'Isotonic':
        test_cal = apply_calibrator(iso, test_probs, 'isotonic')
    else:
        test_cal = test_probs

    # Evaluate on test
    test_uncal = evaluate_calibration(y_test, test_probs, 'Uncalibrated')
    test_cal_res = evaluate_calibration(y_test, test_cal, best['name'])

    print(f"\nTest Calibration Metrics:")
    print(f"  Uncalibrated: Brier={test_uncal['brier']:.4f}, ECE={test_uncal['ece']:.4f}")
    print(f"  {best['name']}: Brier={test_cal_res['brier']:.4f}, ECE={test_cal_res['ece']:.4f}")

    # Save calibrator
    if best['name'] == 'Platt':
        joblib.dump(platt_lr, MODELS_DIR / "calibrator.pkl")
        calibrator_type = 'platt'
    else:
        joblib.dump(iso, MODELS_DIR / "calibrator.pkl")
        calibrator_type = 'isotonic'

    # Save calibrated test predictions
    np.save(MODELS_DIR / "calibrated_test_preds.npy", test_cal)

    # Save calibration info
    calib_info = {
        'type': calibrator_type,
        'val_brier_uncalibrated': uncal['brier'],
        'val_brier_calibrated': best['brier'],
        'val_ece_uncalibrated': uncal['ece'],
        'val_ece_calibrated': best['ece'],
        'test_brier_uncalibrated': test_uncal['brier'],
        'test_brier_calibrated': test_cal_res['brier'],
        'test_ece_uncalibrated': test_uncal['ece'],
        'test_ece_calibrated': test_cal_res['ece']
    }
    with open(MODELS_DIR / "calibration_info.json", 'w') as f:
        json.dump(calib_info, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 7 Complete!")
    print("=" * 60)
    print(f"Best calibrator: {best['name']}")
    print(f"Saved: {MODELS_DIR}/calibrator.pkl ({calibrator_type})")
    print(f"Saved: {MODELS_DIR}/calibrated_test_preds.npy")
    print(f"Saved: {MODELS_DIR}/calibration_info.json")
    print(f"Saved: {EXPERIMENTS}/calibration_curves.png")
    print("\nNext: python enhanced/calibration/threshold.py")


if __name__ == "__main__":
    main()