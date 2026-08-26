"""
Shared utilities for Phase 5 model training scripts.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    matthews_corrcoef, brier_score_loss,
)

ROOT          = Path(__file__).resolve().parent.parent.parent
PROCESSED     = ROOT / "enhanced" / "data" / "processed"
EXPERIMENTS   = ROOT / "enhanced" / "experiments"
MODELS_DIR    = ROOT / "enhanced" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load train/val temporal parquets + selected features."""
    with open(EXPERIMENTS / "selected_features.json") as f:
        sel = json.load(f)

    # Handle both list and dict formats
    if isinstance(sel, list):
        features = sel
    else:
        features = sel.get("final_features", sel.get("selected_features", []))

    target = "SepsisLabel"  # Hardcoded, matches our data

    print("Loading train_temporal.parquet...")
    train = pd.read_parquet(PROCESSED / "train_temporal.parquet")
    print("Loading val_temporal.parquet...")
    val   = pd.read_parquet(PROCESSED / "val_temporal.parquet")

    # Keep only selected features (fill NaN with 0 -- models handle NaN differently)
    available = [f for f in features if f in train.columns]
    print(f"  Selected features : {len(features)}")
    print(f"  Available in data : {len(available)}")

    X_train = train[available].fillna(0).values.astype(np.float32)
    y_train = train[target].values.astype(int)
    X_val   = val[available].fillna(0).values.astype(np.float32)
    y_val   = val[target].values.astype(int)

    n_pos  = y_train.sum()
    n_neg  = len(y_train) - n_pos
    spw    = n_neg / n_pos   # scale_pos_weight for class imbalance

    print(f"\n  Train : {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"  Val   : {X_val.shape[0]:,} rows x {X_val.shape[1]} features")
    print(f"  Sepsis rate (train): {n_pos/len(y_train)*100:.2f}%  | scale_pos_weight = {spw:.2f}")

    return X_train, y_train, X_val, y_val, available, spw


def evaluate(y_true, y_prob, threshold=0.5):
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc"   : float(roc_auc_score(y_true, y_prob)),
        "pr_auc"    : float(average_precision_score(y_true, y_prob)),
        "recall"    : float(recall_score(y_true, y_pred, zero_division=0)),
        "precision" : float(precision_score(y_true, y_pred, zero_division=0)),
        "f1"        : float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc"       : float(matthews_corrcoef(y_true, y_pred)),
        "brier"     : float(brier_score_loss(y_true, y_prob)),
        "threshold" : threshold,
    }


def print_metrics(name, metrics):
    print(f"\n  [{name}] Validation Metrics:")
    print(f"    ROC-AUC  : {metrics['roc_auc']:.4f}   (baseline: 0.7598)")
    print(f"    PR-AUC   : {metrics['pr_auc']:.4f}   (baseline: 0.0714)")
    print(f"    Recall   : {metrics['recall']:.4f}   (baseline: 0.5525)")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    F1       : {metrics['f1']:.4f}")
    print(f"    MCC      : {metrics['mcc']:.4f}")
    print(f"    Brier    : {metrics['brier']:.4f}")
