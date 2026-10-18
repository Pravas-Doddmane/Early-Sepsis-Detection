#!/usr/bin/env python
"""Verify saved model, prediction, feature-selection, and SHAP artifacts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from catboost import CatBoostClassifier
from scipy.stats import binom


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "enhanced" / "models"
DATA_DIR = ROOT / "enhanced" / "data" / "processed"
EXPERIMENTS_DIR = ROOT / "enhanced" / "experiments"
XAI_DIR = EXPERIMENTS_DIR / "xai"
MODEL_NAMES = ("catboost", "xgb", "lgbm", "rf")


def load_model(name: str):
    if name == "catboost":
        model = CatBoostClassifier()
        model.load_model(str(MODELS_DIR / "catboost_model.cbm"))
        return model
    return joblib.load(MODELS_DIR / f"{name}_model.pkl")


def model_imbalance_settings() -> dict[str, str]:
    settings = {}
    for name in MODEL_NAMES:
        model = load_model(name)
        params = model.get_params()
        if name == "catboost":
            settings[name] = f"class_weights={params.get('class_weights')}"
        elif name == "rf":
            settings[name] = f"class_weight={params.get('class_weight')}"
        else:
            settings[name] = (
                f"scale_pos_weight={params.get('scale_pos_weight')}"
            )
    return settings


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if len(labels) != len(scores):
        raise ValueError(f"Prediction/label length mismatch: {len(scores)} != {len(labels)}")
    predicted = (scores >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
    }


def main() -> None:
    val = pd.read_parquet(DATA_DIR / "val_temporal.parquet", columns=["SepsisLabel"])
    labels = val["SepsisLabel"].to_numpy(dtype=int)
    print(f"VALIDATION: rows={len(labels)}, positive labels={int(labels.sum())}")
    print("\nBASE LEARNERS: metrics recomputed from saved validation predictions")
    print("model      ROC-AUC      PR-AUC     recall@0.5  precision@0.5  rows")
    for name in MODEL_NAMES:
        scores = np.asarray(
            np.load(MODELS_DIR / f"{name}_val_preds.npy"), dtype=float
        ).reshape(-1)
        metrics = binary_metrics(labels, scores)
        print(
            f"{name:<10} {metrics['roc_auc']:.9f}  {metrics['pr_auc']:.9f}  "
            f"{metrics['recall']:.9f}  {metrics['precision']:.9f}  {len(scores)}"
        )

    print("\nFINAL META-LEARNER COEFFICIENTS (loaded from meta_learner.pkl)")
    meta = joblib.load(MODELS_DIR / "meta_learner.pkl")
    for name, coefficient in zip(("rf", "xgb", "lgbm", "catboost"), meta.coef_[0]):
        print(f"  {name}: {coefficient:+.12f}")
    print(f"  intercept: {float(meta.intercept_[0]):+.12f}")

    stack_rows = []
    for split, scores_file in (
        ("validation OOF", "stack_oof_val_preds.npy"),
        ("test", "stack_test_preds.npy"),
    ):
        split_labels = (
            labels
            if split == "validation OOF"
            else pd.read_parquet(
                DATA_DIR / "test_temporal.parquet", columns=["SepsisLabel"]
            )["SepsisLabel"].to_numpy(dtype=int)
        )
        scores = np.asarray(np.load(MODELS_DIR / scores_file), dtype=float).reshape(-1)
        metrics = binary_metrics(split_labels, scores)
        stack_rows.append((split, metrics))
    print("\nSTACK PERFORMANCE (saved stacked score arrays)")
    print("split             ROC-AUC        PR-AUC       recall@0.5   precision@0.5")
    for split, metrics in stack_rows:
        print(
            f"{split:<17} {metrics['roc_auc']:.9f}  {metrics['pr_auc']:.9f}  "
            f"{metrics['recall']:.9f}  {metrics['precision']:.9f}"
        )

    print("\nMODEL IMBALANCE PARAMETERS (loaded fitted models)")
    for name, setting in model_imbalance_settings().items():
        print(f"  {name}: {setting}")

    mi = pd.read_csv(EXPERIMENTS_DIR / "mi_scores.csv")
    boruta = pd.read_csv(EXPERIMENTS_DIR / "boruta_hits.csv")
    comparison = pd.read_csv(EXPERIMENTS_DIR / "feature_selection_comparison.csv")
    selected = json.loads((EXPERIMENTS_DIR / "selected_features.json").read_text())
    p_threshold = 0.01 / len(boruta)
    confirmed_hits = [
        hits
        for hits in range(51)
        if 1 - binom.cdf(hits - 1, 50, 0.5) < p_threshold
    ]
    min_confirming_hits = min(confirmed_hits)
    confirmed_count = int((boruta["hits"] >= min_confirming_hits).sum())
    mi_count = int(comparison["in_mi"].sum())
    final_count = len(selected)
    print("\nFEATURE SELECTION COUNTS")
    print(f"  Candidate features (mi_scores.csv rows): {len(mi)}")
    print(f"  MI-selected features (comparison): {mi_count}")
    print(
        f"  Boruta-confirmed features: {confirmed_count} "
        f"(50 iterations, Bonferroni p<{p_threshold:.8g}, minimum hits={min_confirming_hits})"
    )
    print(f"  Final features (selected_features.json): {final_count}")
    print(
        "  Note: selection.py's module docstring says 309 candidates and its call "
        "uses top_k=150; saved MI artifact has 347 rows."
    )

    shap = pd.read_csv(XAI_DIR / "global_shap_importance.csv")
    shap = shap.sort_values("mean_abs_shap", ascending=False).head(10)
    print("\nTOP 10 MEAN ABSOLUTE SHAP FEATURES")
    print("  Source: enhanced/experiments/xai/global_shap_importance.csv")
    print("  Model: CatBoost, per enhanced/xai/explain.py global_shap()")
    print("  Sample: 500 patients selected from test set (seed 42)")
    for row in shap.itertuples(index=False):
        print(f"  {row.feature:<24} {row.mean_abs_shap:.12f}")


if __name__ == "__main__":
    main()
