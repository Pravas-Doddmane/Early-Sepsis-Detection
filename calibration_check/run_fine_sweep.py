#!/usr/bin/env python
"""Fine threshold sweep using saved stacked predictions and calibrators."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "enhanced" / "models"
DATA_DIR = ROOT / "enhanced" / "data" / "processed"
CHECK_DIR = ROOT / "calibration_check"
SEED = 42
TARGET_RECALL = 0.6939
OLD_THRESHOLD = 0.026202020202020202


def get_data(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_path = (
        MODELS_DIR / "stack_oof_val_preds.npy"
        if split == "val"
        else MODELS_DIR / "stack_test_preds.npy"
    )
    parquet_path = DATA_DIR / f"{split}_temporal.parquet"
    scores = np.asarray(np.load(score_path), dtype=float).reshape(-1)
    frame = pd.read_parquet(parquet_path, columns=["patient_id", "SepsisLabel"])
    labels = frame["SepsisLabel"].to_numpy(dtype=int)
    patient_ids = frame["patient_id"].to_numpy()
    if len(scores) != len(labels):
        raise ValueError(f"{split} score/label row mismatch")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{split} scores must be finite probabilities in [0, 1]")
    return scores, labels, patient_ids


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "alarm_rate": float(predictions.mean()),
    }


def closest_target_metrics(
    labels: np.ndarray, scores: np.ndarray, target_recall: float
) -> dict:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    cumulative_true_positives = np.cumsum(sorted_labels)[group_ends]
    predicted_positives = group_ends + 1
    total_positives = int(labels.sum())
    sensitivities = cumulative_true_positives / total_positives
    alarm_rates = predicted_positives / len(labels)
    precisions = cumulative_true_positives / predicted_positives
    f1_values = (
        2 * cumulative_true_positives / (total_positives + predicted_positives)
    )
    candidate_indices = np.lexsort(
        (-sorted_scores[group_ends], alarm_rates, np.abs(sensitivities - target_recall))
    )
    index = int(candidate_indices[0])
    return {
        "threshold": float(sorted_scores[group_ends[index]]),
        "sensitivity": float(sensitivities[index]),
        "precision": float(precisions[index]),
        "f1": float(f1_values[index]),
        "alarm_rate": float(alarm_rates[index]),
    }


def print_metrics_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(
        pd.DataFrame(rows)[
            ["method", "threshold", "sensitivity", "precision", "f1", "alarm_rate"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


def main() -> None:
    raw_val, y_val, patient_ids = get_data("val")
    raw_test, y_test, _ = get_data("test")

    val_metrics_path = MODELS_DIR / "stack_val_metrics.json"
    test_metrics_path = MODELS_DIR / "stack_test_metrics.json"
    saved_val_metrics = json.loads(val_metrics_path.read_text(encoding="utf-8"))
    saved_test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))

    val_recall_at_half = metrics(y_val, raw_val, 0.5)["sensitivity"]
    test_recall_at_half = metrics(y_test, raw_test, 0.5)["sensitivity"]
    print("STACKING METRIC PROVENANCE (current saved artifacts)")
    print(f"  Saved stack_test_metrics.json recall: {saved_test_metrics.get('recall')}")
    print(f"  Recomputed test recall, stack_test_preds >= 0.5: {test_recall_at_half:.9f}")
    print(f"  Saved stack_val_metrics.json keys: {sorted(saved_val_metrics)}")
    print(f"  Recomputed OOF val recall, stack_oof_val_preds >= 0.5: {val_recall_at_half:.9f}")
    print("  stack.py uses probabilities >= 0.5 for test metrics; it does not call predict().")
    print("  Both OOF and final LogisticRegression are instantiated without class_weight.")

    unique_patient_ids = pd.unique(patient_ids)
    patient_a, patient_b = train_test_split(
        unique_patient_ids, test_size=0.5, random_state=SEED, shuffle=True
    )
    mask_a = np.isin(patient_ids, patient_a)
    mask_b = np.isin(patient_ids, patient_b)
    if np.any(mask_a & mask_b) or not np.all(mask_a | mask_b):
        raise RuntimeError("Validation patient split is not a complete disjoint partition")

    isotonic_a = IsotonicRegression(out_of_bounds="clip")
    isotonic_a.fit(raw_val[mask_a], y_val[mask_a])
    calibrated_b = isotonic_a.predict(raw_val[mask_b])
    calibrated_test_a = isotonic_a.predict(raw_test)

    # Include both endpoints; remove the repeated 0.20 once the two grids join.
    low_grid = np.arange(0.001, 0.2000001, 0.0005)
    high_grid = np.arange(0.20, 0.9900001, 0.01)
    thresholds = np.unique(np.round(np.concatenate((low_grid, high_grid)), 12))
    if thresholds[0] != 0.001 or thresholds[-1] != 0.99:
        raise RuntimeError("Threshold grid does not have the requested endpoints")
    qualifying = []
    for threshold in thresholds:
        result = metrics(y_val[mask_b], calibrated_b, float(threshold))
        if result["sensitivity"] >= 0.60:
            qualifying.append(result)
    if not qualifying:
        raise RuntimeError("No fine-grid threshold achieved 60% sensitivity on half B")
    chosen = min(qualifying, key=lambda row: (row["alarm_rate"], -row["threshold"]))

    half_a_test_metrics = metrics(y_test, calibrated_test_a, chosen["threshold"])
    half_a_row = {"method": "Half-A isotonic; threshold selected on half B", **half_a_test_metrics}
    print(f"\nFINE GRID: {len(thresholds)} unique thresholds, 0.001..0.20 step 0.0005, then 0.20..0.99 step 0.01")
    print(
        f"  Half-B selected threshold={chosen['threshold']:.6f}; "
        f"sensitivity={chosen['sensitivity']:.6f}; alarm_rate={chosen['alarm_rate']:.6f}"
    )
    print_metrics_table("TEST metrics at the half-B-selected threshold", [half_a_row])

    calibration_info = json.loads(
        (MODELS_DIR / "calibration_info.json").read_text(encoding="utf-8")
    )
    if calibration_info.get("type") != "isotonic":
        raise ValueError(
            "Saved calibrator is not isotonic; cannot compare the original "
            "full-validation isotonic calibration."
        )
    full_iso = joblib.load(MODELS_DIR / "calibrator.pkl")
    if not isinstance(full_iso, IsotonicRegression):
        raise TypeError(f"Saved calibrator has unexpected type: {type(full_iso).__name__}")
    full_val_cal = full_iso.predict(raw_val)
    full_test_cal = full_iso.predict(raw_test)

    original_threshold_val = metrics(y_val, full_val_cal, OLD_THRESHOLD)
    original_threshold_test = metrics(y_test, full_test_cal, OLD_THRESHOLD)
    same_numeric_threshold_test = metrics(
        y_test, full_test_cal, chosen["threshold"]
    )
    full_rows = [
        {
            "method": "Original full-validation isotonic; old threshold (validation)",
            **original_threshold_val,
        },
        {
            "method": "Original full-validation isotonic; old threshold (test)",
            **original_threshold_test,
        },
        {
            "method": "Original full-validation isotonic; half-B threshold (test)",
            **same_numeric_threshold_test,
        },
    ]
    print_metrics_table("Original full-validation isotonic comparison", full_rows)
    print(
        "  Old supplied validation figures: threshold=0.0262, "
        "sensitivity=0.6624, alarm_rate=0.2520"
    )
    old_saved = json.loads((MODELS_DIR / "optimal_threshold.json").read_text(encoding="utf-8"))
    print(
        "  Saved optimal_threshold.json at 0.026202: "
        f"validation sensitivity={old_saved['val_metrics']['sensitivity']:.6f}, "
        f"validation alarm_rate={old_saved['val_metrics']['alarm_rate']:.6f}; "
        f"test sensitivity={old_saved['test_metrics']['sensitivity']:.6f}, "
        f"test alarm_rate={old_saved['test_metrics']['alarm_rate']:.6f}"
    )

    # Reverse-engineer an operating threshold from the current test scores only
    # to check whether the paper's displayed recall is attainable. This is
    # descriptive and must not be interpreted as held-out test evaluation.
    target = closest_target_metrics(y_test, full_test_cal, TARGET_RECALL)
    target_raw_threshold = closest_target_metrics(y_test, raw_test, TARGET_RECALL)
    target_row = {
        "method": "Full-validation isotonic; test-derived target-recall threshold",
        **target,
    }
    target_raw_row = {
        "method": "Uncalibrated raw; test-derived target-recall threshold",
        **target_raw_threshold,
    }
    print(f"\nPAPER TARGET: closest achievable test recall to {TARGET_RECALL:.2%}")
    print("  Thresholds below are reverse-engineered on test and are descriptive only.")
    print_metrics_table(
        "Test-derived threshold comparison",
        [target_row, target_raw_row],
    )


if __name__ == "__main__":
    main()
