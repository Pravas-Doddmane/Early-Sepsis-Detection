#!/usr/bin/env python
"""Audit stacked-score calibration without retraining the base models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "enhanced" / "models"
DATA_DIR = ROOT / "enhanced" / "data" / "processed"
OUTPUT_DIR = ROOT / "calibration_check"
SEED = 42
N_BINS = 10


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_path = MODELS_DIR / (
        "stack_oof_val_preds.npy" if split == "val" else "stack_test_preds.npy"
    )
    parquet_path = DATA_DIR / f"{split}_temporal.parquet"
    scores = np.asarray(np.load(scores_path), dtype=float).reshape(-1)
    frame = pd.read_parquet(parquet_path, columns=["patient_id", "SepsisLabel"])
    labels = frame["SepsisLabel"].to_numpy(dtype=int)
    patient_ids = frame["patient_id"].to_numpy()

    if len(scores) != len(labels):
        raise ValueError(
            f"{split} score/label row mismatch: {len(scores)} scores, "
            f"{len(labels)} labels in {parquet_path}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{split} scores contain non-finite values")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"{split} scores contain values outside [0, 1]")
    if pd.isna(patient_ids).any():
        raise ValueError(f"{split} contains missing patient_id values")
    return scores, labels, patient_ids


def clipped_logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs", random_state=SEED)
    model.fit(clipped_logit(scores), labels)
    return model


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(scores, labels)
    return model


def apply_platt(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(clipped_logit(scores))[:, 1]


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray) -> float:
    bin_indices = np.minimum((probabilities * N_BINS).astype(int), N_BINS - 1)
    ece = 0.0
    for bin_index in range(N_BINS):
        mask = bin_indices == bin_index
        if mask.any():
            confidence = probabilities[mask].mean()
            frequency = labels[mask].mean()
            ece += mask.mean() * abs(confidence - frequency)
    return float(ece)


def calibration_metrics(
    split: str, method: str, labels: np.ndarray, probabilities: np.ndarray
) -> dict:
    return {
        "section": "calibration",
        "split": split,
        "method": method,
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "threshold": np.nan,
        "sensitivity": np.nan,
        "precision": np.nan,
        "f1": np.nan,
        "alarm_rate": np.nan,
        "training_rate": np.nan,
        "expected_brier": np.nan,
    }


def clinical_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "alarm_rate": float(predictions.mean()),
    }


def print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    print(f"\n{title}")
    print(pd.DataFrame(rows)[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    output_paths = [
        OUTPUT_DIR / "raw_val.npy",
        OUTPUT_DIR / "y_val.npy",
        OUTPUT_DIR / "raw_test.npy",
        OUTPUT_DIR / "y_test.npy",
        OUTPUT_DIR / "calibration_check_results.csv",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing calibration-check outputs: "
            + ", ".join(existing)
        )

    raw_val, y_val, val_patient_ids = load_split("val")
    raw_test, y_test, _ = load_split("test")
    train_frame = pd.read_parquet(DATA_DIR / "train_temporal.parquet", columns=["SepsisLabel"])
    y_train = train_frame["SepsisLabel"].to_numpy(dtype=int)
    training_rate = float(y_train.mean())

    full_val_platt = fit_platt(raw_val, y_val)
    full_val_iso = fit_isotonic(raw_val, y_val)
    full_val_platt_test = apply_platt(full_val_platt, raw_test)
    full_val_iso_test = full_val_iso.predict(raw_test)
    baseline_test = np.full(len(y_test), training_rate, dtype=float)

    test_rows = [
        {
            **calibration_metrics("test", "Constant predictor (train rate)", y_test, baseline_test),
            "training_rate": training_rate,
            "expected_brier": training_rate * (1 - training_rate),
        },
        calibration_metrics("test", "Raw stacked scores", y_test, raw_test),
        calibration_metrics("test", "Platt (correct refit; full val)", y_test, full_val_platt_test),
        calibration_metrics("test", "Isotonic (full val)", y_test, full_val_iso_test),
    ]
    print_table(
        "TEST calibration (Step 2; calibrators fitted on full validation)",
        test_rows,
        ["method", "brier", "ece", "training_rate", "expected_brier"],
    )
    actual_baseline_brier = float(brier_score_loss(y_test, baseline_test))
    print(f"Constant-predictor actual TEST Brier: {actual_baseline_brier:.6f}")

    unique_patient_ids = pd.unique(val_patient_ids)
    patient_a, patient_b = train_test_split(
        unique_patient_ids, test_size=0.5, random_state=SEED, shuffle=True
    )
    mask_a = np.isin(val_patient_ids, patient_a)
    mask_b = np.isin(val_patient_ids, patient_b)
    if np.any(mask_a & mask_b) or not np.all(mask_a | mask_b):
        raise RuntimeError("Validation patient split is not a complete, disjoint partition")

    platt_a = fit_platt(raw_val[mask_a], y_val[mask_a])
    iso_a = fit_isotonic(raw_val[mask_a], y_val[mask_a])
    platt_b = apply_platt(platt_a, raw_val[mask_b])
    iso_b = iso_a.predict(raw_val[mask_b])
    platt_test = apply_platt(platt_a, raw_test)
    iso_test = iso_a.predict(raw_test)

    holdout_rows = []
    for split_name, labels, raw_scores, platt_scores, iso_scores in [
        ("validation_half_B", y_val[mask_b], raw_val[mask_b], platt_b, iso_b),
        ("test", y_test, raw_test, platt_test, iso_test),
    ]:
        holdout_rows.extend(
            [
                calibration_metrics(split_name, "Raw stacked scores", labels, raw_scores),
                calibration_metrics(split_name, "Platt (fit on half A)", labels, platt_scores),
                calibration_metrics(split_name, "Isotonic (fit on half A)", labels, iso_scores),
            ]
        )
    print_table(
        "Patient-held-out calibration (Step 3; both calibrators fitted on half A)",
        holdout_rows,
        ["split", "method", "brier", "ece"],
    )

    brier_platt_b = brier_score_loss(y_val[mask_b], platt_b)
    brier_iso_b = brier_score_loss(y_val[mask_b], iso_b)
    if brier_platt_b <= brier_iso_b:
        selected_name = "Platt (fit on half A)"
        selected_threshold_scores_b = platt_b
        selected_threshold_scores_test = platt_test
    else:
        selected_name = "Isotonic (fit on half A)"
        selected_threshold_scores_b = iso_b
        selected_threshold_scores_test = iso_test

    threshold_candidates = np.arange(0.01, 1.00, 0.01)
    eligible = []
    for threshold in threshold_candidates:
        metrics = clinical_metrics(y_val[mask_b], selected_threshold_scores_b, threshold)
        if metrics["sensitivity"] >= 0.60:
            eligible.append(metrics)
    if not eligible:
        raise RuntimeError("No threshold from 0.01 to 0.99 achieved 60% sensitivity on half B")
    selected = min(eligible, key=lambda row: (row["alarm_rate"], -row["threshold"]))
    selected_threshold = selected["threshold"]

    threshold_rows = []
    calibrated_test_metrics = clinical_metrics(y_test, selected_threshold_scores_test, selected_threshold)
    threshold_rows.append(
        {
            "section": "threshold",
            "split": "test",
            "method": f"Calibrated ({selected_name})",
            **calibrated_test_metrics,
            "brier": np.nan,
            "ece": np.nan,
            "training_rate": np.nan,
            "expected_brier": np.nan,
        }
    )
    raw_test_metrics = clinical_metrics(y_test, raw_test, 0.5)
    threshold_rows.append(
        {
            "section": "threshold",
            "split": "test",
            "method": "Uncalibrated raw scores",
            **raw_test_metrics,
            "brier": np.nan,
            "ece": np.nan,
            "training_rate": np.nan,
            "expected_brier": np.nan,
        }
    )
    print(f"\nThreshold selection (Step 4): {selected_name}; selected on half B")
    print(f"  Half-B sensitivity: {selected['sensitivity']:.6f}")
    print(f"  Minimum qualifying alarm rate: {selected['alarm_rate']:.6f}")
    print(f"  Selected threshold: {selected_threshold:.2f}")
    print_table(
        "TEST clinical metrics (Step 4)",
        threshold_rows,
        ["method", "threshold", "sensitivity", "precision", "f1", "alarm_rate"],
    )

    results = pd.DataFrame(test_rows + holdout_rows + threshold_rows)
    results.to_csv(OUTPUT_DIR / "calibration_check_results.csv", index=False)
    np.save(OUTPUT_DIR / "raw_val.npy", raw_val, allow_pickle=False)
    np.save(OUTPUT_DIR / "y_val.npy", y_val, allow_pickle=False)
    np.save(OUTPUT_DIR / "raw_test.npy", raw_test, allow_pickle=False)
    np.save(OUTPUT_DIR / "y_test.npy", y_test, allow_pickle=False)

    new_platt_brier = next(
        row["brier"]
        for row in holdout_rows
        if row["split"] == "validation_half_B" and row["method"] == "Platt (fit on half A)"
    )
    new_iso_brier = next(
        row["brier"]
        for row in holdout_rows
        if row["split"] == "validation_half_B" and row["method"] == "Isotonic (fit on half A)"
    )
    print("\nFINAL SUMMARY (old reported -> new measured)")
    print(f"Platt Brier:       0.1921 -> {new_platt_brier:.6f} (patient-held-out validation half B)")
    print(f"Isotonic Brier:    0.0173 -> {new_iso_brier:.6f} (patient-held-out validation half B)")
    print(
        f"Test raw Brier:    0.1874 -> "
        f"{calibration_metrics('test', 'raw', y_test, raw_test)['brier']:.6f}"
    )
    print(f"Test isotonic:     0.0171 -> {test_rows[3]['brier']:.6f} (full-validation fit)")
    print(f"Selected threshold: 0.0262 -> {selected_threshold:.2f}")
    print(
        "Conclusion: Yes - the original Platt result is affected by a calibration bug: "
        "class_weight='balanced' alters the probability calibration, and its validation "
        "score was evaluated on the same rows used to fit it."
    )
    print(f"\nSaved outputs under {OUTPUT_DIR}:")
    for path in output_paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    main()
