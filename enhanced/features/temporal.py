"""
Phase 3: Temporal Feature Engineering
======================================
Reads: enhanced/data/processed/{train,val,test}_processed.parquet
Writes: enhanced/data/processed/{train,val,test}_temporal.parquet
        enhanced/data/processed/temporal_feature_cols.json

Rules (STRICT -- no data leakage):
  - All operations are within patient groups (groupby patient_id)
  - Only causal operations: shift() (past values), rolling() with min_periods=1
  - Static columns (Age, Gender, Unit1, Unit2, HospAdmTime) are passed through unchanged
  - Missingness indicator columns (*_was_missing) are passed through unchanged

Feature types generated per clinical variable:
  Lags        : t-1, t-3, t-6
  Differences : diff_1h, diff_3h, pct_change_1h
  Rolling     : mean_3h, mean_6h, mean_12h, std_3h, std_6h, min_6h, max_6h
  Trends      : slope_3h, slope_6h (linear regression slope)

GPU note: This phase is CPU-only (sklearn/pandas). GPU is used in Phase 5.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent   # repo root
PROCESSED_DIR = ROOT / "enhanced" / "data" / "processed"
OUTPUT_DIR    = PROCESSED_DIR

SPLIT_INFO_PATH = PROCESSED_DIR / "split_info.json"

# --- Clinical variables for temporal features ---------------------------------
TEMPORAL_VARS = [
    # Vitals (observed ~85-90%)
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    # Key labs (selected < 50% missing -- from audit)
    "FiO2", "pH", "PaCO2", "SaO2",
    "BUN", "Calcium", "Glucose", "Potassium",
    "Hct", "Hgb", "WBC", "Platelets",
]

STATIC_VARS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
ID_VARS     = ["patient_id", "ICULOS", "SepsisLabel"]


# --- Slope helper -------------------------------------------------------------
def _linear_slope(values: np.ndarray) -> float:
    """Linear regression slope. Returns 0.0 if fewer than 2 non-NaN values."""
    mask = ~np.isnan(values)
    n = mask.sum()
    if n < 2:
        return 0.0
    x = np.where(mask)[0].astype(float)
    y = values[mask]
    slope, *_ = sp_stats.linregress(x, y)
    return float(slope)


# --- Per-patient feature builder ----------------------------------------------
def build_temporal_features_for_group(grp: pd.DataFrame) -> pd.DataFrame:
    """
    For a single patient DataFrame (sorted by ICULOS), compute temporal features.
    Causal: all features at row t use only rows <= t.
    """
    grp = grp.sort_values("ICULOS").reset_index(drop=True)
    new_cols = {}

    for col in TEMPORAL_VARS:
        if col not in grp.columns:
            continue

        s = grp[col]

        # Lags
        new_cols[f"{col}_lag1"] = s.shift(1)
        new_cols[f"{col}_lag3"] = s.shift(3)
        new_cols[f"{col}_lag6"] = s.shift(6)

        # Differences
        new_cols[f"{col}_diff1h"] = s - s.shift(1)
        new_cols[f"{col}_diff3h"] = s - s.shift(3)
        prev1 = s.shift(1)
        new_cols[f"{col}_pct1h"]  = (s - prev1) / prev1.abs().replace(0, np.nan)

        # Rolling statistics (causal window)
        for w, label in [(3, "3h"), (6, "6h"), (12, "12h")]:
            new_cols[f"{col}_mean_{label}"] = s.rolling(window=w, min_periods=1).mean()

        for w, label in [(3, "3h"), (6, "6h")]:
            new_cols[f"{col}_std_{label}"] = s.rolling(window=w, min_periods=1).std(ddof=0)

        new_cols[f"{col}_min_6h"] = s.rolling(window=6, min_periods=1).min()
        new_cols[f"{col}_max_6h"] = s.rolling(window=6, min_periods=1).max()

        # Trend: linear slope over 3h and 6h windows
        for w, label in [(3, "3h"), (6, "6h")]:
            new_cols[f"{col}_slope_{label}"] = (
                s.rolling(window=w, min_periods=1)
                 .apply(_linear_slope, raw=True)
            )

    result = pd.concat([grp, pd.DataFrame(new_cols, index=grp.index)], axis=1)
    return result


# --- Main processing function -------------------------------------------------
def process_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}_processed.parquet"
    print(f"\n{'='*60}")
    print(f"Processing [{name}] split: {path.name}")
    df = pd.read_parquet(path)
    print(f"  Loaded  : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Patients: {df['patient_id'].nunique():,}")

    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)

    t0 = time.time()
    try:
        from tqdm import tqdm
        tqdm.pandas(desc=f"  [{name}] temporal features")
        result = (
            df.groupby("patient_id", sort=False, group_keys=False)
              .progress_apply(build_temporal_features_for_group)
        )
    except ImportError:
        n_patients = df["patient_id"].nunique()
        print(f"  (tqdm not installed -- processing {n_patients:,} patients...)")
        result = (
            df.groupby("patient_id", sort=False, group_keys=False)
              .apply(build_temporal_features_for_group)
        )

    result = result.reset_index(drop=True)
    elapsed = time.time() - t0
    n_new = result.shape[1] - df.shape[1]
    print(f"  Done in {elapsed:.1f}s | {result.shape[0]:,} rows x {result.shape[1]} cols (+{n_new} new features)")

    assert result.shape[0] == df.shape[0], "Row count mismatch -- check groupby logic."
    return result


# --- Entry point --------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase 3: Temporal Feature Engineering")
    print("=" * 60)

    with open(SPLIT_INFO_PATH) as f:
        split_info = json.load(f)

    print(f"\nLoaded split_info.json")
    print(f"  Base features: {len(split_info['feature_columns'])}")
    print(f"  Target       : {split_info['target']}")

    base_features    = set(split_info["feature_columns"])
    valid_temporal   = [v for v in TEMPORAL_VARS if v in base_features]
    skipped          = [v for v in TEMPORAL_VARS if v not in base_features]
    if skipped:
        print(f"  Warning: skipping vars not in processed data: {skipped}")
    print(f"  Temporal vars: {len(valid_temporal)} -> ~{len(valid_temporal) * 13} new features")

    # Process splits
    splits = {}
    for split_name in ["train", "val", "test"]:
        splits[split_name] = process_split(split_name)

    # Build feature column manifest
    train_df = splits["train"]
    original_cols = set(
        ID_VARS + STATIC_VARS + split_info["feature_columns"]
        + [c for c in train_df.columns if c.endswith("_was_missing")]
    )
    temporal_feature_cols = [c for c in train_df.columns if c not in original_cols]
    all_feature_cols      = split_info["feature_columns"] + temporal_feature_cols

    print(f"\nFeature summary:")
    print(f"  Base features      : {len(split_info['feature_columns'])}")
    print(f"  New temporal feats : {len(temporal_feature_cols)}")
    print(f"  Total features     : {len(all_feature_cols)}")

    temporal_info = {
        "base_feature_columns"    : split_info["feature_columns"],
        "temporal_feature_columns": temporal_feature_cols,
        "all_feature_columns"     : all_feature_cols,
        "target"                  : split_info["target"],
        "time_var"                : split_info["time_var"],
        "patient_id_col"          : "patient_id",
        "temporal_vars_used"      : valid_temporal,
        "static_vars"             : STATIC_VARS,
    }

    json_out = OUTPUT_DIR / "temporal_feature_cols.json"
    with open(json_out, "w") as f:
        json.dump(temporal_info, f, indent=2)
    print(f"\n  Saved feature manifest -> {json_out.name}")

    # Save parquets
    print("\nSaving output parquets...")
    for split_name, df in splits.items():
        out_path = OUTPUT_DIR / f"{split_name}_temporal.parquet"
        df.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {split_name}_temporal.parquet -> {size_mb:.1f} MB  ({df.shape[0]:,} rows x {df.shape[1]} cols)")

    # Sanity checks
    print("\nSanity checks:")
    sample = splits["train"]
    pid = sample["patient_id"].iloc[10]
    pat = sample[sample["patient_id"] == pid].sort_values("ICULOS").head(6)
    if "HR_lag1" in pat.columns:
        print(f"\n  Patient {pid} | HR temporal features (first 6 hours):")
        print(pat[["ICULOS", "HR", "HR_lag1", "HR_diff1h", "HR_mean_3h", "HR_slope_3h"]].to_string(index=False))

    first_hrs = sample.groupby("patient_id").first()
    lag1_null = first_hrs["HR_lag1"].isna().mean() if "HR_lag1" in first_hrs else 1.0
    print(f"\n  Lag-1 NaN at first ICU hour: {lag1_null*100:.1f}%  (should be ~100% -- causal check)")

    before = pd.read_parquet(PROCESSED_DIR / "train_processed.parquet")["SepsisLabel"].sum()
    after  = splits["train"]["SepsisLabel"].sum()
    assert before == after, "SepsisLabel sum changed -- data integrity error!"
    print(f"  SepsisLabel sum preserved: {after:,} (before={before:,}) OK")

    print("\n" + "="*60)
    print("Phase 3 COMPLETE")
    print("Next: python enhanced/features/selection.py")
    print("="*60)


if __name__ == "__main__":
    main()
