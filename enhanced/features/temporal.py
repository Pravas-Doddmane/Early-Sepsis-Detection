#!/usr/bin/env python
"""
Phase 3: Temporal Feature Engineering
Creates causal temporal features (lags, rolling stats, trends) per patient.
Constraint: Only hours ≤ current hour (no leakage).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = Path("enhanced/data/processed")
OUTPUT_DIR = Path("enhanced/data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base clinical variables (from Phase 2, <50% missing)
BASE_FEATURES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
    'FiO2', 'pH', 'PaCO2', 'SaO2', 'BUN', 'Calcium', 'Glucose',
    'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets'
]

STATIC_FEATURES = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
TARGET = 'SepsisLabel'
TIME_VAR = 'ICULOS'
ID_VAR = 'patient_id'

# Missingness indicators (created in Phase 2)
MISSING_INDICATORS = [f'{f}_was_missing' for f in BASE_FEATURES]


def load_split(name):
    """Load processed split."""
    df = pd.read_parquet(INPUT_DIR / f"{name}_processed.parquet")
    print(f"Loaded {name}: {df.shape}")
    return df


def create_temporal_features(df):
    """
    Create temporal features per patient (causal - only past hours).
    Groups by patient_id, sorts by ICULOS.
    """
    print("Creating temporal features...")

    # Sort by patient and time
    df = df.sort_values([ID_VAR, TIME_VAR]).reset_index(drop=True)

    # Features to create temporal for
    temporal_cols = [c for c in BASE_FEATURES if c in df.columns]
    print(f"Base temporal columns: {len(temporal_cols)}")

    # Result dataframe
    result_dfs = []

    # Process each patient
    for pid, group in tqdm(df.groupby(ID_VAR), desc="Patients"):
        group = group.copy().reset_index(drop=True)
        n_rows = len(group)

        # Static features (same for all rows)
        static_data = {col: group[col].iloc[0] for col in STATIC_FEATURES if col in group.columns}

        # Create features for each temporal column
        feat_dict = {col: group[col].values for col in temporal_cols}
        feat_dict.update({col: group[col].values for col in MISSING_INDICATORS if col in group.columns})

        # Target and time
        feat_dict[TARGET] = group[TARGET].values
        feat_dict[TIME_VAR] = group[TIME_VAR].values
        feat_dict[ID_VAR] = group[ID_VAR].values

        # Add static
        for col, val in static_data.items():
            feat_dict[col] = np.full(n_rows, val)

        # TEMPORAL FEATURES
        for col in temporal_cols:
            vals = group[col].values

            # 1. LAGS: t-1, t-3, t-6
            for lag in [1, 3, 6]:
                lagged = np.full(n_rows, np.nan)
                lagged[lag:] = vals[:-lag]
                feat_dict[f'{col}_lag{lag}'] = lagged

            # 2. DIFFERENCES: diff_1h, diff_3h, pct_change_1h
            for lag in [1, 3]:
                diff = np.full(n_rows, np.nan)
                diff[lag:] = vals[lag:] - vals[:-lag]
                feat_dict[f'{col}_diff{lag}h'] = diff

            # pct_change_1h
            pct = np.full(n_rows, np.nan)
            pct[1:] = np.where(vals[:-1] != 0, (vals[1:] - vals[:-1]) / np.abs(vals[:-1]), 0)
            feat_dict[f'{col}_pct_change1h'] = pct

            # 3. ROLLING (causal): mean, std, min, max
            for window in [3, 6, 12]:
                # Mean
                rolling_mean = np.full(n_rows, np.nan)
                for i in range(n_rows):
                    start = max(0, i - window + 1)
                    window_vals = vals[start:i+1]
                    if len(window_vals) > 0 and not np.all(np.isnan(window_vals)):
                        rolling_mean[i] = np.nanmean(window_vals)
                feat_dict[f'{col}_mean{window}h'] = rolling_mean

                # Std
                rolling_std = np.full(n_rows, np.nan)
                for i in range(n_rows):
                    start = max(0, i - window + 1)
                    window_vals = vals[start:i+1]
                    if len(window_vals) > 1 and not np.all(np.isnan(window_vals)):
                        rolling_std[i] = np.nanstd(window_vals)
                feat_dict[f'{col}_std{window}h'] = rolling_std

            # Min/Max for 6h window
            for window in [6]:
                rolling_min = np.full(n_rows, np.nan)
                rolling_max = np.full(n_rows, np.nan)
                for i in range(n_rows):
                    start = max(0, i - window + 1)
                    window_vals = vals[start:i+1]
                    if len(window_vals) > 0 and not np.all(np.isnan(window_vals)):
                        rolling_min[i] = np.nanmin(window_vals)
                        rolling_max[i] = np.nanmax(window_vals)
                feat_dict[f'{col}_min{window}h'] = rolling_min
                feat_dict[f'{col}_max{window}h'] = rolling_max

            # 4. TRENDS: Linear slope over 3h, 6h windows
            for window in [3, 6]:
                slope = np.full(n_rows, np.nan)
                for i in range(n_rows):
                    start = max(0, i - window + 1)
                    window_vals = vals[start:i+1]
                    valid_idx = ~np.isnan(window_vals)
                    if np.sum(valid_idx) >= 2:
                        x = np.arange(np.sum(valid_idx))
                        y = window_vals[valid_idx]
                        slope[i] = np.polyfit(x, y, 1)[0]
                feat_dict[f'{col}_slope{window}h'] = slope

        result_dfs.append(pd.DataFrame(feat_dict))

    result = pd.concat(result_dfs, ignore_index=True)
    print(f"Created temporal features: {result.shape}")
    return result


def save_temporal_features(train_df, val_df, test_df):
    """Save temporal features and feature list."""
    print("Saving temporal features...")

    train_df.to_parquet(OUTPUT_DIR / "train_temporal.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR / "val_temporal.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test_temporal.parquet", index=False)

    # Feature columns (exclude target, id, time)
    exclude = {TARGET, ID_VAR, TIME_VAR}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    import json
    with open(OUTPUT_DIR / "temporal_feature_cols.json", 'w') as f:
        json.dump(feature_cols, f)

    print(f"Saved temporal features:")
    print(f"  train: {train_df.shape}")
    print(f"  val:   {val_df.shape}")
    print(f"  test:  {test_df.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature list: {OUTPUT_DIR}/temporal_feature_cols.json")


def main():
    print("=" * 60)
    print("Phase 3: Temporal Feature Engineering")
    print("=" * 60)

    # Load splits
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    # Create temporal features for each split
    print("\n[1/3] Creating train temporal features...")
    train_temp = create_temporal_features(train)

    print("\n[2/3] Creating val temporal features...")
    val_temp = create_temporal_features(val)

    print("\n[3/3] Creating test temporal features...")
    test_temp = create_temporal_features(test)

    # Save
    print("\nSaving...")
    save_temporal_features(train_temp, val_temp, test_temp)

    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)
    print("Next: python enhanced/features/selection.py")


if __name__ == "__main__":
    main()