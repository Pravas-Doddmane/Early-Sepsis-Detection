#!/usr/bin/env python
"""
Phase 2: Preprocessing Pipeline (Optimized)
- Patient-level train/val/test split
- IQR capping (fit on train)
- Imputation benchmark on SAMPLE only
- Scalers fit on train only
- Save transformers + processed splits
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("enhanced/experiments")
OUTPUT_DIR = Path("enhanced/data/processed")
MODEL_DIR = Path("enhanced/models/transformers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Variables from audit with <50% missing
VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
LABS_REASONABLE = ['FiO2', 'pH', 'PaCO2', 'SaO2', 'BUN', 'Calcium', 'Glucose',
                    'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets']
STATIC_VARS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
TARGET = 'SepsisLabel'
TIME_VAR = 'ICULOS'

ALL_FEATURES = VITALS + LABS_REASONABLE + STATIC_VARS


def load_data():
    print("Loading raw data...")
    df = pd.read_parquet(DATA_DIR / "raw_combined.parquet")
    print(f"Loaded {len(df):,} records, {df['patient_id'].nunique():,} patients")
    return df


def patient_level_split(df, test_size=0.15, val_size=0.15, random_state=42):
    patients = df['patient_id'].unique()
    labels = df.groupby('patient_id')[TARGET].max()

    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, stratify=labels, random_state=random_state
    )
    train_labels = labels[train_patients]
    train_patients, val_patients = train_test_split(
        train_patients, test_size=val_size/(1-test_size),
        stratify=train_labels, random_state=random_state
    )

    train_df = df[df['patient_id'].isin(train_patients)].copy()
    val_df = df[df['patient_id'].isin(val_patients)].copy()
    test_df = df[df['patient_id'].isin(test_patients)].copy()

    print(f"Train: {len(train_patients):,} pts, {len(train_df):,} rows")
    print(f"Val:   {len(val_patients):,} pts, {len(val_df):,} rows")
    print(f"Test:  {len(test_patients):,} pts, {len(test_df):,} rows")
    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        rate = d.groupby('patient_id')[TARGET].max().mean() * 100
        print(f"  {name} sepsis rate: {rate:.2f}%")
    return train_df, val_df, test_df


def get_feature_columns(df):
    available = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"Note: Not in data: {missing}")
    return available


def fit_iqr_capper(train_df, feature_cols):
    print("\nFitting IQR capper...")
    capper = {}
    for col in feature_cols:
        if train_df[col].dtype in ['float64', 'int64']:
            q1 = train_df[col].quantile(0.25)
            q3 = train_df[col].quantile(0.75)
            iqr = q3 - q1
            capper[col] = {'lower': q1 - 1.5*iqr, 'upper': q3 + 1.5*iqr}
    joblib.dump(capper, MODEL_DIR / "iqr_capper.pkl")
    return capper


def apply_iqr_capper(df, capper):
    df = df.copy()
    for col, b in capper.items():
        if col in df.columns:
            df[col] = df[col].clip(b['lower'], b['upper'])
    return df


def benchmark_imputers_fast(train_df, val_df, feature_cols, sample_frac=0.02):
    """Benchmark on a small sample for speed."""
    print("\nBenchmarking imputers (on sample)...")
    numeric_cols = [c for c in feature_cols if train_df[c].dtype in ['float64', 'int64']]

    # Sample for benchmarking
    n_sample = min(20000, int(len(train_df) * sample_frac))
    train_sample = train_df[numeric_cols].sample(n=n_sample, random_state=42)
    val_sample = val_df[numeric_cols].sample(n=min(5000, len(val_df)), random_state=42)

    # Add missingness indicators
    for col in numeric_cols:
        train_sample[f'{col}_was_missing'] = train_sample[col].isnull().astype(int)
        val_sample[f'{col}_was_missing'] = val_sample[col].isnull().astype(int)

    results = {}

    # KNN
    print("  KNN...")
    knn = KNNImputer(n_neighbors=5, weights='distance')
    knn.fit(train_sample)
    joblib.dump(knn, MODEL_DIR / "imputer_knn.pkl")
    results['knn'] = knn

    # MICE (fast settings)
    print("  MICE...")
    mice = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1),
        max_iter=5, random_state=42, n_nearest_features=15
    )
    mice.fit(train_sample)
    joblib.dump(mice, MODEL_DIR / "imputer_mice.pkl")
    results['mice'] = mice

    # Quick eval on val sample
    for name, imp in results.items():
        val_imp = imp.transform(val_sample)
        print(f"    {name}: mean={np.nanmean(val_imp):.3f}, std={np.nanstd(val_imp):.3f}")

    return results, numeric_cols


def fit_scalers(train_df, imputer, numeric_cols):
    print("\nFitting scalers on full train (imputed)...")
    # Impute full train
    train_numeric = train_df[numeric_cols].copy()
    for col in numeric_cols:
        train_numeric[f'{col}_was_missing'] = train_numeric[col].isnull().astype(int)
    train_imputed = imputer.transform(train_numeric)

    # Choose scaler per column based on skew
    train_imp_df = pd.DataFrame(train_imputed, columns=numeric_cols + [f'{c}_was_missing' for c in numeric_cols])
    skew = train_imp_df[numeric_cols].skew()

    # Fit per-column scalers (avoids feature name mismatch)
    scalers_std = {}
    scalers_rob = {}
    for col in numeric_cols:
        scaler_std = StandardScaler()
        scaler_rob = RobustScaler()
        scaler_std.fit(train_imp_df[[col]])
        scaler_rob.fit(train_imp_df[[col]])
        scalers_std[col] = scaler_std
        scalers_rob[col] = scaler_rob

    joblib.dump(scalers_std, MODEL_DIR / "scalers_standard.pkl")
    joblib.dump(scalers_rob, MODEL_DIR / "scalers_robust.pkl")

    scaler_choice = {c: ('robust' if abs(skew[c]) > 1.0 else 'standard') for c in numeric_cols}
    return scalers_std, scalers_rob, scaler_choice


def process_split(df, imputer, scalers_std, scalers_rob, scaler_choice, numeric_cols, name):
    print(f"  Processing {name}...")
    df = df.copy()
    numeric_data = df[numeric_cols].copy()
    for col in numeric_cols:
        numeric_data[f'{col}_was_missing'] = numeric_data[col].isnull().astype(int)

    # Impute
    imputed = imputer.transform(numeric_data)
    imp_df = pd.DataFrame(imputed, columns=numeric_cols + [f'{c}_was_missing' for c in numeric_cols])

    # Scale per column
    for col in numeric_cols:
        if scaler_choice[col] == 'standard':
            imp_df[col] = scalers_std[col].transform(imp_df[[col]])
        else:
            imp_df[col] = scalers_rob[col].transform(imp_df[[col]])

    # Add back static + target + time + patient_id
    for col in STATIC_VARS:
        if col in df.columns:
            imp_df[col] = df[col].values
    imp_df[TARGET] = df[TARGET].values
    imp_df[TIME_VAR] = df[TIME_VAR].values
    imp_df['patient_id'] = df['patient_id'].values

    return imp_df


def main():
    print("=" * 60)
    print("Phase 2: Preprocessing (Fast)")
    print("=" * 60)

    df = load_data()
    train_df, val_df, test_df = patient_level_split(df)
    feature_cols = get_feature_columns(train_df)

    # IQR Capping
    capper = fit_iqr_capper(train_df, feature_cols)
    train_c = apply_iqr_capper(train_df, capper)
    val_c = apply_iqr_capper(val_df, capper)
    test_c = apply_iqr_capper(test_df, capper)

    # Imputation benchmark (on sample)
    impute_results, numeric_cols = benchmark_imputers_fast(train_c, val_c, feature_cols)

    # Use MICE as default
    default_imputer = impute_results['mice']

    # Scalers
    scalers_std, scalers_rob, scaler_choice = fit_scalers(train_c, default_imputer, numeric_cols)

    # Process all splits
    for name, split_df in [('train', train_c), ('val', val_c), ('test', test_c)]:
        processed = process_split(split_df, default_imputer, scalers_std, scalers_rob, scaler_choice, numeric_cols, name)
        processed.to_parquet(OUTPUT_DIR / f"{name}_processed.parquet", index=False)
        print(f"    Saved {name}_processed.parquet: {processed.shape}")

    # Save metadata
    import json
    split_info = {
        'feature_columns': feature_cols,
        'numeric_columns': numeric_cols,
        'target': TARGET,
        'time_var': TIME_VAR,
        'scaler_choice': scaler_choice
    }
    with open(OUTPUT_DIR / "split_info.json", 'w') as f:
        json.dump(split_info, f, default=str)

    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print(f"Data: {OUTPUT_DIR}/")
    print(f"Transformers: {MODEL_DIR}/")
    print("Next: python enhanced/features/temporal.py")
    print("=" * 60)


if __name__ == "__main__":
    main()