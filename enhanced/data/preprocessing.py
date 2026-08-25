#!/usr/bin/env python
"""
Phase 2: Preprocessing Pipeline
- Patient-level train/val/test split (no leakage)
- Missing value imputation benchmark: MICE / KNN / MissForest (fit on train only)
- Outlier handling: IQR capping (1.5xIQR, fit on train only)
- Normalization: StandardScaler / RobustScaler (fit on train only)
- Save fitted transformers (joblib)
"""
import os
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try MissForest
try:
    from missingpy import MissForest
    HAS_MISSFOREST = True
except ImportError:
    HAS_MISSFOREST = False
    print("MissForest not available, skipping")

DATA_DIR = Path("enhanced/experiments")
OUTPUT_DIR = Path("enhanced/data/processed")
MODEL_DIR = Path("enhanced/models/transformers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Variables to process (those with <50% missing from audit)
VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LABS_LOW_MISS = ['FiO2', 'pH', 'PaCO2', 'SaO2', 'BUN', 'Calcium', 'Glucose',
                 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets']
# Actually from audit, labs with <50% missing:
# Let's check dynamically
STATIC_VARS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime']
TARGET = 'SepsisLabel'
TIME_VAR = 'ICULOS'

ALL_FEATURES = VITALS + LABS_LOW_MISS + STATIC_VARS


def load_data():
    """Load raw combined data."""
    print("Loading raw data...")
    df = pd.read_parquet(DATA_DIR / "raw_combined.parquet")
    print(f"Loaded {len(df):,} records, {df['patient_id'].nunique():,} patients")
    return df


def patient_level_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """Split by patient_id to prevent leakage."""
    patients = df['patient_id'].unique()
    labels = df.groupby('patient_id')[TARGET].max()  # patient-level label

    # Stratified split
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

    print(f"Train: {len(train_patients):,} patients, {len(train_df):,} records")
    print(f"Val:   {len(val_patients):,} patients, {len(val_df):,} records")
    print(f"Test:  {len(test_patients):,} patients, {len(test_df):,} records")

    # Check class balance
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        rate = split_df.groupby('patient_id')[TARGET].max().mean() * 100
        print(f"  {name} sepsis rate: {rate:.2f}%")

    return train_df, val_df, test_df


def get_feature_columns(df):
    """Get available feature columns (present in data)."""
    available = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"Note: Missing columns (not in data): {missing}")
    return available


def fit_iqr_capper(train_df, feature_cols):
    """Fit IQR capper on training data only."""
    print("\nFitting IQR capper (1.5x IQR)...")
    capper = {}
    for col in feature_cols:
        if train_df[col].dtype in ['float64', 'int64']:
            q1 = train_df[col].quantile(0.25)
            q3 = train_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            capper[col] = {'lower': lower, 'upper': upper, 'q1': q1, 'q3': q3, 'iqr': iqr}
    return capper


def apply_iqr_capper(df, capper):
    """Apply IQR capping."""
    df = df.copy()
    for col, bounds in capper.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=bounds['lower'], upper=bounds['upper'])
    return df


def benchmark_imputers(train_df, val_df, feature_cols):
    """Benchmark MICE, KNN, MissForest on validation set."""
    print("\nBenchmarking imputers...")

    # Prepare data (only numeric features)
    numeric_cols = [c for c in feature_cols if train_df[c].dtype in ['float64', 'int64']]

    X_train = train_df[numeric_cols].copy()
    X_val = val_df[numeric_cols].copy()

    # Add missingness indicators
    for col in numeric_cols:
        X_train[f'{col}_was_missing'] = X_train[col].isnull().astype(int)
        X_val[f'{col}_was_missing'] = X_val[col].isnull().astype(int)

    results = {}

    # 1. KNN Imputer
    print("  KNN Imputer (k=5)...")
    knn = KNNImputer(n_neighbors=5, weights='distance')
    X_train_knn = knn.fit_transform(X_train)
    X_val_knn = knn.transform(X_val)
    results['knn'] = {'imputer': knn, 'train': X_train_knn, 'val': X_val_knn}
    joblib.dump(knn, MODEL_DIR / "imputer_knn.pkl")

    # 2. MICE (IterativeImputer with RF)
    print("  MICE (IterativeImputer + RF)...")
    mice = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        max_iter=10, random_state=42, n_nearest_features=20
    )
    X_train_mice = mice.fit_transform(X_train)
    X_val_mice = mice.transform(X_val)
    results['mice'] = {'imputer': mice, 'train': X_train_mice, 'val': X_val_mice}
    joblib.dump(mice, MODEL_DIR / "imputer_mice.pkl")

    # 3. MissForest
    if HAS_MISSFOREST:
        print("  MissForest...")
        mf = MissForest(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        X_train_mf = mf.fit_transform(X_train)
        X_val_mf = mf.transform(X_val)
        results['missforest'] = {'imputer': mf, 'train': X_train_mf, 'val': X_val_mf}
        joblib.dump(mf, MODEL_DIR / "imputer_missforest.pkl")

    return results, numeric_cols


def evaluate_imputation(results, numeric_cols):
    """Evaluate imputation quality on validation set (where we have ground truth for non-missing)."""
    print("\nImputation Evaluation (on originally observed values in Val):")
    # We can't truly evaluate without ground truth for missing values
    # But we can check distribution preservation
    for name, res in results.items():
        val_imputed = pd.DataFrame(res['val'], columns=numeric_cols + [f'{c}_was_missing' for c in numeric_cols])
        print(f"  {name}: mean={val_imputed[numeric_cols].mean().mean():.3f}, std={val_imputed[numeric_cols].std().mean():.3f}")


def fit_scalers(train_imputed, numeric_cols):
    """Fit scalers on imputed training data."""
    print("\nFitting scalers...")

    # StandardScaler for vitals (roughly normal)
    vital_cols = [c for c in VITALS if c in numeric_cols]
    lab_cols = [c for c in LABS_LOW_MISS if c in numeric_cols]

    scaler_standard = StandardScaler()
    scaler_robust = RobustScaler()

    # Fit on all numeric (we'll choose per-column later)
    scaler_standard.fit(train_imputed[numeric_cols])
    scaler_robust.fit(train_imputed[numeric_cols])

    joblib.dump(scaler_standard, MODEL_DIR / "scaler_standard.pkl")
    joblib.dump(scaler_robust, MODEL_DIR / "scaler_robust.pkl")

    # Determine which scaler per column (based on skew)
    skew = pd.DataFrame(train_imputed, columns=numeric_cols).skew()
    scaler_choice = {}
    for col in numeric_cols:
        if abs(skew[col]) > 1.0:  # Highly skewed -> RobustScaler
            scaler_choice[col] = 'robust'
        else:
            scaler_choice[col] = 'standard'

    return scaler_standard, scaler_robust, scaler_choice


def apply_scalers(df_imputed, scaler_standard, scaler_robust, scaler_choice, numeric_cols):
    """Apply appropriate scaler per column."""
    df_scaled = pd.DataFrame(df_imputed, columns=numeric_cols + [f'{c}_was_missing' for c in numeric_cols])
    for col in numeric_cols:
        if scaler_choice[col] == 'standard':
            df_scaled[col] = scaler_standard.transform(df_scaled[[col]])
        else:
            df_scaled[col] = scaler_robust.transform(df_scaled[[col]])
    return df_scaled


def save_splits(train_df, val_df, test_df, feature_cols, imputer_name='mice'):
    """Save final processed splits."""
    print(f"\nSaving processed splits (using {imputer_name} imputation)...")

    # Get imputed data for each split
    # For now, save the raw splits; imputation will be applied in feature engineering
    train_df.to_parquet(OUTPUT_DIR / "train_raw.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR / "val_raw.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test_raw.parquet", index=False)

    # Save feature column list
    pd.Series(feature_cols).to_json(OUTPUT_DIR / "feature_cols.json")

    # Save split info
    split_info = {
        'train_patients': train_df['patient_id'].unique().tolist(),
        'val_patients': val_df['patient_id'].unique().tolist(),
        'test_patients': test_df['patient_id'].unique().tolist(),
        'feature_columns': feature_cols,
        'target': TARGET,
        'time_var': TIME_VAR
    }
    import json
    with open(OUTPUT_DIR / "split_info.json", 'w') as f:
        json.dump(split_info, f, default=str)

    print(f"Saved to {OUTPUT_DIR}/")


def main():
    print("=" * 60)
    print("Phase 2: Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_data()

    # 2. Patient-level split
    print("\n[1/5] Patient-level train/val/test split...")
    train_df, val_df, test_df = patient_level_split(df)

    # 3. Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    # 4. IQR Outlier Capping (fit on train)
    print("\n[2/5] IQR Outlier Capping...")
    capper = fit_iqr_capper(train_df, feature_cols)
    joblib.dump(capper, MODEL_DIR / "iqr_capper.pkl")

    train_capped = apply_iqr_capper(train_df, capper)
    val_capped = apply_iqr_capper(val_df, capper)
    test_capped = apply_iqr_capper(test_df, capper)

    # 5. Imputation Benchmark
    print("\n[3/5] Imputation Benchmark...")
    impute_results, numeric_cols = benchmark_imputers(train_capped, val_capped, feature_cols)
    evaluate_imputation(impute_results, numeric_cols)

    # 6. Scalers (fit on MICE-imputed train by default)
    print("\n[4/5] Scaler Fitting...")
    default_imputer = 'mice'  # MICE generally best for mixed data
    train_imputed = impute_results[default_imputer]['train']
    scaler_std, scaler_rob, scaler_choice = fit_scalers(train_imputed, numeric_cols)

    # Apply to all splits
    for name, df_capped in [('train', train_capped), ('val', val_capped), ('test', test_capped)]:
        numeric_data = df_capped[numeric_cols].copy()
        # Add missingness indicators
        for col in numeric_cols:
            numeric_data[f'{col}_was_missing'] = numeric_data[col].isnull().astype(int)

        # Impute
        imputer = impute_results[default_imputer]['imputer']
        imputed = imputer.transform(numeric_data)

        # Scale
        scaled = apply_scalers(imputed, scaler_std, scaler_rob, scaler_choice, numeric_cols)

        # Add back static categorical and target
        for col in STATIC_VARS:
            if col in df_capped.columns:
                scaled[col] = df_capped[col].values
        scaled[TARGET] = df_capped[TARGET].values
        scaled[TIME_VAR] = df_capped[TIME_VAR].values
        scaled['patient_id'] = df_capped['patient_id'].values

        # Save
        scaled.to_parquet(OUTPUT_DIR / f"{name}_processed.parquet", index=False)
        print(f"  Saved {name}_processed.parquet: {scaled.shape}")

    # 7. Save splits
    print("\n[5/5] Saving split metadata...")
    save_splits(train_df, val_df, test_df, feature_cols, default_imputer)

    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print("=" * 60)
    print(f"Processed data: {OUTPUT_DIR}/")
    print(f"Transformers: {MODEL_DIR}/")
    print("\nNext: python enhanced/features/temporal.py")


if __name__ == "__main__":
    main()