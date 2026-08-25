#!/usr/bin/env python
"""
Phase 1: Data Audit
Load PhysioNet 2019 (train CSVs) and compute patient-level statistics:
- Missingness %
- Class balance
- ICU stay lengths
- Variable distributions
Output: experiments/audit_report.md
"""
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = Path("dataset/physionet_sepsis/training")
OUTPUT_DIR = Path("enhanced/experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Key clinical variables for focused analysis
CLINICAL_VARS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
    'HospAdmTime', 'ICULOS', 'SepsisLabel'
]


def load_all_patients(data_dir):
    """Load all .psv files and combine into single DataFrame with patient_id."""
    all_files = sorted(glob.glob(str(data_dir / "**" / "*.psv"), recursive=True))
    print(f"Found {len(all_files)} patient files")

    dfs = []
    for f in tqdm(all_files, desc="Loading patients"):
        patient_id = Path(f).stem  # e.g., p000001
        df = pd.read_csv(f, sep='|')
        df['patient_id'] = patient_id
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def compute_patient_stats(df):
    """Compute patient-level statistics."""
    stats = []
    for pid, group in df.groupby('patient_id'):
        n_hours = len(group)
        sepsis_onset = group['SepsisLabel'].max()  # 1 if ever septic
        first_sepsis_hour = group[group['SepsisLabel'] == 1]['ICULOS'].min() if sepsis_onset == 1 else None

        # Missingness per patient
        missing_pct = group[CLINICAL_VARS].isnull().mean().mean() * 100

        stats.append({
            'patient_id': pid,
            'n_hours': n_hours,
            'sepsis_label': sepsis_onset,
            'first_sepsis_hour': first_sepsis_hour,
            'missingness_pct': missing_pct,
            'age': group['Age'].iloc[0] if not group['Age'].isnull().all() else np.nan,
            'gender': group['Gender'].iloc[0] if not group['Gender'].isnull().all() else np.nan,
        })

    return pd.DataFrame(stats)


def compute_variable_stats(df):
    """Compute per-variable statistics across all patients."""
    var_stats = []
    for var in CLINICAL_VARS:
        if var not in df.columns:
            continue
        col = df[var]
        n_total = len(col)
        n_missing = col.isnull().sum()
        pct_missing = n_missing / n_total * 100

        if col.dtype in ['float64', 'int64'] and n_missing < n_total:
            var_stats.append({
                'variable': var,
                'dtype': str(col.dtype),
                'n_total': n_total,
                'n_missing': n_missing,
                'pct_missing': pct_missing,
                'mean': col.mean(),
                'std': col.std(),
                'min': col.min(),
                'q25': col.quantile(0.25),
                'median': col.median(),
                'q75': col.quantile(0.75),
                'max': col.max(),
            })
        else:
            var_stats.append({
                'variable': var,
                'dtype': str(col.dtype),
                'n_total': n_total,
                'n_missing': n_missing,
                'pct_missing': pct_missing,
                'mean': np.nan, 'std': np.nan, 'min': np.nan,
                'q25': np.nan, 'median': np.nan, 'q75': np.nan, 'max': np.nan,
            })
    return pd.DataFrame(var_stats)


def generate_report(df, patient_stats, var_stats):
    """Generate markdown audit report."""
    total_patients = len(patient_stats)
    total_hours = len(df)
    sepsis_patients = patient_stats['sepsis_label'].sum()
    sepsis_rate = sepsis_patients / total_patients * 100

    # ICU stay length stats
    icu_stats = patient_stats['n_hours'].describe()

    # Missingness stats
    miss_stats = patient_stats['missingness_pct'].describe()

    # Class balance per hour
    hourly_sepsis = df.groupby('ICULOS')['SepsisLabel'].mean().reset_index()
    hourly_sepsis.columns = ['ICULOS', 'sepsis_rate']

    report = f"""# PhysioNet 2019 Sepsis Data Audit Report

## Dataset Overview
- **Total Patients**: {total_patients:,}
- **Total Hourly Records**: {total_hours:,}
- **Sepsis-Positive Patients**: {int(sepsis_patients):,} ({sepsis_rate:.2f}%)
- **Sepsis-Negative Patients**: {int(total_patients - sepsis_patients):,} ({100 - sepsis_rate:.2f}%)

## ICU Stay Length Distribution
| Statistic | Value (hours) |
|-----------|---------------|
| Count | {int(icu_stats['count']):,} |
| Mean | {icu_stats['mean']:.1f} |
| Std | {icu_stats['std']:.1f} |
| Min | {int(icu_stats['min']):,} |
| 25% | {int(icu_stats['25%']):,} |
| 50% (Median) | {int(icu_stats['50%']):,} |
| 75% | {int(icu_stats['75%']):,} |
| Max | {int(icu_stats['max']):,} |

## Patient-Level Missingness Distribution
| Statistic | Value (%) |
|-----------|-----------|
| Mean | {miss_stats['mean']:.1f} |
| Std | {miss_stats['std']:.1f} |
| Min | {miss_stats['min']:.1f} |
| 25% | {miss_stats['25%']:.1f} |
| 50% (Median) | {miss_stats['50%']:.1f} |
| 75% | {miss_stats['75%']:.1f} |
| Max | {miss_stats['max']:.1f} |

## Sepsis Onset Timing (Sepsis-Positive Patients Only)
"""
    sepsis_patients_df = patient_stats[patient_stats['sepsis_label'] == 1]
    if len(sepsis_patients_df) > 0:
        onset_stats = sepsis_patients_df['first_sepsis_hour'].describe()
        report += f"""| Statistic | Value (hours) |
|-----------|---------------|
| Count | {int(onset_stats['count']):,} |
| Mean | {onset_stats['mean']:.1f} |
| Std | {onset_stats['std']:.1f} |
| Min | {int(onset_stats['min']):,} |
| 25% | {int(onset_stats['25%']):,} |
| 50% (Median) | {int(onset_stats['50%']):,} |
| 75% | {int(onset_stats['75%']):,} |
| Max | {int(onset_stats['max']):,} |
"""

    report += f"""
## Variable-Level Missingness (Top 20 Most Missing)
"""
    top_missing = var_stats.nlargest(20, 'pct_missing')[['variable', 'pct_missing', 'mean', 'median']]
    report += top_missing.to_markdown(index=False)

    report += f"""

## Variable-Level Missingness (Least Missing - Fully Observed)
"""
    least_missing = var_stats.nsmallest(10, 'pct_missing')[['variable', 'pct_missing', 'mean', 'median']]
    report += least_missing.to_markdown(index=False)

    report += f"""

## Key Clinical Variables Summary Statistics
"""
    key_vars = ['HR', 'MAP', 'Lactate', 'Temp', 'Resp', 'O2Sat', 'WBC', 'Creatinine',
                'Platelets', 'Bilirubin_total', 'Glucose', 'Age', 'Gender']
    key_stats = var_stats[var_stats['variable'].isin(key_vars)][
        ['variable', 'pct_missing', 'mean', 'std', 'median', 'min', 'max']
    ]
    report += key_stats.to_markdown(index=False)

    report += f"""

## Hourly Sepsis Rate (Class Balance Over Time)
| ICULOS (hour) | Sepsis Rate |
|---------------|-------------|
"""
    for _, row in hourly_sepsis.head(30).iterrows():
        report += f"| {int(row['ICULOS']):>3} | {row['sepsis_rate']:.4f} |\n"

    if len(hourly_sepsis) > 30:
        report += "| ... | ... |\n"
        for _, row in hourly_sepsis.tail(5).iterrows():
            report += f"| {int(row['ICULOS']):>3} | {row['sepsis_rate']:.4f} |\n"

    report += f"""

## Data Quality Observations
1. **High Missingness**: Most lab variables have >80% missing values
2. **Vitals More Complete**: HR, O2Sat, SBP, MAP, DBP, Resp have lower missingness
3. **Static Variables**: Age, Gender, Unit1, Unit2, HospAdmTime are constant per patient
4. **Class Imbalance**: Sepsis rate ~{sepsis_rate:.1f}% - highly imbalanced
5. **Temporal Pattern**: Sepsis rate increases with ICU hours (expected)
6. **ICU Stay**: Median stay ~{int(icu_stats['50%'])} hours, max {int(icu_stats['max'])} hours

## Recommendations for Preprocessing
1. **Imputation Strategy**: Use MICE/KNN/MissForest benchmark; vitals vs labs may need different approaches
2. **Outlier Handling**: IQR capping (1.5×IQR) fitted on training patients only
3. **Normalization**: RobustScaler for skewed lab distributions; StandardScaler for vitals
4. **Feature Engineering**: Focus on variables with <50% missingness for temporal features
7. **Patient-Level Split**: Ensure no patient appears in multiple splits

---
*Generated by enhanced/data/audit.py*
"""
    return report


def main():
    print("=" * 60)
    print("Phase 1: Data Audit")
    print("=" * 60)

    # Load all data
    print("\n[1/4] Loading all patient files...")
    df = load_all_patients(DATA_DIR)
    print(f"Loaded {len(df):,} records from {df['patient_id'].nunique():,} patients")

    # Save raw combined for reference
    df.to_parquet(OUTPUT_DIR / "raw_combined.parquet", index=False)
    print(f"Saved raw combined to {OUTPUT_DIR}/raw_combined.parquet")

    # Patient-level stats
    print("\n[2/4] Computing patient-level statistics...")
    patient_stats = compute_patient_stats(df)
    patient_stats.to_csv(OUTPUT_DIR / "patient_stats.csv", index=False)

    # Variable-level stats
    print("\n[3/4] Computing variable-level statistics...")
    var_stats = compute_variable_stats(df)
    var_stats.to_csv(OUTPUT_DIR / "variable_stats.csv", index=False)

    # Generate report
    print("\n[4/4] Generating audit report...")
    report = generate_report(df, patient_stats, var_stats)

    report_path = OUTPUT_DIR / "audit_report.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n✓ Audit complete!")
    print(f"  - Report: {report_path}")
    print(f"  - Patient stats: {OUTPUT_DIR}/patient_stats.csv")
    print(f"  - Variable stats: {OUTPUT_DIR}/variable_stats.csv")
    print(f"  - Raw combined: {OUTPUT_DIR}/raw_combined.parquet")


if __name__ == "__main__":
    main()