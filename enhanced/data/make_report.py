import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("enhanced/experiments")
patient_stats = pd.read_csv(OUTPUT_DIR / "patient_stats.csv")
var_stats = pd.read_csv(OUTPUT_DIR / "variable_stats.csv")
df = pd.read_parquet(OUTPUT_DIR / "raw_combined.parquet")

total_patients = len(patient_stats)
total_hours = len(df)
sepsis_patients = patient_stats["sepsis_label"].sum()
sepsis_rate = sepsis_patients / total_patients * 100
icu_stats = patient_stats["n_hours"].describe()
miss_stats = patient_stats["missingness_pct"].describe()
hourly_sepsis = df.groupby("ICULOS")["SepsisLabel"].mean().reset_index()
hourly_sepsis.columns = ["ICULOS", "sepsis_rate"]
sepsis_patients_df = patient_stats[patient_stats["sepsis_label"] == 1]
onset_stats = sepsis_patients_df["first_sepsis_hour"].describe() if len(sepsis_patients_df) > 0 else None

lines = []
lines.append("# PhysioNet 2019 Sepsis Data Audit Report")
lines.append("")
lines.append("## Dataset Overview")
lines.append("- **Total Patients**: %d" % total_patients)
lines.append("- **Total Hourly Records**: %d" % total_hours)
lines.append("- **Sepsis-Positive Patients**: %d (%.2f%%)" % (sepsis_patients, sepsis_rate))
lines.append("- **Sepsis-Negative Patients**: %d (%.2f%%)" % (total_patients - sepsis_patients, 100 - sepsis_rate))
lines.append("")
lines.append("## ICU Stay Length Distribution")
lines.append("| Statistic | Value (hours) |")
lines.append("|-----------|---------------|")
lines.append("| Count | %d |" % int(icu_stats["count"]))
lines.append("| Mean | %.1f |" % icu_stats["mean"])
lines.append("| Std | %.1f |" % icu_stats["std"])
lines.append("| Min | %d |" % int(icu_stats["min"]))
lines.append("| 25%% | %d |" % int(icu_stats["25%"]))
lines.append("| 50%% (Median) | %d |" % int(icu_stats["50%"]))
lines.append("| 75%% | %d |" % int(icu_stats["75%"]))
lines.append("| Max | %d |" % int(icu_stats["max"]))
lines.append("")
lines.append("## Patient-Level Missingness Distribution")
lines.append("| Statistic | Value (%%%) |")
lines.append("|-----------|-----------|")
lines.append("| Mean | %.1f |" % miss_stats["mean"])
lines.append("| Std | %.1f |" % miss_stats["std"])
lines.append("| Min | %.1f |" % miss_stats["min"])
lines.append("| 25%% | %.1f |" % miss_stats["25%"])
lines.append("| 50%% (Median) | %.1f |" % miss_stats["50%"])
lines.append("| 75%% | %.1f |" % miss_stats["75%"])
lines.append("| Max | %.1f |" % miss_stats["max"])
lines.append("")

if onset_stats is not None:
    lines.append("## Sepsis Onset Timing (Sepsis-Positive Patients Only)")
    lines.append("| Statistic | Value (hours) |")
    lines.append("|-----------|---------------|")
    lines.append("| Count | %d |" % int(onset_stats["count"]))
    lines.append("| Mean | %.1f |" % onset_stats["mean"])
    lines.append("| Std | %.1f |" % onset_stats["std"])
    lines.append("| Min | %d |" % int(onset_stats["min"]))
    lines.append("| 25%% | %d |" % int(onset_stats["25%"]))
    lines.append("| 50%% (Median) | %d |" % int(onset_stats["50%"]))
    lines.append("| 75%% | %d |" % int(onset_stats["75%"]))
    lines.append("| Max | %d |" % int(onset_stats["max"]))
    lines.append("")

lines.append("## Variable-Level Missingness (Top 20 Most Missing)")
top_missing = var_stats.nlargest(20, "pct_missing")[["variable", "pct_missing", "mean", "median"]]
lines.append("| variable | pct_missing | mean | median |")
lines.append("|----------|-------------|------|--------|")
for _, row in top_missing.iterrows():
    lines.append("| %s | %.1f | %.2f | %.2f |" % (row["variable"], row["pct_missing"], row["mean"], row["median"]))
lines.append("")

lines.append("## Variable-Level Missingness (Least Missing - Fully Observed)")
least_missing = var_stats.nsmallest(10, "pct_missing")[["variable", "pct_missing", "mean", "median"]]
lines.append("| variable | pct_missing | mean | median |")
lines.append("|----------|-------------|------|--------|")
for _, row in least_missing.iterrows():
    lines.append("| %s | %.1f | %.2f | %.2f |" % (row["variable"], row["pct_missing"], row["mean"], row["median"]))
lines.append("")

lines.append("## Key Clinical Variables Summary Statistics")
key_vars = ["HR", "MAP", "Lactate", "Temp", "Resp", "O2Sat", "WBC", "Creatinine", "Platelets", "Bilirubin_total", "Glucose", "Age", "Gender"]
key_stats = var_stats[var_stats["variable"].isin(key_vars)][["variable", "pct_missing", "mean", "std", "median", "min", "max"]]
lines.append("| variable | pct_missing | mean | std | median | min | max |")
lines.append("|----------|-------------|------|-----|--------|-----|-----|")
for _, row in key_stats.iterrows():
    lines.append("| %s | %.1f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (row["variable"], row["pct_missing"], row["mean"], row["std"], row["median"], row["min"], row["max"]))
lines.append("")

lines.append("## Hourly Sepsis Rate (Class Balance Over Time)")
lines.append("| ICULOS (hour) | Sepsis Rate |")
lines.append("|---------------|-------------|")
for _, row in hourly_sepsis.head(30).iterrows():
    lines.append("| %3d | %.4f |" % (int(row["ICULOS"]), row["sepsis_rate"]))
if len(hourly_sepsis) > 30:
    lines.append("| ... | ... |")
    for _, row in hourly_sepsis.tail(5).iterrows():
        lines.append("| %3d | %.4f |" % (int(row["ICULOS"]), row["sepsis_rate"]))
lines.append("")

lines.append("## Data Quality Observations")
lines.append("1. **High Missingness**: Most lab variables have >80%% missing values")
lines.append("2. **Vitals More Complete**: HR, O2Sat, SBP, MAP, DBP, Resp have lower missingness")
lines.append("3. **Static Variables**: Age, Gender, Unit1, Unit2, HospAdmTime are constant per patient")
lines.append("4. **Class Imbalance**: Sepsis rate ~%.1f%% - highly imbalanced" % sepsis_rate)
lines.append("5. **Temporal Pattern**: Sepsis rate increases with ICU hours (expected)")
lines.append("6. **ICU Stay**: Median stay ~%d hours, max %d hours" % (int(icu_stats["50%"]), int(icu_stats["max"])))
lines.append("")
lines.append("## Recommendations for Preprocessing")
lines.append("1. **Imputation Strategy**: Use MICE/KNN/MissForest benchmark; vitals vs labs may need different approaches")
lines.append("2. **Outlier Handling**: IQR capping (1.5xIQR) fitted on training patients only")
lines.append("3. **Normalization**: RobustScaler for skewed lab distributions; StandardScaler for vitals")
lines.append("4. **Feature Engineering**: Focus on variables with <50%% missingness for temporal features")
lines.append("5. **Patient-Level Split**: Ensure no patient appears in multiple splits")
lines.append("")
lines.append("---")
lines.append("*Generated by enhanced/data/audit.py*")

report = "\n".join(lines)
report_path = OUTPUT_DIR / "audit_report.md"
with open(report_path, "w") as f:
    f.write(report)
print("Report generated:", report_path)