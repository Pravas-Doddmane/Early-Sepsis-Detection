"""
Phase 9 — Explainable AI (SHAP & LIME)
======================================
Comprehensive explainability pipeline explaining the CatBoost model
(best performing base model, PR-AUC 0.0709, ROC-AUC 0.7808) and
patient-level risk dynamics using:

1. Global SHAP Beeswarm Plot (overall feature rankings & directions)
2. Temporal SHAP Feature Shifts (Early vs Mid vs Late ICU stay)
3. Patient-level SHAP Waterfall Plots (True Positive & True Negative examples)
4. Patient-level LIME Explanations (Feature contribution bar charts)

Outputs saved to: enhanced/experiments/xai/
"""
import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from lime.lime_tabular import LimeTabularExplainer

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "enhanced" / "models"
EXPERIMENTS = ROOT / "enhanced" / "experiments"
DATA_DIR = ROOT / "enhanced" / "data" / "processed"
RAW_DATA_PATH = EXPERIMENTS / "raw_combined.parquet"
XAI_DIR = EXPERIMENTS / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

N_GLOBAL_SAMPLE = 300      # Sample of patients for global SHAP & LIME background
RANDOM_STATE = 42


def compute_temporal_for_patients(df, features_list):
    """Computes causal temporal features for a DataFrame of patients."""
    base_vars = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
                 'FiO2', 'pH', 'PaCO2', 'SaO2', 'BUN', 'Calcium', 'Glucose',
                 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets']
    static_vars = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

    results = []
    for pid, group in df.groupby('patient_id'):
        group = group.sort_values('ICULOS').reset_index(drop=True)
        n = len(group)
        f_dict = {'patient_id': group['patient_id'].values, 'ICULOS': group['ICULOS'].values}
        if 'SepsisLabel' in group.columns:
            f_dict['SepsisLabel'] = group['SepsisLabel'].values

        for s in static_vars:
            if s in group.columns:
                f_dict[s] = group[s].values
            else:
                f_dict[s] = np.zeros(n)

        for col in base_vars:
            if col in group.columns:
                s = group[col].ffill().bfill().fillna(0)
                vals = s.values
                f_dict[col] = group[col].fillna(0).values
                f_dict[f'{col}_was_missing'] = group[col].isna().astype(int).values

                for lag in [1, 3, 6]:
                    lag_v = np.zeros(n)
                    if n > lag:
                        lag_v[lag:] = vals[:-lag]
                    f_dict[f'{col}_lag{lag}'] = lag_v

                for lag in [1, 3]:
                    diff_v = np.zeros(n)
                    if n > lag:
                        diff_v[lag:] = vals[lag:] - vals[:-lag]
                    f_dict[f'{col}_diff{lag}h'] = diff_v

                for w in [3, 6, 12]:
                    f_dict[f'{col}_mean{w}h'] = s.rolling(w, min_periods=1).mean().values
                    f_dict[f'{col}_std{w}h'] = s.rolling(w, min_periods=1).std().fillna(0).values
                f_dict[f'{col}_min6h'] = s.rolling(6, min_periods=1).min().values
                f_dict[f'{col}_max6h'] = s.rolling(6, min_periods=1).max().values

                slopes = np.zeros(n)
                for i in range(n):
                    w_vals = vals[max(0, i-2):i+1]
                    if len(w_vals) >= 2:
                        slopes[i] = (w_vals[-1] - w_vals[0]) / (len(w_vals) - 1)
                f_dict[f'{col}_slope3h'] = slopes

        df_p = pd.DataFrame(f_dict)
        results.append(df_p)

    full_res = pd.concat(results, ignore_index=True)
    for f in features_list:
        if f not in full_res.columns:
            full_res[f] = 0.0
    return full_res


def load_model_and_data():
    with open(EXPERIMENTS / "selected_features.json") as f:
        sel = json.load(f)
    features = sel if isinstance(sel, list) else sel.get("final_features", [])

    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_model.cbm"))

    temporal_test_path = DATA_DIR / "test_temporal.parquet"
    if temporal_test_path.exists():
        print("Loading precomputed test_temporal.parquet...")
        test = pd.read_parquet(temporal_test_path)
    else:
        print(f"Sampling {N_GLOBAL_SAMPLE} patients from raw_combined.parquet and computing temporal features...")
        raw_df = pd.read_parquet(RAW_DATA_PATH)
        
        # Sample both positive and negative patients
        pos_pts = raw_df[raw_df['SepsisLabel'] == 1]['patient_id'].unique()
        neg_pts = raw_df[raw_df['SepsisLabel'] == 0]['patient_id'].unique()
        
        rng = np.random.RandomState(RANDOM_STATE)
        sample_pos = rng.choice(pos_pts, size=min(150, len(pos_pts)), replace=False)
        sample_neg = rng.choice(neg_pts, size=min(150, len(neg_pts)), replace=False)
        sample_pts = np.concatenate([sample_pos, sample_neg])
        
        sample_raw = raw_df[raw_df['patient_id'].isin(sample_pts)].copy()
        test = compute_temporal_for_patients(sample_raw, features)
        print(f"Computed temporal features for {len(sample_pts)} patients: {test.shape}")

    available = [f for f in features if f in test.columns]
    return model, test, available


def global_shap(model, test, available):
    print("\n[1/4] Computing Global SHAP values (Beeswarm & Importance Ranking)...")
    X_sample = test[available].fillna(0).values.astype(np.float32)
    if len(X_sample) > 2000:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X_sample), size=2000, replace=False)
        X_eval = X_sample[idx]
    else:
        X_eval = X_sample

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X_eval))

    # Summary plot
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values, X_eval, feature_names=available, show=False, max_display=20)
    plt.title("Global Feature Impact on Sepsis Prediction (SHAP Summary)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(XAI_DIR / "global_shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {XAI_DIR}/global_shap_beeswarm.png")

    # Feature importance CSV
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': available,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    importance_df.to_csv(XAI_DIR / "global_shap_importance.csv", index=False)
    print(f"  Saved: {XAI_DIR}/global_shap_importance.csv")

    return explainer, X_sample


def temporal_shap(model, test, available, explainer, top_n_features=10):
    print("\n[2/4] Computing Temporal SHAP Importance (Stay Progression Dynamics)...")
    rng = np.random.RandomState(RANDOM_STATE)
    bins = {
        'Early (Hour 1-6)': (1, 6),
        'Mid (Hour 7-24)': (7, 24),
        'Late (Hour 25+)': (25, 999),
    }

    bin_importances = {}
    for bin_name, (lo, hi) in bins.items():
        bin_rows = test[(test['ICULOS'] >= lo) & (test['ICULOS'] <= hi)]
        if len(bin_rows) == 0:
            continue
        n_sample = min(1000, len(bin_rows))
        sample_idx = rng.choice(bin_rows.index, size=n_sample, replace=False)
        X_bin = bin_rows.loc[sample_idx, available].fillna(0).values.astype(np.float32)
        shap_bin = explainer.shap_values(Pool(X_bin))
        bin_importances[bin_name] = np.abs(shap_bin).mean(axis=0)

    importance_df = pd.DataFrame(bin_importances, index=available)
    top_features = importance_df.mean(axis=1).nlargest(top_n_features).index
    plot_df = importance_df.loc[top_features]

    plt.figure(figsize=(10, 7))
    plot_df.plot(kind='barh', figsize=(10, 7), color=['#60a5fa', '#34d399', '#f87171'])
    plt.xlabel('Mean |SHAP value|', fontsize=11, fontweight='bold')
    plt.title('Feature Importance Shift Across ICU Length of Stay', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / "temporal_shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    plot_df.to_csv(XAI_DIR / "temporal_shap_importance.csv")
    print(f"  Saved: {XAI_DIR}/temporal_shap_importance.png")


def per_patient_explanations(model, test, available, explainer):
    print("\n[3/4] Generating Per-Patient SHAP Waterfall Plots...")
    X_mat = test[available].fillna(0).values.astype(np.float32)
    preds = model.predict_proba(Pool(X_mat))[:, 1]
    labels = test['SepsisLabel'].values.astype(int)

    tp_candidates = np.where((labels == 1) & (preds >= 0.05))[0]
    tn_candidates = np.where((labels == 0) & (preds < 0.03))[0]

    rng = np.random.RandomState(RANDOM_STATE)
    tp_row = rng.choice(tp_candidates) if len(tp_candidates) > 0 else 0
    tn_row = rng.choice(tn_candidates) if len(tn_candidates) > 0 else 1

    for name, row_idx in [("TP", tp_row), ("TN", tn_row)]:
        X_row = X_mat[[row_idx]]
        shap_vals_row = explainer.shap_values(Pool(X_row))
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]

        exp_single = shap.Explanation(
            values=shap_vals_row[0],
            base_values=float(base_val),
            data=X_row[0],
            feature_names=available
        )

        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(exp_single, max_display=12, show=False)
        plt.title(f"Patient Decision Waterfall ({name} Case)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(XAI_DIR / f"patient_waterfall_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {XAI_DIR}/patient_waterfall_{name}.png")

    return tp_row, tn_row, X_mat


def per_patient_lime(model, available, tp_row, tn_row, X_mat):
    print("\n[4/4] Generating Per-Patient LIME Explanations...")
    def predict_fn(X):
        p1 = model.predict_proba(Pool(X))[:, 1]
        return np.column_stack([1 - p1, p1])

    # Sample background for LIME
    n_bg = min(500, len(X_mat))
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(X_mat), size=n_bg, replace=False)
    X_bg = X_mat[bg_idx]

    lime_explainer = LimeTabularExplainer(
        X_bg, feature_names=available, class_names=['Non-Sepsis', 'Sepsis'],
        mode='classification', random_state=RANDOM_STATE
    )

    for name, row_idx in [("TP", tp_row), ("TN", tn_row)]:
        X_row = X_mat[row_idx]
        exp = lime_explainer.explain_instance(X_row, predict_fn, num_features=12)
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME Local Explanation ({name} Case)", fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(XAI_DIR / f"patient_lime_{name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {XAI_DIR}/patient_lime_{name}.png")


def main():
    print("=" * 60)
    print("Phase 9: Explainable AI (SHAP & LIME)")
    print("=" * 60)

    model, test, available = load_model_and_data()
    explainer, X_sample = global_shap(model, test, available)
    temporal_shap(model, test, available, explainer)
    tp_row, tn_row, X_mat = per_patient_explanations(model, test, available, explainer)
    per_patient_lime(model, available, tp_row, tn_row, X_mat)

    print("\n" + "=" * 60)
    print("Phase 9 Completed Successfully!")
    print(f"All SHAP and LIME figures saved to: {XAI_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()