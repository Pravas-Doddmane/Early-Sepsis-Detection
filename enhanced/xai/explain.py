"""
Phase 9 — Explainable AI
=========================
Explains the best-performing single base model (CatBoost, test PR-AUC
0.0709 — highest of the four base models) rather than the stacked
ensemble, for two reasons:
  1. CatBoost has native, fast, exact TreeExplainer SHAP support.
  2. Explaining a single interpretable tree model directly is the same
     choice the base paper (Santos et al.) made — keeps our SHAP results
     directly comparable to theirs.
Explaining the meta-learner would require combining 4 separate SHAP
decompositions through a linear layer, which is possible but adds
complexity without adding clarity for a first XAI pass.

Outputs (saved to enhanced/experiments/xai/):
  1. global_shap_beeswarm.png   — overall feature importance (compare
     directly against the base paper's Fig. 4)
  2. temporal_shap_importance.png — NOVEL: how feature importance shifts
     across ICU-LOS (hour t-12 vs t-6 vs t-1 before current point) — the
     base paper's static model cannot produce this, it's our strongest
     interpretability differentiator
  3. patient_waterfall_TP.png / patient_waterfall_TN.png — per-patient
     SHAP waterfall for one true-positive and one true-negative example
  4. patient_lime_TP.png / patient_lime_TN.png — same two patients,
     explained via LIME instead, for a second interpretability method
"""
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier, Pool

MODELS_DIR = Path("enhanced/models")
DATA_DIR = Path("enhanced/data/processed")
EXPERIMENTS = Path("enhanced/experiments")
XAI_DIR = EXPERIMENTS / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

N_GLOBAL_SAMPLE = 500      # patients, not rows — matches base paper's approach
RANDOM_STATE = 42


def load_model_and_data():
    with open(EXPERIMENTS / "selected_features.json") as f:
        sel = json.load(f)
    features = sel if isinstance(sel, list) else sel.get("final_features", [])

    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / "catboost_model.cbm"))

    test = pd.read_parquet(DATA_DIR / "test_temporal.parquet")
    available = [f for f in features if f in test.columns]

    return model, test, available


def get_patient_sample(test, available, n_patients=N_GLOBAL_SAMPLE):
    """Sample whole patients (not random rows) for the global SHAP set —
    keeps the sample clinically coherent and avoids over-weighting
    long-stay patients relative to short-stay ones."""
    rng = np.random.RandomState(RANDOM_STATE)
    all_patients = test['patient_id'].unique()
    sampled_patients = rng.choice(all_patients, size=min(n_patients, len(all_patients)),
                                    replace=False)
    sample = test[test['patient_id'].isin(sampled_patients)].copy()
    X_sample = sample[available].fillna(0).values.astype(np.float32)
    return sample, X_sample


def global_shap(model, sample, X_sample, available):
    print("\n[1/4] Computing global SHAP values (beeswarm)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X_sample))

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=available,
                       show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(XAI_DIR / "global_shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {XAI_DIR}/global_shap_beeswarm.png")

    # Save the raw importance ranking as a table too — useful for the
    # paper's text/appendix, not just the figure
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': available,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    importance_df.to_csv(XAI_DIR / "global_shap_importance.csv", index=False)
    print(f"  Saved: {XAI_DIR}/global_shap_importance.csv")

    return explainer, shap_values


def temporal_shap(model, test, available, explainer, top_n_features=10):
    """
    NOVEL vs base paper: bins rows by ICULOS into early/mid/late ICU
    stay and computes mean |SHAP| separately for each bin, showing
    whether feature importance shifts as patients progress through
    their stay — something a static, stay-level-aggregated model
    (like the base paper's) cannot show, since it only ever sees one
    row per patient.
    """
    print("\n[2/4] Computing temporal SHAP importance (novel vs. base paper)...")

    rng = np.random.RandomState(RANDOM_STATE)
    bins = {
        'Early (hour 1-6)': (1, 6),
        'Mid (hour 7-24)': (7, 24),
        'Late (hour 25+)': (25, 999),
    }

    bin_importances = {}
    for bin_name, (lo, hi) in bins.items():
        bin_rows = test[(test['ICULOS'] >= lo) & (test['ICULOS'] <= hi)]
        if len(bin_rows) == 0:
            continue
        n_sample = min(2000, len(bin_rows))  # cap for speed
        sample_idx = rng.choice(bin_rows.index, size=n_sample, replace=False)
        X_bin = bin_rows.loc[sample_idx, available].fillna(0).values.astype(np.float32)
        shap_bin = explainer.shap_values(Pool(X_bin))
        bin_importances[bin_name] = np.abs(shap_bin).mean(axis=0)
        print(f"  {bin_name}: {n_sample} rows sampled")

    importance_df = pd.DataFrame(bin_importances, index=available)
    top_features = importance_df.mean(axis=1).nlargest(top_n_features).index
    plot_df = importance_df.loc[top_features]

    plt.figure(figsize=(10, 8))
    plot_df.plot(kind='barh', figsize=(10, 8))
    plt.xlabel('Mean |SHAP value|')
    plt.title('Feature Importance Shift Across ICU Stay (Early vs Mid vs Late)')
    plt.tight_layout()
    plt.savefig(XAI_DIR / "temporal_shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    plot_df.to_csv(XAI_DIR / "temporal_shap_importance.csv")
    print(f"  Saved: {XAI_DIR}/temporal_shap_importance.png")
    print(f"  Saved: {XAI_DIR}/temporal_shap_importance.csv")


def per_patient_explanations(model, test, available, explainer):
    """Pick one true-positive and one true-negative example, generate a
    SHAP waterfall for each using the modern Explanation-object API."""
    print("\n[3/4] Generating per-patient SHAP waterfall plots...")

    calibrated_preds = np.load(MODELS_DIR / "calibrated_test_preds.npy")
    threshold = json.load(open(MODELS_DIR / "optimal_threshold.json"))["optimal_threshold"]
    y_true = test["SepsisLabel"].values.astype(int)
    y_pred = (calibrated_preds >= threshold).astype(int)

    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]

    if len(tp_idx) == 0 or len(tn_idx) == 0:
        print("  [WARNING] Could not find both a TP and TN example — skipping.")
        return None, None

    rng = np.random.RandomState(RANDOM_STATE)
    tp_row = rng.choice(tp_idx)
    tn_row = rng.choice(tn_idx)

    for name, row_idx in [("TP", tp_row), ("TN", tn_row)]:
        X_row = test.iloc[[row_idx]][available].fillna(0).values.astype(np.float32)
        # Get SHAP values directly as array (not Explanation object)
        shap_vals = explainer.shap_values(Pool(X_row))

        # shap_vals shape: (1, n_features) for binary classification (positive class)
        # or (1, n_features, 2) for both classes
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]  # take positive class

        # Expected value (base value) for this prediction
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]

        plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals[0],
                base_values=expected_val,
                data=X_row[0],
                feature_names=available
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig(XAI_DIR / f"patient_waterfall_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {XAI_DIR}/patient_waterfall_{name}.png")

    return tp_row, tn_row


def per_patient_lime(model, test, available, tp_row, tn_row):
    print("\n[4/4] Generating LIME explanations for the same patients...")
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("  [SKIPPED] lime not installed. Run: pip install lime --break-system-packages")
        return

    X_all = test[available].fillna(0).values.astype(np.float32)

    def predict_fn(X):
        p1 = model.predict_proba(Pool(X))[:, 1]
        return np.column_stack([1 - p1, p1])

    lime_explainer = LimeTabularExplainer(
        X_all, feature_names=available, class_names=['no_sepsis', 'sepsis'],
        mode='classification', random_state=RANDOM_STATE
    )

    for name, row_idx in [("TP", tp_row), ("TN", tn_row)]:
        X_row = test.iloc[row_idx][available].fillna(0).values.astype(np.float32)
        exp = lime_explainer.explain_instance(X_row, predict_fn, num_features=15)
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(XAI_DIR / f"patient_lime_{name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {XAI_DIR}/patient_lime_{name}.png")


def main():
    print("=" * 60)
    print("Phase 9: Explainable AI")
    print("=" * 60)

    model, test, available = load_model_and_data()
    sample, X_sample = get_patient_sample(test, available)

    explainer, _ = global_shap(model, sample, X_sample, available)
    temporal_shap(model, test, available, explainer)
    tp_row, tn_row = per_patient_explanations(model, test, available, explainer)
    if tp_row is not None:
        per_patient_lime(model, test, available, tp_row, tn_row)

    print("\n" + "=" * 60)
    print("Phase 9 Complete!")
    print("=" * 60)
    print(f"All outputs saved to: {XAI_DIR}/")
    print("\nNext: python enhanced/dashboard/app.py  (or skip to paper writing)")


if __name__ == "__main__":
    main()