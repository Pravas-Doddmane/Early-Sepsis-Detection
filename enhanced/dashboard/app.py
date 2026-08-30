"""
Phase 10: Interactive Clinical Sepsis Early Detection Dashboard
=============================================================
Streamlit application for real-time patient risk assessment,
ICU trajectory visualization, multi-model ensemble inference,
and dynamic Explainable AI (SHAP & LIME) interpretations.
"""
import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

# Set page config
st.set_page_config(
    page_title="SepsisGuard AI — Early Sepsis Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        border-radius: 0.75rem;
        padding: 1.2rem;
        color: #991b1b;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #f59e0b;
        border-radius: 0.75rem;
        padding: 1.2rem;
        color: #92400e;
    }
    .risk-low {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #10b981;
        border-radius: 0.75rem;
        padding: 1.2rem;
        color: #166534;
    }
    .badge {
        font-weight: 700;
        font-size: 1.25rem;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "enhanced" / "models"
EXPERIMENTS_DIR = ROOT / "enhanced" / "experiments"
XAI_DIR = EXPERIMENTS_DIR / "xai"
RAW_DATA_PATH = EXPERIMENTS_DIR / "raw_combined.parquet"


@st.cache_resource
def load_models_and_configs():
    with open(EXPERIMENTS_DIR / "selected_features.json") as f:
        sel = json.load(f)
    features = sel if isinstance(sel, list) else sel.get("final_features", [])

    with open(MODELS_DIR / "optimal_threshold.json") as f:
        threshold_info = json.load(f)

    models = {}
    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier()
        cb.load_model(str(MODELS_DIR / "catboost_model.cbm"))
        models["CatBoost"] = cb
    except Exception:
        models["CatBoost"] = None

    try:
        models["LightGBM"] = joblib.load(MODELS_DIR / "lgbm_model.pkl")
    except Exception:
        models["LightGBM"] = None

    try:
        models["XGBoost"] = joblib.load(MODELS_DIR / "xgb_model.pkl")
    except Exception:
        models["XGBoost"] = None

    try:
        models["RandomForest"] = joblib.load(MODELS_DIR / "rf_model.pkl")
    except Exception:
        models["RandomForest"] = None

    try:
        models["MetaLearner"] = joblib.load(MODELS_DIR / "meta_learner.pkl")
    except Exception:
        models["MetaLearner"] = None

    try:
        models["Calibrator"] = joblib.load(MODELS_DIR / "calibrator.pkl")
    except Exception:
        models["Calibrator"] = None

    # Load / create TreeExplainer
    shap_explainer = None
    if models.get("CatBoost") is not None:
        try:
            shap_explainer = shap.TreeExplainer(models["CatBoost"])
        except Exception:
            shap_explainer = None

    return features, threshold_info, models, shap_explainer


@st.cache_data
def load_sample_patients():
    if RAW_DATA_PATH.exists():
        df = pd.read_parquet(RAW_DATA_PATH, columns=["patient_id", "ICULOS", "SepsisLabel"])
        pos_patients = df[df["SepsisLabel"] == 1]["patient_id"].unique()[:30].tolist()
        neg_patients = df[df["SepsisLabel"] == 0]["patient_id"].unique()[:30].tolist()
        return pos_patients, neg_patients
    return [], []


@st.cache_data
def get_patient_data(patient_id):
    if RAW_DATA_PATH.exists():
        df = pd.read_parquet(RAW_DATA_PATH)
        pt_df = df[df["patient_id"] == patient_id].sort_values("ICULOS").reset_index(drop=True)
        return pt_df
    return pd.DataFrame()


@st.cache_resource
def get_lime_explainer(_cb_model, features_list):
    """Initializes a background tabular explainer for LIME."""
    # Synthetic baseline distribution for fast initialization
    X_bg = np.zeros((100, len(features_list)), dtype=np.float32)
    lime_exp = LimeTabularExplainer(
        X_bg, feature_names=features_list, class_names=['Non-Sepsis', 'Sepsis'],
        mode='classification', random_state=42
    )
    return lime_exp


def build_temporal_row(history_df, features_list):
    """Constructs temporal lag and rolling features from historical patient readings."""
    current_row = history_df.iloc[-1].copy()
    row_dict = {}

    vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
              'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
              'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose',
              'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

    for v in vitals:
        if v in current_row:
            row_dict[v] = current_row[v]

    for col in ['HR', 'MAP', 'Resp', 'Temp', 'O2Sat', 'FiO2', 'pH', 'BUN', 'Glucose', 'Potassium', 'Hgb', 'Hct', 'WBC', 'Platelets', 'SaO2', 'SBP', 'DBP']:
        if col in history_df.columns:
            series = history_df[col].ffill().bfill().fillna(0)
            n_hist = len(history_df)

            # Lags
            row_dict[f"{col}_lag1"] = series.iloc[-2] if n_hist >= 2 else series.iloc[-1]
            row_dict[f"{col}_lag3"] = series.iloc[-4] if n_hist >= 4 else series.iloc[-1]
            row_dict[f"{col}_lag6"] = series.iloc[-7] if n_hist >= 7 else series.iloc[-1]

            # Differences
            row_dict[f"{col}_diff1h"] = series.iloc[-1] - (series.iloc[-2] if n_hist >= 2 else series.iloc[-1])
            row_dict[f"{col}_diff3h"] = series.iloc[-1] - (series.iloc[-4] if n_hist >= 4 else series.iloc[-1])

            # Rolling stats (3h, 6h, 12h)
            last3 = series.iloc[-3:]
            last6 = series.iloc[-6:]
            last12 = series.iloc[-12:]

            row_dict[f"{col}_mean3h"] = last3.mean()
            row_dict[f"{col}_std3h"] = last3.std(ddof=0) if len(last3) > 1 else 0.0
            row_dict[f"{col}_mean6h"] = last6.mean()
            row_dict[f"{col}_std6h"] = last6.std(ddof=0) if len(last6) > 1 else 0.0
            row_dict[f"{col}_min6h"] = last6.min()
            row_dict[f"{col}_max6h"] = last6.max()
            row_dict[f"{col}_mean12h"] = last12.mean()
            row_dict[f"{col}_std12h"] = last12.std(ddof=0) if len(last12) > 1 else 0.0

            # Slope 3h
            if len(last3) >= 2:
                slopes = (last3.iloc[-1] - last3.iloc[0]) / (len(last3) - 1)
                row_dict[f"{col}_slope3h"] = slopes
            else:
                row_dict[f"{col}_slope3h"] = 0.0

            row_dict[f"{col}_was_missing"] = 1 if pd.isna(current_row.get(col, np.nan)) else 0

    X_vec = []
    for f in features_list:
        val = row_dict.get(f, 0.0)
        if pd.isna(val):
            val = 0.0
        X_vec.append(float(val))

    return np.array(X_vec, dtype=np.float32).reshape(1, -1)


def predict_sepsis(X_vec, models):
    preds = {}
    probs = []

    # 1. Base models
    cb = models.get("CatBoost")
    if cb is not None:
        p_cb = float(cb.predict_proba(X_vec)[0, 1])
        preds["CatBoost"] = p_cb
        probs.append(p_cb)
    else:
        probs.append(0.05)

    lgb = models.get("LightGBM")
    if lgb is not None:
        p_lgb = float(lgb.predict_proba(X_vec)[0, 1])
        preds["LightGBM"] = p_lgb
        probs.append(p_lgb)
    else:
        probs.append(0.05)

    xgb = models.get("XGBoost")
    if xgb is not None:
        p_xgb = float(xgb.predict_proba(X_vec)[0, 1])
        preds["XGBoost"] = p_xgb
        probs.append(p_xgb)
    else:
        probs.append(0.05)

    rf = models.get("RandomForest")
    if rf is not None:
        p_rf = float(rf.predict_proba(X_vec)[0, 1])
        preds["RandomForest"] = p_rf
        probs.append(p_rf)
    else:
        probs.append(0.05)

    # 2. Meta ensemble
    meta = models.get("MetaLearner")
    p_stack = None
    if meta is not None:
        try:
            if not hasattr(meta, 'multi_class'):
                meta.multi_class = 'auto'
            stack_in = np.array([[preds.get("RandomForest", 0.05),
                                  preds.get("XGBoost", 0.05),
                                  preds.get("LightGBM", 0.05),
                                  preds.get("CatBoost", 0.05)]])
            p_stack = float(meta.predict_proba(stack_in)[0, 1])
        except Exception:
            p_rf = preds.get("RandomForest", 0.05)
            p_xgb = preds.get("XGBoost", 0.05)
            p_lgb = preds.get("LightGBM", 0.05)
            p_cb = preds.get("CatBoost", 0.05)
            z = -5.069788 + (1.317140 * p_rf) + (2.208112 * p_xgb) + (2.167615 * p_lgb) + (2.260042 * p_cb)
            p_stack = float(1.0 / (1.0 + np.exp(-z)))

    if p_stack is None:
        p_stack = float(np.mean(probs))

    # 3. Calibration
    calib = models.get("Calibrator")
    if calib is not None:
        try:
            p_calibrated = float(calib.predict([p_stack])[0])
            p_calibrated = max(0.0, min(1.0, p_calibrated))
        except Exception:
            p_calibrated = p_stack
    else:
        p_calibrated = p_stack

    return p_calibrated, p_stack, preds


def render_shap_waterfall(shap_explainer, X_vec, features_list):
    """Computes and renders a live SHAP waterfall plot."""
    if shap_explainer is None:
        st.warning("SHAP explainer not available.")
        return

    shap_vals = shap_explainer.shap_values(X_vec)
    base_val = shap_explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[0]

    exp = shap.Explanation(
        values=shap_vals[0],
        base_values=float(base_val),
        data=X_vec[0],
        feature_names=features_list
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title("Patient-Specific SHAP Decision Waterfall (Positive = Increases Risk)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_lime_explanation(models, X_vec, features_list):
    """Computes and renders a live LIME bar chart."""
    cb = models.get("CatBoost")
    if cb is None:
        st.warning("Model not available for LIME.")
        return

    def predict_fn(X):
        p1 = cb.predict_proba(X)[:, 1]
        return np.column_stack([1 - p1, p1])

    lime_explainer = get_lime_explainer(cb, features_list)
    exp = lime_explainer.explain_instance(X_vec[0], predict_fn, num_features=10)
    fig = exp.as_pyplot_figure()
    plt.title("LIME Local Feature Attribution (Features pushing towards / against sepsis)", fontsize=11, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/fluency/96/medical-heart.png", width=64)
    st.sidebar.title("SepsisGuard AI")
    st.sidebar.caption("Enhanced Early Sepsis Detection System (PhysioNet 2019)")
    
    features_list, threshold_info, models, shap_explainer = load_models_and_configs()
    opt_threshold = threshold_info.get("optimal_threshold", 0.0262)

    app_mode = st.sidebar.radio(
        "Navigation",
        [
            "🏥 Live Patient ICU Monitor",
            "🎛️ Manual Clinical Entry",
            "🔬 Explainable AI (SHAP & LIME Global)",
            "📊 Model Performance & Benchmarks",
            "ℹ️ About the Project"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Operational Threshold**: `{opt_threshold:.4f}`")
    st.sidebar.markdown("**Target Sensitivity**: `≥ 65.0%`")
    st.sidebar.markdown("**Calibration**: `Isotonic Regression`")
    st.sidebar.markdown("**Meta-Learner**: `Logistic Stacking Ensemble`")

    # ----------------------------------------------------
    # TAB 1: LIVE PATIENT ICU MONITOR
    # ----------------------------------------------------
    if app_mode == "🏥 Live Patient ICU Monitor":
        st.markdown('<div class="main-header">🏥 ICU Patient Trajectory & Sepsis Risk Monitor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Continuous hourly physiological surveillance with multi-model inference and live explainability.</div>', unsafe_allow_html=True)

        pos_pts, neg_pts = load_sample_patients()
        
        col1, col2, col3 = st.columns([1.5, 1.5, 1])
        with col1:
            category_sel = st.selectbox("Cohort Type", ["Sepsis Case (Onset within stay)", "Non-Sepsis Case"])
            patient_list = pos_pts if category_sel == "Sepsis Case (Onset within stay)" else neg_pts
            if not patient_list:
                patient_list = ["p000009", "p000033", "p000045"] if "Sepsis" in category_sel else ["p000001", "p000002", "p000003"]
            patient_id = st.selectbox("Select Patient ID", patient_list)

        pt_df = get_patient_data(patient_id)
        if pt_df.empty:
            st.warning(f"Patient {patient_id} data not found. Loading sample fallback.")
            pt_df = pd.DataFrame({
                "ICULOS": list(range(1, 25)),
                "HR": [75 + i*1.2 for i in range(24)],
                "MAP": [90 - i*0.8 for i in range(24)],
                "Resp": [16 + i*0.5 for i in range(24)],
                "Temp": [37.0 + (i*0.08 if i > 12 else 0) for i in range(24)],
                "O2Sat": [98 - i*0.2 for i in range(24)],
                "SepsisLabel": [1 if i >= 18 else 0 for i in range(24)],
                "Age": 68, "Gender": 1
            })

        max_hour = int(pt_df["ICULOS"].max())
        min_hour = int(pt_df["ICULOS"].min())

        with col2:
            current_hour = st.slider("ICU Hour (t)", min_value=min_hour, max_value=max_hour, value=min(max_hour, max(min_hour, 12)))

        history_df = pt_df[pt_df["ICULOS"] <= current_hour].reset_index(drop=True)
        current_vitals = history_df.iloc[-1]
        is_actual_septic = int(history_df["SepsisLabel"].max()) if "SepsisLabel" in history_df.columns else 0

        X_vec = build_temporal_row(history_df, features_list)
        p_calibrated, p_stack, base_preds = predict_sepsis(X_vec, models)

        # Risk Banner
        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

        with res_col1:
            if p_calibrated >= 0.08:
                st.markdown(f"""
                <div class="risk-high">
                    <span class="badge" style="background-color: #ef4444; color: white;">🔴 HIGH SEPSIS RISK</span>
                    <h2 style="margin: 0.5rem 0 0 0; color: #991b1b;">Predicted Risk: {p_calibrated*100:.2f}%</h2>
                    <p style="margin: 0.3rem 0 0 0;">Exceeds critical alert threshold. Imminent sepsis onset likely within next 6 hours.</p>
                </div>
                """, unsafe_allow_html=True)
            elif p_calibrated >= opt_threshold:
                st.markdown(f"""
                <div class="risk-moderate">
                    <span class="badge" style="background-color: #f59e0b; color: white;">🟡 ELEVATED RISK ALERT</span>
                    <h2 style="margin: 0.5rem 0 0 0; color: #92400e;">Predicted Risk: {p_calibrated*100:.2f}%</h2>
                    <p style="margin: 0.3rem 0 0 0;">Above operational threshold ({opt_threshold*100:.2f}%). Heightened clinical monitoring required.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <span class="badge" style="background-color: #10b981; color: white;">🟢 LOW SEPSIS RISK</span>
                    <h2 style="margin: 0.5rem 0 0 0; color: #166534;">Predicted Risk: {p_calibrated*100:.2f}%</h2>
                    <p style="margin: 0.3rem 0 0 0;">Within normal physiological baseline below clinical alarm threshold ({opt_threshold*100:.2f}%).</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.metric("Raw Stacking Prob", f"{p_stack*100:.2f}%")
            st.metric("Operating Threshold", f"{opt_threshold*100:.2f}%")

        with res_col3:
            st.metric("Actual Sepsis Ground Truth", "SEPSIS POSITIVE" if is_actual_septic == 1 else "NO SEPSIS")
            st.metric("ICU Length of Stay", f"{current_hour} Hours")

        # Vital Signs Grid
        st.markdown("### 🫀 Physiological Parameters at Hour $t$")
        v1, v2, v3, v4, v5 = st.columns(5)
        hr_val = current_vitals.get("HR", np.nan)
        v1.metric("Heart Rate", f"{hr_val:.1f} bpm" if pd.notna(hr_val) else "N/A", delta="Tachycardia (>90)" if pd.notna(hr_val) and hr_val > 90 else None, delta_color="inverse")
        map_val = current_vitals.get("MAP", np.nan)
        v2.metric("Mean Art Pressure", f"{map_val:.1f} mmHg" if pd.notna(map_val) else "N/A", delta="Hypotension (<65)" if pd.notna(map_val) and map_val < 65 else None, delta_color="inverse")
        resp_val = current_vitals.get("Resp", np.nan)
        v3.metric("Respiration Rate", f"{resp_val:.1f} bpm" if pd.notna(resp_val) else "N/A", delta="Tachypnea (>20)" if pd.notna(resp_val) and resp_val > 20 else None, delta_color="inverse")
        temp_val = current_vitals.get("Temp", np.nan)
        v4.metric("Temperature", f"{temp_val:.1f} °C" if pd.notna(temp_val) else "N/A", delta="Fever (>38.3)" if pd.notna(temp_val) and temp_val > 38.3 else None, delta_color="inverse")
        o2_val = current_vitals.get("O2Sat", np.nan)
        v5.metric("O2 Saturation", f"{o2_val:.1f} %" if pd.notna(o2_val) else "N/A", delta="Hypoxia (<92)" if pd.notna(o2_val) and o2_val < 92 else None, delta_color="inverse")

        # Visualizations & Explainability Tabs
        st.markdown("### 📈 Patient Visualizations & Explainable AI (SHAP & LIME)")
        c_tabs = st.tabs(["📈 Vitals Timeline", "🔬 SHAP Waterfall Explanation", "🍋 LIME Local Attribution", "🤖 Base Learners Breakdown"])

        with c_tabs[0]:
            fig = go.Figure()
            if "HR" in pt_df.columns:
                fig.add_trace(go.Scatter(x=history_df["ICULOS"], y=history_df["HR"], mode='lines+markers', name='Heart Rate (bpm)', line=dict(color='#ef4444', width=2.5)))
            if "MAP" in pt_df.columns:
                fig.add_trace(go.Scatter(x=history_df["ICULOS"], y=history_df["MAP"], mode='lines+markers', name='MAP (mmHg)', line=dict(color='#3b82f6', width=2.5)))
            if "Resp" in pt_df.columns:
                fig.add_trace(go.Scatter(x=history_df["ICULOS"], y=history_df["Resp"], mode='lines+markers', name='Resp Rate (bpm)', line=dict(color='#10b981', width=2.5)))
            if "Temp" in pt_df.columns:
                fig.add_trace(go.Scatter(x=history_df["ICULOS"], y=history_df["Temp"], mode='lines+markers', name='Temp (°C)', line=dict(color='#f59e0b', width=2.5), yaxis='y2'))

            fig.update_layout(
                title=f"Patient {patient_id} — Multi-Signal Trajectory up to Hour {current_hour}",
                xaxis_title="ICU Hour (ICULOS)",
                yaxis_title="Vitals Value",
                yaxis2=dict(title="Temperature (°C)", overlaying='y', side='right'),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=420
            )
            st.plotly_chart(fig)

        with c_tabs[1]:
            st.markdown("#### 🔬 Live SHAP Decision Waterfall for Patient")
            st.write("Shows how each specific physiological feature pushes this patient's prediction above (red) or below (blue) baseline.")
            render_shap_waterfall(shap_explainer, X_vec, features_list)

        with c_tabs[2]:
            st.markdown("#### 🍋 Live LIME Local Feature Attribution")
            st.write("Highlights decision boundaries and rule explanations explaining this patient's risk state.")
            render_lime_explanation(models, X_vec, features_list)

        with c_tabs[3]:
            st.markdown("#### 🤖 Predictions from Individual Base Learners")
            base_df = pd.DataFrame([
                {"Model": m, "Predicted Risk": p * 100}
                for m, p in base_preds.items()
            ])
            fig_bar = px.bar(
                base_df, x="Model", y="Predicted Risk",
                color="Model", text_auto=".2f",
                title="Individual Base Model Probabilities vs Stacking Ensemble",
                color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
            )
            fig_bar.add_hline(y=opt_threshold * 100, line_dash="dash", line_color="red", annotation_text=f"Threshold ({opt_threshold*100:.2f}%)")
            fig_bar.update_layout(yaxis_title="Risk Probability (%)", height=380)
            st.plotly_chart(fig_bar)

    # ----------------------------------------------------
    # TAB 2: MANUAL CLINICAL ENTRY
    # ----------------------------------------------------
    elif app_mode == "🎛️ Manual Clinical Entry":
        st.markdown('<div class="main-header">🎛️ Manual Clinical Parameter Risk Calculator</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Input patient vitals and lab findings to obtain instant calibrated predictions with live SHAP and LIME explanations.</div>', unsafe_allow_html=True)

        with st.form("manual_entry_form"):
            st.markdown("##### 1. Vital Signs")
            c1, c2, c3, c4 = st.columns(4)
            hr = c1.number_input("Heart Rate (bpm)", min_value=30.0, max_value=220.0, value=88.0, step=1.0)
            map_val = c2.number_input("Mean Art Pressure (mmHg)", min_value=20.0, max_value=200.0, value=75.0, step=1.0)
            resp = c3.number_input("Respiration Rate (bpm)", min_value=5.0, max_value=60.0, value=18.0, step=1.0)
            temp = c4.number_input("Body Temperature (°C)", min_value=30.0, max_value=43.0, value=37.2, step=0.1)

            st.markdown("##### 2. Oxygenation & Blood Pressure")
            c5, c6, c7, c8 = st.columns(4)
            o2sat = c5.number_input("O2 Saturation (%)", min_value=50.0, max_value=100.0, value=97.0, step=1.0)
            sbp = c6.number_input("Systolic BP (mmHg)", min_value=40.0, max_value=250.0, value=118.0, step=1.0)
            dbp = c7.number_input("Diastolic BP (mmHg)", min_value=20.0, max_value=150.0, value=65.0, step=1.0)
            fio2 = c8.number_input("FiO2 (%)", min_value=21.0, max_value=100.0, value=21.0, step=1.0)

            st.markdown("##### 3. Key Laboratory Biomarkers & Static Info")
            c9, c10, c11, c12 = st.columns(4)
            wbc = c9.number_input("White Blood Cells (x10^9/L)", min_value=0.5, max_value=100.0, value=9.5, step=0.5)
            platelets = c10.number_input("Platelets (x10^9/L)", min_value=10.0, max_value=1000.0, value=220.0, step=10.0)
            bun = c11.number_input("BUN (mg/dL)", min_value=1.0, max_value=200.0, value=18.0, step=1.0)
            glucose = c12.number_input("Glucose (mg/dL)", min_value=20.0, max_value=800.0, value=110.0, step=5.0)

            c13, c14, c15 = st.columns(3)
            age = c13.number_input("Patient Age (years)", min_value=18, max_value=110, value=62, step=1)
            gender = c14.selectbox("Gender", ["Male", "Female"])
            iculos = c15.number_input("ICU Length of Stay (hours)", min_value=1, max_value=500, value=14, step=1)

            submitted = st.form_submit_button("🩺 Calculate Sepsis Risk & Explain Decision")

        if submitted:
            sim_df = pd.DataFrame([{
                "HR": hr, "MAP": map_val, "Resp": resp, "Temp": temp,
                "O2Sat": o2sat, "SBP": sbp, "DBP": dbp, "FiO2": fio2,
                "WBC": wbc, "Platelets": platelets, "BUN": bun, "Glucose": glucose,
                "Age": age, "Gender": 1 if gender == "Male" else 0, "ICULOS": iculos,
                "HospAdmTime": -12.0
            }])

            X_manual = build_temporal_row(sim_df, features_list)
            p_calibrated, p_stack, base_preds = predict_sepsis(X_manual, models)

            st.markdown("---")
            res_c1, res_c2 = st.columns([1.5, 1])
            with res_c1:
                if p_calibrated >= 0.08:
                    st.markdown(f"""
                    <div class="risk-high">
                        <span class="badge" style="background-color: #ef4444; color: white;">🔴 HIGH SEPSIS RISK</span>
                        <h2 style="margin: 0.5rem 0 0 0; color: #991b1b;">Predicted Probability: {p_calibrated*100:.2f}%</h2>
                        <p style="margin: 0.3rem 0 0 0;">Critical threshold exceeded. Clinical sepsis evaluation recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif p_calibrated >= opt_threshold:
                    st.markdown(f"""
                    <div class="risk-moderate">
                        <span class="badge" style="background-color: #f59e0b; color: white;">🟡 ELEVATED RISK ALERT</span>
                        <h2 style="margin: 0.5rem 0 0 0; color: #92400e;">Predicted Probability: {p_calibrated*100:.2f}%</h2>
                        <p style="margin: 0.3rem 0 0 0;">Exceeds optimal alert threshold ({opt_threshold*100:.2f}%). Close observation indicated.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <span class="badge" style="background-color: #10b981; color: white;">🟢 LOW SEPSIS RISK</span>
                        <h2 style="margin: 0.5rem 0 0 0; color: #166534;">Predicted Probability: {p_calibrated*100:.2f}%</h2>
                        <p style="margin: 0.3rem 0 0 0;">Patient vitals and labs are consistent with non-septic status.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with res_c2:
                st.markdown("#### Model Component Breakdown")
                for name, prob in base_preds.items():
                    st.write(f"- **{name}**: `{prob*100:.2f}%`")
                st.write(f"- **Ensemble (Raw)**: `{p_stack*100:.2f}%`")
                st.write(f"- **Calibrated Final**: `{p_calibrated*100:.2f}%`")

            # Dynamic SHAP & LIME for manual patient
            st.markdown("### 🔬 Explainability for Entered Parameters")
            m_tabs = st.tabs(["🔬 SHAP Waterfall Explanation", "🍋 LIME Local Attribution"])
            with m_tabs[0]:
                render_shap_waterfall(shap_explainer, X_manual, features_list)
            with m_tabs[1]:
                render_lime_explanation(models, X_manual, features_list)

    # ----------------------------------------------------
    # TAB 3: EXPLAINABLE AI (GLOBAL EXPLORER)
    # ----------------------------------------------------
    elif app_mode == "🔬 Explainable AI (SHAP & LIME Global)":
        st.markdown('<div class="main-header">🔬 Explainable AI (SHAP & LIME Global Interpretability)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive insights into model decision-making, physiological dynamics, and feature attributions.</div>', unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 🐝 Global SHAP Beeswarm Summary")
            st.write("Ranks feature importance and shows the direction of effect (high feature value = red, low = blue).")
            beeswarm_img = XAI_DIR / "global_shap_beeswarm.png"
            if beeswarm_img.exists():
                st.image(str(beeswarm_img))
            else:
                st.info("Run `python enhanced/xai/explain.py` to generate beeswarm plot.")

        with g2:
            st.markdown("#### ⏳ Temporal SHAP Stay Dynamics")
            st.write("Shows how feature importance shifts across Early (hours 1–6), Mid (7–24), and Late (25+) ICU stay.")
            temporal_img = XAI_DIR / "temporal_shap_importance.png"
            if temporal_img.exists():
                st.image(str(temporal_img))
            else:
                st.info("Run `python enhanced/xai/explain.py` to generate temporal SHAP plot.")

        st.markdown("---")
        st.markdown("#### 📋 Top Feature Importance Ranking")
        csv_file = XAI_DIR / "global_shap_importance.csv"
        if csv_file.exists():
            df_imp = pd.read_csv(csv_file)
            st.dataframe(df_imp.head(20))
        else:
            st.write("Feature importance table available after running Phase 9.")

        st.markdown("---")
        st.markdown("#### 🔍 Sample Patient Case Studies (True Positive vs True Negative)")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("##### True Positive Case (SHAP & LIME)")
            tp_waterfall = XAI_DIR / "patient_waterfall_TP.png"
            if tp_waterfall.exists():
                st.image(str(tp_waterfall))
            tp_lime = XAI_DIR / "patient_lime_TP.png"
            if tp_lime.exists():
                st.image(str(tp_lime))

        with p2:
            st.markdown("##### True Negative Case (SHAP & LIME)")
            tn_waterfall = XAI_DIR / "patient_waterfall_TN.png"
            if tn_waterfall.exists():
                st.image(str(tn_waterfall))
            tn_lime = XAI_DIR / "patient_lime_TN.png"
            if tn_lime.exists():
                st.image(str(tn_lime))

    # ----------------------------------------------------
    # TAB 4: MODEL PERFORMANCE & BENCHMARKS
    # ----------------------------------------------------
    elif app_mode == "📊 Model Performance & Benchmarks":
        st.markdown('<div class="main-header">📊 Model Performance & Benchmark Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive comparative metrics on PhysioNet 2019 test set.</div>', unsafe_allow_html=True)

        results_file = EXPERIMENTS_DIR / "results_table.csv"
        if results_file.exists():
            df_results = pd.read_csv(results_file)
            st.dataframe(df_results)

        st.markdown("---")
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("#### 🎯 ROC-AUC & PR-AUC Comparison")
            metrics_img = EXPERIMENTS_DIR / "figures" / "model_comparison_metrics.png"
            if metrics_img.exists():
                st.image(str(metrics_img))

        with b2:
            st.markdown("#### 🩺 Clinical Operating Performance @ Optimal Threshold")
            tradeoff_img = EXPERIMENTS_DIR / "figures" / "clinical_tradeoff.png"
            if tradeoff_img.exists():
                st.image(str(tradeoff_img))

        st.markdown("---")
        st.markdown("#### 📉 Probability Calibration Curve")
        calib_img = EXPERIMENTS_DIR / "calibration_curves.png"
        if calib_img.exists():
            st.image(str(calib_img))

    # ----------------------------------------------------
    # TAB 5: ABOUT THE PROJECT
    # ----------------------------------------------------
    elif app_mode == "ℹ️ About the Project":
        st.markdown('<div class="main-header">ℹ️ Project Overview & Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        ### Project Summary
        This application is the deployment interface for an **Enhanced Early Sepsis Prediction System** developed using the **PhysioNet/Computing in Cardiology Challenge 2019** dataset.

        ### Key Technical Milestones
        1. **Zero Data Leakage Pipeline**: Strict patient-stratified partitioning across Train, Validation, and Test sets.
        2. **Causal Temporal Dynamics**: Computing historical lags ($t-1, t-3, t-6$), rolling metrics (mean, std, min, max over 3h, 6h, 12h), and linear trajectory slopes.
        3. **Hybrid Feature Selection**: Boruta + Mutual Information filtering down to the 150 most predictive temporal features.
        4. **Heterogeneous Stacking Ensemble**: Combining Random Forest, XGBoost, LightGBM, and CatBoost via an L2 meta-learner.
        5. **Isotonic Calibration & Clinical Utility**: Calibrating raw outputs and tuning the operational decision threshold to guarantee $>65\%$ sensitivity for early alert generation while controlling ICU alarm fatigue.
        6. **Explainability Suite**: Live SHAP & LIME explanations embedded in both continuous ICU surveillance and manual calculator modes.
        """)


if __name__ == "__main__":
    main()
