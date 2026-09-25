"""Service for loading XAI artifacts (SHAP, LIME)."""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import joblib
import shap
from catboost import CatBoostClassifier, Pool

ENHANCED_DIR = Path(os.environ.get("ENHANCED_DIR", Path(__file__).resolve().parents[4]))
XAI_DIR = ENHANCED_DIR / "experiments" / "xai"
MODELS_DIR = ENHANCED_DIR / "models"
DATA_DIR = ENHANCED_DIR / "data" / "processed"


class XAIArtifacts:
    """Loads and provides access to XAI artifacts."""

    def __init__(self):
        self.explainer = None
        self.model = None
        self.feature_names: List[str] = []
        self.test_data = None
        self.calibrated_preds = None
        self.threshold = 0.026202020202020202
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all XAI artifacts."""
        # Load feature names
        try:
            with open(XAI_DIR.parent / "selected_features.json") as f:
                sel = json.load(f)
                self.feature_names = sel if isinstance(sel, list) else sel.get("final_features", [])
        except Exception:
            pass

        # Load CatBoost model for SHAP
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(str(MODELS_DIR / "catboost_model.cbm"))
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"Warning: Could not load SHAP explainer: {e}")

        # Load test data for reference
        try:
            self.test_data = pd.read_parquet(DATA_DIR / "test_temporal.parquet")
        except Exception as e:
            print(f"Warning: Could not load test data: {e}")

        # Load calibrated predictions
        try:
            self.calibrated_preds = np.load(MODELS_DIR / "calibrated_test_preds.npy")
        except Exception:
            pass

        # Load threshold
        try:
            with open(MODELS_DIR / "optimal_threshold.json") as f:
                data = json.load(f)
                self.threshold = data.get("optimal_threshold", 0.0262)
        except Exception:
            pass

    def get_global_shap_importance(self) -> List[Dict[str, Any]]:
        """Get global SHAP importance from CSV."""
        try:
            df = pd.read_csv(XAI_DIR / "global_shap_importance.csv")
            return df.to_dict("records")
        except Exception:
            return []

    def get_temporal_shap_importance(self) -> List[Dict[str, Any]]:
        """Get temporal SHAP importance from CSV."""
        try:
            df = pd.read_csv(XAI_DIR / "temporal_shap_importance.csv")
            # Handle unnamed index column (feature name is in first column)
            if 'Unnamed: 0' in df.columns:
                df = df.rename(columns={'Unnamed: 0': 'feature'})
            elif df.columns[0] != 'feature':
                df = df.rename(columns={df.columns[0]: 'feature'})
            return df.to_dict("records")
        except Exception:
            return []

    def get_patient_shap(self, patient_id: str, iculos: int) -> Optional[List[Dict[str, Any]]]:
        """Compute SHAP values for a specific patient-hour."""
        if self.explainer is None or self.test_data is None:
            return None

        # Find the row
        row = self.test_data[
            (self.test_data["patient_id"] == patient_id) &
            (self.test_data["ICULOS"] == iculos)
        ]

        if len(row) == 0:
            return None

        X_row = row[self.feature_names].fillna(0).values.astype(np.float32)

        try:
            shap_vals = self.explainer.shap_values(Pool(X_row))
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]  # positive class

            # Get top 15 features by absolute SHAP
            abs_shap = np.abs(shap_vals[0])
            top_indices = np.argsort(abs_shap)[-15:][::-1]

            result = []
            for idx in top_indices:
                result.append({
                    "feature": self.feature_names[idx],
                    "shap_value": float(shap_vals[0, idx]),
                    "feature_value": float(X_row[0, idx])
                })
            return result
        except Exception as e:
            print(f"SHAP computation error: {e}")
            return None

    def get_patient_lime(self, patient_id: str, iculos: int) -> Optional[List[Dict[str, Any]]]:
        """Compute LIME explanation for a specific patient-hour."""
        if self.test_data is None or self.model is None:
            return None

        row = self.test_data[
            (self.test_data["patient_id"] == patient_id) &
            (self.test_data["ICULOS"] == iculos)
        ]

        if len(row) == 0:
            return None

        X_row = row[self.feature_names].fillna(0).values.astype(np.float32)[0]
        X_all = self.test_data[self.feature_names].fillna(0).values.astype(np.float32)

        try:
            from lime.lime_tabular import LimeTabularExplainer

            def predict_fn(X):
                p1 = self.model.predict_proba(Pool(X))[:, 1]
                return np.column_stack([1 - p1, p1])

            lime_explainer = LimeTabularExplainer(
                X_all, feature_names=self.feature_names,
                class_names=['no_sepsis', 'sepsis'],
                mode='classification', random_state=42
            )

            exp = lime_explainer.explain_instance(X_row, predict_fn, num_features=15)
            return [{"feature": f, "weight": w} for f, w in exp.as_list()]
        except Exception as e:
            print(f"LIME computation error: {e}")
            return None

    def get_patient_info(self, patient_id: str, iculos: int) -> Optional[Dict]:
        """Get patient prediction info."""
        if self.test_data is None or self.calibrated_preds is None:
            return None

        row = self.test_data[
            (self.test_data["patient_id"] == patient_id) &
            (self.test_data["ICULOS"] == iculos)
        ]

        if len(row) == 0:
            return None

        idx = row.index[0]
        prob = float(self.calibrated_preds[idx])
        true_label = int(row["SepsisLabel"].iloc[0])
        pred = int(prob >= self.threshold)

        return {
            "patient_id": patient_id,
            "iculos": iculos,
            "prob_sepsis": prob,
            "threshold": self.threshold,
            "prediction": pred,
            "true_label": true_label
        }


# Global instance
_xai_artifacts: Optional[XAIArtifacts] = None


def get_xai_artifacts() -> XAIArtifacts:
    global _xai_artifacts
    if _xai_artifacts is None:
        _xai_artifacts = XAIArtifacts()
    return _xai_artifacts