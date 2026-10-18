"""Service for loading and running ML models."""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
from catboost import CatBoostClassifier, Pool

ENHANCED_DIR = Path(os.environ.get("ENHANCED_DIR", Path(__file__).resolve().parents[4]))
MODELS_DIR = ENHANCED_DIR / "models"
EXPERIMENTS_DIR = ENHANCED_DIR / "experiments"
PROCESSED_DIR = ENHANCED_DIR / "data" / "processed"

BASE_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2", "pH",
    "PaCO2", "SaO2", "BUN", "Calcium", "Glucose", "Potassium", "Hct",
    "Hgb", "WBC", "Platelets",
]
STATIC_FEATURES = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]


class ModelLoader:
    """Loads and manages all trained models for inference."""

    def __init__(self):
        self.models: Dict[str, any] = {}
        self.meta_learner = None
        self.imputer = None
        self.iqr_capper: Dict[str, Dict[str, float]] = {}
        self.scalers_standard: Dict[str, Any] = {}
        self.scalers_robust: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.scaler_choice: Dict[str, str] = {}
        self.feature_names: List[str] = []
        self.calibrator = None
        self.calibration_type: Optional[str] = None
        self.optimal_threshold = 0.026202020202020202
        self._load_models()
        self._load_preprocessing()
        self._load_calibrator()
        self._load_threshold()

    def _load_models(self):
        """Load all trained models."""
        # CatBoost
        try:
            cb = CatBoostClassifier()
            cb.load_model(str(MODELS_DIR / "catboost_model.cbm"))
            self.models["catboost"] = cb
        except Exception as e:
            print(f"Warning: Could not load CatBoost: {e}")

        # XGBoost
        try:
            self.models["xgb"] = joblib.load(MODELS_DIR / "xgb_model.pkl")
        except Exception as e:
            print(f"Warning: Could not load XGBoost: {e}")

        # LightGBM
        try:
            self.models["lgbm"] = joblib.load(MODELS_DIR / "lgbm_model.pkl")
        except Exception as e:
            print(f"Warning: Could not load LightGBM: {e}")

        # Random Forest
        try:
            self.models["rf"] = joblib.load(MODELS_DIR / "rf_model.pkl")
        except Exception as e:
            print(f"Warning: Could not load Random Forest: {e}")

        # Meta-learner (stacking)
        try:
            self.meta_learner = joblib.load(MODELS_DIR / "meta_learner.pkl")
        except Exception as e:
            print(f"Warning: Could not load meta-learner: {e}")

    def _load_preprocessing(self):
        """Load preprocessing artifacts."""
        try:
            transformers_dir = MODELS_DIR / "transformers"
            self.iqr_capper = joblib.load(transformers_dir / "iqr_capper.pkl")
            self.imputer = joblib.load(transformers_dir / "imputer_mice.pkl")
            self.scalers_standard = joblib.load(transformers_dir / "scalers_standard.pkl")
            self.scalers_robust = joblib.load(transformers_dir / "scalers_robust.pkl")
            with (PROCESSED_DIR / "split_info.json").open() as f:
                split_info = json.load(f)
            self.feature_columns = split_info["feature_columns"]
            self.numeric_columns = split_info["numeric_columns"]
            self.scaler_choice = split_info["scaler_choice"]
        except Exception as e:
            print(f"Warning: Could not load preprocessing artifacts: {e}")

        # Load feature names
        try:
            with open(EXPERIMENTS_DIR / "selected_features.json") as f:
                sel = json.load(f)
                self.feature_names = sel if isinstance(sel, list) else sel.get("final_features", [])
        except Exception as e:
            print(f"Warning: Could not load feature names: {e}")

    def _load_calibrator(self):
        try:
            self.calibrator = joblib.load(MODELS_DIR / "calibrator.pkl")
            with (MODELS_DIR / "calibration_info.json").open() as f:
                self.calibration_type = json.load(f)["type"]
        except Exception as e:
            print(f"Warning: Could not load calibrator: {e}")

    def _load_threshold(self):
        """Load optimal threshold."""
        try:
            with open(MODELS_DIR / "optimal_threshold.json") as f:
                data = json.load(f)
                self.optimal_threshold = data.get("optimal_threshold", 0.0262)
        except Exception:
            pass

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_threshold(self) -> float:
        return self.optimal_threshold

    def preprocess_history(self, history: List[Dict[str, Any]]) -> np.ndarray:
        """Apply the fitted training transforms and causal features to hourly raw inputs."""
        required_artifacts = [
            self.imputer, self.feature_names, self.numeric_columns,
            self.scalers_standard, self.scalers_robust,
        ]
        if any(value is None or value == [] for value in required_artifacts):
            raise RuntimeError("Model preprocessing artifacts are incomplete")

        raw_rows = []
        for item in history:
            row = {column: item["features"].get(column) for column in self.feature_columns}
            row["ICULOS"] = item["iculos"]
            raw_rows.append(row)
        raw = pd.DataFrame(raw_rows)

        for column, bounds in self.iqr_capper.items():
            if column in raw:
                raw[column] = raw[column].clip(bounds["lower"], bounds["upper"])

        numeric = raw[self.numeric_columns].copy()
        for column in self.numeric_columns:
            numeric[f"{column}_was_missing"] = numeric[column].isna().astype(int)

        imputer_columns = list(self.imputer.feature_names_in_)
        imputed = self.imputer.transform(numeric.reindex(columns=imputer_columns))
        processed = pd.DataFrame(imputed, columns=imputer_columns)

        for column in self.numeric_columns:
            scalers = self.scalers_standard if self.scaler_choice[column] == "standard" else self.scalers_robust
            processed[column] = scalers[column].transform(processed[[column]]).ravel()
        for column in STATIC_FEATURES:
            if column in raw:
                processed[column] = raw[column].to_numpy()

        temporal = self._engineer_last_row(processed)
        missing_features = [name for name in self.feature_names if name not in temporal]
        if missing_features:
            raise RuntimeError(f"Temporal pipeline is missing model features: {missing_features[:5]}")
        model_features = np.asarray([[temporal[name] for name in self.feature_names]], dtype=np.float32)
        return np.nan_to_num(model_features, nan=0.0)

    @staticmethod
    def _engineer_last_row(processed: pd.DataFrame) -> Dict[str, float]:
        """Match the causal feature definitions in enhanced/features/temporal.py."""
        result: Dict[str, float] = {}
        row_count = len(processed)

        for column in BASE_FEATURES:
            values = processed[column].to_numpy(dtype=float)
            current = values[-1]
            result[column] = current

            for lag in (1, 3, 6):
                result[f"{column}_lag{lag}"] = values[-lag - 1] if row_count > lag else np.nan
            for lag in (1, 3):
                result[f"{column}_diff{lag}h"] = current - values[-lag - 1] if row_count > lag else np.nan
            if row_count > 1:
                previous = values[-2]
                result[f"{column}_pct_change1h"] = (
                    (current - previous) / abs(previous) if previous != 0 else 0.0
                )
            else:
                result[f"{column}_pct_change1h"] = np.nan

            for window in (3, 6, 12):
                recent = values[-window:]
                valid = recent[~np.isnan(recent)]
                result[f"{column}_mean{window}h"] = float(np.mean(valid)) if valid.size else np.nan
                result[f"{column}_std{window}h"] = (
                    float(np.std(valid)) if recent.size > 1 and valid.size else np.nan
                )

            recent = values[-6:]
            valid = recent[~np.isnan(recent)]
            result[f"{column}_min6h"] = float(np.min(valid)) if valid.size else np.nan
            result[f"{column}_max6h"] = float(np.max(valid)) if valid.size else np.nan

            for window in (3, 6):
                recent = values[-window:]
                valid = recent[~np.isnan(recent)]
                result[f"{column}_slope{window}h"] = (
                    float(np.polyfit(np.arange(valid.size), valid, 1)[0])
                    if valid.size >= 2 else np.nan
                )

        last = processed.iloc[-1]
        for column in processed.columns:
            if column.endswith("_was_missing") or column in STATIC_FEATURES:
                value = last[column]
                result[column] = float(value) if pd.notna(value) else np.nan

        return result

    def _calibrate(self, raw_probability: float) -> float:
        if self.calibrator is None or self.calibration_type is None:
            raise RuntimeError("The selected model calibrator is unavailable")
        if self.calibration_type == "platt":
            score = np.clip(raw_probability, 1e-10, 1 - 1e-10)
            logit = np.log(score / (1 - score)).reshape(1, 1)
            return float(self.calibrator.predict_proba(logit)[0, 1])
        return float(self.calibrator.predict([raw_probability])[0])

    def explain_history(self, history: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
        """Return CatBoost SHAP drivers for the exact transformed prediction input."""
        model = self.models.get("catboost")
        if model is None or not self.feature_names:
            raise RuntimeError("CatBoost and selected feature names are required for explanations")

        features = self.preprocess_history(history)
        shap_values = np.asarray(
            model.get_feature_importance(Pool(features), type="ShapValues")
        )[0]
        contributions = shap_values[:-1]
        top_indices = np.argsort(np.abs(contributions))[-top_n:][::-1]
        return [
            {
                "feature": self.feature_names[index],
                "shap_value": float(contributions[index]),
                "feature_value": float(features[0, index]),
            }
            for index in top_indices
        ]

    def predict_ensemble(self, history: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        """Return calibrated stacked risk and individual base-model scores."""
        expected_models = {"rf", "xgb", "lgbm", "catboost"}
        if set(self.models) != expected_models or self.meta_learner is None:
            raise RuntimeError("All four base models and the stacking model must be loaded")

        X = self.preprocess_history(history)
        individual_probs: Dict[str, float] = {}
        for name in ("rf", "xgb", "lgbm", "catboost"):
            model_input = Pool(X) if name == "catboost" else X
            individual_probs[name] = float(self.models[name].predict_proba(model_input)[0, 1])

        meta_input = np.asarray([[
            individual_probs["rf"], individual_probs["xgb"],
            individual_probs["lgbm"], individual_probs["catboost"],
        ]])
        raw_probability = float(self.meta_learner.predict_proba(meta_input)[0, 1])
        return float(np.clip(self._calibrate(raw_probability), 0.0, 1.0)), individual_probs

    def get_model_metrics(self) -> Dict:
        """Load model metrics from disk."""
        metrics = {}

        for name in ["xgb", "lgbm", "catboost", "rf"]:
            try:
                with open(MODELS_DIR / f"{name}_metrics.json") as f:
                    metrics[name] = json.load(f)
            except Exception:
                pass

        # Stacking metrics
        try:
            with open(MODELS_DIR / "stack_test_metrics.json") as f:
                metrics["stacked"] = json.load(f)
        except Exception:
            pass

        # Calibration info
        try:
            with open(MODELS_DIR / "calibration_info.json") as f:
                metrics["calibration"] = json.load(f)
        except Exception:
            pass

        # External validation
        try:
            with open(EXPERIMENTS_DIR / "external_validation_setA_setB.json") as f:
                metrics["external"] = json.load(f)
        except Exception:
            pass

        # Optimal threshold
        try:
            with open(MODELS_DIR / "optimal_threshold.json") as f:
                metrics["threshold"] = json.load(f)
        except Exception:
            pass

        return metrics


# Global instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader