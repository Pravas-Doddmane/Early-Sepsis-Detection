"""Service for loading and running ML models."""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

ENHANCED_DIR = Path(os.environ.get("ENHANCED_DIR", Path(__file__).resolve().parents[4]))
MODELS_DIR = ENHANCED_DIR / "models"
EXPERIMENTS_DIR = ENHANCED_DIR / "experiments"


class ModelLoader:
    """Loads and manages all trained models for inference."""

    def __init__(self):
        self.models: Dict[str, any] = {}
        self.meta_learner = None
        self.scaler = None
        self.imputer = None
        self.feature_names: List[str] = []
        self.optimal_threshold = 0.026202020202020202
        self._load_models()
        self._load_preprocessing()
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
            self.scaler = joblib.load(MODELS_DIR / "transformers" / "scaler_robust.pkl")
        except Exception:
            try:
                self.scaler = joblib.load(MODELS_DIR / "transformers" / "scalers_robust.pkl")
            except Exception as e:
                print(f"Warning: Could not load scaler: {e}")

        try:
            self.imputer = joblib.load(MODELS_DIR / "transformers" / "imputer_knn.pkl")
        except Exception as e:
            print(f"Warning: Could not load imputer: {e}")

        # Load feature names
        try:
            with open(EXPERIMENTS_DIR / "selected_features.json") as f:
                sel = json.load(f)
                self.feature_names = sel if isinstance(sel, list) else sel.get("final_features", [])
        except Exception as e:
            print(f"Warning: Could not load feature names: {e}")

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

    def preprocess(self, features: Dict[str, float]) -> np.ndarray:
        """Preprocess input features for prediction."""
        # Create array in correct feature order
        X = np.array([[features.get(f, 0.0) for f in self.feature_names]], dtype=np.float32)

        # Impute missing (already 0, but imputer may have learned statistics)
        if self.imputer is not None:
            try:
                X = self.imputer.transform(X)
            except Exception:
                pass

        # Scale
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception:
                pass

        return X

    def predict_single(self, model_name: str, features: Dict[str, float]) -> float:
        """Get probability from a single model."""
        X = self.preprocess(features)

        if model_name == "catboost" and "catboost" in self.models:
            pool = Pool(X)
            return float(self.models["catboost"].predict_proba(pool)[0, 1])

        model = self.models.get(model_name)
        if model is not None:
            if hasattr(model, "predict_proba"):
                return float(model.predict_proba(X)[0, 1])
            elif hasattr(model, "predict"):
                return float(model.predict(X)[0])

        return 0.0

    def predict_ensemble(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Get stacked ensemble prediction + individual model probs."""
        individual_probs = {}

        for name in ["rf", "xgb", "lgbm", "catboost"]:
            if name in self.models:
                individual_probs[name] = self.predict_single(name, features)

        # Stack using meta-learner
        if self.meta_learner is not None and len(individual_probs) == 4:
            # Order must match training: rf, xgb, lgbm, catboost
            meta_X = np.array([[
                individual_probs.get("rf", 0),
                individual_probs.get("xgb", 0),
                individual_probs.get("lgbm", 0),
                individual_probs.get("catboost", 0),
            ]])
            ensemble_prob = float(self.meta_learner.predict_proba(meta_X)[0, 1])
        else:
            # Fallback: average
            ensemble_prob = np.mean(list(individual_probs.values())) if individual_probs else 0.0

        return float(np.clip(ensemble_prob, 0.0, 1.0)), individual_probs

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