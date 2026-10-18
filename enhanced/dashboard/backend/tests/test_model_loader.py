import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

from app.schemas import PredictionRequest
from app.services.model_loader import BASE_FEATURES, ENHANCED_DIR, ModelLoader, PROCESSED_DIR


class _FixedClassifier:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, _features):
        return np.array([[1 - self.probability, self.probability]])

    def get_feature_importance(self, _features, type):
        self.asserted_importance_type = type
        return np.array([[0.25, -0.75, 1.5, 0.5]])


class _FixedMetaLearner:
    def predict_proba(self, _features):
        return np.array([[0.98, 0.02]])


class _FixedCalibrator:
    def predict(self, probabilities):
        return np.array([0.03 for _ in probabilities])


class ModelLoaderTests(unittest.TestCase):
    def test_prediction_request_requires_consecutive_recent_hours(self):
        valid = PredictionRequest(
            patient_id="p-test",
            iculos=3,
            history=[
                {"iculos": 1, "features": {}},
                {"iculos": 2, "features": {}},
                {"iculos": 3, "features": {}},
            ],
        )
        self.assertEqual(valid.history[-1].iculos, 3)

        with self.assertRaises(ValidationError):
            PredictionRequest(
                patient_id="p-test",
                iculos=3,
                history=[
                    {"iculos": 1, "features": {}},
                    {"iculos": 3, "features": {}},
                ],
            )

    def test_temporal_features_use_only_the_available_causal_history(self):
        processed = pd.DataFrame({
            **{feature: [10.0, 20.0] for feature in BASE_FEATURES},
            "Age": [60.0, 60.0],
            "HR_was_missing": [0.0, 0.0],
        })

        features = ModelLoader._engineer_last_row(processed)

        self.assertEqual(features["HR"], 20.0)
        self.assertEqual(features["HR_lag1"], 10.0)
        self.assertEqual(features["HR_diff1h"], 10.0)
        self.assertEqual(features["HR_pct_change1h"], 1.0)
        self.assertEqual(features["HR_mean3h"], 15.0)
        self.assertEqual(features["HR_std3h"], 5.0)
        self.assertAlmostEqual(features["HR_slope3h"], 10.0)
        self.assertTrue(np.isnan(features["HR_lag3"]))

    def test_temporal_features_allow_missing_static_patient_details(self):
        processed = pd.DataFrame({
            **{feature: [1.0] for feature in BASE_FEATURES},
            "Age": [None],
            "Gender": [None],
            "Unit2": [None],
        })

        features = ModelLoader._engineer_last_row(processed)

        self.assertTrue(np.isnan(features["Age"]))
        self.assertTrue(np.isnan(features["Gender"]))
        self.assertTrue(np.isnan(features["Unit2"]))

    def test_ensemble_returns_calibrated_probability(self):
        loader = object.__new__(ModelLoader)
        loader.models = {
            name: _FixedClassifier(probability)
            for name, probability in {
                "rf": 0.1,
                "xgb": 0.2,
                "lgbm": 0.3,
                "catboost": 0.4,
            }.items()
        }
        loader.meta_learner = _FixedMetaLearner()
        loader.calibrator = _FixedCalibrator()
        loader.calibration_type = "isotonic"
        loader.preprocess_history = lambda _history: np.zeros((1, 2), dtype=np.float32)

        with patch("app.services.model_loader.Pool", side_effect=lambda values: values):
            probability, model_scores = loader.predict_ensemble([
                {"iculos": 1, "features": {}}
            ])

        self.assertEqual(probability, 0.03)
        self.assertEqual(model_scores["catboost"], 0.4)

    def test_explanation_returns_largest_catboost_shap_drivers(self):
        loader = object.__new__(ModelLoader)
        catboost_model = _FixedClassifier(0.4)
        loader.models = {"catboost": catboost_model}
        loader.feature_names = ["HR", "Temp", "WBC"]
        loader.preprocess_history = lambda _history: np.array([[0.1, 0.2, 0.3]])

        with patch("app.services.model_loader.Pool", side_effect=lambda values: values):
            explanations = loader.explain_history([{"iculos": 1, "features": {}}], top_n=2)

        self.assertEqual(catboost_model.asserted_importance_type, "ShapValues")
        self.assertEqual([item["feature"] for item in explanations], ["WBC", "Temp"])
        self.assertEqual(explanations[1]["shap_value"], -0.75)
        self.assertEqual(explanations[0]["feature_value"], 0.3)

    def test_saved_test_patient_matches_training_features_and_calibrated_prediction(self):
        required_files = [
            PROCESSED_DIR / "test_processed.parquet",
            PROCESSED_DIR / "test_temporal.parquet",
            ENHANCED_DIR / "models" / "calibrated_test_preds.npy",
        ]
        if not all(path.exists() for path in required_files):
            self.skipTest("Generated processed data and model artifacts are unavailable")

        root = ENHANCED_DIR.parent
        identifiers = pd.read_parquet(
            PROCESSED_DIR / "test_processed.parquet",
            columns=["patient_id", "ICULOS"],
        )
        patient_id = next(
            pid for pid, count in identifiers.groupby("patient_id").size().items()
            if count >= 12
        )
        source_paths = [
            root / "dataset/physionet_sepsis/training/training_setA" / f"{patient_id}.psv",
            root / "dataset/physionet_sepsis/training/training_setB" / f"{patient_id}.psv",
        ]
        raw_path = next(path for path in source_paths if path.exists())
        raw_patient = pd.read_csv(raw_path, sep="|")
        end_hour = int(identifiers.loc[identifiers.patient_id == patient_id, "ICULOS"].max())
        first_hour = max(1, end_hour - 11)
        raw_window = raw_patient[
            raw_patient.ICULOS.between(first_hour, end_hour)
        ]
        feature_names = [
            "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2", "pH",
            "PaCO2", "SaO2", "BUN", "Calcium", "Glucose", "Potassium", "Hct",
            "Hgb", "WBC", "Platelets", "Age", "Gender", "Unit1", "Unit2",
            "HospAdmTime",
        ]
        history = [
            {
                "iculos": int(row["ICULOS"]),
                "features": {
                    name: None if pd.isna(row[name]) else float(row[name])
                    for name in feature_names
                },
            }
            for _, row in raw_window.iterrows()
        ]

        loader = ModelLoader()
        predicted_features = loader.preprocess_history(history)[0]
        selected = loader.feature_names
        temporal_patient = pd.read_parquet(
            PROCESSED_DIR / "test_temporal.parquet",
            columns=["patient_id", "ICULOS", *selected],
            filters=[("patient_id", "==", patient_id)],
        )
        saved_row = temporal_patient.loc[
            temporal_patient.ICULOS == end_hour, selected
        ].iloc[0].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(predicted_features, saved_row, rtol=1e-5, atol=1e-5)

        calibrated_probability, _ = loader.predict_ensemble(history)
        test_keys = pd.read_parquet(
            PROCESSED_DIR / "test_temporal.parquet",
            columns=["patient_id", "ICULOS"],
        )
        test_position = np.flatnonzero(
            (test_keys.patient_id.to_numpy() == patient_id)
            & (test_keys.ICULOS.to_numpy() == end_hour)
        )[0]
        saved_probability = np.load(
            ENHANCED_DIR / "models" / "calibrated_test_preds.npy",
            mmap_mode="r",
        )[test_position]
        self.assertAlmostEqual(calibrated_probability, float(saved_probability), places=6)


if __name__ == "__main__":
    unittest.main()