"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


# Patient schemas
class PatientBase(BaseModel):
    patient_id: str
    age: Optional[float] = None
    gender: Optional[int] = None
    unit1: Optional[int] = None
    unit2: Optional[int] = None
    hosp_adm_time: Optional[float] = None
    sepsis_label: Optional[int] = None
    onset_hour: Optional[int] = None


class PatientResponse(PatientBase):
    class Config:
        from_attributes = True


class PatientListResponse(BaseModel):
    patients: List[PatientResponse]
    total: int
    page: int
    page_size: int


# Hourly record schemas
class HourlyRecordBase(BaseModel):
    patient_id: str
    iculos: int
    hr: Optional[float] = None
    o2sat: Optional[float] = None
    temp: Optional[float] = None
    sbp: Optional[float] = None
    map: Optional[float] = None
    dbp: Optional[float] = None
    resp: Optional[float] = None
    etco2: Optional[float] = None
    baseexcess: Optional[float] = None
    hco3: Optional[float] = None
    fio2: Optional[float] = None
    paco2: Optional[float] = None
    sao2: Optional[float] = None
    ast: Optional[float] = None
    bun: Optional[float] = None
    alkphos: Optional[float] = None
    calcium: Optional[float] = None
    chloride: Optional[float] = None
    creatinine: Optional[float] = None
    bilirubin_direct: Optional[float] = None
    glucose: Optional[float] = None
    lactate: Optional[float] = None
    magnesium: Optional[float] = None
    phosphate: Optional[float] = None
    potassium: Optional[float] = None
    bilirubin_total: Optional[float] = None
    troponini: Optional[float] = None
    hct: Optional[float] = None
    hgb: Optional[float] = None
    ptt: Optional[float] = None
    wbc: Optional[float] = None
    fibrinogen: Optional[float] = None
    platelets: Optional[float] = None
    sepsis_label: Optional[int] = None


class HourlyRecordResponse(HourlyRecordBase):
    id: int

    class Config:
        from_attributes = True


# Prediction schemas
class HourlyFeatureInput(BaseModel):
    iculos: int = Field(ge=1, le=336)
    features: Dict[str, Optional[float]]


class PredictionRequest(BaseModel):
    """Request a prediction from a contiguous window of raw hourly measurements."""
    patient_id: str
    iculos: int = Field(ge=1, le=336)
    history: List[HourlyFeatureInput] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_history_window(self):
        expected_hours = list(range(max(1, self.iculos - 11), self.iculos + 1))
        actual_hours = [entry.iculos for entry in self.history]
        if actual_hours != expected_hours:
            raise ValueError(f"history must contain each consecutive ICU hour: {expected_hours}")
        return self


class PredictionResponse(BaseModel):
    id: int
    patient_id: str
    iculos: int
    model_version: str
    prob_sepsis: float
    threshold: float
    prediction: int
    true_label: Optional[int] = None
    shap_top_features: Optional[List[Dict[str, Any]]] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Explainability schemas
class SHAPExplanation(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class LIMEExplanation(BaseModel):
    feature: str
    weight: float


class ExplanationResponse(BaseModel):
    patient_id: str
    iculos: int
    shap: List[SHAPExplanation]
    lime: List[LIMEExplanation]
    prediction_prob: float
    prediction: int
    true_label: Optional[int] = None


# Model metrics schemas
class ModelMetricsResponse(BaseModel):
    version: str
    name: str
    roc_auc: float
    pr_auc: float
    threshold: float
    calibration_type: str
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ModelComparisonResponse(BaseModel):
    models: List[ModelMetricsResponse]
    active_model: Optional[str] = None


# Overview schemas
class OverviewMetrics(BaseModel):
    test_roc_auc: float
    test_pr_auc: float
    test_sensitivity: float
    test_specificity: float
    test_precision: float
    test_f1: float
    test_mcc: float
    threshold: float
    calibration_ece: float
    external_setA_roc: float
    external_setB_roc: float


class OverviewResponse(BaseModel):
    metrics: OverviewMetrics
    feature_importance: List[Dict[str, Any]]
    temporal_importance: List[Dict[str, Any]]


# Pagination
class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)