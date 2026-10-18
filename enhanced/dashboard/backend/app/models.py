"""SQLAlchemy models for the dashboard database."""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Patient(Base):
    """Patient metadata from PhysioNet dataset."""
    __tablename__ = "patients"

    patient_id = Column(String(50), primary_key=True)
    age = Column(Float)
    gender = Column(Integer)  # 0/1
    unit1 = Column(Integer)
    unit2 = Column(Integer)
    hosp_adm_time = Column(Float)
    sepsis_label = Column(Integer)  # 0/1 at patient level
    onset_hour = Column(Integer, nullable=True)  # hour of sepsis onset if positive

    # Relationships
    hourly_records = relationship("HourlyRecord", back_populates="patient", lazy="dynamic")
    predictions = relationship("Prediction", back_populates="patient", lazy="dynamic")


class HourlyRecord(Base):
    """Hourly vital signs and lab values (temporal features)."""
    __tablename__ = "hourly_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    iculos = Column(Integer, index=True)  # ICU length of stay in hours

    # Vitals
    hr = Column(Float)
    o2sat = Column(Float)
    temp = Column(Float)
    sbp = Column(Float)
    map = Column(Float)
    dbp = Column(Float)
    resp = Column(Float)

    # Labs (highly missing)
    etco2 = Column(Float)
    baseexcess = Column(Float)
    hco3 = Column(Float)
    fio2 = Column(Float)
    paco2 = Column(Float)
    sao2 = Column(Float)
    ast = Column(Float)
    bun = Column(Float)
    alkphos = Column(Float)
    calcium = Column(Float)
    chloride = Column(Float)
    creatinine = Column(Float)
    bilirubin_direct = Column(Float)
    glucose = Column(Float)
    lactate = Column(Float)
    magnesium = Column(Float)
    phosphate = Column(Float)
    potassium = Column(Float)
    bilirubin_total = Column(Float)
    troponini = Column(Float)
    hct = Column(Float)
    hgb = Column(Float)
    ptt = Column(Float)
    wbc = Column(Float)
    fibrinogen = Column(Float)
    platelets = Column(Float)

    # Target
    sepsis_label = Column(Integer)  # 0/1 at this hour

    # Relationships
    patient = relationship("Patient", back_populates="hourly_records")

    __table_args__ = (
        Index("ix_patient_iculos", "patient_id", "iculos"),
    )


class Prediction(Base):
    """Model prediction audit trail."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    iculos = Column(Integer)
    model_version = Column(String(50))
    prob_sepsis = Column(Float)
    threshold = Column(Float)
    prediction = Column(Integer)  # 0/1
    true_label = Column(Integer, nullable=True)  # Known later
    shap_top_features = Column(Text)  # JSON string of top SHAP features
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="predictions")


class ModelRegistry(Base):
    """Model version registry."""
    __tablename__ = "models"

    version = Column(String(50), primary_key=True)
    name = Column(String(100))  # e.g., "catboost", "stacked_ensemble"
    roc_auc = Column(Float)
    pr_auc = Column(Float)
    threshold = Column(Float)
    calibration_type = Column(String(50))  # "isotonic", "platt", "none"
    features_json = Column(Text)  # JSON list of feature names
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)