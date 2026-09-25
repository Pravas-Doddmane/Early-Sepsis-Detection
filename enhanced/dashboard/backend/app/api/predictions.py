"""API routes for predictions."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from app.database import get_db
from app.models import Patient, Prediction, ModelRegistry
from app.schemas import (
    PredictionRequest, PredictionResponse, PatientResponse,
    PatientListResponse, PaginationParams
)
from app.services.model_loader import get_model_loader

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionResponse)
def create_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Make a prediction for a patient at a specific hour."""
    loader = get_model_loader()

    # Get ensemble prediction
    prob_sepsis, individual_probs = loader.predict_ensemble(request.features)
    threshold = loader.get_threshold()
    prediction = int(prob_sepsis >= threshold)

    # Get top SHAP features (simplified - just top individual model features)
    shap_features = [
        {"model": k, "probability": v}
        for k, v in sorted(individual_probs.items(), key=lambda x: -x[1])
    ]

    # Save to database
    db_pred = Prediction(
        patient_id=request.patient_id,
        iculos=request.iculos,
        model_version="stacked_ensemble_v1",
        prob_sepsis=prob_sepsis,
        threshold=threshold,
        prediction=prediction,
        true_label=None,  # Unknown at prediction time
        shap_top_features=json.dumps(shap_features)
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)

    # Parse JSON string for response
    db_pred.shap_top_features = shap_features
    return db_pred


@router.get("/patient/{patient_id}", response_model=List[PredictionResponse])
def get_patient_predictions(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get all predictions for a patient."""
    preds = db.query(Prediction).filter(
        Prediction.patient_id == patient_id
    ).order_by(Prediction.iculos).all()
    return preds


@router.get("/recent", response_model=List[PredictionResponse])
def get_recent_predictions(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get most recent predictions."""
    preds = db.query(Prediction).order_by(
        Prediction.created_at.desc()
    ).limit(limit).all()
    return preds