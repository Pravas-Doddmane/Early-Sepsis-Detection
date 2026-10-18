"""API routes for patient data browsing."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional

from app.database import get_db
from app.models import Patient, HourlyRecord
from app.schemas import PatientResponse, PatientListResponse, HourlyRecordResponse, PaginationParams

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.get("/", response_model=PatientListResponse)
def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sepsis_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    """List patients with pagination."""
    query = db.query(Patient)

    if sepsis_only:
        query = query.filter(Patient.sepsis_label == 1)

    total = query.count()
    patients = query.offset((page - 1) * page_size).limit(page_size).all()

    return PatientListResponse(
        patients=patients,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    """Get patient details."""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.get("/{patient_id}/records", response_model=List[HourlyRecordResponse])
def get_patient_records(
    patient_id: str,
    start_hour: int = Query(1),
    end_hour: int = Query(999),
    db: Session = Depends(get_db)
):
    """Get hourly records for a patient."""
    records = db.query(HourlyRecord).filter(
        HourlyRecord.patient_id == patient_id,
        HourlyRecord.iculos >= start_hour,
        HourlyRecord.iculos <= end_hour
    ).order_by(HourlyRecord.iculos).all()

    return records


@router.get("/stats/summary")
def get_patient_stats(db: Session = Depends(get_db)):
    """Get dataset summary statistics."""
    total = db.query(Patient).count()
    sepsis = db.query(Patient).filter(Patient.sepsis_label == 1).count()

    return {
        "total_patients": total,
        "sepsis_patients": sepsis,
        "non_sepsis_patients": total - sepsis,
        "sepsis_rate": sepsis / total if total > 0 else 0
    }