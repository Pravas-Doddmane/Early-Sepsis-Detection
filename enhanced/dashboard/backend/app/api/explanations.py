"""API routes for explainability (SHAP, LIME)."""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from app.services.xai_loader import XAI_DIR, get_xai_artifacts
from app.schemas import ExplanationResponse, SHAPExplanation, LIMEExplanation

router = APIRouter(prefix="/api/explanations", tags=["explanations"])


@router.get("/patient/{patient_id}/{iculos}", response_model=ExplanationResponse)
def get_patient_explanation(
    patient_id: str,
    iculos: int
):
    """Get SHAP and LIME explanations for a specific patient-hour."""
    xai = get_xai_artifacts()

    # Get patient info
    info = xai.get_patient_info(patient_id, iculos)
    if info is None:
        raise HTTPException(status_code=404, detail="Patient/hour not found")

    # Get SHAP
    shap_data = xai.get_patient_shap(patient_id, iculos) or []
    shap_explanations = [
        SHAPExplanation(
            feature=d["feature"],
            shap_value=d["shap_value"],
            feature_value=d["feature_value"]
        )
        for d in shap_data
    ]

    # Get LIME
    lime_data = xai.get_patient_lime(patient_id, iculos) or []
    lime_explanations = [
        LIMEExplanation(feature=d["feature"], weight=d["weight"])
        for d in lime_data
    ]

    return ExplanationResponse(
        patient_id=patient_id,
        iculos=iculos,
        shap=shap_explanations,
        lime=lime_explanations,
        prediction_prob=info["prob_sepsis"],
        prediction=info["prediction"],
        true_label=info["true_label"]
    )


@router.get("/global/shap", response_model=List[dict])
def get_global_shap():
    """Get global SHAP feature importance."""
    xai = get_xai_artifacts()
    return xai.get_global_shap_importance()


@router.get("/temporal/shap", response_model=List[dict])
def get_temporal_shap():
    """Get temporal SHAP importance (Early/Mid/Late ICU stay)."""
    xai = get_xai_artifacts()
    return xai.get_temporal_shap_importance()


@router.get("/images/global_shap")
def get_global_shap_image():
    """Serve global SHAP beeswarm plot."""
    from fastapi.responses import FileResponse
    xai = get_xai_artifacts()
    img_path = XAI_DIR / "global_shap_beeswarm.png"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/temporal_shap")
def get_temporal_shap_image():
    """Serve temporal SHAP importance plot."""
    from fastapi.responses import FileResponse
    xai = get_xai_artifacts()
    img_path = XAI_DIR / "temporal_shap_importance.png"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/examples/{plot_type}/{label}")
def get_example_explanation_image(plot_type: str, label: str):
    """Serve one of the pre-computed TP/TN explanation examples."""
    from fastapi.responses import FileResponse

    if plot_type not in {"waterfall", "lime"} or label not in {"TP", "TN"}:
        raise HTTPException(status_code=404, detail="Explanation example not found")

    xai = get_xai_artifacts()
    img_path = XAI_DIR / f"patient_{plot_type}_{label}.png"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/patient_waterfall/{patient_id}/{iculos}/{label}")
def get_patient_waterfall_image(patient_id: str, iculos: int, label: str):
    """Serve patient waterfall plot (TP or TN)."""
    from fastapi.responses import FileResponse
    xai = get_xai_artifacts()
    img_path = XAI_DIR / f"patient_waterfall_{label}.png"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/images/patient_lime/{patient_id}/{iculos}/{label}")
def get_patient_lime_image(patient_id: str, iculos: int, label: str):
    """Serve patient LIME plot (TP or TN)."""
    from fastapi.responses import FileResponse
    xai = get_xai_artifacts()
    img_path = XAI_DIR / f"patient_lime_{label}.png"
    if img_path.exists():
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Image not found")
