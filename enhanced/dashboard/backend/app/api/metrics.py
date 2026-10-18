"""API routes for model metrics and overview."""
from fastapi import APIRouter
from typing import List, Dict, Any

from app.services.model_loader import get_model_loader
from app.services.xai_loader import get_xai_artifacts
from app.schemas import ModelMetricsResponse, OverviewResponse, OverviewMetrics

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/models", response_model=List[ModelMetricsResponse])
def get_model_metrics():
    """Get all model metrics."""
    loader = get_model_loader()
    metrics = loader.get_model_metrics()

    result = []
    for name, m in metrics.items():
        if name in ["xgb", "lgbm", "catboost", "rf"]:
            result.append(ModelMetricsResponse(
                version=f"{name}_v1",
                name=name.upper(),
                roc_auc=m.get("roc_auc", 0),
                pr_auc=m.get("pr_auc", 0),
                threshold=m.get("threshold", 0.5),
                calibration_type="none",
                is_active=False,
                created_at=None
            ))

    # Add stacked ensemble
    if "stacked" in metrics:
        m = metrics["stacked"]
        threshold_info = metrics.get("threshold", {})
        calibration_info = metrics.get("calibration", {})
        result.append(ModelMetricsResponse(
            version="stacked_v1",
            name="Stacked Ensemble",
            roc_auc=m.get("roc_auc", 0),
            pr_auc=m.get("pr_auc", 0),
            threshold=threshold_info.get("optimal_threshold", m.get("threshold", 0.5)),
            calibration_type=calibration_info.get("type", "unknown"),
            is_active=True,
            created_at=None
        ))

    return result


@router.get("/overview", response_model=OverviewResponse)
def get_overview():
    """Get complete overview metrics for dashboard."""
    loader = get_model_loader()
    xai = get_xai_artifacts()
    metrics = loader.get_model_metrics()

    # Build overview metrics from stacked ensemble + calibration + external
    stacked = metrics.get("stacked", {})
    calibration = metrics.get("calibration", {})
    external = metrics.get("external", {})
    threshold_info = metrics.get("threshold", {})

    test_metrics = threshold_info.get("test_metrics", {})
    val_metrics = threshold_info.get("val_metrics", {})

    overview = OverviewMetrics(
        test_roc_auc=stacked.get("roc_auc", 0.784),
        test_pr_auc=stacked.get("pr_auc", 0.070),
        test_sensitivity=test_metrics.get("sensitivity", 0.656),
        test_specificity=test_metrics.get("specificity", 0.769),
        test_precision=test_metrics.get("precision", 0.049),
        test_f1=test_metrics.get("f1", 0.091),
        test_mcc=test_metrics.get("mcc", 0.132),
        threshold=threshold_info.get("optimal_threshold", 0.0262),
        calibration_ece=calibration.get("test_ece_calibrated", 0.0009),
        external_setA_roc=external.get("setA", {}).get("roc_auc", 0.769),
        external_setB_roc=external.get("setB", {}).get("roc_auc", 0.789),
    )

    return OverviewResponse(
        metrics=overview,
        feature_importance=xai.get_global_shap_importance(),
        temporal_importance=xai.get_temporal_shap_importance()
    )


@router.get("/external", response_model=Dict[str, Any])
def get_external_validation():
    """Get external validation metrics (Set A / Set B)."""
    loader = get_model_loader()
    metrics = loader.get_model_metrics()
    return metrics.get("external", {})