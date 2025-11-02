from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os

from backend.services.feature_importance_analysis_service import FeatureImportanceAnalysisService

router = APIRouter()

class FeatureImportanceRequest(BaseModel):
    importance_filepath: str
    r2_filter_threshold: Optional[float] = 0.95
    top_n_features: Optional[int] = 50

class FeatureImportanceResponse(BaseModel):
    status: str
    message: str
    models_processed: List[str] = []

@router.post("/analyze", response_model=FeatureImportanceResponse)
async def analyze_feature_importance(request: FeatureImportanceRequest) -> Dict[str, Any]:
    """
    Analyze feature importance from a CSV file and save top features for each model.
    
    Args:
        request: FeatureImportanceRequest containing the path to the importance file
        
    Returns:
        FeatureImportanceResponse with status and results
    """
    try:
        # Validate file exists
        if not os.path.exists(request.importance_filepath):
            raise HTTPException(status_code=404, detail=f"File not found: {request.importance_filepath}")
        
        # Initialize service and run analysis
        service = FeatureImportanceAnalysisService(request.importance_filepath)
        
        # Update config if custom values provided
        if request.r2_filter_threshold is not None:
            service.config['R2_FILTER_THRESHOLD'] = request.r2_filter_threshold
        if request.top_n_features is not None:
            service.config['TOP_N_FEATURES'] = request.top_n_features
        
        result = service.analyze_feature_importance()
        
        return FeatureImportanceResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_feature_importance_status() -> Dict[str, str]:
    """Check if the feature importance analysis service is available."""
    return {"status": "available", "service": "feature_importance_analysis"}

@router.get("/config")
async def get_feature_importance_config() -> Dict[str, Any]:
    """Get the current configuration for feature importance analysis."""
    from backend.services.feature_importance_analysis_service import CONFIG
    return {
        "r2_filter_threshold": CONFIG['R2_FILTER_THRESHOLD'],
        "top_n_features": CONFIG['TOP_N_FEATURES'],
        "metrics_to_sort": CONFIG['METRICS_TO_SORT']
    }

@router.post("/analyze-from-prediction")
async def analyze_from_prediction_results() -> Dict[str, Any]:
    """
    Analyze feature importance from the default prediction results location.
    This uses the output from the indicator prediction service.
    """
    try:
        from backend.core.config import settings
        import os
        
        # Use the default feature importance file from prediction results
        importance_filepath = os.path.join(settings.RESULTS_DIR_PATH, "feature_importance.csv")
        
        if not os.path.exists(importance_filepath):
            raise HTTPException(
                status_code=404, 
                detail="No feature importance results found. Please run indicator prediction first."
            )
        
        # Initialize service and run analysis
        service = FeatureImportanceAnalysisService(importance_filepath)
        result = service.analyze_feature_importance()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 