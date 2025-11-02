from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os

from backend.services.indicator_prediction_service import IndicatorPredictionService

router = APIRouter()

class PredictionRequest(BaseModel):
    data_path: str

class PredictionResponse(BaseModel):
    status: str
    message: str
    models_evaluated: list = []

@router.post("/predict", response_model=PredictionResponse)
async def run_indicator_prediction(request: PredictionRequest) -> Dict[str, Any]:
    """
    Run ML model evaluation for indicator prediction.
    
    Args:
        request: PredictionRequest containing the path to the data file
        
    Returns:
        PredictionResponse with status and results
    """
    try:
        # Validate data path exists
        if not os.path.exists(request.data_path):
            raise HTTPException(status_code=404, detail=f"Data file not found: {request.data_path}")
        
        # Initialize service and run prediction
        service = IndicatorPredictionService(request.data_path)
        result = service.run_prediction()
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_prediction_status() -> Dict[str, str]:
    """Check if the indicator prediction service is available."""
    return {"status": "available", "service": "indicator_prediction"} 