# autonomous_trading_system/backend/api/routes/liquidation_hunter_api.py

from fastapi import APIRouter, Depends, HTTPException
from starlette.requests import Request
from typing import Dict, Any

from autonomous_trading_system.backend.services.liquidation_hunter_service import LiquidationHunterService

router = APIRouter()

def get_liquidation_hunter_service(request: Request) -> LiquidationHunterService:
    """Dependency injector to get the LiquidationHunterService instance."""
    service = request.app.state.liquidation_hunter_service
    if not service:
        raise HTTPException(status_code=503, detail="Liquidation Hunter Service is not available.")
    return service

@router.get("/liquidation-hunter/analysis", 
            response_model=Dict[str, Any],
            tags=["Liquidation Hunter"],
            summary="Get the latest market analysis from the Liquidation Hunter",
            description="Returns a JSON object with the most recent market bias analysis, including the target coin, total liquidation risks, and a detailed breakdown per analyzed coin.")
async def get_liquidation_analysis(
    service: LiquidationHunterService = Depends(get_liquidation_hunter_service)
) -> Dict[str, Any]:
    """
    Retrieves the last computed market analysis report from the Liquidation Hunter service.
    """
    analysis_report = service.get_last_analysis()
    if not analysis_report:
        raise HTTPException(status_code=404, detail="No analysis report has been generated yet. Please wait for the service to run its first analysis cycle.")
    return analysis_report
