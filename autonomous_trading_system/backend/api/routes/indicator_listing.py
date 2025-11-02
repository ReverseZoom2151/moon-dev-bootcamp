import logging
from fastapi import APIRouter, HTTPException, status
from backend.services.indicator_listing_service import IndicatorListingService
from backend.core.config import config

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get('/list_indicators')
async def list_indicators():
    """List available technical indicators from pandas_ta and talib libraries."""
    try:
        service = IndicatorListingService()
        await service.list_and_save_indicators()
        return {'message': 'Indicators listed and saved successfully'}
    except Exception as e:
        logger.error(f"Error listing indicators: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) 