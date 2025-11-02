import logging
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel
from backend.services.historical_data_service import HistoricalDataService
from backend.core.config import config

router = APIRouter()
logger = logging.getLogger(__name__)

class HistoricalDataRequest(BaseModel):
    symbol: str = config.get('DEFAULT_SYMBOL', 'UNI/USD')
    timeframe: str = config.get('DEFAULT_TIMEFRAME', '1h')
    weeks: int = config.get('DEFAULT_WEEKS_TO_FETCH', 100)

historical_data_service = HistoricalDataService()

@router.on_event('startup')
async def startup_event():
    """Initialize the historical data service on startup."""
    try:
        await historical_data_service.initialize()
        logger.info("Historical Data Service initialized on API startup.")
    except Exception as e:
        logger.error(f"Failed to initialize Historical Data Service on startup: {e}")

@router.on_event('shutdown')
async def shutdown_event():
    """Close the historical data service on shutdown."""
    try:
        await historical_data_service.close()
        logger.info("Historical Data Service closed on API shutdown.")
    except Exception as e:
        logger.error(f"Error closing Historical Data Service on shutdown: {e}")

@router.post('/fetch', status_code=status.HTTP_200_OK)
async def fetch_historical_data(request: HistoricalDataRequest):
    """Fetch historical OHLCV data for a given symbol, timeframe, and number of weeks."""
    try:
        logger.info(f"Fetching historical data for {request.symbol}, timeframe: {request.timeframe}, weeks: {request.weeks}")
        data = await historical_data_service.get_historical_data(request.symbol, request.timeframe, request.weeks)
        if data.empty:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No historical data found or fetched.")
        return data.to_dict('records')
    except ValueError as ve:
        logger.error(f"ValueError fetching historical data: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error fetching historical data: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while fetching historical data.")