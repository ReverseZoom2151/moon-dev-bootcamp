import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Tuple
from backend.services.yahoo_data_service import YahooDataService
from backend.core.config import config

router = APIRouter()
logger = logging.getLogger(__name__)

class YahooDataRequest(BaseModel):
    symbols: List[Tuple[str, str]] = config.get('YAHOO_SYMBOLS', [
        ('AAPL', 'stock'), ('GOOGL', 'stock'), ('MSFT', 'stock'), ('AMZN', 'stock'),
        ('EURUSD=X', 'forex'), ('GBPUSD=X', 'forex'), ('USDJPY=X', 'forex'), ('AUDUSD=X', 'forex'),
        ('ES=F', 'future'), ('NQ=F', 'future'), ('YM=F', 'future'), ('RTY=F', 'future'),
        ('GC=F', 'future'), ('SI=F', 'future'), ('CL=F', 'future'), ('NG=F', 'future'),
        ('BTC-USD', 'crypto'), ('ETH-USD', 'crypto'), ('XRP-USD', 'crypto'), ('LTC-USD', 'crypto'),
        ('BCH-USD', 'crypto'), ('ADA-USD', 'crypto'), ('DOT-USD', 'crypto'), ('SOL-USD', 'crypto'),
        ('BNB-USD', 'crypto'), ('DOGE-USD', 'crypto'), ('SHIB-USD', 'crypto'), ('AVAX-USD', 'crypto'),
        ('TRX-USD', 'crypto'), ('LINK-USD', 'crypto'), ('MATIC-USD', 'crypto'), ('UNI-USD', 'crypto'),
        ('ATOM-USD', 'crypto'), ('XLM-USD', 'crypto'), ('ETC-USD', 'crypto'), ('TON-USD', 'crypto'),
        ('ICP-USD', 'crypto'), ('HBAR-USD', 'crypto'), ('APT-USD', 'crypto'), ('ARB-USD', 'crypto'),
        ('NEAR-USD', 'crypto'), ('VET-USD', 'crypto'), ('ALGO-USD', 'crypto'), ('QNT-USD', 'crypto'),
        ('FIL-USD', 'crypto'), ('EOS-USD', 'crypto')
    ])
    intervals: List[str] = ['1d', '1h', '1wk', '1mo']

yahoo_data_service = YahooDataService()

@router.post('/fetch', response_model=Dict[str, Dict[str, int]])
async def fetch_yahoo_data(request: YahooDataRequest):
    """Fetch historical data from Yahoo Finance for specified symbols and intervals."""
    try:
        logger.info(f"Fetching Yahoo data for symbols: {request.symbols} at intervals: {request.intervals}")
        result = await yahoo_data_service.download_all_data(request.symbols, request.intervals)
        return result
    except Exception as e:
        logger.error(f"Error fetching Yahoo data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching Yahoo data: {str(e)}"
        )

@router.get('/status')
async def get_yahoo_data_status():
    """Get the status of the Yahoo data service."""
    return {'status': 'Yahoo data service is running'}

