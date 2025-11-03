"""
API Routes for Market Data
===========================
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
import logging

from .. import get_gordon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/price/{symbol}")
async def get_price(
    symbol: str,
    exchange: Optional[str] = Query("binance", description="Exchange name"),
    gordon=Depends(get_gordon)
):
    """Get current price for a symbol."""
    try:
        # Get price via Gordon's exchange adapters
        if exchange in gordon.exchanges:
            exchange_adapter = gordon.exchanges[exchange]
            await exchange_adapter.initialize()
            ticker = await exchange_adapter.get_ticker(symbol)
            
            if ticker:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "price": float(ticker.get('last', 0)),
                    "bid": float(ticker.get('bid', 0)),
                    "ask": float(ticker.get('ask', 0)),
                    "volume": float(ticker.get('volume', 0))
                }
            else:
                raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found on {exchange}")
        else:
            raise HTTPException(status_code=404, detail=f"Exchange {exchange} not available")
    except Exception as e:
        logger.error(f"Failed to get price: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orderbook/{symbol}")
async def get_orderbook(
    symbol: str,
    exchange: Optional[str] = Query("binance", description="Exchange name"),
    limit: Optional[int] = Query(20, description="Number of price levels"),
    gordon=Depends(get_gordon)
):
    """Get order book for a symbol."""
    try:
        if exchange in gordon.exchanges:
            exchange_adapter = gordon.exchanges[exchange]
            await exchange_adapter.initialize()
            orderbook = await exchange_adapter.get_orderbook(symbol, limit=limit)
            
            if orderbook:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "bids": orderbook.get('bids', []),
                    "asks": orderbook.get('asks', []),
                    "timestamp": orderbook.get('timestamp')
                }
            else:
                raise HTTPException(status_code=404, detail=f"Order book not available for {symbol}")
        else:
            raise HTTPException(status_code=404, detail=f"Exchange {exchange} not available")
    except Exception as e:
        logger.error(f"Failed to get orderbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ticker/{symbol}")
async def get_ticker(
    symbol: str,
    exchange: Optional[str] = Query("binance", description="Exchange name"),
    gordon=Depends(get_gordon)
):
    """Get full ticker data for a symbol."""
    try:
        if exchange in gordon.exchanges:
            exchange_adapter = gordon.exchanges[exchange]
            await exchange_adapter.initialize()
            ticker = await exchange_adapter.get_ticker(symbol)
            
            if ticker:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "ticker": ticker
                }
            else:
                raise HTTPException(status_code=404, detail=f"Ticker not available for {symbol}")
        else:
            raise HTTPException(status_code=404, detail=f"Exchange {exchange} not available")
    except Exception as e:
        logger.error(f"Failed to get ticker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

