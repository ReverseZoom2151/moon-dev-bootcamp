"""
Data API Routes - Market data and information endpoints
"""

from fastapi import APIRouter, HTTPException, Request
from datetime import datetime
from data.market_data_manager import MarketDataManager
from services.indicator_service import IndicatorService

router = APIRouter()

async def get_market_data_manager() -> MarketDataManager:
    """Dependency to get market data manager"""
    # This will be injected by the main app
    pass


@router.get("/symbols")
async def get_supported_symbols():
    """Get list of supported trading symbols"""
    try:
        # For now, return a static list
        symbols = [
            "BTC", "ETH", "SOL", "WIF", "POPCAT", "LINK", "ADA", 
            "DOT", "XRP", "LTC", "BCH", "AVAX", "MATIC", "UNI"
        ]
        return {
            "symbols": symbols,
            "total_count": len(symbols)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price/{symbol}")
async def get_current_price(symbol: str):
    """Get current price for a symbol"""
    try:
        # Simulate current price
        base_prices = {
            "BTC": 45000,
            "ETH": 3000,
            "SOL": 100,
            "WIF": 2.5,
            "POPCAT": 1.2,
            "LINK": 15,
            "ADA": 0.5,
            "DOT": 8,
            "XRP": 0.6,
            "LTC": 80
        }
        
        price = base_prices.get(symbol.upper(), 100)
        
        return {
            "symbol": symbol.upper(),
            "price": price,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ohlcv/{symbol}")
async def get_ohlcv_data(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100
):
    """Get OHLCV data for a symbol"""
    try:
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "limit": limit,
            "data": [],  # Would be populated with real data
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-summary")
async def get_market_summary():
    """Get market summary for all symbols"""
    try:
        symbols = ["BTC", "ETH", "SOL", "WIF", "POPCAT"]
        summary = {}
        
        base_prices = {
            "BTC": 45000,
            "ETH": 3000,
            "SOL": 100,
            "WIF": 2.5,
            "POPCAT": 1.2
        }
        
        for symbol in symbols:
            price = base_prices.get(symbol, 100)
            summary[symbol] = {
                "current_price": price,
                "price_change_24h": price * 0.02,  # 2% change
                "price_change_pct_24h": 2.0,
                "volume_24h": 1000000,
                "high_24h": price * 1.05,
                "low_24h": price * 0.95
            }
        
        return {
            "market_summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{symbol}/{timeframe}/{indicator}")
async def get_indicator(request: Request, symbol: str, timeframe: str, indicator: str):
    """Get latest indicator series for a symbol and timeframe."""
    indicator_service: IndicatorService = request.app.state.indicator_service
    if not indicator_service:
        raise HTTPException(status_code=404, detail="Indicator service not running")
    series = indicator_service.get(symbol.upper(), timeframe, indicator)
    if series is None:
        raise HTTPException(status_code=404, detail=f"Indicator {indicator} not found for {symbol}@{timeframe}")
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "indicator": indicator,
        "values": series.tolist(),
        "last": series.iloc[-1] if not series.empty else None
    } 