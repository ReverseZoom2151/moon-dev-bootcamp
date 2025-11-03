"""
API Routes for Backtesting
===========================
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging

from .. import get_gordon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtesting", tags=["backtesting"])


class BacktestRequest(BaseModel):
    """Backtest request model."""
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    timeframe: Optional[str] = "1d"
    parameters: Optional[Dict[str, Any]] = None


@router.post("/run")
async def run_backtest(
    request: BacktestRequest,
    gordon=Depends(get_gordon)
):
    """Run a backtest."""
    try:
        query = f"Backtest {request.strategy} strategy on {request.symbol} from {request.start_date} to {request.end_date}"
        if request.timeframe:
            query += f" timeframe {request.timeframe}"
        
        result = await gordon.backtest(query)
        return {
            "success": True,
            "strategy": request.strategy,
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "result": result
        }
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_backtest_strategies(gordon=Depends(get_gordon)):
    """List available backtesting strategies."""
    try:
        # Get available strategies from backtester
        strategies = [
            "sma_crossover",
            "rsi",
            "mean_reversion",
            "kalman_breakout",
            "liquidation_lliq",
            "liquidation_short",
            "stochrsi_bollinger",
            "multitimeframe_breakout"
        ]
        return {
            "strategies": strategies,
            "total": len(strategies)
        }
    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_strategy(
    strategy: str = Body(..., description="Strategy name"),
    symbol: str = Body(..., description="Trading symbol"),
    start_date: str = Body(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Body(..., description="End date (YYYY-MM-DD)"),
    parameters: Optional[Dict[str, Any]] = Body(None, description="Parameter ranges to optimize"),
    gordon=Depends(get_gordon)
):
    """Optimize strategy parameters."""
    try:
        query = f"Optimize {strategy} strategy on {symbol} from {start_date} to {end_date}"
        result = await gordon.backtest(query)
        return {
            "success": True,
            "strategy": strategy,
            "symbol": symbol,
            "optimization_result": result
        }
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

