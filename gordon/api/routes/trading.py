"""
API Routes for Trading Operations
==================================
"""

from fastapi import APIRouter, HTTPException, Depends, Body, Query
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging

from .. import get_gordon, get_portfolio_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading", tags=["trading"])


class TradeRequest(BaseModel):
    """Trade request model."""
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    order_type: str = "market"
    price: Optional[float] = None
    exchange: Optional[str] = "binance"


@router.post("/execute")
async def execute_trade(
    trade: TradeRequest,
    gordon=Depends(get_gordon)
):
    """Execute a trade."""
    try:
        # This would call Gordon's trading functionality
        # For now, return a placeholder
        return {
            "success": True,
            "symbol": trade.symbol,
            "side": trade.side,
            "amount": trade.amount,
            "order_id": "placeholder_order_id"
        }
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    portfolio_manager=Depends(get_portfolio_manager)
):
    """Get all positions."""
    try:
        positions = await portfolio_manager.get_all_positions(exchange=exchange)
        return {
            "positions": positions,
            "total_count": len(positions)
        }
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-buy")
async def quick_buy(
    symbol: str = Body(..., description="Trading symbol"),
    usd_amount: Optional[float] = Body(None, description="USD amount"),
    gordon=Depends(get_gordon)
):
    """Execute quick buy (Day 51)."""
    try:
        result = await gordon.handle_quick_buysell(f"Quick buy {symbol} for ${usd_amount or 10}")
        return {
            "success": True,
            "message": result,
            "symbol": symbol,
            "amount": usd_amount
        }
    except Exception as e:
        logger.error(f"Quick buy failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-sell")
async def quick_sell(
    symbol: str = Body(..., description="Trading symbol"),
    gordon=Depends(get_gordon)
):
    """Execute quick sell (Day 51)."""
    try:
        result = await gordon.handle_quick_buysell(f"Quick sell {symbol}")
        return {
            "success": True,
            "message": result,
            "symbol": symbol
        }
    except Exception as e:
        logger.error(f"Quick sell failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

