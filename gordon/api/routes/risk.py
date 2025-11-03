"""
API Routes for Risk Management
===============================
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging

from .. import get_risk_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk", tags=["risk"])


class RiskCheckRequest(BaseModel):
    """Risk check request model."""
    exchange: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float] = None


@router.get("/metrics")
async def get_risk_metrics(risk_manager=Depends(get_risk_manager)):
    """Get current risk metrics."""
    try:
        metrics = risk_manager.get_risk_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check")
async def check_trade_risk(
    request: RiskCheckRequest,
    risk_manager=Depends(get_risk_manager)
):
    """Check if a trade passes risk checks."""
    try:
        allowed = await risk_manager.check_trade_allowed(
            request.exchange,
            request.symbol,
            request.side,
            request.amount,
            request.price
        )
        
        metrics = risk_manager.get_risk_metrics()
        
        return {
            "allowed": allowed,
            "risk_score": metrics.get('risk_score', 0),
            "current_drawdown": metrics.get('current_drawdown', 0),
            "max_drawdown": metrics.get('max_drawdown', 0),
            "daily_pnl": metrics.get('daily_pnl', 0)
        }
    except Exception as e:
        logger.error(f"Risk check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/limits")
async def get_risk_limits(risk_manager=Depends(get_risk_manager)):
    """Get current risk limits."""
    try:
        return {
            "max_drawdown": risk_manager.max_drawdown_percent,
            "daily_loss_limit": risk_manager.daily_loss_limit,
            "max_positions": risk_manager.max_positions,
            "max_position_size": risk_manager.max_position_size,
            "risk_per_trade": risk_manager.risk_per_trade_percent
        }
    except Exception as e:
        logger.error(f"Failed to get risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-position-size")
async def calculate_position_size(
    balance: float = Body(..., description="Account balance"),
    stop_loss_percent: float = Body(..., description="Stop loss percentage"),
    risk_manager=Depends(get_risk_manager)
):
    """Calculate position size based on risk."""
    try:
        size = risk_manager.calculate_position_size(balance, stop_loss_percent)
        return {
            "balance": balance,
            "stop_loss_percent": stop_loss_percent,
            "position_size": size,
            "risk_amount": balance * risk_manager.risk_per_trade_percent
        }
    except Exception as e:
        logger.error(f"Position size calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

