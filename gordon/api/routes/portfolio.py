"""
API Routes for Portfolio Management
===================================
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .. import get_portfolio_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


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
            "total_count": len(positions),
            "total_value": sum(p.get('value_usd', 0) for p in positions)
        }
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_portfolio_summary(portfolio_manager=Depends(get_portfolio_manager)):
    """Get portfolio summary."""
    try:
        positions = await portfolio_manager.get_all_positions()
        
        total_value = sum(p.get('value_usd', 0) for p in positions)
        total_pnl = sum(p.get('pnl_usd', 0) for p in positions)
        
        return {
            "total_positions": len(positions),
            "total_value_usd": total_value,
            "total_pnl_usd": total_pnl,
            "total_pnl_percent": (total_pnl / total_value * 100) if total_value > 0 else 0,
            "positions_by_exchange": {}
        }
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

