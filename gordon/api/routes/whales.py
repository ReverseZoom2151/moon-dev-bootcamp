"""
API Routes for Whale Tracking
==============================
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
import logging

from .. import get_gordon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/whales", tags=["whales"])


@router.get("/track")
async def track_whales(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    min_value: Optional[float] = Query(None, description="Minimum position value in USD"),
    gordon=Depends(get_gordon)
):
    """Track whale positions."""
    try:
        query = "Track whale positions"
        if symbol:
            query += f" for {symbol}"
        if min_value:
            query += f" minimum ${min_value}"
        
        result = await gordon.handle_whale_tracking(query)
        return {
            "success": True,
            "symbol": symbol,
            "min_value": min_value,
            "result": result
        }
    except Exception as e:
        logger.error(f"Whale tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/multi-address")
async def track_multi_address(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    min_value: Optional[float] = Query(None, description="Minimum position value"),
    gordon=Depends(get_gordon)
):
    """Track multiple whale addresses (Day 46)."""
    try:
        query = "Track multiple whale addresses"
        if symbol:
            query += f" for {symbol}"
        if min_value:
            query += f" minimum ${min_value}"
        
        result = await gordon.handle_whale_tracking(query)
        return {
            "success": True,
            "symbol": symbol,
            "min_value": min_value,
            "result": result
        }
    except Exception as e:
        logger.error(f"Multi-address tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/liquidation-risk")
async def get_liquidation_risk(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    threshold: float = Query(3.0, description="Liquidation risk threshold (%)"),
    gordon=Depends(get_gordon)
):
    """Analyze liquidation risk (Day 46)."""
    try:
        query = f"Liquidation risk analysis threshold {threshold}%"
        if symbol:
            query += f" for {symbol}"
        
        result = await gordon.handle_whale_tracking(query)
        return {
            "success": True,
            "symbol": symbol,
            "threshold": threshold,
            "result": result
        }
    except Exception as e:
        logger.error(f"Liquidation risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregate")
async def aggregate_positions(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    gordon=Depends(get_gordon)
):
    """Get aggregated positions report (Day 46)."""
    try:
        query = "Aggregate positions"
        if symbol:
            query += f" for {symbol}"
        
        result = await gordon.handle_whale_tracking(query)
        return {
            "success": True,
            "symbol": symbol,
            "result": result
        }
    except Exception as e:
        logger.error(f"Position aggregation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

