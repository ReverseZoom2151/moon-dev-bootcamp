"""
API Routes for Market Dashboard
================================
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional, Any
import logging

from .. import get_gordon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/trending")
async def get_trending_tokens(
    exchange: str = Query("binance", description="Exchange name"),
    limit: int = Query(20, description="Number of tokens"),
    gordon=Depends(get_gordon)
):
    """Get trending tokens."""
    try:
        result = await gordon.handle_market_dashboard(f"Trending tokens {exchange} limit {limit}")
        return {
            "success": True,
            "exchange": exchange,
            "limit": limit,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to get trending tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/new-listings")
async def get_new_listings(
    exchange: str = Query("binance", description="Exchange name"),
    limit: int = Query(20, description="Number of listings"),
    gordon=Depends(get_gordon)
):
    """Get new token listings."""
    try:
        result = await gordon.handle_market_dashboard(f"New listings {exchange} limit {limit}")
        return {
            "success": True,
            "exchange": exchange,
            "limit": limit,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to get new listings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/volume-leaders")
async def get_volume_leaders(
    exchange: str = Query("binance", description="Exchange name"),
    limit: int = Query(20, description="Number of tokens"),
    gordon=Depends(get_gordon)
):
    """Get volume leaders."""
    try:
        result = await gordon.handle_market_dashboard(f"Volume leaders {exchange} limit {limit}")
        return {
            "success": True,
            "exchange": exchange,
            "limit": limit,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to get volume leaders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/funding-rates")
async def get_funding_rates(
    exchange: str = Query("bitfinex", description="Exchange name"),
    gordon=Depends(get_gordon)
):
    """Get funding rates."""
    try:
        result = await gordon.handle_market_dashboard(f"Funding rates {exchange}")
        return {
            "success": True,
            "exchange": exchange,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to get funding rates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/full-analysis")
async def get_full_analysis(
    exchange: str = Query("binance", description="Exchange name"),
    gordon=Depends(get_gordon)
):
    """Get full market dashboard analysis."""
    try:
        result = await gordon.handle_market_dashboard(f"Market dashboard {exchange}")
        return {
            "success": True,
            "exchange": exchange,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to get full analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

