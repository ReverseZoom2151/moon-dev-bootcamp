"""
API Routes for ML Indicators
=============================
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
import logging

from .. import get_gordon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.get("/indicators/discover")
async def discover_indicators(gordon=Depends(get_gordon)):
    """Discover available ML indicators."""
    try:
        result = await gordon.handle_ml_indicators("Discover indicators")
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Indicator discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/indicators/evaluate")
async def evaluate_indicators(
    symbol: str = Body(..., description="Trading symbol"),
    gordon=Depends(get_gordon)
):
    """Evaluate indicator performance."""
    try:
        result = await gordon.handle_ml_indicators(f"Evaluate indicators for {symbol}")
        return {
            "success": True,
            "symbol": symbol,
            "result": result
        }
    except Exception as e:
        logger.error(f"Indicator evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/top")
async def get_top_indicators(
    n: int = Query(10, description="Number of top indicators"),
    gordon=Depends(get_gordon)
):
    """Get top N indicators."""
    try:
        result = await gordon.handle_ml_indicators(f"Show top {n} indicators")
        return {
            "success": True,
            "top_n": n,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to get top indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/indicators/loop")
async def loop_indicators(
    symbol: str = Body(..., description="Trading symbol"),
    generations: int = Body(5, description="Number of generations"),
    gordon=Depends(get_gordon)
):
    """Run indicator looping (multi-generation evaluation)."""
    try:
        result = await gordon.handle_ml_indicators(f"Loop indicators for {symbol} {generations} generations")
        return {
            "success": True,
            "symbol": symbol,
            "generations": generations,
            "result": result
        }
    except Exception as e:
        logger.error(f"Indicator looping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

