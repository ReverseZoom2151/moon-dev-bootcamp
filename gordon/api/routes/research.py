"""
API Routes for Research & Analysis
===================================
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging

from .. import get_gordon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])


class ResearchRequest(BaseModel):
    """Research request model."""
    query: str
    symbol: Optional[str] = None
    include_trading_signals: bool = False


@router.post("/analyze")
async def analyze_stock(
    request: ResearchRequest,
    gordon=Depends(get_gordon)
):
    """Analyze a stock or company."""
    try:
        query = request.query
        if request.symbol:
            query = f"{query} for {request.symbol}"
        
        result = await gordon.research(query)
        return {
            "success": True,
            "query": request.query,
            "symbol": request.symbol,
            "result": result
        }
    except Exception as e:
        logger.error(f"Research analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid")
async def hybrid_analysis(
    symbol: str = Body(..., description="Stock symbol"),
    include_sentiment: bool = Body(True, description="Include sentiment analysis"),
    gordon=Depends(get_gordon)
):
    """Perform hybrid analysis (fundamental + technical + sentiment)."""
    try:
        result = await gordon.research_and_trade(f"Analyze {symbol} with trading signals")
        return {
            "success": True,
            "symbol": symbol,
            "include_sentiment": include_sentiment,
            "result": result
        }
    except Exception as e:
        logger.error(f"Hybrid analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

