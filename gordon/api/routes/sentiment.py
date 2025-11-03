"""
API Routes for Social Sentiment
================================
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
import logging

from .. import get_gordon

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


@router.post("/analyze")
async def analyze_sentiment(
    symbol: str = Body(..., description="Trading symbol"),
    query: Optional[str] = Body(None, description="Search query"),
    method: str = Body("vader", description="Sentiment method (vader, textblob, openai)"),
    gordon=Depends(get_gordon)
):
    """Analyze sentiment for a symbol."""
    try:
        search_query = query or symbol
        result = await gordon.run(f"Analyze sentiment for {symbol} using {method}")
        return {
            "success": True,
            "symbol": symbol,
            "query": search_query,
            "method": method,
            "result": result
        }
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/twitter/{symbol}")
async def get_twitter_sentiment(
    symbol: str,
    limit: int = Query(100, description="Number of tweets"),
    gordon=Depends(get_gordon)
):
    """Get Twitter sentiment for a symbol."""
    try:
        result = await gordon.run(f"Get Twitter sentiment for {symbol} limit {limit}")
        return {
            "success": True,
            "symbol": symbol,
            "limit": limit,
            "result": result
        }
    except Exception as e:
        logger.error(f"Twitter sentiment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

