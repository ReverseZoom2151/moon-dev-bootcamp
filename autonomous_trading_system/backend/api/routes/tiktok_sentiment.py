"""
TikTok Sentiment Analysis API Routes
Provides REST endpoints for TikTok sentiment analysis functionality
"""

import logging
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from services.tiktok_sentiment_service import TikTokSentimentService
from core.config import get_settings

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/tiktok-sentiment", tags=["TikTok Sentiment"])

# Global service instance
tiktok_service: Optional[TikTokSentimentService] = None

def get_tiktok_service() -> TikTokSentimentService:
    """Get or create TikTok sentiment service instance"""
    global tiktok_service
    if tiktok_service is None:
        settings = get_settings()
        tiktok_service = TikTokSentimentService(settings)
    return tiktok_service

@router.get("/status")
async def get_service_status() -> Dict[str, Any]:
    """
    Get TikTok sentiment analysis service status
    
    Returns:
        Service status including configuration and statistics
    """
    try:
        service = get_tiktok_service()
        status = service.get_service_status()
        
        return {
            "success": True,
            "data": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting TikTok service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_service(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Start the TikTok sentiment analysis service
    
    Returns:
        Success confirmation
    """
    try:
        service = get_tiktok_service()
        
        if service.is_running:
            return {
                "success": True,
                "message": "TikTok sentiment service is already running",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Start service in background
        background_tasks.add_task(service.start)
        
        return {
            "success": True,
            "message": "TikTok sentiment service started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting TikTok service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_service() -> Dict[str, Any]:
    """
    Stop the TikTok sentiment analysis service
    
    Returns:
        Success confirmation
    """
    try:
        service = get_tiktok_service()
        await service.stop()
        
        return {
            "success": True,
            "message": "TikTok sentiment service stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping TikTok service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/recent")
async def get_recent_sentiment(
    hours: int = Query(24, description="Hours of recent data to retrieve", ge=1, le=168)
) -> Dict[str, Any]:
    """
    Get recent TikTok sentiment data
    
    Args:
        hours: Number of hours of recent data to retrieve (1-168)
    
    Returns:
        List of recent sentiment data points
    """
    try:
        service = get_tiktok_service()
        sentiment_data = await service.get_recent_sentiment(hours=hours)
        
        # Convert to serializable format
        serialized_data = []
        for data in sentiment_data:
            serialized_data.append({
                "platform": data.platform,
                "symbol": data.symbol,
                "sentiment_score": data.sentiment_score,
                "confidence": data.confidence,
                "mention_count": data.mention_count,
                "engagement_score": data.engagement_score,
                "timestamp": data.timestamp.isoformat(),
                "raw_text": data.raw_text,
                "metadata": data.metadata
            })
        
        return {
            "success": True,
            "data": {
                "sentiment_data": serialized_data,
                "total_records": len(serialized_data),
                "hours_requested": hours,
                "data_range": {
                    "from": (datetime.utcnow() - timedelta(hours=hours)).isoformat(),
                    "to": datetime.utcnow().isoformat()
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recent sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/symbol/{symbol}")
async def get_symbol_sentiment(
    symbol: str,
    hours: int = Query(24, description="Hours of data to analyze", ge=1, le=168)
) -> Dict[str, Any]:
    """
    Get sentiment data for a specific cryptocurrency symbol
    
    Args:
        symbol: Cryptocurrency symbol (e.g., BTC, ETH, SOL)
        hours: Number of hours of data to analyze
    
    Returns:
        Aggregated sentiment data for the symbol
    """
    try:
        service = get_tiktok_service()
        all_sentiment_data = await service.get_recent_sentiment(hours=hours)
        
        # Filter by symbol
        symbol_data = [data for data in all_sentiment_data if data.symbol.upper() == symbol.upper()]
        
        if not symbol_data:
            return {
                "success": True,
                "data": {
                    "symbol": symbol.upper(),
                    "message": "No sentiment data found for this symbol",
                    "total_mentions": 0
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Calculate aggregated metrics
        total_mentions = len(symbol_data)
        avg_sentiment = sum(data.sentiment_score for data in symbol_data) / total_mentions
        avg_confidence = sum(data.confidence for data in symbol_data) / total_mentions
        avg_engagement = sum(data.engagement_score for data in symbol_data) / total_mentions
        
        # Get recent mentions
        recent_mentions = []
        for data in symbol_data[-10:]:  # Last 10 mentions
            recent_mentions.append({
                "timestamp": data.timestamp.isoformat(),
                "sentiment_score": data.sentiment_score,
                "confidence": data.confidence,
                "raw_text": data.raw_text,
                "metadata": data.metadata
            })
        
        return {
            "success": True,
            "data": {
                "symbol": symbol.upper(),
                "total_mentions": total_mentions,
                "avg_sentiment": round(avg_sentiment, 3),
                "avg_confidence": round(avg_confidence, 3),
                "avg_engagement": round(avg_engagement, 3),
                "sentiment_trend": "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral",
                "recent_mentions": recent_mentions,
                "analysis_period_hours": hours
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting symbol sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending")
async def get_trending_tokens() -> Dict[str, Any]:
    """
    Get trending cryptocurrency tokens from TikTok analysis
    
    Returns:
        List of trending tokens with sentiment metrics
    """
    try:
        service = get_tiktok_service()
        trending_tokens = await service.get_trending_tokens()
        
        # Convert timestamps to ISO format
        for token in trending_tokens:
            if 'first_seen' in token:
                token['first_seen'] = token['first_seen'].isoformat()
            if 'last_seen' in token:
                token['last_seen'] = token['last_seen'].isoformat()
        
        return {
            "success": True,
            "data": {
                "trending_tokens": trending_tokens,
                "total_trending": len(trending_tokens),
                "analysis_note": "Tokens with 2+ mentions, ranked by trending score"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting trending tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/summary")
async def get_analytics_summary(
    hours: int = Query(24, description="Hours of data to analyze", ge=1, le=168)
) -> Dict[str, Any]:
    """
    Get comprehensive analytics summary of TikTok sentiment data
    
    Args:
        hours: Number of hours of data to analyze
    
    Returns:
        Analytics summary including top symbols, sentiment distribution, etc.
    """
    try:
        service = get_tiktok_service()
        sentiment_data = await service.get_recent_sentiment(hours=hours)
        
        if not sentiment_data:
            return {
                "success": True,
                "data": {
                    "message": "No sentiment data available for analysis",
                    "total_records": 0
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Analyze data
        symbol_counts = {}
        sentiment_scores = []
        confidence_scores = []
        engagement_scores = []
        
        for data in sentiment_data:
            symbol = data.symbol
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            sentiment_scores.append(data.sentiment_score)
            confidence_scores.append(data.confidence)
            engagement_scores.append(data.engagement_score)
        
        # Calculate statistics
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        avg_engagement = sum(engagement_scores) / len(engagement_scores)
        
        # Top mentioned symbols
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Sentiment distribution
        bullish_count = sum(1 for score in sentiment_scores if score > 0.1)
        bearish_count = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_count = len(sentiment_scores) - bullish_count - bearish_count
        
        return {
            "success": True,
            "data": {
                "summary": {
                    "total_mentions": len(sentiment_data),
                    "unique_symbols": len(symbol_counts),
                    "avg_sentiment": round(avg_sentiment, 3),
                    "avg_confidence": round(avg_confidence, 3),
                    "avg_engagement": round(avg_engagement, 3),
                    "analysis_period_hours": hours
                },
                "sentiment_distribution": {
                    "bullish": bullish_count,
                    "bearish": bearish_count,
                    "neutral": neutral_count,
                    "bullish_percentage": round((bullish_count / len(sentiment_scores)) * 100, 1),
                    "bearish_percentage": round((bearish_count / len(sentiment_scores)) * 100, 1),
                    "neutral_percentage": round((neutral_count / len(sentiment_scores)) * 100, 1)
                },
                "top_mentioned_symbols": [
                    {"symbol": symbol, "mentions": count} 
                    for symbol, count in top_symbols
                ],
                "market_mood": (
                    "Very Bullish" if avg_sentiment > 0.3 else
                    "Bullish" if avg_sentiment > 0.1 else
                    "Very Bearish" if avg_sentiment < -0.3 else
                    "Bearish" if avg_sentiment < -0.1 else
                    "Neutral"
                )
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_service_config() -> Dict[str, Any]:
    """
    Get TikTok sentiment service configuration
    
    Returns:
        Service configuration settings
    """
    try:
        service = get_tiktok_service()
        settings = service.settings
        
        config_data = {
            "service_enabled": service.enabled,
            "max_videos_per_session": service.max_videos_per_session,
            "analysis_interval_minutes": service.analysis_interval // 60,
            "screenshot_enabled": service.screenshot_enabled,
            "headless_mode": service.headless_mode,
            "ai_model": service.ai_model,
            "ai_max_tokens": service.ai_max_tokens,
            "data_directory": str(service.data_dir),
            "crypto_keywords_count": len(service.crypto_keywords),
            "token_patterns_count": len(service.token_patterns)
        }
        
        return {
            "success": True,
            "data": config_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting service config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for TikTok sentiment service
    
    Returns:
        Health status and dependency checks
    """
    try:
        service = get_tiktok_service()
        status = service.get_service_status()
        
        # Determine overall health
        dependencies = status.get('dependencies', {})
        is_healthy = (
            dependencies.get('selenium', False) and
            (dependencies.get('openai', False) or dependencies.get('textblob', False))
        )
        
        health_status = "healthy" if is_healthy else "degraded"
        
        return {
            "success": True,
            "data": {
                "status": health_status,
                "service_running": status.get('is_running', False),
                "dependencies": dependencies,
                "last_analysis": status.get('session_stats', {}).get('last_analysis'),
                "total_videos_analyzed": status.get('session_stats', {}).get('videos_analyzed', 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "success": False,
            "data": {
                "status": "unhealthy",
                "error": str(e)
            },
            "timestamp": datetime.utcnow().isoformat()
        }

# Export router
__all__ = ['router']