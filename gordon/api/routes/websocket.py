"""
API Routes for WebSocket Management
===================================
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging

from .. import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/websocket", tags=["websocket"])


class StreamSubscription(BaseModel):
    """Stream subscription model."""
    exchange: str
    stream_type: str  # 'liquidations', 'trades', 'funding', etc.
    symbols: Optional[List[str]] = None


@router.get("/status")
async def get_websocket_status(websocket_manager=Depends(get_websocket_manager)):
    """Get WebSocket connection status."""
    try:
        status = websocket_manager.get_connection_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get WebSocket status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subscribe")
async def subscribe_stream(
    subscription: StreamSubscription,
    websocket_manager=Depends(get_websocket_manager)
):
    """Subscribe to a WebSocket stream."""
    try:
        await websocket_manager.connect_stream(
            subscription.exchange,
            subscription.stream_type,
            subscription.symbols
        )
        return {
            "success": True,
            "message": f"Subscribed to {subscription.exchange} {subscription.stream_type}",
            "subscription": subscription.dict()
        }
    except Exception as e:
        logger.error(f"Subscription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unsubscribe")
async def unsubscribe_stream(
    connection_id: str = Body(..., description="Connection ID to unsubscribe"),
    websocket_manager=Depends(get_websocket_manager)
):
    """Unsubscribe from a WebSocket stream."""
    try:
        await websocket_manager.disconnect_stream(connection_id)
        return {
            "success": True,
            "message": f"Unsubscribed from {connection_id}",
            "connection_id": connection_id
        }
    except Exception as e:
        logger.error(f"Unsubscription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

