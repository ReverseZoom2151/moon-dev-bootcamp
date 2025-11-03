"""
WebSocket Real-time Updates for API Clients
============================================
Provides WebSocket endpoints for real-time updates to clients.
"""

from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from typing import Dict, List, Set
import json
import logging
from datetime import datetime

from .. import get_strategy_engine, get_risk_manager, get_portfolio_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ws", tags=["websocket-realtime"])

# Active WebSocket connections
active_connections: Set[WebSocket] = set()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        message_str = json.dumps(message)
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client (could be subscription requests)
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "subscribe":
                    # Client wants to subscribe to specific events
                    events = message.get("events", [])
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "events": events,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                elif message_type == "ping":
                    # Heartbeat
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON"
                }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Export functions for use by event bus subscribers
async def broadcast_strategy_signal(signal: Dict):
    """Broadcast strategy signal to all connected clients."""
    await manager.broadcast({
        "type": "strategy_signal",
        "data": signal,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_position_update(position: Dict):
    """Broadcast position update to all connected clients."""
    await manager.broadcast({
        "type": "position_update",
        "data": position,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_risk_alert(alert: Dict):
    """Broadcast risk alert to all connected clients."""
    await manager.broadcast({
        "type": "risk_alert",
        "data": alert,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_trade_execution(trade: Dict):
    """Broadcast trade execution to all connected clients."""
    await manager.broadcast({
        "type": "trade_execution",
        "data": trade,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_market_data(market_data: Dict):
    """Broadcast market data update to all connected clients."""
    await manager.broadcast({
        "type": "market_data",
        "data": market_data,
        "timestamp": datetime.now().isoformat()
    })

