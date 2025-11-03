"""
API Server Entry Point
=======================
FastAPI server for Gordon Trading API.
"""

import uvicorn
import logging
from typing import Optional, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gordon.api import app, initialize_api
from gordon.agent.gordon_agent import GordonAgent
from gordon.core.strategy_manager import StrategyManager
from gordon.core.risk_manager import RiskManager
from gordon.core.position_manager import PositionManager
from gordon.core.websocket_manager import WebSocketManager
from gordon.core.event_bus import EventBus
from gordon.config.config_manager import ConfigManager

# Import WebSocket broadcast functions
from gordon.api.routes.realtime import (
    broadcast_strategy_signal,
    broadcast_position_update,
    broadcast_risk_alert,
    broadcast_trade_execution,
    broadcast_market_data
)

logger = logging.getLogger(__name__)


def create_gordon_components():
    """Create and initialize Gordon components."""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Create event bus
    event_bus = EventBus()
    
    # Initialize database if enabled
    database_manager = None
    database_listener = None
    if config.get('performance', {}).get('save_to_database', False):
        from gordon.core.database import DatabaseManager, DatabaseEventListener
        db_config = config.get('database', {})
        database_manager = DatabaseManager(
            database_url=db_config.get('url'),
            database_dir=db_config.get('directory', './database'),
            database_file=db_config.get('file', 'gordon.db'),
            echo=db_config.get('echo', False)
        )
        database_listener = DatabaseEventListener(database_manager)
        database_listener.setup_event_handlers(event_bus)
        logger.info("Database initialized and event listeners registered")
    
    # Setup event bus subscribers for WebSocket broadcasting
    _setup_websocket_broadcasting(event_bus)
    
    # Create managers
    strategy_manager = StrategyManager(event_bus, config=config)
    risk_manager = RiskManager(event_bus, config_manager, demo_mode=True)
    position_manager = PositionManager(event_bus, config=config)
    
    # Create WebSocket manager (requires market_data_stream)
    # For now, create minimal version
    from gordon.core.market_data_stream import MarketDataStream
    market_data_stream = MarketDataStream(event_bus)
    websocket_manager = WebSocketManager(event_bus, market_data_stream)
    
    # Create Gordon agent
    gordon_agent = GordonAgent(config=config)
    
    return gordon_agent, strategy_manager, risk_manager, position_manager, websocket_manager, database_manager


def _setup_websocket_broadcasting(event_bus: EventBus):
    """Setup event bus subscribers to broadcast to WebSocket clients."""
    async def on_strategy_signal(event: Dict):
        """Handle strategy signal events."""
        await broadcast_strategy_signal(event.get('data', event))
    
    async def on_position_update(event: Dict):
        """Handle position update events."""
        await broadcast_position_update(event.get('data', event))
    
    async def on_risk_alert(event: Dict):
        """Handle risk alert events."""
        await broadcast_risk_alert(event.get('data', event))
    
    async def on_trade_executed(event: Dict):
        """Handle trade executed events."""
        await broadcast_trade_execution(event.get('data', event))
    
    async def on_market_update(event: Dict):
        """Handle market update events."""
        await broadcast_market_data(event.get('data', event))
    
    # Subscribe to events using subscribe() method
    event_bus.subscribe("strategy_signal", on_strategy_signal)
    event_bus.subscribe("position_update", on_position_update)
    event_bus.subscribe("position_opened", on_position_update)
    event_bus.subscribe("position_closed", on_position_update)
    event_bus.subscribe("risk_alert", on_risk_alert)
    event_bus.subscribe("trade_executed", on_trade_executed)
    event_bus.subscribe("order_filled", on_trade_executed)
    event_bus.subscribe("price_update", on_market_update)
    event_bus.subscribe("orderbook_update", on_market_update)


def main():
    """Main entry point for API server."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Gordon Trading API Server...")
    
    # Create Gordon components
    try:
        components = create_gordon_components()
        gordon_agent, strategy_manager, risk_manager, position_manager, websocket_manager, database_manager = components
        
        # Initialize API
        initialize_api(
            gordon_agent,
            strategy_manager,
            risk_manager,
            position_manager,
            websocket_manager
        )
        
        # Store database manager globally if available
        if database_manager:
            from gordon.api import set_database_manager
            set_database_manager(database_manager)
        
        logger.info("Gordon components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Gordon components: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()

