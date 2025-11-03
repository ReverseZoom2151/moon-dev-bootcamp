"""
Database Event Listener
========================
Automatically saves events to database when they occur.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .manager import DatabaseManager
from .models import Trade, Position

logger = logging.getLogger(__name__)


class DatabaseEventListener:
    """
    Listens to EventBus events and automatically saves them to database.
    
    Similar to how ConversationMemory updates on each interaction,
    this automatically persists trading data.
    """
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize database event listener.
        
        Args:
            database_manager: DatabaseManager instance
        """
        self.db = database_manager
        self.logger = logging.getLogger(__name__)
    
    def on_trade_executed(self, event: Dict[str, Any]):
        """Handle trade_executed event."""
        try:
            data = event.get('data', {})
            
            trade_data = {
                'trade_id': data.get('trade_id') or data.get('id', f"trade_{datetime.now().timestamp()}"),
                'exchange': data.get('exchange', 'unknown'),
                'symbol': data.get('symbol', ''),
                'side': data.get('side', 'buy').lower(),
                'order_type': data.get('type', 'market'),
                'amount': float(data.get('amount', 0)),
                'price': float(data.get('price', 0)),
                'fee': float(data.get('fee', 0)),
                'fee_currency': data.get('fee_currency', 'USD'),
                'usd_value': float(data.get('usd_value', 0) or data.get('cost', 0)),
                'strategy_name': data.get('strategy_name'),
                'signal_id': data.get('signal_id'),
                'confidence': float(data.get('confidence', 0.0)),
                'metadata_json': data.get('metadata', {}),
                'timestamp': data.get('timestamp') or datetime.utcnow()
            }
            
            self.db.save_trade(trade_data)
        except Exception as e:
            self.logger.error(f"Error saving trade event: {e}")
    
    def on_order_filled(self, event: Dict[str, Any]):
        """Handle order_filled event (alias for trade_executed)."""
        self.on_trade_executed(event)
    
    def on_position_opened(self, event: Dict[str, Any]):
        """Handle position_opened event."""
        try:
            data = event.get('data', {})
            position = data.get('position', {})
            
            position_data = {
                'exchange': data.get('exchange', 'unknown'),
                'symbol': data.get('symbol', ''),
                'side': position.get('side', 'long').lower(),
                'size': float(position.get('size', 0) or position.get('quantity', 0)),
                'entry_price': float(position.get('entry_price', 0) or position.get('entryPrice', 0)),
                'current_price': float(position.get('current_price', 0) or position.get('price', 0)),
                'leverage': int(position.get('leverage', 1)),
                'usd_value': float(position.get('usd_value', 0) or position.get('value', 0)),
                'margin_used': float(position.get('margin_used', 0) or position.get('collateral', 0)),
                'is_open': True,
                'is_closed': False,
                'strategy_name': data.get('strategy_name'),
                'stop_loss_price': position.get('stop_loss_price'),
                'take_profit_price': position.get('take_profit_price'),
                'stop_loss_pct': position.get('stop_loss_pct'),
                'take_profit_pct': position.get('take_profit_pct'),
                'metadata_json': position.get('metadata', {}),
                'opened_at': data.get('timestamp') or datetime.utcnow()
            }
            
            self.db.save_position(position_data)
        except Exception as e:
            self.logger.error(f"Error saving position_opened event: {e}")
    
    def on_position_updated(self, event: Dict[str, Any]):
        """Handle position_updated event."""
        try:
            data = event.get('data', {})
            position = data.get('position', {})
            
            # Update existing position
            exchange = data.get('exchange', 'unknown')
            symbol = data.get('symbol', '')
            
            position_data = {
                'current_price': float(position.get('current_price', 0) or position.get('price', 0)),
                'unrealized_pnl': float(position.get('unrealized_pnl', 0) or position.get('unrealizedPnl', 0)),
                'unrealized_pnl_pct': float(position.get('pnl_percent', 0) or position.get('percentage', 0)),
                'usd_value': float(position.get('usd_value', 0) or position.get('value', 0)),
                'margin_used': float(position.get('margin_used', 0) or position.get('collateral', 0)),
            }
            
            # Get existing position and update
            session = self.db.get_session()
            try:
                from .models import Position
                existing = session.query(Position).filter(
                    Position.exchange == exchange,
                    Position.symbol == symbol,
                    Position.is_open == True
                ).first()
                
                if existing:
                    for key, value in position_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.updated_at = datetime.utcnow()
                    session.commit()
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error saving position_updated event: {e}")
    
    def on_position_closed(self, event: Dict[str, Any]):
        """Handle position_closed event."""
        try:
            data = event.get('data', {})
            position = data.get('position', {})
            
            exchange = data.get('exchange', 'unknown')
            symbol = data.get('symbol', '')
            
            close_data = {
                'realized_pnl': float(position.get('realized_pnl', 0) or position.get('pnl', 0)),
                'realized_pnl_pct': float(position.get('realized_pnl_pct', 0) or position.get('pnl_percent', 0)),
                'is_open': False,
                'is_closed': True,
                'closed_at': data.get('timestamp') or datetime.utcnow()
            }
            
            self.db.close_position(exchange, symbol, close_data)
        except Exception as e:
            self.logger.error(f"Error saving position_closed event: {e}")
    
    def on_metrics_updated(self, event: Dict[str, Any]):
        """Handle metrics_updated event (strategy metrics)."""
        try:
            data = event.get('data', {})
            strategy_name = data.get('strategy_name')
            
            if not strategy_name:
                return
            
            metrics = data.get('metrics', {})
            
            self.db.save_strategy_metrics(strategy_name, metrics)
        except Exception as e:
            self.logger.error(f"Error saving metrics_updated event: {e}")
    
    def on_pnl_update(self, event: Dict[str, Any]):
        """Handle pnl_update event (risk metrics)."""
        try:
            data = event.get('data', {})
            
            risk_metrics = {
                'total_balance': float(data.get('balance', 0) or data.get('total_balance', 0)),
                'available_balance': float(data.get('available_balance', 0)),
                'used_margin': float(data.get('used_margin', 0)),
                'total_positions': int(data.get('total_positions', 0)),
                'open_positions': int(data.get('open_positions', 0)),
                'total_position_value': float(data.get('total_position_value', 0)),
                'total_pnl': float(data.get('total_pnl', 0)),
                'total_pnl_pct': float(data.get('total_pnl_pct', 0)),
                'daily_pnl': float(data.get('daily_pnl', 0)),
                'daily_pnl_pct': float(data.get('daily_pnl_pct', 0)),
                'current_drawdown': float(data.get('current_drawdown', 0)),
                'current_drawdown_pct': float(data.get('current_drawdown_pct', 0)),
                'max_drawdown': float(data.get('max_drawdown', 0)),
                'max_drawdown_pct': float(data.get('max_drawdown_pct', 0)),
                'peak_balance': float(data.get('peak_balance', 0)) if data.get('peak_balance') else None,
            }
            
            self.db.save_risk_metrics(risk_metrics)
        except Exception as e:
            self.logger.error(f"Error saving pnl_update event: {e}")
    
    def setup_event_handlers(self, event_bus):
        """
        Register all event handlers with EventBus.
        
        Args:
            event_bus: EventBus instance
        """
        event_bus.subscribe('trade_executed', self.on_trade_executed)
        event_bus.subscribe('order_filled', self.on_order_filled)
        event_bus.subscribe('position_opened', self.on_position_opened)
        event_bus.subscribe('position_updated', self.on_position_updated)
        event_bus.subscribe('position_closed', self.on_position_closed)
        event_bus.subscribe('metrics_updated', self.on_metrics_updated)
        event_bus.subscribe('pnl_update', self.on_pnl_update)
        
        self.logger.info("Database event listeners registered")

