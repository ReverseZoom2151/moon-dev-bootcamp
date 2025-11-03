"""
PnL-Based Position Manager
===========================
Day 48: Close positions based on profit/loss thresholds.

Features:
- Track entry prices for positions
- Calculate PnL percentage
- Close positions based on profit/loss thresholds
- Support for both profit targets and stop losses
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PnLPositionManager:
    """
    Manages positions with PnL-based closing logic.
    
    Tracks entry prices and automatically closes positions
    when profit/loss thresholds are reached.
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None,
        positions_file: Optional[str] = None
    ):
        """
        Initialize PnL position manager.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
            positions_file: Path to JSON file for storing position data
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Default thresholds
        self.default_take_profit_pct = self.config.get('take_profit_pct', 10.0)  # 10%
        self.default_stop_loss_pct = self.config.get('stop_loss_pct', 5.0)  # 5%
        
        # Positions tracking file
        self.positions_file = Path(positions_file or './data/pnl_positions.json')
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing positions
        self.positions = self._load_positions()
    
    def _load_positions(self) -> Dict[str, Dict]:
        """Load positions from file."""
        if not self.positions_file.exists():
            return {}
        
        try:
            with open(self.positions_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            return {}
    
    def _save_positions(self):
        """Save positions to file."""
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(self.positions, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def track_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        entry_time: Optional[datetime] = None
    ):
        """
        Track a new position.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            quantity: Position quantity (positive for long, negative for short)
            take_profit_pct: Take profit percentage (override default)
            stop_loss_pct: Stop loss percentage (override default)
            entry_time: Entry timestamp
        """
        position_key = symbol.upper()
        
        self.positions[position_key] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': (entry_time or datetime.now()).isoformat(),
            'take_profit_pct': take_profit_pct or self.default_take_profit_pct,
            'stop_loss_pct': stop_loss_pct or self.default_stop_loss_pct,
            'is_long': quantity > 0
        }
        
        self._save_positions()
        logger.info(f"Tracking position: {symbol} @ {entry_price} (Qty: {quantity})")
    
    def update_position_entry(
        self,
        symbol: str,
        additional_quantity: float,
        average_price: float
    ):
        """
        Update position with additional entry (averaging in).
        
        Args:
            symbol: Trading symbol
            additional_quantity: Additional quantity added
            average_price: New average entry price
        """
        position_key = symbol.upper()
        
        if position_key not in self.positions:
            logger.warning(f"Position {symbol} not tracked. Creating new entry.")
            self.track_position(symbol, average_price, additional_quantity)
            return
        
        position = self.positions[position_key]
        old_quantity = position['quantity']
        old_price = position['entry_price']
        
        # Calculate new average entry price
        old_value = abs(old_quantity) * old_price
        new_value = abs(additional_quantity) * average_price
        total_quantity = old_quantity + additional_quantity
        
        if total_quantity != 0:
            new_avg_price = (old_value + new_value) / abs(total_quantity)
        else:
            new_avg_price = average_price
        
        position['quantity'] = total_quantity
        position['entry_price'] = new_avg_price
        position['is_long'] = total_quantity > 0
        
        self._save_positions()
        logger.info(f"Updated position: {symbol} - New avg: {new_avg_price}, Qty: {total_quantity}")
    
    def check_and_close_position(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Dict]:
        """
        Check if position should be closed based on PnL thresholds.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Dict with close info if threshold reached, None otherwise
        """
        position_key = symbol.upper()
        
        if position_key not in self.positions:
            return None
        
        position = self.positions[position_key]
        entry_price = position['entry_price']
        quantity = position['quantity']
        take_profit_pct = position['take_profit_pct']
        stop_loss_pct = position['stop_loss_pct']
        is_long = position['is_long']
        
        # Calculate PnL percentage
        if is_long:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Check thresholds
        should_close = False
        reason = None
        
        if pnl_pct >= take_profit_pct:
            should_close = True
            reason = f"Take profit reached: {pnl_pct:.2f}% >= {take_profit_pct:.2f}%"
        elif pnl_pct <= -stop_loss_pct:
            should_close = True
            reason = f"Stop loss triggered: {pnl_pct:.2f}% <= -{stop_loss_pct:.2f}%"
        
        if should_close:
            return {
                'symbol': symbol,
                'entry_price': entry_price,
                'current_price': current_price,
                'quantity': quantity,
                'pnl_pct': pnl_pct,
                'reason': reason,
                'should_close': True
            }
        
        return {
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': current_price,
            'quantity': quantity,
            'pnl_pct': pnl_pct,
            'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct,
            'should_close': False
        }
    
    async def close_position(self, symbol: str) -> bool:
        """
        Close a tracked position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Success status
        """
        position_key = symbol.upper()
        
        if position_key not in self.positions:
            logger.warning(f"Position {symbol} not tracked")
            return False
        
        position = self.positions[position_key]
        quantity = position['quantity']
        
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return False
        
        try:
            # Close position using exchange adapter
            if quantity > 0:
                # Long position - sell
                result = await self.exchange_adapter.place_order(
                    symbol=symbol,
                    side='sell',
                    amount=abs(quantity),
                    order_type='market'
                )
            else:
                # Short position - buy to close
                result = await self.exchange_adapter.place_order(
                    symbol=symbol,
                    side='buy',
                    amount=abs(quantity),
                    order_type='market'
                )
            
            if result:
                # Remove from tracking
                del self.positions[position_key]
                self._save_positions()
                logger.info(f"Closed position: {symbol}")
                return True
            else:
                logger.error(f"Failed to close position: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    def remove_position(self, symbol: str):
        """Remove position from tracking."""
        position_key = symbol.upper()
        if position_key in self.positions:
            del self.positions[position_key]
            self._save_positions()
            logger.info(f"Removed position tracking: {symbol}")
    
    def get_position_pnl(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Get current PnL for a position.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Dict with PnL information
        """
        position_key = symbol.upper()
        
        if position_key not in self.positions:
            return None
        
        position = self.positions[position_key]
        entry_price = position['entry_price']
        quantity = position['quantity']
        is_long = position['is_long']
        
        # Calculate PnL
        if is_long:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            pnl_usd = (current_price - entry_price) * quantity
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            pnl_usd = (entry_price - current_price) * abs(quantity)
        
        return {
            'symbol': symbol,
            'entry_price': entry_price,
            'current_price': current_price,
            'quantity': quantity,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'position_value': abs(quantity) * current_price
        }
    
    def get_all_positions(self) -> List[Dict]:
        """Get all tracked positions."""
        return list(self.positions.values())

