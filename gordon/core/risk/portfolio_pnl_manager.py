"""
Portfolio-Level PnL Management
===============================
Day 26: Close all positions when total portfolio value exceeds thresholds.

Features:
- Monitor total portfolio value across all positions
- Close all positions when total value < LOWEST_BALANCE (loss protection)
- Close all positions when total value > TARGET_BALANCE (profit taking)
- Batch position closing with chunking
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class PortfolioPnLManager:
    """
    Portfolio-level PnL management.
    
    Monitors total portfolio value and closes all positions when thresholds are exceeded.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize portfolio PnL manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.lowest_balance = self.config.get('lowest_balance', 1400)  # USD
        self.target_balance = self.config.get('target_balance', 3000)  # USD
        self.min_position_value = self.config.get('min_position_value', 2)  # USD
        self.excluded_symbols = self.config.get('do_not_trade_list', [])

    async def get_portfolio_value(
        self,
        exchange: str,
        position_manager
    ) -> Tuple[float, List[Dict]]:
        """
        Get total portfolio value and all positions.
        
        Args:
            exchange: Exchange name
            position_manager: Position manager instance
            
        Returns:
            Tuple of (total_value, positions_list)
        """
        try:
            # Get all positions
            all_positions = await position_manager.get_all_positions(exchange)
            
            if not all_positions:
                return 0.0, []
            
            # Calculate total value
            total_value = 0.0
            positions_list = []
            
            for pos in all_positions:
                symbol = pos.get('symbol', '')
                usd_value = pos.get('usd_value', 0.0) or 0.0
                
                # Filter out excluded symbols and small positions
                if symbol not in self.excluded_symbols and usd_value > self.min_position_value:
                    total_value += usd_value
                    positions_list.append({
                        'symbol': symbol,
                        'usd_value': usd_value,
                        'size': pos.get('size', 0),
                        'entry_price': pos.get('entry_price', 0),
                        'current_price': pos.get('current_price', 0)
                    })
            
            return total_value, positions_list
            
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0, []

    async def check_and_close_on_thresholds(
        self,
        exchange: str,
        position_manager,
        chunk_closer
    ) -> Dict[str, any]:
        """
        Check portfolio value and close positions if thresholds exceeded.
        
        Args:
            exchange: Exchange name
            position_manager: Position manager instance
            chunk_closer: Function to close positions in chunks
            
        Returns:
            Dictionary with action taken and results
        """
        total_value, positions = await self.get_portfolio_value(exchange, position_manager)
        
        result = {
            'total_value': total_value,
            'lowest_balance': self.lowest_balance,
            'target_balance': self.target_balance,
            'action': None,
            'positions_closed': []
        }
        
        # Check thresholds
        if total_value < self.lowest_balance:
            logger.warning(
                f"Portfolio value ${total_value:.2f} < lowest balance ${self.lowest_balance}. "
                "Closing all positions."
            )
            result['action'] = 'loss_protection'
            
            # Close all positions
            for pos in positions:
                try:
                    await chunk_closer(
                        exchange,
                        pos['symbol'],
                        pos['usd_value']
                    )
                    result['positions_closed'].append({
                        'symbol': pos['symbol'],
                        'value': pos['usd_value'],
                        'reason': 'loss_protection'
                    })
                except Exception as e:
                    logger.error(f"Error closing {pos['symbol']}: {e}")
            
        elif total_value > self.target_balance:
            logger.info(
                f"Portfolio value ${total_value:.2f} > target balance ${self.target_balance}. "
                "Taking profits by closing all positions."
            )
            result['action'] = 'profit_taking'
            
            # Close all positions
            for pos in positions:
                try:
                    await chunk_closer(
                        exchange,
                        pos['symbol'],
                        pos['usd_value']
                    )
                    result['positions_closed'].append({
                        'symbol': pos['symbol'],
                        'value': pos['usd_value'],
                        'reason': 'profit_taking'
                    })
                except Exception as e:
                    logger.error(f"Error closing {pos['symbol']}: {e}")
        else:
            logger.debug(
                f"Portfolio value ${total_value:.2f} within thresholds "
                f"({self.lowest_balance} - {self.target_balance}). No action."
            )
            result['action'] = 'no_action'
        
        return result

    def should_close_portfolio(self, total_value: float) -> Tuple[bool, Optional[str]]:
        """
        Check if portfolio should be closed.
        
        Args:
            total_value: Total portfolio value in USD
            
        Returns:
            Tuple of (should_close, reason)
        """
        if total_value < self.lowest_balance:
            return True, 'loss_protection'
        elif total_value > self.target_balance:
            return True, 'profit_taking'
        return False, None

