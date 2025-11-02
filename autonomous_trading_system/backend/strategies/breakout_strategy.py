"""
Breakout Strategy
Based on Day_17_Projects gap trading and breakout detection
"""

import logging
import pandas as pd
from typing import Dict
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy based on resistance level breaks
    Inspired by Day 18 implementation
    """
    
    def __init__(
        self,
        lookback_hours: int = 1,
        leverage: float = 3.0,
        resistance_days: int = 20,
        min_volume_threshold: float = 1000000,
        stop_loss_percent: float = 18.0,
        take_profit_percent: float = 3.0
    ):
        super().__init__()
        self.lookback_hours = lookback_hours
        self.leverage = leverage
        self.resistance_days = resistance_days
        self.min_volume_threshold = min_volume_threshold
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.resistance_levels = {}
        
    async def initialize(self, symbol: str):
        """Initialize strategy for a symbol"""
        await super().initialize(symbol)
        await self._calculate_resistance_levels(symbol)
        
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Generate trading signal based on breakout detection"""
        try:
            if len(data) < self.resistance_days:
                return self._create_signal('HOLD', 0, "Insufficient data")
                
            current_price = data['close'].iloc[-1]
            current_volume = data['volume'].iloc[-1]
            
            # Check volume threshold
            if current_volume < self.min_volume_threshold:
                return self._create_signal('HOLD', 0, "Volume below threshold")
                
            # Get resistance level for symbol
            resistance_level = self.resistance_levels.get(symbol)
            if not resistance_level:
                await self._calculate_resistance_levels(symbol, data)
                resistance_level = self.resistance_levels.get(symbol)
                
            if not resistance_level:
                return self._create_signal('HOLD', 0, "No resistance level calculated")
                
            # Check for breakout
            if current_price > resistance_level:
                # Calculate position size based on risk
                position_size = await self._calculate_position_size(
                    symbol, current_price, resistance_level
                )
                
                # Calculate stop loss and take profit
                stop_loss = current_price * (1 - self.stop_loss_percent / 100)
                take_profit = current_price * (1 + self.take_profit_percent / 100)
                
                signal_data = {
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'resistance_level': resistance_level,
                    'volume': current_volume
                }
                
                return self._create_signal('BUY', position_size, "Breakout detected", signal_data)
                
            return self._create_signal('HOLD', 0, f"Price {current_price} below resistance {resistance_level}")
            
        except Exception as e:
            logger.error(f"Error generating breakout signal for {symbol}: {e}")
            return self._create_signal('HOLD', 0, f"Error: {e}")
    
    async def _calculate_resistance_levels(self, symbol: str, data: pd.DataFrame = None):
        """Calculate resistance levels for symbols"""
        try:
            if data is None:
                # Fetch historical data
                data = await self._fetch_historical_data(symbol, self.resistance_days)
                
            if len(data) < self.resistance_days:
                logger.warning(f"Insufficient data for resistance calculation: {symbol}")
                return
                
            # Calculate resistance as the highest high in the lookback period
            high_prices = data['high'].tail(self.resistance_days)
            resistance_level = high_prices.max()
            
            self.resistance_levels[symbol] = resistance_level
            
            logger.info(f"Resistance level for {symbol}: {resistance_level}")
            
        except Exception as e:
            logger.error(f"Error calculating resistance levels for {symbol}: {e}")
    
    async def _calculate_position_size(
        self, 
        symbol: str, 
        current_price: float, 
        resistance_level: float
    ) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            account_balance = await self._get_account_balance()
            
            # Calculate risk amount (2% of account)
            risk_amount = account_balance * 0.02
            
            # Calculate stop loss distance
            stop_loss_price = current_price * (1 - self.stop_loss_percent / 100)
            risk_per_unit = current_price - stop_loss_price
            
            if risk_per_unit <= 0:
                return 0
                
            # Calculate position size
            position_size = risk_amount / risk_per_unit
            
            # Apply leverage
            position_size *= self.leverage
            
            # Ensure position size doesn't exceed maximum
            max_position_value = account_balance * 0.1  # Max 10% of account
            max_position_size = max_position_value / current_price
            
            position_size = min(position_size, max_position_size)
            
            return round(position_size, 6)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def should_exit(self, symbol: str, position: Dict, current_data: pd.DataFrame) -> Dict:
        """Check if position should be exited"""
        try:
            current_price = current_data['close'].iloc[-1]
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss', 0)
            take_profit = position.get('take_profit', 0)
            
            # Check stop loss
            if current_price <= stop_loss:
                return {
                    'action': 'SELL',
                    'reason': 'Stop loss triggered',
                    'price': current_price
                }
            
            # Check take profit
            if current_price >= take_profit:
                return {
                    'action': 'SELL',
                    'reason': 'Take profit triggered',
                    'price': current_price
                }
            
            # Check for reversal signals
            if await self._check_reversal_signal(symbol, current_data):
                return {
                    'action': 'SELL',
                    'reason': 'Reversal signal detected',
                    'price': current_price
                }
            
            return {'action': 'HOLD', 'reason': 'No exit conditions met'}
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return {'action': 'HOLD', 'reason': f'Error: {e}'}
    
    async def _check_reversal_signal(self, symbol: str, data: pd.DataFrame) -> bool:
        """Check for reversal signals"""
        try:
            if len(data) < 10:
                return False
                
            # Simple reversal check: price below resistance level again
            current_price = data['close'].iloc[-1]
            resistance_level = self.resistance_levels.get(symbol, 0)
            
            if current_price < resistance_level * 0.98:  # 2% below resistance
                return True
                
            # Check for volume divergence
            recent_volume = data['volume'].tail(5).mean()
            previous_volume = data['volume'].tail(10).head(5).mean()
            
            if recent_volume < previous_volume * 0.7:  # Volume dropped significantly
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking reversal signal: {e}")
            return False
    
    async def _fetch_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch historical data for symbol"""
        # This would integrate with your data provider
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        # This would integrate with your exchange API
        # For now, return default value
        return 10000.0
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': 'Breakout Strategy',
            'description': 'Trades breakouts above resistance levels with volume confirmation',
            'parameters': {
                'lookback_hours': self.lookback_hours,
                'leverage': self.leverage,
                'resistance_days': self.resistance_days,
                'min_volume_threshold': self.min_volume_threshold,
                'stop_loss_percent': self.stop_loss_percent,
                'take_profit_percent': self.take_profit_percent
            },
            'risk_level': 'High',
            'timeframe': '1h',
            'markets': ['Crypto', 'Forex', 'Stocks']
        } 