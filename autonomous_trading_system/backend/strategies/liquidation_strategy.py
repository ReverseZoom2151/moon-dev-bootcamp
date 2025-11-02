"""
Liquidation Strategy - Based on Day_21_Projects implementation
Trades based on liquidation events and volume spikes
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime
from .base_strategy import BaseStrategy, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class LiquidationStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Liquidation Strategy Implementation
    
    Strategy Logic:
    - Monitor liquidation events (short and long liquidations)
    - Enter long positions when significant short liquidations occur
    - Enter short positions when significant long liquidations occur
    - Use volume thresholds and time windows for signal generation
    """
    
    def __init__(
        self,
        min_liquidation_size: float = 100000,
        follow_delay_seconds: int = 5,
        max_position_size_percent: float = 10.0,
        liquidation_momentum_threshold: float = 3.0,
        stop_loss_percent: float = 5.0,
        take_profit_percent: float = 2.0,
        liquidation_window_minutes: int = 15
    ):
        super().__init__()
        self.min_liquidation_size = min_liquidation_size
        self.follow_delay_seconds = follow_delay_seconds
        self.max_position_size_percent = max_position_size_percent
        self.liquidation_momentum_threshold = liquidation_momentum_threshold
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.liquidation_window_minutes = liquidation_window_minutes
        
        # Track recent liquidations
        self.recent_liquidations = {}
        self.liquidation_momentum = {}
        
        logger.info(f"🔧 Liquidation Strategy initialized:")
        logger.info(f"   Min Liquidation Size: ${self.min_liquidation_size:,}")
        logger.info(f"   Follow Delay: {self.follow_delay_seconds} seconds")
        logger.info(f"   Max Position Size Percent: {self.max_position_size_percent}%")
        logger.info(f"   Liquidation Momentum Threshold: {self.liquidation_momentum_threshold}")
        logger.info(f"   Stop Loss Percent: {self.stop_loss_percent}%")
        logger.info(f"   Take Profit Percent: {self.take_profit_percent}%")
        logger.info(f"   Liquidation Window: {self.liquidation_window_minutes} minutes")
    
    async def initialize(self, symbol: str):
        """Initialize strategy for a symbol"""
        await super().initialize(symbol)
        self.recent_liquidations[symbol] = []
        self.liquidation_momentum[symbol] = 0.0
    
    async def process_liquidation_event(self, liquidation_data: Dict):
        """Process incoming liquidation event"""
        try:
            symbol = liquidation_data.get('symbol')
            side = liquidation_data.get('side')  # 'SELL' for long liquidation, 'BUY' for short liquidation
            size = liquidation_data.get('size', 0)
            price = liquidation_data.get('price', 0)
            timestamp = liquidation_data.get('timestamp', datetime.utcnow().timestamp())
            
            if size < self.min_liquidation_size:
                return
                
            if symbol not in self.recent_liquidations:
                self.recent_liquidations[symbol] = []
                
            # Add liquidation to recent list
            liquidation_event = {
                'timestamp': timestamp,
                'side': side,
                'size': size,
                'price': price,
                'type': 'liquidation'
            }
            
            self.recent_liquidations[symbol].append(liquidation_event)
            
            # Clean old liquidations (older than window)
            cutoff_time = timestamp - (self.liquidation_window_minutes * 60)
            self.recent_liquidations[symbol] = [
                liq for liq in self.recent_liquidations[symbol] 
                if liq['timestamp'] > cutoff_time
            ]
            
            # Update liquidation momentum
            await self._update_liquidation_momentum(symbol)
            
            logger.info(f"Liquidation processed: {symbol} {side} ${size:,.0f} @ ${price}")
            
        except Exception as e:
            logger.error(f"Error processing liquidation event: {e}")
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Generate trading signal based on liquidation analysis"""
        try:
            if len(data) < 10:
                return self._create_signal('HOLD', 0, "Insufficient data")
                
            current_price = data['close'].iloc[-1]
            
            # Check if we have recent liquidations
            if symbol not in self.recent_liquidations or not self.recent_liquidations[symbol]:
                return self._create_signal('HOLD', 0, "No recent liquidations")
            
            # Analyze liquidation momentum
            momentum = self.liquidation_momentum.get(symbol, 0)
            
            if abs(momentum) < self.liquidation_momentum_threshold:
                return self._create_signal('HOLD', 0, f"Liquidation momentum too low: {momentum}")
            
            # Determine signal direction based on liquidation momentum
            if momentum > self.liquidation_momentum_threshold:
                # Strong long liquidations -> expect further downside -> SHORT
                signal_direction = 'SELL'
                reason = f"Strong long liquidation momentum: {momentum:.2f}"
            elif momentum < -self.liquidation_momentum_threshold:
                # Strong short liquidations -> expect upside -> LONG
                signal_direction = 'BUY'
                reason = f"Strong short liquidation momentum: {momentum:.2f}"
            else:
                return self._create_signal('HOLD', 0, "Mixed liquidation signals")
            
            # Calculate position size
            position_size = await self._calculate_position_size(symbol, current_price, momentum)
            
            if position_size <= 0:
                return self._create_signal('HOLD', 0, "Position size calculation failed")
            
            # Calculate stop loss and take profit
            if signal_direction == 'BUY':
                stop_loss = current_price * (1 - self.stop_loss_percent / 100)
                take_profit = current_price * (1 + self.take_profit_percent / 100)
            else:  # SELL
                stop_loss = current_price * (1 + self.stop_loss_percent / 100)
                take_profit = current_price * (1 - self.take_profit_percent / 100)
            
            signal_data = {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'liquidation_momentum': momentum,
                'recent_liquidations_count': len(self.recent_liquidations[symbol])
            }
            
            return self._create_signal(signal_direction, position_size, reason, signal_data)
            
        except Exception as e:
            logger.error(f"Error generating liquidation signal for {symbol}: {e}")
            return self._create_signal('HOLD', 0, f"Error: {e}")
    
    async def _update_liquidation_momentum(self, symbol: str):
        """Update liquidation momentum score"""
        try:
            liquidations = self.recent_liquidations.get(symbol, [])
            
            if not liquidations:
                self.liquidation_momentum[symbol] = 0.0
                return
            
            # Calculate momentum based on liquidation direction and size
            momentum_score = 0.0
            total_volume = 0.0
            
            for liq in liquidations:
                size = liq['size']
                side = liq['side']
                
                # Weight by size and recency
                time_weight = self._calculate_time_weight(liq['timestamp'])
                size_weight = min(size / 1000000, 10.0)  # Cap at 10M for weighting
                
                # Long liquidations (SELL side) contribute negative momentum
                # Short liquidations (BUY side) contribute positive momentum
                if side == 'SELL':  # Long liquidation
                    momentum_score -= size_weight * time_weight
                else:  # Short liquidation
                    momentum_score += size_weight * time_weight
                    
                total_volume += size
            
            # Normalize momentum by total volume
            if total_volume > 0:
                self.liquidation_momentum[symbol] = momentum_score
            else:
                self.liquidation_momentum[symbol] = 0.0
                
            logger.debug(f"Liquidation momentum for {symbol}: {self.liquidation_momentum[symbol]:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating liquidation momentum: {e}")
    
    def _calculate_time_weight(self, timestamp: float) -> float:
        """Calculate time-based weight for liquidation events"""
        current_time = datetime.utcnow().timestamp()
        time_diff_minutes = (current_time - timestamp) / 60
        
        # Exponential decay: more recent events have higher weight
        return np.exp(-time_diff_minutes / 5.0)  # Half-life of 5 minutes
    
    async def _calculate_position_size(self, symbol: str, current_price: float, momentum: float) -> float:
        """Calculate position size based on liquidation momentum"""
        try:
            # Get account balance
            account_balance = await self._get_account_balance()
            
            # Base position size on momentum strength
            momentum_factor = min(abs(momentum) / 10.0, 1.0)  # Cap at 1.0
            
            # Calculate position size as percentage of account
            position_percent = self.max_position_size_percent * momentum_factor / 100
            position_value = account_balance * position_percent
            
            # Convert to position size
            position_size = position_value / current_price
            
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
            position_side = position.get('side', 'LONG')
            
            # Check stop loss
            if position_side == 'LONG' and current_price <= stop_loss:
                return {
                    'action': 'SELL',
                    'reason': 'Stop loss triggered',
                    'price': current_price
                }
            elif position_side == 'SHORT' and current_price >= stop_loss:
                return {
                    'action': 'BUY',
                    'reason': 'Stop loss triggered',
                    'price': current_price
                }
            
            # Check take profit
            if position_side == 'LONG' and current_price >= take_profit:
                return {
                    'action': 'SELL',
                    'reason': 'Take profit triggered',
                    'price': current_price
                }
            elif position_side == 'SHORT' and current_price <= take_profit:
                return {
                    'action': 'BUY',
                    'reason': 'Take profit triggered',
                    'price': current_price
                }
            
            # Check for momentum reversal
            current_momentum = self.liquidation_momentum.get(symbol, 0)
            entry_momentum = position.get('liquidation_momentum', 0)
            
            # If momentum has reversed significantly, consider exit
            if (entry_momentum > 0 and current_momentum < -1.0) or \
               (entry_momentum < 0 and current_momentum > 1.0):
                return {
                    'action': 'SELL' if position_side == 'LONG' else 'BUY',
                    'reason': 'Liquidation momentum reversed',
                    'price': current_price
                }
            
            return {'action': 'HOLD', 'reason': 'No exit conditions met'}
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return {'action': 'HOLD', 'reason': f'Error: {e}'}
    
    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        # This would integrate with your exchange API
        return 10000.0
    
    def get_liquidation_summary(self, symbol: str) -> Dict:
        """Get summary of recent liquidations"""
        liquidations = self.recent_liquidations.get(symbol, [])
        
        if not liquidations:
            return {'total_liquidations': 0, 'total_volume': 0, 'momentum': 0}
        
        total_volume = sum(liq['size'] for liq in liquidations)
        long_liquidations = [liq for liq in liquidations if liq['side'] == 'SELL']
        short_liquidations = [liq for liq in liquidations if liq['side'] == 'BUY']
        
        return {
            'total_liquidations': len(liquidations),
            'total_volume': total_volume,
            'long_liquidations': len(long_liquidations),
            'short_liquidations': len(short_liquidations),
            'momentum': self.liquidation_momentum.get(symbol, 0),
            'avg_liquidation_size': total_volume / len(liquidations) if liquidations else 0
        }
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information"""
        return {
            'name': 'Liquidation Strategy',
            'description': 'Follows large liquidation events to capture momentum',
            'parameters': {
                'min_liquidation_size': self.min_liquidation_size,
                'follow_delay_seconds': self.follow_delay_seconds,
                'max_position_size_percent': self.max_position_size_percent,
                'liquidation_momentum_threshold': self.liquidation_momentum_threshold,
                'stop_loss_percent': self.stop_loss_percent,
                'take_profit_percent': self.take_profit_percent,
                'liquidation_window_minutes': self.liquidation_window_minutes
            },
            'risk_level': 'High',
            'timeframe': 'Real-time',
            'markets': ['Crypto Futures']
        } 