"""
Moving Average Reversal Strategy
Based on Day_49_Projects 2xmareversal.py implementation
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MAReversalStrategy(BaseStrategy):
    """
    Moving Average Reversal Strategy using dual moving averages
    
    Strategy Logic:
    - Uses two Simple Moving Averages (fast and slow)
    - Long: Enter when price is above both MAs
    - Short: Enter when price is above fast MA but below slow MA
    - Exit: Take profit/stop loss or when conditions reverse
    """
    
    def __init__(self, market_data_manager, portfolio_manager, risk_manager, config):
        super().__init__(market_data_manager, portfolio_manager, risk_manager, config)
        
        self.name = "ma_reversal"
        self.description = "Dual Moving Average Reversal Strategy"
        
        # Strategy parameters (optimized from Day_49 results)
        self.ma_fast_period = config.get('MA_FAST_PERIOD', 28)
        self.ma_slow_period = config.get('MA_SLOW_PERIOD', 30)
        self.take_profit_pct = config.get('MA_TAKE_PROFIT_PCT', 0.01)  # 1%
        self.stop_loss_pct = config.get('MA_STOP_LOSS_PCT', 0.02)     # 2%
        
        # Position tracking
        self.positions = {}
        self.entry_prices = {}
        self.ma_data = {}
        
        # Performance tracking
        self.trades_count = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        # Data requirements
        self.min_data_points = max(self.ma_fast_period, self.ma_slow_period) + 10
    
    async def initialize(self):
        """Initialize the strategy"""
        try:
            logger.info(f"🔄 Initializing MA Reversal Strategy")
            logger.info(f"📊 Parameters: Fast MA={self.ma_fast_period}, Slow MA={self.ma_slow_period}")
            logger.info(f"💰 TP={self.take_profit_pct:.1%}, SL={self.stop_loss_pct:.1%}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize MA Reversal Strategy: {e}")
            return False
    
    async def should_trade(self, symbol: str, data: Dict) -> bool:
        """Check if we should trade this symbol"""
        try:
            # Get historical data
            historical_data = await self.market_data_manager.get_historical_data(
                symbol, 
                timeframe='1h', 
                limit=self.min_data_points
            )
            
            if historical_data is None or len(historical_data) < self.min_data_points:
                return False
            
            # Calculate moving averages
            await self._calculate_moving_averages(symbol, historical_data)
            
            # Check if we have valid MA data
            if symbol not in self.ma_data:
                return False
            
            ma_data = self.ma_data[symbol]
            if len(ma_data['ma_fast']) < 2 or len(ma_data['ma_slow']) < 2:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking if should trade {symbol}: {e}")
            return False
    
    async def generate_signals(self, symbol: str, data: Dict) -> Dict:
        """Generate trading signals based on MA crossover"""
        try:
            if symbol not in self.ma_data:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No MA data'}
            
            ma_data = self.ma_data[symbol]
            current_price = data.get('price', 0)
            
            if current_price <= 0:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Invalid price'}
            
            # Get latest MA values
            ma_fast_current = ma_data['ma_fast'].iloc[-1]
            ma_slow_current = ma_data['ma_slow'].iloc[-1]
            ma_fast_prev = ma_data['ma_fast'].iloc[-2]
            ma_slow_prev = ma_data['ma_slow'].iloc[-2]
            
            # Current position
            current_position = await self.portfolio_manager.get_position(symbol)
            position_size = current_position.quantity if current_position else 0
            
            # Check exit conditions first
            if position_size != 0:
                exit_signal = await self._check_exit_conditions(symbol, current_price, position_size)
                if exit_signal['action'] != 'hold':
                    return exit_signal
            
            # Entry conditions
            signal = {'action': 'hold', 'confidence': 0.0, 'reason': 'No signal'}
            
            # Long entry: Price above both MAs
            if (current_price > ma_fast_current and 
                current_price > ma_slow_current and
                position_size <= 0):  # Not already long
                
                # Additional confirmation: Fast MA crossing above slow MA
                if ma_fast_current > ma_slow_current and ma_fast_prev <= ma_slow_prev:
                    signal = {
                        'action': 'buy',
                        'confidence': 0.8,
                        'reason': f'Price above both MAs, fast MA crossing above slow MA'
                    }
                elif ma_fast_current > ma_slow_current:
                    signal = {
                        'action': 'buy',
                        'confidence': 0.6,
                        'reason': f'Price above both MAs'
                    }
            
            # Short entry: Price above fast MA but below slow MA
            elif (current_price > ma_fast_current and 
                  current_price < ma_slow_current and
                  position_size >= 0):  # Not already short
                
                signal = {
                    'action': 'sell',
                    'confidence': 0.7,
                    'reason': f'Price above fast MA but below slow MA'
                }
            
            # Add technical indicators to signal
            signal['ma_fast'] = ma_fast_current
            signal['ma_slow'] = ma_slow_current
            signal['price'] = current_price
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating signals for {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    async def _check_exit_conditions(self, symbol: str, current_price: float, position_size: float) -> Dict:
        """Check if we should exit current position"""
        try:
            if symbol not in self.entry_prices:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No entry price recorded'}
            
            entry_price = self.entry_prices[symbol]
            ma_data = self.ma_data[symbol]
            ma_slow_current = ma_data['ma_slow'].iloc[-1]
            
            # Calculate P&L percentage
            if position_size > 0:  # Long position
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Take profit
                if pnl_pct >= self.take_profit_pct:
                    return {
                        'action': 'sell',
                        'confidence': 1.0,
                        'reason': f'Take profit: {pnl_pct:.2%} >= {self.take_profit_pct:.2%}'
                    }
                
                # Stop loss
                if pnl_pct <= -self.stop_loss_pct:
                    return {
                        'action': 'sell',
                        'confidence': 1.0,
                        'reason': f'Stop loss: {pnl_pct:.2%} <= {-self.stop_loss_pct:.2%}'
                    }
                
                # Technical exit: Price below slow MA
                if current_price < ma_slow_current:
                    return {
                        'action': 'sell',
                        'confidence': 0.8,
                        'reason': f'Price below slow MA: {current_price:.4f} < {ma_slow_current:.4f}'
                    }
            
            elif position_size < 0:  # Short position
                pnl_pct = (entry_price - current_price) / entry_price
                
                # Take profit
                if pnl_pct >= self.take_profit_pct:
                    return {
                        'action': 'buy',
                        'confidence': 1.0,
                        'reason': f'Take profit: {pnl_pct:.2%} >= {self.take_profit_pct:.2%}'
                    }
                
                # Stop loss
                if pnl_pct <= -self.stop_loss_pct:
                    return {
                        'action': 'buy',
                        'confidence': 1.0,
                        'reason': f'Stop loss: {pnl_pct:.2%} <= {-self.stop_loss_pct:.2%}'
                    }
                
                # Technical exit: Price above slow MA
                if current_price > ma_slow_current:
                    return {
                        'action': 'buy',
                        'confidence': 0.8,
                        'reason': f'Price above slow MA: {current_price:.4f} > {ma_slow_current:.4f}'
                    }
            
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'No exit conditions met'}
            
        except Exception as e:
            logger.error(f"❌ Error checking exit conditions for {symbol}: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    async def _calculate_moving_averages(self, symbol: str, data: pd.DataFrame):
        """Calculate moving averages for the symbol"""
        try:
            if len(data) < max(self.ma_fast_period, self.ma_slow_period):
                return
            
            # Calculate Simple Moving Averages
            ma_fast = data['close'].rolling(window=self.ma_fast_period).mean()
            ma_slow = data['close'].rolling(window=self.ma_slow_period).mean()
            
            # Store MA data
            self.ma_data[symbol] = {
                'ma_fast': ma_fast,
                'ma_slow': ma_slow,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating moving averages for {symbol}: {e}")
    
    async def calculate_position_size(self, symbol: str, signal: Dict) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            account_balance = await self.portfolio_manager.get_total_value()
            
            # Risk per trade (default 2% of account)
            risk_per_trade = account_balance * 0.02
            
            # Calculate position size based on stop loss
            current_price = signal.get('price', 0)
            if current_price <= 0:
                return 0
            
            # Position size = Risk / (Price * Stop Loss %)
            position_size = risk_per_trade / (current_price * self.stop_loss_pct)
            
            # Apply maximum position size limit (10% of account)
            max_position_value = account_balance * 0.10
            max_position_size = max_position_value / current_price
            
            position_size = min(position_size, max_position_size)
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size for {symbol}: {e}")
            return 0
    
    async def on_trade_executed(self, symbol: str, side: str, quantity: float, price: float):
        """Handle trade execution"""
        try:
            if side in ['buy', 'long']:
                # Record entry price for long position
                self.entry_prices[symbol] = price
                logger.info(f"🟢 MA Reversal LONG: {symbol} @ {price:.4f} | Size: {quantity:.4f}")
                
            elif side in ['sell', 'short']:
                # Record entry price for short position or exit for long
                current_position = await self.portfolio_manager.get_position(symbol)
                
                if current_position and current_position.quantity > 0:
                    # Closing long position
                    if symbol in self.entry_prices:
                        entry_price = self.entry_prices[symbol]
                        pnl_pct = (price - entry_price) / entry_price
                        self.total_pnl += pnl_pct
                        self.trades_count += 1
                        
                        if pnl_pct > 0:
                            self.winning_trades += 1
                        
                        logger.info(f"🔴 MA Reversal CLOSE LONG: {symbol} @ {price:.4f} | P&L: {pnl_pct:.2%}")
                        del self.entry_prices[symbol]
                else:
                    # Opening short position
                    self.entry_prices[symbol] = price
                    logger.info(f"🔴 MA Reversal SHORT: {symbol} @ {price:.4f} | Size: {quantity:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error handling trade execution for {symbol}: {e}")
    
    async def get_strategy_stats(self) -> Dict:
        """Get strategy performance statistics"""
        try:
            win_rate = (self.winning_trades / self.trades_count * 100) if self.trades_count > 0 else 0
            avg_pnl = (self.total_pnl / self.trades_count * 100) if self.trades_count > 0 else 0
            
            return {
                'strategy_name': self.name,
                'total_trades': self.trades_count,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'total_pnl_pct': self.total_pnl * 100,
                'avg_trade_pnl_pct': avg_pnl,
                'active_positions': len(self.entry_prices),
                'parameters': {
                    'ma_fast_period': self.ma_fast_period,
                    'ma_slow_period': self.ma_slow_period,
                    'take_profit_pct': self.take_profit_pct,
                    'stop_loss_pct': self.stop_loss_pct
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy stats: {e}")
            return {'strategy_name': self.name, 'error': str(e)}
    
    async def update_parameters(self, new_params: Dict):
        """Update strategy parameters"""
        try:
            if 'ma_fast_period' in new_params:
                self.ma_fast_period = new_params['ma_fast_period']
            if 'ma_slow_period' in new_params:
                self.ma_slow_period = new_params['ma_slow_period']
            if 'take_profit_pct' in new_params:
                self.take_profit_pct = new_params['take_profit_pct']
            if 'stop_loss_pct' in new_params:
                self.stop_loss_pct = new_params['stop_loss_pct']
            
            # Update minimum data points requirement
            self.min_data_points = max(self.ma_fast_period, self.ma_slow_period) + 10
            
            logger.info(f"📊 MA Reversal parameters updated: {new_params}")
            
        except Exception as e:
            logger.error(f"❌ Error updating parameters: {e}")
    
    async def cleanup(self):
        """Cleanup strategy resources"""
        try:
            # Clear data structures
            self.ma_data.clear()
            self.entry_prices.clear()
            self.positions.clear()
            
            logger.info("🧹 MA Reversal strategy cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up MA Reversal strategy: {e}") 