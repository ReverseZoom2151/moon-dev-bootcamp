"""
2x MA Reversal Strategy
=======================
Day 49: Dual Moving Average Reversal Strategy

Features:
- Fast MA (25) and Slow MA (30) crossover signals
- Long entries: price above both MAs
- Short entries: price above fast MA but below slow MA
- Take profit: 5%
- Stop loss: 5%
- Close short when price moves above slow MA

Optimized parameters from Moon Dev testing:
- Fast MA: 25 periods
- Slow MA: 30 periods
- Take Profit: 5%
- Stop Loss: 5%
"""

import logging
import asyncio
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MAReversalStrategy:
    """
    2x Moving Average Reversal Strategy (Day 49).
    
    Uses dual MA crossover to identify reversal opportunities:
    - Long: Price above both fast and slow MA
    - Short: Price above fast MA but below slow MA
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize 2x MA Reversal Strategy.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Optimized parameters from Day 49
        self.ma_fast = self.config.get('ma_fast', 25)
        self.ma_slow = self.config.get('ma_slow', 30)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.05)  # 5%
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)  # 5%
        
        # Trading configuration
        self.position_size_usd = self.config.get('position_size_usd', 1000.0)
        self.symbol = self.config.get('symbol', 'BTCUSDT')
        self.timeframe = self.config.get('timeframe', '1d')
        self.lookback_periods = self.config.get('lookback_periods', 100)
        
        # Position tracking
        self.position = 0.0  # Current position (positive for long, negative for short)
        self.entry_price = 0.0
        
        logger.info(
            f"MA Reversal Strategy initialized: "
            f"Fast MA={self.ma_fast}, Slow MA={self.ma_slow}, "
            f"TP={self.take_profit_pct*100}%, SL={self.stop_loss_pct*100}%"
        )
    
    async def calculate_moving_averages(
        self,
        symbol: str,
        timeframe: str = None,
        lookback: int = None
    ) -> Optional[Tuple[float, float, float]]:
        """
        Calculate fast and slow moving averages.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            lookback: Number of periods to look back
            
        Returns:
            Tuple of (current_price, fast_ma, slow_ma) or None
        """
        if not self.exchange_adapter:
            logger.error("Exchange adapter not available")
            return None
        
        timeframe = timeframe or self.timeframe
        lookback = lookback or self.lookback_periods
        
        try:
            # Get historical data
            ohlcv_data = await self.exchange_adapter.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback
            )
            
            if not ohlcv_data or len(ohlcv_data) < self.ma_slow:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(ohlcv_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Calculate moving averages
            from gordon.core.utilities import EnhancedTradingUtils
            utils = EnhancedTradingUtils()
            
            close_prices = df['close']
            fast_ma = utils.calculate_sma(close_prices, window=self.ma_fast)
            slow_ma = utils.calculate_sma(close_prices, window=self.ma_slow)
            
            if fast_ma.empty or slow_ma.empty:
                return None
            
            current_price = float(close_prices.iloc[-1])
            current_fast_ma = float(fast_ma.iloc[-1])
            current_slow_ma = float(slow_ma.iloc[-1])
            
            return (current_price, current_fast_ma, current_slow_ma)
            
        except Exception as e:
            logger.error(f"Error calculating MAs: {e}")
            return None
    
    async def check_signals(
        self,
        symbol: str = None,
        price: float = None,
        fast_ma: float = None,
        slow_ma: float = None
    ) -> Optional[Dict]:
        """
        Check for trading signals.
        
        Args:
            symbol: Trading symbol
            price: Current price (will fetch if not provided)
            fast_ma: Fast MA value (will calculate if not provided)
            slow_ma: Slow MA value (will calculate if not provided)
            
        Returns:
            Dict with signal information or None
        """
        symbol = symbol or self.symbol
        
        # Calculate MAs if not provided
        if price is None or fast_ma is None or slow_ma is None:
            ma_data = await self.calculate_moving_averages(symbol)
            if not ma_data:
                return None
            price, fast_ma, slow_ma = ma_data
        
        signal = None
        signal_type = None
        
        # Check for short setup: price above fast MA but below slow MA
        if price > fast_ma and price < slow_ma:
            signal = 'SHORT'
            signal_type = 'ENTRY'
        
        # Check for long setup: price above both MAs
        elif price > fast_ma and price > slow_ma:
            signal = 'LONG'
            signal_type = 'ENTRY'
        
        # Check for short close: price moves above slow MA
        if self.position < 0 and price > slow_ma:
            signal = 'CLOSE_SHORT'
            signal_type = 'EXIT'
        
        if signal:
            return {
                'signal': signal,
                'signal_type': signal_type,
                'price': price,
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'symbol': symbol
            }
        
        return None
    
    async def execute_long(
        self,
        symbol: str,
        price: float
    ) -> bool:
        """
        Execute long position entry.
        
        Args:
            symbol: Trading symbol
            price: Entry price
            
        Returns:
            Success status
        """
        if not self.exchange_adapter:
            return False
        
        try:
            # Calculate position size
            quantity = self.position_size_usd / price
            
            # Place buy order
            order_result = await self.exchange_adapter.place_order(
                symbol=symbol,
                side='buy',
                amount=quantity,
                order_type='market'
            )
            
            if order_result:
                self.position = quantity
                self.entry_price = price
                logger.info(f"Long position opened: {symbol} @ {price} (Qty: {quantity})")
                return True
            else:
                logger.error(f"Failed to open long position: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing long: {e}")
            return False
    
    async def execute_short(
        self,
        symbol: str,
        price: float
    ) -> bool:
        """
        Execute short position entry.
        
        Args:
            symbol: Trading symbol
            price: Entry price
            
        Returns:
            Success status
        """
        if not self.exchange_adapter:
            return False
        
        try:
            # Calculate position size
            quantity = self.position_size_usd / price
            
            # Place sell order (short)
            order_result = await self.exchange_adapter.place_order(
                symbol=symbol,
                side='sell',
                amount=quantity,
                order_type='market'
            )
            
            if order_result:
                self.position = -quantity  # Negative for short
                self.entry_price = price
                logger.info(f"Short position opened: {symbol} @ {price} (Qty: {quantity})")
                return True
            else:
                logger.error(f"Failed to open short position: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing short: {e}")
            return False
    
    async def check_exit_conditions(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Dict]:
        """
        Check if position should be closed based on TP/SL or strategy rules.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Dict with exit signal or None
        """
        if abs(self.position) < 0.0001 or self.entry_price == 0:
            return None
        
        # Calculate PnL
        if self.position > 0:  # Long position
            pnl_pct = ((current_price - self.entry_price) / self.entry_price)
            
            # Check take profit
            if pnl_pct >= self.take_profit_pct:
                return {
                    'exit_signal': 'TAKE_PROFIT',
                    'reason': f'Take profit reached: {pnl_pct*100:.2f}%',
                    'price': current_price,
                    'entry_price': self.entry_price,
                    'pnl_pct': pnl_pct
                }
            
            # Check stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return {
                    'exit_signal': 'STOP_LOSS',
                    'reason': f'Stop loss triggered: {pnl_pct*100:.2f}%',
                    'price': current_price,
                    'entry_price': self.entry_price,
                    'pnl_pct': pnl_pct
                }
        
        elif self.position < 0:  # Short position
            pnl_pct = ((self.entry_price - current_price) / self.entry_price)
            
            # Check take profit
            if pnl_pct >= self.take_profit_pct:
                return {
                    'exit_signal': 'TAKE_PROFIT',
                    'reason': f'Take profit reached: {pnl_pct*100:.2f}%',
                    'price': current_price,
                    'entry_price': self.entry_price,
                    'pnl_pct': pnl_pct
                }
            
            # Check stop loss
            if pnl_pct <= -self.stop_loss_pct:
                return {
                    'exit_signal': 'STOP_LOSS',
                    'reason': f'Stop loss triggered: {pnl_pct*100:.2f}%',
                    'price': current_price,
                    'entry_price': self.entry_price,
                    'pnl_pct': pnl_pct
                }
        
        return None
    
    async def close_position(
        self,
        symbol: str
    ) -> bool:
        """
        Close current position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Success status
        """
        if not self.exchange_adapter or abs(self.position) < 0.0001:
            return False
        
        try:
            if self.position > 0:
                # Close long position (sell)
                order_result = await self.exchange_adapter.place_order(
                    symbol=symbol,
                    side='sell',
                    amount=abs(self.position),
                    order_type='market'
                )
            else:
                # Close short position (buy)
                order_result = await self.exchange_adapter.place_order(
                    symbol=symbol,
                    side='buy',
                    amount=abs(self.position),
                    order_type='market'
                )
            
            if order_result:
                logger.info(f"Position closed: {symbol} (Was: {self.position})")
                self.position = 0.0
                self.entry_price = 0.0
                return True
            else:
                logger.error(f"Failed to close position: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def execute(
        self,
        symbol: str = None
    ) -> Optional[Dict]:
        """
        Execute strategy logic and check for signals.
        
        Args:
            symbol: Trading symbol (uses default if not provided)
            
        Returns:
            Dict with execution results
        """
        symbol = symbol or self.symbol
        
        try:
            # Check for exit conditions first
            if abs(self.position) > 0.0001:
                # Get current price
                ticker = await self.exchange_adapter.get_ticker(symbol)
                if ticker:
                    current_price = float(ticker.get('last', 0))
                    
                    # Check exit conditions
                    exit_signal = await self.check_exit_conditions(symbol, current_price)
                    if exit_signal:
                        logger.info(f"Exit signal detected: {exit_signal['exit_signal']}")
                        await self.close_position(symbol)
                        return {
                            'action': 'CLOSED',
                            'exit_signal': exit_signal,
                            'symbol': symbol
                        }
            
            # Check for entry signals
            signal_info = await self.check_signals(symbol)
            
            if signal_info:
                signal = signal_info['signal']
                
                if signal == 'LONG' and abs(self.position) < 0.0001:
                    # Open long position
                    success = await self.execute_long(symbol, signal_info['price'])
                    return {
                        'action': 'OPENED_LONG',
                        'success': success,
                        'signal': signal_info,
                        'symbol': symbol
                    }
                
                elif signal == 'SHORT' and abs(self.position) < 0.0001:
                    # Open short position
                    success = await self.execute_short(symbol, signal_info['price'])
                    return {
                        'action': 'OPENED_SHORT',
                        'success': success,
                        'signal': signal_info,
                        'symbol': symbol
                    }
                
                elif signal == 'CLOSE_SHORT' and self.position < 0:
                    # Close short position
                    success = await self.close_position(symbol)
                    return {
                        'action': 'CLOSED_SHORT',
                        'success': success,
                        'signal': signal_info,
                        'symbol': symbol
                    }
            
            return {
                'action': 'NO_SIGNAL',
                'symbol': symbol,
                'current_position': self.position
            }
            
        except Exception as e:
            logger.error(f"Error executing strategy: {e}")
            return None

