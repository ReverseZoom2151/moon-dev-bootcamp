"""
Mean Reversion Strategy - Based on Day_20_Projects implementation
Trades when price deviates significantly from SMA and reverts to mean
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Mean Reversion Strategy Implementation
    
    Strategy Logic:
    - Calculate SMA for specified period
    - Buy when price is below SMA by specified percentage range
    - Sell when price is above SMA by specified percentage range
    - Use randomized thresholds within ranges to avoid predictable entries
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "MeanReversion"):
        super().__init__(config, market_data_manager, name)
        
        # Strategy-specific configuration
        self.sma_period = config.get("sma_period", 14)
        self.buy_range = config.get("buy_range", (12, 15))  # % below SMA
        self.sell_range = config.get("sell_range", (14, 22))  # % above SMA
        self.order_size_usd = config.get("order_size_usd", 10)
        self.leverage = config.get("leverage", 3)
        
        # Minimum data points required
        self.min_data_points = max(self.sma_period + 10, 50)
        
        logger.info(f"🔧 Mean Reversion Strategy initialized:")
        logger.info(f"   SMA Period: {self.sma_period}")
        logger.info(f"   Buy Range: {self.buy_range}% below SMA")
        logger.info(f"   Sell Range: {self.sell_range}% above SMA")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize strategy-specific components"""
        # Validate configuration
        if self.sma_period <= 0:
            raise ValueError("SMA period must be positive")
        
        if len(self.buy_range) != 2 or self.buy_range[0] >= self.buy_range[1]:
            raise ValueError("Buy range must be a tuple of (min, max) with min < max")
        
        if len(self.sell_range) != 2 or self.sell_range[0] >= self.sell_range[1]:
            raise ValueError("Sell range must be a tuple of (min, max) with min < max")
        
        logger.info("✅ Mean Reversion strategy validation complete")
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate trading signal based on mean reversion logic"""
        try:
            # Process each symbol
            for symbol in self.symbols:
                signal = await self._analyze_symbol(symbol)
                if signal and signal.action != SignalAction.HOLD:
                    return signal
            
            return None  # No signals generated
            
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}", exc_info=True)
            return None
    
    async def _analyze_symbol(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze a specific symbol for mean reversion opportunities"""
        try:
            # Get market data
            data = await self._get_market_data(symbol, limit=self.min_data_points)
            if data is None or len(data) < self.min_data_points:
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
                return None
            
            # Calculate indicators
            indicators = await self._calculate_indicators(symbol, data)
            if not indicators:
                return None
            
            # Generate signal based on mean reversion logic
            signal = await self._evaluate_mean_reversion(symbol, data, indicators)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str, data: Any) -> Dict[str, Any]:
        """Calculate technical indicators for mean reversion strategy"""
        try:
            # Calculate SMA
            sma = self.calculate_sma(data['close'], self.sma_period)
            
            if sma is None or sma.isna().all():
                logger.warning(f"⚠️ Could not calculate SMA for {symbol}")
                return {}
            
            # Get the latest values
            current_price = float(data['close'].iloc[-1])
            current_sma = float(sma.dropna().iloc[-1])
            
            # Calculate price deviation from SMA
            price_deviation_pct = ((current_price - current_sma) / current_sma) * 100
            
            # Calculate dynamic thresholds with randomization
            buy_threshold_pct = np.random.uniform(self.buy_range[0], self.buy_range[1])
            sell_threshold_pct = np.random.uniform(self.sell_range[0], self.sell_range[1])
            
            buy_threshold_price = current_sma * (1 - buy_threshold_pct / 100)
            sell_threshold_price = current_sma * (1 + sell_threshold_pct / 100)
            
            indicators = {
                'sma': current_sma,
                'current_price': current_price,
                'price_deviation_pct': price_deviation_pct,
                'buy_threshold_pct': buy_threshold_pct,
                'sell_threshold_pct': sell_threshold_pct,
                'buy_threshold_price': buy_threshold_price,
                'sell_threshold_price': sell_threshold_price,
                'sma_series': sma
            }
            
            logger.debug(f"📊 {symbol} indicators: Price={current_price:.4f}, SMA={current_sma:.4f}, "
                        f"Deviation={price_deviation_pct:.2f}%")
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {symbol}: {e}")
            return {}
    
    async def _evaluate_mean_reversion(self, symbol: str, data: Any, indicators: Dict[str, Any]) -> Optional[StrategySignal]:
        """Evaluate mean reversion conditions and generate signal"""
        try:
            current_price = indicators['current_price']
            buy_threshold_price = indicators['buy_threshold_price']
            sell_threshold_price = indicators['sell_threshold_price']
            price_deviation_pct = indicators['price_deviation_pct']
            
            # Determine action based on mean reversion logic
            action = SignalAction.HOLD
            confidence = 0.0
            
            if current_price < buy_threshold_price:
                # Price is significantly below SMA - potential buy signal
                action = SignalAction.BUY
                # Confidence increases with larger deviation
                confidence = min(abs(price_deviation_pct) / 20.0, 1.0)  # Max confidence at 20% deviation
                
            elif current_price > sell_threshold_price:
                # Price is significantly above SMA - potential sell signal
                action = SignalAction.SELL
                # Confidence increases with larger deviation
                confidence = min(abs(price_deviation_pct) / 20.0, 1.0)
            
            # Only generate signal if confidence is above threshold
            min_confidence = 0.3
            if confidence < min_confidence:
                return None
            
            # Create metadata with strategy details
            metadata = {
                'sma': indicators['sma'],
                'price_deviation_pct': price_deviation_pct,
                'buy_threshold_price': buy_threshold_price,
                'sell_threshold_price': sell_threshold_price,
                'buy_threshold_pct': indicators['buy_threshold_pct'],
                'sell_threshold_pct': indicators['sell_threshold_pct'],
                'strategy_type': 'mean_reversion'
            }
            
            # Create and return signal
            if action != SignalAction.HOLD:
                signal = self._create_signal(
                    symbol=symbol,
                    action=action,
                    price=current_price,
                    confidence=confidence,
                    metadata=metadata
                )
                
                logger.info(f"🎯 Mean Reversion Signal: {action.value} {symbol} @ {current_price:.4f} "
                           f"(confidence: {confidence:.2f}, deviation: {price_deviation_pct:.2f}%)")
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error evaluating mean reversion for {symbol}: {e}")
            return None
    
    def _calculate_position_size(self, signal: StrategySignal) -> float:
        """Calculate position size based on strategy parameters"""
        # Use configured USD size and leverage
        base_size = self.order_size_usd
        
        # Adjust size based on confidence
        confidence_multiplier = signal.confidence
        adjusted_size = base_size * confidence_multiplier
        
        # Apply leverage
        leveraged_size = adjusted_size * self.leverage
        
        return min(leveraged_size, self.max_position_size)
    
    def _calculate_stop_loss(self, entry_price: float, action: SignalAction) -> float:
        """Calculate stop loss for mean reversion strategy"""
        # More aggressive stop loss for mean reversion (expecting quick reversals)
        stop_loss_pct = 0.7  # 70% stop loss as in original implementation
        
        if action == SignalAction.BUY:
            return entry_price * (1 - stop_loss_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 + stop_loss_pct)
        
        return entry_price
    
    def _calculate_take_profit(self, entry_price: float, action: SignalAction) -> float:
        """Calculate take profit based on mean reversion to SMA"""
        # Take profit when price reverts to SMA area
        # This would ideally use the current SMA value, but we'll use a reasonable estimate
        reversion_pct = 0.05  # 5% reversion target
        
        if action == SignalAction.BUY:
            return entry_price * (1 + reversion_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 - reversion_pct)
        
        return entry_price
    
    async def _validate_signal(self, signal: StrategySignal) -> bool:
        """Additional validation for mean reversion signals"""
        # Check if we're not already in a position for this symbol
        # This would be handled by the portfolio manager, but we can add extra checks
        
        # Ensure minimum time between signals for same symbol
        if hasattr(self, '_last_signal_time'):
            time_since_last = datetime.utcnow() - self._last_signal_time.get(signal.symbol, datetime.min)
            if time_since_last.total_seconds() < 300:  # 5 minutes minimum
                return False
        
        return True
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        base_info = self.get_status()
        
        strategy_specific = {
            'strategy_type': 'mean_reversion',
            'parameters': {
                'sma_period': self.sma_period,
                'buy_range': self.buy_range,
                'sell_range': self.sell_range,
                'order_size_usd': self.order_size_usd,
                'leverage': self.leverage
            },
            'description': 'Trades when price deviates significantly from SMA and reverts to mean'
        }
        
        base_info.update(strategy_specific)
        return base_info 