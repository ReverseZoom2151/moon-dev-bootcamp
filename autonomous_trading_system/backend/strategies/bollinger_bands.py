"""
Bollinger Bands Strategy - Based on Day_10_Projects implementation
Trades on band compression (tight bands) indicating potential breakouts
"""

import logging
from typing import Dict, Any, Optional
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class BollingerBandsStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Bollinger Bands Strategy Implementation
    
    Strategy Logic:
    - Calculate Bollinger Bands (SMA ± std_dev * standard deviation)
    - Detect band compression (tight bands) as entry signal
    - Place both long and short orders when bands are compressed
    - Exit when bands expand or profit/loss targets are hit
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "BollingerBands"):
        super().__init__(config, market_data_manager, name)
        
        # Strategy-specific configuration
        self.period = config.get("period", 20)
        self.std_dev = config.get("std_dev", 2)
        self.compression_threshold = config.get("compression_threshold", 0.02)  # 2% band width
        self.position_size = config.get("position_size", 1000)
        
        # Minimum data points required
        self.min_data_points = max(self.period + 10, 50)
        
        logger.info(f"🔧 Bollinger Bands Strategy initialized:")
        logger.info(f"   Period: {self.period}")
        logger.info(f"   Standard Deviation: {self.std_dev}")
        logger.info(f"   Compression Threshold: {self.compression_threshold * 100}%")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize strategy-specific components"""
        # Validate configuration
        if self.period <= 0:
            raise ValueError("Bollinger Bands period must be positive")
        
        if self.std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
        
        if not 0 < self.compression_threshold < 1:
            raise ValueError("Compression threshold must be between 0 and 1")
        
        logger.info("✅ Bollinger Bands strategy validation complete")
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate trading signal based on Bollinger Bands compression"""
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
        """Analyze a specific symbol for Bollinger Bands opportunities"""
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
            
            # Generate signal based on band compression
            signal = await self._evaluate_bollinger_bands(symbol, data, indicators)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str, data: Any) -> Dict[str, Any]:
        """Calculate Bollinger Bands indicators"""
        try:
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(
                data['close'], self.period, self.std_dev
            )
            
            if any(x is None for x in [upper_band, middle_band, lower_band]):
                logger.warning(f"⚠️ Could not calculate Bollinger Bands for {symbol}")
                return {}
            
            # Get the latest values
            current_price = float(data['close'].iloc[-1])
            current_upper = float(upper_band.dropna().iloc[-1])
            current_middle = float(middle_band.dropna().iloc[-1])
            current_lower = float(lower_band.dropna().iloc[-1])
            
            # Calculate band width (compression indicator)
            band_width = (current_upper - current_lower) / current_middle
            
            # Check if bands are compressed
            bands_compressed = band_width < self.compression_threshold
            
            # Calculate price position within bands
            if current_upper != current_lower:
                price_position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                price_position = 0.5  # Middle position if bands are identical
            
            indicators = {
                'upper_band': current_upper,
                'middle_band': current_middle,
                'lower_band': current_lower,
                'current_price': current_price,
                'band_width': band_width,
                'bands_compressed': bands_compressed,
                'price_position': price_position,
                'upper_band_series': upper_band,
                'middle_band_series': middle_band,
                'lower_band_series': lower_band
            }
            
            logger.debug(f"📊 {symbol} BB indicators: Price={current_price:.4f}, "
                        f"Bands=[{current_lower:.4f}, {current_middle:.4f}, {current_upper:.4f}], "
                        f"Width={band_width:.4f}, Compressed={bands_compressed}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {symbol}: {e}")
            return {}
    
    async def _evaluate_bollinger_bands(self, symbol: str, data: Any, indicators: Dict[str, Any]) -> Optional[StrategySignal]:
        """Evaluate Bollinger Bands conditions and generate signal"""
        try:
            current_price = indicators['current_price']
            bands_compressed = indicators['bands_compressed']
            price_position = indicators['price_position']
            band_width = indicators['band_width']
            
            # Only trade when bands are compressed (indicating potential breakout)
            if not bands_compressed:
                return None
            
            # Determine action based on price position and compression
            action = SignalAction.HOLD
            confidence = 0.0
            
            # When bands are compressed, we expect a breakout
            # Place orders on both sides to catch the breakout
            if bands_compressed:
                # Prefer direction based on price position within bands
                if price_position > 0.6:
                    # Price near upper band - expect upward breakout
                    action = SignalAction.BUY
                    confidence = min((1 - band_width) * 2, 1.0)  # Higher confidence with tighter bands
                elif price_position < 0.4:
                    # Price near lower band - expect downward breakout
                    action = SignalAction.SELL
                    confidence = min((1 - band_width) * 2, 1.0)
                else:
                    # Price in middle - use random selection as in original
                    import random
                    action = SignalAction.BUY if random.random() > 0.5 else SignalAction.SELL
                    confidence = min((1 - band_width) * 1.5, 0.8)  # Moderate confidence
            
            # Only generate signal if confidence is above threshold
            min_confidence = 0.4
            if confidence < min_confidence:
                return None
            
            # Create metadata with strategy details
            metadata = {
                'upper_band': indicators['upper_band'],
                'middle_band': indicators['middle_band'],
                'lower_band': indicators['lower_band'],
                'band_width': band_width,
                'bands_compressed': bands_compressed,
                'price_position': price_position,
                'compression_threshold': self.compression_threshold,
                'strategy_type': 'bollinger_bands'
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
                
                logger.info(f"🎯 Bollinger Bands Signal: {action.value} {symbol} @ {current_price:.4f} "
                           f"(confidence: {confidence:.2f}, band_width: {band_width:.4f})")
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error evaluating Bollinger Bands for {symbol}: {e}")
            return None
    
    def _calculate_stop_loss(self, entry_price: float, action: SignalAction) -> float:
        """Calculate stop loss based on Bollinger Bands"""
        # Use the opposite band as stop loss
        stop_loss_pct = 0.03  # 3% stop loss for breakout strategies
        
        if action == SignalAction.BUY:
            return entry_price * (1 - stop_loss_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 + stop_loss_pct)
        
        return entry_price
    
    def _calculate_take_profit(self, entry_price: float, action: SignalAction) -> float:
        """Calculate take profit for breakout strategy"""
        # Target profit based on expected breakout magnitude
        take_profit_pct = 0.05  # 5% take profit for breakouts
        
        if action == SignalAction.BUY:
            return entry_price * (1 + take_profit_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 - take_profit_pct)
        
        return entry_price
    
    async def _detect_band_expansion(self, symbol: str) -> bool:
        """Detect if bands are expanding (exit signal)"""
        try:
            # Get recent data to check for expansion
            data = await self._get_market_data(symbol, limit=10)
            if data is None or len(data) < 5:
                return False
            
            # Calculate recent band widths
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(
                data['close'], self.period, self.std_dev
            )
            
            if any(x is None for x in [upper_band, middle_band, lower_band]):
                return False
            
            # Check if band width is increasing (expansion)
            recent_widths = []
            for i in range(-3, 0):  # Last 3 periods
                try:
                    upper = float(upper_band.iloc[i])
                    middle = float(middle_band.iloc[i])
                    lower = float(lower_band.iloc[i])
                    width = (upper - lower) / middle
                    recent_widths.append(width)
                except (IndexError, ValueError):
                    continue
            
            if len(recent_widths) >= 2:
                # Check if width is increasing
                return recent_widths[-1] > recent_widths[-2] * 1.1  # 10% increase
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error detecting band expansion for {symbol}: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        base_info = self.get_status()
        
        strategy_specific = {
            'strategy_type': 'bollinger_bands',
            'parameters': {
                'period': self.period,
                'std_dev': self.std_dev,
                'compression_threshold': self.compression_threshold,
                'position_size': self.position_size
            },
            'description': 'Trades on Bollinger Band compression indicating potential breakouts'
        }
        
        base_info.update(strategy_specific)
        return base_info 