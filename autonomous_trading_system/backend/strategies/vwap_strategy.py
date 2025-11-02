"""
VWAP Strategy - Based on Day_12_Projects implementation
Trades based on Volume Weighted Average Price deviations and probabilities
"""

import logging
import numpy as np
import random
from typing import Dict, Any, Optional
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class VWAPStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    VWAP Strategy Implementation
    
    Strategy Logic:
    - Calculate Volume Weighted Average Price (VWAP)
    - Compare current price to VWAP
    - Use probabilistic approach for direction selection
    - Higher probability of long when price > VWAP
    - Lower probability of long when price < VWAP
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "VWAP"):
        super().__init__(config, market_data_manager, name)
        
        # Strategy-specific configuration
        self.deviation_threshold = config.get("deviation_threshold", 0.002)  # 0.2%
        self.volume_threshold = config.get("volume_threshold", 1000000)
        self.long_prob_above_vwap = config.get("long_prob_above_vwap", 0.7)
        self.long_prob_below_vwap = config.get("long_prob_below_vwap", 0.3)
        
        # Minimum data points required
        self.min_data_points = 50
        
        logger.info(f"🔧 VWAP Strategy initialized:")
        logger.info(f"   Deviation Threshold: {self.deviation_threshold * 100}%")
        logger.info(f"   Volume Threshold: {self.volume_threshold:,}")
        logger.info(f"   Long Prob Above VWAP: {self.long_prob_above_vwap}")
        logger.info(f"   Long Prob Below VWAP: {self.long_prob_below_vwap}")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize strategy-specific components"""
        # Validate configuration
        if not 0 < self.deviation_threshold < 1:
            raise ValueError("Deviation threshold must be between 0 and 1")
        
        if self.volume_threshold <= 0:
            raise ValueError("Volume threshold must be positive")
        
        if not 0 <= self.long_prob_above_vwap <= 1:
            raise ValueError("Long probability above VWAP must be between 0 and 1")
        
        if not 0 <= self.long_prob_below_vwap <= 1:
            raise ValueError("Long probability below VWAP must be between 0 and 1")
        
        logger.info("✅ VWAP strategy validation complete")
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate trading signal based on VWAP analysis"""
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
        """Analyze a specific symbol for VWAP opportunities"""
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
            
            # Generate signal based on VWAP analysis
            signal = await self._evaluate_vwap(symbol, data, indicators)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str, data: Any) -> Dict[str, Any]:
        """Calculate VWAP and related indicators"""
        try:
            # Calculate VWAP
            vwap = self.calculate_vwap(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            if vwap is None or vwap.isna().all():
                logger.warning(f"⚠️ Could not calculate VWAP for {symbol}")
                return {}
            
            # Get the latest values
            current_price = float(data['close'].iloc[-1])
            current_vwap = float(vwap.dropna().iloc[-1])
            current_volume = float(data['volume'].iloc[-1])
            
            # Calculate price deviation from VWAP
            price_deviation = (current_price - current_vwap) / current_vwap
            price_deviation_pct = abs(price_deviation) * 100
            
            # Determine price position relative to VWAP
            above_vwap = current_price > current_vwap
            
            indicators = {
                'current_price': current_price,
                'current_vwap': current_vwap,
                'current_volume': current_volume,
                'price_deviation': price_deviation,
                'price_deviation_pct': price_deviation_pct,
                'above_vwap': above_vwap,
                'vwap_series': vwap
            }
            
            logger.debug(f"📊 {symbol} VWAP indicators: Price={current_price:.4f}, "
                        f"VWAP={current_vwap:.4f}, Deviation={price_deviation:.4f}, "
                        f"Volume={current_volume:,.0f}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {symbol}: {e}")
            return {}
    
    async def _evaluate_vwap(self, symbol: str, data: Any, indicators: Dict[str, Any]) -> Optional[StrategySignal]:
        """Evaluate VWAP conditions and generate signal"""
        try:
            current_price = indicators['current_price']
            current_vwap = indicators['current_vwap']
            above_vwap = indicators['above_vwap']
            price_deviation_pct = indicators['price_deviation_pct']
            current_volume = indicators['current_volume']
            
            # Check volume threshold
            if current_volume < self.volume_threshold:
                logger.debug(f"Volume {current_volume:,.0f} below threshold {self.volume_threshold:,}")
                return None
            
            # Determine action based on VWAP position and probability
            action = SignalAction.HOLD
            confidence = 0.0
            
            # Use probabilistic approach as in original strategy
            random_chance = random.random()
            
            if above_vwap:
                # Price above VWAP - higher probability of going long
                going_long = random_chance <= self.long_prob_above_vwap
                base_confidence = self.long_prob_above_vwap
            else:
                # Price below VWAP - lower probability of going long
                going_long = random_chance <= self.long_prob_below_vwap
                base_confidence = self.long_prob_below_vwap if going_long else (1 - self.long_prob_below_vwap)
            
            # Set action based on direction
            if going_long:
                action = SignalAction.BUY
            else:
                action = SignalAction.SELL
            
            # Calculate confidence based on deviation and volume
            deviation_factor = min(price_deviation_pct / 2.0, 1.0)  # Max factor at 2% deviation
            volume_factor = min(current_volume / (self.volume_threshold * 2), 1.0)  # Max factor at 2x threshold
            
            confidence = base_confidence * (0.5 + 0.3 * deviation_factor + 0.2 * volume_factor)
            confidence = min(confidence, 1.0)
            
            # Only generate signal if confidence is above threshold
            min_confidence = 0.4
            if confidence < min_confidence:
                return None
            
            # Create metadata with strategy details
            metadata = {
                'vwap': current_vwap,
                'price_deviation': indicators['price_deviation'],
                'price_deviation_pct': price_deviation_pct,
                'above_vwap': above_vwap,
                'volume': current_volume,
                'volume_threshold': self.volume_threshold,
                'random_chance': random_chance,
                'long_probability': self.long_prob_above_vwap if above_vwap else self.long_prob_below_vwap,
                'strategy_type': 'vwap'
            }
            
            # Create and return signal
            signal = self._create_signal(
                symbol=symbol,
                action=action,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
            
            logger.info(f"🎯 VWAP Signal: {action.value} {symbol} @ {current_price:.4f} "
                       f"(confidence: {confidence:.2f}, VWAP: {current_vwap:.4f}, "
                       f"above_vwap: {above_vwap})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error evaluating VWAP for {symbol}: {e}")
            return None
    
    def _calculate_stop_loss(self, entry_price: float, action: SignalAction) -> float:
        """Calculate stop loss for VWAP strategy"""
        # Conservative stop loss for VWAP mean reversion
        stop_loss_pct = 0.02  # 2% stop loss
        
        if action == SignalAction.BUY:
            return entry_price * (1 - stop_loss_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 + stop_loss_pct)
        
        return entry_price
    
    def _calculate_take_profit(self, entry_price: float, action: SignalAction) -> float:
        """Calculate take profit for VWAP strategy"""
        # Target profit when price reverts to VWAP
        take_profit_pct = 0.015  # 1.5% take profit
        
        if action == SignalAction.BUY:
            return entry_price * (1 + take_profit_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 - take_profit_pct)
        
        return entry_price
    
    async def _get_vwap_deviation_strength(self, symbol: str) -> float:
        """Calculate the strength of VWAP deviation"""
        try:
            data = await self._get_market_data(symbol, limit=20)
            if data is None or len(data) < 10:
                return 0.0
            
            # Calculate recent VWAP deviations
            vwap = self.calculate_vwap(
                data['high'], data['low'], data['close'], data['volume']
            )
            
            if vwap is None:
                return 0.0
            
            # Calculate average deviation over recent periods
            deviations = []
            for i in range(-10, 0):  # Last 10 periods
                try:
                    price = float(data['close'].iloc[i])
                    vwap_val = float(vwap.iloc[i])
                    deviation = abs(price - vwap_val) / vwap_val
                    deviations.append(deviation)
                except (IndexError, ValueError):
                    continue
            
            if deviations:
                avg_deviation = sum(deviations) / len(deviations)
                return min(avg_deviation * 10, 1.0)  # Scale to 0-1 range
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating VWAP deviation strength for {symbol}: {e}")
            return 0.0
    
    async def _check_volume_surge(self, symbol: str) -> bool:
        """Check if there's a volume surge indicating strong momentum"""
        try:
            data = await self._get_market_data(symbol, limit=20)
            if data is None or len(data) < 10:
                return False
            
            # Calculate average volume over recent periods
            recent_volumes = data['volume'].tail(10).values
            avg_volume = np.mean(recent_volumes[:-1])  # Exclude current volume
            current_volume = recent_volumes[-1]
            
            # Check if current volume is significantly higher
            volume_surge = current_volume > avg_volume * 1.5  # 50% above average
            
            return volume_surge
            
        except Exception as e:
            logger.error(f"❌ Error checking volume surge for {symbol}: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        base_info = self.get_status()
        
        strategy_specific = {
            'strategy_type': 'vwap',
            'parameters': {
                'deviation_threshold': self.deviation_threshold,
                'volume_threshold': self.volume_threshold,
                'long_prob_above_vwap': self.long_prob_above_vwap,
                'long_prob_below_vwap': self.long_prob_below_vwap
            },
            'description': 'Trades based on VWAP deviations with probabilistic direction selection'
        }
        
        base_info.update(strategy_specific)
        return base_info 