"""
StochRSI Strategy - Based on Day_16_Projects implementation
Trades based on Stochastic RSI momentum signals and crossovers
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class StochRSIStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    StochRSI Strategy Implementation
    
    Strategy Logic:
    - Calculate Stochastic RSI (StochRSI %K and %D lines)
    - Detect crossovers between %K and %D lines
    - Use overbought/oversold levels for entry signals
    - Combine with other indicators for confirmation
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "StochRSI"):
        super().__init__(config, market_data_manager, name)
        
        # Strategy-specific configuration
        self.rsi_period = config.get("rsi_period", 14)
        self.stoch_period = config.get("stoch_period", 14)
        self.smooth_k = config.get("smooth_k", 3)
        self.smooth_d = config.get("smooth_d", 3)
        self.overbought = config.get("overbought", 80)
        self.oversold = config.get("oversold", 20)
        
        # Minimum data points required
        self.min_data_points = max(self.rsi_period + self.stoch_period + 20, 80)
        
        logger.info(f"🔧 StochRSI Strategy initialized:")
        logger.info(f"   RSI Period: {self.rsi_period}")
        logger.info(f"   Stoch Period: {self.stoch_period}")
        logger.info(f"   Smooth K: {self.smooth_k}, Smooth D: {self.smooth_d}")
        logger.info(f"   Overbought: {self.overbought}, Oversold: {self.oversold}")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize strategy-specific components"""
        # Validate configuration
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        
        if self.stoch_period <= 0:
            raise ValueError("Stochastic period must be positive")
        
        if not 0 < self.oversold < self.overbought < 100:
            raise ValueError("Invalid overbought/oversold levels")
        
        logger.info("✅ StochRSI strategy validation complete")
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate trading signal based on StochRSI analysis"""
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
        """Analyze a specific symbol for StochRSI opportunities"""
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
            
            # Generate signal based on StochRSI analysis
            signal = await self._evaluate_stochrsi(symbol, data, indicators)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str, data: Any) -> Dict[str, Any]:
        """Calculate StochRSI and related indicators"""
        try:
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Calculate RSI first
            rsi = self.calculate_rsi(df['close'], self.rsi_period)
            
            if rsi is None or rsi.isna().all():
                logger.warning(f"⚠️ Could not calculate RSI for {symbol}")
                return {}
            
            # Calculate Stochastic RSI
            stoch_k, stoch_d = self._calculate_stochastic_rsi(rsi)
            
            if stoch_k is None or stoch_d is None:
                logger.warning(f"⚠️ Could not calculate StochRSI for {symbol}")
                return {}
            
            # Get the latest values
            current_price = float(df['close'].iloc[-1])
            current_rsi = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50
            current_stoch_k = float(stoch_k.dropna().iloc[-1]) if not stoch_k.dropna().empty else 50
            current_stoch_d = float(stoch_d.dropna().iloc[-1]) if not stoch_d.dropna().empty else 50
            
            # Check for crossovers
            crossover_bullish = self._detect_bullish_crossover(stoch_k, stoch_d)
            crossover_bearish = self._detect_bearish_crossover(stoch_k, stoch_d)
            
            # Determine overbought/oversold conditions
            is_oversold = current_stoch_k < self.oversold and current_stoch_d < self.oversold
            is_overbought = current_stoch_k > self.overbought and current_stoch_d > self.overbought
            
            indicators = {
                'current_price': current_price,
                'current_rsi': current_rsi,
                'current_stoch_k': current_stoch_k,
                'current_stoch_d': current_stoch_d,
                'crossover_bullish': crossover_bullish,
                'crossover_bearish': crossover_bearish,
                'is_oversold': is_oversold,
                'is_overbought': is_overbought,
                'rsi_series': rsi,
                'stoch_k_series': stoch_k,
                'stoch_d_series': stoch_d
            }
            
            logger.debug(f"📊 {symbol} StochRSI indicators: Price={current_price:.4f}, "
                        f"RSI={current_rsi:.2f}, StochK={current_stoch_k:.2f}, "
                        f"StochD={current_stoch_d:.2f}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {symbol}: {e}")
            return {}
    
    def _calculate_stochastic_rsi(self, rsi_series: pd.Series) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Calculate Stochastic RSI %K and %D lines"""
        try:
            # Calculate %K line
            rsi_min = rsi_series.rolling(window=self.stoch_period).min()
            rsi_max = rsi_series.rolling(window=self.stoch_period).max()
            
            # Avoid division by zero
            rsi_range = rsi_max - rsi_min
            stoch_k_raw = ((rsi_series - rsi_min) / rsi_range.replace(0, 1)) * 100
            
            # Smooth %K line
            stoch_k = stoch_k_raw.rolling(window=self.smooth_k).mean()
            
            # Calculate %D line (smoothed %K)
            stoch_d = stoch_k.rolling(window=self.smooth_d).mean()
            
            return stoch_k, stoch_d
            
        except Exception as e:
            logger.error(f"❌ Error calculating Stochastic RSI: {e}")
            return None, None
    
    def _detect_bullish_crossover(self, stoch_k: pd.Series, stoch_d: pd.Series) -> bool:
        """Detect bullish crossover (%K crosses above %D)"""
        try:
            if len(stoch_k) < 2 or len(stoch_d) < 2:
                return False
            
            # Check if %K crossed above %D in the last period
            k_current = stoch_k.iloc[-1]
            k_previous = stoch_k.iloc[-2]
            d_current = stoch_d.iloc[-1]
            d_previous = stoch_d.iloc[-2]
            
            # Bullish crossover: %K was below %D and is now above
            bullish_cross = (k_previous <= d_previous) and (k_current > d_current)
            
            return bullish_cross
            
        except Exception as e:
            logger.error(f"❌ Error detecting bullish crossover: {e}")
            return False
    
    def _detect_bearish_crossover(self, stoch_k: pd.Series, stoch_d: pd.Series) -> bool:
        """Detect bearish crossover (%K crosses below %D)"""
        try:
            if len(stoch_k) < 2 or len(stoch_d) < 2:
                return False
            
            # Check if %K crossed below %D in the last period
            k_current = stoch_k.iloc[-1]
            k_previous = stoch_k.iloc[-2]
            d_current = stoch_d.iloc[-1]
            d_previous = stoch_d.iloc[-2]
            
            # Bearish crossover: %K was above %D and is now below
            bearish_cross = (k_previous >= d_previous) and (k_current < d_current)
            
            return bearish_cross
            
        except Exception as e:
            logger.error(f"❌ Error detecting bearish crossover: {e}")
            return False
    
    async def _evaluate_stochrsi(self, symbol: str, data: Any, indicators: Dict[str, Any]) -> Optional[StrategySignal]:
        """Evaluate StochRSI conditions and generate signal"""
        try:
            current_price = indicators['current_price']
            current_stoch_k = indicators['current_stoch_k']
            current_stoch_d = indicators['current_stoch_d']
            crossover_bullish = indicators['crossover_bullish']
            crossover_bearish = indicators['crossover_bearish']
            is_oversold = indicators['is_oversold']
            is_overbought = indicators['is_overbought']
            
            # Determine action based on StochRSI signals
            action = SignalAction.HOLD
            confidence = 0.0
            signal_reason = ""
            
            # Bullish signal: Bullish crossover in oversold territory
            if crossover_bullish and is_oversold:
                action = SignalAction.BUY
                confidence = 0.8
                signal_reason = "Bullish crossover in oversold territory"
            
            # Moderate bullish signal: Bullish crossover (not necessarily oversold)
            elif crossover_bullish:
                action = SignalAction.BUY
                confidence = 0.6
                signal_reason = "Bullish crossover"
            
            # Bearish signal: Bearish crossover in overbought territory
            elif crossover_bearish and is_overbought:
                action = SignalAction.SELL
                confidence = 0.8
                signal_reason = "Bearish crossover in overbought territory"
            
            # Moderate bearish signal: Bearish crossover (not necessarily overbought)
            elif crossover_bearish:
                action = SignalAction.SELL
                confidence = 0.6
                signal_reason = "Bearish crossover"
            
            # Additional signals based on extreme levels
            elif is_oversold and current_stoch_k > current_stoch_d:
                # In oversold territory with %K above %D
                action = SignalAction.BUY
                confidence = 0.5
                signal_reason = "Oversold with K > D"
            
            elif is_overbought and current_stoch_k < current_stoch_d:
                # In overbought territory with %K below %D
                action = SignalAction.SELL
                confidence = 0.5
                signal_reason = "Overbought with K < D"
            
            # Only generate signal if confidence is above threshold
            min_confidence = 0.4
            if confidence < min_confidence:
                return None
            
            # Create metadata with strategy details
            metadata = {
                'stoch_k': current_stoch_k,
                'stoch_d': current_stoch_d,
                'rsi': indicators['current_rsi'],
                'crossover_bullish': crossover_bullish,
                'crossover_bearish': crossover_bearish,
                'is_oversold': is_oversold,
                'is_overbought': is_overbought,
                'signal_reason': signal_reason,
                'overbought_level': self.overbought,
                'oversold_level': self.oversold,
                'strategy_type': 'stochrsi'
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
                
                logger.info(f"🎯 StochRSI Signal: {action.value} {symbol} @ {current_price:.4f} "
                           f"(confidence: {confidence:.2f}, reason: {signal_reason})")
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error evaluating StochRSI for {symbol}: {e}")
            return None
    
    def _calculate_stop_loss(self, entry_price: float, action: SignalAction) -> float:
        """Calculate stop loss for StochRSI strategy"""
        # Conservative stop loss for momentum strategies
        stop_loss_pct = 0.025  # 2.5% stop loss
        
        if action == SignalAction.BUY:
            return entry_price * (1 - stop_loss_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 + stop_loss_pct)
        
        return entry_price
    
    def _calculate_take_profit(self, entry_price: float, action: SignalAction) -> float:
        """Calculate take profit for StochRSI strategy"""
        # Target profit based on momentum continuation
        take_profit_pct = 0.04  # 4% take profit
        
        if action == SignalAction.BUY:
            return entry_price * (1 + take_profit_pct)
        elif action == SignalAction.SELL:
            return entry_price * (1 - take_profit_pct)
        
        return entry_price
    
    async def _get_momentum_strength(self, symbol: str) -> float:
        """Calculate momentum strength based on recent StochRSI movements"""
        try:
            data = await self._get_market_data(symbol, limit=20)
            if data is None or len(data) < 10:
                return 0.0
            
            # Calculate recent RSI
            rsi = self.calculate_rsi(data['close'], self.rsi_period)
            if rsi is None:
                return 0.0
            
            # Calculate StochRSI
            stoch_k, stoch_d = self._calculate_stochastic_rsi(rsi)
            if stoch_k is None or stoch_d is None:
                return 0.0
            
            # Calculate momentum based on recent changes
            recent_k_values = stoch_k.dropna().tail(5).values
            if len(recent_k_values) < 2:
                return 0.0
            
            # Calculate average change in %K
            k_changes = np.diff(recent_k_values)
            avg_change = np.mean(np.abs(k_changes))
            
            # Normalize to 0-1 range
            momentum_strength = min(avg_change / 20, 1.0)  # Max strength at 20 point change
            
            return momentum_strength
            
        except Exception as e:
            logger.error(f"❌ Error calculating momentum strength for {symbol}: {e}")
            return 0.0
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information"""
        base_info = self.get_status()
        
        strategy_specific = {
            'strategy_type': 'stochrsi',
            'parameters': {
                'rsi_period': self.rsi_period,
                'stoch_period': self.stoch_period,
                'smooth_k': self.smooth_k,
                'smooth_d': self.smooth_d,
                'overbought': self.overbought,
                'oversold': self.oversold
            },
            'description': 'Trades based on Stochastic RSI momentum signals and crossovers'
        }
        
        base_info.update(strategy_specific)
        return base_info 