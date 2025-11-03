"""
Market Regime Detector
======================
RL component that learns to identify market regimes.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

from .base import BaseRLComponent

logger = logging.getLogger(__name__)


class MarketRegimeDetector(BaseRLComponent):
    """
    RL component that detects market regimes.
    
    Regimes:
        - 0: Trending Up
        - 1: Trending Down
        - 2: Ranging/Balanced
        - 3: High Volatility
        - 4: Low Volatility
        - 5: Consolidation
        - 6: Breakout
    """
    
    REGIME_NAMES = [
        'Trending Up',
        'Trending Down',
        'Ranging',
        'High Volatility',
        'Low Volatility',
        'Consolidation',
        'Breakout'
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize market regime detector.
        
        Args:
            config: Configuration dictionary
        """
        # State: market features + price action patterns
        state_size = 20  # Market features + technical indicators
        
        super().__init__(
            name="market_regime_detector",
            state_size=state_size,
            action_size=len(self.REGIME_NAMES),  # 7 regimes
            model_dir=config.get('model_dir', './models/rl') if config else './models/rl',
            config=config or {}
        )
        
        self.current_regime = 2  # Default: Ranging
        self.regime_history: List[Dict] = []
    
    def get_state(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract state vector from context.
        
        State includes:
        - Price action (OHLCV data)
        - Technical indicators
        - Volume patterns
        - Volatility metrics
        """
        market = context.get('market', {})
        price_data = context.get('price_data', {})
        
        # Price action features
        price_change_1h = price_data.get('price_change_1h', 0.0)
        price_change_24h = price_data.get('price_change_24h', 0.0)
        price_change_7d = price_data.get('price_change_7d', 0.0)
        high_low_ratio = price_data.get('high_low_ratio', 1.0)
        
        # Technical indicators
        rsi = market.get('rsi', 50.0) / 100.0
        macd = market.get('macd', 0.0)
        macd_signal = market.get('macd_signal', 0.0)
        bb_position = market.get('bb_position', 0.5)
        sma_trend = market.get('sma_trend', 0.0)  # Price vs SMA
        
        # Volume features
        volume_ratio = market.get('volume_ratio', 1.0)
        volume_trend = market.get('volume_trend', 0.0)
        
        # Volatility features
        volatility = market.get('volatility', 0.0)
        volatility_ratio = market.get('volatility_ratio', 1.0)  # Current vs average
        
        # Trend strength
        trend_strength = market.get('trend_strength', 0.0)
        trend_direction = market.get('trend_direction', 0.0)  # -1 to 1
        
        # Support/Resistance
        support_distance = market.get('support_distance', 0.0)
        resistance_distance = market.get('resistance_distance', 0.0)
        
        # Market structure
        higher_highs = market.get('higher_highs', 0.0)  # -1 to 1
        higher_lows = market.get('higher_lows', 0.0)  # -1 to 1
        
        # Consolidation indicators
        consolidation_score = market.get('consolidation_score', 0.0)
        
        state = np.array([
            price_change_1h,
            price_change_24h,
            price_change_7d,
            high_low_ratio,
            rsi,
            macd,
            macd_signal,
            bb_position,
            sma_trend,
            volume_ratio,
            volume_trend,
            volatility,
            volatility_ratio,
            trend_strength,
            trend_direction,
            support_distance,
            resistance_distance,
            higher_highs,
            higher_lows,
            consolidation_score
        ])
        
        return state.astype(np.float32)
    
    def calculate_reward(self, action: int, context: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calculate reward based on regime detection accuracy.
        
        Reward considers:
        - How well the detected regime matches actual market behavior
        - Strategy performance in detected regime
        """
        detected_regime = action
        regime_name = self.REGIME_NAMES[detected_regime]
        
        # Get actual market behavior
        price_change = context.get('price_data', {}).get('price_change_24h', 0.0)
        volatility = context.get('market', {}).get('volatility', 0.0)
        trend = context.get('market', {}).get('trend', 0.0)
        
        reward = 0.0
        
        # Reward for correct regime detection
        if detected_regime == 0 and price_change > 0.02 and trend > 0.3:  # Trending Up
            reward += 10
        elif detected_regime == 1 and price_change < -0.02 and trend < -0.3:  # Trending Down
            reward += 10
        elif detected_regime == 2 and abs(price_change) < 0.01 and abs(trend) < 0.2:  # Ranging
            reward += 10
        elif detected_regime == 3 and volatility > 0.05:  # High Volatility
            reward += 10
        elif detected_regime == 4 and volatility < 0.01:  # Low Volatility
            reward += 10
        
        # Penalty for wrong detection
        if detected_regime == 0 and price_change < -0.02:  # Detected up but price down
            reward -= 10
        elif detected_regime == 1 and price_change > 0.02:  # Detected down but price up
            reward -= 10
        
        # Reward for regime-appropriate strategy performance
        strategy_performance = result.get('strategy_performance', 0.0)
        if strategy_performance > 0:
            reward += strategy_performance * 5
        
        return reward
    
    def detect_regime(self, market_context: Dict[str, Any], price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect current market regime.
        
        Returns:
            Dictionary with detected regime and confidence
        """
        context = {
            'market': market_context,
            'price_data': price_data
        }
        
        prediction = self.predict(context)
        action = prediction['action']
        regime_name = self.REGIME_NAMES[action]
        
        self.current_regime = action
        
        result = {
            'regime': action,
            'regime_name': regime_name,
            'confidence': prediction['confidence'],
            'all_probabilities': {
                self.REGIME_NAMES[i]: float(prediction['q_values'][i])
                for i in range(len(self.REGIME_NAMES))
            }
        }
        
        self.regime_history.append(result)
        return result
    
    def get_current_regime(self) -> Dict[str, Any]:
        """Get current market regime."""
        return {
            'regime': self.current_regime,
            'regime_name': self.REGIME_NAMES[self.current_regime]
        }

