"""
Position Size Optimizer
=======================
RL component that learns optimal position sizes based on market conditions and strategy confidence.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

from .base import BaseRLComponent

logger = logging.getLogger(__name__)


class PositionSizeOptimizer(BaseRLComponent):
    """
    RL component that optimizes position sizes dynamically.
    
    Actions represent size multipliers:
        - 0: 0.25x (quarter size)
        - 1: 0.5x (half size)
        - 2: 1.0x (normal size)
        - 3: 1.5x (1.5x size)
        - 4: 2.0x (double size)
    """
    
    SIZE_MULTIPLIERS = [0.25, 0.5, 1.0, 1.5, 2.0]
    
    def __init__(self, base_size: float = 0.01, config: Optional[Dict] = None):
        """
        Initialize position size optimizer.
        
        Args:
            base_size: Base position size (default 1% of portfolio)
            config: Configuration dictionary
        """
        # State: signal confidence + market conditions + portfolio state
        state_size = 15
        
        super().__init__(
            name="position_size_optimizer",
            state_size=state_size,
            action_size=len(self.SIZE_MULTIPLIERS),  # 5 size levels
            model_dir=config.get('model_dir', './models/rl') if config else './models/rl',
            config=config or {}
        )
        
        self.base_size = base_size
        self.current_multiplier = 1.0
    
    def get_state(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract state vector from context.
        
        State includes:
        - Signal confidence and strength
        - Market conditions (volatility, trend, etc.)
        - Portfolio state (available balance, correlation, etc.)
        """
        signal = context.get('signal', {})
        market = context.get('market', {})
        portfolio = context.get('portfolio', {})
        
        # Signal features
        confidence = signal.get('confidence', 0.5)
        signal_strength = signal.get('strength', 0.0)  # -1 to 1
        strategy_performance = signal.get('strategy_performance', 0.0)
        recent_win_rate = signal.get('recent_win_rate', 0.5)
        
        # Market features
        volatility = market.get('volatility', 0.0)
        trend = market.get('trend', 0.0)
        volume = market.get('volume_ratio', 1.0)
        liquidity = market.get('liquidity', 1.0)
        spread = market.get('spread', 0.0)
        
        # Portfolio features
        available_balance = portfolio.get('available_balance', 10000.0)
        current_positions = portfolio.get('position_count', 0)
        correlation = portfolio.get('correlation', 0.0)
        max_position_size = portfolio.get('max_position_size', 0.1)
        drawdown = portfolio.get('drawdown_pct', 0.0)
        
        # Normalize features
        state = np.array([
            confidence,
            signal_strength,
            strategy_performance,
            recent_win_rate,
            volatility,
            trend,
            volume,
            liquidity,
            spread,
            available_balance / 100000.0,  # Normalize
            current_positions / 10.0,  # Normalize
            correlation,
            max_position_size,
            drawdown,
            1.0  # Bias
        ])
        
        return state.astype(np.float32)
    
    def calculate_reward(self, action: int, context: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calculate reward based on position size performance.
        
        Reward considers:
        - Risk-adjusted returns per position
        - Capital efficiency
        - Drawdown control
        """
        multiplier = self.SIZE_MULTIPLIERS[action]
        position_size = self.base_size * multiplier
        
        # Get performance metrics
        pnl = result.get('pnl', 0.0)
        pnl_pct = result.get('pnl_pct', 0.0)
        roi = result.get('roi', 0.0)  # Return on investment for this position
        
        # Base reward from ROI
        reward = roi * 10
        
        # Bonus for efficient capital usage
        if multiplier > 1.0 and pnl_pct > 0.02:  # Used more capital, got good return
            reward += 5
        
        # Penalty for over-sizing
        if multiplier > 1.5 and pnl_pct < -0.02:  # Over-sized and lost
            reward -= abs(pnl_pct) * 30
        
        # Penalty for under-sizing good opportunities
        confidence = context.get('signal', {}).get('confidence', 0.5)
        if multiplier < 1.0 and confidence > 0.8 and pnl_pct > 0.02:  # Missed opportunity
            reward -= 5
        
        # Reward for appropriate sizing given volatility
        volatility = context.get('market', {}).get('volatility', 0.0)
        if volatility > 0.05 and multiplier > 1.0:  # High volatility but large position
            reward -= 10
        elif volatility < 0.02 and multiplier < 1.0:  # Low volatility but small position
            reward -= 5
        
        # Reward for correlation awareness
        correlation = context.get('portfolio', {}).get('correlation', 0.0)
        if correlation > 0.7 and multiplier > 1.0:  # High correlation but large position
            reward -= 10
        
        return reward
    
    def optimize_size(self, signal: Dict[str, Any], market_context: Dict[str, Any], 
                     portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize position size based on signal and context.
        
        Returns:
            Dictionary with optimized position size and multiplier
        """
        context = {
            'signal': signal,
            'market': market_context,
            'portfolio': portfolio_context
        }
        
        prediction = self.predict(context)
        action = prediction['action']
        multiplier = self.SIZE_MULTIPLIERS[action]
        optimized_size = self.base_size * multiplier
        
        self.current_multiplier = multiplier
        
        result = {
            'size_multiplier': multiplier,
            'position_size': optimized_size,
            'base_size': self.base_size,
            'size_name': ['Quarter', 'Half', 'Normal', '1.5x', 'Double'][action],
            'confidence': prediction['confidence'],
            'recommendation': 'increase' if action > 2 else 'decrease' if action < 2 else 'maintain'
        }
        
        return result
    
    def get_current_multiplier(self) -> float:
        """Get current size multiplier."""
        return self.current_multiplier

