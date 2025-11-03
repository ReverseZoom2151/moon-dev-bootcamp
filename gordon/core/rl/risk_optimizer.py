"""
Dynamic Risk Optimizer
======================
RL component that learns optimal risk parameters based on market conditions.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

from .base import BaseRLComponent

logger = logging.getLogger(__name__)


class RiskOptimizer(BaseRLComponent):
    """
    RL component that optimizes risk parameters dynamically.
    
    Actions represent risk levels:
        - 0: Very Conservative (0.5% risk per trade)
        - 1: Conservative (1% risk per trade)
        - 2: Moderate (2% risk per trade)
        - 3: Aggressive (3% risk per trade)
        - 4: Very Aggressive (5% risk per trade)
    """
    
    RISK_LEVELS = [0.005, 0.01, 0.02, 0.03, 0.05]  # Risk percentages per trade
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize risk optimizer.
        
        Args:
            config: Configuration dictionary
        """
        # State: portfolio state + market conditions + recent performance
        state_size = 20  # portfolio metrics + market features + performance history
        
        super().__init__(
            name="risk_optimizer",
            state_size=state_size,
            action_size=len(self.RISK_LEVELS),  # 5 risk levels
            model_dir=config.get('model_dir', './models/rl') if config else './models/rl',
            config=config or {}
        )
        
        # Current risk level
        self.current_risk_level = 2  # Default: Moderate (2%)
        self.current_risk_pct = self.RISK_LEVELS[self.current_risk_level]
        
        # Performance tracking
        self.risk_performance_history: List[Dict[str, Any]] = []
    
    def get_state(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract state vector from context.
        
        State includes:
        - Portfolio metrics (balance, drawdown, leverage, etc.)
        - Market conditions (volatility, trend, etc.)
        - Recent performance (win rate, Sharpe ratio, etc.)
        """
        portfolio = context.get('portfolio', {})
        market = context.get('market', {})
        performance = context.get('performance', {})
        
        # Portfolio features
        balance = portfolio.get('balance', 10000.0)
        drawdown = portfolio.get('drawdown', 0.0)
        drawdown_pct = portfolio.get('drawdown_pct', 0.0)
        leverage = portfolio.get('leverage', 1.0)
        position_count = portfolio.get('position_count', 0)
        margin_used = portfolio.get('margin_used', 0.0)
        available_balance = portfolio.get('available_balance', balance)
        
        # Market features
        volatility = market.get('volatility', 0.0)
        trend = market.get('trend', 0.0)
        volume = market.get('volume_ratio', 1.0)
        price_change = market.get('price_change_24h', 0.0)
        fear_greed = market.get('fear_greed_index', 50.0) / 100.0
        
        # Performance features
        win_rate = performance.get('win_rate', 0.5)
        sharpe_ratio = performance.get('sharpe_ratio', 0.0)
        total_pnl = performance.get('total_pnl', 0.0)
        total_pnl_pct = performance.get('total_pnl_pct', 0.0)
        daily_pnl = performance.get('daily_pnl', 0.0)
        daily_pnl_pct = performance.get('daily_pnl_pct', 0.0)
        consecutive_losses = performance.get('consecutive_losses', 0)
        
        # Normalize features
        state = np.array([
            balance / 100000.0,  # Normalize to 0-1 scale
            drawdown,
            drawdown_pct,
            leverage / 10.0,  # Normalize leverage
            position_count / 10.0,  # Normalize position count
            margin_used / balance if balance > 0 else 0.0,
            available_balance / balance if balance > 0 else 1.0,
            volatility,
            trend,
            volume,
            price_change,
            fear_greed,
            win_rate,
            sharpe_ratio / 3.0,  # Normalize Sharpe (typically -3 to 3)
            total_pnl / balance if balance > 0 else 0.0,
            total_pnl_pct,
            daily_pnl / balance if balance > 0 else 0.0,
            daily_pnl_pct,
            consecutive_losses / 10.0,  # Normalize
            1.0  # Bias
        ])
        
        return state.astype(np.float32)
    
    def calculate_reward(self, action: int, context: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calculate reward based on risk-adjusted performance.
        
        Reward considers:
        - Risk-adjusted returns (Sharpe ratio)
        - Drawdown control
        - Capital preservation
        """
        risk_level = self.RISK_LEVELS[action]
        
        # Get performance metrics
        sharpe_ratio = result.get('sharpe_ratio', 0.0)
        drawdown_pct = result.get('drawdown_pct', 0.0)
        total_pnl_pct = result.get('total_pnl_pct', 0.0)
        win_rate = result.get('win_rate', 0.5)
        
        # Base reward from Sharpe ratio (risk-adjusted returns)
        reward = sharpe_ratio * 10
        
        # Bonus for good returns
        reward += total_pnl_pct * 5
        
        # Strong penalty for high drawdown
        if drawdown_pct > 0.1:  # 10% drawdown
            reward -= drawdown_pct * 50
        
        # Penalty for excessive drawdown
        if drawdown_pct > 0.2:  # 20% drawdown
            reward -= 100  # Large penalty
        
        # Reward for appropriate risk level given market conditions
        volatility = context.get('market', {}).get('volatility', 0.0)
        if volatility > 0.05 and action < 2:  # High volatility, should be conservative
            reward += 5
        elif volatility < 0.02 and action > 2:  # Low volatility, can be aggressive
            reward += 5
        
        # Penalty for wrong risk level
        if drawdown_pct > 0.1 and action > 2:  # High drawdown but still aggressive
            reward -= 20
        
        # Reward for capital preservation
        balance = context.get('portfolio', {}).get('balance', 10000.0)
        initial_balance = context.get('portfolio', {}).get('initial_balance', balance)
        if balance >= initial_balance:
            reward += 2
        
        return reward
    
    def optimize_risk(self, portfolio_context: Dict[str, Any], market_context: Dict[str, Any], 
                     performance_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize risk parameters based on current conditions.
        
        Returns:
            Dictionary with optimized risk parameters
        """
        context = {
            'portfolio': portfolio_context,
            'market': market_context,
            'performance': performance_context
        }
        
        prediction = self.predict(context)
        action = prediction['action']
        risk_level = self.RISK_LEVELS[action]
        
        # Update current risk level
        self.current_risk_level = action
        self.current_risk_pct = risk_level
        
        # Calculate derived risk parameters
        max_position_size = risk_level * 5  # Max position size as multiple of risk
        max_leverage = min(3.0, 1.0 + risk_level * 10)  # Conservative leverage
        stop_loss_pct = risk_level * 2  # Stop loss based on risk
        
        result = {
            'risk_per_trade': risk_level,
            'risk_level': action,
            'risk_level_name': ['Very Conservative', 'Conservative', 'Moderate', 'Aggressive', 'Very Aggressive'][action],
            'max_position_size': max_position_size,
            'max_leverage': max_leverage,
            'stop_loss_pct': stop_loss_pct,
            'confidence': prediction['confidence'],
            'recommendation': 'increase' if action > self.current_risk_level else 'decrease' if action < self.current_risk_level else 'maintain'
        }
        
        self.risk_performance_history.append(result)
        return result
    
    def get_current_risk(self) -> float:
        """Get current risk per trade percentage."""
        return self.current_risk_pct

