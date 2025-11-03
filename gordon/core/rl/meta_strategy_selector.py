"""
Meta-Strategy Selector
=======================
RL component that learns which strategy to use based on market conditions.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

from .base import BaseRLComponent

logger = logging.getLogger(__name__)


class MetaStrategySelector(BaseRLComponent):
    """
    RL component that selects the best strategy based on market conditions.
    
    Actions:
        - 0: SMA Strategy
        - 1: RSI Strategy
        - 2: VWAP Strategy
        - 3: Bollinger Strategy
        - 4: Breakout Strategy
        - 5: Mean Reversion Strategy
        - 6: DQN Strategy
        - 7: Market Maker Strategy
        - ... (one action per available strategy)
    """
    
    def __init__(self, available_strategies: List[str], config: Optional[Dict] = None):
        """
        Initialize meta-strategy selector.
        
        Args:
            available_strategies: List of available strategy names
            config: Configuration dictionary
        """
        self.available_strategies = available_strategies
        self.strategy_to_index = {s: i for i, s in enumerate(available_strategies)}
        self.index_to_strategy = {i: s for s, i in self.strategy_to_index.items()}
        
        # State: market features + strategy performance history
        state_size = 15 + len(available_strategies) * 3  # market features + per-strategy metrics
        
        super().__init__(
            name="meta_strategy_selector",
            state_size=state_size,
            action_size=len(available_strategies),
            model_dir=config.get('model_dir', './models/rl') if config else './models/rl',
            config=config or {}
        )
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, Dict] = {}
        for strategy in available_strategies:
            self.strategy_performance[strategy] = {
                'win_rate': 0.5,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'recent_signals': []
            }
    
    def get_state(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract state vector from context.
        
        State includes:
        - Market features (volatility, trend, volume, etc.)
        - Strategy performance metrics
        """
        # Market features
        market = context.get('market', {})
        volatility = market.get('volatility', 0.0)
        trend = market.get('trend', 0.0)  # -1 to 1
        volume_ratio = market.get('volume_ratio', 1.0)
        price_change = market.get('price_change_24h', 0.0)
        rsi = market.get('rsi', 50.0) / 100.0  # Normalize to 0-1
        macd = market.get('macd', 0.0)
        bb_position = market.get('bb_position', 0.5)  # Position in Bollinger Bands
        spread = market.get('spread', 0.0)
        liquidity = market.get('liquidity', 1.0)
        time_of_day = market.get('time_of_day', 0.5)  # 0-1 (0=midnight, 1=midnight)
        day_of_week = market.get('day_of_week', 3) / 7.0  # 0-1
        market_regime = market.get('regime', 0.5)  # 0=ranging, 1=trending
        fear_greed = market.get('fear_greed_index', 50.0) / 100.0  # Normalize to 0-1
        correlation = market.get('correlation', 0.0)
        
        market_features = np.array([
            volatility,
            trend,
            volume_ratio,
            price_change,
            rsi,
            macd,
            bb_position,
            spread,
            liquidity,
            time_of_day,
            day_of_week,
            market_regime,
            fear_greed,
            correlation,
            1.0  # Bias term
        ])
        
        # Strategy performance metrics
        strategy_features = []
        for strategy in self.available_strategies:
            perf = self.strategy_performance[strategy]
            strategy_features.extend([
                perf['win_rate'],
                perf['avg_return'],
                perf['sharpe_ratio']
            ])
        
        state = np.concatenate([market_features, np.array(strategy_features)])
        return state.astype(np.float32)
    
    def calculate_reward(self, action: int, context: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calculate reward based on strategy performance.
        
        Reward considers:
        - PnL of selected strategy
        - Risk-adjusted returns (Sharpe ratio)
        - Comparison to other strategies
        """
        selected_strategy = self.index_to_strategy[action]
        
        # Get performance metrics
        pnl = result.get('pnl', 0.0)
        pnl_pct = result.get('pnl_pct', 0.0)
        sharpe_ratio = result.get('sharpe_ratio', 0.0)
        win_rate = result.get('win_rate', 0.5)
        
        # Base reward from PnL
        reward = pnl_pct * 10  # Scale up percentage returns
        
        # Add Sharpe ratio component (risk-adjusted)
        reward += sharpe_ratio * 2
        
        # Add win rate component
        reward += (win_rate - 0.5) * 5  # Bonus for >50% win rate
        
        # Penalty for high drawdown
        drawdown = result.get('drawdown', 0.0)
        if drawdown > 0.1:  # 10% drawdown
            reward -= drawdown * 20
        
        # Bonus for outperforming other strategies
        other_strategies_pnl = result.get('other_strategies_pnl', {})
        if other_strategies_pnl:
            avg_other_pnl = np.mean(list(other_strategies_pnl.values()))
            if pnl_pct > avg_other_pnl:
                reward += (pnl_pct - avg_other_pnl) * 5
        
        # Update strategy performance
        self.strategy_performance[selected_strategy] = {
            'win_rate': win_rate,
            'avg_return': pnl_pct,
            'sharpe_ratio': sharpe_ratio,
            'recent_signals': result.get('recent_signals', [])
        }
        
        return reward
    
    def select_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select best strategy based on current market conditions.
        
        Returns:
            Dictionary with selected strategy name and confidence
        """
        prediction = self.predict(context)
        action = prediction['action']
        selected_strategy = self.index_to_strategy[action]
        
        return {
            'strategy': selected_strategy,
            'confidence': prediction['confidence'],
            'action_index': action,
            'all_probabilities': {
                self.index_to_strategy[i]: float(prediction['q_values'][i])
                for i in range(len(self.available_strategies))
            }
        }
    
    def update_strategy_performance(self, strategy_name: str, metrics: Dict[str, Any]):
        """Update performance metrics for a strategy."""
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name].update(metrics)

