"""
Signal Aggregator
=================
RL component that learns how to combine signals from multiple strategies.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

from .base import BaseRLComponent

logger = logging.getLogger(__name__)


class SignalAggregator(BaseRLComponent):
    """
    RL component that learns how to aggregate signals from multiple strategies.
    
    Actions:
        - 0: Reject all signals (HOLD)
        - 1: Accept buy signal (weighted)
        - 2: Accept sell signal (weighted)
    
    The component learns optimal weights for each strategy's signals.
    """
    
    def __init__(self, strategy_names: List[str], config: Optional[Dict] = None):
        """
        Initialize signal aggregator.
        
        Args:
            strategy_names: List of strategy names to aggregate
            config: Configuration dictionary
        """
        self.strategy_names = strategy_names
        
        # State: signals from all strategies + market context
        # Each strategy contributes: action (BUY=1, SELL=-1, HOLD=0), confidence, recent_performance
        state_size = 10 + len(strategy_names) * 3  # market features + per-strategy signal info
        
        super().__init__(
            name="signal_aggregator",
            state_size=state_size,
            action_size=3,  # HOLD, BUY, SELL
            model_dir=config.get('model_dir', './models/rl') if config else './models/rl',
            config=config or {}
        )
        
        # Signal weights (learned by RL, but can be initialized)
        self.signal_weights: Dict[str, float] = {s: 1.0 for s in strategy_names}
        
        # Aggregation history
        self.aggregation_history: List[Dict] = []
    
    def get_state(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract state vector from context.
        
        State includes:
        - Market features
        - Signals from all strategies (action, confidence, performance)
        """
        # Market features
        market = context.get('market', {})
        volatility = market.get('volatility', 0.0)
        trend = market.get('trend', 0.0)
        volume = market.get('volume_ratio', 1.0)
        price_change = market.get('price_change_24h', 0.0)
        rsi = market.get('rsi', 50.0) / 100.0
        spread = market.get('spread', 0.0)
        liquidity = market.get('liquidity', 1.0)
        market_regime = market.get('regime', 0.5)
        correlation = market.get('correlation', 0.0)
        
        market_features = np.array([
            volatility,
            trend,
            volume,
            price_change,
            rsi,
            spread,
            liquidity,
            market_regime,
            correlation,
            1.0  # Bias
        ])
        
        # Strategy signals
        signals = context.get('signals', {})
        signal_features = []
        
        for strategy in self.strategy_names:
            signal = signals.get(strategy, {})
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.0)
            recent_perf = signal.get('recent_performance', 0.0)
            
            # Encode action: BUY=1, SELL=-1, HOLD=0
            action_encoded = 0.0
            if action == 'BUY':
                action_encoded = 1.0
            elif action == 'SELL':
                action_encoded = -1.0
            
            signal_features.extend([action_encoded, confidence, recent_perf])
        
        state = np.concatenate([market_features, np.array(signal_features)])
        return state.astype(np.float32)
    
    def calculate_reward(self, action: int, context: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calculate reward based on aggregated signal performance.
        
        Reward considers:
        - PnL of aggregated signal
        - Whether aggregation improved over individual signals
        - Risk-adjusted returns
        """
        # Action: 0=HOLD, 1=BUY, 2=SELL
        action_taken = ['HOLD', 'BUY', 'SELL'][action]
        
        # Get performance metrics
        pnl = result.get('pnl', 0.0)
        pnl_pct = result.get('pnl_pct', 0.0)
        
        # Base reward from PnL
        reward = pnl_pct * 10
        
        # Compare to individual strategy performance
        individual_signals_pnl = result.get('individual_signals_pnl', {})
        if individual_signals_pnl:
            avg_individual_pnl = np.mean(list(individual_signals_pnl.values()))
            if pnl_pct > avg_individual_pnl:
                reward += (pnl_pct - avg_individual_pnl) * 10  # Bonus for outperforming
        
        # Penalty for wrong direction
        if action_taken == 'BUY' and pnl_pct < -0.02:  # -2% loss on buy
            reward -= abs(pnl_pct) * 20
        elif action_taken == 'SELL' and pnl_pct > 0.02:  # +2% gain when we sold
            reward -= abs(pnl_pct) * 20
        
        # Bonus for good aggregation (high confidence, correct prediction)
        if result.get('confidence', 0.0) > 0.7 and pnl_pct > 0:
            reward += 5
        
        return reward
    
    def aggregate_signals(self, signals: Dict[str, Dict], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate signals from multiple strategies.
        
        Args:
            signals: Dictionary mapping strategy names to their signals
            market_context: Market context data
            
        Returns:
            Aggregated signal with action, confidence, and weights
        """
        context = {
            'signals': signals,
            'market': market_context
        }
        
        prediction = self.predict(context)
        action_index = prediction['action']
        action = ['HOLD', 'BUY', 'SELL'][action_index]
        
        # Calculate weighted confidence
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for strategy, signal in signals.items():
            weight = self.signal_weights.get(strategy, 1.0)
            conf = signal.get('confidence', 0.0)
            weighted_confidence += conf * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_confidence /= total_weight
        
        # Use RL prediction confidence if higher
        final_confidence = max(weighted_confidence, prediction['confidence'])
        
        result = {
            'action': action,
            'confidence': final_confidence,
            'weighted_confidence': weighted_confidence,
            'rl_confidence': prediction['confidence'],
            'individual_signals': signals,
            'weights': self.signal_weights.copy(),
            'action_index': action_index
        }
        
        self.aggregation_history.append(result)
        return result
    
    def update_weights(self, strategy_name: str, weight: float):
        """Update weight for a strategy."""
        if strategy_name in self.signal_weights:
            self.signal_weights[strategy_name] = weight

