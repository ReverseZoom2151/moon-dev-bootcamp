"""
Portfolio Allocator
==================
RL component that learns optimal portfolio allocation across symbols and strategies.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

from .base import BaseRLComponent

logger = logging.getLogger(__name__)


class PortfolioAllocator(BaseRLComponent):
    """
    RL component that optimizes portfolio allocation.
    
    Actions represent allocation strategies:
        - 0: Conservative (70% cash, 30% positions)
        - 1: Balanced (50% cash, 50% positions)
        - 2: Aggressive (30% cash, 70% positions)
        - 3: Very Aggressive (10% cash, 90% positions)
        - 4: All-in (0% cash, 100% positions)
    """
    
    ALLOCATION_STRATEGIES = [
        {'cash': 0.70, 'positions': 0.30},  # Conservative
        {'cash': 0.50, 'positions': 0.50},  # Balanced
        {'cash': 0.30, 'positions': 0.70},  # Aggressive
        {'cash': 0.10, 'positions': 0.90},  # Very Aggressive
        {'cash': 0.0, 'positions': 1.0}     # All-in
    ]
    
    def __init__(self, symbols: List[str], config: Optional[Dict] = None):
        """
        Initialize portfolio allocator.
        
        Args:
            symbols: List of symbols to allocate across
            config: Configuration dictionary
        """
        self.symbols = symbols
        
        # State: portfolio state + symbol performance + market conditions
        state_size = 15 + len(symbols) * 2  # Base features + per-symbol metrics
        
        super().__init__(
            name="portfolio_allocator",
            state_size=state_size,
            action_size=len(self.ALLOCATION_STRATEGIES),  # 5 allocation strategies
            model_dir=config.get('model_dir', './models/rl') if config else './models/rl',
            config=config or {}
        )
        
        # Current allocation
        self.current_allocation = {symbol: 0.0 for symbol in symbols}
        self.current_strategy = 1  # Default: Balanced
        
        # Symbol performance tracking
        self.symbol_performance: Dict[str, Dict] = {}
        for symbol in symbols:
            self.symbol_performance[symbol] = {
                'return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'correlation': 0.0
            }
    
    def get_state(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract state vector from context.
        
        State includes:
        - Portfolio metrics
        - Symbol performance metrics
        - Market conditions
        """
        portfolio = context.get('portfolio', {})
        market = context.get('market', {})
        
        # Portfolio features
        total_balance = portfolio.get('balance', 10000.0)
        available_balance = portfolio.get('available_balance', total_balance)
        used_balance = total_balance - available_balance
        position_count = portfolio.get('position_count', 0)
        total_pnl = portfolio.get('total_pnl', 0.0)
        total_pnl_pct = portfolio.get('total_pnl_pct', 0.0)
        drawdown_pct = portfolio.get('drawdown_pct', 0.0)
        max_drawdown = portfolio.get('max_drawdown', 0.0)
        diversification_score = portfolio.get('diversification_score', 0.5)
        avg_correlation = portfolio.get('avg_correlation', 0.0)
        sharpe_ratio = portfolio.get('sharpe_ratio', 0.0)
        
        # Market features
        overall_trend = market.get('overall_trend', 0.0)
        overall_volatility = market.get('overall_volatility', 0.0)
        market_regime = market.get('regime', 0.5)
        fear_greed = market.get('fear_greed_index', 50.0) / 100.0
        
        base_features = np.array([
            total_balance / 100000.0,  # Normalize
            available_balance / total_balance if total_balance > 0 else 1.0,
            used_balance / total_balance if total_balance > 0 else 0.0,
            position_count / 10.0,  # Normalize
            total_pnl / total_balance if total_balance > 0 else 0.0,
            total_pnl_pct,
            drawdown_pct,
            max_drawdown,
            diversification_score,
            avg_correlation,
            sharpe_ratio / 3.0,  # Normalize
            overall_trend,
            overall_volatility,
            market_regime,
            fear_greed
        ])
        
        # Symbol features
        symbol_features = []
        for symbol in self.symbols:
            perf = self.symbol_performance.get(symbol, {})
            symbol_features.extend([
                perf.get('return', 0.0),
                perf.get('sharpe_ratio', 0.0) / 3.0  # Normalize
            ])
        
        state = np.concatenate([base_features, np.array(symbol_features)])
        return state.astype(np.float32)
    
    def calculate_reward(self, action: int, context: Dict[str, Any], result: Dict[str, Any]) -> float:
        """
        Calculate reward based on portfolio performance.
        
        Reward considers:
        - Risk-adjusted returns
        - Drawdown control
        - Diversification
        """
        allocation = self.ALLOCATION_STRATEGIES[action]
        
        # Get performance metrics
        total_pnl_pct = result.get('total_pnl_pct', 0.0)
        sharpe_ratio = result.get('sharpe_ratio', 0.0)
        drawdown_pct = result.get('drawdown_pct', 0.0)
        diversification = result.get('diversification_score', 0.5)
        
        # Base reward from Sharpe ratio
        reward = sharpe_ratio * 10
        
        # Bonus for good returns
        reward += total_pnl_pct * 5
        
        # Strong penalty for high drawdown
        if drawdown_pct > 0.15:  # 15% drawdown
            reward -= drawdown_pct * 50
        
        # Reward for appropriate allocation given market conditions
        market_regime = context.get('market', {}).get('regime', 0.5)
        volatility = context.get('market', {}).get('overall_volatility', 0.0)
        
        # High volatility should be more conservative
        if volatility > 0.05 and allocation['positions'] > 0.7:
            reward -= 10
        elif volatility < 0.02 and allocation['positions'] < 0.5:
            reward -= 5  # Low volatility, can be more aggressive
        
        # Reward for diversification
        reward += diversification * 5
        
        # Penalty for over-allocation in bad market
        if market_regime < 0.3 and allocation['positions'] > 0.7:  # Bad market, too aggressive
            reward -= 15
        
        return reward
    
    def optimize_allocation(self, portfolio_context: Dict[str, Any], 
                           market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize portfolio allocation.
        
        Returns:
            Dictionary with optimized allocation weights
        """
        context = {
            'portfolio': portfolio_context,
            'market': market_context
        }
        
        prediction = self.predict(context)
        action = prediction['action']
        allocation_strategy = self.ALLOCATION_STRATEGIES[action]
        
        self.current_strategy = action
        
        # Calculate per-symbol allocation
        # For now, equal weight across symbols (can be enhanced)
        num_symbols = len(self.symbols)
        if num_symbols > 0:
            per_symbol_weight = allocation_strategy['positions'] / num_symbols
            symbol_allocations = {symbol: per_symbol_weight for symbol in self.symbols}
        else:
            symbol_allocations = {}
        
        result = {
            'allocation_strategy': action,
            'allocation_name': ['Conservative', 'Balanced', 'Aggressive', 'Very Aggressive', 'All-in'][action],
            'cash_percent': allocation_strategy['cash'],
            'positions_percent': allocation_strategy['positions'],
            'symbol_allocations': symbol_allocations,
            'confidence': prediction['confidence']
        }
        
        self.current_allocation = symbol_allocations
        return result
    
    def update_symbol_performance(self, symbol: str, metrics: Dict[str, Any]):
        """Update performance metrics for a symbol."""
        if symbol in self.symbol_performance:
            self.symbol_performance[symbol].update(metrics)
    
    def get_current_allocation(self) -> Dict[str, Any]:
        """Get current allocation."""
        return {
            'strategy': self.current_strategy,
            'allocation': self.current_allocation.copy()
        }

