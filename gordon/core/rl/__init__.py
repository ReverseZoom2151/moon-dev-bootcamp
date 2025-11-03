"""
RL Module
=========
Reinforcement Learning components for Gordon.
"""

from .base import BaseRLComponent, DQNAgent
from .meta_strategy_selector import MetaStrategySelector
from .signal_aggregator import SignalAggregator
from .risk_optimizer import RiskOptimizer
from .position_sizer import PositionSizeOptimizer
from .regime_detector import MarketRegimeDetector
from .portfolio_allocator import PortfolioAllocator
from .manager import RLManager

__all__ = [
    'BaseRLComponent',
    'DQNAgent',
    'MetaStrategySelector',
    'SignalAggregator',
    'RiskOptimizer',
    'PositionSizeOptimizer',
    'MarketRegimeDetector',
    'PortfolioAllocator',
    'RLManager',
]

