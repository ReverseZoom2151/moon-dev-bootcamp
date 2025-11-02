"""
Trading Strategies Module
==========================
Contains all trading strategy implementations organized by day/type.
Each strategy is self-contained in its own file.

Extracted from strategy_manager.py for clean separation and modularity.
"""

from .base import BaseStrategy

# Day 6-12 Strategy Imports
from .day6_sma_strategy import SMAStrategy
from .day7_rsi_strategy import RSIStrategy
from .day8_vwap_strategy import VWAPStrategy
from .day9_vwma_strategy import VWMAStrategy
from .day10_bollinger_strategy import BollingerStrategy
from .day10_volume_strategy import VolumeStrategy
from .day11_breakout_strategy import BreakoutStrategy
from .day11_supply_demand_strategy import SupplyDemandStrategy
from .day12_engulfing_strategy import EngulfingStrategy
from .day12_vwap_probability_strategy import VWAPProbabilityStrategy

# Import all strategy classes for easy access
__all__ = [
    'BaseStrategy',
    # Day 6-12 Strategies
    'SMAStrategy',
    'RSIStrategy',
    'VWAPStrategy',
    'VWMAStrategy',
    'BollingerStrategy',
    'VolumeStrategy',
    'BreakoutStrategy',
    'SupplyDemandStrategy',
    'EngulfingStrategy',
    'VWAPProbabilityStrategy',
]