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

# Day 26 Strategy Imports
from .day26_eth_sma_strategy import EthSMAStrategy
from .day26_market_maker_strategy import MarketMakerStrategy

# Day 36 Social Signal Trading
from .social_signal_trader import SocialSignalTrader, SocialSignalDetector, SocialSignalExecutor

# Day 37 RRS Strategy
from .rrs_strategy import RRSStrategy

# Day 45 Enhanced Strategies
from .liquidation_hunter_strategy import LiquidationHunterStrategy

# Day 48 Easy Entry Strategy
from .easy_entry_strategy import EasyEntryStrategy

# Day 51 Quick Buy/Sell Strategy
from .quick_buysell_strategy import QuickBuySellStrategy

# Day 49 MA Reversal Strategy
from .ma_reversal_strategy import MAReversalStrategy

# Day 50 Enhanced Supply/Demand Strategy
from .enhanced_sd_strategy import EnhancedSupplyDemandStrategy

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
    # Day 20 Strategies
    'MeanReversionStrategy',
    # Day 26 Strategies
    'EthSMAStrategy',
    'MarketMakerStrategy',
    # Day 36 Social Signal Trading
    'SocialSignalTrader',
    'SocialSignalDetector',
    'SocialSignalExecutor',
    # Day 37 RRS Strategy
    'RRSStrategy',
    # Day 45 Enhanced Strategies
    'LiquidationHunterStrategy',
    # Day 48 Easy Entry Strategy
    'EasyEntryStrategy',
    # Day 51 Quick Buy/Sell Strategy
    'QuickBuySellStrategy',
    # Day 49 MA Reversal Strategy
    'MAReversalStrategy',
    # Day 50 Enhanced Supply/Demand Strategy
    'EnhancedSupplyDemandStrategy',
]