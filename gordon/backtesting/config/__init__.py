"""Configuration modules for backtesting"""

from .strategy_configs import STRATEGY_CONFIGS, get_strategy_config
from .backtest_config import BacktestDefaults

__all__ = ['STRATEGY_CONFIGS', 'get_strategy_config', 'BacktestDefaults']
