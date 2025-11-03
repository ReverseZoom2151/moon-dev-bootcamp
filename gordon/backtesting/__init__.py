"""
Refactored Backtesting Module

A modular, extensible backtesting system supporting multiple frameworks
and strategies with clean separation of concerns.

Modules:
  - base: Framework-agnostic base classes
  - data: Data fetching and validation
  - analysis: Results analysis and reporting
  - runners: Framework-specific backtesting runners
  - strategies: Strategy implementations
  - config: Configuration management
  - utils: Utility functions and helpers

Examples:
  from gordon.backtesting import BacktestRunner
  runner = BacktestRunner()
  results = runner.run_all_strategies()
"""

from .base import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    Trade,
    BaseStrategy,
    DataProvider,
    BacktestRunner
)

from .data import DataFetcher, MockDataProvider
from .analysis import ResultsAnalyzer, StrategyComparator, ResultsReporter
from .config import get_strategy_config, BacktestDefaults
from .utils import IndicatorHelper, timeit_backtest

__version__ = "2.0.0"
__author__ = "ATC Bootcamp"

__all__ = [
    # Base classes
    'BacktestConfig',
    'BacktestMetrics',
    'BacktestResult',
    'Trade',
    'BaseStrategy',
    'DataProvider',
    'BacktestRunner',
    
    # Data
    'DataFetcher',
    'MockDataProvider',
    
    # Analysis
    'ResultsAnalyzer',
    'StrategyComparator',
    'ResultsReporter',
    
    # Configuration
    'get_strategy_config',
    'BacktestDefaults',
    
    # Utils
    'IndicatorHelper',
    'timeit_backtest'
]
