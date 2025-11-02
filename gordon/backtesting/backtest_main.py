"""
Comprehensive Backtesting Orchestrator
========================================
Central orchestrator for all backtesting strategies and frameworks.

This module provides a unified interface for backtesting various strategies
with different frameworks and optimization capabilities.

Strategies Included:
- Day 13: SMA Crossover with Stop Loss and Trailing Stop (backtrader)
- Day 16: StochRSI + Bollinger Bands Strategy (backtesting.py)
- Day 17: Enhanced EMA Strategy with Optimization (backtesting.py)
- Day 18: Multi-timeframe Breakout Strategy with Bollinger Bands (backtesting.py)
- Day 20: Mean Reversion Strategy with SMA (backtesting.py)
"""

import sys
import os
import json
import logging
import warnings
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import runners and utilities
from .runners import BacktestRunner, StrategyOptimizer
from .data.fetcher import DataFetcher

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveBacktester:
    """
    Master orchestrator for all backtesting operations.
    Provides a clean interface for running and comparing strategies.
    """

    def __init__(self, initial_cash: float = 10000, commission: float = 0.001):
        """
        Initialize the comprehensive backtester.

        Args:
            initial_cash: Starting capital for backtests
            commission: Trading commission rate
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.runner = BacktestRunner(initial_cash, commission)
        self.optimizer = StrategyOptimizer()
        self.data_fetcher = DataFetcher()

    def run_strategy(self, strategy_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run a specific backtesting strategy.

        Args:
            strategy_name: Name of the strategy to run
            **kwargs: Strategy-specific parameters

        Returns:
            Dict with backtest results

        Available strategies:
            - 'sma_crossover': Day 13 SMA strategy
            - 'stochrsi_bollinger': Day 16 StochRSI + BB
            - 'enhanced_ema': Day 17 Enhanced EMA
            - 'multitimeframe_breakout': Day 18 MTF Breakout
            - 'mean_reversion': Day 20 Mean Reversion
        """
        strategy_methods = {
            'sma_crossover': self.runner.run_sma_crossover_backtest,
            'stochrsi_bollinger': self.runner.run_stochrsi_bollinger_backtest,
            'enhanced_ema': self.runner.run_enhanced_ema_backtest,
            'multitimeframe_breakout': self.runner.run_multitimeframe_breakout_backtest,
            'mean_reversion': self.runner.run_mean_reversion_backtest
        }

        if strategy_name not in strategy_methods:
            logger.error(f"Unknown strategy: {strategy_name}")
            logger.info(f"Available strategies: {list(strategy_methods.keys())}")
            return {}

        return strategy_methods[strategy_name](**kwargs)

    def run_all_strategies(
        self,
        optimize_ema: bool = False,
        optimize_mtf: bool = False,
        optimize_mr: bool = False
    ) -> Dict[str, Any]:
        """
        Run all backtesting strategies with optional optimization.

        Args:
            optimize_ema: Whether to optimize Enhanced EMA strategy
            optimize_mtf: Whether to optimize Multi-timeframe strategy
            optimize_mr: Whether to optimize Mean Reversion strategy

        Returns:
            Dict with all results and comparison
        """
        return self.runner.run_all_strategies(optimize_ema, optimize_mtf, optimize_mr)

    def optimize_strategy(
        self,
        strategy_name: str,
        param_grid: Dict,
        metric: str = 'sharpe_ratio',
        method: str = 'grid'
    ) -> Any:
        """
        Optimize a strategy's parameters.

        Args:
            strategy_name: Strategy to optimize
            param_grid: Parameter search space
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
            method: Optimization method ('grid' or 'random')

        Returns:
            Best parameters and results
        """
        logger.info(f"Optimizing {strategy_name} strategy using {method} search")

        # Create backtest function for the strategy
        def backtest_func(**params):
            return self.run_strategy(strategy_name, **params)

        if method == 'grid':
            return self.optimizer.grid_search(backtest_func, param_grid, metric)
        elif method == 'random':
            return self.optimizer.random_search(backtest_func, param_grid, n_iter=50, metric=metric)
        else:
            logger.error(f"Unknown optimization method: {method}")
            return None

    def compare_frameworks(self) -> Dict:
        """
        Compare Backtrader vs backtesting.py frameworks.

        Returns:
            Comparison of framework performance and features
        """
        logger.info("Comparing backtesting frameworks...")

        results = {
            'backtrader': {
                'strategies': ['sma_crossover'],
                'features': [
                    'Complex strategy development',
                    'Built-in indicators',
                    'Advanced order types',
                    'Multiple data feeds',
                    'Live trading support'
                ]
            },
            'backtesting.py': {
                'strategies': [
                    'stochrsi_bollinger',
                    'enhanced_ema',
                    'multitimeframe_breakout',
                    'mean_reversion'
                ],
                'features': [
                    'Simple and intuitive API',
                    'Fast execution',
                    'Built-in optimization',
                    'Vectorized operations',
                    'Lightweight'
                ]
            }
        }

        # Run sample backtests for comparison
        bt_result = self.run_strategy('sma_crossover')
        btpy_result = self.run_strategy('enhanced_ema')

        results['performance_comparison'] = {
            'backtrader_return': bt_result.get('total_return', 0),
            'backtesting_py_return': btpy_result.get('total_return', 0)
        }

        return results

    def save_results(self, filename: str = 'backtest_results.json'):
        """
        Save backtest results to file.

        Args:
            filename: Output filename
        """
        try:
            import numpy as np

            # Convert numpy types for JSON serialization
            json_results = {}
            for key, value in self.runner.results.items():
                json_results[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in value.items()
                    if k != 'trades'  # Exclude complex objects
                }

            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)

            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Could not save results: {e}")

    def load_results(self, filename: str = 'backtest_results.json') -> Dict:
        """
        Load backtest results from file.

        Args:
            filename: Input filename

        Returns:
            Loaded results
        """
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
            logger.info(f"Results loaded from {filename}")
            return results
        except Exception as e:
            logger.error(f"Could not load results: {e}")
            return {}


def main():
    """
    Main execution function demonstrating the orchestrator.
    """
    # Initialize orchestrator
    orchestrator = ComprehensiveBacktester(initial_cash=10000, commission=0.001)

    # Example 1: Run a single strategy
    logger.info("\n=== Running Single Strategy ===")
    sma_result = orchestrator.run_strategy('sma_crossover', sma_period=20)

    # Example 2: Run all strategies with optimization
    logger.info("\n=== Running All Strategies ===")
    all_results = orchestrator.run_all_strategies(optimize_ema=True)

    # Example 3: Optimize a specific strategy
    logger.info("\n=== Optimizing Mean Reversion Strategy ===")
    optimization_result = orchestrator.optimize_strategy(
        'mean_reversion',
        param_grid={
            'sma_period': [10, 14, 20, 30],
            'buy_pct': [5, 10, 15],
            'sell_pct': [15, 20, 25]
        },
        metric='sharpe_ratio'
    )

    # Example 4: Compare frameworks
    logger.info("\n=== Comparing Frameworks ===")
    framework_comparison = orchestrator.compare_frameworks()

    # Save results
    orchestrator.save_results()

    return orchestrator


if __name__ == "__main__":
    orchestrator = main()