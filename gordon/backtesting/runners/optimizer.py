"""
Strategy Optimizer Module
=========================
Advanced optimization utilities for backtesting strategies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from itertools import product
import concurrent.futures
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    params: Dict[str, Any]
    metric: float
    stats: Dict[str, Any]
    rank: int = 0


class StrategyOptimizer:
    """
    Advanced strategy parameter optimizer.
    Supports grid search, random search, and Bayesian optimization.
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize optimizer.

        Args:
            max_workers: Maximum parallel workers for optimization
        """
        self.max_workers = max_workers
        self.results_history = []

    def grid_search(
        self,
        backtest_func: Callable,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio',
        constraint_func: Callable = None
    ) -> OptimizationResult:
        """
        Perform grid search optimization.

        Args:
            backtest_func: Function to run backtest
            param_grid: Dictionary of parameter lists
            metric: Metric to optimize
            constraint_func: Optional constraint function

        Returns:
            Best optimization result
        """
        logger.info(f"Starting grid search optimization for {metric}")

        # Generate all parameter combinations
        param_combinations = self._generate_combinations(param_grid)

        # Filter by constraints if provided
        if constraint_func:
            param_combinations = [
                p for p in param_combinations
                if constraint_func(p)
            ]

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        # Run backtests in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(backtest_func, **params): params
                for params in param_combinations
            }

            for future in concurrent.futures.as_completed(futures):
                params = futures[future]
                try:
                    stats = future.result()
                    if stats and metric in stats:
                        results.append(OptimizationResult(
                            params=params,
                            metric=stats[metric],
                            stats=stats
                        ))
                except Exception as e:
                    logger.debug(f"Failed with params {params}: {e}")

        # Sort by metric
        results.sort(key=lambda x: x.metric, reverse=True)

        # Add ranks
        for i, result in enumerate(results):
            result.rank = i + 1

        self.results_history = results

        if results:
            best = results[0]
            logger.info(f"Best {metric}: {best.metric:.4f} with params: {best.params}")
            return best
        else:
            logger.warning("No valid results found")
            return None

    def random_search(
        self,
        backtest_func: Callable,
        param_distributions: Dict[str, Any],
        n_iter: int = 50,
        metric: str = 'sharpe_ratio'
    ) -> OptimizationResult:
        """
        Perform random search optimization.

        Args:
            backtest_func: Function to run backtest
            param_distributions: Parameter distributions
            n_iter: Number of iterations
            metric: Metric to optimize

        Returns:
            Best optimization result
        """
        logger.info(f"Starting random search optimization for {metric}")

        # Generate random parameter combinations
        param_combinations = []
        for _ in range(n_iter):
            params = {}
            for key, dist in param_distributions.items():
                if isinstance(dist, list):
                    params[key] = np.random.choice(dist)
                elif isinstance(dist, tuple) and len(dist) == 2:
                    # Assume uniform distribution between min and max
                    params[key] = np.random.uniform(dist[0], dist[1])
                else:
                    params[key] = dist
            param_combinations.append(params)

        # Run backtests
        results = []
        for params in param_combinations:
            try:
                stats = backtest_func(**params)
                if stats and metric in stats:
                    results.append(OptimizationResult(
                        params=params,
                        metric=stats[metric],
                        stats=stats
                    ))
            except Exception as e:
                logger.debug(f"Failed with params {params}: {e}")

        # Sort by metric
        results.sort(key=lambda x: x.metric, reverse=True)

        self.results_history = results

        if results:
            best = results[0]
            logger.info(f"Best {metric}: {best.metric:.4f} with params: {best.params}")
            return best
        else:
            return None

    def walk_forward_optimization(
        self,
        backtest_func: Callable,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        window_size: int,
        step_size: int,
        metric: str = 'sharpe_ratio'
    ) -> List[OptimizationResult]:
        """
        Perform walk-forward optimization.

        Args:
            backtest_func: Function to run backtest
            data: Full dataset
            param_grid: Parameter grid
            window_size: Size of optimization window
            step_size: Step size for moving window
            metric: Metric to optimize

        Returns:
            List of optimization results for each period
        """
        logger.info("Starting walk-forward optimization")

        results = []
        n_windows = (len(data) - window_size) // step_size + 1

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size

            if end_idx > len(data):
                break

            # Split data
            train_data = data.iloc[start_idx:end_idx]
            test_start = end_idx
            test_end = min(test_start + step_size, len(data))
            test_data = data.iloc[test_start:test_end]

            logger.info(f"Window {i+1}/{n_windows}: Training on {len(train_data)} samples")

            # Optimize on training data
            best_result = self.grid_search(
                lambda **params: backtest_func(train_data, **params),
                param_grid,
                metric
            )

            if best_result:
                # Test on out-of-sample data
                test_stats = backtest_func(test_data, **best_result.params)
                best_result.stats['out_of_sample'] = test_stats
                results.append(best_result)

        return results

    def get_optimization_summary(self) -> pd.DataFrame:
        """
        Get summary of optimization results.

        Returns:
            DataFrame with optimization results
        """
        if not self.results_history:
            return pd.DataFrame()

        data = []
        for result in self.results_history:
            row = {'rank': result.rank, 'metric': result.metric}
            row.update(result.params)
            row.update({
                f'stat_{k}': v
                for k, v in result.stats.items()
                if k != 'trades'  # Exclude complex objects
            })
            data.append(row)

        return pd.DataFrame(data)

    def plot_optimization_surface(
        self,
        param1: str,
        param2: str,
        metric: str = 'sharpe_ratio'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for plotting optimization surface.

        Args:
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to plot

        Returns:
            Tuple of (X, Y, Z) arrays for surface plot
        """
        if not self.results_history:
            return None, None, None

        # Extract unique values
        param1_values = sorted(set(r.params.get(param1) for r in self.results_history))
        param2_values = sorted(set(r.params.get(param2) for r in self.results_history))

        # Create grid
        X, Y = np.meshgrid(param1_values, param2_values)
        Z = np.zeros_like(X, dtype=float)

        # Fill Z values
        for result in self.results_history:
            p1 = result.params.get(param1)
            p2 = result.params.get(param2)
            if p1 in param1_values and p2 in param2_values:
                i = param1_values.index(p1)
                j = param2_values.index(p2)
                Z[j, i] = result.metric

        return X, Y, Z

    def _generate_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict]:
        """Generate all parameter combinations from grid."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations