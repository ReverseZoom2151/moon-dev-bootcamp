"""
Backtest Template Utilities
============================
Day 23: Utilities for batch processing multiple CSV files and running backtests.

Features:
- Load and prepare data from CSV files
- Run backtests on multiple files
- Batch optimization across datasets
- Results aggregation and reporting
"""

import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
# Import external backtesting package, avoiding conflict with gordon.backtesting
from .backtesting_import import get_backtesting_strategy

# Import Backtest from backtest_runner which handles the import correctly
import sys
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
try:
    from backtesting import Backtest
except ImportError:
    Backtest = None
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

Strategy = get_backtesting_strategy()

if Strategy is None or Backtest is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")

logger = logging.getLogger(__name__)


class BacktestTemplate:
    """
    Template utility for batch backtesting across multiple CSV files.
    
    Provides methods to load data, run backtests, and aggregate results.
    """

    def __init__(self, data_folder: str, output_file: Optional[str] = None):
        """
        Initialize backtest template.
        
        Args:
            data_folder: Path to folder containing CSV files
            output_file: Optional path to save results
        """
        self.data_folder = Path(data_folder)
        self.output_file = output_file
        self.results = []

    def load_prepare_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load CSV data, set index, assign standard columns.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Prepared DataFrame or None if error
        """
        try:
            data = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
            
            # Standardize column names if needed
            if 'Open' not in data.columns:
                # Assuming fixed column order: Open, High, Low, Close, Volume
                if len(data.columns) >= 5:
                    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                else:
                    raise ValueError("Insufficient columns in CSV")
            
            # Basic validation
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required OHLCV columns. Found: {data.columns.tolist()}")
            
            # Handle NaNs
            if data.isnull().values.any():
                logger.warning(f"NaNs found in {os.path.basename(file_path)}, attempting forward fill.")
                data.ffill(inplace=True)
                data.bfill(inplace=True)  # Backfill any remaining NaNs at the start
            
            data.dropna(inplace=True)  # Drop any rows that still have NaNs
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading or preparing data from {file_path}: {e}")
            return None

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_class: Type[Strategy],
        cash: float = 100000,
        commission: float = 0.002,
        **strategy_params
    ) -> Optional[Any]:
        """
        Run a single backtest and return statistics.
        
        Args:
            data: Prepared DataFrame
            strategy_class: Strategy class to use
            cash: Initial cash
            commission: Commission rate
            **strategy_params: Strategy-specific parameters
            
        Returns:
            Backtest statistics or None if error
        """
        try:
            bt = Backtest(data, strategy_class, cash=cash, commission=commission)
            stats = bt.run(**strategy_params)
            return stats
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None

    def run_optimization(
        self,
        data: pd.DataFrame,
        strategy_class: Type[Strategy],
        cash: float = 100000,
        commission: float = 0.002,
        params_to_optimize: Optional[Dict] = None,
        maximize: str = 'Equity Final [$]',
        n_iter: int = 50,
        method: str = 'skopt'
    ) -> Optional[Any]:
        """
        Run optimization using skopt or grid search.
        
        Args:
            data: Prepared DataFrame
            strategy_class: Strategy class to use
            cash: Initial cash
            commission: Commission rate
            params_to_optimize: Dictionary of parameter ranges
            maximize: Metric to maximize
            n_iter: Number of iterations (for skopt)
            method: Optimization method ('skopt' or 'grid')
            
        Returns:
            Optimization results or None if error
        """
        try:
            bt = Backtest(data, strategy_class, cash=cash, commission=commission)
            
            if method == 'skopt':
                try:
                    results = bt.optimize(
                        **params_to_optimize,
                        maximize=maximize,
                        method='skopt',
                        n_iter=n_iter
                    )
                    return results
                except ImportError:
                    logger.error("scikit-optimize not found. Install with: pip install scikit-optimize")
                    return None
            else:
                # Grid search fallback
                results = bt.optimize(
                    **params_to_optimize,
                    maximize=maximize
                )
                return results
                
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return None

    def process_folder(
        self,
        strategy_class: Type[Strategy],
        cash: float = 100000,
        commission: float = 0.002,
        optimize: bool = False,
        params_to_optimize: Optional[Dict] = None,
        maximize: str = 'Equity Final [$]',
        **strategy_params
    ) -> List[Dict[str, Any]]:
        """
        Process all CSV files in the data folder.
        
        Args:
            strategy_class: Strategy class to use
            cash: Initial cash
            commission: Commission rate
            optimize: Whether to run optimization
            params_to_optimize: Parameter ranges for optimization
            maximize: Metric to maximize
            **strategy_params: Strategy-specific parameters
            
        Returns:
            List of results dictionaries
        """
        if not self.data_folder.is_dir():
            logger.error(f"Data folder not found at {self.data_folder}")
            return []
        
        data_files = [f for f in self.data_folder.iterdir() if f.suffix == '.csv']
        if not data_files:
            logger.warning(f"No CSV files found in {self.data_folder}")
            return []
        
        logger.info(f"Found {len(data_files)} CSV files in {self.data_folder}")
        
        results = []
        
        for data_file in data_files:
            logger.info(f"Processing {data_file.name}...")
            data = self.load_prepare_data(str(data_file))
            
            if data is None or data.empty:
                logger.warning(f"Skipping {data_file.name} due to data loading error")
                continue
            
            if optimize and params_to_optimize:
                optimization_results = self.run_optimization(
                    data,
                    strategy_class,
                    cash=cash,
                    commission=commission,
                    params_to_optimize=params_to_optimize,
                    maximize=maximize
                )
                
                if optimization_results:
                    results.append({
                        'file': data_file.name,
                        'type': 'optimized',
                        'results': optimization_results,
                        'stats': optimization_results
                    })
            else:
                stats = self.run_backtest(
                    data,
                    strategy_class,
                    cash=cash,
                    commission=commission,
                    **strategy_params
                )
                
                if stats:
                    results.append({
                        'file': data_file.name,
                        'type': 'default',
                        'results': stats,
                        'stats': stats
                    })
        
        self.results = results
        return results

    def save_results(self, results: Optional[List[Dict[str, Any]]] = None):
        """
        Save results to output file.
        
        Args:
            results: Optional results list (uses self.results if not provided)
        """
        if not self.output_file:
            return
        
        results = results or self.results
        
        try:
            with open(self.output_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("BACKTEST RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                for result in results:
                    f.write(f"File: {result['file']}\n")
                    f.write(f"Type: {result['type']}\n")
                    f.write(f"Results:\n{result['stats']}\n")
                    f.write("-" * 80 + "\n\n")
            
            logger.info(f"Results saved to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

