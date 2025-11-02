#!/usr/bin/env python3
"""
Mean Reversion Strategy Backtesting Script

This script backtests a Simple Moving Average (SMA) mean reversion strategy
with configurable parameters for SMA period, buy threshold, and sell threshold.
"""

import os
import argparse
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from backtesting import Backtest, Strategy
from backtesting.test import SMA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger('mr_backtest')

# Force single process on Windows to avoid handle limit errors
if os.name == 'nt':  # Windows
    # Set start method to 'spawn' to avoid handle issues
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass

class SMAMeanReversionStrategy(Strategy):
    """
    Mean Reversion strategy based on SMA crossovers.
    
    Parameters:
        sma_period: Period for SMA calculation
        buy_pct: Percentage below SMA to buy
        sell_pct: Percentage above SMA to sell
    """
    # Default parameters, will be optimized
    sma_period = 14  
    buy_pct = 1.0    
    sell_pct = 1.0   
    stop_loss = 0.0   # Optional stop loss percentage (0 = disabled)
    take_profit = 0.0 # Optional take profit percentage (0 = disabled)

    def init(self):
        """Initialize indicators used in the strategy."""
        # Calculate the SMA using the Close price
        self.sma = self.I(SMA, self.data.Close, self.sma_period)
        
        # Store trade statistics
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0

    def next(self):
        """
        Main strategy logic run on each candle.
        Executes buy/sell decisions based on price relationship to SMA.
        """
        # Calculate the buying and selling thresholds
        buy_threshold = self.sma[-1] * (1 - self.buy_pct / 100)
        sell_threshold = self.sma[-1] * (1 + self.sell_pct / 100)
        
        # Get current price
        price = self.data.Close[-1]
        
        # Entry logic - if not in a position and price is below buy threshold
        if not self.position and price < buy_threshold:
            self.buy()
            self.trade_count += 1
            
            # Set stop loss and take profit if enabled
            if self.stop_loss > 0:
                sl_price = price * (1 - self.stop_loss / 100)
                self.position.sl = sl_price
                
            if self.take_profit > 0:
                tp_price = price * (1 + self.take_profit / 100)
                self.position.tp = tp_price

        # Exit logic - if in a position and price is above sell threshold
        elif self.position and price > sell_threshold:
            # Get entry price - different backtesting versions use different attribute names
            try:
                # Calculate P&L directly without accessing entry price
                pnl_pct = (price / self.position.price - 1) * 100
            except AttributeError:
                # If we can't get the entry price, just close the position
                pnl_pct = 0
            
            if pnl_pct > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
                
            self.position.close()

def load_and_prepare_data(file_path):
    """
    Load data from CSV and prepare it for backtesting.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Prepared DataFrame with OHLCV data
    """
    try:
        # Load the data
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Check if we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns.str.lower() for col in required_cols):
            logger.error(f"CSV file missing required columns. Found: {data.columns}")
            raise ValueError("CSV file must have open, high, low, close, volume columns")
        
        # Normalize column names
        data = data[[col for col in data.columns if col.lower() in required_cols]]
        data.columns = [col.capitalize() for col in data.columns]
        
        # Sort the data index in ascending order
        data = data.sort_index()
        
        logger.info(f"Loaded data from {file_path}: {len(data)} records from {data.index.min()} to {data.index.max()}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def optimize_strategy(bt, param_ranges):
    """
    Optimize strategy parameters using grid search.
    
    Args:
        bt: Backtest instance
        param_ranges: Dictionary of parameter ranges to optimize
        
    Returns:
        Optimization stats and heatmap
    """
    logger.info(f"Starting optimization with parameter ranges: {param_ranges}")
    
    # Define constraint to ensure valid parameter combinations
    constraint = lambda param: (
        param.sma_period > 0 and 
        param.buy_pct > 0 and 
        param.sell_pct > 0
    )
    
    # Run optimization
    try:
        # On Windows, force serial optimization to avoid handle limit
        if os.name == 'nt':
            logger.info("Running optimization in serial mode on Windows")
            opt_stats, heatmap = bt.optimize(
                maximize='Equity Final [$]',
                constraint=constraint,
                return_heatmap=True,
                max_tries=100,  # Limit overall optimization attempts
                random_state=42,  # For reproducibility
                method='grid',   # Explicit grid search 
                n_jobs=1,        # Force single process
                **param_ranges
            )
        else:
            # On non-Windows systems, use parallel optimization
            opt_stats, heatmap = bt.optimize(
                maximize='Equity Final [$]',
                constraint=constraint,
                return_heatmap=True,
                **param_ranges
            )
        
        logger.info(f"Optimization complete. Best parameters: SMA={opt_stats.sma_period}, "
                   f"Buy%={opt_stats.buy_pct}, Sell%={opt_stats.sell_pct}")
        return opt_stats, heatmap
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        # Try running without optimization as fallback
        logger.warning("Falling back to default parameters")
        return None, None

def plot_optimization_heatmap(heatmap, param_name='buy_pct', figsize=(12, 10)):
    """
    Plot heatmap of optimization results.
    
    Args:
        heatmap: Heatmap data from optimization
        param_name: Parameter to use for the x-axis
        figsize: Size of the figure
    """
    # Skip if heatmap is None (optimization failed)
    if heatmap is None:
        logger.warning("Skipping heatmap plotting as no optimization data is available")
        return None
        
    try:
        # Convert heatmap to DataFrame for plotting
        heatmap_df = heatmap.unstack(level=param_name).T
        
        # Create the plot
        plt.figure(figsize=figsize)
        ax = sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap='viridis', 
                     annot_kws={"size": 8}, cbar_kws={'label': 'Final Equity ($)'})
        
        # Add titles and labels
        plt.title("Parameter Optimization Heatmap", fontsize=16)
        plt.xlabel(f"Sell Percentage (%)", fontsize=12)
        plt.ylabel(f"Buy Percentage (%)", fontsize=12)
        
        # Improve readability of axis labels
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Adjust layout for better display
        plt.tight_layout()
        
        # Save the heatmap
        plt.savefig("optimization_heatmap.png")
        logger.info("Heatmap saved to optimization_heatmap.png")
        
        return ax
        
    except Exception as e:
        logger.error(f"Failed to plot heatmap: {e}")
        raise

def run_backtest(data_path, initial_cash=100000, commission=0.002, 
                optimize=True, save_results=True):
    """
    Run the full backtest process.
    
    Args:
        data_path: Path to the CSV data file
        initial_cash: Starting capital for backtest
        commission: Commission rate as a decimal
        optimize: Whether to run parameter optimization
        save_results: Whether to save results and plots
    """
    try:
        # Extract symbol name from filename
        symbol = os.path.basename(data_path).split('_')[0]
        logger.info(f"Running backtest for {symbol}")
        
        # Load and prepare data
        data = load_and_prepare_data(data_path)
        
        # Create backtest instance
        bt = Backtest(data, SMAMeanReversionStrategy, 
                      cash=initial_cash, 
                      commission=commission,
                      trade_on_close=True)
        
        # Disable optimization on Windows by default due to handle limits
        if optimize and os.name != 'nt':
            logger.info("Running optimization (non-Windows platform)")
            # Define parameter ranges for optimization
            param_ranges = {
                'sma_period': [10, 14, 20],         # Just test 3 values
                'buy_pct': [5, 15, 25],             # Just test 3 values
                'sell_pct': [5, 15, 25],            # Just test 3 values
            }
            
            # Run optimization
            opt_stats, heatmap = optimize_strategy(bt, param_ranges)
            
            # Plot and save heatmap
            if heatmap is not None:
                try:
                    plot_optimization_heatmap(heatmap)
                except Exception as e:
                    logger.error(f"Failed to plot heatmap: {e}")
            
            # Run final backtest with optimal parameters
            if opt_stats is not None:
                logger.info("Running final backtest with optimal parameters")
                results = bt.run(
                    sma_period=opt_stats.sma_period, 
                    buy_pct=opt_stats.buy_pct, 
                    sell_pct=opt_stats.sell_pct
                )
            else:
                # Run with default parameters if optimization failed
                logger.info("Running backtest with default parameters (optimization failed)")
                results = bt.run()
        elif optimize and os.name == 'nt':
            logger.warning("Optimization disabled on Windows due to handle limit issues")
            logger.info("Running parameter tests manually instead")
            
            # Test common parameter combinations manually instead of using optimization
            best_result = None
            best_equity = 0
            best_params = {}
            
            # Smaller test grid to avoid excessive runtime
            test_params = [
                {'sma_period': 10, 'buy_pct': 5, 'sell_pct': 15},
                {'sma_period': 14, 'buy_pct': 10, 'sell_pct': 20},
                {'sma_period': 20, 'buy_pct': 15, 'sell_pct': 25},
                {'sma_period': 30, 'buy_pct': 20, 'sell_pct': 30}
            ]
            
            # Run backtest with each parameter set
            for params in test_params:
                logger.info(f"Testing parameters: {params}")
                result = bt.run(**params)
                final_equity = result['Equity Final [$]']
                logger.info(f"Result: Equity=${final_equity:.2f}, Return={result['Return [%]']:.2f}%")
                
                if final_equity > best_equity:
                    best_equity = final_equity
                    best_result = result
                    best_params = params
            
            logger.info(f"Best parameters found: {best_params}")
            logger.info(f"Best result: Equity=${best_equity:.2f}, Return={best_result['Return [%]']:.2f}%")
            
            # Use the best parameters for the final result
            results = best_result
        else:
            # Run with default parameters
            logger.info("Running backtest with default parameters")
            results = bt.run()
        
        # Print and log results
        logger.info(f"Backtest Results for {symbol}:")
        logger.info(f"Return: {results['Return [%]']:.2f}%")
        logger.info(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
        logger.info(f"# Trades: {results['# Trades']}")
        
        # Create plots
        plt.figure(figsize=(15, 10))
        bt.plot(filename=f"{symbol}_backtest_results.png" if save_results else None)
        
        if save_results:
            # Save results to CSV
            results_df = pd.DataFrame(results).T
            results_df.to_csv(f"{symbol}_backtest_results.csv")
            logger.info(f"Results saved to {symbol}_backtest_results.csv")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mean Reversion Strategy Backtesting')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the CSV data file')
    
    parser.add_argument('--cash', type=float, default=100000,
                        help='Initial cash for backtest (default: 100000)')
    
    parser.add_argument('--commission', type=float, default=0.002,
                        help='Commission rate (default: 0.002)')
    
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip parameter optimization')
    
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results and plots')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    try:
        args = parse_arguments()
    except:
        # If run interactively without arguments, use defaults
        # Create parser and print help
        parser = argparse.ArgumentParser(description='Mean Reversion Strategy Backtesting')
        parser.print_help()
        
        # Create args with default values for testing
        class Args:
            data = 'WIF_4h_50000.csv'
            cash = 100000
            commission = 0.002
            no_optimize = False
            no_save = False
        args = Args()
        
    # Run the backtest
    run_backtest(
        data_path=args.data,
        initial_cash=args.cash,
        commission=args.commission,
        optimize=not args.no_optimize,
        save_results=not args.no_save
    )