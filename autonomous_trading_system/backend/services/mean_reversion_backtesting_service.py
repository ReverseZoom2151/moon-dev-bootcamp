#!/usr/bin/env python3
"""
Mean Reversion Strategy Backtesting Service

This service implements comprehensive backtesting for the Mean Reversion strategy
using the backtesting.py library, similar to mr_bt.py but integrated with the ATS framework.
"""

import os
import asyncio
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from typing import Dict, List, Optional, Tuple, Any
from backtesting import Backtest, Strategy
from backtesting.test import SMA
from core.config import get_settings, get_trading_config
from services.trading_utils import get_ohlcv2, process_data_to_df

# Configure logging
logger = logging.getLogger('mean_reversion_backtest')

# Force single process on Windows to avoid handle limit errors
if os.name == 'nt':  # Windows
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Method already set

class MeanReversionBacktestStrategy(Strategy):
    """
    Mean Reversion strategy for backtesting using backtesting.py library.
    
    This mirrors the logic from mr_bt.py but integrates with ATS configuration.
    """
    # Default parameters - will be overridden by config
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
            # Calculate P&L
            try:
                pnl_pct = (price / self.position.price - 1) * 100
            except AttributeError:
                pnl_pct = 0
            
            if pnl_pct > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
                
            self.position.close()

class MeanReversionBacktestingService:
    """
    Service for running Mean Reversion strategy backtests using backtesting.py library.
    Integrates with the ATS configuration system and provides comprehensive analysis.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.trading_config = get_trading_config()
        self.mr_config = self.trading_config.get('mean_reversion', {})
        logger.info("🔬 Mean Reversion Backtesting Service initialized")

    async def load_and_prepare_data(self, symbol: str, timeframe: str = "4h", days_back: int = 100) -> pd.DataFrame:
        """
        Load market data for backtesting.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            days_back: Number of days of historical data
            
        Returns:
            Prepared DataFrame with OHLCV data
        """
        try:
            # Use the existing trading utils to fetch data
            snapshot_data = get_ohlcv2(symbol, timeframe, days_back)
            if not snapshot_data:
                raise ValueError(f"No data available for {symbol}")
            
            # Convert to DataFrame with proper format for backtesting.py
            df = process_data_to_df(snapshot_data)
            
            if df.empty:
                raise ValueError(f"Empty dataset for {symbol}")
            
            # Ensure we have the required columns with proper capitalization
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df_cols_lower = [col.lower() for col in df.columns]
            
            if not all(col in df_cols_lower for col in required_cols):
                raise ValueError(f"Missing required columns. Found: {df.columns}")
            
            # Rename columns to match backtesting.py expectations
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in required_cols:
                    column_mapping[col] = col_lower.capitalize()
            
            df = df.rename(columns=column_mapping)
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Sort by date
            df = df.sort_index()
            
            logger.info(f"📊 Loaded {len(df)} records for {symbol} from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data for {symbol}: {e}")
            raise

    async def run_single_backtest(
        self, 
        symbol: str, 
        data: Optional[pd.DataFrame] = None,
        parameters: Optional[Dict[str, Any]] = None,
        initial_cash: float = 100000,
        commission: float = 0.002
    ) -> Dict[str, Any]:
        """
        Run a single backtest for the Mean Reversion strategy.
        
        Args:
            symbol: Trading symbol
            data: Market data (if None, will be fetched)
            parameters: Strategy parameters (if None, uses config defaults)
            initial_cash: Starting capital
            commission: Commission rate
            
        Returns:
            Backtest results dictionary
        """
        try:
            # Load data if not provided
            if data is None:
                timeframe = self.mr_config.get('timeframe', '4h')
                data = await self.load_and_prepare_data(symbol, timeframe)
            
            # Use config parameters if not specified
            if parameters is None:
                parameters = {
                    'sma_period': self.mr_config.get('sma_period', 14),
                    'buy_pct': self.mr_config.get('buy_range', [12, 15])[0],  # Use lower bound
                    'sell_pct': self.mr_config.get('sell_range', [14, 22])[0],  # Use lower bound
                }
            
            logger.info(f"🚀 Running backtest for {symbol} with parameters: {parameters}")
            
            # Create backtest instance
            bt = Backtest(
                data, 
                MeanReversionBacktestStrategy,
                cash=initial_cash,
                commission=commission,
                trade_on_close=True
            )
            
            # Run backtest
            results = await asyncio.to_thread(bt.run, **parameters)
            
            # Convert results to dictionary format
            result_dict = {
                'symbol': symbol,
                'parameters': parameters,
                'initial_cash': initial_cash,
                'commission': commission,
                'final_equity': float(results['Equity Final [$]']),
                'total_return_pct': float(results['Return [%]']),
                'sharpe_ratio': float(results['Sharpe Ratio']) if pd.notna(results['Sharpe Ratio']) else 0.0,
                'max_drawdown_pct': float(results['Max. Drawdown [%]']),
                'total_trades': int(results['# Trades']),
                'win_rate_pct': float(results['Win Rate [%]']) if '# Trades' in results and results['# Trades'] > 0 else 0.0,
                'profit_factor': float(results.get('Profit Factor', 0.0)) if pd.notna(results.get('Profit Factor', 0.0)) else 0.0,
                'start_date': data.index[0].isoformat(),
                'end_date': data.index[-1].isoformat(),
                'data_points': len(data)
            }
            
            logger.info(f"✅ Backtest completed for {symbol}")
            logger.info(f"   Return: {result_dict['total_return_pct']:.2f}%")
            logger.info(f"   Sharpe: {result_dict['sharpe_ratio']:.2f}")
            logger.info(f"   Max DD: {result_dict['max_drawdown_pct']:.2f}%")
            logger.info(f"   Trades: {result_dict['total_trades']}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"❌ Backtest failed for {symbol}: {e}")
            raise

    async def optimize_parameters(
        self,
        symbol: str,
        data: Optional[pd.DataFrame] = None,
        parameter_ranges: Optional[Dict[str, List]] = None,
        initial_cash: float = 100000,
        commission: float = 0.002,
        maximize: str = 'Equity Final [$]'
    ) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            symbol: Trading symbol
            data: Market data
            parameter_ranges: Parameter ranges for optimization
            initial_cash: Starting capital
            commission: Commission rate
            maximize: Metric to maximize
            
        Returns:
            Tuple of (best_result_dict, heatmap_dataframe)
        """
        try:
            # Load data if not provided
            if data is None:
                timeframe = self.mr_config.get('timeframe', '4h')
                data = await self.load_and_prepare_data(symbol, timeframe)
            
            # Default parameter ranges
            if parameter_ranges is None:
                buy_range = self.mr_config.get('buy_range', [12, 15])
                sell_range = self.mr_config.get('sell_range', [14, 22])
                parameter_ranges = {
                    'sma_period': [10, 14, 20],
                    'buy_pct': list(range(buy_range[0], buy_range[1] + 1, 2)),
                    'sell_pct': list(range(sell_range[0], sell_range[1] + 1, 3)),
                }
            
            logger.info(f"🔧 Starting optimization for {symbol} with ranges: {parameter_ranges}")
            
            # Create backtest instance
            bt = Backtest(
                data,
                MeanReversionBacktestStrategy,
                cash=initial_cash,
                commission=commission,
                trade_on_close=True
            )
            
            # Define constraint
            constraint = lambda param: (
                param.sma_period > 0 and 
                param.buy_pct > 0 and 
                param.sell_pct > 0
            )
            
            # Run optimization
            if os.name == 'nt':  # Windows - use serial mode
                logger.info("Running optimization in serial mode (Windows)")
                opt_result, heatmap = await asyncio.to_thread(
                    bt.optimize,
                    maximize=maximize,
                    constraint=constraint,
                    return_heatmap=True,
                    max_tries=100,
                    random_state=42,
                    method='grid',
                    n_jobs=1,
                    **parameter_ranges
                )
            else:
                # Non-Windows - can use parallel
                opt_result, heatmap = await asyncio.to_thread(
                    bt.optimize,
                    maximize=maximize,
                    constraint=constraint,
                    return_heatmap=True,
                    **parameter_ranges
                )
            
            # Convert result to dictionary
            best_params = {
                'sma_period': opt_result.sma_period,
                'buy_pct': opt_result.buy_pct,
                'sell_pct': opt_result.sell_pct
            }
            
            result_dict = {
                'symbol': symbol,
                'best_parameters': best_params,
                'optimization_metric': maximize,
                'final_equity': float(opt_result['Equity Final [$]']),
                'total_return_pct': float(opt_result['Return [%]']),
                'sharpe_ratio': float(opt_result['Sharpe Ratio']) if pd.notna(opt_result['Sharpe Ratio']) else 0.0,
                'max_drawdown_pct': float(opt_result['Max. Drawdown [%]']),
                'total_trades': int(opt_result['# Trades']),
                'win_rate_pct': float(opt_result['Win Rate [%]']) if opt_result['# Trades'] > 0 else 0.0,
                'data_points': len(data)
            }
            
            logger.info(f"✅ Optimization completed for {symbol}")
            logger.info(f"   Best params: {best_params}")
            logger.info(f"   Best return: {result_dict['total_return_pct']:.2f}%")
            
            return result_dict, heatmap
            
        except Exception as e:
            logger.error(f"❌ Optimization failed for {symbol}: {e}")
            return {}, None

    async def plot_optimization_heatmap(
        self, 
        heatmap: pd.DataFrame, 
        symbol: str,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Plot and save optimization heatmap.
        
        Args:
            heatmap: Heatmap data from optimization
            symbol: Trading symbol
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot or None if failed
        """
        if heatmap is None or heatmap.empty:
            logger.warning("No heatmap data available for plotting")
            return None
            
        try:
            # Prepare heatmap for plotting
            if 'buy_pct' in heatmap.index.names:
                heatmap_df = heatmap.unstack(level='buy_pct').T
            else:
                heatmap_df = heatmap
            
            # Create the plot
            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(
                heatmap_df, 
                annot=True, 
                fmt=".0f", 
                cmap='viridis',
                annot_kws={"size": 8}, 
                cbar_kws={'label': 'Final Equity ($)'}
            )
            
            # Add titles and labels
            plt.title(f"Mean Reversion Optimization Heatmap - {symbol}", fontsize=16)
            plt.xlabel("Sell Percentage (%)", fontsize=12)
            plt.ylabel("Buy Percentage (%)", fontsize=12)
            
            # Improve readability
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the plot
            if save_path is None:
                save_path = f"mr_optimization_heatmap_{symbol}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory
            
            logger.info(f"📈 Heatmap saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Failed to plot heatmap: {e}")
            return None

    async def run_comprehensive_backtest(
        self,
        symbol: str,
        optimize: bool = True,
        save_results: bool = True,
        initial_cash: float = 100000,
        commission: float = 0.002
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest including optimization and visualization.
        
        Args:
            symbol: Trading symbol
            optimize: Whether to run parameter optimization
            save_results: Whether to save plots and results
            initial_cash: Starting capital
            commission: Commission rate
            
        Returns:
            Comprehensive results dictionary
        """
        try:
            logger.info(f"🎯 Starting comprehensive backtest for {symbol}")
            
            # Load data
            timeframe = self.mr_config.get('timeframe', '4h')
            data = await self.load_and_prepare_data(symbol, timeframe)
            
            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data_period': f"{data.index[0].date()} to {data.index[-1].date()}",
                'data_points': len(data)
            }
            
            if optimize:
                # Run optimization
                opt_result, heatmap = await self.optimize_parameters(
                    symbol, data, initial_cash=initial_cash, commission=commission
                )
                results['optimization'] = opt_result
                
                # Plot heatmap if requested
                if save_results and heatmap is not None:
                    heatmap_path = await self.plot_optimization_heatmap(heatmap, symbol)
                    results['heatmap_path'] = heatmap_path
                
                # Run final backtest with optimal parameters
                if opt_result.get('best_parameters'):
                    final_result = await self.run_single_backtest(
                        symbol, data, opt_result['best_parameters'], 
                        initial_cash, commission
                    )
                    results['final_backtest'] = final_result
            else:
                # Run single backtest with default parameters
                default_result = await self.run_single_backtest(
                    symbol, data, initial_cash=initial_cash, commission=commission
                )
                results['backtest'] = default_result
            
            logger.info(f"✅ Comprehensive backtest completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive backtest failed for {symbol}: {e}")
            raise

    async def batch_backtest(
        self,
        symbols: Optional[List[str]] = None,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Run backtests for multiple symbols.
        
        Args:
            symbols: List of symbols (if None, uses config symbols)
            optimize: Whether to run optimization for each symbol
            
        Returns:
            Dictionary of results by symbol
        """
        if symbols is None:
            symbols = self.mr_config.get('symbols', ['WIF', 'POPCAT', 'BTC', 'ETH'])
        
        logger.info(f"🚀 Starting batch backtest for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                symbol_result = await self.run_comprehensive_backtest(
                    symbol, optimize=optimize, save_results=False
                )
                results[symbol] = symbol_result
                logger.info(f"✅ Completed backtest for {symbol}")
            except Exception as e:
                logger.error(f"❌ Failed backtest for {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        # Summary statistics
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        if successful_results:
            summary = self._calculate_batch_summary(successful_results)
            results['_summary'] = summary
        
        logger.info(f"🎯 Batch backtest completed: {len(successful_results)}/{len(symbols)} successful")
        return results

    def _calculate_batch_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across multiple backtests."""
        try:
            returns = []
            sharpes = []
            max_dds = []
            
            for symbol, data in results.items():
                backtest_data = data.get('final_backtest') or data.get('backtest') or data.get('optimization', {})
                if isinstance(backtest_data, dict):
                    if 'total_return_pct' in backtest_data:
                        returns.append(backtest_data['total_return_pct'])
                    if 'sharpe_ratio' in backtest_data:
                        sharpes.append(backtest_data['sharpe_ratio'])
                    if 'max_drawdown_pct' in backtest_data:
                        max_dds.append(backtest_data['max_drawdown_pct'])
            
            if returns:
                return {
                    'avg_return_pct': sum(returns) / len(returns),
                    'best_return_pct': max(returns),
                    'worst_return_pct': min(returns),
                    'avg_sharpe': sum(sharpes) / len(sharpes) if sharpes else 0,
                    'avg_max_drawdown_pct': sum(max_dds) / len(max_dds) if max_dds else 0,
                    'symbols_tested': len(returns),
                    'positive_returns': len([r for r in returns if r > 0])
                }
            else:
                return {'error': 'No valid backtest data found'}
                
        except Exception as e:
            logger.error(f"Error calculating batch summary: {e}")
            return {'error': str(e)} 