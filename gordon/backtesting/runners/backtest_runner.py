"""
Backtest Runner Module
======================
Unified runner for executing backtesting strategies across different frameworks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Backtesting frameworks - make backtrader optional
try:
    import backtrader as bt
    import backtrader.analyzers as btanalyzers
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None  # type: ignore
    btanalyzers = None  # type: ignore
    logger.warning("Backtrader not available. Some backtesting features will be disabled.")

# Import the external backtesting package, not the local module
import sys
import importlib
# Temporarily remove the parent directory from path to import external backtesting
parent_dir = str(__file__).rsplit('gordon', 1)[0] + 'gordon'
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
try:
    from backtesting import Backtest
except ImportError:
    # If external package not installed, create a dummy
    Backtest = None
# Add the parent directory back
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import strategies - make backtrader strategy optional
SmaCrossStrategy = None
if BACKTRADER_AVAILABLE:
    try:
        from ..strategies.backtrader import SmaCrossStrategy
    except ImportError:
        logger.warning("Could not import SmaCrossStrategy. Backtrader strategies will be disabled.")
        SmaCrossStrategy = None
from ..strategies.backtesting_lib import (
    StochRSIBollingerStrategy,
    EnhancedEMAStrategy,
    MultiTimeframeBreakoutStrategy,
    MeanReversionStrategy,
    LiquidationSLiqStrategy,
    LiquidationLLiqStrategy,
    LiquidationShortSLiqStrategy,
    DelayedLiquidationShortStrategy,
    KalmanBreakoutReversalStrategy
)
from ..utils.liquidation_data_prep import (
    prepare_liquidation_data_for_sliq_strategy,
    prepare_liquidation_data_for_lliq_strategy,
    prepare_liquidation_data_for_short_strategy
)
from ..utils.alpha_decay_test import run_alpha_decay_test, print_alpha_decay_report
from ..utils.data_filters import DataFilter
from ..utils.backtest_template import BacktestTemplate
from ..evolution import GPEvolutionRunner, GPConfig
from ..data.fetcher import DataFetcher


class BacktestRunner:
    """
    Unified runner for all backtesting strategies.
    Handles execution across both Backtrader and backtesting.py frameworks.
    """

    def __init__(self, initial_cash: float = 10000, commission: float = 0.001, config: Optional[Dict] = None):
        """
        Initialize backtest runner.

        Args:
            initial_cash: Starting capital
            commission: Trading commission (0.1% default)
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = {}
        self.data_fetcher = DataFetcher()
        
        # Initialize GP runner if config available
        if 'backtesting' in self.config and 'gp_evolution' in self.config['backtesting']:
            if self.config['backtesting']['gp_evolution'].get('enabled', False):
                try:
                    from gordon.backtesting.evolution import GPEvolutionRunner, GPConfig
                    gp_config = GPConfig(**self.config['backtesting']['gp_evolution'])
                    self.gp_runner = GPEvolutionRunner(gp_config)
                except Exception as e:
                    logger.warning(f"Could not initialize GP runner: {e}")
                    self.gp_runner = None
            else:
                self.gp_runner = None
        else:
            self.gp_runner = None
        
        # Initialize ML indicator manager if config available
        if 'ml' in self.config and self.config['ml'].get('indicator_evaluation', {}).get('enabled', False):
            try:
                from gordon.ml import MLIndicatorManager
                self.ml_manager = MLIndicatorManager(self.config)
            except Exception as e:
                logger.warning(f"Could not initialize ML indicator manager: {e}")
                self.ml_manager = None
        else:
            self.ml_manager = None
        
        # Initialize historical data manager if config available
        if self.config.get('data_collection', {}).get('historical_data'):
            from ..data.historical_data_manager import HistoricalDataManager
            self.historical_data_manager = HistoricalDataManager(self.config)
        else:
            self.historical_data_manager = None

    def run_sma_crossover_backtest(
        self,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1d',
        sma_period: int = 20,
        stop_loss: float = 0.02,
        trailing_stop: float = 0.01
    ) -> Dict[str, Any]:
        """
        Run Day 13 SMA Crossover backtest using Backtrader.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            sma_period: SMA period
            stop_loss: Stop loss percentage
            trailing_stop: Trailing stop percentage

        Returns:
            Dict with backtest results and statistics
        """
        if not BACKTRADER_AVAILABLE or SmaCrossStrategy is None:
            logger.error("Backtrader not available. Install with: pip install backtrader")
            return {'error': 'Backtrader not available'}

        logger.info("=== Running SMA Crossover Backtest (Day 13) ===")

        try:
            # Initialize Cerebro engine
            cerebro = bt.Cerebro()

            # Add strategy with parameters
            cerebro.addstrategy(
                SmaCrossStrategy,
                sma_period=sma_period,
                stop_loss=stop_loss,
                trailing_stop=trailing_stop
            )

            # Fetch and add data
            df = self.data_fetcher.fetch_for_backtrader(symbol, timeframe)
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)

            # Set broker parameters
            cerebro.broker.set_cash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)

            # Add sizer (95% of capital)
            cerebro.addsizer(bt.sizers.AllInSizer, percents=95)

            # Add analyzers
            cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
            cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')

            # Run backtest
            logger.info("Starting SMA Crossover backtest...")
            results = cerebro.run()

            if not results:
                raise ValueError("No results returned from backtest")

            strategy = results[0]

            # Extract statistics
            stats = {
                'initial_value': self.initial_cash,
                'final_value': cerebro.broker.getvalue(),
                'total_return': ((cerebro.broker.getvalue() / self.initial_cash - 1) * 100),
                'sharpe_ratio': strategy.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A'),
                'max_drawdown': strategy.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 'N/A'),
                'trades': strategy.analyzers.trades.get_analysis()
            }

            # Print results
            self._print_results("SMA Crossover", stats)

            self.results['sma_crossover'] = stats
            return stats

        except Exception as e:
            logger.error(f"Error in SMA Crossover backtest: {e}")
            return {}

    def run_stochrsi_bollinger_backtest(
        self,
        symbol: str = 'ETHUSDT',
        timeframe: str = '1h'
    ) -> Dict[str, Any]:
        """
        Run Day 16 StochRSI + Bollinger Bands backtest.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe

        Returns:
            Dict with backtest results
        """
        logger.info("=== Running StochRSI + Bollinger Bands Backtest (Day 16) ===")

        try:
            # Fetch data
            data_df = self.data_fetcher.fetch_for_backtesting_lib(symbol, timeframe)

            if data_df.empty:
                logger.error("Could not fetch data")
                return {}

            # Run backtest
            backtester = Backtest(
                data_df,
                StochRSIBollingerStrategy,
                cash=self.initial_cash,
                commission=self.commission
            )

            stats = backtester.run()

            # Convert stats to dict
            stats_dict = {
                'initial_value': self.initial_cash,
                'final_value': stats['Equity Final [$]'],
                'total_return': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]']
            }

            # Print results
            self._print_results("StochRSI + Bollinger", stats_dict)

            self.results['stochrsi_bollinger'] = stats_dict
            return stats_dict

        except Exception as e:
            logger.error(f"Error in StochRSI + Bollinger backtest: {e}")
            return {}

    def run_enhanced_ema_backtest(
        self,
        symbol: str = 'SOLUSDT',
        timeframe: str = '1h',
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Run Day 17 Enhanced EMA backtest with optional optimization.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            optimize: Whether to run parameter optimization

        Returns:
            Dict with backtest results
        """
        logger.info("=== Running Enhanced EMA Backtest (Day 17) ===")

        try:
            # Fetch data
            data_df = self.data_fetcher.fetch_for_backtesting_lib(symbol, timeframe, limit=3000)

            if data_df.empty:
                logger.error("Could not fetch data")
                return {}

            # Initialize backtester
            backtester = Backtest(
                data_df,
                EnhancedEMAStrategy,
                cash=self.initial_cash,
                commission=self.commission
            )

            if optimize:
                logger.info("Running parameter optimization...")
                stats = backtester.optimize(
                    ema_short=range(5, 15),
                    ema_long=range(20, 35),
                    maximize='Sharpe Ratio',
                    constraint=lambda p: p.ema_short < p.ema_long
                )
            else:
                stats = backtester.run()

            # Convert stats to dict
            stats_dict = {
                'initial_value': self.initial_cash,
                'final_value': stats['Equity Final [$]'],
                'total_return': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'] if '# Trades' in stats and stats['# Trades'] > 0 else 0
            }

            # Print results
            self._print_results("Enhanced EMA", stats_dict)

            self.results['enhanced_ema'] = stats_dict
            return stats_dict

        except Exception as e:
            logger.error(f"Error in Enhanced EMA backtest: {e}")
            return {}

    def run_multitimeframe_breakout_backtest(
        self,
        symbol: str = 'BNBUSDT',
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Run Day 18 Multi-timeframe Breakout backtest.

        Args:
            symbol: Trading pair
            optimize: Whether to run parameter optimization

        Returns:
            Dict with backtest results
        """
        logger.info("=== Running Multi-timeframe Breakout Backtest (Day 18) ===")

        try:
            # Fetch data for main timeframe (4H)
            logger.info(f"Fetching data for {symbol}...")
            df = self.data_fetcher.fetch_for_backtesting_lib(symbol, '4h', limit=1000)

            if df.empty:
                logger.error(f"Could not fetch data for {symbol}")
                return {}

            # Initialize backtester
            backtester = Backtest(
                df,
                MultiTimeframeBreakoutStrategy,
                cash=self.initial_cash,
                commission=self.commission
            )

            if optimize:
                logger.info("Running parameter optimization...")
                stats = backtester.optimize(
                    bb_period=range(15, 25),
                    bb_std=np.arange(1.5, 2.5, 0.5),
                    maximize='Sharpe Ratio'
                )
            else:
                # Run with default parameters
                stats = backtester.run()

            # Convert stats to dict
            stats_dict = {
                'initial_value': self.initial_cash,
                'final_value': stats['Equity Final [$]'],
                'total_return': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'] if '# Trades' in stats and stats['# Trades'] > 0 else 0
            }

            # Print results
            self._print_results("Multi-timeframe Breakout", stats_dict)

            self.results['multitimeframe_breakout'] = stats_dict
            return stats_dict

        except Exception as e:
            logger.error(f"Error in Multi-timeframe Breakout backtest: {e}")
            return {}

    def run_mean_reversion_backtest(
        self,
        symbol: str = 'WIFUSDT',
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Run Day 20 Mean Reversion backtest.

        Args:
            symbol: Trading pair (default WIFUSDT as per Day 20 bot)
            optimize: Whether to run parameter optimization

        Returns:
            Dict with backtest results
        """
        logger.info("=== Running Mean Reversion Backtest (Day 20) ===")

        try:
            # Fetch data (using 4h timeframe as per Day 20 bot)
            logger.info(f"Fetching data for {symbol}...")
            df = self.data_fetcher.fetch_for_backtesting_lib(symbol, '4h', limit=500)

            if df.empty:
                logger.error(f"Could not fetch data for {symbol}")
                return {}

            # Initialize backtester
            backtester = Backtest(
                df,
                MeanReversionStrategy,
                cash=self.initial_cash,
                commission=self.commission
            )

            if optimize:
                logger.info("Running parameter optimization...")

                # Define optimization parameters
                opt_params = dict(
                    sma_period=[10, 14, 20, 30],
                    buy_pct=[5, 10, 15, 20],
                    sell_pct=[10, 15, 20, 25],
                    stop_loss=[0, 5, 10],  # 0 means disabled
                    take_profit=[0, 10, 20]  # 0 means disabled
                )

                # Run optimization
                best_sharpe = -np.inf
                best_params = {}
                best_stats = None

                for params in self._generate_param_combinations(opt_params):
                    try:
                        # Skip invalid combinations
                        if params['buy_pct'] >= params['sell_pct']:
                            continue

                        temp_stats = backtester.run(**params)

                        if temp_stats['Sharpe Ratio'] > best_sharpe:
                            best_sharpe = temp_stats['Sharpe Ratio']
                            best_params = params
                            best_stats = temp_stats

                    except Exception as e:
                        logger.debug(f"Skipped params {params}: {e}")

                if best_stats:
                    stats = best_stats
                    logger.info(f"Best parameters found: {best_params}")
                else:
                    logger.warning("No valid parameters found, using defaults")
                    stats = backtester.run()
            else:
                # Run with default parameters
                stats = backtester.run()

            # Convert stats to dict
            stats_dict = {
                'initial_value': self.initial_cash,
                'final_value': stats['Equity Final [$]'],
                'total_return': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'] if '# Trades' in stats and stats['# Trades'] > 0 else 0
            }

            # Add optimization parameters if optimized
            if optimize and 'best_params' in locals() and best_params:
                stats_dict['optimal_params'] = best_params

            # Print results
            self._print_results("Mean Reversion", stats_dict)

            self.results['mean_reversion'] = stats_dict
            return stats_dict

        except Exception as e:
            logger.error(f"Error in Mean Reversion backtest: {e}")
            return {}

    def run_liquidation_sliq_backtest(
        self,
        data_path: str,
        symbol: Optional[str] = None,
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Run Day 21 Liquidation S LIQ backtest.
        
        Args:
            data_path: Path to liquidation CSV file
            symbol: Optional symbol filter
            optimize: Whether to run parameter optimization
            
        Returns:
            Dict with backtest results
        """
        logger.info("=== Running Liquidation S LIQ Backtest (Day 21) ===")
        
        try:
            # Prepare liquidation data
            data_df = prepare_liquidation_data_for_sliq_strategy(data_path, symbol)
            
            if data_df.empty:
                logger.error("Could not prepare liquidation data")
                return {}
            
            # Initialize backtester
            backtester = Backtest(
                data_df,
                LiquidationSLiqStrategy,
                cash=self.initial_cash,
                commission=self.commission
            )
            
            if optimize:
                logger.info("Running parameter optimization...")
                stats = backtester.optimize(
                    s_liq_entry_thresh=range(10000, 500000, 10000),
                    entry_time_window_mins=range(5, 60, 5),
                    take_profit=[i / 1000 for i in range(5, 31, 5)],
                    stop_loss=[i / 1000 for i in range(5, 31, 5)],
                    maximize='Equity Final [$]'
                )
            else:
                stats = backtester.run()
            
            # Convert stats to dict
            stats_dict = {
                'initial_value': self.initial_cash,
                'final_value': stats['Equity Final [$]'],
                'total_return': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'] if '# Trades' in stats and stats['# Trades'] > 0 else 0
            }
            
            # Print results
            self._print_results("Liquidation S LIQ", stats_dict)
            
            self.results['liquidation_sliq'] = stats_dict
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error in Liquidation S LIQ backtest: {e}")
            return {}

    def run_liquidation_lliq_backtest(
        self,
        data_path: str,
        symbol: Optional[str] = None,
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Run Day 21 Liquidation L LIQ backtest.
        
        Args:
            data_path: Path to liquidation CSV file
            symbol: Optional symbol filter
            optimize: Whether to run parameter optimization
            
        Returns:
            Dict with backtest results
        """
        logger.info("=== Running Liquidation L LIQ Backtest (Day 21) ===")
        
        try:
            # Prepare liquidation data
            data_df = prepare_liquidation_data_for_lliq_strategy(data_path, symbol)
            
            if data_df.empty:
                logger.error("Could not prepare liquidation data")
                return {}
            
            # Initialize backtester
            backtester = Backtest(
                data_df,
                LiquidationLLiqStrategy,
                cash=self.initial_cash,
                commission=self.commission
            )
            
            if optimize:
                logger.info("Running parameter optimization...")
                stats = backtester.optimize(
                    l_liq_entry_thresh=range(10000, 500000, 10000),
                    entry_time_window_mins=range(1, 11, 1),
                    s_liq_closure_thresh=range(10000, 500000, 10000),
                    exit_time_window_mins=range(1, 11, 1),
                    take_profit=[i / 100 for i in range(1, 5, 1)],
                    stop_loss=[i / 100 for i in range(1, 5, 1)],
                    maximize='Equity Final [$]'
                )
            else:
                stats = backtester.run()
            
            # Convert stats to dict
            stats_dict = {
                'initial_value': self.initial_cash,
                'final_value': stats['Equity Final [$]'],
                'total_return': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'] if '# Trades' in stats and stats['# Trades'] > 0 else 0
            }
            
            # Print results
            self._print_results("Liquidation L LIQ", stats_dict)
            
            self.results['liquidation_lliq'] = stats_dict
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error in Liquidation L LIQ backtest: {e}")
            return {}

    def run_liquidation_short_sliq_backtest(
        self,
        data_path: str,
        symbol: Optional[str] = None,
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Run Day 22 Liquidation Short S LIQ backtest.
        
        Args:
            data_path: Path to liquidation CSV file
            symbol: Optional symbol filter
            optimize: Whether to run parameter optimization
            
        Returns:
            Dict with backtest results
        """
        logger.info("=== Running Liquidation Short S LIQ Backtest (Day 22) ===")
        
        try:
            # Prepare liquidation data
            data_df = prepare_liquidation_data_for_short_strategy(data_path, symbol)
            
            if data_df.empty:
                logger.error("Could not prepare liquidation data")
                return {}
            
            # Initialize backtester
            backtester = Backtest(
                data_df,
                LiquidationShortSLiqStrategy,
                cash=self.initial_cash,
                commission=self.commission
            )
            
            if optimize:
                logger.info("Running parameter optimization...")
                stats = backtester.optimize(
                    short_liquidation_thresh=range(10000, 500000, 10000),
                    entry_time_window_mins=range(1, 11, 1),
                    long_liquidation_closure_thresh=range(10000, 500000, 10000),
                    exit_time_window_mins=range(1, 11, 1),
                    take_profit_pct=[i / 100 for i in range(1, 5, 1)],
                    stop_loss_pct=[i / 100 for i in range(1, 5, 1)],
                    maximize='Equity Final [$]'
                )
            else:
                stats = backtester.run()
            
            # Convert stats to dict
            stats_dict = {
                'initial_value': self.initial_cash,
                'final_value': stats['Equity Final [$]'],
                'total_return': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'] if '# Trades' in stats and stats['# Trades'] > 0 else 0
            }
            
            # Print results
            self._print_results("Liquidation Short S LIQ", stats_dict)
            
            self.results['liquidation_short_sliq'] = stats_dict
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error in Liquidation Short S LIQ backtest: {e}")
            return {}

    def run_alpha_decay_test(
        self,
        data_path: str,
        symbol: Optional[str] = None,
        delays: Optional[List[int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run alpha decay test for short liquidation strategy.
        
        Tests how strategy performance degrades with entry delays.
        
        Args:
            data_path: Path to liquidation CSV file
            symbol: Optional symbol filter
            delays: List of delay values in minutes (default: [0, 1, 2, 5, 10, 15, 30, 60])
            
        Returns:
            Dictionary mapping delay to backtest results
        """
        logger.info("=== Running Alpha Decay Test (Day 22) ===")
        
        if delays is None:
            delays = [0, 1, 2, 5, 10, 15, 30, 60]
        
        try:
            # Prepare liquidation data
            data_df = prepare_liquidation_data_for_short_strategy(data_path, symbol)
            
            if data_df.empty:
                logger.error("Could not prepare liquidation data")
                return {}
            
            # Run alpha decay test
            results = run_alpha_decay_test(
                data_df,
                DelayedLiquidationShortStrategy,
                delays,
                initial_cash=self.initial_cash,
                commission=self.commission
            )
            
            # Print analysis report
            print_alpha_decay_report(results, metric='return_pct')
            
            self.results['alpha_decay'] = results
            return results
            
        except Exception as e:
            logger.error(f"Error in alpha decay test: {e}")
            return {}

    def run_kalman_breakout_backtest(
        self,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1h',
        optimize: bool = False,
        filter_exclude_summer: bool = False,
        filter_market_hours: Optional[str] = None  # 'market', 'non_market', or None
    ) -> Dict[str, Any]:
        """
        Run Day 23 Kalman Filter Breakout/Reversal backtest.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            optimize: Whether to run parameter optimization
            filter_exclude_summer: Whether to exclude summer months (May-September)
            filter_market_hours: Filter by market hours ('market', 'non_market', or None)
            
        Returns:
            Dict with backtest results
        """
        logger.info("=== Running Kalman Filter Breakout/Reversal Backtest (Day 23) ===")
        
        try:
            # Fetch data
            df = self.data_fetcher.fetch_for_backtesting_lib(symbol, timeframe, limit=2000)
            
            if df.empty:
                logger.error("Could not fetch data")
                return {}
            
            # Apply data filters if requested
            if filter_exclude_summer:
                logger.info("Filtering out summer months (May-September)")
                df = DataFilter.filter_exclude_months(df, start_month=5, end_month=9)
            
            if filter_market_hours:
                logger.info(f"Filtering for {filter_market_hours} hours")
                market_df, non_market_df = DataFilter.filter_market_hours(df)
                if filter_market_hours == 'market':
                    df = market_df
                elif filter_market_hours == 'non_market':
                    df = non_market_df
            
            if df.empty:
                logger.error("No data remaining after filtering")
                return {}
            
            # Initialize backtester
            backtester = Backtest(
                df,
                KalmanBreakoutReversalStrategy,
                cash=self.initial_cash,
                commission=self.commission
            )
            
            if optimize:
                logger.info("Running parameter optimization...")
                stats = backtester.optimize(
                    window=range(20, 100, 10),
                    take_profit=[i / 100 for i in range(1, 11, 1)],
                    stop_loss=[i / 100 for i in range(1, 11, 1)],
                    maximize='Equity Final [$]',
                    method='skopt',
                    n_iter=50
                )
            else:
                stats = backtester.run()
            
            # Convert stats to dict
            stats_dict = {
                'initial_value': self.initial_cash,
                'final_value': stats['Equity Final [$]'],
                'total_return': stats['Return [%]'],
                'sharpe_ratio': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'num_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'] if '# Trades' in stats and stats['# Trades'] > 0 else 0
            }
            
            # Print results
            self._print_results("Kalman Breakout/Reversal", stats_dict)
            
            self.results['kalman_breakout'] = stats_dict
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error in Kalman Breakout backtest: {e}")
            return {}

    def run_ml_indicator_evaluation(
        self,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1h',
        pandas_ta_indicators: List[Dict] = None,
        talib_indicators: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run ML indicator evaluation for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            pandas_ta_indicators: List of pandas_ta indicator configs
            talib_indicators: List of talib indicator names
            
        Returns:
            Evaluation results
        """
        if not self.ml_manager:
            logger.error("ML indicator manager not initialized")
            return {'error': 'ML manager not available'}
        
        try:
            # Fetch data
            data_df = self.data_fetcher.fetch_for_backtesting_lib(symbol, timeframe, limit=2000)
            
            if data_df.empty:
                logger.error(f"Could not fetch data for {symbol}")
                return {'error': 'Data fetch failed'}
            
            # Run evaluation
            result = self.ml_manager.evaluate_indicator_set(
                data_df,
                pandas_ta_indicators=pandas_ta_indicators,
                talib_indicators=talib_indicators
            )
            
            logger.info("ML indicator evaluation completed")
            return result
            
        except Exception as e:
            logger.error(f"Error in ML indicator evaluation: {e}")
            return {'error': str(e)}
    
    def run_ml_indicator_looping(
        self,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1h',
        generations: int = 10
    ) -> Dict[str, Any]:
        """
        Run ML indicator looping for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            generations: Number of generations to run
            
        Returns:
            Looping results
        """
        if not self.ml_manager:
            logger.error("ML indicator manager not initialized")
            return {'error': 'ML manager not available'}
        
        try:
            # Update generations in config
            if 'ml' not in self.config:
                self.config['ml'] = {}
            if 'indicator_looping' not in self.config['ml']:
                self.config['ml']['indicator_looping'] = {}
            self.config['ml']['indicator_looping']['generations'] = generations
            
            # Reinitialize manager with updated config
            from gordon.ml import MLIndicatorManager
            ml_manager = MLIndicatorManager(self.config)
            
            # Fetch data
            data_df = self.data_fetcher.fetch_for_backtesting_lib(symbol, timeframe, limit=2000)
            
            if data_df.empty:
                logger.error(f"Could not fetch data for {symbol}")
                return {'error': 'Data fetch failed'}
            
            # Run looping
            results = ml_manager.run_indicator_looping(data_df)
            
            logger.info(f"ML indicator looping completed: {len(results)} generations")
            return {
                'generations': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in ML indicator looping: {e}")
            return {'error': str(e)}
    
    def run_ml_top_indicators(
        self,
        top_n: int = 50
    ) -> Dict[str, Any]:
        """
        Get top indicators from ML evaluation.
        
        Args:
            top_n: Number of top indicators to return
            
        Returns:
            Dictionary with rankings
        """
        if not self.ml_manager:
            logger.error("ML indicator manager not initialized")
            return {'error': 'ML manager not available'}
        
        try:
            rankings = self.ml_manager.get_top_indicators(top_n=top_n)
            return rankings
        except Exception as e:
            logger.error(f"Error getting top indicators: {e}")
            return {'error': str(e)}
    
    def evolve_strategy_gp(
        self,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1h',
        data_path: Optional[str] = None,
        generations: int = 30,
        population_size: int = 200,
        use_multiprocessing: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evolve trading strategy using Genetic Programming (Day 29).
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            data_path: Optional path to CSV file
            generations: Number of generations to evolve
            population_size: Size of population
            use_multiprocessing: Whether to use multiprocessing
            save_results: Whether to save results
            
        Returns:
            Dictionary with evolution results
        """
        logger.info("=== Starting GP Strategy Evolution (Day 29) ===")
        
        try:
            # Create GP configuration
            gp_config = GPConfig(
                initial_cash=self.initial_cash,
                commission_pct=self.commission,
                population_size=population_size,
                generations=generations
            )
            
            # Run evolution
            results = self.gp_runner.evolve_strategy(
                symbol=symbol,
                timeframe=timeframe,
                data_path=data_path,
                gp_config=gp_config,
                use_multiprocessing=use_multiprocessing,
                save_results=save_results
            )
            
            if 'error' in results:
                logger.error(f"GP evolution failed: {results['error']}")
                return results
            
            # Log results
            logger.info(f"Evolution completed for {symbol}")
            logger.info(f"Found {results.get('hall_of_fame_size', 0)} best strategies")
            
            if results.get('best_strategies'):
                logger.info("Top strategies:")
                for strategy in results['best_strategies'][:3]:
                    logger.info(
                        f"  Rank {strategy['rank']}: "
                        f"Fitness={strategy['fitness']:.2f}, "
                        f"Size={strategy['size']}"
                    )
            
            self.results['gp_evolution'] = results
            return results
            
        except Exception as e:
            logger.error(f"Error in GP evolution: {e}", exc_info=True)
            return {'error': str(e)}

    def _generate_param_combinations(self, opt_params):
        """Generate parameter combinations for optimization."""
        import itertools
        keys, values = zip(*opt_params.items())
        for v in itertools.product(*values):
            yield dict(zip(keys, v))

    def run_all_strategies(
        self,
        optimize_ema: bool = False,
        optimize_mtf: bool = False,
        optimize_mr: bool = False
    ) -> Dict[str, Any]:
        """
        Run all backtesting strategies and compare results.

        Args:
            optimize_ema: Whether to optimize Enhanced EMA strategy
            optimize_mtf: Whether to optimize Multi-timeframe strategy
            optimize_mr: Whether to optimize Mean Reversion strategy

        Returns:
            Dict with all results and comparison
        """
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING ALL BACKTESTING STRATEGIES")
        logger.info("=" * 60 + "\n")

        # Run all strategies
        self.run_sma_crossover_backtest()
        self.run_stochrsi_bollinger_backtest()
        self.run_enhanced_ema_backtest(optimize=optimize_ema)
        self.run_multitimeframe_breakout_backtest(optimize=optimize_mtf)
        self.run_mean_reversion_backtest(optimize=optimize_mr)

        # Compare results
        self._compare_strategies()

        return self.results

    def _print_results(self, strategy_name: str, stats: Dict):
        """Print formatted backtest results."""
        print(f"\n--- {strategy_name} Results ---")
        print(f"Initial Capital: ${stats['initial_value']:,.2f}")
        print(f"Final Value: ${stats['final_value']:,.2f}")
        print(f"Total Return: {stats['total_return']:.2f}%")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']}")
        print(f"Max Drawdown: {stats['max_drawdown']}%")

        if 'num_trades' in stats:
            print(f"Number of Trades: {stats['num_trades']}")
        if 'win_rate' in stats:
            print(f"Win Rate: {stats['win_rate']:.2f}%")

    def _compare_strategies(self):
        """Compare and rank all strategies."""
        if not self.results:
            return

        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON")
        print("=" * 60)

        # Create comparison DataFrame
        comparison_data = []
        for name, stats in self.results.items():
            comparison_data.append({
                'Strategy': name.replace('_', ' ').title(),
                'Return %': stats.get('total_return', 0),
                'Sharpe': stats.get('sharpe_ratio', 0),
                'Max DD %': stats.get('max_drawdown', 0),
                'Final Value': stats.get('final_value', 0)
            })

        df = pd.DataFrame(comparison_data)

        # Sort by return
        df = df.sort_values('Return %', ascending=False)

        print("\n" + df.to_string(index=False))

        # Identify best strategy
        best_strategy = df.iloc[0]['Strategy']
        best_return = df.iloc[0]['Return %']

        print(f"\n🏆 Best Strategy: {best_strategy} with {best_return:.2f}% return")