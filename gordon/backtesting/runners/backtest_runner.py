"""
Backtest Runner Module
======================
Unified runner for executing backtesting strategies across different frameworks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime

# Backtesting frameworks
import backtrader as bt
import backtrader.analyzers as btanalyzers
from backtesting import Backtest

# Import strategies
from ..strategies.backtrader import SmaCrossStrategy
from ..strategies.backtesting_lib import (
    StochRSIBollingerStrategy,
    EnhancedEMAStrategy,
    MultiTimeframeBreakoutStrategy,
    MeanReversionStrategy
)
from ..data.fetcher import DataFetcher

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    Unified runner for all backtesting strategies.
    Handles execution across both Backtrader and backtesting.py frameworks.
    """

    def __init__(self, initial_cash: float = 10000, commission: float = 0.001):
        """
        Initialize backtest runner.

        Args:
            initial_cash: Starting capital
            commission: Trading commission (0.1% default)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = {}
        self.data_fetcher = DataFetcher()

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