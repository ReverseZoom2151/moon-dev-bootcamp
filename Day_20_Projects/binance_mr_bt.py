#!/usr/bin/env python3
"""
Mean Reversion Strategy Backtesting Script (Binance Version)
"""

import os
import argparse
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
import sys
from backtesting import Backtest, Strategy
from backtesting.test import SMA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.binance_nice_funcs import create_exchange, get_ohlcv2, process_data_to_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('backtest.log')])
logger = logging.getLogger('mr_backtest')

if os.name == 'nt':
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

class SMAMeanReversionStrategy(Strategy):
    sma_period = 14  
    buy_pct = 1.0    
    sell_pct = 1.0   
    stop_loss = 0.0   
    take_profit = 0.0 
    def init(self):
        self.sma = self.I(SMA, self.data.Close, self.sma_period)
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
    def next(self):
        buy_threshold = self.sma[-1] * (1 - self.buy_pct / 100)
        sell_threshold = self.sma[-1] * (1 + self.sell_pct / 100)
        price = self.data.Close[-1]
        if not self.position and price < buy_threshold:
            self.buy()
            self.trade_count += 1
            if self.stop_loss > 0:
                sl_price = price * (1 - self.stop_loss / 100)
                self.position.sl = sl_price
            if self.take_profit > 0:
                tp_price = price * (1 + self.take_profit / 100)
                self.position.tp = tp_price
        elif self.position and price > sell_threshold:
            try:
                pnl_pct = (price / self.position.price - 1) * 100
            except AttributeError:
                pnl_pct = 0
            if pnl_pct > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
            self.position.close()

def fetch_and_prepare_data(symbol='WIFUSDT', timeframe='1h', lookback_days=365):
    exchange = create_exchange()
    data = get_ohlcv2(symbol, timeframe, lookback_days)
    df = process_data_to_df(data)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df.sort_index()
    logger.info(f"Loaded data: {len(df)} records")
    return df

def optimize_strategy(bt, param_ranges):
    logger.info(f"Starting optimization with parameter ranges: {param_ranges}")
    constraint = lambda param: (param.sma_period > 0 and param.buy_pct > 0 and param.sell_pct > 0)
    try:
        if os.name == 'nt':
            logger.info("Running optimization in serial mode on Windows")
            opt_stats, heatmap = bt.optimize(maximize='Equity Final [$]', constraint=constraint, return_heatmap=True, max_tries=100, random_state=42, method='grid', n_jobs=1, **param_ranges)
        else:
            opt_stats, heatmap = bt.optimize(maximize='Equity Final [$]', constraint=constraint, return_heatmap=True, **param_ranges)
        logger.info(f"Optimization complete. Best parameters: SMA={opt_stats.sma_period}, Buy%={opt_stats.buy_pct}, Sell%={opt_stats.sell_pct}")
        return opt_stats, heatmap
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return None, None

def plot_optimization_heatmap(heatmap, param_name='buy_pct', figsize=(12, 10)):
    if heatmap is None:
        logger.warning("Skipping heatmap plotting as no optimization data is available")
        return None
    try:
        heatmap_df = heatmap.unstack(level=param_name).T
        plt.figure(figsize=figsize)
        ax = sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap='viridis', annot_kws={"size": 8}, cbar_kws={'label': 'Final Equity ($)'})
        plt.title("Parameter Optimization Heatmap", fontsize=16)
        plt.xlabel(f"Sell Percentage (%)", fontsize=12)
        plt.ylabel(f"Buy Percentage (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("optimization_heatmap.png")
        logger.info("Heatmap saved to optimization_heatmap.png")
        return ax
    except Exception as e:
        logger.error(f"Failed to plot heatmap: {e}")
        raise

def run_backtest(initial_cash=100000, commission=0.002, optimize=True, save_results=True):
    try:
        data = fetch_and_prepare_data()
        bt = Backtest(data, SMAMeanReversionStrategy, cash=initial_cash, commission=commission, trade_on_close=True)
        if optimize and os.name != 'nt':
            param_ranges = {'sma_period': [10, 14, 20], 'buy_pct': [5, 15, 25], 'sell_pct': [5, 15, 25]}
            opt_stats, heatmap = optimize_strategy(bt, param_ranges)
            plot_optimization_heatmap(heatmap)
            if opt_stats is not None:
                results = bt.run(sma_period=opt_stats.sma_period, buy_pct=opt_stats.buy_pct, sell_pct=opt_stats.sell_pct)
            else:
                results = bt.run()
        elif optimize and os.name == 'nt':
            logger.warning("Optimization disabled on Windows due to handle limit issues")
            logger.info("Running parameter tests manually instead")
            best_result = None
            best_equity = 0
            best_params = {}
            test_params = [{'sma_period': 10, 'buy_pct': 5, 'sell_pct': 15}, {'sma_period': 14, 'buy_pct': 10, 'sell_pct': 20}, {'sma_period': 20, 'buy_pct': 15, 'sell_pct': 25}, {'sma_period': 30, 'buy_pct': 20, 'sell_pct': 30}]
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
            results = best_result
        else:
            results = bt.run()
        logger.info(f"Backtest Results:")
        logger.info(f"Return: {results['Return [%]']:.2f}%")
        logger.info(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
        logger.info(f"# Trades: {results['# Trades']}")
        plt.figure(figsize=(15, 10))
        bt.plot(filename="backtest_results.png" if save_results else None)
        if save_results:
            results_df = pd.DataFrame(results).T
            results_df.to_csv("backtest_results.csv")
            logger.info("Results saved to backtest_results.csv")
        return results
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(description='Mean Reversion Strategy Backtesting')
    parser.add_argument('--cash', type=float, default=100000, help='Initial cash for backtest (default: 100000)')
    parser.add_argument('--commission', type=float, default=0.002, help='Commission rate (default: 0.002)')
    parser.add_argument('--no-optimize', action='store_true', help='Skip parameter optimization')
    parser.add_argument('--no-save', action='store_true', help='Do not save results and plots')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
    except:
        class Args:
            cash = 100000
            commission = 0.002
            no_optimize = False
            no_save = False
        args = Args()
    run_backtest(initial_cash=args.cash, commission=args.commission, optimize=not args.no_optimize, save_results=not args.no_save) 