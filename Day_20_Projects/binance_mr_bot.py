#!/usr/bin/env python3
"""
Mean Reversion Trading Bot for Binance (USDT Futures).
"""

import os
import sys
import time
import logging
import schedule
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Day_12_Projects.binance_nice_funcs import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('trading_bot.log')])
logger = logging.getLogger('mr_bot')

DEFAULT_ORDER_USD_SIZE = 10
DEFAULT_LEVERAGE = 3
DEFAULT_TIMEFRAME = '4h'

SYMBOLS = ['WIFUSDT', 'POPCATUSDT']
SYMBOLS_DATA = {
    'WIFUSDT': {'sma_period': 14, 'buy_range': (14, 15), 'sell_range': (14, 22)},
    'POPCATUSDT': {'sma_period': 14, 'buy_range': (12, 13), 'sell_range': (14, 18)}
}

def calculate_sma(data, period):
    return data['close'].rolling(window=period).mean()

def mean_reversion_strategy(symbol, data, sma_period, buy_range, sell_range):
    try:
        data['SMA'] = calculate_sma(data, sma_period)
        if len(data) < sma_period:
            logger.warning(f"Not enough data to calculate SMA for period {sma_period}")
            return "HOLD", None, None, None
        last_valid_sma = data['SMA'].dropna().iloc[-1]
        buy_threshold = last_valid_sma * (1 - np.random.uniform(buy_range[0], buy_range[1]) / 100)
        sell_threshold = last_valid_sma * (1 + np.random.uniform(sell_range[0], sell_range[1]) / 100)
        current_price = float(data['close'].iloc[-1])
        buy_threshold = float(buy_threshold)
        sell_threshold = float(sell_threshold)
        if current_price < buy_threshold:
            action = "BUY"
        elif current_price > sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"
        return action, buy_threshold, sell_threshold, current_price
    except Exception as e:
        logger.error(f"Error in mean reversion strategy for {symbol}: {e}")
        return "HOLD", None, None, None

def execute_trade(symbol, action, current_price, buy_threshold, sell_threshold, order_usd_size, leverage):
    try:
        acct_value = acct_bal()
        if action == "BUY":
            logger.info(f"Executing BUY order for {symbol}")
            lev, size = adjust_leverage_size_signal(symbol, leverage, acct_value)
            positions, in_position, _, _, _, _, _ = get_position(symbol)
            if not in_position:
                logger.info(f"Opening new position for {symbol}")
                entry_price = round(float(buy_threshold), 3)
                stop_loss = round(float(buy_threshold * 0.3), 3)
                take_profit = round(float(sell_threshold), 3)
                symbol_info = {"Symbol": symbol, "Entry Price": entry_price, "Stop Loss": stop_loss, "Take Profit": take_profit}
                open_order_deluxe(symbol_info, size)
                logger.info(f"Order placed for {symbol} at {entry_price}")
                return True
            else:
                logger.info(f"Already in position for {symbol}, no new order placed")
                return False
        elif action == "SELL":
            logger.info(f"SELL signal for {symbol} - orders should already be in place via take-profit")
            return False
        else:
            logger.info(f"No action needed for {symbol}")
            return False
    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {e}")
        return False

def run_trading_strategy(symbols=None, symbols_data=None, order_usd_size=None, leverage=None, timeframe=None):
    symbols = symbols or SYMBOLS
    symbols_data = symbols_data or SYMBOLS_DATA
    order_usd_size = order_usd_size or DEFAULT_ORDER_USD_SIZE
    leverage = leverage or DEFAULT_LEVERAGE
    timeframe = timeframe or DEFAULT_TIMEFRAME
    orders_placed = 0
    for symbol in symbols:
        try:
            if symbol not in symbols_data:
                logger.warning(f"No configuration for {symbol}, skipping")
                continue
            sym_config = symbols_data[symbol]
            sma_period = sym_config['sma_period']
            buy_range = sym_config['buy_range']
            sell_range = sym_config['sell_range']
            snapshot_data = get_ohlcv2(symbol, timeframe, 20)
            df = process_data_to_df(snapshot_data)
            if df.empty:
                logger.warning(f"No data available for {symbol}, skipping")
                continue
            action, buy_threshold, sell_threshold, current_price = mean_reversion_strategy(symbol, df, sma_period, buy_range, sell_range)
            logger.info(f"{symbol} - Action: {action}, Buy: {buy_threshold}, Sell: {sell_threshold}, Current: {current_price}")
            if execute_trade(symbol, action, current_price, buy_threshold, sell_threshold, order_usd_size, leverage):
                orders_placed += 1
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
    return orders_placed

def main():
    logger.info("Starting MR trading bot")
    orders = run_trading_strategy()
    logger.info(f"Initial run complete, placed {orders} orders")
    schedule.every(1).minutes.do(run_trading_strategy)
    logger.info("Bot running on schedule (every 1 minute)")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main() 