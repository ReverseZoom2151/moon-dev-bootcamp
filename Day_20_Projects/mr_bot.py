#!/usr/bin/env python3
"""
Mean Reversion Trading Bot for HyperLiquid Exchange.
This bot implements a mean reversion strategy for trading cryptocurrency.
"""

import os
import sys
import time
import logging

# Fix numpy compatibility issue - Must be before other imports
import numpy
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan

import schedule
import numpy as np
import requests
import eth_account

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import nice_funcs as n

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger('mr_bot')

# Trading configuration
DEFAULT_ORDER_USD_SIZE = 10
DEFAULT_LEVERAGE = 3
DEFAULT_TIMEFRAME = '4h'

# Define symbol-specific parameters
SYMBOLS = ['WIF', 'POPCAT']
SYMBOLS_DATA = {
    'WIF': {
        'sma_period': 14,
        'buy_range': (14, 15),
        'sell_range': (14, 22)
    },
    'POPCAT': {
        'sma_period': 14,
        'buy_range': (12, 13),
        'sell_range': (14, 18)
    }
}

def get_trading_key():
    """Get the trading key from environment variable or config file."""
    # Try to get from environment variable (more secure)
    key = os.environ.get('TRADING_KEY')
    
    # If not in environment, try to import from a config file
    if not key:
        try:
            # Add project root to path for imports
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            # Now import project modules
            from Day_4_Projects import dontshare as d
            key = d.key  # Assuming the variable is named 'key' in the dontshare module
        except ImportError:
            logger.error("Trading key not found! Set TRADING_KEY environment variable or create dontshareconfig.py with 'secret' variable")
            raise ValueError("Trading key not found. Please set TRADING_KEY environment variable.")
    
    return key

def fetch_candle_snapshot(symbol, interval, start_time, end_time):
    """
    Fetch candle data for a given symbol and time range.
    
    Args:
        symbol: Trading symbol
        interval: Time interval ('1m', '5m', '15m', '1h', '4h', '1d')
        start_time: Start datetime
        end_time: End datetime
        
    Returns:
        List of candle data or None if the request fails
    """
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000)
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        snapshot_data = response.json()
        
        # Handle different response structures
        if 'candles' in snapshot_data:
            return snapshot_data['candles']
        return snapshot_data
        
    except requests.RequestException as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None
    except (ValueError, KeyError) as e:
        logger.error(f"Error parsing data for {symbol}: {e}")
        return None

def calculate_sma(data, period):
    """
    Calculate Simple Moving Average for a DataFrame.
    
    Args:
        data: DataFrame with 'close' column
        period: Period for SMA calculation
        
    Returns:
        Series containing SMA values
    """
    return data['close'].rolling(window=period).mean()

def mean_reversion_strategy(symbol, data, sma_period, buy_range, sell_range):
    """
    Implement mean reversion trading strategy.
    
    Args:
        symbol: Trading symbol
        data: OHLCV DataFrame
        sma_period: Period for SMA calculation
        buy_range: Tuple of (min, max) percentages below SMA to buy
        sell_range: Tuple of (min, max) percentages above SMA to sell
        
    Returns:
        Tuple of (action, buy_threshold, sell_threshold, current_price)
    """
    try:
        # Calculate SMA
        data['SMA'] = calculate_sma(data, sma_period)
        
        # Ensure there are enough data points after calculating SMA
        if len(data) < sma_period:
            logger.warning(f"Not enough data to calculate SMA for period {sma_period}")
            return "HOLD", None, None, None

        # Get the last valid SMA value (non-NaN)
        last_valid_sma = data['SMA'].dropna().iloc[-1]

        # Calculate buying and selling thresholds
        buy_threshold = last_valid_sma * (1 - np.random.uniform(buy_range[0], buy_range[1]) / 100)
        sell_threshold = last_valid_sma * (1 + np.random.uniform(sell_range[0], sell_range[1]) / 100)

        # Get the latest closing price
        current_price = float(data['close'].iloc[-1])
        buy_threshold = float(buy_threshold)
        sell_threshold = float(sell_threshold)

        # Strategy: Buy if current price is below the buy threshold; Sell if current price is above the sell threshold
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

def execute_trade(symbol, action, current_price, buy_threshold, sell_threshold, order_usd_size, leverage, account):
    """
    Execute a trade based on the strategy signal.
    
    Args:
        symbol: Trading symbol
        action: Trading action ('BUY', 'SELL', 'HOLD')
        current_price: Current price of the symbol
        buy_threshold: Price threshold for buying
        sell_threshold: Price threshold for selling
        order_usd_size: USD size of the order
        leverage: Leverage to use
        account: Trading account object
        
    Returns:
        True if order was placed, False otherwise
    """
    try:
        if action == "BUY":
            logger.info(f"Executing BUY order for {symbol}")
            
            # Set leverage and calculate size
            lev, size = n.adjust_leverage_usd_size(symbol, order_usd_size, leverage, account)
            
            # Check if we already have a position
            positions, in_position, pos_size, pos_sym, entry_px, pnl_perc, is_long = n.get_position(symbol, account)
            
            if not in_position:
                logger.info(f"Opening new position for {symbol}")
                
                # Round prices to appropriate decimal places
                entry_price = round(float(buy_threshold), 3)
                stop_loss = round(float(buy_threshold * 0.3), 3)  # 70% stop loss
                take_profit = round(float(sell_threshold), 3)
                
                # Prepare order information
                symbol_info = {
                    "Symbol": symbol,
                    "Entry Price": entry_price,
                    "Stop Loss": stop_loss,
                    "Take Profit": take_profit
                }
                
                # Place the order
                n.open_order_deluxe(symbol_info, size, account)
                logger.info(f"Order placed for {symbol} at {entry_price}")
                return True
            else:
                logger.info(f"Already in position for {symbol}, no new order placed")
                return False
                
        elif action == "SELL":
            logger.info(f"SELL signal for {symbol} - orders should already be in place via take-profit")
            return False
            
        else:  # HOLD
            logger.info(f"No action needed for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {e}")
        return False

def run_trading_strategy(symbols=None, symbols_data=None, order_usd_size=None, leverage=None, timeframe=None):
    """
    Run the main trading strategy for all symbols.
    
    Args:
        symbols: List of symbols to trade
        symbols_data: Dictionary of symbol-specific parameters
        order_usd_size: USD size of orders
        leverage: Leverage to use
        timeframe: Timeframe for analysis
        
    Returns:
        Number of orders placed
    """
    # Use default values if not provided
    symbols = symbols or SYMBOLS
    symbols_data = symbols_data or SYMBOLS_DATA
    order_usd_size = order_usd_size or DEFAULT_ORDER_USD_SIZE
    leverage = leverage or DEFAULT_LEVERAGE
    timeframe = timeframe or DEFAULT_TIMEFRAME
    
    try:
        # Get trading key and initialize account
        key = get_trading_key()
        account = eth_account.Account.from_key(key)
        
        orders_placed = 0
        
        for symbol in symbols:
            try:
                # Get symbol-specific parameters
                if symbol not in symbols_data:
                    logger.warning(f"No configuration for {symbol}, skipping")
                    continue
                    
                sym_config = symbols_data[symbol]
                sma_period = sym_config['sma_period']
                buy_range = sym_config['buy_range']
                sell_range = sym_config['sell_range']
                
                # Fetch market data
                snapshot_data = n.get_ohlcv2(symbol, timeframe, 20)
                df = n.process_data_to_df(snapshot_data)
                
                if df.empty:
                    logger.warning(f"No data available for {symbol}, skipping")
                    continue
                    
                # Run the strategy
                action, buy_threshold, sell_threshold, current_price = mean_reversion_strategy(
                    symbol, df, sma_period, buy_range, sell_range
                )
                
                logger.info(f"{symbol} - Action: {action}, Buy: {buy_threshold}, Sell: {sell_threshold}, Current: {current_price}")
                
                # Execute the trade if needed
                if execute_trade(symbol, action, current_price, buy_threshold, sell_threshold, 
                                order_usd_size, leverage, account):
                    orders_placed += 1
                    
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                
        return orders_placed
        
    except Exception as e:
        logger.error(f"Fatal error in trading strategy: {e}")
        return 0

def main():
    """Main entry point for the trading bot."""
    logger.info("Starting MR trading bot")
    
    # Initial run
    orders = run_trading_strategy()
    logger.info(f"Initial run complete, placed {orders} orders")
    
    # Schedule regular runs
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