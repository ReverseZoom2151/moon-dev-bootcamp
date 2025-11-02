############# Coding trading bot #3 - engulfing candle 2024

'''
ENGULFING candle strategy:
An engulfing pattern occurs when the current candle completely engulfs the previous candle.
- Bullish engulfing: Current candle's body completely engulfs previous bearish candle's body
- Bearish engulfing: Current candle's body completely engulfs previous bullish candle's body

Trading logic:
1. Check for engulfing patterns in the most recent candles
2. Take a position in the direction of the engulfing pattern if confirmed by SMA
3. Manage risk with target profit and stop loss
'''

# Fix numpy.NaN import issue in pandas_ta
import numpy
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan

# Standard library imports
import os, sys, time, logging
from datetime import datetime

# Third-party imports
import ccxt, schedule
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Custom imports
from Day_4_Projects.key_file import key as xP_hmv_KEY, secret as xP_hmv_SECRET

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("engulfing_bot.log")
    ]
)
logger = logging.getLogger("engulfing_bot")

# Set up exchange connection
phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': xP_hmv_KEY,
    'secret': xP_hmv_SECRET
})

# Configuration parameters
symbol = 'BTC/USD:BTC'
pos_size = 1
target = 9  # % gain target
max_loss = -8  # % max loss
timeframe = '15m'
limit = 100  # Need enough candles for analysis
sma_period = 20
retry_delay = 30  # seconds to wait after errors

# Order parameters
params = {'timeInForce': 'PostOnly'}

# Global flag to track if order was recently placed
recent_order = False
recent_order_time = None

# Utility functions to replace nice_funcs
def df_sma(symbol, timeframe, limit, sma_period):
    """
    Fetch OHLCV data and calculate SMA.
    """
    logger.info(f"Fetching {limit} {timeframe} candles for {symbol}")
    try:
        # Fetch OHLCV data
        ohlcv = phemex.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Calculate SMA
        sma_col = f'sma{sma_period}_{timeframe}'
        df[sma_col] = df['close'].rolling(window=sma_period).mean()
        
        logger.info(f"Successfully calculated SMA{sma_period}")
        return df
    except Exception as e:
        logger.error(f"Error in df_sma: {e}")
        raise

def ask_bid(symbol):
    """
    Get current ask/bid prices from order book.
    """
    logger.info(f"Fetching order book for {symbol}")
    try:
        orderbook = phemex.fetch_order_book(symbol)
        bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
        ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None
        
        # Create a simplified L2 data structure
        l2_data = [
            [{'px': price, 'qty': size} for price, size in orderbook['bids'][:20]],  # Top 20 bids
            [{'px': price, 'qty': size} for price, size in orderbook['asks'][:20]]   # Top 20 asks
        ]
        
        logger.info(f"Current bid: {bid}, ask: {ask}")
        return ask, bid, l2_data
    except Exception as e:
        logger.error(f"Error in ask_bid: {e}")
        return None, None, [[], []]

def open_positions(symbol):
    """
    Get information about current open positions.
    """
    logger.info(f"Checking open positions for {symbol}")
    try:
        positions = phemex.fetch_positions([symbol])
        
        if positions and len(positions) > 0:
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['contracts']) > 0:
                    # Found an open position
                    size = float(pos['contracts'])
                    is_long = pos['side'] == 'long'
                    entry_price = float(pos['entryPrice'])
                    pnl = float(pos['unrealizedPnl'])
                    
                    logger.info(f"Open position: {size} contracts, {'long' if is_long else 'short'}, entry: {entry_price}")
                    return {
                        'symbol': symbol,
                        'size': size,
                        'entry_price': entry_price,
                        'pnl': pnl
                    }, True, size, is_long, entry_price, pnl, 0
                    
        # No open positions found
        logger.info(f"No open positions for {symbol}")
        return {}, False, 0, False, 0, 0, 0
    except Exception as e:
        logger.error(f"Error in open_positions: {e}")
        return {}, False, 0, False, 0, 0, 0

def custom_pnl_close(symbol, target_percent, max_loss_percent):
    """
    Custom implementation of pnl_close that works with phemex exchange.
    Closes position when profit target or max loss is reached.
    """
    logger.info("Checking PnL for position closing conditions")
    try:
        # Get position information
        positions_info = open_positions(symbol)
        in_position = positions_info[1]
        
        if not in_position:
            logger.info("No open positions to check")
            return
            
        position_size = float(positions_info[2]) if positions_info[2] else 0
        is_long = positions_info[3]
        entry_price = float(positions_info[4]) if positions_info[4] else 0
        
        # Get current market price
        ticker = phemex.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Calculate PnL percentage
        if is_long:
            pnl_percent = ((current_price / entry_price) - 1) * 100
        else:
            pnl_percent = ((entry_price / current_price) - 1) * 100
            
        logger.info(f"Current PnL: {pnl_percent:.2f}% (Target: {target_percent}%, Stop: {max_loss_percent}%)")
        
        # Check if we should close the position
        if pnl_percent >= target_percent or pnl_percent <= max_loss_percent:
            close_type = "profit target" if pnl_percent >= target_percent else "stop loss"
            logger.info(f"Closing position at {pnl_percent:.2f}% {close_type}")
            
            # Close the position with a market order
            if is_long:
                phemex.create_market_sell_order(symbol, position_size)
            else:
                phemex.create_market_buy_order(symbol, position_size)
                
            logger.info(f"Position closed at {current_price}")
    except Exception as e:
        logger.error(f"Error in custom_pnl_close: {e}")

def detect_engulfing_pattern(df):
    """
    Detect engulfing candle patterns in the dataframe.
    
    Returns:
        str: 'bullish', 'bearish', or None
    """
    if len(df) < 3:
        logger.warning("Not enough candles to detect engulfing pattern")
        return None
    
    # Get the last two completed candles (not the current forming one)
    prev_candle = df.iloc[-3]
    curr_candle = df.iloc[-2]
    
    # Calculate candle bodies
    prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
    curr_body_size = abs(curr_candle['close'] - curr_candle['open'])
    
    # Determine if candles are bullish or bearish
    prev_bullish = prev_candle['close'] > prev_candle['open']
    curr_bullish = curr_candle['close'] > curr_candle['open']
    
    # Check for bullish engulfing pattern
    if (not prev_bullish and curr_bullish and
        curr_body_size > prev_body_size and
        curr_candle['open'] <= prev_candle['close'] and
        curr_candle['close'] >= prev_candle['open']):
        return "bullish"
    
    # Check for bearish engulfing pattern
    elif (prev_bullish and not curr_bullish and
          curr_body_size > prev_body_size and
          curr_candle['open'] >= prev_candle['close'] and
          curr_candle['close'] <= prev_candle['open']):
        return "bearish"
    
    return None

def get_market_data():
    """Get market data and indicators."""
    try:
        # Get OHLCV data with SMA
        df = df_sma(symbol, timeframe, limit, sma_period)
        
        # Fix column name to match what's expected
        sma_col = f'sma{sma_period}_{timeframe}'
        
        # Get current bid/ask prices
        askbid = ask_bid(symbol)
        if askbid[0] is None:
            logger.error("Failed to get bid/ask prices")
            return None
        
        ask, bid = askbid[0], askbid[1]
        
        # Get current position information
        allposinfo = open_positions(symbol)
        in_pos = allposinfo[1]
        curr_size = int(allposinfo[2]) if allposinfo[2] else 0
        pos_direction = "long" if allposinfo[3] else "short" if curr_size > 0 else None
        
        # Get the latest SMA and price values
        last_close = df.iloc[-1]['close']
        sma_value = df.iloc[-1][sma_col]
        
        # Detect engulfing pattern
        pattern = detect_engulfing_pattern(df)
        
        logger.info(f"Position: {in_pos}, Size: {curr_size}, Direction: {pos_direction}")
        logger.info(f"Bid: {bid}, Ask: {ask}, SMA: {sma_value}, Last Close: {last_close}")
        logger.info(f"Engulfing Pattern: {pattern}")
        
        return {
            'df': df,
            'ask': ask,
            'bid': bid,
            'in_position': in_pos,
            'position_size': curr_size,
            'position_direction': pos_direction,
            'sma': sma_value,
            'last_close': last_close,
            'pattern': pattern
        }
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return None

def create_order(order_type, order_price):
    """Create an order with proper error handling."""
    global recent_order, recent_order_time
    
    try:
        phemex.cancel_all_orders(symbol)
        
        if order_type == "buy":
            phemex.create_limit_buy_order(symbol, pos_size, order_price, params)
            logger.info(f"Created BUY order at price {order_price}")
        else:
            phemex.create_limit_sell_order(symbol, pos_size, order_price, params)
            logger.info(f"Created SELL order at price {order_price}")
        
        # Set recent order flag and timestamp
        recent_order = True
        recent_order_time = datetime.now()
        
        return True
    except Exception as e:
        logger.error(f"Error creating {order_type} order: {e}")
        return False

def bot():
    """Main trading bot function."""
    global recent_order, recent_order_time
    
    # Check if we need to wait after recent order
    if recent_order and recent_order_time:
        elapsed = (datetime.now() - recent_order_time).total_seconds()
        if elapsed < 120:  # 2 minutes
            logger.info(f"Recent order placed {elapsed:.0f} seconds ago, waiting...")
            return
        else:
            recent_order = False
    
    try:
        # Check if we need to close positions based on profit/loss targets
        custom_pnl_close(symbol, target, max_loss)
        
        # Get market data
        data = get_market_data()
        if not data:
            logger.error("Failed to get market data, skipping cycle")
            return
        
        # If already in position, don't open a new one
        if data['in_position'] and data['position_size'] > 0:
            logger.info(f"Already in position ({data['position_direction']}), not entering new trade")
            return
        
        # Make trading decisions based on engulfing pattern and SMA confirmation
        pattern = data['pattern']
        bid = data['bid']
        ask = data['ask']
        sma = data['sma']
        
        if pattern == "bullish" and bid > sma:
            logger.info("Bullish engulfing pattern with SMA confirmation")
            order_price = bid * 0.999  # Slightly below current bid for higher fill probability
            create_order("buy", order_price)
            
        elif pattern == "bearish" and ask < sma:
            logger.info("Bearish engulfing pattern with SMA confirmation")
            order_price = ask * 1.001  # Slightly above current ask for higher fill probability
            create_order("sell", order_price)
            
        else:
            if pattern:
                logger.info(f"{pattern.capitalize()} pattern detected but no SMA confirmation")
            else:
                logger.info("No engulfing pattern detected")
                
    except Exception as e:
        logger.error(f"Error in bot main function: {e}")

def run_bot():
    """Wrapper to safely run the bot."""
    try:
        bot()
    except Exception as e:
        logger.error(f"Unexpected error in bot: {e}")
        time.sleep(retry_delay)

def shutdown_handler():
    """Handle graceful shutdown."""
    logger.info("Shutting down bot...")
    try:
        # Cancel all open orders
        phemex.cancel_all_orders(symbol)
        logger.info("Cancelled all orders")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Schedule the bot to run every 30 seconds
schedule.every(30).seconds.do(run_bot)

# Main loop
if __name__ == "__main__":
    logger.info(f"=== Starting Engulfing Candle Trading Bot ===")
    logger.info(f"Trading {symbol} with {pos_size} position size")
    logger.info(f"Target profit: {target}%, Max loss: {max_loss}%")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)  # Small sleep to prevent CPU hogging
    except KeyboardInterrupt:
        shutdown_handler()
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        shutdown_handler()











