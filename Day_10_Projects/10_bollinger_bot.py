'''
Bollinger Band Trading Bot

This bot trades based on Bollinger Band compression (tight bands), 
entering both long and short positions when bands are tight.
'''
# Fix numpy NaN import issue in pandas_ta
import numpy
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan

import sys, os, nice_funcs as n, eth_account, time, schedule, pandas as pd, datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects import dontshare as d 
from datetime import datetime

# Configuration Constants
SYMBOL = 'WIF'
TIMEFRAME = '15m'
SMA_WINDOW = 20
LOOKBACK_DAYS = 1 
SIZE = 1 
TARGET_PROFIT = 5       # 5% profit target
MAX_LOSS = -10          # 10% stop loss
LEVERAGE = 3
MAX_POSITIONS = 1 

# API Key
SECRET_KEY = d.private_key

def check_position_status(account):
    """
    Check the current position status for the configured symbol
    
    Args:
        account (LocalAccount): User's trading account
        
    Returns:
        tuple: Position details and status
    """
    try:
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long, num_of_pos = n.get_position_andmaxpos(
            SYMBOL, account, MAX_POSITIONS
        )
        
        print(f'Current positions for {SYMBOL}: {positions}')
        return positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long, num_of_pos
        
    except Exception as e:
        print(f"Error checking position status: {e}")
        return [], False, 0, None, 0, 0, None, 0

def prepare_trading_parameters(account):
    """
    Set leverage and calculate position size for trading
    
    Args:
        account (LocalAccount): User's trading account
        
    Returns:
        tuple: (leverage, position_size)
    """
    try:
        leverage, pos_size = n.adjust_leverage_size_signal(SYMBOL, LEVERAGE, account)
        
        # Reduce position size by half to manage risk
        adjusted_size = pos_size / 2
        print(f"Adjusted position size: {adjusted_size} (half of calculated size {pos_size})")
        
        return leverage, adjusted_size
        
    except Exception as e:
        print(f"Error preparing trading parameters: {e}")
        return LEVERAGE, SIZE  # Default fallback values

def check_bollinger_bands():
    """
    Check if Bollinger Bands are currently tight, indicating potential breakout
    
    Returns:
        tuple: (DataFrame with BB data, bool indicating if bands are tight)
    """
    try:
        # Get Bitcoin data as market indicator
        snapshot_data = n.get_ohlcv2('BTC', '1m', 500)
        if not snapshot_data:
            print("Failed to get BTC data for Bollinger Band calculation")
            return pd.DataFrame(), False
            
        df = n.process_data_to_df(snapshot_data)
        if df.empty:
            print("Empty dataframe from BTC data")
            return df, False
            
        bb_df, bands_tight, _ = n.calculate_bollinger_bands(df)
        
        print(f'Bollinger bands compression detected: {bands_tight}')
        return bb_df, bands_tight
        
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        return pd.DataFrame(), False

def place_entry_orders(account, pos_size):
    """
    Place both buy and sell limit orders at specified prices
    
    Args:
        account (LocalAccount): User's trading account
        pos_size (float): Position size to trade
    """
    try:
        # Get current market prices
        ask, bid, l2_data = n.ask_bid(SYMBOL)
        print(f'Current prices - Ask: {ask}, Bid: {bid}')
        
        # Get prices 10 levels deep for better entry
        if len(l2_data) >= 2 and len(l2_data[0]) > 10 and len(l2_data[1]) > 10:
            bid10 = float(l2_data[0][10]['px'])
            ask10 = float(l2_data[1][10]['px'])
            print(f'Level 10 prices - Ask: {ask10}, Bid: {bid10}')
        else:
            # Fall back to regular bid/ask if order book isn't deep enough
            bid10 = bid
            ask10 = ask
            print("Order book not deep enough, using top of book prices")
        
        # Cancel any existing orders first
        n.cancel_all_orders(account)
        print('Cancelled existing orders before placing new ones')
        
        # Place buy order
        n.limit_order(SYMBOL, True, pos_size, bid10, False, account)
        print(f'Placed buy order for {pos_size} at {bid10}')
        
        # Place sell order
        n.limit_order(SYMBOL, False, pos_size, ask10, False, account)
        print(f'Placed sell order for {pos_size} at {ask10}')
        
    except Exception as e:
        print(f"Error placing entry orders: {e}")

def bot_iteration():
    """
    Main bot trading logic, executed on each scheduled interval
    """
    try:
        print(f"\n--- Bot iteration starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        # Initialize account
        account = eth_account.Account.from_key(SECRET_KEY)
        
        # Check current position status
        _, im_in_pos, _, _, _, _, _, _ = check_position_status(account)
        
        # Set leverage and calculate position size
        _, pos_size = prepare_trading_parameters(account)
        
        # If in position, check profit/loss targets
        if im_in_pos:
            print('Currently in position, monitoring PnL targets')
            n.cancel_all_orders(account)
            n.pnl_close(SYMBOL, TARGET_PROFIT, MAX_LOSS, account)
        else:
            print('Not in position, checking for entry conditions')
        
        # Check Bollinger Band compression
        _, bands_tight = check_bollinger_bands()
        
        # Trading decision logic
        if not im_in_pos and bands_tight:
            print('Entry condition met: Bollinger bands are tight and no current position')
            place_entry_orders(account, pos_size)
            
        elif not bands_tight:
            print('Bollinger bands not tight, cancelling orders and closing positions')
            n.cancel_all_orders(account)
            n.close_all_positions(account)
            
        else:
            print(f'No action taken. In position: {im_in_pos}, Bands tight: {bands_tight}')
            
        print(f"--- Bot iteration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
    except Exception as e:
        print(f"Critical error in bot execution: {e}")

def run_bot():
    """
    Run the bot once immediately, then schedule recurring executions
    """
    print("Starting Bollinger Band Trading Bot")
    
    # Run once immediately
    bot_iteration()
    
    # Schedule recurring execution
    schedule.every(30).seconds.do(bot_iteration)
    
    print("Bot scheduled to run every 30 seconds")
    
    # Main loop with error handling
    while True:
        try:
            schedule.run_pending()
            time.sleep(10)
        except Exception as e:
            print(f'Connection issue or error occurred: {e}')
            print('Waiting 30 seconds before retry...')
            time.sleep(30)

# Execute the bot when script is run directly
if __name__ == "__main__":
    run_bot()