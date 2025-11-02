# Fix numpy NaN import issue in pandas_ta
import numpy
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan

import json, sys, os, time, requests, pandas as pd, pandas_ta as ta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects import dontshare as d 
from hyperliquid.info import Info 
from hyperliquid.exchange import Exchange 
from hyperliquid.utils import constants 
from datetime import datetime, timedelta

# Constants
SECRET_KEY = d.private_key
DEFAULT_SYMBOL = 'WIF'

def ask_bid(symbol):
    '''
    Gets the ask and bid prices for any symbol passed in
    
    Args:
        symbol (str): The trading symbol to query
        
    Returns:
        tuple: (ask_price, bid_price, l2_data) where l2_data contains the full order book data
        
    Raises:
        Exception: If API request fails or response has unexpected format
    '''
    try:
        url = 'https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}

        data = {
            'type': 'l2Book', 
            'coin': symbol
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        
        l2_data = response.json()
        
        if 'levels' not in l2_data:
            raise ValueError(f"Unexpected API response format: 'levels' key missing in response for {symbol}")
            
        l2_data = l2_data['levels']
        
        if len(l2_data) < 2 or not l2_data[0] or not l2_data[1]:
            raise ValueError(f"Insufficient order book data for {symbol}")

        # get ask bid 
        bid = float(l2_data[0][0]['px'])
        ask = float(l2_data[1][0]['px'])

        return ask, bid, l2_data
        
    except requests.exceptions.RequestException as e:
        print(f"API request error for {symbol}: {e}")
        raise
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error processing data for {symbol}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error for {symbol}: {e}")
        raise

def get_sz_px_decimals(coin):
    ''' 
    Returns size decimals and price decimals for a given coin/symbol
    
    Args:
        coin (str): The trading symbol
        
    Returns:
        tuple: (sz_decimals, px_decimals) where sz_decimals is the size precision and px_decimals is the price precision
    '''
    try:
        url = 'https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}
        data = {'type': 'meta'}

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()

        data = response.json()
        if 'universe' not in data:
            raise ValueError(f"Unexpected API response format: 'universe' key missing")
            
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == coin), None)
        
        if not symbol_info:
            raise ValueError(f"Symbol {coin} not found in API response")
            
        sz_decimals = symbol_info['szDecimals']

        # Get current ask price to determine price decimals
        ask = ask_bid(coin)[0]
        ask_str = str(ask)
        
        if '.' in ask_str:
            px_decimals = len(ask_str.split('.')[1])
        else:
            px_decimals = 0 

        print(f'{coin} size precision: {sz_decimals} decimals, price precision: {px_decimals} decimals')

        return sz_decimals, px_decimals
        
    except Exception as e:
        print(f"Error getting decimals for {coin}: {e}")
        raise


# MAKE A BUY AND A SELL ORDER
def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    """
    Place a limit order on the exchange
    
    Args:
        coin (str): Trading symbol
        is_buy (bool): True for buy order, False for sell order
        sz (float): Order size
        limit_px (float): Limit price
        reduce_only (bool): Whether this is a reduce-only order
        account (LocalAccount): User's account
        
    Returns:
        dict: Order result from the exchange
        
    Raises:
        Exception: If order placement fails
    """
    try:
        exchange = Exchange(account, constants.MAINNET_API_URL)
        rounding = get_sz_px_decimals(coin)[0]
        sz = round(sz, rounding)
        
        print(f'Placing limit order for {coin}: {"BUY" if is_buy else "SELL"} {sz} @ {limit_px}')
        
        order_result = exchange.order(
            coin, is_buy, sz, limit_px, 
            {"limit": {"tif": 'Gtc'}}, 
            reduce_only=reduce_only
        )

        if is_buy:
            print(f"Limit BUY order placed: {order_result['response']['data']['statuses'][0]}")
        else:
            print(f"Limit SELL order placed: {order_result['response']['data']['statuses'][0]}")

        return order_result
        
    except Exception as e:
        print(f"Error placing limit order for {coin}: {e}")
        raise

def acct_bal(account):
    """
    Get the account balance
    
    Args:
        account (LocalAccount): User's account
        
    Returns:
        float: Current account value
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        acct_value = user_state["marginSummary"]["accountValue"]
        
        print(f'Current account value: {acct_value}')
        return float(acct_value)
        
    except Exception as e:
        print(f"Error getting account balance: {e}")
        raise


def adjust_leverage_size_signal(symbol, leverage, account):
    """
    Calculate position size based on account value and leverage
    
    Args:
        symbol (str): Trading symbol
        leverage (float): Desired leverage
        account (LocalAccount): User's account
        
    Returns:
        tuple: (leverage, size) The leverage and calculated position size
    """
    try:
        print(f'Setting leverage to {leverage}')

        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get the user state and account value
        user_state = info.user_state(account.address)
        acct_value = float(user_state["marginSummary"]["accountValue"])
        acct_val95 = acct_value * 0.95

        # Update leverage
        leverage_result = exchange.update_leverage(leverage, symbol)
        print(f"Leverage update result: {leverage_result}")

        # Get current price
        price = ask_bid(symbol)[0]

        # Calculate size
        size = (acct_val95 / price) * leverage
        
        # Round size to appropriate precision
        rounding = get_sz_px_decimals(symbol)[0]
        size = round(size, rounding)
        
        return leverage, size
        
    except Exception as e:
        print(f"Error adjusting leverage and size for {symbol}: {e}")
        raise


def get_position(symbol, account):
    """
    Get current position information for a symbol
    
    Args:
        symbol (str): Trading symbol
        account (LocalAccount): User's account
        
    Returns:
        tuple: (positions, in_pos, size, pos_sym, entry_px, pnl_perc, long)
            positions (list): List of position objects
            in_pos (bool): Whether user is in a position
            size (float): Position size
            pos_sym (str): Position symbol
            entry_px (float): Entry price
            pnl_perc (float): PnL percentage
            long (bool): True if long position, False if short, None if no position
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        print(f'Current account value: {user_state["marginSummary"]["accountValue"]}')
        
        positions = []
        
        for position in user_state["assetPositions"]:
            if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
                positions.append(position["position"])
                in_pos = True 
                size = float(position["position"]["szi"])
                pos_sym = position["position"]["coin"]
                entry_px = float(position["position"]["entryPx"])
                pnl_perc = float(position["position"]["returnOnEquity"])*100
                print(f'Current PnL: {pnl_perc:.2f}%')
                break 
        else:
            in_pos = False 
            size = 0 
            pos_sym = None 
            entry_px = 0 
            pnl_perc = 0

        if size > 0:
            long = True 
        elif size < 0:
            long = False 
        else:
            long = None 

        return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long
        
    except Exception as e:
        print(f"Error getting position for {symbol}: {e}")
        raise


def cancel_all_orders(account):
    """
    Cancel all open orders
    
    Args:
        account (LocalAccount): User's account
    """
    try:
        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        open_orders = info.open_orders(account.address)
        if not open_orders:
            print("No open orders to cancel")
            return

        print(f'Cancelling {len(open_orders)} open orders...')
        for open_order in open_orders:
            try:
                exchange.cancel(open_order['coin'], open_order['oid'])
                print(f"Cancelled order for {open_order['coin']}")
            except Exception as e:
                print(f"Error cancelling order {open_order['oid']} for {open_order['coin']}: {e}")
                
    except Exception as e:
        print(f"Error cancelling orders: {e}")
        raise

def kill_switch(symbol, account):
    """
    Emergency function to close a position
    
    Args:
        symbol (str): Trading symbol
        account (LocalAccount): User's account
    """
    try:
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
        
        if not im_in_pos:
            print(f"No position to close for {symbol}")
            return
            
        print(f"Executing kill switch for {symbol} position")
        
        while im_in_pos:
            try:
                cancel_all_orders(account)

                ask, bid, l2 = ask_bid(pos_sym)
                pos_size = abs(pos_size)

                if long:
                    print(f"Closing LONG position of {pos_size} {pos_sym} at {bid}")
                    limit_order(pos_sym, False, pos_size, bid, True, account)
                else:
                    print(f"Closing SHORT position of {pos_size} {pos_sym} at {ask}")
                    limit_order(pos_sym, True, pos_size, ask, True, account)
                
                # Wait a moment for order to execute
                time.sleep(5)
                
                # Check if position is closed
                positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
                
            except Exception as e:
                print(f"Error during kill switch execution: {e}")
                time.sleep(2)  # Brief pause before retrying
                positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

        print(f"Position successfully closed for {symbol}")
        
    except Exception as e:
        print(f"Fatal error in kill switch for {symbol}: {e}")
        raise


def pnl_close(symbol, target, max_loss, account):
    """
    Monitors positions for their PnL and closes the position when target or max loss is hit
    
    Args:
        symbol (str): Trading symbol
        target (float): Target profit percentage
        max_loss (float): Maximum loss percentage (should be negative)
        account (LocalAccount): User's account
    """
    try:
        print(f'Monitoring {symbol} for PnL targets: profit {target}%, loss {max_loss}%')
        
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
        
        if not im_in_pos:
            print(f"No position to monitor for {symbol}")
            return
            
        position_type = "LONG" if long else "SHORT"
        print(f"Current {position_type} position: Size {pos_size}, Entry {entry_px}, PnL {pnl_perc:.2f}%")
        
        if pnl_perc >= target:
            print(f"🎯 Target reached! PnL: {pnl_perc:.2f}% >= Target: {target}%. Closing position.")
            kill_switch(pos_sym, account)
            return True
        elif pnl_perc <= max_loss:
            print(f"⚠️ Stop loss hit! PnL: {pnl_perc:.2f}% <= Max Loss: {max_loss}%. Closing position.")
            kill_switch(pos_sym, account)
            return True
        else:
            print(f"Position within parameters. PnL: {pnl_perc:.2f}%, Target: {target}%, Stop: {max_loss}%")
            return False
            
    except Exception as e:
        print(f"Error in PnL monitoring for {symbol}: {e}")
        raise


def get_position_andmaxpos(symbol, account, max_positions):
    """
    Gets current position info and enforces maximum position limit
    
    Args:
        symbol (str): Trading symbol
        account (LocalAccount): User's account
        max_positions (int): Maximum number of positions allowed
        
    Returns:
        tuple: Same as get_position() plus the number of open positions
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        print(f'Current account value: {user_state["marginSummary"]["accountValue"]}')
        
        # Get all open positions
        open_positions = []
        for position in user_state["assetPositions"]:
            if float(position["position"]["szi"]) != 0:
                open_positions.append(position["position"]["coin"])
        
        num_of_pos = len(open_positions)
        
        # Check if we need to close positions due to exceeding max_positions
        if num_of_pos > max_positions:
            print(f"⚠️ Position limit exceeded: {num_of_pos}/{max_positions}. Closing all positions.")
            for position_symbol in open_positions:
                kill_switch(position_symbol, account)
            # After closing positions, we should have 0 open positions
            num_of_pos = 0
            # Return default empty position info
            return [], False, 0, None, 0, 0, None, 0
        else:
            print(f"Position count: {num_of_pos}/{max_positions}")
        
        # Now get specific position info for the requested symbol
        positions = []
        in_pos = False
        size = 0
        pos_sym = None
        entry_px = 0
        pnl_perc = 0
        long = None
        
        for position in user_state["assetPositions"]:
            if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
                positions.append(position["position"])
                in_pos = True 
                size = float(position["position"]["szi"])
                pos_sym = position["position"]["coin"]
                entry_px = float(position["position"]["entryPx"])
                pnl_perc = float(position["position"]["returnOnEquity"])*100
                print(f'Current {symbol} PnL: {pnl_perc:.2f}%')
                break 
        
        if size > 0:
            long = True 
        elif size < 0:
            long = False 
        
        return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long, num_of_pos
    
    except Exception as e:
        print(f"Error getting position and enforcing max positions: {e}")
        raise


def close_all_positions(account):
    """
    Close all open positions across all symbols
    
    Args:
        account (LocalAccount): User's account
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        print(f'Current account value: {user_state["marginSummary"]["accountValue"]}')
        
        # Cancel all orders first
        cancel_all_orders(account)
        print('All orders have been cancelled')
        
        # Find all open positions
        open_positions = []
        for position in user_state["assetPositions"]:
            if float(position["position"]["szi"]) != 0:
                open_positions.append(position["position"]["coin"])
        
        if not open_positions:
            print("No open positions to close")
            return
            
        print(f"Closing {len(open_positions)} open positions: {', '.join(open_positions)}")
        
        # Close each position
        for position_symbol in open_positions:
            kill_switch(position_symbol, account)
        
        print('All positions have been closed')
        
    except Exception as e:
        print(f"Error closing all positions: {e}")
        raise

def calculate_bollinger_bands(df, length=20, std_dev=2):
    """
    Calculate Bollinger Bands for a given DataFrame and classify when the bands are tight vs wide.

    Args:
        df (DataFrame): DataFrame with a 'close' column
        length (int): The period over which the SMA is calculated. Default is 20
        std_dev (int): Number of standard deviations for the bands. Default is 2

    Returns:
        tuple: (df, tight, wide)
            df (DataFrame): Original DataFrame with Bollinger Bands columns added
            tight (bool): True if bands are currently tight
            wide (bool): True if bands are currently wide
    """
    try:
        # Ensure 'close' is numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Calculate Bollinger Bands using pandas_ta
        bollinger_bands = ta.bbands(df['close'], length=length, std=std_dev)

        # Handle empty bollinger_bands result
        if bollinger_bands.empty:
            print("Warning: Unable to calculate Bollinger Bands (not enough data)")
            df['BBL'] = df['BBM'] = df['BBU'] = df['BandWidth'] = None
            return df, False, False

        # Select only the Bollinger Bands columns
        bollinger_bands = bollinger_bands.iloc[:, [0, 1, 2]]  # BBL, BBM, BBU
        bollinger_bands.columns = ['BBL', 'BBM', 'BBU']

        # Merge the Bollinger Bands into the original DataFrame
        df = pd.concat([df, bollinger_bands], axis=1)

        # Calculate Band Width
        df['BandWidth'] = df['BBU'] - df['BBL']

        # Check if we have enough data for quantile calculations
        if len(df) < 5:
            print("Warning: Not enough data for Bollinger Band width classification")
            return df, False, False

        # Determine thresholds for 'tight' and 'wide' bands
        tight_threshold = df['BandWidth'].quantile(0.2)
        wide_threshold = df['BandWidth'].quantile(0.8)

        # Classify the current state of the bands
        current_band_width = df['BandWidth'].iloc[-1]
        tight = current_band_width <= tight_threshold
        wide = current_band_width >= wide_threshold

        return df, tight, wide
        
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        # Return original dataframe and False flags as fallback
        return df, False, False

def process_data_to_df(snapshot_data):
    """
    Process raw OHLCV data into a DataFrame with additional indicators
    
    Args:
        snapshot_data (list): List of candle data dictionaries
        
    Returns:
        DataFrame: Processed dataframe with OHLCV data and indicators
    """
    try:
        if not snapshot_data:
            print("Warning: No data to process")
            return pd.DataFrame()
            
        # Set up DataFrame columns
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        
        for snapshot in snapshot_data:
            # Convert timestamps and extract data
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            open_price = snapshot['o']
            high_price = snapshot['h']
            low_price = snapshot['l']
            close_price = snapshot['c']
            volume = snapshot['v']
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=columns)
        
        # Ensure numeric types for calculations
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Calculate support and resistance
        if len(df) > 2:
            df['support'] = df[:-2]['close'].min()
            df['resis'] = df[:-2]['close'].max()
        else:
            # If DataFrame has 2 or fewer rows
            df['support'] = df['close'].min()
            df['resis'] = df['close'].max()

        return df
        
    except Exception as e:
        print(f"Error processing data to DataFrame: {e}")
        return pd.DataFrame()

def get_ohlcv2(symbol, interval, lookback_days):
    """
    Fetch OHLCV data for a symbol within specified timeframe
    
    Args:
        symbol (str): Trading symbol
        interval (str): Candle interval (e.g., '1m', '5m', '1h')
        lookback_days (int): Number of days to look back
        
    Returns:
        list: List of candle data dictionaries or None if error
    """
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
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

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        snapshot_data = response.json()
        
        if not snapshot_data:
            print(f"Warning: No data returned for {symbol} with interval {interval}")
            
        return snapshot_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OHLCV data for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching OHLCV data for {symbol}: {e}")
        return None
    
def fetch_candle_snapshot(symbol, interval, start_time, end_time):
    """
    Fetch candle data for a symbol in a specific time range
    
    Args:
        symbol (str): Trading symbol
        interval (str): Candle interval (e.g., '1m', '5m', '1h')
        start_time (datetime): Start time
        end_time (datetime): End time
        
    Returns:
        list: List of candle data dictionaries or None if error
    """
    try:
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

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        snapshot_data = response.json()
        
        if not snapshot_data:
            print(f"Warning: No data returned for {symbol} with interval {interval}")
            
        return snapshot_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching candle snapshot for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching candle snapshot for {symbol}: {e}")
        return None

def calculate_sma(prices, window):
    """
    Calculate Simple Moving Average for a series of prices
    
    Args:
        prices (Series): Price series
        window (int): SMA window/period
        
    Returns:
        float: Latest SMA value or None if error
    """
    try:
        if len(prices) < window:
            print(f"Warning: Not enough data to calculate {window}-period SMA")
            return None
            
        sma = prices.rolling(window=window).mean()
        return sma.iloc[-1]
        
    except Exception as e:
        print(f"Error calculating SMA: {e}")
        return None

def get_latest_sma(symbol, interval, window, lookback_days=1):
    """
    Get the latest SMA value for a symbol
    
    Args:
        symbol (str): Trading symbol
        interval (str): Candle interval (e.g., '1m', '5m', '1h')
        window (int): SMA window/period
        lookback_days (int): Number of days to look back
        
    Returns:
        float: Latest SMA value or None if error
    """
    try:
        start_time = datetime.now() - timedelta(days=lookback_days)
        end_time = datetime.now()

        snapshots = fetch_candle_snapshot(symbol, interval, start_time, end_time)

        if not snapshots or len(snapshots) < window:
            print(f"Warning: Not enough data to calculate {window}-period SMA for {symbol}")
            return None

        prices = pd.Series([float(snapshot['c']) for snapshot in snapshots])
        latest_sma = calculate_sma(prices, window)
        
        return latest_sma
        
    except Exception as e:
        print(f"Error getting latest SMA for {symbol}: {e}")
        return None

def supply_demand_zones_hl(symbol, timeframe, limit):
    """
    Calculate supply and demand zones for a symbol
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Candle interval (e.g., '1m', '5m', '1h')
        limit (int): Number of days to look back
        
    Returns:
        DataFrame: Supply and demand zones or empty DataFrame if error
    """
    try:
        print(f'Calculating supply and demand zones for {symbol} on {timeframe} timeframe...')

        sd_df = pd.DataFrame()

        snapshot_data = get_ohlcv2(symbol, timeframe, limit)
        if not snapshot_data:
            print(f"No data available for {symbol} on {timeframe} timeframe")
            return sd_df
            
        df = process_data_to_df(snapshot_data)
        if df.empty:
            print(f"Failed to process data for {symbol}")
            return sd_df

        # Calculate support and resistance levels
        supp = df.iloc[-1]['support']
        resis = df.iloc[-1]['resis']

        # Calculate additional levels using lows and highs
        df['supp_lo'] = df[:-2]['low'].min()
        supp_lo = df.iloc[-1]['supp_lo']

        df['res_hi'] = df[:-2]['high'].max()
        res_hi = df.iloc[-1]['res_hi']

        # Store results in DataFrame
        sd_df[f'{timeframe}_dz'] = [supp_lo, supp]  # Demand zones
        sd_df[f'{timeframe}_sz'] = [res_hi, resis]  # Supply zones

        print(f'Supply and demand zones for {symbol} on {timeframe}:')
        print(sd_df)

        return sd_df
        
    except Exception as e:
        print(f"Error calculating supply and demand zones for {symbol}: {e}")
        return pd.DataFrame()

def calculate_vwap_with_symbol(symbol):
    """
    Calculate Volume Weighted Average Price (VWAP) for a symbol
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        tuple: (df, latest_vwap)
            df (DataFrame): DataFrame with VWAP column added
            latest_vwap (float): Latest VWAP value
    """
    try:
        # Fetch and process data
        snapshot_data = get_ohlcv2(symbol, '15m', 300)
        if not snapshot_data:
            print(f"No data available for {symbol}")
            return pd.DataFrame(), None
            
        df = process_data_to_df(snapshot_data)
        if df.empty:
            print(f"Failed to process data for {symbol}")
            return df, None

        # Convert the 'timestamp' column to datetime and set as the index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Ensure all columns used for VWAP calculation are of numeric type
        numeric_columns = ['high', 'low', 'close', 'volume']
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        # Drop rows with NaNs created during type conversion
        df.dropna(subset=numeric_columns, inplace=True)

        # Ensure the DataFrame is ordered by datetime
        df.sort_index(inplace=True)

        # Check if we have enough data for VWAP
        if len(df) < 2:
            print(f"Not enough data to calculate VWAP for {symbol}")
            return df, None

        # Calculate VWAP and add it as a new column
        df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])

        # Retrieve the latest VWAP value from the DataFrame
        latest_vwap = df['VWAP'].iloc[-1]

        return df, latest_vwap
        
    except Exception as e:
        print(f"Error calculating VWAP for {symbol}: {e}")
        return pd.DataFrame(), None
