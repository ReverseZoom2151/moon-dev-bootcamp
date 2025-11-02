import os, sys, eth_account, json, time, requests, logging, datetime, pandas as pd, logging
from dotenv import load_dotenv

# Fix numpy NaN import issue for pandas_ta
import numpy as np
np.NaN = np.nan  # Define NaN for backward compatibility

# Now import pandas_ta after the fix
import pandas_ta as ta

from Day_4_Projects import dontshare as d 
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info 
from hyperliquid.exchange import Exchange 
from hyperliquid.utils import constants 
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()
secret_key = os.getenv("HYPERLIQUID_SECRET_KEY", d.private_key)

# Simple time-based cache
cache = {}
cache_ttl = {}

def cached_get(key, ttl_seconds, fetch_func):
    now = time.time()
    if key in cache and now - cache_ttl[key] < ttl_seconds:
        return cache[key]
    
    result = fetch_func()
    if result is not None:
        cache[key] = result
        cache_ttl[key] = now
    return result

class HyperliquidClient:
    def __init__(self, account):
        self.account = account
        self.exchange = Exchange(account, constants.MAINNET_API_URL)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    def get_position(self, symbol):
        # Implementation here
        pass

def ask_bid(symbol, max_retries=3, retry_delay=2):
    """
    Get current ask and bid prices for a symbol with retry logic.
    
    Args:
        symbol (str): Trading symbol
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Seconds to wait between retries
        
    Returns:
        tuple: (ask_price, bid_price, l2_data) or (None, None, None) on error
    """
    retry_count = 0
    
    # Remove exchange-specific formatting if present
    api_symbol = symbol
    if '/' in symbol:
        parts = symbol.split('/')
        api_symbol = parts[0]
    if ':' in api_symbol:
        api_symbol = api_symbol.split(':')[0]
    
    logger.info(f"Using API symbol {api_symbol} for {symbol}")
    
    while retry_count < max_retries:
        try:
            url = 'https://api.hyperliquid.xyz/info'
            headers = {'Content-Type': 'application/json'}
            
            data = {
                'type': 'l2Book', 
                'coin': api_symbol
            }
            
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                l2_data = response.json()
                l2_data = l2_data['levels']
                
                # Get ask and bid
                bid = float(l2_data[0][0]['px'])
                ask = float(l2_data[1][0]['px'])
                
                return ask, bid, l2_data
            else:
                logger.error(f"Error fetching ask/bid for {symbol}: {response.status_code}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying ask_bid ({retry_count}/{max_retries})...")
                    time.sleep(retry_delay)
                    
        except Exception as e:
            logger.error(f"Error in ask_bid for {symbol}: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying ask_bid ({retry_count}/{max_retries})...")
                time.sleep(retry_delay)
    
    # All retries failed, check cache for last known values
    cached_key = f"ask_bid_{api_symbol}"
    if cached_key in cache:
        logger.warning(f"Using cached values for {symbol}")
        return cache[cached_key]
    
    logger.error(f"All ask_bid attempts failed for {symbol}")
    return None, None, None

def get_sz_px_decimals(symbol):
    """
    Get size and price decimals for a symbol.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        tuple: (size_decimals, price_decimals) or (None, None) on error
    """
    try:
        url = 'https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}
        data = {'type': 'meta'}

        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            data = response.json()
            symbols = data['universe']
            symbol_info = next((s for s in symbols if s['name'] == symbol), None)
            if symbol_info:
                sz_decimals = symbol_info['szDecimals']
            else:
                logger.error(f'Symbol {symbol} not found')
                return None, None
        else:
            logger.error(f'Error in meta API: {response.status_code}')
            return None, None
        
        # Get price decimals from ask price
        price_data = ask_bid(symbol)
        if price_data[0] is None:
            logger.warning(f"Couldn't get price for {symbol}, using 0 decimals")
            return sz_decimals, 0
            
        ask = price_data[0]
        ask_str = str(ask)
        if '.' in ask_str:
            px_decimals = len(ask_str.split('.')[1])
        else:
            px_decimals = 0 

        logger.info(f'{symbol} - size decimals: {sz_decimals}, price decimals: {px_decimals}')
        return sz_decimals, px_decimals
    except Exception as e:
        logger.error(f"Error in get_sz_px_decimals for {symbol}: {str(e)}")
        return None, None

def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    """
    Place a limit order.
    
    Args:
        coin (str): Trading symbol
        is_buy (bool): True for buy, False for sell
        sz (float): Order size
        limit_px (float): Limit price
        reduce_only (bool): Whether the order is reduce-only
        account: Account object for authentication
        
    Returns:
        dict: Order result or None on error
    """
    try:
        exchange = Exchange(account, constants.MAINNET_API_URL)
        
        # Round size to appropriate decimals
        decimals = get_sz_px_decimals(coin)
        if decimals and decimals[0] is not None:
            rounding = decimals[0]
            sz = round(sz, rounding)
        else:
            logger.warning(f"Could not get size decimals for {coin}, using unrounded size")
        
        order_type = "BUY" if is_buy else "SELL"
        logger.info(f'Placing limit {order_type} order for {coin} {sz} @ {limit_px}')
        
        order_result = exchange.order(
            coin, 
            is_buy, 
            sz, 
            limit_px, 
            {"limit": {"tif": 'Gtc'}}, 
            reduce_only=reduce_only
        )

        logger.info(f"Limit {order_type} order placed: {order_result['response']['data']['statuses'][0]}")
        return order_result
    except Exception as e:
        logger.error(f"Error placing limit order for {coin}: {str(e)}")
        return None

def acct_bal(account):
    """
    Get account balance.
    
    Args:
        account: Account object for authentication
        
    Returns:
        float: Account value or 0 on error
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        acct_value = float(user_state["marginSummary"]["accountValue"])
        logger.info(f'Current account value: {acct_value}')
        return acct_value
    except Exception as e:
        logger.error(f"Error getting account balance: {str(e)}")
        return 0

def adjust_leverage_size_signal(symbol, leverage, account):
    """
    Adjust leverage for a symbol and calculate potential position size.
    
    Args:
        symbol (str): Trading symbol
        leverage (int): Desired leverage
        account: Account object for authentication
        
    Returns:
        tuple: (leverage, potential_size) or (leverage, 0) on error
    """
    try:
        logger.info(f'Setting leverage to {leverage} for {symbol}')

        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get current account value
        user_state = info.user_state(account.address)
        acct_value = float(user_state["marginSummary"]["accountValue"])
        acct_val95 = acct_value * 0.95  # Use 95% of balance

        # Update leverage
        leverage_result = exchange.update_leverage(leverage, symbol)
        logger.info(f"Leverage update result: {leverage_result}")

        # Get current price
        price_data = ask_bid(symbol)
        if price_data[0] is None:
            logger.error(f"Could not get price for {symbol}")
            return leverage, 0
            
        price = price_data[0]

        # Calculate size based on account value and leverage
        size = (acct_val95 / price) * leverage
        
        # Round to appropriate decimal places
        decimals = get_sz_px_decimals(symbol)
        if decimals and decimals[0] is not None:
            rounding = decimals[0]
            size = round(size, rounding)
        
        logger.info(f"Calculated potential position size: {size} {symbol}")
        return leverage, size
    except Exception as e:
        logger.error(f"Error adjusting leverage for {symbol}: {str(e)}")
        return leverage, 0

def get_position(symbol: str, account: LocalAccount) -> tuple[list, bool, float, str, float, float, bool]:
    """
    Get current position information for a symbol.
    
    Args:
        symbol (str): Trading symbol
        account: Account object for authentication
        
    Returns:
        tuple: (positions, in_position, size, symbol, entry_price, pnl_percent, is_long)
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        
        positions = []
        in_pos = False
        size = 0
        pos_sym = None
        entry_px = 0
        pnl_perc = 0
        long = None
        
        logger.info(f'Checking position for {symbol}')
        
        for position in user_state["assetPositions"]:
            if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
                positions.append(position["position"])
                in_pos = True 
                size = float(position["position"]["szi"])
                pos_sym = position["position"]["coin"]
                entry_px = float(position["position"]["entryPx"])
                pnl_perc = float(position["position"]["returnOnEquity"])*100
                logger.info(f'Found position: size={size}, entry={entry_px}, PNL={pnl_perc:.2f}%')
                break 
        
        # Determine if long or short based on position size
        if size > 0:
            long = True
        elif size < 0:
            long = False
        else:
            long = None
            
        if not in_pos:
            logger.info(f"No position for {symbol}")
            
        return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {str(e)}")
        return [], False, 0, None, 0, 0, None

def get_position_andmaxpos(symbol, account, max_positions):
    """
    Get position and check against maximum allowed positions.
    If exceeded, close positions.
    
    Args:
        symbol (str): Trading symbol
        account: Account object for authentication
        max_positions (int): Maximum allowed positions
        
    Returns:
        tuple: Position information (see get_position)
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        
        open_positions = []
        
        logger.info(f'Checking position count against max {max_positions}')
        
        # Check for open positions
        for position in user_state["assetPositions"]:
            if float(position["position"]["szi"]) != 0:
                open_positions.append(position["position"]["coin"])

        # If over max positions, close positions
        if len(open_positions) > max_positions:
            logger.warning(f'In {len(open_positions)} positions, max is {max_positions} - closing excess positions')
            
            # Close positions until under limit
            for position in open_positions:
                kill_switch(position, account)
                max_positions -= 1
                if max_positions <= 0:
                    break
        else:
            logger.info(f'In {len(open_positions)} positions, max is {max_positions} - within limits')

        # Get position info for the specific symbol
        return get_position(symbol, account)
    except Exception as e:
        logger.error(f"Error in get_position_andmaxpos: {str(e)}")
        return [], False, 0, None, 0, 0, None

def cancel_all_orders(account):
    """
    Cancel all open orders.
    
    Args:
        account: Account object for authentication
    """
    try:
        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        open_orders = info.open_orders(account.address)
        logger.info(f'Cancelling {len(open_orders)} open orders')
        
        for open_order in open_orders:
            exchange.cancel(open_order['coin'], open_order['oid'])
            
        logger.info("All orders cancelled")
    except Exception as e:
        logger.error(f"Error cancelling orders: {str(e)}")

def kill_switch(symbol, account):
    """
    Emergency position close function.
    Will retry until position is closed.
    
    Args:
        symbol (str): Trading symbol
        account: Account object for authentication
    """
    try:
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
        
        if not im_in_pos:
            logger.info(f"No position to close for {symbol}")
            return
            
        logger.warning(f"Emergency position close initiated for {symbol}")
        attempts = 0
        max_attempts = 5
        
        while im_in_pos and attempts < max_attempts:
            attempts += 1
            logger.info(f"Close attempt {attempts}/{max_attempts}")
            
            # Cancel any existing orders
            cancel_all_orders(account)

            # Get current market prices
            ask, bid, l2_data = ask_bid(pos_sym)
            if ask is None or bid is None:
                logger.error("Error getting market prices, retrying...")
                time.sleep(3)
                continue

            pos_size = abs(pos_size)

            # Place appropriate order to close position
            if long:
                limit_order(pos_sym, False, pos_size, ask, True, account)
                logger.info(f'Placed sell order to close position: {pos_size} @ {ask}')
            else:
                limit_order(pos_sym, True, pos_size, bid, True, account)
                logger.info(f'Placed buy order to close position: {pos_size} @ {bid}')
                
            time.sleep(5)
            
            # Check if position is closed
            positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

        if im_in_pos:
            logger.error(f"Failed to close position after {max_attempts} attempts")
        else:
            logger.info('Position successfully closed')
    except Exception as e:
        logger.error(f"Error in kill_switch for {symbol}: {str(e)}")

def pnl_close(symbol, target=None, max_loss=None, account=None):
    """
    Check if position PNL has hit target or stop loss levels and close if needed.
    This function supports two calling patterns:
    1. pnl_close(symbol) - For compatibility with 11_bot2.py
    2. pnl_close(symbol, target, max_loss, account) - Full version
    
    Args:
        symbol (str): Trading symbol
        target (float, optional): Target profit percentage
        max_loss (float, optional): Maximum loss percentage (negative value)
        account (object, optional): Account object for authentication
    """
    try:
        # Handle case where only symbol is provided (backward compatibility with 11_bot2.py)
        if target is None and max_loss is None and account is None:
            logger.info(f'Legacy pnl_close call for {symbol}')
            # Use default values for backward compatibility
            target = 9  # Default from 11_bot2.py
            max_loss = -8  # Default from 11_bot2.py
            account = eth_account.Account.from_key(secret_key)  # Use global secret key
        
        logger.info(f'Checking PNL for {symbol} (target: {target}%, max_loss: {max_loss}%)')
        
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
        
        if not im_in_pos:
            logger.info(f"No position to check PNL for {symbol}")
            return False

        if pnl_perc > target:
            logger.info(f'PNL gain is {pnl_perc:.2f}% and target is {target}% - closing position (WIN)')
            kill_switch(pos_sym, account)
            return True
        elif pnl_perc <= max_loss:
            logger.info(f'PNL loss is {pnl_perc:.2f}% and max loss is {max_loss}% - closing position (LOSS)')
            kill_switch(pos_sym, account)
            return True
        else:
            logger.info(f'PNL is {pnl_perc:.2f}%, target is {target}%, max loss is {max_loss}% - holding position')
            return False
    except Exception as e:
        logger.error(f"Error in pnl_close for {symbol}: {str(e)}")
        return False

def close_all_positions(account):
    """
    Close all open positions.
    
    Args:
        account: Account object for authentication
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        
        open_positions = []
        logger.info('Closing all positions')

        # Cancel all orders first
        cancel_all_orders(account)

        # Find all open positions
        for position in user_state["assetPositions"]:
            if float(position["position"]["szi"]) != 0:
                open_positions.append(position["position"]["coin"])

        logger.info(f"Found {len(open_positions)} positions to close")
        
        # Close all positions
        for position in open_positions:
            kill_switch(position, account)

        logger.info('All positions closed')
    except Exception as e:
        logger.error(f"Error closing all positions: {str(e)}")

def calculate_bollinger_bands(df, length=20, std_dev=2):
    """
    Calculate Bollinger Bands for a DataFrame.
    
    Args:
        df (DataFrame): DataFrame with close prices
        length (int): Period for calculation
        std_dev (int): Standard deviation multiplier
        
    Returns:
        tuple: (DataFrame with bands, is_tight, is_wide)
    """
    try:
        # Ensure 'close' is numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Calculate Bollinger Bands
        bollinger_bands = ta.bbands(df['close'], length=length, std=std_dev)
        
        # Select only the main Bollinger Bands columns
        bollinger_bands = bollinger_bands.iloc[:, [0, 1, 2]]  # BBL, BBM, BBU
        bollinger_bands.columns = ['BBL', 'BBM', 'BBU']

        # Merge the Bollinger Bands into the original DataFrame
        df = pd.concat([df, bollinger_bands], axis=1)

        # Calculate Band Width
        df['BandWidth'] = df['BBU'] - df['BBL']

        # Determine thresholds for 'tight' and 'wide' bands
        tight_threshold = df['BandWidth'].quantile(0.2)
        wide_threshold = df['BandWidth'].quantile(0.8)

        # Classify the current state of the bands
        current_band_width = df['BandWidth'].iloc[-1]
        tight = current_band_width <= tight_threshold
        wide = current_band_width >= wide_threshold

        return df, tight, wide
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return df, False, False

def process_data_to_df(snapshot_data):
    """
    Process snapshot data into a DataFrame with calculated fields.
    
    Args:
        snapshot_data (list): Candle data
        
    Returns:
        DataFrame: Processed data with support and resistance
    """
    try:
        if not snapshot_data:
            logger.warning("No snapshot data to process")
            return pd.DataFrame()
            
        # Process candle data
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        
        for snapshot in snapshot_data:
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            open_price = float(snapshot['o'])
            high_price = float(snapshot['h'])
            low_price = float(snapshot['l'])
            close_price = float(snapshot['c'])
            volume = float(snapshot['v'])
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=columns)
        
        # Ensure all numeric columns are float type
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate support and resistance
        if len(df) > 2:  # Check if DataFrame has more than 2 rows
            df['support'] = df[:-2]['close'].min()
            df['resis'] = df[:-2]['close'].max()
        else:  # If DataFrame has 2 or fewer rows, use all data
            df['support'] = df['close'].min()
            df['resis'] = df['close'].max()
            
        # Ensure support and resistance are float type
        df['support'] = pd.to_numeric(df['support'], errors='coerce')
        df['resis'] = pd.to_numeric(df['resis'], errors='coerce')

        return df
    except Exception as e:
        logger.error(f"Error processing data to DataFrame: {str(e)}")
        return pd.DataFrame()

def get_ohlcv2(symbol, interval, lookback_days, max_retries=3, retry_delay=2):
    """
    Get OHLCV data for a symbol with retry logic.
    
    Args:
        symbol (str): Trading symbol
        interval (str): Timeframe interval (e.g. '15m', '1h')
        lookback_days (int): Number of days to look back
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Seconds to wait between retries
        
    Returns:
        list: Candle data or None on error
    """
    retry_count = 0
    
    # Remove exchange-specific formatting if present
    api_symbol = symbol
    if '/' in symbol:
        parts = symbol.split('/')
        api_symbol = parts[0]
    if ':' in api_symbol:
        api_symbol = api_symbol.split(':')[0]
    
    logger.info(f"Using API symbol {api_symbol} for {symbol} OHLCV")
    
    while retry_count < max_retries:
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            url = 'https://api.hyperliquid.xyz/info'
            headers = {'Content-Type': 'application/json'}
            data = {
                "type": "candleSnapshot",
                "req": {
                    "coin": api_symbol,
                    "interval": interval,
                    "startTime": int(start_time.timestamp() * 1000),
                    "endTime": int(end_time.timestamp() * 1000)
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                snapshot_data = response.json()
                # Cache the successful result
                cache_key = f"ohlcv_{api_symbol}_{interval}_{lookback_days}"
                cache[cache_key] = snapshot_data
                cache_ttl[cache_key] = time.time()
                return snapshot_data
            else:
                logger.error(f"Error fetching OHLCV data for {symbol}: {response.status_code}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying get_ohlcv2 ({retry_count}/{max_retries})...")
                    time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Error in get_ohlcv2 for {symbol}: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying get_ohlcv2 ({retry_count}/{max_retries})...")
                time.sleep(retry_delay)
    
    # All retries failed, check cache for last known values
    cache_key = f"ohlcv_{api_symbol}_{interval}_{lookback_days}"
    if cache_key in cache:
        ttl = 3600  # Use cached data for up to 1 hour
        if time.time() - cache_ttl[cache_key] < ttl:
            logger.warning(f"Using cached OHLCV data for {symbol} (age: {int(time.time() - cache_ttl[cache_key])}s)")
            return cache[cache_key]
    
    logger.error(f"All get_ohlcv2 attempts failed for {symbol}")
    
    # No data available, provide minimal mock data for the bot to continue
    if 'mock_data' not in globals():
        global mock_data
        mock_data = []
        for i in range(10):
            timestamp = int((datetime.now() - timedelta(minutes=i*15)).timestamp() * 1000)
            mock_data.append({
                't': timestamp,
                'o': 20000,
                'h': 20100,
                'l': 19900,
                'c': 20050,
                'v': 100
            })
    
    logger.warning(f"Using mock data for {symbol}")
    return mock_data

async def fetch_candle_snapshot_async(symbol, interval, start_time, end_time):
    # Async implementation
    pass

def calculate_sma(prices, window):
    """
    Calculate Simple Moving Average.
    
    Args:
        prices (Series): Price series
        window (int): SMA window
        
    Returns:
        float: SMA value or None on error
    """
    try:
        sma = prices.rolling(window=window).mean()
        return sma.iloc[-1]  # Return the most recent SMA value
    except Exception as e:
        logger.error(f"Error calculating SMA: {str(e)}")
        return None

def fetch_candle_snapshot(symbol, interval, start_time, end_time):
    """
    Fetch candle snapshot data for a specific time range.
    
    Args:
        symbol (str): Trading symbol
        interval (str): Timeframe interval
        start_time (datetime): Start time
        end_time (datetime): End time
        
    Returns:
        list: Candle data or None on error
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
        if response.status_code == 200:
            snapshot_data = response.json()
            return snapshot_data
        else:
            logger.error(f"Error fetching candle snapshot for {symbol}: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error in fetch_candle_snapshot for {symbol}: {str(e)}")
        return None

def get_latest_sma(symbol, interval, window, lookback_days=1):
    """
    Get the latest SMA value for a symbol.
    
    Args:
        symbol (str): Trading symbol
        interval (str): Timeframe interval
        window (int): SMA window
        lookback_days (int): Days to look back
        
    Returns:
        float: SMA value or None on error
    """
    try:
        start_time = datetime.now() - timedelta(days=lookback_days)
        end_time = datetime.now()

        snapshots = fetch_candle_snapshot(symbol, interval, start_time, end_time)

        if snapshots:
            prices = pd.Series([float(snapshot['c']) for snapshot in snapshots])
            latest_sma = calculate_sma(prices, window)
            return latest_sma
        else:
            logger.warning(f"No data for SMA calculation for {symbol}")
            return None
    except Exception as e:
        logger.error(f"Error in get_latest_sma for {symbol}: {str(e)}")
        return None

def supply_demand_zones_hl(symbol, timeframe, limit):
    """
    Calculate supply and demand zones.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe interval
        limit (int): Lookback days
        
    Returns:
        DataFrame: Supply and demand zones
    """
    try:
        logger.info(f'Calculating supply and demand zones for {symbol}')

        sd_df = pd.DataFrame()

        snapshot_data = get_ohlcv2(symbol, timeframe, limit)
        if snapshot_data is None:
            logger.error(f"No data for supply/demand calculation for {symbol}")
            return pd.DataFrame()
            
        df = process_data_to_df(snapshot_data)
        if df.empty:
            logger.error(f"Empty DataFrame for {symbol}")
            return pd.DataFrame()

        # Calculate supply and demand zones
        supp = df.iloc[-1]['support']
        resis = df.iloc[-1]['resis']

        # Add low support and high resistance
        df['supp_lo'] = df[:-2]['low'].min()
        supp_lo = df.iloc[-1]['supp_lo']

        df['res_hi'] = df[:-2]['high'].max()
        res_hi = df.iloc[-1]['res_hi']

        # Create the result DataFrame
        sd_df[f'{timeframe}_dz'] = [supp_lo, supp]
        sd_df[f'{timeframe}_sz'] = [res_hi, resis]

        logger.info(f'Supply and demand zones for {symbol}:\n{sd_df}')
        return sd_df
    except Exception as e:
        logger.error(f"Error calculating supply/demand zones for {symbol}: {str(e)}")
        return pd.DataFrame()

def df_sma(symbol, timeframe='15m', limit=100, sma=20):
    """
    Get DataFrame with data and SMA calculation.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe interval
        limit (int): Number of candles to fetch
        sma (int): SMA window
        
    Returns:
        DataFrame: Data with SMA, support, and resistance
    """
    try:
        logger.info(f'Getting SMA data for {symbol} on {timeframe} timeframe')
        
        # Calculate appropriate lookback days based on timeframe and limit
        # Convert common timeframes to fraction of a day
        timeframe_days_map = {
            '1m': 1/1440,    # 1 minute / 1440 minutes in a day
            '5m': 5/1440,
            '15m': 15/1440,
            '30m': 30/1440,
            '1h': 1/24,
            '4h': 4/24,
            '1d': 1
        }
        
        # Get days multiplier from map or default to 15m
        timeframe_days = timeframe_days_map.get(timeframe, 15/1440)
        
        # Calculate total days needed
        lookback_days = max(1, (limit * timeframe_days) * 1.1)  # Add 10% buffer
        
        logger.info(f'Fetching {limit} candles with {lookback_days:.2f} days lookback')
        
        # Get OHLCV data
        snapshot_data = get_ohlcv2(symbol, timeframe, lookback_days)
        
        # Convert to DataFrame
        df = process_data_to_df(snapshot_data)
        
        # Create empty DataFrame with necessary columns if we got no data
        if df.empty:
            logger.warning(f"Creating mock DataFrame for {symbol}")
            # Create a basic DataFrame with required columns
            timestamps = [(datetime.now() - timedelta(minutes=i*int(timeframe.replace('m', '')))).strftime('%Y-%m-%d %H:%M:%S') 
                         for i in range(limit)]
            
            # Create mock data with some volatility
            base_price = 20000
            prices = []
            for i in range(limit):
                # Create some random walk
                mod = (i % 10) - 5  # Oscillate between -5 and 5
                price = base_price + mod * 100  # Add/subtract $100 per step
                prices.append(price)
            
            data = []
            for i, ts in enumerate(timestamps):
                price = prices[i]
                data.append([
                    ts,  # timestamp
                    price - 10,  # open
                    price + 20,  # high
                    price - 20,  # low
                    price,  # close
                    100  # volume
                ])
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate support and resistance
            df['support'] = min(prices) - 50
            df['resis'] = max(prices) + 50
        
        # Limit to requested number of candles
        if len(df) > limit:
            df = df.iloc[-limit:]
            
        # Calculate SMA
        df[f'sma_{sma}'] = df['close'].rolling(window=sma).mean()
        
        # Ensure support and resistance are in DataFrame
        if 'support' not in df.columns:
            df['support'] = df['close'].min()
        if 'resis' not in df.columns:
            df['resis'] = df['close'].max()
        
        # Fill any NaN values in the support and resistance columns
        df['support'] = df['support'].fillna(df['close'].min())
        df['resis'] = df['resis'].fillna(df['close'].max())
        
        logger.info(f'Successfully built DataFrame with {len(df)} rows for {symbol}')
        
        # Cache successful result
        cache_key = f"df_sma_{symbol}_{timeframe}_{limit}_{sma}"
        cache[cache_key] = df
        cache_ttl[cache_key] = time.time()
        
        return df
        
    except Exception as e:
        logger.error(f"Error in df_sma for {symbol}: {str(e)}")
        
        # Check cache for previous result
        cache_key = f"df_sma_{symbol}_{timeframe}_{limit}_{sma}"
        if cache_key in cache:
            ttl = 3600  # Use cached data for up to 1 hour
            if time.time() - cache_ttl[cache_key] < ttl:
                logger.warning(f"Using cached DataFrame for {symbol}")
                return cache[cache_key]
        
        # Create minimal DataFrame so the bot can continue running
        df = pd.DataFrame({
            'timestamp': [(datetime.now() - timedelta(minutes=i*15)).strftime('%Y-%m-%d %H:%M:%S') for i in range(limit)],
            'open': [20000] * limit,
            'high': [20100] * limit,
            'low': [19900] * limit,
            'close': [20050] * limit,
            'volume': [100] * limit,
            'support': [19500] * limit,
            'resis': [20500] * limit,
            f'sma_{sma}': [20025] * limit
        })
        
        return df

def calculate_vwap_with_symbol(symbol):
    """
    Calculate VWAP (Volume-Weighted Average Price) for a symbol.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        tuple: (DataFrame with VWAP, latest VWAP value) or (empty DataFrame, None) on error
    """
    try:
        # Fetch data
        snapshot_data = get_ohlcv2(symbol, '15m', 300)
        if snapshot_data is None:
            logger.error(f"No data for VWAP calculation for {symbol}")
            return pd.DataFrame(), None
            
        df = process_data_to_df(snapshot_data)
        if df.empty:
            logger.error(f"Empty DataFrame for {symbol}")
            return df, None

        # Convert the 'timestamp' column to datetime and set as the index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Ensure all columns used for VWAP calculation are numeric
        numeric_columns = ['high', 'low', 'close', 'volume']
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        df.dropna(subset=numeric_columns, inplace=True)

        # Ensure the DataFrame is ordered by datetime
        df.sort_index(inplace=True)

        # Calculate VWAP
        df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])

        # Get the latest VWAP value
        latest_vwap = df['VWAP'].iloc[-1]
        logger.info(f'VWAP for {symbol}: {latest_vwap}')

        return df, latest_vwap
    except Exception as e:
        logger.error(f"Error calculating VWAP for {symbol}: {str(e)}")
        return pd.DataFrame(), None

def open_positions(symbol):
    """
    Get open positions information for a symbol.
    
    This is maintained for backward compatibility with 11_bot2.py.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        tuple: (positions_list, in_position, position_size, is_long, index_pos)
    """
    try:
        # Create account using global secret key
        account = eth_account.Account.from_key(secret_key)
        
        # For exchange-specific symbols like 'BTC/USD:BTC' on Phemex, extract the base symbol
        actual_symbol = symbol
        if ':' in symbol:
            # Handle symbols like 'BTC/USD:BTC' - extract just the base
            parts = symbol.split('/')
            if len(parts) > 0 and ':' in parts[0]:
                actual_symbol = parts[0].split(':')[0]
            elif len(parts) > 0:
                actual_symbol = parts[0]
        
        logger.info(f"Converting exchange symbol {symbol} to {actual_symbol} for API call")
        
        # Use get_position to get the data
        positions, in_pos, size, pos_sym, entry_px, pnl_perc, long = get_position(actual_symbol, account)
        
        # Format the return to match what the bot expects
        index_pos = 3  # This value seems hardcoded in the bot
        
        logger.info(f"open_positions result for {symbol}: in_pos={in_pos}, size={size}")
        return positions, in_pos, size, long, index_pos
    except Exception as e:
        logger.error(f"Error in open_positions for {symbol}: {str(e)}")
        return [], False, 0, False, 3

def sleep_on_close(symbol, pause_time):
    """
    Check if a position was recently closed and pause trading if so.
    
    Args:
        symbol (str): Trading symbol
        pause_time (int): Minutes to pause
        
    Returns:
        bool: True if paused, False otherwise
    """
    try:
        # This is a simple implementation that always returns False
        # In a real implementation, we would need to track position closes
        # and return True for a period after a close
        
        # For now, just log the call and return False
        logger.info(f"Sleep on close called for {symbol} with pause time {pause_time} minutes")
        
        # To implement a real pause, we would need to:
        # 1. Track when positions are closed in a global or persistent variable
        # 2. Check if enough time has passed since the last close
        # 3. Return True if still in pause period, False otherwise
        
        # Mock implementation - always return False
        return False
    except Exception as e:
        logger.error(f"Error in sleep_on_close for {symbol}: {str(e)}")
        return False

def handle_api_response(response, operation):
    if response.status_code == 429:
        logger.warning(f"Rate limit hit during {operation}, backing off")
        time.sleep(2)  # Back off
        return None
    elif response.status_code != 200:
        logger.error(f"API error during {operation}: {response.status_code}")
        return None
    return response.json()

def validate_symbol(symbol):
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("Symbol must be a non-empty string")
    return symbol
