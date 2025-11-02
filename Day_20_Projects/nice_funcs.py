#from dontshare import key  
#key = 'klhjklhjklhjklhjk' -- this is whats in the dontshareconfig.py (private key, keep safe af)

import eth_account
import json
import time
import pandas as pd

# Fix numpy compatibility issue - Must be before pandas_ta import
import numpy
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan

import pandas_ta as ta
import requests
import ccxt
import logging
import os, sys
from datetime import datetime, timedelta
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from typing import Tuple, List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading.log')
    ]
)
logger = logging.getLogger('trading')

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
            key = d.key
        except ImportError:
            logger.error("Trading key not found! Set TRADING_KEY environment variable or create dontshare.py with 'key' variable")
            raise ValueError("Trading key not found. Please set TRADING_KEY environment variable.")
    
    return key

# Get the key once at module level
try:
    key = get_trading_key()
except Exception as e:
    logger.warning(f"Warning: {e}. Some functions requiring authentication may not work.")
    key = None

# Constants
DEFAULT_SYMBOL = 'SOL'
DEFAULT_TIMEFRAME = '15m'
DEFAULT_MAX_LOSS = -1
DEFAULT_TARGET = 5
DEFAULT_POSITION_SIZE = 200
DEFAULT_LEVERAGE = 10
DEFAULT_VOLUME_MULTIPLIER = 3
DEFAULT_ROUNDING = 4

# Configuration
symbol = DEFAULT_SYMBOL
timeframe = DEFAULT_TIMEFRAME
max_loss = DEFAULT_MAX_LOSS
target = DEFAULT_TARGET
pos_size = DEFAULT_POSITION_SIZE
leverage = DEFAULT_LEVERAGE
vol_multiplier = DEFAULT_VOLUME_MULTIPLIER
rounding = DEFAULT_ROUNDING

cb_symbol = symbol + '/USDT' #BTC/USD

logger.info('Nice trading functions module initialized')

class TradingConfig:
    """
    Trading configuration class to centralize trading parameters.
    """
    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        max_loss: float = DEFAULT_MAX_LOSS,
        target: float = DEFAULT_TARGET,
        position_size: float = DEFAULT_POSITION_SIZE,
        leverage: int = DEFAULT_LEVERAGE,
        volume_multiplier: float = DEFAULT_VOLUME_MULTIPLIER,
        rounding: int = DEFAULT_ROUNDING
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_loss = max_loss
        self.target = target
        self.position_size = position_size
        self.leverage = leverage
        self.volume_multiplier = volume_multiplier
        self.rounding = rounding
        self.cb_symbol = f"{symbol}/USDT"
        
    def __str__(self) -> str:
        """Return string representation of config for logging."""
        return (
            f"TradingConfig(symbol={self.symbol}, timeframe={self.timeframe}, "
            f"max_loss={self.max_loss}, target={self.target}, "
            f"position_size={self.position_size}, leverage={self.leverage})"
        )
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key == 'symbol':
                    self.cb_symbol = f"{value}/USDT"
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        logger.info(f"Updated configuration: {self}")

# Default configuration instance
config = TradingConfig()

def ask_bid(symbol: str) -> Tuple[float, float, List]:
    """
    Fetches the current ask and bid prices for a given trading symbol from HyperLiquid API.
    
    Args:
        symbol: The trading symbol to fetch prices for (e.g., 'BTC', 'SOL')
        
    Returns:
        A tuple containing (ask_price, bid_price, level2_data)
        
    Raises:
        RequestException: If the API request fails
        ValueError: If the response data is invalid
    """
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        'type': 'l2Book',
        'coin': symbol
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for HTTP errors
        l2_data = response.json()
        
        if 'levels' not in l2_data:
            raise ValueError(f"Invalid response format: {l2_data}")
            
        l2_data = l2_data['levels']
        
        # Get bid and ask
        bid = float(l2_data[0][0]['px'])
        ask = float(l2_data[1][0]['px'])

        return ask, bid, l2_data
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        raise
    except (ValueError, KeyError, IndexError) as e:
        print(f"Error parsing response data: {e}")
        raise ValueError(f"Failed to parse orderbook data: {e}")

def get_sz_px_decimals(symbol: str) -> Tuple[int, int]:
    """
    Determines the size and price decimals for a given trading symbol.
    
    This is required to properly format order sizes and prices according to 
    exchange requirements. If size isn't properly formatted, you'll get an
    "Invalid order size" error.
    
    Args:
        symbol: The trading symbol to check
        
    Returns:
        Tuple containing (size_decimals, price_decimals)
        
    Raises:
        RequestException: If the API request fails
        ValueError: If the symbol is not found or response is invalid
    """
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    try:
        # Get symbol metadata for size decimals
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        data = response.json()
        symbols = data.get('universe', [])
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        
        if not symbol_info:
            raise ValueError(f'Symbol {symbol} not found')
            
        sz_decimals = symbol_info['szDecimals']
        
        # Get current ask price to determine price decimals
        try:
            ask = ask_bid(symbol)[0]
            ask_str = str(ask)
            
            # Compute the number of decimal points in the ask price
            if '.' in ask_str:
                px_decimals = len(ask_str.split('.')[1])
            else:
                px_decimals = 0
                
            return sz_decimals, px_decimals
            
        except Exception as e:
            print(f"Error getting ask price for {symbol}: {e}")
            # Fall back to size decimals if we can't get price decimals
            return sz_decimals, sz_decimals
            
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        raise
    except Exception as e:
        print(f"Error determining decimals for {symbol}: {e}")
        raise ValueError(f"Failed to get decimals for {symbol}: {e}")

def limit_order(coin: str, is_buy: bool, sz: float, limit_px: float, 
                reduce_only: bool, account) -> Dict:
    """
    Places a limit order on HyperLiquid exchange.
    
    Args:
        coin: Trading symbol (e.g., 'BTC', 'SOL')
        is_buy: True for buy orders, False for sell orders
        sz: Order size (quantity)
        limit_px: Limit price for the order
        reduce_only: Whether the order should be reduce-only
        account: The account object with trading permissions
        
    Returns:
        The order result dictionary from the exchange API
        
    Raises:
        ValueError: If the parameters are invalid
        Exception: If the order placement fails
    """
    try:
        # Initialize exchange connection
        exchange = Exchange(account, constants.MAINNET_API_URL)
        
        # Get proper decimal rounding for the symbol
        sz_decimals, _ = get_sz_px_decimals(coin)
        sz = round(sz, sz_decimals)
        
        # Input validation
        if sz <= 0:
            raise ValueError(f"Order size must be positive: {sz}")
        if limit_px <= 0:
            raise ValueError(f"Limit price must be positive: {limit_px}")
            
        # Log order details
        order_type = "BUY" if is_buy else "SELL"
        print(f"Placing limit {order_type} order for {coin}: {sz} @ {limit_px}")
        
        # Place the order
        order_result = exchange.order(
            coin, 
            is_buy, 
            sz, 
            limit_px, 
            {"limit": {"tif": "Gtc"}}, 
            reduce_only=reduce_only
        )
        
        # Log the result
        status = order_result.get('response', {}).get('data', {}).get('statuses', [{}])[0]
        print(f"Limit {order_type} order placed: {status}")
        
        return order_result
        
    except Exception as e:
        print(f"Error placing limit order for {coin}: {e}")
        raise

def adjust_leverage_size_signal(symbol, leverage, account):

        '''
        this calculates size based off what we want.
        95% of balance
        '''

        print('leverage:', leverage)

        #account: LocalAccount = eth_account.Account.from_key(key)
        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get the user state and print out leverage information for ETH
        user_state = info.user_state(account.address)
        acct_value = user_state["marginSummary"]["accountValue"]
        acct_value = float(acct_value)
        #print(acct_value)
        acct_val95 = acct_value * .95

        print(exchange.update_leverage(leverage, symbol))

        price = ask_bid(symbol)[0]

        # size == balance / price * leverage
        # INJ 6.95 ... at 10x lev... 10 INJ == $cost 6.95
        size = (acct_val95 / price) * leverage
        size = float(size)
        rounding = get_sz_px_decimals(symbol)[0]
        size = round(size, rounding)
        #print(f'this is the size we can use 95% fo acct val {size}')
    
        user_state = info.user_state(account.address)
            
        return leverage, size

def adjust_leverage_usd_size(symbol, usd_size, leverage, account):

        '''
        this calculates size based off a specific USD dollar amount
        '''

        print('leverage:', leverage)

        #account: LocalAccount = eth_account.Account.from_key(key)
        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get the user state and print out leverage information for ETH
        user_state = info.user_state(account.address)
        acct_value = user_state["marginSummary"]["accountValue"]
        acct_value = float(acct_value)

        print(exchange.update_leverage(leverage, symbol))

        price = ask_bid(symbol)[0]

        # size == balance / price * leverage
        # INJ 6.95 ... at 10x lev... 10 INJ == $cost 6.95
        size = (usd_size / price) * leverage
        size = float(size)
        rounding = get_sz_px_decimals(symbol)[0]
        size = round(size, rounding)
        print(f'this is the size of crypto we will be using {size}')
    
        user_state = info.user_state(account.address)
            
        return leverage, size

def adjust_leverage(symbol, leverage):
    """
    Adjusts the leverage for a trading symbol.
    
    Args:
        symbol: Trading symbol
        leverage: Leverage value to set
        
    Returns:
        Result of the leverage update
    """
    try:
        if key is None:
            logger.error("No trading key available. Cannot adjust leverage.")
            return False
            
        account = eth_account.Account.from_key(key)
        exchange = Exchange(account, constants.MAINNET_API_URL)
        
        logger.info(f"Setting leverage to {leverage}x for {symbol}")
        result = exchange.update_leverage(leverage, symbol)
        return result
    except Exception as e:
        logger.error(f"Failed to adjust leverage for {symbol}: {e}")
        return False

def get_ohclv(cb_symbol, timeframe, limit):

    coinbase = ccxt.kraken()

    ohlcv = coinbase.fetch_ohlcv(cb_symbol, timeframe, limit)
    #print(ohlcv)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df = df.tail(limit)

    df['support'] = df[:-2]['close'].min()
    df['resis'] = df[:-2]['close'].max()

    # Save the dataframe to a CSV file
    df.to_csv('ohlcv_data.csv', index=False)

    return df 

def get_ohlcv2(symbol: str, interval: str, lookback_days: int) -> List[Dict]:
    """
    Fetches historical OHLCV (Open, High, Low, Close, Volume) data from HyperLiquid API.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC', 'SOL')
        interval: Time interval (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
        lookback_days: Number of days of historical data to fetch
        
    Returns:
        List of candlestick data dictionaries
        
    Raises:
        RequestException: If the API request fails
    """
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

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        snapshot_data = response.json()
        return snapshot_data
    except requests.RequestException as e:
        print(f"Error fetching OHLCV data for {symbol}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error fetching OHLCV data: {e}")
        return []

def process_data_to_df(snapshot_data: List[Dict], time_period: int = 20) -> pd.DataFrame:
    """
    Converts raw candlestick data into a pandas DataFrame with additional calculations.
    
    Args:
        snapshot_data: List of candlestick dictionaries from the API
        time_period: Rolling window size for technical indicators (default: 20)
        
    Returns:
        DataFrame with processed OHLCV data and technical indicators
    """
    if not snapshot_data:
        return pd.DataFrame()
        
    try:
        # Prepare data for DataFrame
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

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        # Ensure data types are numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Calculate rolling support and resistance
        df['support'] = df['close'].rolling(window=time_period, min_periods=1).min().shift(1)
        df['resis'] = df['close'].rolling(window=time_period, min_periods=1).max().shift(1)

        return df
    except Exception as e:
        print(f"Error processing OHLCV data: {e}")
        return pd.DataFrame()

def calculate_vwap_with_symbol(symbol: str) -> Tuple[pd.DataFrame, float]:
    """
    Calculates Volume Weighted Average Price (VWAP) for a given symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC', 'SOL')
        
    Returns:
        Tuple containing (DataFrame with VWAP data, latest VWAP value)
    """
    try:
        # Fetch and process data
        snapshot_data = get_ohlcv2(symbol, '15m', 300)
        df = process_data_to_df(snapshot_data)
        
        if df.empty:
            print(f"No data available for {symbol}")
            return df, 0.0

        # Convert the 'timestamp' column to datetime and set as the index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Ensure all columns used for VWAP calculation are of numeric type
        numeric_columns = ['high', 'low', 'close', 'volume']
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        # Drop rows with NaNs
        df.dropna(subset=numeric_columns, inplace=True)

        # Ensure the DataFrame is ordered by datetime
        df.sort_index(inplace=True)

        # Calculate VWAP
        df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])

        # Get the latest VWAP value
        latest_vwap = df['VWAP'].iloc[-1] if not df.empty else 0.0

        return df, latest_vwap
        
    except Exception as e:
        print(f"Error calculating VWAP for {symbol}: {e}")
        return pd.DataFrame(), 0.0

def supply_demand_zones_hl(symbol, timeframe, limit):

    print('starting moons supply and demand zone calculations..')

    sd_df = pd.DataFrame()

    snapshot_data = get_ohlcv2(symbol, timeframe, limit)
    df = process_data_to_df(snapshot_data)

    
    supp = df.iloc[-1]['support']
    resis = df.iloc[-1]['resis']
    #print(f'this is moons support for 1h {supp_1h} this is resis: {resis_1h}')

    df['supp_lo'] = df[:-2]['low'].min()
    supp_lo = df.iloc[-1]['supp_lo']

    df['res_hi'] = df[:-2]['high'].max()
    res_hi = df.iloc[-1]['res_hi']

    print(df)


    sd_df[f'{timeframe}_dz'] = [supp_lo, supp]
    sd_df[f'{timeframe}_sz'] = [res_hi, resis]

    print('here are moons supply and demand zones')
    print(sd_df)

    return sd_df 

class PositionManager:
    """
    Manages trading positions including getting position information,
    adjusting leverage, and handling position closing.
    """
    
    def __init__(self, account):
        """
        Initialize the position manager with an account.
        
        Args:
            account: The trading account object
        """
        self.account = account
        self.exchange = Exchange(account, constants.MAINNET_API_URL)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    def get_position(self, symbol: str) -> Tuple[List, bool, float, Optional[str], float, float, Optional[bool]]:
        """
        Gets the current position information for a symbol.
        
        Args:
            symbol: The trading symbol to check
            
        Returns:
            Tuple containing:
            - positions: List of position data
            - in_position: Whether there's an active position
            - size: Position size (positive for long, negative for short)
            - pos_symbol: Symbol of the position
            - entry_px: Entry price
            - pnl_perc: PnL percentage
            - is_long: True if long, False if short, None if no position
        """
        try:
            user_state = self.info.user_state(self.account.address)
            logger.info(f"Account value: {user_state['marginSummary']['accountValue']}")
            
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
                    pnl_perc = float(position["position"]["returnOnEquity"]) * 100
                    logger.info(f"Position PnL: {pnl_perc}%")
                    break
            
            if size > 0:
                long = True 
            elif size < 0:
                long = False
                
            return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long
            
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return [], False, 0, None, 0, 0, None
    
    def get_position_and_check_max(self, symbol: str, max_positions: int) -> Tuple[List, bool, float, Optional[str], float, float, Optional[bool], int]:
        """
        Gets position info and checks if the maximum number of positions has been reached.
        If maximum positions exceeded, closes all positions.
        
        Args:
            symbol: The trading symbol to check
            max_positions: Maximum allowed open positions
            
        Returns:
            Same as get_position() plus number of current positions
        """
        try:
            user_state = self.info.user_state(self.account.address)
            logger.info(f"Account value: {user_state['marginSummary']['accountValue']}")
            
            positions = []
            open_positions = []
            
            # Check all open positions first
            for position in user_state["assetPositions"]:
                if float(position["position"]["szi"]) != 0:
                    open_positions.append(position["position"]["coin"])
            
            num_of_pos = len(open_positions)
            
            # Close positions if max exceeded
            if num_of_pos > max_positions:
                logger.warning(f"Position limit exceeded: {num_of_pos}/{max_positions}, closing positions")
                for position in open_positions:
                    self.kill_switch(position)
            else:
                logger.info(f"Position count: {num_of_pos}/{max_positions}")
            
            # Get info for the requested symbol
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
                    pnl_perc = float(position["position"]["returnOnEquity"]) * 100
                    logger.info(f"Position PnL: {pnl_perc}%")
                    break
            
            if size > 0:
                long = True 
            elif size < 0:
                long = False
                
            return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long, num_of_pos
            
        except Exception as e:
            logger.error(f"Error checking positions for {symbol}: {e}")
            return [], False, 0, None, 0, 0, None, 0
    
    def adjust_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Adjusts the leverage for a trading symbol.
        
        Args:
            symbol: Trading symbol
            leverage: Leverage value to set
            
        Returns:
            Success status (True/False)
        """
        try:
            logger.info(f"Setting leverage to {leverage}x for {symbol}")
            result = self.exchange.update_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to adjust leverage for {symbol}: {e}")
            return False
    
    def adjust_leverage_and_calculate_size(self, symbol: str, leverage: int) -> Tuple[int, float]:
        """
        Adjusts leverage and calculates appropriate position size based on account value.
        
        Args:
            symbol: Trading symbol
            leverage: Leverage value to set
            
        Returns:
            Tuple of (leverage, position_size)
        """
        try:
            logger.info(f"Setting leverage to {leverage}x for {symbol}")
            self.exchange.update_leverage(leverage, symbol)
            
            user_state = self.info.user_state(self.account.address)
            acct_value = float(user_state["marginSummary"]["accountValue"])
            acct_val95 = acct_value * 0.95  # Use 95% of account value
            
            price = ask_bid(symbol)[0]
            
            # Calculate size based on account value, price, and leverage
            size = (acct_val95 / price) * leverage
            rounding_val = get_sz_px_decimals(symbol)[0]
            size = round(size, rounding_val)
            
            logger.info(f"Calculated position size: {size} {symbol} at {leverage}x leverage")
            return leverage, size
            
        except Exception as e:
            logger.error(f"Error adjusting leverage and calculating size: {e}")
            return leverage, 0
    
    def adjust_leverage_usd_size(self, symbol: str, usd_size: float, leverage: int) -> Tuple[int, float]:
        """
        Adjusts leverage and calculates position size based on a specific USD amount.
        
        Args:
            symbol: Trading symbol
            usd_size: Position size in USD
            leverage: Leverage value to set
            
        Returns:
            Tuple of (leverage, position_size)
        """
        try:
            logger.info(f"Setting leverage to {leverage}x for {symbol} with ${usd_size} position")
            self.exchange.update_leverage(leverage, symbol)
            
            price = ask_bid(symbol)[0]
            
            # Calculate size based on USD amount, price, and leverage
            size = (usd_size / price) * leverage
            rounding_val = get_sz_px_decimals(symbol)[0]
            size = round(size, rounding_val)
            
            logger.info(f"Calculated position size: {size} {symbol} at {leverage}x leverage")
            return leverage, size
            
        except Exception as e:
            logger.error(f"Error adjusting leverage for USD size: {e}")
            return leverage, 0
    
    def kill_switch(self, symbol: str) -> bool:
        """
        Emergency position closer - closes all positions for a symbol.
        
        Args:
            symbol: Symbol to close positions for
            
        Returns:
            Success status (True/False)
        """
        try:
            positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = self.get_position(symbol)
            
            if not in_pos:
                logger.info(f"No position to close for {symbol}")
                return True
                
            logger.warning(f"KILL SWITCH activated for {symbol}")
            
            while in_pos:
                # Cancel any open orders
                self.cancel_all_orders()
                
                # Get bid/ask
                ask, bid, _ = ask_bid(pos_sym)
                pos_size = abs(pos_size)
                
                # Close position with market order
                if long:
                    limit_order(pos_sym, False, pos_size, bid, True, self.account)
                    logger.info(f"Kill switch - SELL TO CLOSE submitted for {pos_sym}")
                else:
                    limit_order(pos_sym, True, pos_size, ask, True, self.account)
                    logger.info(f"Kill switch - BUY TO CLOSE submitted for {pos_sym}")
                
                time.sleep(5)  # Wait for order to execute
                positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = self.get_position(symbol)
            
            logger.info(f"Position successfully closed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error in kill switch for {symbol}: {e}")
            return False
    
    def pnl_close(self, symbol: str, target: float, max_loss: float) -> None:
        """
        Closes positions based on profit/loss targets.
        
        Args:
            symbol: Trading symbol
            target: Target profit percentage to close at
            max_loss: Maximum loss percentage to close at
        """
        try:
            logger.info(f"Checking PnL for {symbol} (target: {target}%, max loss: {max_loss}%)")
            positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = self.get_position(symbol)
            
            if not in_pos:
                logger.info(f"No position to check for {symbol}")
                return
                
            if pnl_perc > target:
                logger.info(f"Target reached: {pnl_perc}% > {target}% for {symbol}, closing position")
                self.kill_switch(pos_sym)
            elif pnl_perc <= max_loss:
                logger.warning(f"Stop loss hit: {pnl_perc}% <= {max_loss}% for {symbol}, closing position")
                self.kill_switch(pos_sym)
            else:
                logger.info(f"Position within parameters: {pnl_perc}% (target: {target}%, max loss: {max_loss}%)")
        
        except Exception as e:
            logger.error(f"Error in PnL close for {symbol}: {e}")
    
    def cancel_all_orders(self) -> bool:
        """
        Cancels all open orders.
        
        Returns:
            Success status (True/False)
        """
        try:
            open_orders = self.info.open_orders(self.account.address)
            logger.info(f"Cancelling {len(open_orders)} open orders")
            
            for open_order in open_orders:
                self.exchange.cancel(open_order['coin'], open_order['oid'])
                
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False
    
    def cancel_symbol_orders(self, symbol: str) -> bool:
        """
        Cancels all open orders for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Success status (True/False)
        """
        try:
            open_orders = self.info.open_orders(self.account.address)
            symbol_orders = [order for order in open_orders if order['coin'] == symbol]
            
            logger.info(f"Cancelling {len(symbol_orders)} open orders for {symbol}")
            
            for open_order in symbol_orders:
                self.exchange.cancel(open_order['coin'], open_order['oid'])
                
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling orders for {symbol}: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """
        Closes all open positions across all symbols.
        
        Returns:
            Success status (True/False)
        """
        try:
            user_state = self.info.user_state(self.account.address)
            open_positions = []
            
            # Cancel all orders first
            self.cancel_all_orders()
            
            # Find all open positions
            for position in user_state["assetPositions"]:
                if float(position["position"]["szi"]) != 0:
                    open_positions.append(position["position"]["coin"])
            
            logger.warning(f"Closing all {len(open_positions)} open positions")
            
            # Close each position
            for symbol in open_positions:
                self.kill_switch(symbol)
                
            return True
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

def get_open_interest():
    
    # read in this csv and get the most recent which is the 2nd column
    open_interest = pd.read_csv('/Users/md/Dropbox/dev/github/liquidation-trading-bot/data/oi_total.csv')
    oi = int(open_interest.iloc[-1, 1]) 

    return oi

def get_liquidations():
    # Read in the CSV file
    liquidations_df = pd.read_csv('/Users/md/Dropbox/dev/github/liquidation-trading-bot/data/recent_liqs.csv')

    # Convert DataFrame to JSON
    liquidations_json = {}

    for index, row in liquidations_df.iterrows():
        interval = row['Interval']
        liquidations_json[interval] = {
            'Total Liquidations': row['Total Liquidations'],
            'Long Liquidations': row['Long Liquidations'],
            'Short Liquidations': row['Short Liquidations']
        }

    return json.dumps(liquidations_json, indent=4)


def get_funding_rate():
    # Read in the CSV file
    funding_df = pd.read_csv('/Users/md/Dropbox/dev/github/liquidation-trading-bot/data/funding.csv')

    # Convert DataFrame to JSON
    funding_json = {}

    for index, row in funding_df.iterrows():
        symbol = row['symbol']
        funding_json[symbol] = {
            'Funding Rate': row['funding_rate'],
            'Yearly Funding Rate': row['yearly_funding_rate']
        }

    return json.dumps(funding_json, indent=4)

class MarketAnalyzer:
    """
    Class for analyzing market data and conditions to inform trading decisions.
    """
    
    def __init__(self, config: TradingConfig = None):
        """
        Initialize the market analyzer with configuration.
        
        Args:
            config: Trading configuration object
        """
        self.config = config or TradingConfig()
    
    def get_market_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        """
        Get market data for a symbol and convert to DataFrame.
        
        Args:
            symbol: Trading symbol
            interval: Time interval for data
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            snapshot_data = get_ohlcv2(symbol, interval, lookback_days)
            df = process_data_to_df(snapshot_data)
            return df
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def should_quote_orders(self) -> bool:
        """
        Determines if market conditions are suitable for quoting orders.
        
        Returns:
            Boolean indicating whether to quote orders
        """
        try:
            logger.info("Checking if we should quote orders")
            
            # Get BTC price range over last 4 hours to determine volatility
            snapshot_data = get_ohlcv2('BTC', '1m', 500)
            df = process_data_to_df(snapshot_data)
            
            if df.empty:
                logger.warning("No data available to determine quote decision")
                return False
                
            price_range = self.calculate_range(df, 240)  # 240 mins = 4 hours
            logger.info(f"BTC price range in last 4 hours: ${price_range}")
            
            # Don't quote orders if market is too volatile
            if price_range > 500:
                logger.warning("Market too volatile for quoting orders")
                return False
            else:
                logger.info("Market conditions acceptable for quoting orders")
                return True
                
        except Exception as e:
            logger.error(f"Error determining if we should quote orders: {e}")
            return False
    
    def calculate_range(self, df: pd.DataFrame, window: int) -> float:
        """
        Calculate the range between highest high and lowest low for recent periods.
        
        Args:
            df: DataFrame with high and low data
            window: Number of periods to analyze
            
        Returns:
            Price range as a float
        """
        try:
            # Ensure window isn't larger than data
            window = min(window, len(df))
            
            # Get recent data
            recent_df = df[-window:].copy()
            
            # Ensure numeric columns
            recent_df['high'] = pd.to_numeric(recent_df['high'], errors='coerce')
            recent_df['low'] = pd.to_numeric(recent_df['low'], errors='coerce')
            
            # Drop any rows with NaN values
            recent_df.dropna(subset=['high', 'low'], inplace=True)
            
            if recent_df.empty:
                return 0.0
                
            # Calculate range
            highest_high = recent_df['high'].max()
            lowest_low = recent_df['low'].min()
            price_range = highest_high - lowest_low
            
            return price_range
            
        except Exception as e:
            logger.error(f"Error calculating price range: {e}")
            return 0.0
    
    def volume_spike(self, df: pd.DataFrame) -> bool:
        """
        Detects if there's a volume spike in recent data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Boolean indicating if there's a volume spike with downtrend
        """
        try:
            # Calculate volume moving average
            df['MA_Volume'] = df['volume'].rolling(window=20).mean()
            
            # Calculate price moving average
            df['MA_Close'] = df['close'].rolling(window=20).mean()
            
            if df.empty or df['MA_Volume'].isna().all() or df['MA_Close'].isna().all():
                return False
                
            latest_data = df.iloc[-1]
            
            # Check for volume spike with price downtrend
            volume_spike_and_price_downtrend = (
                latest_data['volume'] > self.config.volume_multiplier * latest_data['MA_Volume'] and 
                latest_data['MA_Close'] > latest_data['close']
            )
            
            return volume_spike_and_price_downtrend
            
        except Exception as e:
            logger.error(f"Error detecting volume spike: {e}")
            return False
    
    def linear_regression_bollinger(self, df: pd.DataFrame, 
                                   bb_length: int = 20, 
                                   bb_std_dev: int = 2, 
                                   lrc_length: int = 20, 
                                   proximity_threshold: float = 0.02) -> Tuple[pd.DataFrame, bool, bool, bool]:
        """
        Determines when to quote buy, sell, or both orders based on Bollinger Bands and Linear Regression Channel.
        
        Args:
            df: DataFrame with OHLCV data
            bb_length: Period for Bollinger Bands SMA calculation
            bb_std_dev: Standard deviations for Bollinger Bands
            lrc_length: Period for Linear Regression Channel calculation
            proximity_threshold: Proximity to LRC Middle Line as percentage of channel width
            
        Returns:
            Tuple of (DataFrame, quote_buy_orders, quote_sell_orders, quote_both_orders)
        """
        try:
            # Calculate Bollinger Bands
            df, tight, wide = self.calculate_bollinger_bands(df, length=bb_length, std_dev=bb_std_dev)
            
            # Calculate Linear Regression Channel and determine quoting conditions
            df, lrc_quote_buy, lrc_quote_sell, lrc_quote_both = self.calculate_linear_regression_channel(
                df, length=lrc_length, proximity_threshold=proximity_threshold
            )
            
            # Integrate decisions
            quote_buy_orders = lrc_quote_buy and not wide  # Buy in uptrend when not wide
            quote_sell_orders = lrc_quote_sell and not wide  # Sell in downtrend when not wide
            quote_both_orders = lrc_quote_both or tight  # Both in sideways market
            
            return df, quote_buy_orders, quote_sell_orders, quote_both_orders
            
        except Exception as e:
            logger.error(f"Error in linear regression Bollinger analysis: {e}")
            return df, False, False, False
    
    def calculate_linear_regression_channel(self, df: pd.DataFrame, 
                                           length: int = 20, 
                                           proximity_threshold: float = 0.02) -> Tuple[pd.DataFrame, bool, bool, bool]:
        """
        Calculate Linear Regression Channel and determine quoting conditions.
        
        Args:
            df: DataFrame with OHLCV data
            length: Period for calculation
            proximity_threshold: Proximity threshold
            
        Returns:
            Tuple of (DataFrame, quote_buy_orders, quote_sell_orders, quote_both_orders)
        """
        try:
            # Ensure 'close' is numeric
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            # Calculate Linear Regression
            linreg_values = ta.linreg(df['close'], length=length)
            df['LRCM_'] = linreg_values  # Middle Line
            
            # Calculate channel width based on standard deviation
            channel_width = df['close'].rolling(window=length).std()
            
            # Add channel lines
            df['LRCT_'] = df['LRCM_'] + channel_width * 2  # Top Channel
            df['LRCB_'] = df['LRCM_'] - channel_width * 2  # Bottom Channel
            
            # Determine the slope of the channel
            if len(df) >= length:
                slope = (df['LRCM_'].iloc[-1] - df['LRCM_'].iloc[-length]) / length
            else:
                slope = 0
            
            # Determine conditions for quoting orders
            quote_buy_orders = slope > 0 and df['close'].iloc[-1] < df['LRCT_'].iloc[-1]
            quote_sell_orders = slope < 0 and df['close'].iloc[-1] > df['LRCB_'].iloc[-1]
            
            # Check if price is near the middle line
            proximity_to_middle = abs(df['close'].iloc[-1] - df['LRCM_'].iloc[-1])
            is_near_middle = proximity_to_middle <= (channel_width.iloc[-1] * 2 * proximity_threshold)
            
            # Quote both if near middle
            quote_both_orders = is_near_middle
            
            return df, quote_buy_orders, quote_sell_orders, quote_both_orders
            
        except Exception as e:
            logger.error(f"Error calculating linear regression channel: {e}")
            return df, False, False, False
    
    def supply_demand_zones(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Calculates supply and demand zones.
        
        Args:
            symbol: Trading symbol
            timeframe: Time interval
            limit: Number of periods to analyze
            
        Returns:
            DataFrame with supply and demand zones
        """
        try:
            logger.info(f"Calculating supply and demand zones for {symbol}")
            sd_df = pd.DataFrame()
            
            snapshot_data = get_ohlcv2(symbol, timeframe, limit)
            df = process_data_to_df(snapshot_data)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return sd_df
                
            supp = df.iloc[-1]['support']
            resis = df.iloc[-1]['resis']
            
            df['supp_lo'] = df[:-2]['low'].min()
            supp_lo = df.iloc[-1]['supp_lo']
            
            df['res_hi'] = df[:-2]['high'].max()
            res_hi = df.iloc[-1]['res_hi']
            
            sd_df[f'{timeframe}_dz'] = [supp_lo, supp]
            sd_df[f'{timeframe}_sz'] = [res_hi, resis]
            
            logger.info(f"Supply and demand zones for {symbol}: {sd_df}")
            return sd_df
            
        except Exception as e:
            logger.error(f"Error calculating supply and demand zones: {e}")
            return pd.DataFrame()
            
    def calculate_bollinger_bands(self, df: pd.DataFrame, length: int = 20, std_dev: int = 2) -> Tuple[pd.DataFrame, bool, bool]:
        """
        Calculate Bollinger Bands for a given DataFrame and classify when the bands are tight vs wide.

        Args:
            df: DataFrame with a 'close' column
            length: Period for the simple moving average (default: 20)
            std_dev: Number of standard deviations for the bands (default: 2)

        Returns:
            Tuple containing (DataFrame with Bollinger Bands, is_tight, is_wide)
        """
        try:
            # Ensure 'close' is numeric
            df['close'] = pd.to_numeric(df['close'], errors='coerce')

            # Calculate Bollinger Bands
            bollinger_bands = ta.bbands(df['close'], length=length, std=std_dev)
            
            if bollinger_bands is None or bollinger_bands.empty:
                logger.warning("Failed to calculate Bollinger Bands")
                return df, False, False

            # Extract the main Bollinger Bands columns
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
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return df, False, False


