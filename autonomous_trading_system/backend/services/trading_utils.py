import time
import logging
import requests
import asyncio
import pandas as pd
import pandas_ta as pta
import numpy as np
import os
import sys
from core.config import get_exchange_config
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from datetime import datetime, timedelta
from typing import List, Tuple, Any, Dict, Callable

# Fix numpy compatibility issue - Must be before pandas_ta import
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

logger = logging.getLogger("trading_utils")

# HTTP helper constants
API_URL = constants.MAINNET_API_URL
API_HEADERS = {'Content-Type': 'application/json'}
DEFAULT_TIMEOUT = 10  # seconds
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds between retries

# Default configuration constants
DEFAULT_SYMBOL = 'SOL'
DEFAULT_TIMEFRAME = '15m'
DEFAULT_MAX_LOSS = -1
DEFAULT_TARGET = 5
DEFAULT_POSITION_SIZE = 200
DEFAULT_LEVERAGE = 10
DEFAULT_VOLUME_MULTIPLIER = 3
DEFAULT_ROUNDING = 4

class TradingConfig:
    """
    Trading configuration class to centralize trading parameters.
    Enhanced from nice_funcs.py for better parameter management.
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

def get_trading_key():
    """Get the trading key from environment variable or config file."""
    # Try to get from environment variable (more secure)
    key = os.environ.get('TRADING_KEY') or os.environ.get('HYPERLIQUID_PRIVATE_KEY')
    
    # If not in environment, try to import from a config file
    if not key:
        try:
            # Add project root to path for imports
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Try to import from day projects if available
            try:
                from Day_4_Projects import dontshare as d
                key = d.key
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"Could not load key from config file: {e}")
    
    if not key:
        logger.error("Trading key not found! Set TRADING_KEY or HYPERLIQUID_PRIVATE_KEY environment variable")
    
    return key

def ask_bid(symbol: str) -> Tuple[float, float, Any]:
    """
    Get ask, bid, and full L2 order book data from Hyperliquid REST.
    Enhanced with better error handling from nice_funcs.py.
    """
    cfg = get_exchange_config("hyperliquid")
    url = f"{cfg['base_url']}/info"
    headers = {'Content-Type': 'application/json'}
    payload = {'type': 'l2Book', 'coin': symbol}
    
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        if 'levels' not in data:
            raise ValueError(f"Invalid response format: {data}")
            
        levels = data.get('levels', data)
        bid = float(levels[0][0]['px'])
        ask = float(levels[1][0]['px'])
        return ask, bid, levels
        
    except requests.RequestException as e:
        logger.error(f"API request failed for {symbol}: {e}")
        raise
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Error parsing response data for {symbol}: {e}")
        raise ValueError(f"Failed to parse orderbook data: {e}")

def get_sz_px_decimals(coin: str) -> Tuple[int, int]:
    """
    Return size and price decimal precision for a given coin via Hyperliquid metadata.
    Enhanced with better error handling from nice_funcs.py.
    """
    cfg = get_exchange_config("hyperliquid")
    url = f"{cfg['base_url']}/info"
    headers = {'Content-Type': 'application/json'}
    payload = {'type': 'meta'}
    
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        universe = resp.json().get('universe', [])
        symbol_info = next((s for s in universe if s['name'] == coin), None)
        
        if not symbol_info:
            raise ValueError(f"Symbol {coin} not found in metadata")
            
        sz_decimals = symbol_info['szDecimals']
        
        # Get current ask price to determine price decimals
        try:
            ask = ask_bid(coin)[0]
            ask_str = str(ask)
            px_decimals = len(ask_str.split('.')[1]) if '.' in ask_str else 0
            return sz_decimals, px_decimals
        except Exception as e:
            logger.warning(f"Error getting ask price for {coin}: {e}")
            # Fall back to size decimals if we can't get price decimals
            return sz_decimals, sz_decimals
            
    except requests.RequestException as e:
        logger.error(f"API request failed for {coin}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error determining decimals for {coin}: {e}")
        raise ValueError(f"Failed to get decimals for {coin}: {e}")

def limit_order(coin: str, is_buy: bool, sz: float, limit_px: float, reduce_only: bool, account) -> Dict[str, Any]:
    """
    Place a limit order on Hyperliquid.
    Enhanced with better validation and logging from nice_funcs.py.
    """
    try:
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
        logger.info(f"Placing limit {order_type} order for {coin}: {sz} @ {limit_px}")
        
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
        logger.info(f"Limit {order_type} order placed: {status}")
        
        return order_result
        
    except Exception as e:
        logger.error(f"Error placing limit order for {coin}: {e}")
        raise

def acct_bal(account) -> float:
    """Get account balance via Hyperliquid Info API"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    state = info.user_state(account.address)
    val = float(state['marginSummary']['accountValue'])
    logger.info(f"Account value: {val}")
    return val

def adjust_leverage_size_signal(symbol: str, leverage: float, account) -> Tuple[float, float]:
    """
    Adjust leverage and calculate position size based on account value.
    Enhanced from nice_funcs.py with 95% account usage.
    """
    logger.info(f'Setting leverage to {leverage}x for {symbol}')
    
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    # Get the user state and account value
    state = info.user_state(account.address)
    acct_value = float(state['marginSummary']['accountValue'])
    acct_val95 = acct_value * 0.95  # Use 95% of account value
    
    # Update leverage
    result = exchange.update_leverage(leverage, symbol)
    logger.info(f"Leverage update result: {result}")
    
    # Calculate position size
    ask, _, _ = ask_bid(symbol)
    sz = (acct_val95 / ask) * leverage
    
    # Round to proper decimals
    rounding = get_sz_px_decimals(symbol)[0]
    sz = round(sz, rounding)
    
    logger.info(f'Calculated position size: {sz} {symbol} with 95% account usage')
    return leverage, sz

def adjust_leverage_usd_size(symbol: str, usd_size: float, leverage: float, account) -> Tuple[float, float]:
    """
    Calculate position size based on specific USD dollar amount.
    New function from nice_funcs.py.
    """
    logger.info(f'Setting leverage to {leverage}x for {symbol} with ${usd_size} position')
    
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    # Get account value for validation
    state = info.user_state(account.address)
    acct_value = float(state['marginSummary']['accountValue'])
    
    # Update leverage
    result = exchange.update_leverage(leverage, symbol)
    logger.info(f"Leverage update result: {result}")
    
    # Calculate size based on USD amount
    ask, _, _ = ask_bid(symbol)
    sz = (usd_size / ask) * leverage
    
    # Round to proper decimals
    rounding = get_sz_px_decimals(symbol)[0]
    sz = round(sz, rounding)
    
    logger.info(f'Calculated position size: {sz} {symbol} for ${usd_size} at {leverage}x leverage')
    return leverage, sz

def get_position(symbol: str, account) -> Tuple[Any, bool, float, Any, float, float, Any]:
    """
    Get current position info via Hyperliquid Info API.
    Enhanced with better logging from nice_funcs.py.
    """
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    state = info.user_state(account.address)
    logger.info(f"Account value: {state['marginSummary']['accountValue']}")
    
    positions = []
    in_pos = False
    size = 0.0
    entry_px = 0.0
    pnl_perc = 0.0
    long = None
    
    for pos in state['assetPositions']:
        p = pos['position']
        if p['coin'] == symbol and float(p['szi']) != 0:
            positions.append(p)
            in_pos = True
            size = float(p['szi'])
            entry_px = float(p['entryPx'])
            pnl_perc = float(p['returnOnEquity']) * 100
            long = size > 0
            logger.info(f"Position PnL for {symbol}: {pnl_perc}%")
            break
            
    logger.info(f"Position info for {symbol}: in_pos={in_pos}, size={size}, entry={entry_px}, pnl={pnl_perc}%")
    return positions, in_pos, size, symbol, entry_px, pnl_perc, long

def cancel_all_orders(account):
    """Cancel all open orders via Hyperliquid"""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    orders = info.open_orders(account.address)
    if not orders:
        logger.info("No open orders to cancel")
        return
    logger.info(f"Cancelling {len(orders)} orders")
    for o in orders:
        try:
            exchange.cancel(o['coin'], o['oid'])
        except Exception as e:
            logger.error(f"Error cancelling order {o['oid']}: {e}")

def kill_switch(symbol: str, account):
    """
    Emergency function to close a position via Hyperliquid.
    Enhanced with better error handling from nice_funcs.py.
    """
    positions, in_pos, size, sym, entry_px, pnl, long = get_position(symbol, account)
    if not in_pos:
        logger.info(f"No position to close for {symbol}")
        return
        
    logger.warning(f"KILL SWITCH activated for {symbol}")
    
    while in_pos:
        # Cancel any open orders first
        cancel_all_orders(account)
        
        # Get current bid/ask
        ask, bid, _ = ask_bid(sym)
        pos_size = abs(size)
        
        # Close position with market order
        if long:
            limit_order(sym, False, pos_size, bid, True, account)
            logger.info(f"Kill switch - SELL TO CLOSE submitted for {sym}")
        else:
            limit_order(sym, True, pos_size, ask, True, account)
            logger.info(f"Kill switch - BUY TO CLOSE submitted for {sym}")
        
        time.sleep(5)  # Wait for order to execute
        positions, in_pos, size, sym, entry_px, pnl, long = get_position(symbol, account)
    
    logger.info(f"Position successfully closed for {symbol}")

def pnl_close(symbol: str, target: float, max_loss: float, account) -> bool:
    """
    Monitor PnL and close position when thresholds are hit.
    Enhanced with better logging from nice_funcs.py.
    """
    logger.info(f"Checking PnL for {symbol} (target: {target}%, max loss: {max_loss}%)")
    positions, in_pos, size, sym, entry_px, pnl_perc, long = get_position(symbol, account)
    
    if not in_pos:
        logger.info(f"No position to check for {symbol}")
        return False
        
    if pnl_perc >= target:
        logger.info(f"Target reached: {pnl_perc}% >= {target}% for {symbol}, closing position")
        kill_switch(sym, account)
        return True
    elif pnl_perc <= max_loss:
        logger.warning(f"Stop loss hit: {pnl_perc}% <= {max_loss}% for {symbol}, closing position")
        kill_switch(sym, account)
        return True
    else:
        logger.info(f"Position within parameters: {pnl_perc}% (target: {target}%, max loss: {max_loss}%)")
        return False

def get_position_andmaxpos(symbol: str, account, max_positions: int) -> Tuple[Any, bool, float, Any, float, float, Any, int]:
    """
    Get position info and enforce maximum number of positions.
    Enhanced with better position management from nice_funcs.py.
    """
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    state = info.user_state(account.address)
    
    # Check all open positions first
    open_syms = []
    for position in state["assetPositions"]:
        if float(position["position"]["szi"]) != 0:
            open_syms.append(position["position"]["coin"])
    
    num = len(open_syms)
    
    # Close positions if max exceeded
    if num > max_positions:
        logger.warning(f"Position limit exceeded: {num}/{max_positions}, closing positions")
        for sym in open_syms:
            kill_switch(sym, account)
        num = 0
        return [], False, 0, None, 0.0, 0.0, None, num
    else:
        logger.info(f"Position count: {num}/{max_positions}")
    
    return get_position(symbol, account) + (num,)

def close_all_positions(account):
    """
    Close all open positions across all coins.
    Enhanced with better error handling from nice_funcs.py.
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        state = info.user_state(account.address)
        open_positions = []
        
        # Cancel all orders first
        cancel_all_orders(account)
        
        # Find all open positions
        for position in state["assetPositions"]:
            if float(position["position"]["szi"]) != 0:
                open_positions.append(position["position"]["coin"])
        
        logger.warning(f"Closing all {len(open_positions)} open positions")
        
        # Close each position
        for symbol in open_positions:
            kill_switch(symbol, account)
            
        return True
        
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        return False

def calculate_bollinger_bands(df: pd.DataFrame, length: int = 20, std_dev: int = 2) -> Tuple[pd.DataFrame, bool, bool]:
    """
    Calculate BBands and return booleans indicating if current bandwidth is tight or wide.
    Enhanced from nice_funcs.py with better band width classification.
    """
    try:
        # Ensure 'close' is numeric
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Calculate Bollinger Bands
        bb = pta.bbands(df['close'], length=length, std=std_dev)
        
        if bb is None or bb.empty:
            logger.warning("Failed to calculate Bollinger Bands")
            return df, False, False
            
        # Extract the main Bollinger Bands columns
        bb = bb.iloc[:, [0, 1, 2]]  # BBL, BBM, BBU
        bb.columns = ['BBL', 'BBM', 'BBU']
        
        # Merge with original DataFrame
        df = pd.concat([df, bb], axis=1)
        
        # Calculate Band Width
        df['BandWidth'] = df['BBU'] - df['BBL']
        
        # Need at least 5 data points to classify band width
        if len(df) < 5:
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
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return df, False, False

def process_data_to_df(snapshot_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw OHLCV dicts to DataFrame with support/resistance"""
    if not snapshot_data:
        return pd.DataFrame()
    rows = []
    for s in snapshot_data:
        ts = datetime.fromtimestamp(s['t']/1000)
        rows.append([ts, s['o'], s['h'], s['l'], s['c'], s['v']])
    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')
    if len(df) > 2:
        df['support'] = df['low'][:-2].min()
        df['resistance'] = df['high'][:-2].max()
    else:
        df['support'] = df['low'].min()
        df['resistance'] = df['high'].max()
    return df

def get_ohlcv2(symbol: str, interval: str, lookback_days: int) -> Any:
    """Fetch OHLCV data snapshot via Hyperliquid REST"""
    cfg = get_exchange_config('hyperliquid')
    url = f"{cfg['base_url']}/info"
    headers = {'Content-Type': 'application/json'}
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    payload = {
        'type':'candleSnapshot',
        'req':{
            'coin':symbol,
            'interval':interval,
            'startTime':int(start_time.timestamp()*1000),
            'endTime':int(end_time.timestamp()*1000)
        }
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

def fetch_candle_snapshot(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> Any:
    """Fetch OHLCV for a specified timeframe"""
    cfg = get_exchange_config('hyperliquid')
    url = f"{cfg['base_url']}/info"
    headers = {'Content-Type': 'application/json'}
    payload = {
        'type':'candleSnapshot',
        'req':{
            'coin':symbol,
            'interval':interval,
            'startTime':int(start_time.timestamp()*1000),
            'endTime':int(end_time.timestamp()*1000)
        }
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

def calculate_sma(prices: pd.Series, window: int) -> Any:
    """Compute latest SMA value"""
    if len(prices) < window:
        return None
    sma = prices.rolling(window).mean()
    return sma.iloc[-1]

def get_latest_sma(symbol: str, interval: str, window: int, lookback_days: int = 1) -> Any:
    """Get latest SMA via REST candle snapshot"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    data = fetch_candle_snapshot(symbol, interval, start_time, end_time)
    if not data or len(data) < window:
        return None
    series = pd.Series([float(d['c']) for d in data])
    return calculate_sma(series, window)

def supply_demand_zones_hl(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Compute supply/demand zones"""
    snap = get_ohlcv2(symbol, timeframe, limit)
    df = process_data_to_df(snap)
    if df.empty:
        return df
    supp = df['support'].iloc[-1]
    res = df['resistance'].iloc[-1]
    df_out = pd.DataFrame({f'{timeframe}_dz':[df['support'].min(), supp], f'{timeframe}_sz':[df['resistance'].max(), res]})
    return df_out

def calculate_vwap_with_symbol(symbol: str) -> Tuple[pd.DataFrame, Any]:
    """Calculate VWAP for a symbol via pandas_ta"""
    cfg = get_exchange_config('hyperliquid')
    df, latest = pd.DataFrame(), None
    try:
        snap = get_ohlcv2(symbol, '15m', 1)
        df = process_data_to_df(snap)
        df['VWAP'] = pta.vwap(df['high'], df['low'], df['close'], df['volume'])
        latest = df['VWAP'].iloc[-1]
    except Exception as e:
        logger.error(f"Error calculating VWAP for {symbol}: {e}")
    return df, latest

def df_sma(symbol: str, timeframe: str = '15m', limit: int = 100, sma_window: int = 20) -> pd.DataFrame:
    """Fetch OHLCV snapshot and return DataFrame with SMA, support, and resistance for the last 'limit' bars."""
    # Map timeframe to days
    timeframe_map = {
        '1m': 1/1440, '5m': 5/1440, '15m': 15/1440,
        '30m': 30/1440, '1h': 1/24, '4h': 4/24, '1d': 1
    }
    days = timeframe_map.get(timeframe, 15/1440)
    lookback_days = max(1, int(limit * days * 1.1))
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    # Fetch raw OHLCV data
    data = fetch_candle_snapshot(symbol, timeframe, start_time, end_time)
    df = process_data_to_df(data)
    if df.empty:
        return df
    # Limit to last 'limit' bars
    if len(df) > limit:
        df = df.iloc[-limit:]
    # Compute SMA
    df[f'sma_{sma_window}'] = df['close'].rolling(window=sma_window).mean()
    return df

def open_positions(symbol: str, account) -> Tuple[Any, bool, float, bool]:
    """Return positions list, in_position flag, size, and long flag."""
    positions, in_pos, size, _, _, _, long = get_position(symbol, account)
    return positions, in_pos, size, long

def sleep_on_close(symbol: str, pause_minutes: int) -> bool:
    """Stub for pausing after close; always returns False."""
    return False

def validate_symbol(symbol: str) -> str:
    """Ensure symbol is a non-empty string."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    return symbol

# In-memory cache for generic fetch operations
_cache: Dict[str, Any] = {}
_cache_ttl: Dict[str, float] = {}

def cached_get(key: str, ttl_seconds: int, fetch_func: Callable[[], Any]) -> Any:
    """Return cached result if fresh, otherwise call fetch_func and cache."""
    now = time.time()
    if key in _cache and now - _cache_ttl[key] < ttl_seconds:
        return _cache[key]
    result = fetch_func()
    if result is not None:
        _cache[key] = result
        _cache_ttl[key] = now
    return result

def handle_api_response(response: requests.Response, operation: str) -> Any:
    """Process HTTP response with backoff and error logging."""
    if response.status_code == 429:
        logger.warning(f"Rate limit hit during {operation}, backing off")
        time.sleep(2)
        return None
    if response.status_code != 200:
        logger.error(f"API error during {operation}: {response.status_code}")
        return None
    return response.json()

async def fetch_candle_snapshot_async(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> Any:
    """Async wrapper around fetch_candle_snapshot."""
    return await asyncio.to_thread(fetch_candle_snapshot, symbol, interval, start_time, end_time)

def make_api_request(url: str, data: Dict[str, Any], method: str = "post",
                     timeout: int = DEFAULT_TIMEOUT,
                     max_retries: int = MAX_RETRY_ATTEMPTS) -> Dict[str, Any]:
    """
    Make API request with retries and error handling.
    """
    for attempt in range(1, max_retries + 1):
        try:
            if method.lower() == "post":
                resp = requests.post(url, headers=API_HEADERS, json=data, timeout=timeout)
            else:
                resp = requests.get(url, headers=API_HEADERS, params=data, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"API request attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                logger.error(f"API request failed after {max_retries} attempts: {e}")
                raise
            time.sleep(RETRY_DELAY)

# Added helper for cancelling orders by symbol
def cancel_symbol_orders(symbol: str, account):
    """Cancel all open orders for a given symbol."""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    orders = info.open_orders(account.address)
    if not orders:
        logger.info(f"No open orders to cancel for {symbol}")
        return
    for o in orders:
        if o.get('coin') == symbol:
            try:
                exchange = Exchange(account, constants.MAINNET_API_URL)
                exchange.cancel(symbol, o['oid'])
            except Exception as e:
                logger.error(f"Error cancelling order {o.get('oid')} for {symbol}: {e}")
    logger.info(f"Cancelled orders for {symbol}")

# Added deluxe order placement function
def open_order_deluxe(symbol_info: Dict[str, Any], size: float, account):
    """Place entry, stop-loss, and take-profit orders based on symbol_info."""
    symbol = symbol_info["Symbol"]
    entry_price = symbol_info["Entry Price"]
    stop_loss = symbol_info["Stop Loss"]
    take_profit = symbol_info["Take Profit"]
    exchange = Exchange(account, constants.MAINNET_API_URL)
    # Determine decimal rounding for price
    time.sleep(RETRY_DELAY)

class PositionManager:
    """
    Manages trading positions including getting position information,
    adjusting leverage, and handling position closing.
    Enhanced from nice_funcs.py for comprehensive position management.
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
    
    def get_position(self, symbol: str) -> Tuple[List, bool, float, str, float, float, bool]:
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
            pos_sym = symbol
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
            return [], False, 0, symbol, 0, 0, None
    
    def get_position_and_check_max(self, symbol: str, max_positions: int) -> Tuple[List, bool, float, str, float, float, bool, int]:
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
            pos_sym = symbol
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
            return [], False, 0, symbol, 0, 0, None, 0
    
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

class MarketAnalyzer:
    """
    Class for analyzing market data and conditions to inform trading decisions.
    Enhanced from nice_funcs.py for comprehensive market analysis.
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
            df, tight, wide = calculate_bollinger_bands(df, length=bb_length, std_dev=bb_std_dev)
            
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
            linreg_values = pta.linreg(df['close'], length=length)
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
            resis = df.iloc[-1]['resistance']
            
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

# Additional utility functions from nice_funcs.py

def calculate_vwap_with_symbol_enhanced(symbol: str) -> Tuple[pd.DataFrame, float]:
    """
    Enhanced VWAP calculation with better error handling from nice_funcs.py.
    
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
            logger.warning(f"No data available for {symbol}")
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
        df['VWAP'] = pta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])

        # Get the latest VWAP value
        latest_vwap = df['VWAP'].iloc[-1] if not df.empty else 0.0

        return df, latest_vwap
        
    except Exception as e:
        logger.error(f"Error calculating VWAP for {symbol}: {e}")
        return pd.DataFrame(), 0.0

# Note: These functions require external data sources that may not be available in this environment
# They are included for completeness but may need to be adapted based on available data sources

def get_open_interest():
    """
    Get open interest data from external source.
    Note: This requires external data source configuration.
    """
    try:
        # This would need to be adapted to your specific data source
        # Example implementation shows the structure
        logger.warning("get_open_interest requires external data source configuration")
        return 0
    except Exception as e:
        logger.error(f"Error getting open interest: {e}")
        return 0

def get_liquidations():
    """
    Get liquidation data from external source.
    Note: This requires external data source configuration.
    """
    try:
        # This would need to be adapted to your specific data source
        # Example implementation shows the structure
        logger.warning("get_liquidations requires external data source configuration")
        return "{}"
    except Exception as e:
        logger.error(f"Error getting liquidations: {e}")
        return "{}"

def get_funding_rate():
    """
    Get funding rate data from external source.
    Note: This requires external data source configuration.
    """
    try:
        # This would need to be adapted to your specific data source
        # Example implementation shows the structure
        logger.warning("get_funding_rate requires external data source configuration")
        return "{}"
    except Exception as e:
        logger.error(f"Error getting funding rate: {e}")
        return "{}"

# Create default configuration instance
default_config = TradingConfig() 