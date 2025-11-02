import requests
import pandas as pd
import time
import pandas_ta as ta
import os
import json
import hmac
import hashlib
from bitfinex_CONFIG import *
from termcolor import cprint 
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Constants
CLOSED_POSITIONS_TXT = CLOSED_POSITIONS_TXT  # From config
SELL_AMOUNT_PERC = 0.5  # 50% of balance to sell per transaction
DEFAULT_SYMBOL = "tBTCUSD"  # Default symbol for testing (with 't' prefix)

# --- Helper Functions ---

def _get_signature(path, nonce, body):
    """Generate HMAC SHA384 signature for Bitfinex API"""
    message = f"/api/v2/{path}{nonce}{body}"
    return hmac.new(
        API_SECRET.encode(),
        message.encode(),
        hashlib.sha384
    ).hexdigest()

def _make_bitfinex_request(endpoint: str, params: dict = None, authenticated: bool = False, method: str = 'GET'):
    """Helper function to make requests to the Bitfinex API."""
    if params is None:
        params = {}
    
    if authenticated:
        url = f"{BITFINEX_AUTH_URL}{endpoint}"
        
        if not API_KEY or not API_SECRET:
            cprint("Error: Bitfinex API credentials not configured", "red")
            return None
        
        nonce = str(int(time.time() * 1000000))
        body = json.dumps(params)
        signature = _get_signature(endpoint, nonce, body)
        
        headers = {
            'bfx-nonce': nonce,
            'bfx-apikey': API_KEY,
            'bfx-signature': signature,
            'content-type': 'application/json'
        }
        
        try:
            if method.upper() == 'POST':
                response = requests.post(url, headers=headers, data=body)
            else:
                response = requests.get(url, headers=headers, data=body)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            cprint(f"Bitfinex authenticated API request failed: {e}", "red")
            return None
    else:
        url = f"{BITFINEX_BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            cprint(f"Bitfinex public API request failed: {e}", "red")
            return None

def _read_closed_positions():
    """Reads and returns a set of token symbols from closed positions file."""
    if not os.path.exists(CLOSED_POSITIONS_TXT):
        return set()
    
    with open(CLOSED_POSITIONS_TXT, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def _add_to_closed_positions(symbol: str):
    """Adds a trading symbol to closed positions file if not already present."""
    closed_positions = _read_closed_positions()
    if symbol not in closed_positions:
        os.makedirs(os.path.dirname(CLOSED_POSITIONS_TXT), exist_ok=True)
        with open(CLOSED_POSITIONS_TXT, 'a') as f:
            f.write(f"{symbol}\n")
        cprint(f"Added {symbol} to closed positions", "green")

def get_account_balance(currency: str = "USD"):
    """Get account balance for a specific currency"""
    endpoint = "/r/wallets"
    data = _make_bitfinex_request(endpoint, authenticated=True)
    
    if data:
        for wallet in data:
            if len(wallet) >= 3:
                wallet_type = wallet[0]
                wallet_currency = wallet[1]
                balance = float(wallet[2]) if wallet[2] else 0.0
                
                if (wallet_currency.upper() == currency.upper() and 
                    wallet_type in ['exchange', 'margin']):
                    return balance
    
    return 0.0

def token_price(symbol: str):
    """Get current price for a symbol"""
    if not symbol.startswith('t'):
        symbol = f"t{symbol}"
    
    endpoint = f"/ticker/{symbol}"
    data = _make_bitfinex_request(endpoint)
    
    if data and len(data) >= 7:
        return float(data[6]) if data[6] else None
    return None

def get_position(symbol: str):
    """Get current position for a symbol"""
    if not symbol.startswith('t'):
        symbol = f"t{symbol}"
    
    endpoint = "/r/positions"
    data = _make_bitfinex_request(endpoint, authenticated=True)
    
    if data:
        for position in data:
            if len(position) >= 3 and position[0] == symbol:
                return float(position[2]) if position[2] else 0.0
    
    return 0.0

def market_buy(symbol: str, amount: float = None, quote_amount: float = None):
    """Execute a market buy order"""
    if not symbol.startswith('t'):
        symbol = f"t{symbol}"
    
    if quote_amount:
        current_price = token_price(symbol)
        if not current_price:
            cprint(f"Could not get price for {symbol}", "red")
            return None
        amount = quote_amount / current_price
    
    if not amount:
        cprint("Error: No amount specified for market buy", "red")
        return None
    
    endpoint = "/w/order/submit"
    params = {
        'type': 'EXCHANGE MARKET',
        'symbol': symbol,
        'amount': str(amount),
        'price': '0',
    }
    
    try:
        result = _make_bitfinex_request(endpoint, params=params, authenticated=True, method='POST')
        
        if result and len(result) > 0 and result[6] == 'SUCCESS':
            order_details = result[4]
            cprint(f"Market buy executed: {symbol} - OrderId: {order_details[0]}", "green")
            return order_details
        else:
            cprint(f"Market buy failed for {symbol}", "red")
            return None
    except Exception as e:
        cprint(f"Error executing market buy: {e}", "red")
        return None

def market_sell(symbol: str, amount: float):
    """Execute a market sell order"""
    if not symbol.startswith('t'):
        symbol = f"t{symbol}"
    
    endpoint = "/w/order/submit"
    params = {
        'type': 'EXCHANGE MARKET',
        'symbol': symbol,
        'amount': str(-abs(amount)),
        'price': '0',
    }
    
    try:
        result = _make_bitfinex_request(endpoint, params=params, authenticated=True, method='POST')
        
        if result and len(result) > 0 and result[6] == 'SUCCESS':
            order_details = result[4]
            cprint(f"Market sell executed: {symbol} - OrderId: {order_details[0]}", "green")
            return order_details
        else:
            cprint(f"Market sell failed for {symbol}", "red")
            return None
    except Exception as e:
        cprint(f"Error executing market sell: {e}", "red")
        return None

def get_historical_data(symbol: str, timeframe: str = '1h', limit: int = 100):
    """Get historical candlestick data"""
    if not symbol.startswith('t'):
        symbol = f"t{symbol}"
    
    tf_map = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '3h': '3h', '6h': '6h', '12h': '12h',
        '1D': '1D', '7D': '7D', '14D': '14D', '1M': '1M'
    }
    
    bitfinex_tf = tf_map.get(timeframe, '1h')
    
    end_time = int(time.time() * 1000)
    if bitfinex_tf == '1m':
        start_time = end_time - (limit * 60 * 1000)
    elif bitfinex_tf == '1h':
        start_time = end_time - (limit * 60 * 60 * 1000)
    else:
        start_time = end_time - (limit * 24 * 60 * 60 * 1000)
    
    endpoint = f"/candles/trade:{bitfinex_tf}:{symbol}/hist"
    params = {
        'start': start_time,
        'end': end_time,
        'limit': limit,
        'sort': 1
    }
    
    data = _make_bitfinex_request(endpoint, params=params)
    
    if data:
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'close', 'high', 'low', 'volume'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return df.sort_values('timestamp').reset_index(drop=True)
    
    return None

def check_trend(symbol: str, periods: int = 21):
    """Professional trend analysis"""
    df = get_historical_data(symbol, timeframe=SMA_TIMEFRAME, limit=periods + 50)
    if df is None or len(df) < periods:
        return None
    
    df['sma'] = ta.sma(df['close'], length=periods)
    df['ema_fast'] = ta.ema(df['close'], length=EMA_FAST)
    df['ema_slow'] = ta.ema(df['close'], length=EMA_SLOW)
    df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)
    
    current_price = df['close'].iloc[-1]
    current_sma = df['sma'].iloc[-1]
    current_ema_fast = df['ema_fast'].iloc[-1]
    current_ema_slow = df['ema_slow'].iloc[-1]
    current_rsi = df['rsi'].iloc[-1]
    
    trend_signals = 0
    if current_price > current_ema_fast:
        trend_signals += 1
    if current_price > current_ema_slow:
        trend_signals += 1
    if current_ema_fast > current_ema_slow:
        trend_signals += 1
    if 50 < current_rsi < 70:
        trend_signals += 1
    
    is_uptrend = trend_signals >= 3
    trend_strength = (current_price - current_sma) / current_sma * 100
    
    return {
        'is_uptrend': is_uptrend,
        'trend_strength': trend_strength,
        'price': current_price,
        'sma': current_sma,
        'ema_fast': current_ema_fast,
        'ema_slow': current_ema_slow,
        'rsi': current_rsi
    }

def supply_demand_zones(symbol: str, days_back: int = 2, timeframe: str = '1h'):
    """Professional supply and demand zone analysis"""
    if timeframe == '1h':
        periods = days_back * 24
    else:
        periods = days_back * 24
    
    df = get_historical_data(symbol, timeframe=timeframe, limit=min(periods, 1000))
    if df is None or len(df) < 20:
        return None
    
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['support1'] = 2 * df['pivot'] - df['high']
    df['resistance1'] = 2 * df['pivot'] - df['low']
    df['support2'] = df['pivot'] - (df['high'] - df['low'])
    df['resistance2'] = df['pivot'] + (df['high'] - df['low'])
    
    current_price = df['close'].iloc[-1]
    latest_pivot = df['pivot'].iloc[-1]
    latest_support = df['support1'].iloc[-1]
    latest_resistance = df['resistance1'].iloc[-1]
    
    near_support = abs(current_price - latest_support) / current_price < 0.02
    near_resistance = abs(current_price - latest_resistance) / current_price < 0.02
    
    return {
        'current_price': current_price,
        'pivot': latest_pivot,
        'support': latest_support,
        'resistance': latest_resistance,
        'near_support': near_support,
        'near_resistance': near_resistance,
        'support_distance': (current_price - latest_support) / current_price * 100,
        'resistance_distance': (latest_resistance - current_price) / current_price * 100
    }

def kill_switch(symbol: str):
    """Emergency sell all position for a symbol"""
    position = get_position(symbol)
    if position > 0:
        cprint(f"KILL SWITCH ACTIVATED for {symbol} - Selling {position}", "red")
        result = market_sell(symbol, position)
        if result:
            _add_to_closed_positions(symbol)
            cprint(f"Emergency exit completed for {symbol}", "green")
        return result
    else:
        cprint(f"No position to close for {symbol}", "yellow")
        return None

def pnl_close(symbol: str):
    """Calculate PnL and close position"""
    position = get_position(symbol)
    if position <= 0:
        cprint(f"No position to close for {symbol}", "yellow")
        return None
    
    current_price = token_price(symbol)
    if not current_price:
        cprint(f"Could not get price for {symbol}", "red")
        return None
    
    position_value = abs(position) * current_price
    
    cprint(f"Closing position for {symbol}: {position} units at ${current_price:.6f}", "cyan")
    
    result = market_sell(symbol, position) if position > 0 else market_buy(symbol, amount=abs(position))
    if result:
        _add_to_closed_positions(symbol)
        cprint(f"Position closed for {symbol}. Value: ${position_value:.2f}", "green")
        update_leaderboard("bitfinex_trader", position_value)
    
    return result

def update_leaderboard(trader_name: str, pnl: float):
    """Update trader leaderboard"""
    try:
        if os.path.exists(LEADERBOARD_CSV):
            df = pd.read_csv(LEADERBOARD_CSV)
        else:
            df = pd.DataFrame(columns=['trader', 'total_pnl', 'trades', 'last_trade'])
        
        if trader_name in df['trader'].values:
            df.loc[df['trader'] == trader_name, 'total_pnl'] += pnl
            df.loc[df['trader'] == trader_name, 'trades'] += 1
            df.loc[df['trader'] == trader_name, 'last_trade'] = datetime.now()
        else:
            new_row = pd.DataFrame({
                'trader': [trader_name],
                'total_pnl': [pnl],
                'trades': [1],
                'last_trade': [datetime.now()]
            })
            df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_csv(LEADERBOARD_CSV, index=False)
        cprint(f"Updated leaderboard for {trader_name}: ${pnl:.2f}", "green")
    except Exception as e:
        cprint(f"Error updating leaderboard: {e}", "red")

# Compatibility functions
def round_down(value: float, decimals: int):
    """Round down to specified decimal places"""
    multiplier = 10 ** decimals
    return int(value * multiplier) / multiplier

def get_data(symbol: str, days_back: int = 5, timeframe: str = '1h'):
    """Get historical data for analysis"""
    return get_historical_data(symbol, timeframe=timeframe, limit=days_back * 24)

def token_overview(symbol: str):
    """Get token overview"""
    info = get_symbol_info(symbol)
    return info

def get_symbol_info(symbol: str):
    """Get symbol information"""
    if not symbol.startswith('t'):
        symbol = f"t{symbol}"
    
    endpoint = f"/ticker/{symbol}"
    data = _make_bitfinex_request(endpoint)
    
    if data and len(data) >= 11:
        return {
            'symbol': symbol,
            'price': float(data[6]) if data[6] else 0,
            'daily_change_percent': float(data[5]) * 100 if data[5] else 0,
            'volume': float(data[7]) if data[7] else 0,
            'high': float(data[8]) if data[8] else 0,
            'low': float(data[9]) if data[9] else 0
        }
    return None

def chunk_kill(symbol: str, max_order_size_usd: float, sleep_between_orders: int = 120):
    """Sell position in chunks"""
    position = get_position(symbol)
    if position <= 0:
        cprint(f"No position to sell for {symbol}", "yellow")
        return None
    
    current_price = token_price(symbol)
    if not current_price:
        return None
    
    total_value = abs(position) * current_price
    max_quantity = max_order_size_usd / current_price
    chunks = int(abs(position) / max_quantity) + (1 if abs(position) % max_quantity > 0 else 0)
    
    results = []
    remaining = abs(position)
    
    for i in range(chunks):
        if remaining <= 0:
            break
        
        chunk_quantity = min(remaining, max_quantity)
        result = market_sell(symbol, chunk_quantity)
        
        if result:
            results.append(result)
            remaining -= chunk_quantity
            cprint(f"Chunk {i+1} sold successfully", "green")
        
        if i < chunks - 1 and remaining > 0:
            time.sleep(sleep_between_orders)
    
    if remaining <= 0:
        _add_to_closed_positions(symbol)
    
    return results
