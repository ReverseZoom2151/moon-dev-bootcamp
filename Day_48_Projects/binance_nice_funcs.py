import requests
import pandas as pd
import time
import pandas_ta as ta
import os
import json
import hmac
import hashlib
import urllib.parse
from binance_CONFIG import *
from termcolor import cprint 
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Constants
CLOSED_POSITIONS_TXT = CLOSED_POSITIONS_TXT  # From config
SELL_AMOUNT_PERC = 0.5  # 50% of balance to sell per transaction
DEFAULT_SYMBOL = "BTCUSDT"  # Default symbol for testing

# --- Helper Functions ---

def _get_signature(params, api_secret):
    """Generate HMAC SHA256 signature for Binance API"""
    query_string = urllib.parse.urlencode(params)
    return hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def _make_binance_request(endpoint: str, params: dict = None, authenticated: bool = False, method: str = 'GET'):
    """Helper function to make requests to the Binance API."""
    if params is None:
        params = {}
    
    base_url = BINANCE_FAPI_URL if TRADING_MODE == "FUTURES" else BINANCE_BASE_URL
    url = f"{base_url}{endpoint}"
    headers = {}
    
    if authenticated:
        if not API_KEY or not API_SECRET:
            cprint("Error: Binance API credentials not configured", "red")
            return None
        
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = _get_signature(params, API_SECRET)
        headers['X-MBX-APIKEY'] = API_KEY
    
    try:
        if method.upper() == 'POST':
            response = requests.post(url, headers=headers, params=params)
        else:
            response = requests.get(url, headers=headers, params=params)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        cprint(f"Binance API request failed: {e}", "red")
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

def get_account_balance(asset: str = "USDT"):
    """Get account balance for a specific asset"""
    if TRADING_MODE == "FUTURES":
        endpoint = "/fapi/v2/balance"
    else:
        endpoint = "/account"
    
    data = _make_binance_request(endpoint, authenticated=True)
    if not data:
        return 0.0
    
    if TRADING_MODE == "FUTURES":
        for balance in data:
            if balance['asset'] == asset:
                return float(balance['balance'])
    else:
        for balance in data['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
    
    return 0.0

def get_symbol_info(symbol: str):
    """Get symbol information including price, volume, etc."""
    endpoint = "/ticker/24hr"
    params = {"symbol": symbol}
    
    data = _make_binance_request(endpoint, params=params)
    if data:
        return {
            'symbol': data['symbol'],
            'price': float(data['lastPrice']),
            'volume': float(data['volume']),
            'quote_volume': float(data['quoteVolume']),
            'price_change_percent': float(data['priceChangePercent']),
            'high': float(data['highPrice']),
            'low': float(data['lowPrice']),
            'trades': int(data['count'])
        }
    return None

def token_price(symbol: str):
    """Get current price for a symbol"""
    endpoint = "/ticker/price"
    params = {"symbol": symbol}
    
    data = _make_binance_request(endpoint, params=params)
    if data:
        return float(data['price'])
    return None

def get_position(symbol: str):
    """Get current position for a symbol"""
    if TRADING_MODE == "FUTURES":
        endpoint = "/fapi/v2/positionRisk"
        data = _make_binance_request(endpoint, authenticated=True)
        
        if data:
            for position in data:
                if position['symbol'] == symbol:
                    return float(position['positionAmt'])
    else:
        # For spot trading, get balance of base asset
        base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('BTC', '').replace('ETH', '')
        endpoint = "/account"
        data = _make_binance_request(endpoint, authenticated=True)
        
        if data:
            for balance in data['balances']:
                if balance['asset'] == base_asset:
                    return float(balance['free'])
    
    return 0.0

def market_buy(symbol: str, quantity: float = None, quote_quantity: float = None):
    """Execute a market buy order"""
    if TRADING_MODE == "FUTURES":
        endpoint = "/fapi/v1/order"
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET'
        }
        if quote_quantity:
            params['quoteOrderQty'] = quote_quantity
        else:
            params['quantity'] = quantity
    else:
        endpoint = "/order"
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET'
        }
        if quote_quantity:
            params['quoteOrderQty'] = quote_quantity
        else:
            params['quantity'] = quantity
    
    try:
        result = _make_binance_request(endpoint, params=params, authenticated=True, method='POST')
        if result:
            cprint(f"Market buy executed: {symbol} - OrderId: {result.get('orderId')}", "green")
            return result
        else:
            cprint(f"Market buy failed for {symbol}", "red")
            return None
    except Exception as e:
        cprint(f"Error executing market buy: {e}", "red")
        return None

def market_sell(symbol: str, quantity: float):
    """Execute a market sell order"""
    if TRADING_MODE == "FUTURES":
        endpoint = "/fapi/v1/order"
    else:
        endpoint = "/order"
    
    params = {
        'symbol': symbol,
        'side': 'SELL',
        'type': 'MARKET',
        'quantity': quantity
    }
    
    try:
        result = _make_binance_request(endpoint, params=params, authenticated=True, method='POST')
        if result:
            cprint(f"Market sell executed: {symbol} - OrderId: {result.get('orderId')}", "green")
            return result
        else:
            cprint(f"Market sell failed for {symbol}", "red")
            return None
    except Exception as e:
        cprint(f"Error executing market sell: {e}", "red")
        return None

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

def get_historical_data(symbol: str, interval: str = '5m', limit: int = 100):
    """Get historical kline/candlestick data"""
    endpoint = "/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    data = _make_binance_request(endpoint, params=params)
    if data:
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        return df
    return None

def check_trend(symbol: str, periods: int = 20):
    """Check if symbol is in uptrend using SMA"""
    df = get_historical_data(symbol, interval=SMA_TIMEFRAME, limit=periods + 10)
    if df is not None and len(df) >= periods:
        df['sma'] = ta.sma(df['close'], length=periods)
        current_price = df['close'].iloc[-1]
        current_sma = df['sma'].iloc[-1]
        
        return {
            'is_uptrend': current_price > current_sma,
            'price': current_price,
            'sma': current_sma,
            'trend_strength': (current_price - current_sma) / current_sma * 100
        }
    return None

def supply_demand_zones(symbol: str, days_back: int = 1, timeframe: str = '15m'):
    """Identify supply and demand zones using pivot points"""
    # Calculate periods based on days and timeframe
    if timeframe == '1m':
        periods = days_back * 24 * 60
    elif timeframe == '5m':
        periods = days_back * 24 * 12
    elif timeframe == '15m':
        periods = days_back * 24 * 4
    elif timeframe == '1h':
        periods = days_back * 24
    else:
        periods = 100  # Default
    
    df = get_historical_data(symbol, interval=timeframe, limit=min(periods, 1000))
    if df is None or len(df) < 20:
        return None
    
    # Calculate pivot points
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['support1'] = 2 * df['pivot'] - df['high']
    df['resistance1'] = 2 * df['pivot'] - df['low']
    df['support2'] = df['pivot'] - (df['high'] - df['low'])
    df['resistance2'] = df['pivot'] + (df['high'] - df['low'])
    
    current_price = df['close'].iloc[-1]
    latest_pivot = df['pivot'].iloc[-1]
    latest_support = df['support1'].iloc[-1]
    latest_resistance = df['resistance1'].iloc[-1]
    
    # Determine if current price is near support/demand zone
    near_support = abs(current_price - latest_support) / current_price < 0.02  # Within 2%
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
    
    # For simplicity, assuming average cost basis (would need to track this properly)
    # This is a simplified implementation
    position_value = position * current_price
    
    cprint(f"Closing position for {symbol}: {position} units at ${current_price}", "cyan")
    
    result = market_sell(symbol, position)
    if result:
        _add_to_closed_positions(symbol)
        cprint(f"Position closed for {symbol}. Value: ${position_value:.2f}", "green")
        
        # Update leaderboard (simplified)
        update_leaderboard("binance_trader", position_value)
    
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

def get_top_volume_symbols(limit: int = 50):
    """Get top symbols by 24h volume"""
    endpoint = "/ticker/24hr"
    data = _make_binance_request(endpoint)
    
    if data:
        # Filter for USDT pairs and sort by volume
        usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        
        return [item['symbol'] for item in usdt_pairs[:limit]]
    
    return []

def get_trending_symbols():
    """Get trending symbols based on price change and volume"""
    endpoint = "/ticker/24hr"
    data = _make_binance_request(endpoint)
    
    if data:
        # Filter for significant price changes and volume
        trending = []
        for item in data:
            if (item['symbol'].endswith('USDT') and 
                float(item['priceChangePercent']) > 5.0 and  # > 5% price change
                float(item['quoteVolume']) > MIN_VOLUME_24H and
                int(item['count']) > MIN_TRADES_LAST_HOUR):
                trending.append({
                    'symbol': item['symbol'],
                    'price_change': float(item['priceChangePercent']),
                    'volume': float(item['quoteVolume']),
                    'price': float(item['lastPrice'])
                })
        
        # Sort by price change
        trending.sort(key=lambda x: x['price_change'], reverse=True)
        return trending[:20]  # Return top 20
    
    return []

def open_position(symbol: str, usd_amount: float = None):
    """Open a position for a symbol"""
    if not usd_amount:
        usd_amount = DEFAULT_TRADE_SIZE_USDT
    
    # Check if symbol is in do not trade list
    if symbol in DO_NOT_TRADE_LIST:
        cprint(f"Symbol {symbol} is in do not trade list", "yellow")
        return None
    
    # Check account balance
    balance = get_account_balance()
    if balance < usd_amount:
        cprint(f"Insufficient balance. Available: ${balance:.2f}, Required: ${usd_amount:.2f}", "red")
        return None
    
    # Get current price
    current_price = token_price(symbol)
    if not current_price:
        cprint(f"Could not get price for {symbol}", "red")
        return None
    
    cprint(f"Opening position for {symbol} with ${usd_amount:.2f} at price ${current_price:.6f}", "cyan")
    
    # Execute market buy
    result = market_buy(symbol, quote_quantity=usd_amount)
    
    if result:
        cprint(f"Position opened successfully for {symbol}", "green")
        # Log the trade
        try:
            trade_data = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'BUY',
                'amount_usd': usd_amount,
                'price': current_price,
                'order_id': result.get('orderId')
            }
            
            # Save to CSV
            if os.path.exists(ALL_TRENDING_EVER_CSV):
                df = pd.read_csv(ALL_TRENDING_EVER_CSV)
                new_row = pd.DataFrame([trade_data])
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = pd.DataFrame([trade_data])
            
            df.to_csv(ALL_TRENDING_EVER_CSV, index=False)
            cprint(f"Trade logged to {ALL_TRENDING_EVER_CSV}", "green")
        except Exception as e:
            cprint(f"Error logging trade: {e}", "red")
    
    return result

def chunk_kill(symbol: str, max_order_size_usdt: float, sleep_between_orders: int = 60):
    """Sell position in chunks to minimize market impact"""
    position = get_position(symbol)
    if position <= 0:
        cprint(f"No position to sell for {symbol}", "yellow")
        return None
    
    current_price = token_price(symbol)
    if not current_price:
        cprint(f"Could not get price for {symbol}", "red")
        return None
    
    total_value = position * current_price
    cprint(f"Chunked selling {symbol}: {position} units (${total_value:.2f})", "cyan")
    
    # Calculate number of chunks
    max_quantity = max_order_size_usdt / current_price
    chunks = int(position / max_quantity) + (1 if position % max_quantity > 0 else 0)
    
    results = []
    remaining = position
    
    for i in range(chunks):
        if remaining <= 0:
            break
        
        # Calculate quantity for this chunk
        chunk_quantity = min(remaining, max_quantity)
        
        cprint(f"Selling chunk {i+1}/{chunks}: {chunk_quantity:.6f} {symbol}", "yellow")
        
        result = market_sell(symbol, chunk_quantity)
        if result:
            results.append(result)
            remaining -= chunk_quantity
            cprint(f"Chunk {i+1} sold successfully", "green")
        else:
            cprint(f"Failed to sell chunk {i+1}", "red")
            break
        
        # Sleep between orders except for the last one
        if i < chunks - 1 and remaining > 0:
            cprint(f"Waiting {sleep_between_orders} seconds before next chunk...", "cyan")
            time.sleep(sleep_between_orders)
    
    if remaining <= 0:
        _add_to_closed_positions(symbol)
        cprint(f"All chunks sold successfully for {symbol}", "green")
    else:
        cprint(f"Partial sell completed. Remaining: {remaining:.6f} {symbol}", "yellow")
    
    return results

def round_down(value: float, decimals: int):
    """Round down to specified decimal places"""
    multiplier = 10 ** decimals
    return int(value * multiplier) / multiplier

def get_data(symbol: str, days_back: int = 3, timeframe: str = '5m'):
    """Get historical data for analysis"""
    return get_historical_data(symbol, interval=timeframe, limit=days_back * 288)  # 288 5-min intervals per day

# Compatibility functions for existing code
def token_overview(symbol: str):
    """Get token overview (compatibility function)"""
    return get_symbol_info(symbol)

def ask_bid(symbol: str):
    """Get current price (compatibility function)"""
    return token_price(symbol)

def fetch_wallet_holdings_nosaving_names(wallet_address: str = None, token_filter: str = None):
    """Compatibility function - returns account balances"""
    if TRADING_MODE == "FUTURES":
        endpoint = "/fapi/v2/balance"
    else:
        endpoint = "/account"
    
    data = _make_binance_request(endpoint, authenticated=True)
    if data:
        if TRADING_MODE == "FUTURES":
            balances = [{'asset': item['asset'], 'balance': float(item['balance'])} 
                       for item in data if float(item['balance']) > 0]
        else:
            balances = [{'asset': item['asset'], 'balance': float(item['free'])} 
                       for item in data['balances'] if float(item['free']) > 0]
        
        return pd.DataFrame(balances)
    
    return pd.DataFrame()

def get_time_range(days_back: int):
    """Get time range for data queries"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    return int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000)

# Print helper functions
def print_pretty_json(data):
    """Print JSON data in a readable format"""
    print(json.dumps(data, indent=2))

def find_urls(text: str):
    """Find URLs in text (basic implementation)"""
    import re
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)
