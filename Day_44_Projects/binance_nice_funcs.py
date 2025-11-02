# Binance Trading Utilities - Comprehensive Trading Functions
# Binance equivalent of the Hyperliquid nice_funcs.py module

import requests
import time
import pandas as pd
import pandas_ta as ta
import hmac
import hashlib
import urllib.parse
import warnings
import sys, os
from dontshareconfig import binance_api_key, binance_api_secret

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

print('Binance trading utilities loaded 🚀')

# Configuration
symbol = 'BTCUSDT'
timeframe = '15m'
max_loss = -1
target = 5
pos_size = 200
leverage = 10
vol_multiplier = 3
rounding = 4

API_KEY = binance_api_key
API_SECRET = binance_api_secret
BASE_URL = 'https://api.binance.com'

def generate_signature(query_string):
    """Generate HMAC SHA256 signature for Binance API."""
    return hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def make_request(endpoint, params=None, method='GET', signed=True):
    """Make authenticated request to Binance API."""
    if params is None:
        params = {}
    
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
    
    if signed:
        params['timestamp'] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        signature = generate_signature(query_string)
        params['signature'] = signature
    
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, params=params, headers=headers, timeout=10)
        elif method == 'POST':
            response = requests.post(url, params=params, headers=headers, timeout=10)
        elif method == 'DELETE':
            response = requests.delete(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def ask_bid(symbol):
    """Get current bid/ask prices for a symbol."""
    try:
        data = make_request('/api/v3/ticker/bookTicker', {'symbol': symbol}, signed=False)
        if data:
            ask = float(data['askPrice'])
            bid = float(data['bidPrice'])
            return ask, bid, data
        return None, None, None
    except Exception as e:
        print(f"Error getting bid/ask for {symbol}: {e}")
        return None, None, None

def get_current_price(symbol):
    """Fetch the current price for a given symbol."""
    try:
        data = make_request('/api/v3/ticker/price', {'symbol': symbol}, signed=False)
        if data:
            return float(data['price'])
        return None
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
        return None

def get_symbol_info(symbol):
    """Get symbol trading information and precision."""
    try:
        data = make_request('/api/v3/exchangeInfo', signed=False)
        if data and 'symbols' in data:
            for sym_info in data['symbols']:
                if sym_info['symbol'] == symbol:
                    # Extract precision info
                    price_precision = sym_info['quotePrecision']
                    qty_precision = sym_info['baseAssetPrecision']
                    
                    # Get lot size filter
                    min_qty = 0
                    for filter_info in sym_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            min_qty = float(filter_info['minQty'])
                            break
                    
                    return {
                        'symbol': symbol,
                        'price_precision': price_precision,
                        'qty_precision': qty_precision,
                        'min_qty': min_qty,
                        'status': sym_info['status']
                    }
        return None
    except Exception as e:
        print(f"Error getting symbol info for {symbol}: {e}")
        return None

def limit_order(symbol, side, quantity, price, time_in_force='GTC'):
    """Place a limit order on Binance."""
    try:
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': time_in_force,
            'quantity': quantity,
            'price': price
        }
        
        response = make_request('/api/v3/order', params, method='POST')
        return response
    except Exception as e:
        print(f"Error placing limit order: {e}")
        return None

def market_order(symbol, side, quantity):
    """Place a market order on Binance."""
    try:
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': quantity
        }
        
        response = make_request('/api/v3/order', params, method='POST')
        return response
    except Exception as e:
        print(f"Error placing market order: {e}")
        return None

def get_account_balance():
    """Get current account balance."""
    try:
        data = make_request('/api/v3/account')
        if data and 'balances' in data:
            balances = {}
            for balance in data['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            return balances
        return {}
    except Exception as e:
        print(f"Error getting account balance: {e}")
        return {}

def get_balance(asset='USDT'):
    """Get balance for a specific asset."""
    try:
        balances = get_account_balance()
        if asset in balances:
            return balances[asset]['total']
        return 0.0
    except Exception as e:
        print(f"Error getting {asset} balance: {e}")
        return 0.0

def get_open_orders(symbol=None):
    """Get open orders for a symbol or all symbols."""
    try:
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = make_request('/api/v3/openOrders', params)
        return response if response else []
    except Exception as e:
        print(f"Error getting open orders: {e}")
        return []

def cancel_order(symbol, order_id):
    """Cancel a specific order."""
    try:
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        response = make_request('/api/v3/order', params, method='DELETE')
        return response
    except Exception as e:
        print(f"Error canceling order: {e}")
        return None

def cancel_all_orders(symbol=None):
    """Cancel all open orders for a symbol or all symbols."""
    try:
        if symbol:
            params = {'symbol': symbol}
            response = make_request('/api/v3/openOrders', params, method='DELETE')
            return response
        else:
            # Cancel all orders for all symbols
            open_orders = get_open_orders()
            results = []
            for order in open_orders:
                result = cancel_order(order['symbol'], order['orderId'])
                if result:
                    results.append(result)
            return results
    except Exception as e:
        print(f"Error canceling all orders: {e}")
        return []

def get_ohlcv(symbol, interval='15m', limit=500):
    """Get OHLCV data for a symbol."""
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        data = make_request('/api/v3/klines', params, signed=False)
        if data:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error getting OHLCV data: {e}")
        return pd.DataFrame()

def get_ohlcv2(symbol, interval, lookback_days):
    """Get OHLCV data with lookback period."""
    try:
        end_time = int(time.time() * 1000)
        start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        
        data = make_request('/api/v3/klines', params, signed=False)
        if data:
            return data
        return []
    except Exception as e:
        print(f"Error getting OHLCV data with lookback: {e}")
        return []

def process_data_to_df(snapshot_data, time_period=20):
    """Process snapshot data into DataFrame with indicators."""
    try:
        if not snapshot_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(snapshot_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Convert to proper types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Add technical indicators
        if len(df) >= time_period:
            df['sma'] = ta.sma(df['close'], length=time_period)
            df['ema'] = ta.ema(df['close'], length=time_period)
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        return df
    except Exception as e:
        print(f"Error processing data to DataFrame: {e}")
        return pd.DataFrame()

def calculate_vwap_with_symbol(symbol, interval='15m', limit=100):
    """Calculate VWAP for a symbol."""
    try:
        df = get_ohlcv(symbol, interval, limit)
        if not df.empty:
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            return df['vwap'].iloc[-1] if not pd.isna(df['vwap'].iloc[-1]) else None
        return None
    except Exception as e:
        print(f"Error calculating VWAP: {e}")
        return None

def calculate_atr(df, window=14):
    """Calculate Average True Range."""
    try:
        if len(df) < window:
            return df
        
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=window)
        return df
    except Exception as e:
        print(f"Error calculating ATR: {e}")
        return df

def calculate_bollinger_bands(df, length=20, std_dev=2):
    """Calculate Bollinger Bands."""
    try:
        if len(df) < length:
            return df, False, False
        
        bbands = ta.bbands(df['close'], length=length, std=std_dev)
        df['bb_upper'] = bbands[f'BBU_{length}_{std_dev}.0']
        df['bb_middle'] = bbands[f'BBM_{length}_{std_dev}.0']
        df['bb_lower'] = bbands[f'BBL_{length}_{std_dev}.0']
        
        # Determine if bands are tight or wide
        current_width = df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]
        avg_width = (df['bb_upper'] - df['bb_lower']).rolling(50).mean().iloc[-1]
        
        tight = current_width < (avg_width * 0.8)
        wide = current_width > (avg_width * 1.2)
        
        return df, tight, wide
    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        return df, False, False

def volume_spike(df, multiplier=2):
    """Detect volume spikes."""
    try:
        if len(df) < 20:
            return False
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        return current_volume > (avg_volume * multiplier)
    except Exception as e:
        print(f"Error detecting volume spike: {e}")
        return False

def get_order_book(symbol, limit=100):
    """Get order book depth."""
    try:
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        data = make_request('/api/v3/depth', params, signed=False)
        if data:
            return data
        return None
    except Exception as e:
        print(f"Error getting order book: {e}")
        return None

def kill_switch(symbol):
    """Emergency close all positions for a symbol."""
    try:
        # Cancel all orders first
        cancel_all_orders(symbol)
        
        # Get current balance of base asset
        base_asset = symbol.replace('USDT', '').replace('BTC', '').replace('ETH', '')
        balance = get_balance(base_asset)
        
        if balance > 0:
            # Market sell all holdings
            result = market_order(symbol, 'SELL', balance)
            print(f"Kill switch executed: Sold {balance} {base_asset}")
            return result
        
        print("No position to close")
        return True
    except Exception as e:
        print(f"Error in kill switch: {e}")
        return False

def pnl_close(symbol, target_profit=5, max_loss=-1):
    """Close position based on PnL targets."""
    try:
        # This is a simplified version - would need to track entry prices
        current_price = get_current_price(symbol)
        if not current_price:
            return False
        
        # Get account balance to estimate PnL
        # In a real implementation, you'd track entry prices and calculate actual PnL
        print(f"PnL monitoring for {symbol} at price {current_price}")
        return True
    except Exception as e:
        print(f"Error in PnL close: {e}")
        return False

def adjust_leverage_usd_size(symbol, usd_size, leverage=1):
    """Calculate position size based on USD amount and leverage."""
    try:
        current_price = get_current_price(symbol)
        if not current_price:
            return 0
        
        # For spot trading, leverage is 1
        # For futures, you'd use the leverage parameter
        quantity = usd_size / current_price
        
        # Round to appropriate precision
        symbol_info = get_symbol_info(symbol)
        if symbol_info:
            precision = symbol_info['qty_precision']
            quantity = round(quantity, precision)
        
        return quantity
    except Exception as e:
        print(f"Error calculating position size: {e}")
        return 0

def get_funding_rate(symbol=None):
    """Get funding rates (for futures)."""
    try:
        # This would be for Binance Futures API
        # /fapi/v1/premiumIndex
        print("Funding rate function - would need Futures API")
        return None
    except Exception as e:
        print(f"Error getting funding rate: {e}")
        return None

def get_open_interest():
    """Get open interest data (for futures)."""
    try:
        # This would be for Binance Futures API
        print("Open interest function - would need Futures API")
        return None
    except Exception as e:
        print(f"Error getting open interest: {e}")
        return None

def get_liquidations():
    """Get liquidation data (for futures)."""
    try:
        # This would be for Binance Futures API
        print("Liquidations function - would need Futures API")
        return None
    except Exception as e:
        print(f"Error getting liquidations: {e}")
        return None

def connect():
    """Test connection to Binance API."""
    try:
        data = make_request('/api/v3/time', signed=False)
        if data and 'serverTime' in data:
            print(f"✅ Connected to Binance API. Server time: {data['serverTime']}")
            return {'status': 'connected', 'server_time': data['serverTime']}
        return {'status': 'failed'}
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def test_connection():
    """Test API connection and permissions."""
    try:
        # Test public endpoint
        time_data = make_request('/api/v3/time', signed=False)
        if not time_data:
            return False
        
        # Test private endpoint
        account_data = make_request('/api/v3/account')
        if not account_data:
            return False
        
        print("✅ Binance API connection and authentication successful")
        return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

# Spot-specific functions
def spot_limit_order(symbol, side, quantity, price):
    """Place a spot limit order."""
    return limit_order(symbol, side, quantity, price)

def spot_market_order(symbol, side, quantity):
    """Place a spot market order."""
    return market_order(symbol, side, quantity)

def get_trade_history(symbol, limit=100):
    """Get trade history for a symbol."""
    try:
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        response = make_request('/api/v3/myTrades', params)
        return response if response else []
    except Exception as e:
        print(f"Error getting trade history: {e}")
        return []

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window

def get_latest_sma(symbol, interval='15m', window=20, lookback_days=1):
    """Get latest SMA value."""
    try:
        df = get_ohlcv(symbol, interval, window + 10)
        if len(df) >= window:
            return calculate_sma(df['close'].tolist(), window)
        return None
    except Exception as e:
        print(f"Error getting latest SMA: {e}")
        return None

def ob_data(symbol):
    """Get comprehensive order book data."""
    try:
        # Get Binance order book
        binance_ob = get_order_book(symbol, 100)
        if not binance_ob:
            return None
        
        # Process order book data
        bids = [[float(price), float(qty)] for price, qty in binance_ob['bids'][:50]]
        asks = [[float(price), float(qty)] for price, qty in binance_ob['asks'][:50]]
        
        # Find largest bid/ask
        max_bid = max(bids, key=lambda x: x[1]) if bids else [0, 0]
        max_ask = max(asks, key=lambda x: x[1]) if asks else [0, 0]
        
        return {
            'bids': bids,
            'asks': asks,
            'max_bid_price': max_bid[0],
            'max_ask_price': max_ask[0],
            'max_bid_size': max_bid[1],
            'max_ask_size': max_ask[1]
        }
    except Exception as e:
        print(f"Error getting order book data: {e}")
        return None

print("🚀 Binance nice_funcs loaded successfully!")
