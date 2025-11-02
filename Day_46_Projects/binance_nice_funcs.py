"""
🌙 Moon Dev's Binance Trading Functions
Built with love by Moon Dev 🚀

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.

Binance version of nice_funcs.py with comprehensive trading utilities for
spot and futures trading, market analysis, position management, and risk controls.
"""

import os
import sys
import time
import hmac
import hashlib
import urllib.parse
import requests
import pandas as pd
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Day_4_Projects.dontshareconfig import BINANCE_API_KEY, BINANCE_SECRET_KEY
except ImportError:
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

print('🌙 Binance Nice Functions Loaded! 🚀')

# Configuration
BASE_URL = "https://api.binance.com"
FUTURES_URL = "https://fapi.binance.com"
HEADERS = {'X-MBX-APIKEY': BINANCE_API_KEY} if BINANCE_API_KEY else {}

# Global parameters
symbol = 'BTCUSDT'
timeframe = '15m'
max_loss = -1
target = 5
pos_size = 200
leverage = 10

def _generate_signature(query_string: str) -> str:
    """Generate HMAC SHA256 signature"""
    if not BINANCE_SECRET_KEY:
        raise ValueError("Secret key required")
    return hmac.new(BINANCE_SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def _make_request(url: str, params: dict = None, signed: bool = False, method: str = 'GET') -> dict:
    """Make authenticated request to Binance API"""
    params = params or {}
    
    if signed:
        params['timestamp'] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        params['signature'] = _generate_signature(query_string)
    
    try:
        if method == 'POST':
            response = requests.post(url, headers=HEADERS, params=params, timeout=10)
        else:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error: {response.status_code}")
            return {}
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return {}

def ask_bid(symbol):
    """Get bid/ask prices"""
    url = f"{BASE_URL}/api/v3/ticker/bookTicker"
    response = _make_request(url, {'symbol': symbol})
    if response:
        return {
            'bid': float(response['bidPrice']),
            'ask': float(response['askPrice']),
            'mid': (float(response['bidPrice']) + float(response['askPrice'])) / 2
        }

def spot_price_and_symbol_info(symbol):
    """Get symbol info and current price"""
    price_url = f"{BASE_URL}/api/v3/ticker/price"
    info_url = f"{BASE_URL}/api/v3/exchangeInfo"
    
    price_data = _make_request(price_url, {'symbol': symbol})
    info_data = _make_request(info_url, {'symbol': symbol})
    
    if price_data and info_data and 'symbols' in info_data:
        symbol_info = info_data['symbols'][0]
        return {
            'symbol': symbol,
            'price': float(price_data['price']),
            'base_asset': symbol_info['baseAsset'],
            'quote_asset': symbol_info['quoteAsset']
        }

def spot_limit_order(symbol, side, quantity, price):
    """Place spot limit order"""
    url = f"{BASE_URL}/api/v3/order"
    params = {
        'symbol': symbol, 'side': side.upper(), 'type': 'LIMIT',
        'timeInForce': 'GTC', 'quantity': quantity, 'price': price
    }
    return _make_request(url, params, signed=True, method='POST')

def limit_order(symbol, side, quantity, price, reduce_only=False):
    """Place futures limit order"""
    url = f"{FUTURES_URL}/fapi/v1/order"
    params = {
        'symbol': symbol, 'side': side.upper(), 'type': 'LIMIT',
        'timeInForce': 'GTC', 'quantity': quantity, 'price': price
    }
    if reduce_only:
        params['reduceOnly'] = 'true'
    return _make_request(url, params, signed=True, method='POST')

def get_ohlcv(symbol, interval, limit=500):
    """Get OHLCV data"""
    url = f"{BASE_URL}/api/v3/klines"
    data = _make_request(url, {'symbol': symbol, 'interval': interval, 'limit': limit})
    
    if data:
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols]
    return pd.DataFrame()

def get_position(symbol):
    """Get futures position"""
    url = f"{FUTURES_URL}/fapi/v2/positionRisk"
    data = _make_request(url, signed=True)
    
    if data:
        for pos in data:
            if pos['symbol'] == symbol:
                return {
                    'symbol': pos['symbol'],
                    'position_amt': float(pos['positionAmt']),
                    'entry_price': float(pos['entryPrice']),
                    'pnl': float(pos['unRealizedProfit']),
                    'pnl_pct': float(pos['percentage']),
                    'side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT' if float(pos['positionAmt']) < 0 else 'NONE'
                }

def kill_switch_mkt(symbol):
    """Close position with market order"""
    position = get_position(symbol)
    if not position or position['position_amt'] == 0:
        return None
        
    close_side = 'SELL' if position['position_amt'] > 0 else 'BUY'
    close_qty = abs(position['position_amt'])
    
    url = f"{FUTURES_URL}/fapi/v1/order"
    params = {
        'symbol': symbol, 'side': close_side, 'type': 'MARKET',
        'quantity': close_qty, 'reduceOnly': 'true'
    }
    return _make_request(url, params, signed=True, method='POST')

def get_balance():
    """Get USDT balance"""
    url = f"{FUTURES_URL}/fapi/v2/balance"
    data = _make_request(url, signed=True)
    
    if data:
        for balance in data:
            if balance['asset'] == 'USDT':
                return float(balance['balance'])
    return 0.0

def pnl_close(symbol, target_pnl, max_loss):
    """Close position based on PnL thresholds"""
    position = get_position(symbol)
    if not position or position['position_amt'] == 0:
        return None
        
    current_pnl_pct = position['pnl_pct']
    
    if current_pnl_pct >= target_pnl:
        print(f"🎯 Target reached! PnL: {current_pnl_pct:.2f}%")
        return kill_switch_mkt(symbol)
    elif current_pnl_pct <= max_loss:
        print(f"🛑 Stop loss hit! PnL: {current_pnl_pct:.2f}%")
        return kill_switch_mkt(symbol)
    
def connect():
    """Test API connection"""
    url = f"{BASE_URL}/api/v3/ping"
    response = _make_request(url)
    return response == {}

# Technical analysis functions
def calculate_sma(prices, window):
    """Calculate SMA"""
    return sum(prices[-window:]) / window if len(prices) >= window else None

def calculate_atr(df, window=14):
    """Calculate ATR using pandas_ta"""
    return ta.atr(df['high'], df['low'], df['close'], length=window)

def calculate_bollinger_bands(df, length=20, std_dev=2):
    """Calculate Bollinger Bands using pandas_ta"""
    return ta.bbands(df['close'], length=length, std=std_dev)

# Additional utility functions
def cancel_all_orders():
    """Cancel all open orders"""
    # Cancel futures orders
    url = f"{FUTURES_URL}/fapi/v1/openOrders"
    orders = _make_request(url, signed=True)
    if orders:
        for order in orders:
            cancel_url = f"{FUTURES_URL}/fapi/v1/order"
            params = {'symbol': order['symbol'], 'orderId': order['orderId']}
            _make_request(cancel_url, params, signed=True, method='DELETE')
    return True

def adjust_leverage(symbol, leverage_target):
    """Adjust leverage"""
    url = f"{FUTURES_URL}/fapi/v1/leverage"
    params = {'symbol': symbol, 'leverage': leverage_target}
    return _make_request(url, params, signed=True, method='POST')

def close_all_positions_mkt():
    """Close all positions with market orders"""
    url = f"{FUTURES_URL}/fapi/v2/positionRisk"
    positions = _make_request(url, signed=True)
    closed = []
    
    if positions:
        for pos in positions:
            if float(pos['positionAmt']) != 0:
                result = kill_switch_mkt(pos['symbol'])
                if result:
                    closed.append(pos['symbol'])
    return closed
