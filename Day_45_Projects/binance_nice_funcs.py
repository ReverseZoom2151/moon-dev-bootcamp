"""
🚀 Binance Trading Utilities - Moon Dev Style
Comprehensive Binance API wrapper with advanced trading functions

Built with love by Moon Dev 🌙 ✨
Disclaimer: This is not financial advice. Use at your own risk.
"""

import os
import sys
import time
import hmac
import hashlib
import urllib.parse
import requests
import warnings

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dontshareconfig import binance_api_key, binance_api_secret
except ImportError:
    binance_api_key = os.getenv('BINANCE_API_KEY')
    binance_api_secret = os.getenv('BINANCE_SECRET_KEY')

print('🚀 Binance Nice Funcs Loaded!')

# Global configuration
symbol = 'BTCUSDT'
timeframe = '15m'
max_loss = -1
target = 5
pos_size = 200
leverage = 10
vol_multiplier = 3
rounding = 4

class BinanceTrader:
    def __init__(self):
        self.api_key = binance_api_key
        self.api_secret = binance_api_secret
        self.base_url = "https://api.binance.com"
        self.futures_url = "https://fapi.binance.com"
        self.rate_limit_delay = 0.1
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API keys not found!")
    
    def _generate_signature(self, query_string):
        """Generate HMAC SHA256 signature for Binance API"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint, params=None, signed=False, futures=False, method='GET'):
        """Make authenticated request to Binance API"""
        if params is None:
            params = {}
        
        base_url = self.futures_url if futures else self.base_url
        url = f"{base_url}{endpoint}"
        
        headers = {'X-MBX-APIKEY': self.api_key} if self.api_key else {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = urllib.parse.urlencode(params)
            signature = self._generate_signature(query_string)
            params['signature'] = signature
        
        time.sleep(self.rate_limit_delay)
        
        if method == 'POST':
            response = requests.post(url, params=params, headers=headers)
        elif method == 'DELETE':
            response = requests.delete(url, params=params, headers=headers)
        else:
            response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None

# Initialize global trader instance
trader = BinanceTrader()

def ask_bid(symbol):
    """Get current bid and ask prices from order book"""
    try:
        data = trader._make_request('/api/v3/depth', {'symbol': symbol, 'limit': 5})
        if data:
            bids = data['bids']
            asks = data['asks']
            
            bid = float(bids[0][0]) if bids else 0
            ask = float(asks[0][0]) if asks else 0
            
            return ask, bid, {'bids': bids, 'asks': asks}
        return None, None, None
    except Exception as e:
        print(f"Error getting bid/ask for {symbol}: {e}")
        return None, None, None

def spot_price_and_hoe_ass_spot_symbol(symbol):
    """Get current price and symbol info for Binance"""
    try:
        # Get current price
        price_data = trader._make_request('/api/v3/ticker/price', {'symbol': symbol})
        if not price_data:
            return f"Symbol {symbol} not found."
        
        # Get symbol info
        exchange_info = trader._make_request('/api/v3/exchangeInfo')
        if not exchange_info:
            return f"Could not get exchange info for {symbol}."
        
        symbol_info = None
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                symbol_info = s
                break
        
        if not symbol_info:
            return f"Symbol info for {symbol} not found."
        
        mid_px = float(price_data['price'])
        
        # Calculate decimals from filters
        price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        
        px_decimals = 8  # Default
        sz_decimals = 8  # Default
        
        if price_filter and price_filter['tickSize']:
            tick_size = float(price_filter['tickSize'])
            px_decimals = len(str(tick_size).split('.')[-1]) if '.' in str(tick_size) else 0
        
        if lot_size_filter and lot_size_filter['stepSize']:
            step_size = float(lot_size_filter['stepSize'])
            sz_decimals = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
        
        return mid_px, symbol, sz_decimals, px_decimals
        
    except Exception as e:
        print(f"Error getting symbol info: {e}")
        return f"Error getting symbol info: {e}"

def spot_limit_order(coin, is_buy, sz, limit_px, account=None, sz_decimals=8, px_decimals=8):
    """Place a spot limit order on Binance"""
    try:
        side = 'BUY' if is_buy else 'SELL'
        
        # Round to proper decimals
        quantity = round(float(sz), sz_decimals)
        price = round(float(limit_px), px_decimals)
        
        params = {
            'symbol': coin,
            'side': side,
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': str(quantity),
            'price': str(price)
        }
        
        result = trader._make_request('/api/v3/order', params, signed=True, method='POST')
        
        if result:
            print(f"✅ {side} order placed: {quantity} {coin} @ ${price}")
            return result
        else:
            print(f"❌ Failed to place {side} order")
            return None
            
    except Exception as e:
        print(f"Error placing spot limit order: {e}")
        return None

def all_spot_symbols():
    """Get all available spot trading symbols"""
    try:
        exchange_info = trader._make_request('/api/v3/exchangeInfo')
        if exchange_info:
            symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            return symbols
        return []
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []

def get_sz_px_decimals(symbol):
    """Get size and price decimals for a symbol"""
    try:
        result = spot_price_and_hoe_ass_spot_symbol(symbol)
        if isinstance(result, tuple) and len(result) == 4:
            _, _, sz_decimals, px_decimals = result
            return sz_decimals, px_decimals
        return 8, 8  # Default values
    except Exception as e:
        print(f"Error getting decimals for {symbol}: {e}")
        return 8, 8

def limit_order(coin, is_buy, sz, limit_px, reduce_only=False, account=None):
    """Place a futures limit order"""
    try:
        side = 'BUY' if is_buy else 'SELL'
        
        params = {
            'symbol': coin,
            'side': side,
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': str(float(sz)),
            'price': str(float(limit_px))
        }
        
        if reduce_only:
            params['reduceOnly'] = 'true'
        
        result = trader._make_request('/fapi/v1/order', params, signed=True, futures=True, method='POST')
        
        if result:
            print(f"✅ Futures {side} order placed: {sz} {coin} @ ${limit_px}")
            return result
        else:
            print(f"❌ Failed to place futures {side} order")
            return None
            
    except Exception as e:
        print(f"Error placing futures limit order: {e}")
        return None

def adjust_leverage_size_signal(symbol, leverage, account=None):
    """Calculate position size based on 95% of balance with leverage"""
    try:
        # Get account balance
        balance_info = trader._make_request('/fapi/v2/balance', signed=True, futures=True)
        if not balance_info:
            return 0
        
        usdt_balance = 0
        for balance in balance_info:
            if balance['asset'] == 'USDT':
                usdt_balance = float(balance['availableBalance'])
                break
        
        # Get current price
        price_data = trader._make_request('/fapi/v1/ticker/price', {'symbol': symbol}, futures=True)
        if not price_data:
            return 0
        
        current_price = float(price_data['price'])
        
        # Calculate position size (95% of balance with leverage)
        position_value = usdt_balance * 0.95 * leverage
        quantity = position_value / current_price
        
        # Round to appropriate decimals
        sz_decimals, _ = get_sz_px_decimals(symbol)
        quantity = round(quantity, sz_decimals)
        
        print(f"📊 Position size for {symbol}: {quantity} (${position_value:.2f} value)")
        return quantity
        
    except Exception as e:
        print(f"Error calculating position size: {e}")
        return 0

def adjust_leverage_usd_size(symbol, usd_size, leverage, account=None):
    """Calculate position size based on specific USD amount with leverage"""
    try:
        # Get current price
        price_data = trader._make_request('/fapi/v1/ticker/price', {'symbol': symbol}, futures=True)
        if not price_data:
            return 0
        
        current_price = float(price_data['price'])
        
        # Calculate position size
        position_value = usd_size * leverage
        quantity = position_value / current_price
        
        # Round to appropriate decimals
        sz_decimals, _ = get_sz_px_decimals(symbol)
        quantity = round(quantity, sz_decimals)
        
        print(f"📊 Position size for {symbol}: {quantity} (${position_value:.2f} value)")
        return quantity
        
    except Exception as e:
        print(f"Error calculating USD position size: {e}")
        return 0

def adjust_leverage(symbol, leverage):
    """Set leverage for a symbol"""
    try:
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        
        result = trader._make_request('/fapi/v1/leverage', params, signed=True, futures=True, method='POST')
        
        if result:
            print(f"✅ Leverage set to {leverage}x for {symbol}")
            return result
        else:
            print(f"❌ Failed to set leverage for {symbol}")
            return None
            
    except Exception as e:
        print(f"Error setting leverage: {e}")
        return None

def get_current_price(symbol):
    """Fetch current price for a symbol"""
    try:
        data = trader._make_request('/api/v3/ticker/price', {'symbol': symbol})
        if data:
            return float(data['price'])
        return None
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def get_balance(account=None):
    """Get current USDT balance"""
    try:
        balance_info = trader._make_request('/fapi/v2/balance', signed=True, futures=True)
        if balance_info:
            for balance in balance_info:
                if balance['asset'] == 'USDT':
                    return float(balance['availableBalance'])
        return 0
    except Exception as e:
        print(f"Error getting balance: {e}")
        return 0

def connect():
    """Connect to Binance API - returns trader instance"""
    return {'exchange': trader, 'wallet': trader}
