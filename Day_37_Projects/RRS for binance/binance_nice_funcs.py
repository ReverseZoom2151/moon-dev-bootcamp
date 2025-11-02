# binance_nice_funcs.py - Binance Trading Utility Functions for RRS System
import requests
import time
import hmac
import hashlib
import pandas as pd
import logging
import os
import sys
from urllib.parse import urlencode

# Local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import dontshareconfig as d
except ImportError:
    logging.error("Could not import 'dontshareconfig.py' from parent directory.")
    sys.exit(1)

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# --- Binance API Configuration ---
BINANCE_API_KEY = getattr(d, 'binance_api_key', None)
BINANCE_SECRET_KEY = getattr(d, 'binance_secret_key', None)
BASE_URL = "https://api.binance.com"

if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    logger.error("Binance API credentials not found in dontshareconfig.py")
    sys.exit(1)

# --- Trading Configuration ---
DEFAULT_LEVERAGE = 10
MAX_LOSS = -1  # -1%
TARGET_PROFIT = 5  # 5%
DEFAULT_POS_SIZE = 200  # USDT
VOLUME_MULTIPLIER = 3
ROUNDING_PRECISION = 4

print('🚀 Binance RRS Trading Utilities Loaded')

def _get_server_time():
    """Get Binance server time for timestamp sync."""
    try:
        response = requests.get(f"{BASE_URL}/api/v3/time")
        return response.json()['serverTime']
    except Exception as e:
        logger.error(f"Failed to get server time: {e}")
        return int(time.time() * 1000)

def _create_signature(query_string: str, secret: str) -> str:
    """Create HMAC SHA256 signature for Binance API."""
    return hmac.new(
        secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def _authenticated_request(method: str, endpoint: str, params: dict = None) -> dict:
    """Make authenticated request to Binance API."""
    if params is None:
        params = {}
    
    # Add timestamp
    params['timestamp'] = _get_server_time()
    
    # Create query string
    query_string = urlencode(params)
    
    # Create signature
    signature = _create_signature(query_string, BINANCE_SECRET_KEY)
    params['signature'] = signature
    
    # Headers
    headers = {
        'X-MBX-APIKEY': BINANCE_API_KEY,
        'Content-Type': 'application/json'
    }
    
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == 'GET':
            response = requests.get(url, params=params, headers=headers)
        elif method.upper() == 'POST':
            response = requests.post(url, params=params, headers=headers)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, params=params, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Binance API request failed: {e}")
        return {}

def ask_bid(symbol: str) -> dict:
    """Get current bid/ask prices for a symbol."""
    try:
        symbol_formatted = symbol.upper() + 'USDT' if not symbol.endswith('USDT') else symbol.upper()
        response = requests.get(f"{BASE_URL}/api/v3/ticker/bookTicker", 
                               params={'symbol': symbol_formatted})
        response.raise_for_status()
        
        data = response.json()
        return {
            'symbol': symbol,
            'bid_price': float(data['bidPrice']),
            'bid_qty': float(data['bidQty']),
            'ask_price': float(data['askPrice']),
            'ask_qty': float(data['askQty']),
            'spread': float(data['askPrice']) - float(data['bidPrice']),
            'spread_pct': ((float(data['askPrice']) - float(data['bidPrice'])) / float(data['bidPrice'])) * 100
        }
    except Exception as e:
        logger.error(f"Failed to get bid/ask for {symbol}: {e}")
        return {}

def get_price(symbol: str) -> float:
    """Get current price for a symbol."""
    try:
        symbol_formatted = symbol.upper() + 'USDT' if not symbol.endswith('USDT') else symbol.upper()
        response = requests.get(f"{BASE_URL}/api/v3/ticker/price", 
                               params={'symbol': symbol_formatted})
        response.raise_for_status()
        return float(response.json()['price'])
    except Exception as e:
        logger.error(f"Failed to get price for {symbol}: {e}")
        return 0.0

def get_account_balance() -> dict:
    """Get account balance information."""
    try:
        result = _authenticated_request('GET', '/api/v3/account')
        if not result:
            return {}
        
        balances = {}
        for balance in result.get('balances', []):
            asset = balance['asset']
            free = float(balance['free'])
            locked = float(balance['locked'])
            total = free + locked
            
            if total > 0:  # Only include non-zero balances
                balances[asset] = {
                    'free': free,
                    'locked': locked,
                    'total': total
                }
        
        return balances
        
    except Exception as e:
        logger.error(f"Failed to get account balance: {e}")
        return {}

def get_position_size(symbol: str) -> dict:
    """Get current position size for a symbol (for futures)."""
    try:
        # For spot trading, check wallet balance
        balances = get_account_balance()
        base_asset = symbol.replace('USDT', '').upper()
        
        position_info = {
            'symbol': symbol,
            'size': 0.0,
            'side': 'none',
            'entry_price': 0.0,
            'unrealized_pnl': 0.0,
            'position_value_usdt': 0.0
        }
        
        if base_asset in balances:
            size = balances[base_asset]['total']
            if size > 0:
                current_price = get_price(symbol)
                position_info.update({
                    'size': size,
                    'side': 'long',
                    'position_value_usdt': size * current_price
                })
        
        return position_info
        
    except Exception as e:
        logger.error(f"Failed to get position size for {symbol}: {e}")
        return {}

def market_buy(symbol: str, usdt_amount: float) -> dict:
    """Execute a market buy order."""
    try:
        symbol_formatted = symbol.upper() + 'USDT' if not symbol.endswith('USDT') else symbol.upper()
        
        params = {
            'symbol': symbol_formatted,
            'side': 'BUY',
            'type': 'MARKET',
            'quoteOrderQty': round(usdt_amount, 2)  # Buy with USDT amount
        }
        
        result = _authenticated_request('POST', '/api/v3/order', params)
        
        if result:
            logger.info(f"Market buy executed: {symbol} for ${usdt_amount}")
            return {
                'success': True,
                'orderId': result.get('orderId'),
                'executedQty': float(result.get('executedQty', 0)),
                'cummulativeQuoteQty': float(result.get('cummulativeQuoteQty', 0)),
                'status': result.get('status'),
                'fills': result.get('fills', [])
            }
        else:
            return {'success': False, 'error': 'API request failed'}
            
    except Exception as e:
        logger.error(f"Market buy failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def market_sell(symbol: str, quantity: float = None) -> dict:
    """Execute a market sell order."""
    try:
        symbol_formatted = symbol.upper() + 'USDT' if not symbol.endswith('USDT') else symbol.upper()
        
        # If no quantity specified, sell all available balance
        if quantity is None:
            balances = get_account_balance()
            base_asset = symbol.replace('USDT', '').upper()
            if base_asset in balances:
                quantity = balances[base_asset]['free']
            else:
                logger.warning(f"No {base_asset} balance to sell")
                return {'success': False, 'error': 'No balance to sell'}
        
        params = {
            'symbol': symbol_formatted,
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': round(quantity, 6)
        }
        
        result = _authenticated_request('POST', '/api/v3/order', params)
        
        if result:
            logger.info(f"Market sell executed: {quantity} {symbol}")
            return {
                'success': True,
                'orderId': result.get('orderId'),
                'executedQty': float(result.get('executedQty', 0)),
                'cummulativeQuoteQty': float(result.get('cummulativeQuoteQty', 0)),
                'status': result.get('status'),
                'fills': result.get('fills', [])
            }
        else:
            return {'success': False, 'error': 'API request failed'}
            
    except Exception as e:
        logger.error(f"Market sell failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def limit_order(symbol: str, side: str, quantity: float, price: float) -> dict:
    """Place a limit order."""
    try:
        symbol_formatted = symbol.upper() + 'USDT' if not symbol.endswith('USDT') else symbol.upper()
        
        params = {
            'symbol': symbol_formatted,
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': round(quantity, 6),
            'price': round(price, 2)
        }
        
        result = _authenticated_request('POST', '/api/v3/order', params)
        
        if result:
            logger.info(f"Limit order placed: {side} {quantity} {symbol} @ ${price}")
            return {
                'success': True,
                'orderId': result.get('orderId'),
                'status': result.get('status'),
                'symbol': result.get('symbol'),
                'side': result.get('side'),
                'quantity': result.get('origQty'),
                'price': result.get('price')
            }
        else:
            return {'success': False, 'error': 'API request failed'}
            
    except Exception as e:
        logger.error(f"Limit order failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def cancel_all_orders(symbol: str = None) -> dict:
    """Cancel all open orders for a symbol or all symbols."""
    try:
        if symbol:
            symbol_formatted = symbol.upper() + 'USDT' if not symbol.endswith('USDT') else symbol.upper()
            params = {'symbol': symbol_formatted}
            result = _authenticated_request('DELETE', '/api/v3/openOrders', params)
        else:
            # Get all open orders and cancel them
            open_orders = _authenticated_request('GET', '/api/v3/openOrders')
            results = []
            for order in open_orders:
                cancel_result = _authenticated_request('DELETE', '/api/v3/order', 
                                                     {'symbol': order['symbol'], 'orderId': order['orderId']})
                results.append(cancel_result)
            result = {'cancelled_orders': len(results)}
        
        if result:
            logger.info(f"Orders cancelled for {symbol if symbol else 'all symbols'}")
            return {'success': True, 'result': result}
        else:
            return {'success': False, 'error': 'API request failed'}
            
    except Exception as e:
        logger.error(f"Cancel orders failed: {e}")
        return {'success': False, 'error': str(e)}

def get_ohlcv(symbol: str, interval: str = '15m', limit: int = 1000) -> pd.DataFrame:
    """Get OHLCV data for a symbol."""
    try:
        symbol_formatted = symbol.upper() + 'USDT' if not symbol.endswith('USDT') else symbol.upper()
        
        params = {
            'symbol': symbol_formatted,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(f"{BASE_URL}/api/v3/klines", params=params)
        response.raise_for_status()
        
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to proper types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to get OHLCV for {symbol}: {e}")
        return pd.DataFrame()

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Volume Weighted Average Price."""
    if df.empty:
        return df
    
    df = df.copy()
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df

def calculate_bollinger_bands(df: pd.DataFrame, length: int = 20, std_dev: float = 2) -> tuple:
    """Calculate Bollinger Bands and determine market conditions."""
    if df.empty or len(df) < length:
        return df, False, False
    
    df = df.copy()
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=length).mean()
    bb_std = df['close'].rolling(window=length).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Determine if bands are tight or wide
    current_width = df['bb_width'].iloc[-1]
    avg_width = df['bb_width'].rolling(window=50).mean().iloc[-1]
    
    tight = current_width < (avg_width * 0.8)  # 20% below average
    wide = current_width > (avg_width * 1.2)   # 20% above average
    
    return df, tight, wide

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Average True Range."""
    if df.empty or len(df) < window:
        return df
    
    df = df.copy()
    
    # True Range calculation
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # ATR is the moving average of True Range
    df['atr'] = df['true_range'].rolling(window=window).mean()
    
    # Clean up temporary columns
    df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
    
    return df

def volume_spike_detector(df: pd.DataFrame, multiplier: float = 2.0) -> bool:
    """Detect if current volume is significantly higher than average."""
    if df.empty or len(df) < 20:
        return False
    
    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
    
    return current_volume > (avg_volume * multiplier)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate RSI indicator."""
    if df.empty or len(df) < period:
        return df
    
    df = df.copy()
    delta = df['close'].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def kill_switch(symbol: str) -> dict:
    """Emergency close all positions for a symbol."""
    try:
        # Cancel all open orders first
        cancel_result = cancel_all_orders(symbol)
        
        # Then close position (sell all holdings)
        position = get_position_size(symbol)
        sell_result = {'success': True}
        
        if position.get('size', 0) > 0:
            sell_result = market_sell(symbol)
        
        logger.info(f"Kill switch activated for {symbol}")
        return {
            'success': True,
            'orders_cancelled': cancel_result.get('success', False),
            'position_closed': sell_result.get('success', False)
        }
        
    except Exception as e:
        logger.error(f"Kill switch failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def pnl_monitor(symbol: str, target_profit: float = TARGET_PROFIT, 
                max_loss: float = MAX_LOSS) -> dict:
    """Monitor PnL and close position if targets hit."""
    try:
        position = get_position_size(symbol)
        
        if position.get('size', 0) == 0:
            return {'action': 'none', 'reason': 'no_position'}
        
        current_price = get_price(symbol)
        entry_price = position.get('entry_price', current_price)  # Fallback to current price
        
        if entry_price <= 0:
            return {'action': 'none', 'reason': 'invalid_entry_price'}
        
        pnl_pct = ((current_price - entry_price) / entry_price) * 100
        
        if pnl_pct >= target_profit:
            sell_result = market_sell(symbol)
            logger.info(f"Profit target hit for {symbol}: {pnl_pct:.2f}%")
            return {'action': 'close_profit', 'pnl_pct': pnl_pct, 'result': sell_result}
        
        elif pnl_pct <= max_loss:
            sell_result = market_sell(symbol)
            logger.info(f"Stop loss hit for {symbol}: {pnl_pct:.2f}%")
            return {'action': 'close_loss', 'pnl_pct': pnl_pct, 'result': sell_result}
        
        else:
            return {'action': 'hold', 'pnl_pct': pnl_pct}
            
    except Exception as e:
        logger.error(f"PnL monitor failed for {symbol}: {e}")
        return {'action': 'error', 'error': str(e)}

def get_all_symbols() -> list:
    """Get all available trading symbols."""
    try:
        response = requests.get(f"{BASE_URL}/api/v3/exchangeInfo")
        response.raise_for_status()
        
        data = response.json()
        symbols = []
        
        for symbol_info in data['symbols']:
            if (symbol_info['status'] == 'TRADING' and 
                symbol_info['symbol'].endswith('USDT')):
                symbols.append(symbol_info['symbol'])
        
        return symbols
        
    except Exception as e:
        logger.error(f"Failed to get symbols: {e}")
        return []

def calculate_position_size(symbol: str, usdt_amount: float, leverage: int = 1) -> dict:
    """Calculate appropriate position size based on USDT amount."""
    try:
        current_price = get_price(symbol)
        if current_price <= 0:
            return {'error': 'Invalid price'}
        
        # For spot trading, leverage is effectively 1
        effective_amount = usdt_amount * leverage if leverage > 1 else usdt_amount
        quantity = effective_amount / current_price
        
        return {
            'symbol': symbol,
            'usdt_amount': usdt_amount,
            'current_price': current_price,
            'quantity': round(quantity, 6),
            'total_value': effective_amount,
            'leverage': leverage
        }
        
    except Exception as e:
        logger.error(f"Position size calculation failed for {symbol}: {e}")
        return {'error': str(e)}

def get_order_book(symbol: str, limit: int = 100) -> dict:
    """Get order book data."""
    try:
        symbol_formatted = symbol.upper() + 'USDT' if not symbol.endswith('USDT') else symbol.upper()
        
        params = {'symbol': symbol_formatted, 'limit': limit}
        response = requests.get(f"{BASE_URL}/api/v3/depth", params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to proper format
        bids = [[float(price), float(qty)] for price, qty in data['bids']]
        asks = [[float(price), float(qty)] for price, qty in data['asks']]
        
        # Calculate liquidity metrics
        bid_liquidity = sum(price * qty for price, qty in bids[:10])
        ask_liquidity = sum(price * qty for price, qty in asks[:10])
        spread = asks[0][0] - bids[0][0] if bids and asks else 0
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'bid_liquidity': bid_liquidity,
            'ask_liquidity': ask_liquidity,
            'spread': spread,
            'spread_pct': (spread / bids[0][0] * 100) if bids else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to get order book for {symbol}: {e}")
        return {}

# Test functions
def test_connection():
    """Test Binance API connection."""
    try:
        response = requests.get(f"{BASE_URL}/api/v3/ping")
        response.raise_for_status()
        logger.info("✅ Binance API connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Binance API connection failed: {e}")
        return False

if __name__ == "__main__":
    # Test the functions
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Binance nice_funcs...")
    
    # Test connection
    if test_connection():
        # Test basic functions
        btc_price = get_price('BTC')
        print(f"BTC Price: ${btc_price:,.2f}")
        
        # Test bid/ask
        btc_book = ask_bid('BTC')
        if btc_book:
            print(f"BTC Spread: ${btc_book['spread']:.2f} ({btc_book['spread_pct']:.3f}%)")
        
        # Test OHLCV
        btc_data = get_ohlcv('BTC', '1h', 100)
        if not btc_data.empty:
            print(f"BTC OHLCV data: {len(btc_data)} candles")
            
            # Test technical indicators
            btc_data = calculate_vwap(btc_data)
            btc_data = calculate_atr(btc_data)
            btc_data = calculate_rsi(btc_data)
            print(f"Current BTC RSI: {btc_data['rsi'].iloc[-1]:.2f}")
    
    print("✅ Binance nice_funcs test complete")
