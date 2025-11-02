# bitfinex_nice_funcs.py - Bitfinex Professional Trading Utility Functions
import requests
import json
import time
import hmac
import hashlib
import base64
import pandas as pd
import logging
import os
import sys

# Local imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import dontshareconfig as d
except ImportError:
    logging.error("Could not import 'dontshareconfig.py' from parent directory.")
    sys.exit(1)

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# --- Bitfinex Professional Configuration ---
BITFINEX_API_KEY = getattr(d, 'bitfinex_api_key', None)
BITFINEX_SECRET_KEY = getattr(d, 'bitfinex_secret_key', None)
BASE_URL = "https://api.bitfinex.com"

if not BITFINEX_API_KEY or not BITFINEX_SECRET_KEY:
    logger.error("Bitfinex API credentials not found in dontshareconfig.py")
    sys.exit(1)

# Professional trading parameters
DEFAULT_LEVERAGE = 10
MAX_LOSS = -2
TARGET_PROFIT = 8
DEFAULT_POS_SIZE = 500
VOLUME_MULTIPLIER = 2.5

print('🏛️ Bitfinex Professional RRS Trading Utilities Loaded')

def _get_nonce() -> str:
    """Generate nonce for authentication."""
    return str(int(time.time() * 1000000))

def _create_signature(path: str, nonce: str, body: str = "") -> str:
    """Create HMAC SHA384 signature."""
    message = f"/api/{path}{nonce}{body}"
    return hmac.new(
        BITFINEX_SECRET_KEY.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha384
    ).hexdigest()

def _authenticated_request(method: str, endpoint: str, params: dict = None) -> dict:
    """Make authenticated request to Bitfinex API."""
    if params is None:
        params = {}
    
    nonce = _get_nonce()
    path = endpoint.replace('/v1/', '').replace('/v2/', '')
    body = json.dumps(params) if params else ""
    
    headers = {
        'X-BFX-APIKEY': BITFINEX_API_KEY,
        'X-BFX-PAYLOAD': base64.b64encode(body.encode('utf-8')).decode('utf-8'),
        'X-BFX-SIGNATURE': _create_signature(path, nonce, body),
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Bitfinex API request failed: {e}")
        return {}

def ask_bid(symbol: str) -> dict:
    """Get professional bid/ask prices."""
    try:
        symbol_formatted = f"t{symbol.upper()}"
        response = requests.get(f"{BASE_URL}/v2/ticker/{symbol_formatted}", timeout=10)
        response.raise_for_status()
        
        ticker = response.json()
        if len(ticker) >= 10:
            return {
                'symbol': symbol,
                'bid_price': float(ticker[0]),
                'bid_size': float(ticker[1]),
                'ask_price': float(ticker[2]),
                'ask_size': float(ticker[3]),
                'spread': float(ticker[2]) - float(ticker[0]),
                'spread_pct': ((float(ticker[2]) - float(ticker[0])) / float(ticker[0])) * 100
            }
    except Exception as e:
        logger.error(f"Failed to get bid/ask for {symbol}: {e}")
        return {}

def get_price(symbol: str) -> float:
    """Get current price."""
    try:
        symbol_formatted = f"t{symbol.upper()}"
        response = requests.get(f"{BASE_URL}/v2/ticker/{symbol_formatted}", timeout=10)
        response.raise_for_status()
        ticker = response.json()
        return float(ticker[6]) if len(ticker) >= 7 else 0.0
    except Exception as e:
        logger.error(f"Failed to get price for {symbol}: {e}")
        return 0.0

def get_account_balance() -> dict:
    """Get account balance."""
    try:
        wallets = _authenticated_request('POST', '/v2/auth/r/wallets')
        if not wallets:
            return {}
        
        balances = {}
        for wallet in wallets:
            if len(wallet) >= 4 and wallet[2] != 0:
                currency = wallet[1]
                balance = float(wallet[2])
                available = float(wallet[4]) if len(wallet) > 4 else balance
                balances[currency] = {
                    'balance': balance,
                    'available': available,
                    'wallet_type': wallet[0]
                }
        return balances
    except Exception as e:
        logger.error(f"Failed to get account balance: {e}")
        return {}

def get_position_size(symbol: str) -> dict:
    """Get position size."""
    try:
        positions = _authenticated_request('POST', '/v2/auth/r/positions')
        if not positions:
            return {'symbol': symbol, 'size': 0.0, 'side': 'none'}
        
        symbol_formatted = f"t{symbol.upper()}"
        for position in positions:
            if len(position) >= 8 and position[0] == symbol_formatted:
                size = float(position[2])
                return {
                    'symbol': symbol,
                    'size': size,
                    'side': 'long' if size > 0 else 'short' if size < 0 else 'none',
                    'entry_price': float(position[3]),
                    'unrealized_pnl': float(position[6]),
                    'unrealized_pnl_pct': float(position[7])
                }
        
        return {'symbol': symbol, 'size': 0.0, 'side': 'none'}
    except Exception as e:
        logger.error(f"Failed to get position size for {symbol}: {e}")
        return {}

def market_buy(symbol: str, usd_amount: float) -> dict:
    """Execute market buy order."""
    try:
        symbol_formatted = f"t{symbol.upper()}"
        current_price = get_price(symbol)
        if current_price <= 0:
            return {'success': False, 'error': 'Invalid price'}
        
        quantity = usd_amount / current_price
        
        order_params = {
            'type': 'EXCHANGE MARKET',
            'symbol': symbol_formatted,
            'amount': str(quantity),
            'price': '1'
        }
        
        result = _authenticated_request('POST', '/v2/auth/w/order/submit', order_params)
        
        if result and len(result) > 0:
            logger.info(f"Market buy executed: {symbol} for ${usd_amount}")
            return {'success': True, 'orderId': result[0][0], 'amount': quantity}
        return {'success': False, 'error': 'Order submission failed'}
    except Exception as e:
        logger.error(f"Market buy failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def market_sell(symbol: str, quantity: float = None) -> dict:
    """Execute market sell order."""
    try:
        symbol_formatted = f"t{symbol.upper()}"
        
        if quantity is None:
            position = get_position_size(symbol)
            quantity = position.get('size', 0)
            if quantity <= 0:
                return {'success': False, 'error': 'No position to sell'}
        
        order_params = {
            'type': 'EXCHANGE MARKET',
            'symbol': symbol_formatted,
            'amount': str(-abs(quantity)),
            'price': '1'
        }
        
        result = _authenticated_request('POST', '/v2/auth/w/order/submit', order_params)
        
        if result and len(result) > 0:
            logger.info(f"Market sell executed: {quantity} {symbol}")
            return {'success': True, 'orderId': result[0][0], 'amount': quantity}
        return {'success': False, 'error': 'Sell order failed'}
    except Exception as e:
        logger.error(f"Market sell failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def limit_order(symbol: str, side: str, quantity: float, price: float) -> dict:
    """Place limit order."""
    try:
        symbol_formatted = f"t{symbol.upper()}"
        signed_quantity = quantity if side.upper() == 'BUY' else -quantity
        
        order_params = {
            'type': 'EXCHANGE LIMIT',
            'symbol': symbol_formatted,
            'amount': str(signed_quantity),
            'price': str(price)
        }
        
        result = _authenticated_request('POST', '/v2/auth/w/order/submit', order_params)
        
        if result and len(result) > 0:
            logger.info(f"Limit order placed: {side} {quantity} {symbol} @ ${price}")
            return {'success': True, 'orderId': result[0][0]}
        return {'success': False, 'error': 'Limit order failed'}
    except Exception as e:
        logger.error(f"Limit order failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def cancel_all_orders(symbol: str = None) -> dict:
    """Cancel orders."""
    try:
        if symbol:
            cancel_params = {'symbol': f"t{symbol.upper()}"}
            result = _authenticated_request('POST', '/v2/auth/w/order/cancel/multi', cancel_params)
        else:
            result = _authenticated_request('POST', '/v2/auth/w/order/cancel/all')
        
        if result:
            logger.info(f"Orders cancelled for {symbol if symbol else 'all symbols'}")
            return {'success': True, 'cancelled_orders': len(result) if isinstance(result, list) else 1}
        return {'success': False, 'error': 'Cancel request failed'}
    except Exception as e:
        logger.error(f"Cancel orders failed: {e}")
        return {'success': False, 'error': str(e)}

def get_ohlcv(symbol: str, timeframe: str = '15m', limit: int = 1000) -> pd.DataFrame:
    """Get OHLCV data."""
    try:
        symbol_formatted = f"t{symbol.upper()}"
        tf_mapping = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1D'}
        interval = tf_mapping.get(timeframe, '15m')
        
        url = f"{BASE_URL}/v2/candles/trade:{interval}:{symbol_formatted}/hist"
        response = requests.get(url, params={'limit': limit, 'sort': 1}, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return pd.DataFrame()
        
        df_data = []
        for candle in data:
            if len(candle) >= 6:
                df_data.append({
                    'timestamp': pd.to_datetime(candle[0], unit='ms'),
                    'open': float(candle[1]),
                    'close': float(candle[2]),
                    'high': float(candle[3]),
                    'low': float(candle[4]),
                    'volume': float(candle[5])
                })
        
        df = pd.DataFrame(df_data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Failed to get OHLCV for {symbol}: {e}")
        return pd.DataFrame()

def kill_switch(symbol: str) -> dict:
    """Emergency position closure."""
    try:
        cancel_result = cancel_all_orders(symbol)
        position = get_position_size(symbol)
        close_result = {'success': True}
        
        if abs(position.get('size', 0)) > 0:
            close_result = market_sell(symbol, abs(position['size']))
        
        logger.warning(f"Kill switch activated for {symbol}")
        return {
            'success': True,
            'orders_cancelled': cancel_result.get('success', False),
            'position_closed': close_result.get('success', False)
        }
    except Exception as e:
        logger.error(f"Kill switch failed for {symbol}: {e}")
        return {'success': False, 'error': str(e)}

def pnl_monitor(symbol: str, target_profit: float = TARGET_PROFIT, max_loss: float = MAX_LOSS) -> dict:
    """Monitor PnL and close positions."""
    try:
        position = get_position_size(symbol)
        if abs(position.get('size', 0)) == 0:
            return {'action': 'none', 'reason': 'no_position'}
        
        pnl_pct = position.get('unrealized_pnl_pct', 0)
        
        if pnl_pct >= target_profit:
            close_result = market_sell(symbol)
            logger.info(f"Profit target hit for {symbol}: {pnl_pct:.2f}%")
            return {'action': 'close_profit', 'pnl_pct': pnl_pct, 'result': close_result}
        
        elif pnl_pct <= max_loss:
            close_result = market_sell(symbol)
            logger.warning(f"Stop loss hit for {symbol}: {pnl_pct:.2f}%")
            return {'action': 'close_loss', 'pnl_pct': pnl_pct, 'result': close_result}
        
        return {'action': 'hold', 'pnl_pct': pnl_pct}
    except Exception as e:
        logger.error(f"PnL monitor failed for {symbol}: {e}")
        return {'action': 'error', 'error': str(e)}

def test_connection():
    """Test Bitfinex API connection."""
    try:
        response = requests.get(f"{BASE_URL}/v2/platform/status")
        response.raise_for_status()
        data = response.json()
        if data[0] == 1:  # 1 = operative, 0 = maintenance
            logger.info("✅ Bitfinex API connection successful")
            return True
        else:
            logger.warning("⚠️ Bitfinex in maintenance mode")
            return False
    except Exception as e:
        logger.error(f"❌ Bitfinex API connection failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🧪 Testing Bitfinex professional nice_funcs...")
    
    if test_connection():
        btc_price = get_price('BTCUSD')
        print(f"BTC Price: ${btc_price:,.2f}")
        
        btc_book = ask_bid('BTCUSD')
        if btc_book:
            print(f"BTC Spread: ${btc_book['spread']:.2f} ({btc_book['spread_pct']:.3f}%)")
        
        btc_data = get_ohlcv('BTCUSD', '1h', 100)
        if not btc_data.empty:
            print(f"BTC OHLCV data: {len(btc_data)} candles")
    
    print("✅ Bitfinex professional nice_funcs test complete")
