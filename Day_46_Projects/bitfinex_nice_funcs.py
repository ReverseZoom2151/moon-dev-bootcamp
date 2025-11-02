"""
🌙 Moon Dev's Bitfinex Trading Functions
Built with love by Moon Dev 🚀

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.

Bitfinex version of nice_funcs.py with institutional-grade trading utilities for
margin trading, derivatives, position management, and professional risk controls.
"""

import os
import sys
import time
import json
import hmac
import hashlib
import requests
import pandas as pd
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Day_4_Projects.dontshareconfig import BITFINEX_API_KEY, BITFINEX_SECRET_KEY
except ImportError:
    BITFINEX_API_KEY = os.getenv('BITFINEX_API_KEY')
    BITFINEX_SECRET_KEY = os.getenv('BITFINEX_SECRET_KEY')

print('🌙 Bitfinex Professional Functions Loaded! 🚀')

# Configuration
BASE_URL = "https://api-pub.bitfinex.com"
AUTH_URL = "https://api.bitfinex.com"

# Global parameters
symbol = 'tBTCUSD'
timeframe = '15m'
max_loss = -1
target = 5
pos_size = 200
leverage = 10

def _generate_signature(url_path: str, nonce: str, body: str = "") -> str:
    """Generate HMAC SHA384 signature for authenticated requests"""
    if not BITFINEX_SECRET_KEY:
        raise ValueError("Secret key required")
    
    message = '/api/' + url_path + nonce + body
    signature = hmac.new(
        BITFINEX_SECRET_KEY.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha384
    ).hexdigest()
    return signature

def _make_request(endpoint: str, params: dict = None, signed: bool = False, method: str = 'GET') -> any:
    """Make authenticated request to Bitfinex API"""
    params = params or {}
    
    try:
        if signed:
            nonce = str(int(time.time() * 1000000))
            url_path = endpoint.lstrip('/')
            body = json.dumps(params) if params else ""
            
            signature = _generate_signature(url_path, nonce, body)
            
            headers = {
                'bfx-nonce': nonce,
                'bfx-apikey': BITFINEX_API_KEY,
                'bfx-signature': signature,
                'content-type': 'application/json'
            }
            
            url = f"{AUTH_URL}{endpoint}"
            response = requests.post(url, headers=headers, data=body, timeout=10)
        else:
            url = f"{BASE_URL}{endpoint}"
            response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return None

def ask_bid(symbol):
    """Get bid/ask prices for a symbol"""
    try:
        endpoint = f"/v2/ticker/{symbol}"
        data = _make_request(endpoint)
        
        if data and len(data) >= 10:
            # Bitfinex ticker format: [BID, BID_SIZE, ASK, ASK_SIZE, ...]
            bid = float(data[0])
            ask = float(data[2])
            return {
                'bid': bid,
                'ask': ask,
                'mid': (bid + ask) / 2,
                'spread': ask - bid,
                'spread_pct': ((ask - bid) / ask) * 100
            }
    except Exception as e:
        print(f"❌ Error getting bid/ask for {symbol}: {str(e)}")
        return None

def spot_price_and_symbol_info(symbol):
    """Get spot price and symbol information"""
    try:
        # Get current ticker
        ticker_data = ask_bid(symbol)
        if not ticker_data:
            return None
            
        # Extract base and quote assets
        if symbol.startswith('t'):
            symbol_clean = symbol[1:]  # Remove 't' prefix
            if len(symbol_clean) >= 6:
                base_asset = symbol_clean[:3]
                quote_asset = symbol_clean[3:]
            else:
                base_asset = symbol_clean[:3]
                quote_asset = 'USD'
        else:
            base_asset = symbol[:3]
            quote_asset = 'USD'
        
        return {
            'symbol': symbol,
            'price': ticker_data['mid'],
            'base_asset': base_asset,
            'quote_asset': quote_asset,
            'bid': ticker_data['bid'],
            'ask': ticker_data['ask'],
            'spread': ticker_data['spread']
        }
    except Exception as e:
        print(f"❌ Error getting symbol info for {symbol}: {str(e)}")
        return None

def margin_limit_order(symbol, side, amount, price, order_type='EXCHANGE LIMIT'):
    """Place a margin limit order"""
    try:
        if not BITFINEX_API_KEY or not BITFINEX_SECRET_KEY:
            print("❌ API credentials required for trading")
            return None
        
        params = {
            'type': order_type,
            'symbol': symbol,
            'amount': str(abs(float(amount)) if side.upper() == 'BUY' else -abs(float(amount))),
            'price': str(price)
        }
        
        response = _make_request("/v2/auth/w/order/submit", params, signed=True)
        if response:
            print(f"✅ {side} order placed: {abs(float(amount))} {symbol} @ ${price}")
            return response
        else:
            print(f"❌ Failed to place {side} order")
            return None
            
    except Exception as e:
        print(f"❌ Error placing margin order: {str(e)}")
        return None

def spot_limit_order(symbol, side, quantity, price):
    """Place a spot limit order (alias for margin order)"""
    return margin_limit_order(symbol, side, quantity, price, 'EXCHANGE LIMIT')

def limit_order(symbol, side, quantity, price, reduce_only=False):
    """Place a derivatives limit order"""
    try:
        order_type = 'LIMIT'
        if reduce_only:
            order_type = 'LIMIT'  # Bitfinex handles reduce-only through position sizing
            
        return margin_limit_order(symbol, side, quantity, price, order_type)
    except Exception as e:
        print(f"❌ Error placing derivatives order: {str(e)}")
        return None

def get_ohlcv(symbol, timeframe, limit=500):
    """Get OHLCV data for a symbol"""
    try:
        # Convert timeframe to Bitfinex format
        tf_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h',
            '4h': '4h', '6h': '6h', '12h': '12h', '1d': '1D', '1w': '7D'
        }
        
        bitfinex_tf = tf_map.get(timeframe, '15m')
        endpoint = f"/v2/candles/trade:{bitfinex_tf}:{symbol}/hist"
        params = {'limit': min(limit, 5000)}  # Bitfinex allows up to 5000
        
        data = _make_request(endpoint, params)
        
        if data:
            # Bitfinex format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
            
            # Convert timestamp and reorder columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()  # Ensure chronological order
            
            return df
    except Exception as e:
        print(f"❌ Error getting OHLCV data: {str(e)}")
        return pd.DataFrame()

def get_ohlcv2(symbol, interval, lookback_days=30):
    """Get OHLCV data with lookback period"""
    try:
        # Calculate approximate limit based on timeframe
        interval_hours = {
            '1m': 1/60, '5m': 5/60, '15m': 15/60, '30m': 0.5, '1h': 1,
            '4h': 4, '6h': 6, '12h': 12, '1d': 24, '1w': 168
        }
        
        if interval in interval_hours:
            hours_per_candle = interval_hours[interval]
            total_hours = lookback_days * 24
            limit = min(int(total_hours / hours_per_candle), 5000)
        else:
            limit = 1000
        
        return get_ohlcv(symbol, interval, limit)
    except Exception as e:
        print(f"❌ Error getting OHLCV2 data: {str(e)}")
        return pd.DataFrame()

def get_positions():
    """Get all active positions"""
    try:
        if not BITFINEX_API_KEY or not BITFINEX_SECRET_KEY:
            print("❌ API credentials required")
            return []
        
        positions = _make_request("/v2/auth/r/positions", signed=True)
        
        if positions:
            processed_positions = []
            for pos in positions:
                # Bitfinex position format: [SYMBOL, STATUS, AMOUNT, BASE_PRICE, ...]
                if len(pos) >= 4 and float(pos[2]) != 0:
                    processed_positions.append({
                        'symbol': pos[0],
                        'status': pos[1],
                        'position_amt': float(pos[2]),
                        'base_price': float(pos[3]),
                        'funding_cost': float(pos[4]) if len(pos) > 4 else 0,
                        'unrealized_pnl': float(pos[6]) if len(pos) > 6 else 0,
                        'side': 'LONG' if float(pos[2]) > 0 else 'SHORT',
                        'notional': abs(float(pos[2]) * float(pos[3]))
                    })
            return processed_positions
    except Exception as e:
        print(f"❌ Error getting positions: {str(e)}")
        return []

def get_position(symbol):
    """Get position for a specific symbol"""
    try:
        positions = get_positions()
        for pos in positions:
            if pos['symbol'] == symbol:
                return pos
        return None
    except Exception as e:
        print(f"❌ Error getting position for {symbol}: {str(e)}")
        return None

def kill_switch_mkt(symbol):
    """Close position with market order"""
    try:
        position = get_position(symbol)
        if not position or position['position_amt'] == 0:
            print(f"No position to close for {symbol}")
            return None
        
        # Calculate close amount (opposite of position)
        close_amount = -position['position_amt']  # Opposite sign
        
        params = {
            'type': 'MARKET',
            'symbol': symbol,
            'amount': str(close_amount)
        }
        
        response = _make_request("/v2/auth/w/order/submit", params, signed=True)
        if response:
            print(f"✅ Position closed with market order: {symbol}")
            return response
    except Exception as e:
        print(f"❌ Error closing position: {str(e)}")
        return None

def kill_switch(symbol):
    """Close position with limit order at favorable price"""
    try:
        position = get_position(symbol)
        if not position or position['position_amt'] == 0:
            return None
        
        # Get current prices
        prices = ask_bid(symbol)
        if not prices:
            return None
        
        # Use favorable price (bid for selling, ask for buying)
        close_amount = -position['position_amt']
        close_price = prices['ask'] if close_amount > 0 else prices['bid']
        
        params = {
            'type': 'EXCHANGE LIMIT',
            'symbol': symbol,
            'amount': str(close_amount),
            'price': str(close_price)
        }
        
        response = _make_request("/v2/auth/w/order/submit", params, signed=True)
        if response:
            print(f"✅ Position close order placed: {symbol}")
            return response
    except Exception as e:
        print(f"❌ Error placing close order: {str(e)}")
        return None

def get_balance():
    """Get account balance"""
    try:
        if not BITFINEX_API_KEY or not BITFINEX_SECRET_KEY:
            print("❌ API credentials required")
            return None
        
        wallets = _make_request("/v2/auth/r/wallets", signed=True)
        
        if wallets:
            total_usd = 0
            for wallet in wallets:
                # Format: [WALLET_TYPE, CURRENCY, BALANCE, ...]
                if len(wallet) >= 3:
                    currency = wallet[1]
                    balance = float(wallet[2])
                    
                    if currency == 'USD' or currency == 'UST':
                        total_usd += balance
                    elif currency == 'BTC' and balance > 0:
                        # Convert BTC to USD (approximate)
                        btc_price = spot_price_and_symbol_info('tBTCUSD')
                        if btc_price:
                            total_usd += balance * btc_price['price']
            
            return total_usd
    except Exception as e:
        print(f"❌ Error getting balance: {str(e)}")
        return None

def pnl_close(symbol, target_pnl, max_loss):
    """Close position based on PnL thresholds"""
    try:
        position = get_position(symbol)
        if not position or position['position_amt'] == 0:
            return None
        
        # Calculate PnL percentage
        current_price = ask_bid(symbol)
        if not current_price:
            return None
        
        entry_price = position['base_price']
        current_mid = current_price['mid']
        
        if position['side'] == 'LONG':
            pnl_pct = ((current_mid - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_mid) / entry_price) * 100
        
        if pnl_pct >= target_pnl:
            print(f"🎯 Target reached! PnL: {pnl_pct:.2f}% >= {target_pnl}%")
            return kill_switch_mkt(symbol)
        elif pnl_pct <= max_loss:
            print(f"🛑 Stop loss hit! PnL: {pnl_pct:.2f}% <= {max_loss}%")
            return kill_switch_mkt(symbol)
        else:
            print(f"📊 Current PnL: {pnl_pct:.2f}% (Target: {target_pnl}%, Stop: {max_loss}%)")
            return None
    except Exception as e:
        print(f"❌ Error checking PnL: {str(e)}")
        return None

def connect():
    """Test API connection"""
    try:
        # Test public endpoint
        data = _make_request("/v2/platform/status")
        if data and len(data) > 0:
            print("✅ Connected to Bitfinex API")
            return True
        else:
            print("❌ Failed to connect to Bitfinex API")
            return False
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

def cancel_all_orders():
    """Cancel all open orders"""
    try:
        if not BITFINEX_API_KEY or not BITFINEX_SECRET_KEY:
            print("❌ API credentials required")
            return False
        
        # Get all orders
        orders = _make_request("/v2/auth/r/orders", signed=True)
        
        if orders:
            for order in orders:
                order_id = order[0]
                params = {'id': order_id}
                _make_request("/v2/auth/w/order/cancel", params, signed=True)
        
        print("✅ All orders cancelled")
        return True
    except Exception as e:
        print(f"❌ Error cancelling orders: {str(e)}")
        return False

def close_all_positions_mkt():
    """Close all positions with market orders"""
    try:
        positions = get_positions()
        closed_positions = []
        
        for pos in positions:
            if pos['position_amt'] != 0:
                result = kill_switch_mkt(pos['symbol'])
                if result:
                    closed_positions.append(pos['symbol'])
                time.sleep(0.2)  # Rate limiting
        
        print(f"✅ Closed {len(closed_positions)} positions: {closed_positions}")
        return closed_positions
    except Exception as e:
        print(f"❌ Error closing all positions: {str(e)}")
        return []

def adjust_leverage(symbol, leverage_target):
    """Adjust leverage (Bitfinex uses margin ratio instead)"""
    try:
        print(f"📊 Bitfinex uses margin ratio instead of leverage")
        print(f"Target leverage {leverage_target}x = {100/leverage_target:.1f}% margin requirement")
        return True
    except Exception as e:
        print(f"❌ Error with leverage adjustment: {str(e)}")
        return False

# Technical analysis functions
def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    return sum(prices[-window:]) / window if len(prices) >= window else None

def calculate_atr(df, window=14):
    """Calculate ATR using pandas_ta"""
    if df.empty or len(df) < window:
        return None
    return ta.atr(df['high'], df['low'], df['close'], length=window)

def calculate_bollinger_bands(df, length=20, std_dev=2):
    """Calculate Bollinger Bands using pandas_ta"""
    if df.empty or len(df) < length:
        return None
    return ta.bbands(df['close'], length=length, std=std_dev)

def calculate_vwap_with_symbol(symbol, interval='15m', limit=100):
    """Calculate VWAP for a symbol"""
    try:
        df = get_ohlcv(symbol, interval, limit)
        if df.empty:
            return None
        
        # Calculate VWAP using pandas_ta
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        return vwap.iloc[-1] if not vwap.empty else None
    except Exception as e:
        print(f"❌ Error calculating VWAP: {str(e)}")
        return None

def get_funding_rate(symbol):
    """Get funding rate for derivatives"""
    try:
        endpoint = f"/v2/stats1/{symbol}/hist"
        params = {"key": "funding.close", "limit": 1}
        
        stats = _make_request(endpoint, params)
        
        if stats and len(stats) > 0:
            return {
                'symbol': symbol,
                'funding_rate': float(stats[0][1]),
                'funding_rate_pct': float(stats[0][1]) * 100,
                'timestamp': pd.to_datetime(stats[0][0], unit='ms')
            }
    except Exception as e:
        print(f"❌ Error getting funding rate: {str(e)}")
        return None

def get_open_interest(symbol):
    """Get open interest for derivatives"""
    try:
        endpoint = f"/v2/stats1/{symbol}/hist"
        params = {"key": "pos.size", "limit": 1}
        
        stats = _make_request(endpoint, params)
        
        if stats and len(stats) > 0:
            return {
                'symbol': symbol,
                'open_interest': abs(float(stats[0][1])),
                'timestamp': pd.to_datetime(stats[0][0], unit='ms')
            }
    except Exception as e:
        print(f"❌ Error getting open interest: {str(e)}")
        return None

def volume_spike(df, threshold=2.0):
    """Detect volume spikes"""
    try:
        if df.empty or len(df) < 20:
            return False
        
        avg_volume = df['volume'].rolling(20).mean()
        current_volume = df['volume'].iloc[-1]
        latest_avg = avg_volume.iloc[-1]
        
        return current_volume > (latest_avg * threshold)
    except Exception as e:
        print(f"❌ Error detecting volume spike: {str(e)}")
        return False
