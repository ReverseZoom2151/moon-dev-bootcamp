# Bitfinex Professional Trading Utilities - Institutional Grade Trading Functions
# Bitfinex equivalent of the Hyperliquid nice_funcs.py module with professional features

import requests
import json
import time
import pandas as pd
import pandas_ta as ta
import hmac
import hashlib
import warnings
import sys, os
from dontshareconfig import bitfinex_api_key, bitfinex_api_secret

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

print('Bitfinex Professional Trading Utilities loaded 🏛️')

# Professional Configuration
symbol = 'tBTCUSD'  # Bitfinex format with 't' prefix
timeframe = '1h'    # Professional timeframe
max_loss = -2.5     # Professional risk management
target = 8.0        # Institutional target
pos_size = 500      # Larger position sizes
leverage = 3        # Conservative institutional leverage
vol_multiplier = 2.5
rounding = 6        # Higher precision

API_KEY = bitfinex_api_key
API_SECRET = bitfinex_api_secret
BASE_URL = 'https://api-pub.bitfinex.com'
AUTH_URL = 'https://api.bitfinex.com'

def generate_professional_signature(path, nonce, body=''):
    """Generate HMAC SHA384 signature for Bitfinex professional API."""
    message = f'/api{path}{nonce}{body}'
    signature = hmac.new(
        API_SECRET.encode(),
        message.encode(),
        hashlib.sha384
    ).hexdigest()
    return signature

def make_professional_request(endpoint, params=None, method='GET', authenticated=True, version='v2'):
    """Make professional authenticated request to Bitfinex API."""
    try:
        if authenticated:
            url = f"{AUTH_URL}/api/{version}{endpoint}"
            nonce = str(int(time.time() * 1000000))
            
            if method == 'POST':
                body = json.dumps(params) if params else ''
                signature = generate_professional_signature(f"/{version}{endpoint}", nonce, body)
                
                headers = {
                    'bfx-nonce': nonce,
                    'bfx-apikey': API_KEY,
                    'bfx-signature': signature,
                    'content-type': 'application/json'
                }
                
                response = requests.post(url, headers=headers, data=body, timeout=30)
            else:
                query_string = ''
                if params:
                    query_string = '?' + '&'.join([f"{k}={v}" for k, v in params.items()])
                    url += query_string
                
                signature = generate_professional_signature(f"/{version}{endpoint}{query_string}", nonce)
                
                headers = {
                    'bfx-nonce': nonce,
                    'bfx-apikey': API_KEY,
                    'bfx-signature': signature
                }
                
                response = requests.get(url, headers=headers, timeout=30)
        else:
            # Public endpoint
            url = f"{BASE_URL}/{version}{endpoint}"
            if params:
                response = requests.get(url, params=params, timeout=30)
            else:
                response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("⚠️ Professional rate limit hit, implementing backoff...")
            time.sleep(60)
            return None
        else:
            print(f"Professional API Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"Professional request failed: {e}")
        return None

def ask_bid(symbol):
    """Get professional bid/ask prices for a symbol."""
    try:
        # Ensure symbol has 't' prefix for trading pairs
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        data = make_professional_request(f'/ticker/{symbol}', authenticated=False)
        if data and len(data) >= 7:
            # Bitfinex ticker format: [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, DAILY_CHANGE_PERC, LAST_PRICE, ...]
            bid = float(data[0])
            ask = float(data[2])
            last_price = float(data[6])
            
            return ask, bid, {
                'bid': bid,
                'ask': ask,
                'last': last_price,
                'spread': ask - bid,
                'spread_pct': ((ask - bid) / bid) * 100
            }
        return None, None, None
    except Exception as e:
        print(f"Professional error getting bid/ask for {symbol}: {e}")
        return None, None, None

def get_current_price(symbol):
    """Fetch current professional price for a symbol."""
    try:
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        data = make_professional_request(f'/ticker/{symbol}', authenticated=False)
        if data and len(data) >= 7:
            return float(data[6])  # Last price
        return None
    except Exception as e:
        print(f"Professional error getting price for {symbol}: {e}")
        return None

def get_professional_symbol_info(symbol):
    """Get professional symbol trading information."""
    try:
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        # Get symbol details from pairs endpoint
        data = make_professional_request('/conf/pub:info:pair', authenticated=False)
        if data and isinstance(data, list):
            for pair_info in data:
                if isinstance(pair_info, list) and len(pair_info) >= 2:
                    if pair_info[0] == symbol[1:]:  # Remove 't' prefix for comparison
                        return {
                            'symbol': symbol,
                            'price_precision': 5,  # Bitfinex default
                            'amount_precision': 8,
                            'min_order_size': pair_info[1] if len(pair_info) > 1 else 0.001,
                            'status': 'TRADING'
                        }
        
        # Default values if not found
        return {
            'symbol': symbol,
            'price_precision': 5,
            'amount_precision': 8,
            'min_order_size': 0.001,
            'status': 'TRADING'
        }
    except Exception as e:
        print(f"Professional error getting symbol info: {e}")
        return None

def professional_limit_order(symbol, side, amount, price, flags=0):
    """Place a professional limit order on Bitfinex."""
    try:
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        # Bitfinex uses positive amounts for buy, negative for sell
        if side.upper() == 'SELL':
            amount = -abs(float(amount))
        else:
            amount = abs(float(amount))
        
        order_data = [
            0,  # Order ID (0 for new orders)
            symbol,
            amount,
            price,
            'EXCHANGE LIMIT',  # Order type
            flags  # Order flags
        ]
        
        response = make_professional_request('/auth/w/order/submit', order_data, method='POST')
        return response
    except Exception as e:
        print(f"Professional error placing limit order: {e}")
        return None

def professional_market_order(symbol, side, amount):
    """Place a professional market order on Bitfinex."""
    try:
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        # Bitfinex uses positive amounts for buy, negative for sell
        if side.upper() == 'SELL':
            amount = -abs(float(amount))
        else:
            amount = abs(float(amount))
        
        order_data = [
            0,  # Order ID
            symbol,
            amount,
            0,  # Price (0 for market orders)
            'EXCHANGE MARKET'  # Order type
        ]
        
        response = make_professional_request('/auth/w/order/submit', order_data, method='POST')
        return response
    except Exception as e:
        print(f"Professional error placing market order: {e}")
        return None

def get_professional_account_balance():
    """Get professional account balance with institutional details."""
    try:
        data = make_professional_request('/auth/r/wallets')
        if data:
            balances = {}
            total_usd_value = 0
            
            for wallet in data:
                if len(wallet) >= 5:
                    wallet_type = wallet[0]  # exchange, margin, funding
                    currency = wallet[1]
                    balance = float(wallet[2])
                    unsettled = float(wallet[3]) if wallet[3] else 0
                    available = float(wallet[4]) if wallet[4] else balance
                    
                    if balance != 0 or available != 0:
                        balances[f"{currency}_{wallet_type}"] = {
                            'currency': currency,
                            'wallet_type': wallet_type,
                            'balance': balance,
                            'available': available,
                            'unsettled': unsettled
                        }
                        
                        # Estimate USD value for major currencies
                        if currency == 'USD':
                            total_usd_value += balance
                        elif currency == 'BTC':
                            btc_price = get_current_price('tBTCUSD')
                            if btc_price:
                                total_usd_value += balance * btc_price
            
            return {
                'balances': balances,
                'total_usd_value': total_usd_value,
                'margin_available': any(w['wallet_type'] == 'margin' for w in balances.values()),
                'professional_grade': 'institutional' if total_usd_value > 100000 else 'standard'
            }
        return {}
    except Exception as e:
        print(f"Professional error getting account balance: {e}")
        return {}

def get_professional_balance(currency='USD', wallet_type='exchange'):
    """Get professional balance for specific currency and wallet."""
    try:
        account_data = get_professional_account_balance()
        key = f"{currency}_{wallet_type}"
        if 'balances' in account_data and key in account_data['balances']:
            return account_data['balances'][key]['available']
        return 0.0
    except Exception as e:
        print(f"Professional error getting {currency} balance: {e}")
        return 0.0

def get_professional_open_orders(symbol=None):
    """Get professional open orders."""
    try:
        if symbol and not symbol.startswith('t'):
            symbol = 't' + symbol
        
        data = make_professional_request('/auth/r/orders')
        if data:
            orders = []
            for order in data:
                if len(order) >= 17:
                    order_symbol = order[3]
                    if symbol is None or order_symbol == symbol:
                        orders.append({
                            'id': order[0],
                            'gid': order[1],
                            'cid': order[2],
                            'symbol': order_symbol,
                            'created': order[4],
                            'updated': order[5],
                            'amount': order[6],
                            'amount_orig': order[7],
                            'type': order[8],
                            'status': order[13],
                            'price': order[16] if order[16] else 0
                        })
            return orders
        return []
    except Exception as e:
        print(f"Professional error getting open orders: {e}")
        return []

def cancel_professional_order(order_id):
    """Cancel a professional order."""
    try:
        order_data = [order_id]
        response = make_professional_request('/auth/w/order/cancel', order_data, method='POST')
        return response
    except Exception as e:
        print(f"Professional error canceling order: {e}")
        return None

def cancel_all_professional_orders(symbol=None):
    """Cancel all professional orders."""
    try:
        open_orders = get_professional_open_orders(symbol)
        results = []
        
        for order in open_orders:
            result = cancel_professional_order(order['id'])
            if result:
                results.append(result)
        
        print(f"🏛️ Professional orders canceled: {len(results)}")
        return results
    except Exception as e:
        print(f"Professional error canceling all orders: {e}")
        return []

def get_professional_ohlcv(symbol, timeframe='1h', limit=500):
    """Get professional OHLCV data with institutional analysis."""
    try:
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        # Map timeframes to Bitfinex format
        tf_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '3h': '3h', '6h': '6h', '12h': '12h',
            '1d': '1D', '1w': '7D', '2w': '14D', '1M': '1M'
        }
        
        bf_timeframe = tf_map.get(timeframe, '1h')
        endpoint = f'/candles/trade:{bf_timeframe}:{symbol}/hist'
        
        params = {
            'limit': min(limit, 5000),  # Bitfinex max
            'sort': -1  # Newest first
        }
        
        data = make_professional_request(endpoint, params, authenticated=False)
        if data:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'close', 'high', 'low', 'volume'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert to proper types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Professional error getting OHLCV data: {e}")
        return pd.DataFrame()

def process_professional_data_to_df(df, time_period=20):
    """Process professional data with institutional indicators."""
    try:
        if df.empty or len(df) < time_period:
            return df
        
        # Professional technical indicators
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        
        # Professional momentum indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
        
        # Professional volatility indicators
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['bb_upper'] = ta.bbands(df['close'])['BBU_20_2.0']
        df['bb_middle'] = ta.bbands(df['close'])['BBM_20_2.0']
        df['bb_lower'] = ta.bbands(df['close'])['BBL_20_2.0']
        
        # Professional volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Professional trend indicators
        df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
        
        return df
    except Exception as e:
        print(f"Professional error processing data: {e}")
        return df

def calculate_professional_vwap(symbol, timeframe='1h', limit=100):
    """Calculate professional VWAP with institutional precision."""
    try:
        df = get_professional_ohlcv(symbol, timeframe, limit)
        if not df.empty:
            df = process_professional_data_to_df(df)
            return df['vwap'].iloc[-1] if not pd.isna(df['vwap'].iloc[-1]) else None
        return None
    except Exception as e:
        print(f"Professional error calculating VWAP: {e}")
        return None

def professional_kill_switch(symbol):
    """Professional emergency close with institutional controls."""
    try:
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        print(f"🚨 Professional kill switch activated for {symbol}")
        
        # Cancel all orders first
        cancel_all_professional_orders(symbol)
        
        # Get positions
        positions = make_professional_request('/auth/r/positions')
        closed_positions = []
        
        if positions:
            for pos in positions:
                if len(pos) >= 4 and pos[0] == symbol:
                    amount = float(pos[2])  # Position size
                    if abs(amount) > 0:
                        # Close position with market order
                        close_amount = -amount  # Opposite sign to close
                        result = professional_market_order(symbol, 'SELL' if amount > 0 else 'BUY', abs(close_amount))
                        if result:
                            closed_positions.append({
                                'symbol': symbol,
                                'amount': amount,
                                'close_result': result
                            })
        
        print(f"🏛️ Professional kill switch completed: {len(closed_positions)} positions closed")
        return closed_positions
    except Exception as e:
        print(f"Professional kill switch error: {e}")
        return []

def get_professional_funding_rate(symbol=None):
    """Get professional funding rates for margin/derivatives."""
    try:
        if symbol and not symbol.startswith('t'):
            symbol = 't' + symbol
        
        endpoint = '/stats1/funding'
        if symbol:
            endpoint += f':{symbol}'
        endpoint += '/hist'
        
        data = make_professional_request(endpoint, authenticated=False)
        if data:
            funding_data = []
            for item in data[:10]:  # Last 10 periods
                if len(item) >= 4:
                    funding_data.append({
                        'timestamp': pd.to_datetime(item[0], unit='ms'),
                        'funding_rate': float(item[3]),
                        'funding_rate_pct': float(item[3]) * 100
                    })
            return funding_data
        return []
    except Exception as e:
        print(f"Professional error getting funding rate: {e}")
        return []

def get_professional_order_book(symbol, precision='P0', length=25):
    """Get professional order book with institutional depth."""
    try:
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        endpoint = f'/book/{symbol}/{precision}'
        params = {'len': min(length, 100)}
        
        data = make_professional_request(endpoint, params, authenticated=False)
        if data:
            bids = []
            asks = []
            
            for item in data:
                if len(item) >= 3:
                    price = float(item[0])
                    count = item[1]
                    amount = float(item[2])
                    
                    if amount > 0:
                        bids.append([price, amount, count])
                    else:
                        asks.append([price, abs(amount), count])
            
            # Sort bids (highest first) and asks (lowest first)
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])
            
            # Find largest sizes
            max_bid = max(bids, key=lambda x: x[1]) if bids else [0, 0, 0]
            max_ask = max(asks, key=lambda x: x[1]) if asks else [0, 0, 0]
            
            return {
                'bids': bids,
                'asks': asks,
                'max_bid_price': max_bid[0],
                'max_ask_price': max_ask[0],
                'max_bid_size': max_bid[1],
                'max_ask_size': max_ask[1],
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0,
                'mid_price': (asks[0][0] + bids[0][0]) / 2 if bids and asks else 0
            }
        return None
    except Exception as e:
        print(f"Professional error getting order book: {e}")
        return None

def professional_position_sizing(symbol, usd_amount, leverage=1, risk_pct=2.0):
    """Professional position sizing with institutional risk management."""
    try:
        current_price = get_current_price(symbol)
        if not current_price:
            return 0
        
        # Professional risk-adjusted position sizing
        account_data = get_professional_account_balance()
        total_balance = account_data.get('total_usd_value', 0)
        
        # Risk management: don't risk more than risk_pct of portfolio
        max_risk_amount = total_balance * (risk_pct / 100)
        position_amount = min(usd_amount, max_risk_amount)
        
        # Calculate quantity
        quantity = (position_amount * leverage) / current_price
        
        # Round to appropriate precision
        symbol_info = get_professional_symbol_info(symbol)
        if symbol_info:
            precision = symbol_info['amount_precision']
            quantity = round(quantity, precision)
        
        return quantity
    except Exception as e:
        print(f"Professional error calculating position size: {e}")
        return 0

def professional_pnl_monitor(symbol, target_profit=8.0, max_loss=-2.5):
    """Professional PnL monitoring with institutional controls."""
    try:
        positions = make_professional_request('/auth/r/positions')
        if not positions:
            return {'action': 'no_positions'}
        
        for pos in positions:
            if len(pos) >= 8 and pos[0] == symbol:
                unrealized_pnl_pct = float(pos[6]) * 100 if pos[6] else 0
                
                if unrealized_pnl_pct >= target_profit:
                    return {
                        'action': 'close_profit',
                        'pnl_pct': unrealized_pnl_pct,
                        'professional_grade': 'institutional_profit_taking'
                    }
                elif unrealized_pnl_pct <= max_loss:
                    return {
                        'action': 'close_loss',
                        'pnl_pct': unrealized_pnl_pct,
                        'professional_grade': 'institutional_risk_management'
                    }
        
        return {'action': 'hold', 'professional_grade': 'within_parameters'}
    except Exception as e:
        print(f"Professional PnL monitoring error: {e}")
        return {'action': 'error'}

def connect_professional():
    """Professional connection test with institutional validation."""
    try:
        # Test public endpoint
        public_data = make_professional_request('/platform/status', authenticated=False)
        if not public_data or public_data[0] != 1:
            return {'status': 'failed', 'reason': 'platform_maintenance'}
        
        # Test private endpoint
        balance_data = get_professional_account_balance()
        if not balance_data:
            return {'status': 'failed', 'reason': 'authentication_failed'}
        
        account_grade = balance_data.get('professional_grade', 'standard')
        total_value = balance_data.get('total_usd_value', 0)
        
        print(f"✅ Professional Bitfinex connection established")
        print(f"🏛️ Account Grade: {account_grade.title()}")
        print(f"💰 Total Portfolio Value: ${total_value:,.2f}")
        
        return {
            'status': 'connected',
            'account_grade': account_grade,
            'portfolio_value': total_value,
            'margin_available': balance_data.get('margin_available', False)
        }
    except Exception as e:
        print(f"❌ Professional connection failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def test_professional_connection():
    """Test professional API connection and permissions."""
    try:
        result = connect_professional()
        return result['status'] == 'connected'
    except Exception as e:
        print(f"❌ Professional connection test failed: {e}")
        return False

print("🏛️ Bitfinex Professional Trading Utilities loaded successfully!")
