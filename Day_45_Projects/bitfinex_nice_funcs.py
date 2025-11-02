"""
🏛️ Bitfinex Professional Trading Utilities - Moon Dev Style  
Institutional-grade Bitfinex API wrapper with advanced trading functions

Built with love by Moon Dev 🌙 ✨
Disclaimer: This is not financial advice. Use at your own risk.
"""

import os
import sys
import time
import hmac
import hashlib
import json
import requests
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dontshareconfig import bitfinex_api_key, bitfinex_api_secret
except ImportError:
    bitfinex_api_key = os.getenv('BITFINEX_API_KEY')
    bitfinex_api_secret = os.getenv('BITFINEX_SECRET_KEY')

print('🏛️ Bitfinex Professional Nice Funcs Loaded!')

# Global configuration
symbol = 'tBTCUSD'
timeframe = '15m'
max_loss = -1
target = 5
pos_size = 200
leverage = 10
vol_multiplier = 3
rounding = 4

class BitfinexTrader:
    def __init__(self):
        self.api_key = bitfinex_api_key
        self.api_secret = bitfinex_api_secret
        self.base_url = "https://api-pub.bitfinex.com/v2"
        self.auth_url = "https://api.bitfinex.com/v2/auth"
        self.rate_limit_delay = 0.2  # More conservative for Bitfinex
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Bitfinex API keys not found!")
    
    def _generate_signature(self, path, nonce, body=""):
        """Generate HMAC SHA384 signature for Bitfinex API"""
        message = f'/v2/auth{path}{nonce}{body}'
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha384
        ).hexdigest()
        return signature
    
    def _make_request(self, endpoint, params=None, authenticated=False, method='GET'):
        """Make authenticated request to Bitfinex API"""
        if params is None:
            params = {}
        
        if authenticated:
            url = f"{self.auth_url}{endpoint}"
            nonce = str(int(time.time() * 1000000))
            body = json.dumps(params) if params else ""
            
            signature = self._generate_signature(endpoint, nonce, body)
            
            headers = {
                'bfx-nonce': nonce,
                'bfx-apikey': self.api_key,
                'bfx-signature': signature,
                'content-type': 'application/json'
            }
            
            time.sleep(self.rate_limit_delay)
            
            if method == 'POST':
                response = requests.post(url, data=body, headers=headers)
            else:
                response = requests.get(url, headers=headers)
        else:
            url = f"{self.base_url}{endpoint}"
            time.sleep(self.rate_limit_delay)
            response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return None

# Initialize global trader instance
trader = BitfinexTrader()

def ask_bid(symbol):
    """Get current bid and ask prices from order book"""
    try:
        data = trader._make_request(f'/book/{symbol}/P0', {'len': '25'})
        if data:
            bids = [order for order in data if float(order[2]) > 0]
            asks = [order for order in data if float(order[2]) < 0]
            
            bid = float(bids[0][0]) if bids else 0
            ask = float(asks[0][0]) if asks else 0
            
            return ask, bid, {'bids': bids, 'asks': asks}
        return None, None, None
    except Exception as e:
        print(f"Error getting bid/ask for {symbol}: {e}")
        return None, None, None

def spot_price_and_hoe_ass_spot_symbol(symbol):
    """Get current price and symbol info for Bitfinex"""
    try:
        # Get current ticker
        ticker_data = trader._make_request(f'/ticker/{symbol}')
        if not ticker_data:
            return f"Symbol {symbol} not found."
        
        mid_px = float(ticker_data[6])  # Last price
        
        # Bitfinex typically uses consistent decimal places
        # Most pairs have 5 decimal places for price, varying for size
        px_decimals = 5  # Standard for most pairs
        sz_decimals = 8  # Standard for size
        
        # Adjust decimals based on symbol
        if 'USD' in symbol:
            px_decimals = 5 if symbol.startswith('t') else 2
            sz_decimals = 8
        
        return mid_px, symbol, sz_decimals, px_decimals
        
    except Exception as e:
        print(f"Error getting symbol info: {e}")
        return f"Error getting symbol info: {e}"

def spot_limit_order(coin, is_buy, sz, limit_px, account=None, sz_decimals=8, px_decimals=5):
    """Place a spot limit order on Bitfinex"""
    try:
        amount = float(sz) if is_buy else -float(sz)  # Negative for sell
        
        # Round to proper decimals
        amount = round(amount, sz_decimals)
        price = round(float(limit_px), px_decimals)
        
        order_params = {
            "type": "EXCHANGE LIMIT",
            "symbol": coin,
            "amount": str(amount),
            "price": str(price)
        }
        
        result = trader._make_request('/w/order/submit', order_params, authenticated=True, method='POST')
        
        if result:
            side = "BUY" if is_buy else "SELL"
            print(f"✅ {side} order placed: {abs(amount)} {coin} @ ${price}")
            return result
        else:
            print(f"❌ Failed to place order")
            return None
            
    except Exception as e:
        print(f"Error placing spot limit order: {e}")
        return None

def all_spot_symbols():
    """Get all available trading symbols"""
    try:
        symbols_data = trader._make_request('/conf/pub:list:pair:exchange')
        if symbols_data and isinstance(symbols_data[0], list):
            # Filter for trading pairs (start with 't')
            symbols = [s for s in symbols_data[0] if s.startswith('t')]
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
        return 8, 5  # Default values for Bitfinex
    except Exception as e:
        print(f"Error getting decimals for {symbol}: {e}")
        return 8, 5

def limit_order(coin, is_buy, sz, limit_px, reduce_only=False, account=None):
    """Place a margin limit order"""
    try:
        amount = float(sz) if is_buy else -float(sz)
        
        order_params = {
            "type": "LIMIT",  # Margin trading
            "symbol": coin,
            "amount": str(amount),
            "price": str(float(limit_px))
        }
        
        if reduce_only:
            order_params["flags"] = 64  # Reduce-only flag
        
        result = trader._make_request('/w/order/submit', order_params, authenticated=True, method='POST')
        
        if result:
            side = "BUY" if is_buy else "SELL"
            print(f"✅ Margin {side} order placed: {abs(amount)} {coin} @ ${limit_px}")
            return result
        else:
            print(f"❌ Failed to place margin order")
            return None
            
    except Exception as e:
        print(f"Error placing margin limit order: {e}")
        return None

def adjust_leverage_size_signal(symbol, leverage, account=None):
    """Calculate position size based on 95% of balance with leverage"""
    try:
        # Get wallet balances
        wallets = trader._make_request('/w/wallets', authenticated=True)
        if not wallets:
            return 0
        
        # Find USD balance
        usd_balance = 0
        for wallet in wallets:
            if wallet[1] == 'USD' and wallet[0] == 'exchange':  # Exchange wallet USD
                usd_balance = float(wallet[2])  # Available balance
                break
        
        # Get current price
        ticker = trader._make_request(f'/ticker/{symbol}')
        if not ticker:
            return 0
        
        current_price = float(ticker[6])  # Last price
        
        # Calculate position size (95% of balance with leverage effect)
        position_value = usd_balance * 0.95 * leverage
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
        ticker = trader._make_request(f'/ticker/{symbol}')
        if not ticker:
            return 0
        
        current_price = float(ticker[6])
        
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
    """Set leverage for margin trading (Bitfinex doesn't have explicit leverage setting)"""
    print(f"ℹ️ Bitfinex uses margin-based trading. Leverage is implicit through position sizing.")
    return True

def get_current_price(symbol):
    """Fetch current price for a symbol"""
    try:
        ticker = trader._make_request(f'/ticker/{symbol}')
        if ticker:
            return float(ticker[6])  # Last price
        return None
    except Exception as e:
        print(f"Error getting current price: {e}")
        return None

def get_balance(account=None):
    """Get current USD balance"""
    try:
        wallets = trader._make_request('/w/wallets', authenticated=True)
        if wallets:
            for wallet in wallets:
                if wallet[1] == 'USD' and wallet[0] == 'exchange':
                    return float(wallet[2])  # Available balance
        return 0
    except Exception as e:
        print(f"Error getting balance: {e}")
        return 0

def get_account_value(account=None):
    """Get account information"""
    try:
        wallets = trader._make_request('/w/wallets', authenticated=True)
        if wallets:
            total_usd = 0
            for wallet in wallets:
                if wallet[1] == 'USD':
                    total_usd += float(wallet[2])
            
            return {
                'total_wallet_balance': total_usd,
                'available_balance': total_usd
            }
        return None
    except Exception as e:
        print(f"Error getting account value: {e}")
        return None

def connect():
    """Connect to Bitfinex API - returns trader instance"""
    return {'exchange': trader, 'wallet': trader}

def get_position(symbol, account=None):
    """Get current position for a symbol"""
    try:
        positions = trader._make_request('/w/positions', authenticated=True)
        if not positions:
            return None
        
        for pos in positions:
            if pos[0] == symbol:  # Symbol
                position_size = float(pos[2])  # Amount
                
                if position_size != 0:
                    return {
                        'symbol': symbol,
                        'size': abs(position_size),
                        'side': 'long' if position_size > 0 else 'short',
                        'entry_price': float(pos[3]),  # Base price
                        'mark_price': float(pos[3]),   # Using base price as mark
                        'unrealized_pnl': float(pos[6]),  # P&L
                        'percentage': (float(pos[6]) / float(pos[3])) * 100 if pos[3] else 0
                    }
        
        return None
        
    except Exception as e:
        print(f"Error getting position: {e}")
        return None

def kill_switch(symbol, account=None):
    """Emergency close position"""
    try:
        position = get_position(symbol, account)
        if not position:
            print(f"No position to close for {symbol}")
            return True
        
        # Close with market order (opposite side)
        amount = -position['size'] if position['side'] == 'long' else position['size']
        
        order_params = {
            "type": "MARKET",
            "symbol": symbol,
            "amount": str(amount)
        }
        
        result = trader._make_request('/w/order/submit', order_params, authenticated=True, method='POST')
        
        if result:
            print(f"🚨 KILL SWITCH: Closed position for {symbol}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error in kill switch: {e}")
        return False

def cancel_all_orders(account=None):
    """Cancel all open orders"""
    try:
        result = trader._make_request('/w/order/cancel/all', {}, authenticated=True, method='POST')
        
        if result:
            print("✅ All orders cancelled")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error cancelling orders: {e}")
        return False

def cancel_symbol_orders(account, symbol):
    """Cancel all orders for a specific symbol"""
    try:
        # Get active orders for symbol
        orders = trader._make_request('/w/orders', authenticated=True)
        if not orders:
            return True
        
        cancelled = 0
        for order in orders:
            if order[3] == symbol:  # Symbol field
                cancel_params = {
                    "id": order[0]  # Order ID
                }
                result = trader._make_request('/w/order/cancel', cancel_params, authenticated=True, method='POST')
                if result:
                    cancelled += 1
        
        print(f"✅ Cancelled {cancelled} orders for {symbol}")
        return True
        
    except Exception as e:
        print(f"Error cancelling symbol orders: {e}")
        return False

def pnl_close(symbol, target, max_loss, account=None):
    """Close position based on PnL targets"""
    try:
        position = get_position(symbol, account)
        if not position:
            return False
        
        pnl_percentage = position.get('percentage', 0)
        
        if pnl_percentage >= target:
            print(f"🎯 Taking profit at {pnl_percentage:.2f}% for {symbol}")
            return kill_switch(symbol, account)
        
        if pnl_percentage <= max_loss:
            print(f"🛑 Stopping loss at {pnl_percentage:.2f}% for {symbol}")
            return kill_switch(symbol, account)
        
        return False
        
    except Exception as e:
        print(f"Error in PnL close: {e}")
        return False

def get_ohclv(cb_symbol, timeframe, limit=500):
    """Get OHLCV data from Bitfinex"""
    try:
        # Convert timeframe to Bitfinex format
        timeframe_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '3h': '3h', '6h': '6h', '12h': '12h',
            '1d': '1D', '1w': '7D', '1M': '1M'
        }
        
        tf = timeframe_map.get(timeframe, '15m')
        
        # Convert symbol format (remove /USD, /USDT and add 't' prefix if needed)
        symbol = cb_symbol.replace('/USD', 'USD').replace('/USDT', 'USD')
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        # Get candles
        params = {
            'limit': min(limit, 1000),
            'sort': 1  # Ascending order
        }
        
        data = trader._make_request(f'/candles/trade:{tf}:{symbol}/hist', params)
        
        if data:
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            
            return df.sort_values('timestamp')
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error getting OHLCV data: {e}")
        return pd.DataFrame()

def get_ohlcv2(symbol, interval, lookback_days):
    """Get historical OHLCV data with lookback period"""
    try:
        # Calculate start time
        end_time = int(time.time() * 1000)
        start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)
        
        # Convert interval to Bitfinex format
        timeframe_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '3h': '3h', '6h': '6h', '12h': '12h',
            '1d': '1D', '1w': '7D', '1M': '1M'
        }
        
        tf = timeframe_map.get(interval, '15m')
        
        # Ensure symbol has 't' prefix
        if not symbol.startswith('t'):
            symbol = 't' + symbol
        
        params = {
            'limit': 1000,
            'start': start_time,
            'end': end_time,
            'sort': 1
        }
        
        data = trader._make_request(f'/candles/trade:{tf}:{symbol}/hist', params)
        
        return data if data else []
        
    except Exception as e:
        print(f"Error getting OHLCV2 data: {e}")
        return []

def process_data_to_df(snapshot_data, time_period=20):
    """Process raw OHLCV data to DataFrame"""
    if not snapshot_data:
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(snapshot_data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        
        return df.sort_values('timestamp')
        
    except Exception as e:
        print(f"Error processing data to DataFrame: {e}")
        return pd.DataFrame()

def calculate_vwap_with_symbol(symbol):
    """Calculate VWAP for a symbol"""
    try:
        df = get_ohclv(symbol, '15m', 100)
        if df.empty:
            return None
        
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['typical_price'] * df['volume']
        df['cumulative_pv'] = df['pv'].cumsum()
        df['cumulative_volume'] = df['volume'].cumsum()
        df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
        
        return df['vwap'].iloc[-1]
        
    except Exception as e:
        print(f"Error calculating VWAP: {e}")
        return None

def supply_demand_zones_hl(symbol, timeframe, limit=100):
    """Identify supply and demand zones"""
    try:
        df = get_ohclv(symbol, timeframe, limit)
        if df.empty:
            return None
        
        # Professional supply/demand zone identification
        df['high_resistance'] = df['high'].rolling(window=20).max()
        df['low_support'] = df['low'].rolling(window=20).min()
        
        # Calculate institutional levels
        df['institutional_resistance'] = df['high'].quantile(0.95)
        df['institutional_support'] = df['low'].quantile(0.05)
        
        current_price = df['close'].iloc[-1]
        resistance = df['high_resistance'].iloc[-1]
        support = df['low_support'].iloc[-1]
        
        return {
            'resistance': resistance,
            'support': support,
            'institutional_resistance': df['institutional_resistance'].iloc[-1],
            'institutional_support': df['institutional_support'].iloc[-1],
            'current': current_price,
            'zone_strength': abs(resistance - support) / current_price,
            'professional_bias': 'bullish' if current_price > (resistance + support) / 2 else 'bearish'
        }
        
    except Exception as e:
        print(f"Error calculating supply/demand zones: {e}")
        return None

# Additional professional trading functions
def get_funding_stats():
    """Get comprehensive funding rate statistics"""
    try:
        funding_symbols = ['fUSD', 'fBTC', 'fETH', 'fEUR']
        funding_data = []
        
        for symbol in funding_symbols:
            stats = trader._make_request(f'/stats1/{symbol}:30d:FUNDING/last')
            if stats:
                funding_data.append({
                    'symbol': symbol,
                    'funding_rate': float(stats[1]) if len(stats) > 1 else 0,
                    'timestamp': pd.to_datetime(stats[0], unit='ms') if len(stats) > 0 else datetime.utcnow()
                })
        
        return funding_data
        
    except Exception as e:
        print(f"Error getting funding stats: {e}")
        return []

def institutional_order_flow_analysis(symbol):
    """Analyze institutional order flow patterns"""
    try:
        # Get recent trades
        trades = trader._make_request(f'/trades/{symbol}/hist', {'limit': 1000})
        if not trades:
            return None
        
        df = pd.DataFrame(trades, columns=['ID', 'Timestamp', 'Amount', 'Price'])
        df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df['amount'] = df['Amount'].astype(float)
        df['price'] = df['Price'].astype(float)
        df['value_usd'] = df['price'] * df['amount'].abs()
        
        # Institutional classification
        df['institutional_tier'] = pd.cut(df['value_usd'], 
                                        bins=[0, 50000, 200000, 1000000, float('inf')],
                                        labels=['Retail', 'Professional', 'Institutional', 'Sovereign'])
        
        # Flow analysis
        institutional_flow = df[df['institutional_tier'].isin(['Institutional', 'Sovereign'])]
        buy_flow = institutional_flow[institutional_flow['amount'] > 0]['value_usd'].sum()
        sell_flow = institutional_flow[institutional_flow['amount'] < 0]['value_usd'].sum()
        
        return {
            'buy_flow': buy_flow,
            'sell_flow': sell_flow,
            'net_flow': buy_flow - sell_flow,
            'flow_ratio': buy_flow / sell_flow if sell_flow > 0 else float('inf'),
            'institutional_bias': 'bullish' if buy_flow > sell_flow else 'bearish'
        }
        
    except Exception as e:
        print(f"Error in institutional flow analysis: {e}")
        return None
