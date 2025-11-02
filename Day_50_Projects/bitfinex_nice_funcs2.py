# Bitfinex Professional Trading Utilities for Supply & Demand Zone Strategy
# Institutional-grade functions for Bitfinex exchange operations

import requests
import json
import time
import hmac
import hashlib
import pandas as pd
import pandas_ta as ta
import os
from dotenv import load_dotenv
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from termcolor import cprint

# Load environment variables
load_dotenv()

def get_signature(api_secret: str, message: str) -> str:
    """Generate HMAC SHA384 signature for Bitfinex API"""
    return hmac.new(
        api_secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha384
    ).hexdigest()

def make_bitfinex_request(config: Dict, endpoint: str, params: Dict = None, signed: bool = False, method: str = 'GET') -> Optional[Dict]:
    """Make authenticated or public request to Bitfinex API"""
    if params is None:
        params = {}
    
    base_url = config.get('BITFINEX_BASE_URL', 'https://api.bitfinex.com')
    url = f"{base_url}{endpoint}"
    
    headers = {'Content-Type': 'application/json'}
    
    if signed:
        api_key = config.get('BITFINEX_API_KEY')
        api_secret = config.get('BITFINEX_API_SECRET')
        
        if not api_key or not api_secret:
            cprint("Error: Bitfinex API credentials not configured", "red")
            return None
        
        nonce = str(int(time.time() * 1000000))
        body = json.dumps(params) if params else ''
        message = f'/api{endpoint}{nonce}{body}'
        signature = get_signature(api_secret, message)
        
        headers.update({
            'bfx-nonce': nonce,
            'bfx-apikey': api_key,
            'bfx-signature': signature
        })
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=15)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=params, timeout=15)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        cprint(f"Bitfinex API request failed: {e}", "red")
        return None

def token_overview(config: Dict, symbol: str) -> Optional[Dict]:
    """Get comprehensive token overview for a Bitfinex symbol"""
    try:
        # Get ticker data
        ticker_symbol = symbol[1:] if symbol.startswith('t') else symbol  # Remove 't' prefix for ticker
        ticker_data = make_bitfinex_request(config, f"/v1/pubticker/{ticker_symbol}")
        
        if not ticker_data:
            return None
        
        # Get stats data
        stats_data = make_bitfinex_request(config, f"/v1/stats/{ticker_symbol}")
        
        # Get order book depth
        depth_data = make_bitfinex_request(config, f"/v1/book/{ticker_symbol}", {"limit_bids": 50, "limit_asks": 50})
        
        # Parse ticker data
        current_price = float(ticker_data.get('last_price', 0))
        volume_24h = float(ticker_data.get('volume', 0))
        high_24h = float(ticker_data.get('high', 0))
        low_24h = float(ticker_data.get('low', 0))
        change_24h = float(ticker_data.get('daily_change_perc', 0)) * 100  # Convert to percentage
        
        # Calculate spread and liquidity
        spread = 0
        bid_liquidity = 0
        ask_liquidity = 0
        
        if depth_data and depth_data.get('bids') and depth_data.get('asks'):
            best_bid = float(depth_data['bids'][0]['price'])
            best_ask = float(depth_data['asks'][0]['price'])
            spread = (best_ask - best_bid) / best_bid * 100
            
            # Calculate liquidity (sum of top 10 levels)
            bid_liquidity = sum(float(bid['amount']) for bid in depth_data['bids'][:10])
            ask_liquidity = sum(float(ask['amount']) for ask in depth_data['asks'][:10])
        
        return {
            'symbol': symbol,
            'price': current_price,
            'price_change_24h': change_24h,
            'volume_24h': volume_24h,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'spread_pct': spread,
            'bid_liquidity': bid_liquidity,
            'ask_liquidity': ask_liquidity,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        cprint(f"Error getting token overview for {symbol}: {e}", "red")
        return None

def get_historical_data(config: Dict, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
    """Get historical candlestick data from Bitfinex"""
    try:
        # Map intervals to Bitfinex format
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '3h': '3h', '6h': '6h', '12h': '12h',
            '1d': '1D', '1w': '7D', '2w': '14D', '1M': '1M'
        }
        
        bf_interval = interval_map.get(interval, '1h')
        
        # Remove 't' prefix if present for API call
        api_symbol = symbol[1:] if symbol.startswith('t') else symbol
        
        # Calculate start time
        now = datetime.now()
        if bf_interval == '1m':
            start_time = now - timedelta(minutes=limit)
        elif bf_interval == '5m':
            start_time = now - timedelta(minutes=limit * 5)
        elif bf_interval == '15m':
            start_time = now - timedelta(minutes=limit * 15)
        elif bf_interval == '30m':
            start_time = now - timedelta(minutes=limit * 30)
        elif bf_interval == '1h':
            start_time = now - timedelta(hours=limit)
        elif bf_interval == '3h':
            start_time = now - timedelta(hours=limit * 3)
        elif bf_interval == '6h':
            start_time = now - timedelta(hours=limit * 6)
        elif bf_interval == '12h':
            start_time = now - timedelta(hours=limit * 12)
        elif bf_interval == '1D':
            start_time = now - timedelta(days=limit)
        elif bf_interval == '7D':
            start_time = now - timedelta(weeks=limit)
        elif bf_interval == '14D':
            start_time = now - timedelta(weeks=limit * 2)
        else:
            start_time = now - timedelta(days=limit * 30)  # Monthly
        
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(now.timestamp() * 1000)
        
        endpoint = f"/v2/candles/trade:{bf_interval}:{symbol}/hist"
        params = {
            'start': start_timestamp,
            'end': end_timestamp,
            'limit': limit,
            'sort': 1  # Sort ascending (oldest first)
        }
        
        data = make_bitfinex_request(config, endpoint, params)
        
        if not data:
            return None
        
        # Convert to DataFrame
        # Bitfinex candles format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
        df = pd.DataFrame(data, columns=['timestamp', 'Open', 'Close', 'High', 'Low', 'Volume'])
        
        # Convert timestamp to datetime
        df['Datetime (UTC)'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Reorder columns and sort
        df = df[['Datetime (UTC)', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Datetime (UTC)')
        
        return df
    
    except Exception as e:
        cprint(f"Error getting historical data for {symbol}: {e}", "red")
        return None

def market_buy(config: Dict, symbol: str, amount_usd: float, slippage_bps: int = None) -> Optional[Dict]:
    """Execute a market buy order on Bitfinex"""
    if slippage_bps is None:
        slippage_bps = config.get('SLIPPAGE', 75)
    
    try:
        # Get current price for quantity calculation
        ticker_symbol = symbol[1:] if symbol.startswith('t') else symbol
        ticker_data = make_bitfinex_request(config, f"/v1/pubticker/{ticker_symbol}")
        
        if not ticker_data:
            cprint(f"Failed to get price for {symbol}", "red")
            return None
        
        current_price = float(ticker_data.get('ask', 0))  # Use ask price for buying
        if current_price <= 0:
            cprint(f"Invalid price for {symbol}: {current_price}", "red")
            return None
        
        quantity = amount_usd / current_price
        
        # Prepare order parameters
        order_params = {
            "symbol": symbol,
            "amount": str(quantity),
            "price": "1",  # Market orders use price "1"
            "exchange": "bitfinex",
            "side": "buy",
            "type": "market"
        }
        
        # Add professional features
        if config.get('USE_HIDDEN_ORDERS'):
            order_params["is_hidden"] = True
        
        if config.get('USE_POST_ONLY'):
            order_params["is_postonly"] = True
        
        # Execute order
        result = make_bitfinex_request(config, "/v1/order/new", order_params, signed=True, method='POST')
        
        if result and not result.get('message'):
            cprint(f"✅ Market BUY executed: {symbol} for ${amount_usd}", "green")
            return result
        else:
            error_msg = result.get('message', 'Unknown error') if result else 'API call failed'
            cprint(f"❌ Market BUY failed: {symbol} - {error_msg}", "red")
            return None
    
    except Exception as e:
        cprint(f"Error in market_buy: {e}", "red")
        return None

def market_sell(config: Dict, symbol: str, quantity: float, slippage_bps: int = None) -> Optional[Dict]:
    """Execute a market sell order on Bitfinex"""
    if slippage_bps is None:
        slippage_bps = config.get('SLIPPAGE', 75)
    
    try:
        if quantity <= 0:
            cprint(f"Invalid quantity: {quantity}", "red")
            return None
        
        # Prepare order parameters
        order_params = {
            "symbol": symbol,
            "amount": str(quantity),
            "price": "1",  # Market orders use price "1"
            "exchange": "bitfinex",
            "side": "sell",
            "type": "market"
        }
        
        # Add professional features
        if config.get('USE_HIDDEN_ORDERS'):
            order_params["is_hidden"] = True
        
        # Execute order
        result = make_bitfinex_request(config, "/v1/order/new", order_params, signed=True, method='POST')
        
        if result and not result.get('message'):
            cprint(f"✅ Market SELL executed: {quantity} {symbol}", "green")
            return result
        else:
            error_msg = result.get('message', 'Unknown error') if result else 'API call failed'
            cprint(f"❌ Market SELL failed: {symbol} - {error_msg}", "red")
            return None
    
    except Exception as e:
        cprint(f"Error in market_sell: {e}", "red")
        return None

def get_account_balance(config: Dict, currency: str = 'USD') -> float:
    """Get account balance for specific currency"""
    try:
        data = make_bitfinex_request(config, "/v1/balances", signed=True, method='POST')
        
        if not data:
            return 0.0
        
        for balance in data:
            if balance.get('currency', '').upper() == currency.upper() and balance.get('type') == 'exchange':
                return float(balance.get('available', 0))
        
        return 0.0
    
    except Exception as e:
        cprint(f"Error getting balance for {currency}: {e}", "red")
        return 0.0

def get_position_value(config: Dict, symbol: str) -> float:
    """Get current position value in USD"""
    try:
        # Extract base currency from symbol
        base_currency = symbol.replace('t', '').replace('USD', '').replace('UST', '').replace('USDT', '')
        
        # Get balance
        balance = get_account_balance(config, base_currency)
        
        if balance <= 0:
            return 0.0
        
        # Get current price
        ticker_symbol = symbol[1:] if symbol.startswith('t') else symbol
        ticker_data = make_bitfinex_request(config, f"/v1/pubticker/{ticker_symbol}")
        
        if not ticker_data:
            return 0.0
        
        current_price = float(ticker_data.get('last_price', 0))
        return balance * current_price
    
    except Exception as e:
        cprint(f"Error getting position value for {symbol}: {e}", "red")
        return 0.0

def kill_switch(config: Dict, symbol: str) -> bool:
    """Emergency exit - sell all positions for a symbol"""
    try:
        # Extract base currency
        base_currency = symbol.replace('t', '').replace('USD', '').replace('UST', '').replace('USDT', '')
        balance = get_account_balance(config, base_currency)
        
        if balance <= 0:
            cprint(f"No position to close for {symbol}", "yellow")
            return True
        
        cprint(f"🚨 KILL SWITCH ACTIVATED for {symbol} - Selling {balance} {base_currency}", "red", attrs=['bold'])
        
        # Execute market sell
        result = market_sell(config, symbol, balance)
        
        if result:
            cprint(f"✅ Emergency exit completed for {symbol}", "green")
            return True
        else:
            cprint(f"❌ Kill switch failed for {symbol}", "red")
            return False
    
    except Exception as e:
        cprint(f"Error in kill_switch: {e}", "red")
        return False

def get_funding_rate(config: Dict, symbol: str) -> Optional[Dict]:
    """Get current funding rate for symbol"""
    try:
        # Convert to funding symbol format
        if symbol.endswith('USD'):
            funding_symbol = symbol.replace('USD', 'F0:USTF0')
        else:
            funding_symbol = f"{symbol}F0:USTF0"
        
        ticker_symbol = funding_symbol[1:] if funding_symbol.startswith('t') else funding_symbol
        data = make_bitfinex_request(config, f"/v1/pubticker/{ticker_symbol}")
        
        if data:
            return {
                'symbol': symbol,
                'funding_rate': float(data.get('last_price', 0)),
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        cprint(f"Could not get funding rate for {symbol}: {e}", "yellow")
    
    return None

def calculate_supply_demand_zones(config: Dict, symbol: str, timeframe: str = '1h', lookback_periods: int = 100) -> Dict:
    """Calculate supply and demand zones using advanced technical analysis"""
    df = get_historical_data(config, symbol, timeframe, lookback_periods)
    
    if df is None or len(df) < 20:
        return {'supply_zones': [], 'demand_zones': [], 'current_price': 0}
    
    # Calculate technical indicators
    df['sma_20'] = ta.sma(df['Close'], length=20)
    df['sma_50'] = ta.sma(df['Close'], length=50)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df['volume_sma'] = ta.sma(df['Volume'], length=20)
    df['high_volume'] = df['Volume'] > (df['volume_sma'] * config['SDZ_CONFIG']['VOLUME_CONFIRMATION_MULTIPLIER'])
    
    # Calculate price momentum
    df['price_change'] = df['Close'].pct_change()
    df['price_momentum'] = ta.roc(df['Close'], length=5)
    
    # Identify pivot points with improved logic
    window = 5
    df['pivot_high'] = df['High'] == df['High'].rolling(window=window, center=True).max()
    df['pivot_low'] = df['Low'] == df['Low'].rolling(window=window, center=True).min()
    
    supply_zones = []
    demand_zones = []
    
    current_price = df['Close'].iloc[-1]
    zone_width_pct = config['SDZ_CONFIG']['ZONE_WIDTH_PCT']
    min_strength = config['SDZ_CONFIG']['MIN_ZONE_STRENGTH']
    max_zones = config['SDZ_CONFIG']['MAX_ZONES_TO_TRACK']
    
    # Find supply zones (resistance levels)
    pivot_highs = df[df['pivot_high'] == True].copy()
    for _, pivot in pivot_highs.iterrows():
        zone_high = pivot['High']
        zone_low = zone_high * (1 - zone_width_pct)
        
        # Count zone interactions
        interactions = len(df[(df['High'] >= zone_low) & (df['High'] <= zone_high)])
        
        # Check for rejections (candles with long upper wicks)
        rejections = len(df[(df['High'] >= zone_low) & (df['High'] <= zone_high) & 
                           (df['Close'] < df['High'] * 0.95)])
        
        strength_score = interactions + (rejections * 2)  # Weight rejections more
        
        if strength_score >= min_strength and zone_low > current_price * 1.001:  # Above current price
            supply_zones.append({
                'high': zone_high,
                'low': zone_low,
                'strength': strength_score,
                'interactions': interactions,
                'rejections': rejections,
                'timestamp': pivot['Datetime (UTC)'],
                'volume_confirmation': pivot['high_volume'],
                'distance_pct': (zone_low - current_price) / current_price,
                'rsi_at_level': pivot['rsi'] if not pd.isna(pivot['rsi']) else 50
            })
    
    # Find demand zones (support levels)
    pivot_lows = df[df['pivot_low'] == True].copy()
    for _, pivot in pivot_lows.iterrows():
        zone_low = pivot['Low']
        zone_high = zone_low * (1 + zone_width_pct)
        
        # Count zone interactions
        interactions = len(df[(df['Low'] >= zone_low) & (df['Low'] <= zone_high)])
        
        # Check for bounces (candles with long lower wicks)
        bounces = len(df[(df['Low'] >= zone_low) & (df['Low'] <= zone_high) & 
                        (df['Close'] > df['Low'] * 1.05)])
        
        strength_score = interactions + (bounces * 2)  # Weight bounces more
        
        if strength_score >= min_strength and zone_high < current_price * 0.999:  # Below current price
            demand_zones.append({
                'high': zone_high,
                'low': zone_low,
                'strength': strength_score,
                'interactions': interactions,
                'bounces': bounces,
                'timestamp': pivot['Datetime (UTC)'],
                'volume_confirmation': pivot['high_volume'],
                'distance_pct': (current_price - zone_high) / current_price,
                'rsi_at_level': pivot['rsi'] if not pd.isna(pivot['rsi']) else 50
            })
    
    # Sort zones by strength and proximity to current price
    supply_zones = sorted(supply_zones, 
                         key=lambda x: (x['strength'], -abs(x['distance_pct'])), 
                         reverse=True)[:max_zones]
    
    demand_zones = sorted(demand_zones, 
                         key=lambda x: (x['strength'], -abs(x['distance_pct'])), 
                         reverse=True)[:max_zones]
    
    return {
        'supply_zones': supply_zones,
        'demand_zones': demand_zones,
        'current_price': current_price,
        'sma_20': df['sma_20'].iloc[-1] if not df['sma_20'].isna().iloc[-1] else current_price,
        'sma_50': df['sma_50'].iloc[-1] if not df['sma_50'].isna().iloc[-1] else current_price,
        'rsi': df['rsi'].iloc[-1] if not df['rsi'].isna().iloc[-1] else 50,
        'volume_avg': df['volume_sma'].iloc[-1] if not df['volume_sma'].isna().iloc[-1] else 0,
        'momentum': df['price_momentum'].iloc[-1] if not df['price_momentum'].isna().iloc[-1] else 0
    }

def check_trend(config: Dict, symbol: str, timeframe: str = '1h') -> str:
    """Check trend using advanced SMA analysis"""
    df = get_historical_data(config, symbol, timeframe, config.get('SMA_DAYS_BACK', 5) * 24)
    
    if df is None or len(df) < config.get('SMA_BARS', 20):
        return 'sideways'
    
    # Calculate multiple SMAs for trend confirmation
    sma_bars = config.get('SMA_BARS', 20)
    df['sma_fast'] = ta.sma(df['Close'], length=sma_bars)
    df['sma_slow'] = ta.sma(df['Close'], length=sma_bars * 2)
    
    current_price = df['Close'].iloc[-1]
    current_sma_fast = df['sma_fast'].iloc[-1]
    current_sma_slow = df['sma_slow'].iloc[-1]
    
    buffer_pct = config.get('SMA_BUFFER_PCT', 0.01)
    
    # Check trend direction
    if (current_price > current_sma_fast * (1 + buffer_pct) and 
        current_sma_fast > current_sma_slow and
        df['sma_fast'].iloc[-1] > df['sma_fast'].iloc[-3]):  # SMA trending up
        return 'up'
    elif (current_price < current_sma_fast * (1 - buffer_pct) and 
          current_sma_fast < current_sma_slow and
          df['sma_fast'].iloc[-1] < df['sma_fast'].iloc[-3]):  # SMA trending down
        return 'down'
    else:
        return 'sideways'

def check_market_conditions(config: Dict) -> Dict:
    """Check overall market conditions using major cryptocurrencies"""
    major_symbols = config.get('MARKET_CONDITIONS', {}).get('MAJOR_COINS_FOR_SENTIMENT', ['tBTCUSD', 'tETHUSD', 'tLTCUSD'])
    sentiment_timeframe = config.get('MARKET_CONDITIONS', {}).get('SENTIMENT_TIMEFRAME', '4h')
    
    trends = {}
    bullish_count = 0
    
    for symbol in major_symbols:
        trend = check_trend(config, symbol, sentiment_timeframe)
        trends[symbol] = trend
        
        if trend == 'up':
            bullish_count += 1
    
    total_symbols = len(major_symbols)
    bullish_percentage = bullish_count / total_symbols if total_symbols > 0 else 0
    
    bullish_threshold = config.get('MARKET_CONDITIONS', {}).get('BULLISH_THRESHOLD', 0.67)
    bearish_threshold = config.get('MARKET_CONDITIONS', {}).get('BEARISH_THRESHOLD', 0.33)
    
    if bullish_percentage >= bullish_threshold:
        market_condition = 'bullish'
    elif bullish_percentage <= bearish_threshold:
        market_condition = 'bearish'
    else:
        market_condition = 'neutral'
    
    return {
        'condition': market_condition,
        'bullish_percentage': bullish_percentage,
        'individual_trends': trends,
        'analysis_timeframe': sentiment_timeframe
    }

def log_trade(config: Dict, symbol: str, action: str, quantity: float, price: float, value_usd: float):
    """Log trade information with institutional detail"""
    trade_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'price': price,
        'value_usd': value_usd,
        'exchange': 'Bitfinex'
    }
    
    # Save to trade history CSV
    trade_file = config.get('TRADE_HISTORY_CSV', 'csvs/bitfinex/sdz_trades.csv')
    os.makedirs(os.path.dirname(trade_file), exist_ok=True)
    
    try:
        if os.path.exists(trade_file):
            df = pd.read_csv(trade_file)
            new_row = pd.DataFrame([trade_data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([trade_data])
        
        df.to_csv(trade_file, index=False)
        cprint(f"📊 Trade logged: {action} {quantity:.6f} {symbol} at ${price:.6f}", "cyan")
    
    except Exception as e:
        cprint(f"Error logging trade: {e}", "red")

def is_market_open() -> bool:
    """Check if market is open (crypto markets are always open)"""
    return True

def print_pretty_json(data: Any):
    """Print formatted JSON data"""
    print(json.dumps(data, indent=2, default=str))

# Compatibility functions to match original interface
def get_token_overview(config: Dict, symbol: str) -> Optional[Dict]:
    """Alias for token_overview for compatibility"""
    return token_overview(config, symbol)

def fetch_wallet_holdings_nosaving_names(config: Dict, wallet_address: str, token_mint_address: str) -> float:
    """Get position value for compatibility (returns USD value)"""
    return get_position_value(config, token_mint_address)
