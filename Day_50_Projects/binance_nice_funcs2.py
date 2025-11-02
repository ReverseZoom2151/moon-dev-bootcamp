# Binance Nice Functions for Supply & Demand Zone Trading
# Professional trading utilities adapted for Binance exchange

import requests
import pandas as pd
import pandas_ta as ta
import os
import time
import hmac
import hashlib
from urllib.parse import urlencode
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv
from typing import Dict, Optional, Any

# Load environment variables
load_dotenv()

def get_signature(query_string: str, api_secret: str) -> str:
    """Generate HMAC SHA256 signature for Binance API"""
    return hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def make_binance_request(config: Dict, endpoint: str, params: Dict = None, signed: bool = False, method: str = 'GET') -> Optional[Dict]:
    """Make authenticated or public request to Binance API"""
    if params is None:
        params = {}
    
    base_url = config.get('BINANCE_BASE_URL', 'https://api.binance.com')
    url = f"{base_url}{endpoint}"
    
    headers = {}
    
    if signed:
        api_key = config.get('BINANCE_API_KEY')
        api_secret = config.get('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            cprint("Error: Binance API credentials not configured", "red")
            return None
        
        params['timestamp'] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = get_signature(query_string, api_secret)
        params['signature'] = signature
        
        headers['X-MBX-APIKEY'] = api_key
    
    try:
        if method == 'GET':
            response = requests.get(url, params=params, headers=headers, timeout=10)
        elif method == 'POST':
            response = requests.post(url, params=params, headers=headers, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        cprint(f"Binance API request failed: {e}", "red")
        return None

def token_overview(config: Dict, symbol: str) -> Optional[Dict]:
    """Get comprehensive token overview for a Binance symbol"""
    # Get 24hr ticker statistics
    ticker_data = make_binance_request(config, "/api/v3/ticker/24hr", {"symbol": symbol})
    
    if not ticker_data:
        return None
    
    # Get current price
    price_data = make_binance_request(config, "/api/v3/ticker/price", {"symbol": symbol})
    current_price = float(price_data['price']) if price_data else 0
    
    # Get order book depth
    depth_data = make_binance_request(config, "/api/v3/depth", {"symbol": symbol, "limit": 100})
    
    # Calculate spread and liquidity metrics
    spread = 0
    bid_liquidity = 0
    ask_liquidity = 0
    
    if depth_data and depth_data.get('bids') and depth_data.get('asks'):
        best_bid = float(depth_data['bids'][0][0])
        best_ask = float(depth_data['asks'][0][0])
        spread = (best_ask - best_bid) / best_bid * 100
        
        # Calculate liquidity (sum of top 10 levels)
        bid_liquidity = sum(float(bid[1]) for bid in depth_data['bids'][:10])
        ask_liquidity = sum(float(ask[1]) for ask in depth_data['asks'][:10])
    
    return {
        'symbol': symbol,
        'price': current_price,
        'price_change_24h': float(ticker_data.get('priceChangePercent', 0)),
        'volume_24h': float(ticker_data.get('volume', 0)),
        'quote_volume_24h': float(ticker_data.get('quoteVolume', 0)),
        'high_24h': float(ticker_data.get('highPrice', 0)),
        'low_24h': float(ticker_data.get('lowPrice', 0)),
        'trade_count_24h': int(ticker_data.get('count', 0)),
        'spread_pct': spread,
        'bid_liquidity': bid_liquidity,
        'ask_liquidity': ask_liquidity,
        'timestamp': datetime.now().isoformat()
    }

def get_historical_data(config: Dict, symbol: str, interval: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
    """Get historical kline/candlestick data from Binance"""
    endpoint = "/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    data = make_binance_request(config, endpoint, params)
    
    if not data:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert data types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
        df[col] = pd.to_numeric(df[col])
    
    # Rename columns to standard format
    df = df.rename(columns={
        'open_time': 'Datetime (UTC)',
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    return df[['Datetime (UTC)', 'Open', 'High', 'Low', 'Close', 'Volume']].sort_values('Datetime (UTC)')

def market_buy(config: Dict, symbol: str, amount_usdt: float, slippage_bps: int = None) -> Optional[Dict]:
    """Execute a market buy order on Binance"""
    if slippage_bps is None:
        slippage_bps = config.get('SLIPPAGE', 100)
    
    try:
        # Get current price
        price_data = make_binance_request(config, "/api/v3/ticker/price", {"symbol": symbol})
        if not price_data:
            cprint(f"Failed to get price for {symbol}", "red")
            return None
        
        current_price = float(price_data['price'])
        
        # Get symbol info for precision
        symbol_info = get_symbol_info(config, symbol)
        if not symbol_info:
            cprint(f"Failed to get symbol info for {symbol}", "red")
            return None
        
        # Calculate quantity with precision
        base_precision = symbol_info['baseAssetPrecision']
        quantity = round(amount_usdt / current_price, base_precision)
        
        if quantity <= 0:
            cprint(f"Invalid quantity calculated: {quantity}", "red")
            return None
        
        # Place market buy order
        endpoint = "/api/v3/order"
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET',
            'quoteOrderQty': f"{amount_usdt:.2f}"  # Use quote quantity for market buy
        }
        
        result = make_binance_request(config, endpoint, params, signed=True, method='POST')
        
        if result:
            cprint(f"✅ Market BUY executed: {symbol} for ${amount_usdt}", "green")
            return result
        else:
            cprint(f"❌ Market BUY failed: {symbol}", "red")
            return None
    
    except Exception as e:
        cprint(f"Error in market_buy: {e}", "red")
        return None

def market_sell(config: Dict, symbol: str, quantity: float, slippage_bps: int = None) -> Optional[Dict]:
    """Execute a market sell order on Binance"""
    if slippage_bps is None:
        slippage_bps = config.get('SLIPPAGE', 100)
    
    try:
        # Get symbol info for precision
        symbol_info = get_symbol_info(config, symbol)
        if not symbol_info:
            cprint(f"Failed to get symbol info for {symbol}", "red")
            return None
        
        # Round quantity to appropriate precision
        base_precision = symbol_info['baseAssetPrecision']
        quantity = round(quantity, base_precision)
        
        if quantity <= 0:
            cprint(f"Invalid quantity: {quantity}", "red")
            return None
        
        # Place market sell order
        endpoint = "/api/v3/order"
        params = {
            'symbol': symbol,
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': f"{quantity:.{base_precision}f}"
        }
        
        result = make_binance_request(config, endpoint, params, signed=True, method='POST')
        
        if result:
            cprint(f"✅ Market SELL executed: {quantity} {symbol}", "green")
            return result
        else:
            cprint(f"❌ Market SELL failed: {symbol}", "red")
            return None
    
    except Exception as e:
        cprint(f"Error in market_sell: {e}", "red")
        return None

def get_account_balance(config: Dict, asset: str = 'USDT') -> float:
    """Get account balance for specific asset"""
    endpoint = "/api/v3/account"
    data = make_binance_request(config, endpoint, signed=True)
    
    if not data or 'balances' not in data:
        return 0.0
    
    for balance in data['balances']:
        if balance['asset'] == asset:
            return float(balance['free'])
    
    return 0.0

def get_position_value(config: Dict, symbol: str) -> float:
    """Get current position value in USDT"""
    # Extract base asset from symbol (e.g., 'BTC' from 'BTCUSDT')
    base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
    
    # Get balance
    balance = get_account_balance(config, base_asset)
    
    if balance <= 0:
        return 0.0
    
    # Get current price
    price_data = make_binance_request(config, "/api/v3/ticker/price", {"symbol": symbol})
    if not price_data:
        return 0.0
    
    current_price = float(price_data['price'])
    return balance * current_price

def get_symbol_info(config: Dict, symbol: str) -> Optional[Dict]:
    """Get symbol trading information"""
    endpoint = "/api/v3/exchangeInfo"
    data = make_binance_request(config, endpoint)
    
    if not data or 'symbols' not in data:
        return None
    
    for symbol_info in data['symbols']:
        if symbol_info['symbol'] == symbol:
            return symbol_info
    
    return None

def kill_switch(config: Dict, symbol: str) -> bool:
    """Emergency exit - sell all positions for a symbol"""
    try:
        # Get current position
        base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        balance = get_account_balance(config, base_asset)
        
        if balance <= 0:
            cprint(f"No position to close for {symbol}", "yellow")
            return True
        
        cprint(f"🚨 KILL SWITCH ACTIVATED for {symbol} - Selling {balance} {base_asset}", "red", attrs=['bold'])
        
        # Execute market sell
        result = market_sell(config, symbol, balance)
        
        if result:
            # Log to closed positions
            closed_positions_file = config.get('CLOSED_POSITIONS_TXT', 'data/closed_positions.txt')
            os.makedirs(os.path.dirname(closed_positions_file), exist_ok=True)
            
            with open(closed_positions_file, 'a') as f:
                f.write(f"{symbol}\n")
            
            cprint(f"✅ Emergency exit completed for {symbol}", "green")
            return True
        else:
            cprint(f"❌ Kill switch failed for {symbol}", "red")
            return False
    
    except Exception as e:
        cprint(f"Error in kill_switch: {e}", "red")
        return False

def calculate_supply_demand_zones(config: Dict, symbol: str, timeframe: str = '1h', lookback_periods: int = 100) -> Dict:
    """Calculate supply and demand zones using technical analysis"""
    df = get_historical_data(config, symbol, timeframe, lookback_periods)
    
    if df is None or len(df) < 20:
        return {'supply_zones': [], 'demand_zones': [], 'current_price': 0}
    
    # Calculate technical indicators
    df['sma_20'] = ta.sma(df['Close'], length=20)
    df['sma_50'] = ta.sma(df['Close'], length=50)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    df['volume_sma'] = ta.sma(df['Volume'], length=20)
    df['high_volume'] = df['Volume'] > (df['volume_sma'] * 1.5)
    
    # Identify pivot highs and lows
    df['pivot_high'] = df['High'] == df['High'].rolling(window=5, center=True).max()
    df['pivot_low'] = df['Low'] == df['Low'].rolling(window=5, center=True).min()
    
    supply_zones = []
    demand_zones = []
    
    current_price = df['Close'].iloc[-1]
    
    # Find supply zones (resistance levels)
    pivot_highs = df[df['pivot_high'] == True].copy()
    for _, pivot in pivot_highs.iterrows():
        zone_high = pivot['High']
        zone_low = zone_high * 0.995  # 0.5% zone width
        
        # Check if zone has been respected (price bounced off it)
        touches = len(df[(df['High'] >= zone_low) & (df['High'] <= zone_high)])
        
        if touches >= 2 and zone_low > current_price:  # Above current price
            supply_zones.append({
                'high': zone_high,
                'low': zone_low,
                'strength': touches,
                'timestamp': pivot['Datetime (UTC)'],
                'volume_confirmation': pivot['high_volume']
            })
    
    # Find demand zones (support levels)
    pivot_lows = df[df['pivot_low'] == True].copy()
    for _, pivot in pivot_lows.iterrows():
        zone_low = pivot['Low']
        zone_high = zone_low * 1.005  # 0.5% zone width
        
        # Check if zone has been respected
        touches = len(df[(df['Low'] >= zone_low) & (df['Low'] <= zone_high)])
        
        if touches >= 2 and zone_high < current_price:  # Below current price
            demand_zones.append({
                'high': zone_high,
                'low': zone_low,
                'strength': touches,
                'timestamp': pivot['Datetime (UTC)'],
                'volume_confirmation': pivot['high_volume']
            })
    
    # Sort zones by strength and proximity to current price
    supply_zones = sorted(supply_zones, key=lambda x: (x['strength'], -abs(current_price - x['low'])), reverse=True)
    demand_zones = sorted(demand_zones, key=lambda x: (x['strength'], -abs(current_price - x['high'])), reverse=True)
    
    return {
        'supply_zones': supply_zones[:5],  # Top 5 supply zones
        'demand_zones': demand_zones[:5],  # Top 5 demand zones
        'current_price': current_price,
        'sma_20': df['sma_20'].iloc[-1],
        'sma_50': df['sma_50'].iloc[-1],
        'rsi': df['rsi'].iloc[-1]
    }

def check_trend(config: Dict, symbol: str, timeframe: str = '1h') -> str:
    """Check trend using SMA analysis"""
    df = get_historical_data(config, symbol, timeframe, 50)
    
    if df is None or len(df) < 20:
        return 'sideways'
    
    # Calculate SMAs
    sma_bars = config.get('SMA_BARS', 20)
    df['sma'] = ta.sma(df['Close'], length=sma_bars)
    
    current_price = df['Close'].iloc[-1]
    current_sma = df['sma'].iloc[-1]
    
    buffer_pct = config.get('SMA_BUFFER_PCT', 0.10)
    
    if current_price > current_sma * (1 + buffer_pct):
        return 'up'
    elif current_price < current_sma * (1 - buffer_pct):
        return 'down'
    else:
        return 'sideways'

def check_market_conditions(config: Dict) -> Dict:
    """Check overall market conditions using major coins"""
    major_symbols = config.get('MARKET_CONDITIONS', {}).get('MAJOR_COINS_FOR_SENTIMENT', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    
    trends = {}
    bullish_count = 0
    
    for symbol in major_symbols:
        trend = check_trend(config, symbol, '4h')  # Use 4h for market sentiment
        trends[symbol] = trend
        
        if trend == 'up':
            bullish_count += 1
    
    total_symbols = len(major_symbols)
    bullish_percentage = bullish_count / total_symbols
    
    if bullish_percentage >= 0.67:  # 67% or more bullish
        market_condition = 'bullish'
    elif bullish_percentage <= 0.33:  # 33% or less bullish
        market_condition = 'bearish'
    else:
        market_condition = 'neutral'
    
    return {
        'condition': market_condition,
        'bullish_percentage': bullish_percentage,
        'individual_trends': trends
    }

def log_trade(config: Dict, symbol: str, action: str, quantity: float, price: float, value_usdt: float):
    """Log trade information"""
    trade_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'action': action,
        'quantity': quantity,
        'price': price,
        'value_usdt': value_usdt,
        'exchange': 'Binance'
    }
    
    # Save to trade history CSV
    trade_file = config.get('TRADE_HISTORY_CSV', 'csvs/binance/trade_history.csv')
    os.makedirs(os.path.dirname(trade_file), exist_ok=True)
    
    try:
        if os.path.exists(trade_file):
            df = pd.read_csv(trade_file)
            new_row = pd.DataFrame([trade_data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([trade_data])
        
        df.to_csv(trade_file, index=False)
        cprint(f"📊 Trade logged: {action} {quantity} {symbol} at ${price:.6f}", "cyan")
    
    except Exception as e:
        cprint(f"Error logging trade: {e}", "red")

def get_24h_stats(config: Dict, symbol: str) -> Optional[Dict]:
    """Get 24-hour trading statistics"""
    return make_binance_request(config, "/api/v3/ticker/24hr", {"symbol": symbol})

def is_market_open() -> bool:
    """Check if market is open (crypto markets are always open)"""
    return True

def print_pretty_json(data: Any):
    """Print formatted JSON data"""
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

# Compatibility functions to match original interface
def get_token_overview(config: Dict, symbol: str) -> Optional[Dict]:
    """Alias for token_overview for compatibility"""
    return token_overview(config, symbol)

def fetch_wallet_holdings_nosaving_names(config: Dict, wallet_address: str, token_mint_address: str) -> float:
    """Get position value for compatibility (returns USDT value)"""
    return get_position_value(config, token_mint_address)
