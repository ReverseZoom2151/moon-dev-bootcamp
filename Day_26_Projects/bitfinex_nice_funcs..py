"""
Trading utility functions for Bitfinex exchange interactions.

This module contains functions for token data retrieval, trade execution,
position management, and market analysis adapted for Bitfinex API.
"""

import json
import math
import time
import hmac
import hashlib
import base64
import pandas as pd
import requests
from termcolor import cprint
from datetime import datetime
from config_bitfinex import *

try:
    from openai import OpenAI
    from Day_4_Projects import dontshare as d
    client = OpenAI(api_key=d.openai_key)
except ImportError:
    client = None

# --- Constants ---
BASE_URL = "https://api.bitfinex.com"

# --- Bitfinex API Helper ---
class BitfinexAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
    
    def _nonce(self):
        return str(int(time.time() * 1000000))
    
    def _sign_payload(self, payload):
        j = json.dumps(payload)
        data = base64.standard_b64encode(j.encode('utf8'))
        h = hmac.new(self.api_secret.encode('utf8'), data, hashlib.sha384)
        signature = h.hexdigest()
        return {
            "X-BFX-APIKEY": self.api_key,
            "X-BFX-SIGNATURE": signature,
            "X-BFX-PAYLOAD": data
        }
    
    def _public_request(self, endpoint):
        url = f"{BASE_URL}{endpoint}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            cprint(f"Bitfinex public API error: {e}", 'red')
            raise e
    
    def _private_request(self, endpoint, params=None):
        url = f"{BASE_URL}{endpoint}"
        if params is None:
            params = {}
        
        nonce = self._nonce()
        payload = {"request": endpoint, "nonce": nonce, **params}
        headers = self._sign_payload(payload)
        
        try:
            response = requests.post(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            cprint(f"Bitfinex private API error: {e}", 'red')
            raise e

api = BitfinexAPI(API_KEY, API_SECRET)

# --- Core Functions ---
def round_down(value, decimals):
    factor = 10 ** decimals
    return math.floor(value * factor) / factor

def print_pretty_json(data):
    import pprint
    pprint.pprint(data)

# --- Data Retrieval ---
def token_price(symbol):
    try:
        data = api._public_request(f'/v1/pubticker/{symbol.lower()}')
        return float(data['last_price'])
    except Exception as e:
        cprint(f"Error fetching price for {symbol}: {e}", 'red')
        return None

def get_position(symbol):
    # Extract base currency from symbol (e.g., 'btc' from 'btcusd')
    base_currency = symbol.replace('usd', '').replace('USD', '').lower()
    
    try:
        balances = api._private_request('/v1/balances')
        for balance in balances:
            if (balance['currency'].lower() == base_currency and 
                balance['type'] == 'exchange'):
                return float(balance['available'])
        return 0.0
    except Exception as e:
        cprint(f"Error fetching position for {symbol}: {e}", 'red')
        return 0.0

def get_decimals(symbol):
    # Bitfinex uses standard precision for most pairs
    # Return a reasonable default precision mapping
    return {
        'price_precision': 5,
        'amount_precision': 8
    }

def fetch_wallet_holdings_og(unused=None):
    try:
        balances = api._private_request('/v1/balances')
        holdings = []
        
        for balance in balances:
            if balance['type'] == 'exchange':
                available = float(balance['available'])
                if available > 0:
                    currency = balance['currency'].upper()
                    
                    # Calculate USD value
                    usd_value = available
                    if currency not in ['USD', 'USDT']:
                        try:
                            price = token_price(f"{currency.lower()}usd")
                            usd_value = available * price if price else 0
                        except:
                            usd_value = 0
                    
                    holdings.append({
                        'Mint Address': currency,
                        'Amount': available,
                        'USD Value': usd_value
                    })
        
        return pd.DataFrame(holdings)
    except Exception as e:
        cprint(f"Error fetching holdings: {e}", 'red')
        return pd.DataFrame()

def token_overview(symbol):
    try:
        # Get 24hr ticker statistics
        ticker = api._public_request(f'/v1/stats/{symbol.lower()}')
        
        # Get recent trades for activity assessment
        trades = api._public_request(f'/v1/trades/{symbol.lower()}?limit_trades=100')
        
        # Calculate basic metrics
        volume_24h = float(ticker[0]['volume']) if ticker else 0
        
        # Approximate hourly trades from recent activity
        recent_trades = len([t for t in trades if 
                           datetime.now().timestamp() - float(t['timestamp']) < 3600])
        
        return {
            'buy1h': recent_trades // 2,  # Approximate
            'sell1h': recent_trades // 2,
            'trade1h': recent_trades,
            'buy_percentage': 50,  # Default
            'sell_percentage': 50,
            'minimum_trades_met': recent_trades > 10,
            'priceChangesXhrs': {'24h': 0},  # Not easily available
            'rug_pull': False,
            'v24USD': volume_24h,
            'liquidity': volume_24h
        }
    except Exception as e:
        cprint(f"Error fetching overview for {symbol}: {e}", 'red')
        return None

def token_security_info(symbol):
    # Centralized exchanges don't have DeFi security concerns
    # Return basic trading status
    return {
        'symbol': symbol,
        'status': 'ACTIVE',
        'tradingEnabled': True,
        'exchange': 'Bitfinex'
    }

def token_creation_info(symbol):
    # Not applicable for centralized exchanges
    return {
        'symbol': symbol,
        'exchange': 'Bitfinex',
        'listing_date': 'N/A'
    }

# --- Trading Functions ---
def market_buy(symbol, amount_usd, slippage=None):
    try:
        result = api._private_request('/v1/order/new', {
            'symbol': symbol.lower(),
            'amount': str(amount_usd),
            'price': '1',
            'exchange': 'bitfinex',
            'side': 'buy',
            'type': 'market'
        })
        cprint(f"Market buy order placed for {symbol}", 'green')
        return f"Order ID: {result.get('id', 'N/A')}"
    except Exception as e:
        cprint(f"Buy error for {symbol}: {e}", 'red')
        raise e

def market_sell(symbol, quantity, slippage=None):
    try:
        result = api._private_request('/v1/order/new', {
            'symbol': symbol.lower(),
            'amount': str(quantity),
            'price': '1',
            'exchange': 'bitfinex',
            'side': 'sell',
            'type': 'market'
        })
        cprint(f"Market sell order placed for {symbol}", 'green')
        return f"Order ID: {result.get('id', 'N/A')}"
    except Exception as e:
        cprint(f"Sell error for {symbol}: {e}", 'red')
        raise e

def elegant_entry(symbol, buy_under_price):
    print(f'Executing elegant entry for {symbol}...')
    
    pos = get_position(symbol)
    price = token_price(symbol)
    pos_usd = pos * price if price else 0
    
    if pos_usd >= (0.97 * USD_SIZE):
        cprint(f'Position already filled (${round(pos_usd, 2)})', 'green')
        return pos_usd
    
    print(f'Initial: Position={round(pos, 2)}, Price=${round(price, 8)}, '
          f'Value=${round(pos_usd, 2)}, Target=${USD_SIZE}')
    
    while pos_usd < (0.97 * USD_SIZE) and price <= buy_under_price:
        size_needed = USD_SIZE - pos_usd
        chunk_size = min(size_needed, MAX_USD_ORDER_SIZE)
        
        try:
            for _ in range(ORDERS_PER_OPEN):
                market_buy(symbol, chunk_size)
                cprint(f'Chunk buy: ${chunk_size}', 'white', 'on_blue')
                time.sleep(0.5)
            
            time.sleep(TX_SLEEP)
            
        except Exception as e:
            cprint(f'Buy error: {e}. Retrying in 30s...', 'red')
            time.sleep(30)
            
            try:
                for _ in range(ORDERS_PER_OPEN):
                    market_buy(symbol, chunk_size)
                    time.sleep(0.5)
                time.sleep(TX_SLEEP)
            except Exception as retry_error:
                cprint(f'Final buy error: {retry_error}', 'white', 'on_red')
                break
        
        # Update position
        pos = get_position(symbol)
        price = token_price(symbol)
        pos_usd = pos * price if price else 0
    
    final_status = "filled" if pos_usd >= (0.97 * USD_SIZE) else "partial"
    cprint(f'Entry complete - {final_status}. Final: ${round(pos_usd, 2)}', 'cyan')
    return pos_usd

def elegant_time_entry(symbol, buy_under, seconds_to_sleep):
    # Similar to elegant_entry but with custom sleep time
    print(f'Time-based elegant entry for {symbol}...')
    
    pos = get_position(symbol)
    price = token_price(symbol)
    pos_usd = pos * price if price else 0
    
    if pos_usd >= (0.97 * USD_SIZE):
        print('Position already filled')
        return pos_usd
    
    while pos_usd < (0.97 * USD_SIZE) and price <= buy_under:
        size_needed = USD_SIZE - pos_usd
        chunk_size = min(size_needed, MAX_USD_ORDER_SIZE)
        
        try:
            for _ in range(ORDERS_PER_OPEN):
                market_buy(symbol, chunk_size)
                cprint(f'Chunk buy: {symbol} ${chunk_size}', 'white', 'on_blue')
                time.sleep(1)
            
            time.sleep(seconds_to_sleep)
            
        except Exception as e:
            cprint(f'Buy error: {e}', 'red')
            time.sleep(30)
        
        pos = get_position(symbol)
        price = token_price(symbol)
        pos_usd = pos * price if price else 0
    
    return pos_usd

def chunk_kill(symbol, max_usd_sell_size, slippage):
    balance = get_position(symbol)
    if balance <= 0:
        print(f'No position to close for {symbol}')
        return True
    
    cprint(f'Closing position: {balance} {symbol}', 'white', 'on_magenta')
    
    price = token_price(symbol)
    if not price:
        cprint(f'Cannot get price for {symbol}', 'red')
        return False
    
    while balance > 0.001:  # Close until very small amount remains
        usd_value = balance * price
        
        # Calculate sell size for this chunk
        if usd_value < max_usd_sell_size:
            sell_size = balance
        else:
            sell_size = max_usd_sell_size / price
        
        sell_size = round_down(sell_size, 6)
        
        cprint(f'Selling {sell_size} {symbol} (~${round(sell_size * price, 2)})', 'white', 'on_blue')
        
        try:
            for _ in range(3):  # Multiple attempts
                market_sell(symbol, sell_size)
                time.sleep(1)
            
            time.sleep(5)  # Wait for execution
            
        except Exception as e:
            cprint(f'Sell error: {e}', 'red')
            time.sleep(5)
        
        # Update position
        balance = get_position(symbol)
        price = token_price(symbol)
    
    cprint(f'Position closed for {symbol}', 'white', 'on_green')
    return True

def chunk_kill_mm(symbol, max_usd_sell_size, slippage, sell_over_p, seconds_to_sleep):
    """Market making version of chunk_kill"""
    balance = get_position(symbol)
    if balance <= 0:
        return True
    
    price = token_price(symbol)
    if not price or price <= sell_over_p:
        print(f'Price ${price} not above threshold ${sell_over_p}')
        return False
    
    while balance > 0.001 and price > sell_over_p:
        usd_value = balance * price
        sell_size = min(balance, max_usd_sell_size / price)
        sell_size = round_down(sell_size, 6)
        
        try:
            market_sell(symbol, sell_size)
            cprint(f'MM sell: {sell_size} {symbol}', 'white', 'on_blue')
            time.sleep(seconds_to_sleep)
        except Exception as e:
            cprint(f'MM sell error: {e}', 'red')
            time.sleep(5)
        
        balance = get_position(symbol)
        price = token_price(symbol)
    
    return balance <= 0.001

# --- Market Data ---
def fetch_candle_data_with_smas(symbol, timeframe='1D', limit=200, sma_windows=[20, 41]):
    try:
        # Bitfinex candles endpoint
        endpoint = f'/v2/candles/trade:{timeframe}:t{symbol.upper()}/hist'
        params = f'?limit={limit}'
        
        data = api._public_request(endpoint + params)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'Timestamp', 'Open', 'Close', 'High', 'Low', 'Volume'
        ])
        
        # Convert timestamp and sort
        df['Datetime (UTC)'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df.sort_values('Timestamp')
        
        # Add price column
        df['Price'] = df['Close']
        
        # Calculate SMAs
        for window in sma_windows:
            sma_col = f'SMA_{window}'
            df[sma_col] = df['Close'].rolling(window=window).mean()
            df[f'Price > {sma_col}'] = df['Close'] > df[sma_col]
        
        return df
    except Exception as e:
        cprint(f"Error fetching candle data for {symbol}: {e}", 'red')
        return None

# --- Position Management ---
def pnl_close(symbol):
    balance = get_position(symbol)
    if balance <= 0:
        return False
    
    price = token_price(symbol)
    if not price:
        return False
    
    usd_value = balance * price
    
    # Define thresholds
    tp = 2.0 * USD_SIZE  # Take profit at 2x
    sl = 0.5 * USD_SIZE  # Stop loss at 50%
    
    print(f'{symbol}: Position={round(balance, 6)}, Value=${round(usd_value, 2)}')
    print(f'TP=${round(tp, 2)}, SL=${round(sl, 2)}')
    
    if usd_value > tp:
        cprint(f'Take profit triggered for {symbol}', 'white', 'on_green')
        try:
            chunk_kill(symbol, MAX_USD_ORDER_SIZE, SLIPPAGE_PERCENT)
            return True
        except:
            return False
    
    elif usd_value < sl and usd_value > 1:
        cprint(f'Stop loss triggered for {symbol}', 'white', 'on_blue')
        try:
            chunk_kill(symbol, MAX_USD_ORDER_SIZE, SLIPPAGE_PERCENT)
            return True
        except:
            return False
    
    return False

def close_all_positions():
    print('Closing all open positions...')
    
    holdings_df = fetch_wallet_holdings_og()
    if holdings_df.empty:
        print('No positions found')
        return 0
    
    # Filter out USD and excluded assets
    exclude_assets = ['USD', 'USDT'] + DO_NOT_TRADE_LIST
    positions_to_close = holdings_df[~holdings_df['Mint Address'].isin(exclude_assets)]
    
    closed_count = 0
    for index, row in positions_to_close.iterrows():
        asset = row['Mint Address']
        usd_value = row['USD Value']
        
        if usd_value < 1:
            continue
        
        try:
            symbol = f"{asset.lower()}usd"
            if chunk_kill(symbol, MAX_USD_ORDER_SIZE, SLIPPAGE_PERCENT):
                closed_count += 1
        except Exception as e:
            print(f'Error closing {asset}: {e}')
    
    print(f'Closed {closed_count} positions')
    return closed_count

# --- Utility Functions ---
def calculate_chunk_size(pos, price, target, max_chunk):
    pos_value = pos * price
    size_needed = target - pos_value
    chunk_size = min(size_needed, max_chunk) if size_needed > 0 else 0
    return chunk_size, size_needed

# --- AI Functions ---
def vibe_check(name):
    if not client or not name:
        return None
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "Rate token names 1-10 for meme potential."},
                {"role": "user", "content": f"Rate this token name: {name}. Reply with just a number."}
            ]
        )
        
        score_text = response.choices[0].message.content.strip()
        score = int(score_text.split()[0] if ' ' in score_text else score_text)
        print(f"Vibe score for '{name}': {score}/10")
        return score
    except Exception as e:
        print(f"Vibe check error for {name}: {e}")
        return None

def gpt4(prompt, df):
    if not client:
        return None
    
    try:
        df_str = df.tail(10).to_string() if df is not None else "No data"
        
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "Trading expert. Return only True or False."},
                {"role": "user", "content": f"{prompt}\n\nData:\n{df_str}\n\nBuy (True) or sell (False)?"}
            ]
        )
        
        decision = response.choices[0].message.content.strip()
        print(f"AI decision: {decision}")
        return decision
    except Exception as e:
        print(f"AI decision error: {e}")
        return None

def ai_vibe_check(df, min_vibe_score):
    if df is None or df.empty or 'name' not in df.columns:
        return df
    
    print(f"Running vibe check on {len(df)} tokens...")
    df['Vibe Score'] = df['name'].apply(vibe_check)
    
    filtered_df = df[df['Vibe Score'] >= min_vibe_score]
    print(f"Filtered to {len(filtered_df)} tokens with vibe >= {min_vibe_score}")
    
    return filtered_df

def serialize_df_for_prompt(df, max_rows=20):
    if df is None or df.empty:
        return "No data available."
    
    # Focus on relevant columns
    key_columns = ['Datetime (UTC)', 'Open', 'High', 'Low', 'Close', 'Volume']
    available_columns = [col for col in key_columns if col in df.columns]
    
    df_portion = df.tail(max_rows)[available_columns]
    return df_portion.to_string(index=False)

# --- Funding Rate (from futures if available) ---
def get_btc_funding_rate():
    # Bitfinex doesn't have the same funding rate concept as perpetual futures
    # Return None or implement alternative metric
    try:
        # Get lending rates as alternative
        lending_data = api._public_request('/v1/lendbook/btc')
        if lending_data and 'asks' in lending_data:
            avg_rate = sum(float(ask['rate']) for ask in lending_data['asks'][:5]) / 5
            return avg_rate * 365 * 100  # Annualized percentage
        return None
    except:
        return None
