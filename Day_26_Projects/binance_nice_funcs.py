"""
Trading utility functions for Binance exchange interactions.

This module contains functions for token data retrieval, trade execution,
position management, and market analysis adapted for Binance API.
"""

import math
import time
import hmac
import hashlib
import pandas as pd
import requests
from termcolor import cprint
from urllib.parse import urlencode
from config_binance import *

try:
    from openai import OpenAI
    from Day_4_Projects import dontshare as d
    client = OpenAI(api_key=d.openai_key)
except ImportError:
    client = None

# --- Constants ---
BASE_URL = "https://api.binance.com"
FAPI_URL = "https://fapi.binance.com"

# --- Binance API Helper ---
class BinanceAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.headers = {"X-MBX-APIKEY": self.api_key}
    
    def _get_timestamp(self):
        return int(time.time() * 1000)
    
    def _sign(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method, endpoint, params=None, signed=False):
        url = f"{BASE_URL}{endpoint}"
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = self._get_timestamp()
            params['signature'] = self._sign(params)
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, data=params, timeout=10)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            cprint(f"API error: {e}", 'red')
            raise e

api = BinanceAPI(API_KEY, API_SECRET)

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
        data = api._request('GET', '/api/v3/ticker/price', {'symbol': symbol.upper()})
        return float(data['price'])
    except Exception as e:
        cprint(f"Error fetching price for {symbol}: {e}", 'red')
        return None

def get_position(symbol):
    base_asset = symbol.replace('USDT', '').replace('BUSD', '')
    
    try:
        account_info = api._request('GET', '/api/v3/account', signed=True)
        for balance in account_info['balances']:
            if balance['asset'].upper() == base_asset.upper():
                return float(balance['free'])
        return 0.0
    except Exception as e:
        cprint(f"Error fetching position for {symbol}: {e}", 'red')
        return 0.0

def get_decimals(symbol):
    try:
        exchange_info = api._request('GET', '/api/v3/exchangeInfo')
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol.upper():
                return {
                    'baseAssetPrecision': symbol_info['baseAssetPrecision'],
                    'quotePrecision': symbol_info['quotePrecision']
                }
        return {'baseAssetPrecision': 8, 'quotePrecision': 8}
    except Exception as e:
        return {'baseAssetPrecision': 8, 'quotePrecision': 8}

def fetch_wallet_holdings_og(unused=None):
    try:
        account_info = api._request('GET', '/api/v3/account', signed=True)
        holdings = []
        
        for balance in account_info['balances']:
            free_balance = float(balance['free'])
            if free_balance > 0:
                usd_value = free_balance
                if balance['asset'] not in ['USDT', 'BUSD', 'USDC']:
                    try:
                        price = token_price(f"{balance['asset']}USDT")
                        usd_value = free_balance * price if price else 0
                    except:
                        usd_value = 0
                
                holdings.append({
                    'Mint Address': balance['asset'],
                    'Amount': free_balance,
                    'USD Value': usd_value
                })
        
        return pd.DataFrame(holdings)
    except Exception as e:
        return pd.DataFrame()

def token_overview(symbol):
    try:
        ticker = api._request('GET', '/api/v3/ticker/24hr', {'symbol': symbol.upper()})
        return {
            'buy1h': 0,  # Not available in Binance
            'sell1h': 0,
            'trade1h': int(ticker['count']) / 24,  # Approximate hourly from 24h
            'buy_percentage': 50,  # Default
            'sell_percentage': 50,
            'minimum_trades_met': int(ticker['count']) > 100,
            'priceChangesXhrs': {'24h': float(ticker['priceChangePercent'])},
            'rug_pull': float(ticker['priceChangePercent']) < -80,
            'v24USD': float(ticker['quoteVolume']),
            'liquidity': float(ticker['quoteVolume'])
        }
    except Exception as e:
        return None

# --- Trading Functions ---
def market_buy(symbol, amount_usdt, slippage=None):
    try:
        result = api._request('POST', '/api/v3/order', {
            'symbol': symbol.upper(),
            'side': 'BUY',
            'type': 'MARKET',
            'quoteOrderQty': str(amount_usdt)
        }, signed=True)
        return f"Order ID: {result['orderId']}"
    except Exception as e:
        cprint(f"Buy error: {e}", 'red')
        raise e

def market_sell(symbol, quantity, slippage=None):
    try:
        result = api._request('POST', '/api/v3/order', {
            'symbol': symbol.upper(),
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': str(quantity)
        }, signed=True)
        return f"Order ID: {result['orderId']}"
    except Exception as e:
        cprint(f"Sell error: {e}", 'red')
        raise e

def elegant_entry(symbol, buy_under_price):
    print(f'Executing elegant entry for {symbol}...')
    
    pos = get_position(symbol)
    price = token_price(symbol)
    pos_usd = pos * price if price else 0
    
    if pos_usd >= (0.97 * USD_SIZE):
        cprint(f'Position already filled (${round(pos_usd, 2)})', 'green')
        return pos_usd
    
    while pos_usd < (0.97 * USD_SIZE) and price <= buy_under_price:
        size_needed = USD_SIZE - pos_usd
        chunk_size = min(size_needed, MAX_USD_ORDER_SIZE)
        
        try:
            for _ in range(ORDERS_PER_OPEN):
                market_buy(symbol, chunk_size)
                cprint(f'Chunk buy: ${chunk_size}', 'white', 'on_blue')
                time.sleep(0.1)
            
            time.sleep(TX_SLEEP)
        except Exception as e:
            cprint(f'Buy error: {e}. Retrying...', 'red')
            time.sleep(30)
        
        pos = get_position(symbol)
        price = token_price(symbol)
        pos_usd = pos * price if price else 0
    
    cprint(f'Entry complete. Final: ${round(pos_usd, 2)}', 'cyan')
    return pos_usd

def chunk_kill(symbol, max_usd_sell_size, slippage):
    balance = get_position(symbol)
    if balance <= 0:
        return True
    
    price = token_price(symbol)
    if not price:
        return False
    
    while balance > 0.001:
        usd_value = balance * price
        sell_size = min(balance, max_usd_sell_size / price)
        sell_size = round_down(sell_size, 6)
        
        try:
            market_sell(symbol, sell_size)
            time.sleep(2)
        except Exception as e:
            cprint(f'Sell error: {e}', 'red')
            time.sleep(5)
        
        balance = get_position(symbol)
        price = token_price(symbol)
    
    cprint(f'Position closed for {symbol}', 'green')
    return True

# --- Market Data ---
def fetch_candle_data_with_smas(symbol, interval='1d', limit=200, sma_windows=[20, 41]):
    try:
        klines = api._request('GET', '/api/v3/klines', {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        })
        
        df = pd.DataFrame(klines, columns=[
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Volume', 'Trades', 'Taker Buy Base',
            'Taker Buy Quote', 'Ignore'
        ])
        
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col].astype(float)
        
        df['Price'] = df['Close']
        
        for window in sma_windows:
            sma_col = f'SMA_{window}'
            df[sma_col] = df['Close'].rolling(window=window).mean()
            df[f'Price > {sma_col}'] = df['Close'] > df[sma_col]
        
        return df
    except Exception as e:
        return None

def get_btc_funding_rate():
    try:
        url = f"{FAPI_URL}/fapi/v1/premiumIndex?symbol=BTCUSDT"
        response = requests.get(url)
        data = response.json()
        
        funding_rate = float(data['lastFundingRate'])
        return (funding_rate * 3 * 365) * 100
    except Exception as e:
        return None

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
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None

# --- Utility Functions ---
def calculate_chunk_size(pos, price, target, max_chunk):
    pos_value = pos * price
    size_needed = target - pos_value
    chunk_size = min(size_needed, max_chunk) if size_needed > 0 else 0
    return chunk_size, size_needed

def pnl_close(symbol):
    balance = get_position(symbol)
    if balance <= 0:
        return False
    
    price = token_price(symbol)
    if not price:
        return False
    
    usd_value = balance * price
    tp = 2.0 * USD_SIZE  # Take profit at 2x
    sl = 0.5 * USD_SIZE  # Stop loss at 50%
    
    if usd_value > tp or usd_value < sl:
        try:
            chunk_kill(symbol, MAX_USD_ORDER_SIZE, SLIPPAGE_PERCENT)
            return True
        except:
            return False
    
    return False
