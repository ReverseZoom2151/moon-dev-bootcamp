#!/usr/bin/env python3
"""
Binance Historical Data Fetcher

Fetches historical OHLCV data from Binance API with intelligent caching.
Supports multiple timeframes and extensive historical data collection.
"""

import pandas as pd
import datetime
import os
import time
import requests
import hmac
import hashlib
from urllib.parse import urlencode
from math import ceil
from typing import Optional, Dict, List

# Import Binance configuration
try:
    from Day_26_Projects.binance_config import API_KEY, API_SECRET, PRIMARY_SYMBOL
except ImportError:
    print("Warning: binance_config not found, using default values")
    API_KEY = ""
    API_SECRET = ""
    PRIMARY_SYMBOL = "BTCUSDT"

# --- Configuration ---
CONFIG = {
    'SYMBOL': PRIMARY_SYMBOL,  # Binance symbol format (e.g., BTCUSDT, ETHUSDT)
    'TIMEFRAME': '1h',         # Timeframe: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    'WEEKS_TO_FETCH': 52,      # Number of weeks of historical data (1 year)
    'CACHE_DIR': './binance_data',  # Directory to save cached data files
    'FETCH_LIMIT': 1000,       # Max 1000 klines per request (Binance limit)
    'RATE_LIMIT_DELAY': 0.2,   # Delay between requests (seconds)
    'RETRY_ATTEMPTS': 3,       # Number of retry attempts for failed requests
    'RETRY_DELAY': 5,          # Delay between retries (seconds)
}

# Binance API configuration
BINANCE_API = {
    'BASE_URL': 'https://api.binance.com/api/v3',
    'KLINES_ENDPOINT': '/klines',
    'EXCHANGE_INFO_ENDPOINT': '/exchangeInfo',
    'WEIGHT_LIMIT': 1200,      # Request weight limit per minute
    'CURRENT_WEIGHT': 0,       # Track current weight usage
}

# Ensure cache directory exists
os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)

# --- Binance API Functions ---

class BinanceDataFetcher:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BINANCE_API['BASE_URL']
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': self.api_key}) if api_key else None
        
    def _get_timestamp(self):
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    def _sign_request(self, params: Dict) -> str:
        """Sign request parameters for authenticated endpoints."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def get_exchange_info(self) -> Optional[Dict]:
        """Get exchange trading rules and symbol information."""
        try:
            url = f"{self.base_url}{BINANCE_API['EXCHANGE_INFO_ENDPOINT']}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching exchange info: {e}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate that symbol exists on Binance."""
        exchange_info = self.get_exchange_info()
        if not exchange_info:
            return False
        
        symbols = [s['symbol'] for s in exchange_info.get('symbols', [])]
        return symbol.upper() in symbols
    
    def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int = None, limit: int = 1000) -> Optional[List]:
        """Fetch klines (candlestick) data from Binance."""
        try:
            url = f"{self.base_url}{BINANCE_API['KLINES_ENDPOINT']}"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'startTime': start_time,
                'limit': min(limit, 1000)  # Binance max is 1000
            }
            
            if end_time:
                params['endTime'] = end_time
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Update weight tracking (klines endpoint weight is 1)
            BINANCE_API['CURRENT_WEIGHT'] += 1
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching klines: {e}")
            return None
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return None

# --- Helper Functions ---

def timeframe_to_seconds(timeframe: str) -> int:
    """Convert Binance interval to seconds."""
    interval_map = {
        '1m': 60,
        '3m': 3 * 60,
        '5m': 5 * 60,
        '15m': 15 * 60,
        '30m': 30 * 60,
        '1h': 60 * 60,
        '2h': 2 * 60 * 60,
        '4h': 4 * 60 * 60,
        '6h': 6 * 60 * 60,
        '8h': 8 * 60 * 60,
        '12h': 12 * 60 * 60,
        '1d': 24 * 60 * 60,
        '3d': 3 * 24 * 60 * 60,
        '1w': 7 * 24 * 60 * 60,
        '1M': 30 * 24 * 60 * 60  # Approximation
    }
    
    return interval_map.get(timeframe, 3600)  # Default to 1 hour

def get_cache_filename(symbol: str, timeframe: str, weeks: int) -> str:
    """Generate cache filename."""
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    return os.path.join(CONFIG['CACHE_DIR'], f'binance_{safe_symbol}_{timeframe}_{weeks}w.csv')

def load_cached_data(cache_filename: str) -> Optional[pd.DataFrame]:
    """Load data from cache file."""
    if not os.path.exists(cache_filename):
        return None
    
    try:
        print(f"📁 Loading cached data from: {cache_filename}")
        df = pd.read_csv(cache_filename, index_col='datetime', parse_dates=True)
        print(f"✅ Loaded {len(df)} cached records")
        return df
    except Exception as e:
        print(f"❌ Error loading cache file {cache_filename}: {e}")
        return None

def save_data_to_cache(dataframe: pd.DataFrame, cache_filename: str) -> bool:
    """Save DataFrame to cache file."""
    try:
        dataframe.to_csv(cache_filename)
        print(f"💾 Data saved to cache: {cache_filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving to cache: {e}")
        return False

def process_klines_data(raw_klines: List) -> pd.DataFrame:
    """Process raw klines data into pandas DataFrame."""
    if not raw_klines:
        return pd.DataFrame()
    
    # Binance klines format:
    # [open_time, open, high, low, close, volume, close_time, quote_volume, 
    #  trades, taker_buy_base_volume, taker_buy_quote_volume, ignore]
    
    df = pd.DataFrame(raw_klines, columns=[
        'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df.set_index('datetime', inplace=True)
    
    # Keep only OHLCV columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any NaN values
    df.dropna(inplace=True)
    
    # Remove duplicates
    initial_len = len(df)
    df = df[~df.index.duplicated(keep='first')]
    if len(df) < initial_len:
        print(f"🔄 Removed {initial_len - len(df)} duplicate records")
    
    # Sort by datetime
    df.sort_index(inplace=True)
    
    return df

def get_historical_data(symbol: str, timeframe: str, weeks: int) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given symbol, timeframe, and number of weeks.
    Uses intelligent caching to avoid redundant API calls.
    """
    cache_filename = get_cache_filename(symbol, timeframe, weeks)
    
    # Try to load from cache first
    cached_data = load_cached_data(cache_filename)
    if cached_data is not None and not cached_data.empty:
        return cached_data
    
    print(f"🚀 Fetching fresh data from Binance API...")
    print(f"📊 Symbol: {symbol}")
    print(f"⏰ Timeframe: {timeframe}")
    print(f"📅 Weeks: {weeks}")
    
    # Initialize Binance fetcher
    fetcher = BinanceDataFetcher(API_KEY, API_SECRET)
    
    # Validate symbol
    if not fetcher.validate_symbol(symbol):
        raise ValueError(f"❌ Symbol {symbol} not found on Binance")
    
    # Calculate time parameters
    interval_seconds = timeframe_to_seconds(timeframe)
    total_seconds = weeks * 7 * 24 * 60 * 60
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(seconds=total_seconds)
    
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    # Calculate number of API calls needed
    max_candles_per_call = CONFIG['FETCH_LIMIT']
    total_candles_needed = total_seconds // interval_seconds
    api_calls_needed = ceil(total_candles_needed / max_candles_per_call)
    
    print(f"📈 Estimated candles needed: {total_candles_needed:,}")
    print(f"🔄 API calls required: {api_calls_needed}")
    print(f"📅 Data range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Fetch data in chunks
    all_klines = []
    current_start = start_timestamp
    
    for i in range(api_calls_needed):
        # Calculate end time for this chunk
        chunk_duration = max_candles_per_call * interval_seconds * 1000  # Convert to ms
        current_end = min(current_start + chunk_duration, end_timestamp)
        
        print(f"📊 Fetching chunk {i+1}/{api_calls_needed}...")
        
        # Retry logic for each chunk
        for attempt in range(CONFIG['RETRY_ATTEMPTS']):
            try:
                klines = fetcher.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=current_start,
                    end_time=current_end,
                    limit=max_candles_per_call
                )
                
                if klines:
                    all_klines.extend(klines)
                    print(f"✅ Fetched {len(klines)} candles")
                    
                    # Update start time for next chunk
                    if klines:
                        last_close_time = klines[-1][6]  # close_time from klines
                        current_start = last_close_time + 1
                    
                    break  # Success, exit retry loop
                else:
                    print(f"⚠️ No data returned for chunk {i+1}")
                    break
                    
            except Exception as e:
                print(f"❌ Attempt {attempt+1} failed: {e}")
                if attempt < CONFIG['RETRY_ATTEMPTS'] - 1:
                    print(f"🔄 Retrying in {CONFIG['RETRY_DELAY']} seconds...")
                    time.sleep(CONFIG['RETRY_DELAY'])
                else:
                    print(f"💥 Failed to fetch chunk {i+1} after {CONFIG['RETRY_ATTEMPTS']} attempts")
        
        # Rate limiting
        if i < api_calls_needed - 1:  # Don't delay after last request
            time.sleep(CONFIG['RATE_LIMIT_DELAY'])
        
        # Stop if we've reached the end timestamp
        if current_start >= end_timestamp:
            break
    
    if not all_klines:
        print("❌ No data was fetched")
        return pd.DataFrame()
    
    print(f"✅ Total klines fetched: {len(all_klines):,}")
    
    # Process data
    dataframe = process_klines_data(all_klines)
    
    if dataframe.empty:
        print("❌ No valid data after processing")
        return dataframe
    
    # Save to cache
    save_data_to_cache(dataframe, cache_filename)
    
    return dataframe

def analyze_data(df: pd.DataFrame, symbol: str) -> None:
    """Analyze and display data statistics."""
    if df.empty:
        print("❌ No data to analyze")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 BINANCE DATA ANALYSIS: {symbol}")
    print(f"{'='*60}")
    print(f"📈 Total Records: {len(df):,}")
    print(f"📅 Date Range: {df.index.min().strftime('%Y-%m-%d %H:%M')} to {df.index.max().strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 Price Range: ${df['Low'].min():.8g} - ${df['High'].max():.8g}")
    print(f"📊 Avg Volume: {df['Volume'].mean():,.0f}")
    print(f"📈 Price Change: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:+.2f}%")
    
    print(f"\n📈 FIRST 5 RECORDS:")
    print(df.head().to_string())
    
    print(f"\n📉 LAST 5 RECORDS:")
    print(df.tail().to_string())
    
    # Basic statistics
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(df.describe().round(8))

# --- Main Execution ---

def main():
    """Main execution function."""
    symbol = CONFIG['SYMBOL']
    timeframe = CONFIG['TIMEFRAME']
    weeks = CONFIG['WEEKS_TO_FETCH']
    
    print("🟠" + "="*70 + "🟠")
    print("🚀         BINANCE HISTORICAL DATA FETCHER         🚀")
    print("🟠" + "="*70 + "🟠")
    
    try:
        # Fetch historical data
        historical_data = get_historical_data(symbol, timeframe, weeks)
        
        if not historical_data.empty:
            analyze_data(historical_data, symbol)
            print(f"\n✅ Successfully fetched and cached {len(historical_data):,} records!")
        else:
            print("\n❌ No historical data was fetched.")
    
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
