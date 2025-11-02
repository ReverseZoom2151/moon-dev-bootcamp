#!/usr/bin/env python3
"""
Bitfinex Historical Data Fetcher

Fetches historical OHLCV data from Bitfinex API with intelligent caching.
Supports multiple timeframes and extensive historical data collection.
"""

import pandas as pd
import datetime
import os
import time
import requests
import hmac
import hashlib
import base64
import json
from math import ceil
from typing import Optional, Dict, List

# Import Bitfinex configuration
try:
    from Day_26_Projects.bitfinex_config import API_KEY, API_SECRET, PRIMARY_SYMBOL
except ImportError:
    print("Warning: bitfinex_config not found, using default values")
    API_KEY = ""
    API_SECRET = ""
    PRIMARY_SYMBOL = "btcusd"

# --- Configuration ---
CONFIG = {
    'SYMBOL': PRIMARY_SYMBOL,  # Bitfinex symbol format (e.g., btcusd, ethusd)
    'TIMEFRAME': '1h',         # Timeframe: 1m, 5m, 15m, 30m, 1h, 3h, 6h, 12h, 1D, 7D, 14D, 1M
    'WEEKS_TO_FETCH': 52,      # Number of weeks of historical data (1 year)
    'CACHE_DIR': './bitfinex_data',  # Directory to save cached data files
    'FETCH_LIMIT': 5000,       # Max 5000 candles per request (Bitfinex limit)
    'RATE_LIMIT_DELAY': 0.5,   # Delay between requests (seconds)
    'RETRY_ATTEMPTS': 3,       # Number of retry attempts for failed requests
    'RETRY_DELAY': 10,         # Delay between retries (seconds)
}

# Bitfinex API configuration
BITFINEX_API = {
    'BASE_URL': 'https://api.bitfinex.com',
    'CANDLES_ENDPOINT': '/v2/candles/trade:{timeframe}:t{symbol}/hist',
    'SYMBOLS_ENDPOINT': '/v1/symbols',
    'RATE_LIMIT': 90,          # Requests per minute
    'CURRENT_REQUESTS': 0,     # Track current request count
}

# Ensure cache directory exists
os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)

# --- Bitfinex API Functions ---

class BitfinexDataFetcher:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BITFINEX_API['BASE_URL']
        self.session = requests.Session()
        self.request_count = 0
        self.last_request_time = time.time()
        
    def _nonce(self) -> str:
        """Generate nonce for authenticated requests."""
        return str(int(time.time() * 1000000))
    
    def _sign_payload(self, payload: Dict) -> Dict:
        """Sign payload for authenticated Bitfinex API requests."""
        j = json.dumps(payload)
        data = base64.standard_b64encode(j.encode('utf8'))
        h = hmac.new(
            self.api_secret.encode('utf8'), 
            data, 
            hashlib.sha384
        )
        signature = h.hexdigest()
        
        return {
            "X-BFX-APIKEY": self.api_key,
            "X-BFX-SIGNATURE": signature,
            "X-BFX-PAYLOAD": data
        }
    
    def _rate_limit_check(self):
        """Check and enforce rate limits."""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        if time_diff < 60:  # Within the same minute
            if self.request_count >= BITFINEX_API['RATE_LIMIT'] - 10:  # Buffer of 10 requests
                sleep_time = 60 - time_diff + 1  # Wait until next minute + 1 sec buffer
                print(f"⏳ Rate limit approaching. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        else:
            # More than a minute has passed, reset counter
            self.request_count = 0
            self.last_request_time = current_time
    
    def get_symbols(self) -> Optional[List[str]]:
        """Get available trading symbols."""
        try:
            self._rate_limit_check()
            url = f"{self.base_url}{BITFINEX_API['SYMBOLS_ENDPOINT']}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            self.request_count += 1
            return response.json()
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate that symbol exists on Bitfinex."""
        symbols = self.get_symbols()
        if not symbols:
            return False
        return symbol.lower() in symbols
    
    def get_candles(self, symbol: str, timeframe: str, start_time: int = None, end_time: int = None, limit: int = 5000, sort: int = 1) -> Optional[List]:
        """
        Fetch candles data from Bitfinex.
        
        Args:
            symbol: Trading pair (e.g., 'btcusd')
            timeframe: Candle timeframe (e.g., '1h', '1D')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds  
            limit: Number of candles to return (max 5000)
            sort: 1 for ascending, -1 for descending
        """
        try:
            self._rate_limit_check()
            
            # Format symbol for Bitfinex (add 't' prefix if not present)
            formatted_symbol = symbol.upper()
            if not formatted_symbol.startswith('T'):
                formatted_symbol = 'T' + formatted_symbol
            
            # Build URL
            endpoint = BITFINEX_API['CANDLES_ENDPOINT'].format(
                timeframe=timeframe,
                symbol=formatted_symbol
            )
            url = f"{self.base_url}{endpoint}"
            
            # Build parameters
            params = {
                'limit': min(limit, 5000),  # Bitfinex max is 5000
                'sort': sort
            }
            
            if start_time:
                params['start'] = start_time
            if end_time:
                params['end'] = end_time
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.request_count += 1
            
            data = response.json()
            
            # Handle error responses
            if isinstance(data, dict) and 'error' in data:
                print(f"Bitfinex API error: {data}")
                return None
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching candles: {e}")
            return None
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return None

# --- Helper Functions ---

def timeframe_to_seconds(timeframe: str) -> int:
    """Convert Bitfinex timeframe to seconds."""
    timeframe_map = {
        '1m': 60,
        '5m': 5 * 60,
        '15m': 15 * 60,
        '30m': 30 * 60,
        '1h': 60 * 60,
        '3h': 3 * 60 * 60,
        '6h': 6 * 60 * 60,
        '12h': 12 * 60 * 60,
        '1D': 24 * 60 * 60,
        '7D': 7 * 24 * 60 * 60,
        '14D': 14 * 24 * 60 * 60,
        '1M': 30 * 24 * 60 * 60  # Approximation
    }
    
    return timeframe_map.get(timeframe, 3600)  # Default to 1 hour

def get_cache_filename(symbol: str, timeframe: str, weeks: int) -> str:
    """Generate cache filename."""
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    return os.path.join(CONFIG['CACHE_DIR'], f'bitfinex_{safe_symbol}_{timeframe}_{weeks}w.csv')

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

def process_candles_data(raw_candles: List) -> pd.DataFrame:
    """Process raw candles data into pandas DataFrame."""
    if not raw_candles:
        return pd.DataFrame()
    
    # Bitfinex candles format:
    # [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
    # MTS is timestamp in milliseconds
    
    df = pd.DataFrame(raw_candles, columns=[
        'timestamp', 'Open', 'Close', 'High', 'Low', 'Volume'
    ])
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('datetime', inplace=True)
    
    # Reorder columns to standard OHLCV format
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
    
    print(f"🚀 Fetching fresh data from Bitfinex API...")
    print(f"📊 Symbol: {symbol}")
    print(f"⏰ Timeframe: {timeframe}")
    print(f"📅 Weeks: {weeks}")
    
    # Initialize Bitfinex fetcher
    fetcher = BitfinexDataFetcher(API_KEY, API_SECRET)
    
    # Validate symbol
    if not fetcher.validate_symbol(symbol):
        print(f"⚠️  Warning: Symbol {symbol} validation failed. Proceeding anyway...")
    
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
    
    # Fetch data in chunks (Bitfinex works better with time-based chunking)
    all_candles = []
    
    if api_calls_needed > 1:
        # Multiple calls needed - use time-based chunking
        chunk_duration_ms = (end_timestamp - start_timestamp) // api_calls_needed
        
        for i in range(api_calls_needed):
            chunk_start = start_timestamp + (i * chunk_duration_ms)
            chunk_end = start_timestamp + ((i + 1) * chunk_duration_ms)
            
            # Make sure last chunk goes to end
            if i == api_calls_needed - 1:
                chunk_end = end_timestamp
            
            print(f"📊 Fetching chunk {i+1}/{api_calls_needed}...")
            
            # Retry logic for each chunk
            for attempt in range(CONFIG['RETRY_ATTEMPTS']):
                try:
                    candles = fetcher.get_candles(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=chunk_start,
                        end_time=chunk_end,
                        limit=max_candles_per_call,
                        sort=1  # Ascending order
                    )
                    
                    if candles:
                        all_candles.extend(candles)
                        print(f"✅ Fetched {len(candles)} candles")
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
            
            # Rate limiting between chunks
            if i < api_calls_needed - 1:
                time.sleep(CONFIG['RATE_LIMIT_DELAY'])
    
    else:
        # Single call sufficient
        print(f"📊 Fetching all data in single request...")
        
        for attempt in range(CONFIG['RETRY_ATTEMPTS']):
            try:
                candles = fetcher.get_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_timestamp,
                    end_time=end_timestamp,
                    limit=max_candles_per_call,
                    sort=1
                )
                
                if candles:
                    all_candles.extend(candles)
                    print(f"✅ Fetched {len(candles)} candles")
                    break
                else:
                    print("⚠️ No data returned")
                    break
                    
            except Exception as e:
                print(f"❌ Attempt {attempt+1} failed: {e}")
                if attempt < CONFIG['RETRY_ATTEMPTS'] - 1:
                    print(f"🔄 Retrying in {CONFIG['RETRY_DELAY']} seconds...")
                    time.sleep(CONFIG['RETRY_DELAY'])
                else:
                    print(f"💥 Failed to fetch data after {CONFIG['RETRY_ATTEMPTS']} attempts")
    
    if not all_candles:
        print("❌ No data was fetched")
        return pd.DataFrame()
    
    print(f"✅ Total candles fetched: {len(all_candles):,}")
    
    # Process data
    dataframe = process_candles_data(all_candles)
    
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
    print(f"📊 BITFINEX DATA ANALYSIS: {symbol.upper()}")
    print(f"{'='*60}")
    print(f"📈 Total Records: {len(df):,}")
    print(f"📅 Date Range: {df.index.min().strftime('%Y-%m-%d %H:%M')} to {df.index.max().strftime('%Y-%m-%d %H:%M')}")
    print(f"💰 Price Range: ${df['Low'].min():.8g} - ${df['High'].max():.8g}")
    print(f"📊 Avg Volume: {df['Volume'].mean():,.0f}")
    print(f"📈 Price Change: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:+.2f}%")
    
    # Volume analysis
    volume_usd = df['Volume'] * df['Close']
    print(f"💵 Avg Daily Volume (USD): ${volume_usd.mean():,.0f}")
    
    print(f"\n📈 FIRST 5 RECORDS:")
    print(df.head().to_string())
    
    print(f"\n📉 LAST 5 RECORDS:")
    print(df.tail().to_string())
    
    # Basic statistics
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(df.describe().round(8))
    
    # Volatility analysis
    df['Returns'] = df['Close'].pct_change()
    daily_volatility = df['Returns'].std()
    annualized_volatility = daily_volatility * (365 ** 0.5)
    print(f"\n📊 VOLATILITY ANALYSIS:")
    print(f"Daily Volatility: {daily_volatility:.4f} ({daily_volatility*100:.2f}%)")
    print(f"Annualized Volatility: {annualized_volatility:.4f} ({annualized_volatility*100:.1f}%)")

# --- Main Execution ---

def main():
    """Main execution function."""
    symbol = CONFIG['SYMBOL']
    timeframe = CONFIG['TIMEFRAME']
    weeks = CONFIG['WEEKS_TO_FETCH']
    
    print("🔶" + "="*70 + "🔶")
    print("🚀        BITFINEX HISTORICAL DATA FETCHER         🚀")
    print("🔶" + "="*70 + "🔶")
    
    try:
        # Fetch historical data
        historical_data = get_historical_data(symbol, timeframe, weeks)
        
        if not historical_data.empty:
            analyze_data(historical_data, symbol)
            print(f"\n✅ Successfully fetched and cached {len(historical_data):,} records!")
            print(f"📁 Cache location: {CONFIG['CACHE_DIR']}")
        else:
            print("\n❌ No historical data was fetched.")
    
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
