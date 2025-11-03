"""
Binance Historical Data Provider
=================================
Day 31: Binance-specific historical data fetcher with intelligent caching.

Fetches extensive historical OHLCV data from Binance API with:
- Intelligent caching to CSV files
- Chunked fetching to handle large date ranges
- Rate limit handling
- Automatic retry logic
- Duplicate removal
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
from typing import Optional, Dict
from pathlib import Path
import logging

from .provider import DataProvider

logger = logging.getLogger(__name__)


class BinanceDataProvider(DataProvider):
    """
    Binance historical data provider with caching.
    
    Fetches OHLCV data from Binance API with intelligent caching.
    """
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        cache_dir: str = "./binance_data",
        fetch_limit: int = 1000,
        rate_limit_delay: float = 0.2,
        retry_attempts: int = 3
    ):
        """
        Initialize Binance data provider.
        
        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            cache_dir: Directory to cache data files
            fetch_limit: Max candles per request (Binance limit: 1000)
            rate_limit_delay: Delay between requests (seconds)
            retry_attempts: Number of retry attempts
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.fetch_limit = fetch_limit
        self.rate_limit_delay = rate_limit_delay
        self.retry_attempts = retry_attempts
        
        self.base_url = 'https://api.binance.com/api/v3'
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-MBX-APIKEY': api_key})
    
    @property
    def column_mapping(self) -> Dict[str, str]:
        """Return column mapping for frameworks."""
        return {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
    
    def _get_timestamp(self) -> int:
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
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        amount = int("".join(filter(str.isdigit, timeframe)))
        unit = "".join(filter(str.isalpha, timeframe)).lower()
        
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000
        }
        
        return amount * multipliers.get(unit, 60 * 1000)
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for symbol/timeframe."""
        cache_filename = f'{symbol.replace("/", "_")}-{timeframe}-data.csv'
        return self.cache_dir / cache_filename
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_csv(cache_path, index_col='datetime', parse_dates=True)
            logger.info(f"Loaded {len(df)} rows from cache: {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"Error loading cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save data to cache file."""
        try:
            df.to_csv(cache_path)
            logger.info(f"Saved {len(df)} rows to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache {cache_path}: {e}")
    
    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> Optional[list]:
        """Fetch klines (candlestick) data from Binance."""
        url = f"{self.base_url}/klines"
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'startTime': start_time,
            'limit': min(limit, self.fetch_limit)
        }
        
        if end_time:
            params['endTime'] = end_time
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                time.sleep(self.rate_limit_delay)
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error fetching klines: {e}")
                    if attempt == self.retry_attempts - 1:
                        return None
            except Exception as e:
                logger.error(f"Error fetching klines: {e}")
                if attempt == self.retry_attempts - 1:
                    return None
                time.sleep(5)
        
        return None
    
    def fetch(
        self,
        symbol: str = 'BTCUSDT',
        timeframe: str = '1h',
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        limit: int = 2000
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start: Start date (defaults to limit weeks ago)
            end: End date (defaults to now)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_path = self._get_cache_path(symbol, timeframe)
        cached_data = self._load_from_cache(cache_path)
        
        # Determine date range
        if end is None:
            end = datetime.datetime.now(datetime.timezone.utc)
        
        if start is None:
            # Calculate start based on limit and timeframe
            timeframe_ms = self._timeframe_to_ms(timeframe)
            start = end - datetime.timedelta(milliseconds=limit * timeframe_ms)
        
        # Convert to timestamps
        start_timestamp_ms = int(start.timestamp() * 1000)
        end_timestamp_ms = int(end.timestamp() * 1000)
        
        # Check if we need to fetch new data
        if cached_data is not None and not cached_data.empty:
            cache_end = cached_data.index[-1]
            cache_end_ts = int(cache_end.timestamp() * 1000)
            
            # If cache covers requested range, return cached data
            if cache_end_ts >= end_timestamp_ms:
                filtered = cached_data[
                    (cached_data.index >= start) & (cached_data.index <= end)
                ]
                if not filtered.empty:
                    logger.info(f"Returning cached data for {symbol} ({timeframe})")
                    return filtered
        
        # Fetch new data
        logger.info(f"Fetching Binance data for {symbol} ({timeframe}) from {start} to {end}")
        
        all_ohlcv = []
        current_timestamp_ms = start_timestamp_ms
        timeframe_ms = self._timeframe_to_ms(timeframe)
        
        while current_timestamp_ms < end_timestamp_ms:
            chunk_data = self._fetch_klines(
                symbol=symbol,
                interval=timeframe,
                start_time=current_timestamp_ms,
                end_time=end_timestamp_ms,
                limit=self.fetch_limit
            )
            
            if not chunk_data:
                break
            
            all_ohlcv.extend(chunk_data)
            
            # Update timestamp for next chunk
            if len(chunk_data) < self.fetch_limit:
                break  # No more data
            
            last_candle_ts = chunk_data[-1][0]
            current_timestamp_ms = last_candle_ts + timeframe_ms
            
            # Don't exceed limit
            if len(all_ohlcv) >= limit:
                break
        
        if not all_ohlcv:
            logger.warning(f"No data fetched for {symbol}")
            return cached_data if cached_data is not None else pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if initial_rows != len(df):
            logger.info(f"Removed {initial_rows - len(df)} duplicate entries")
        
        df.sort_index(inplace=True)
        
        # Merge with cached data if exists
        if cached_data is not None and not cached_data.empty:
            combined = pd.concat([cached_data, df])
            df = combined[~combined.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
        
        # Filter to requested range
        df = df[(df.index >= start) & (df.index <= end)]
        
        # Save to cache
        if not df.empty:
            self._save_to_cache(df, cache_path)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data integrity."""
        if df.empty:
            return False
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Check for negative values
        if (df[['Open', 'High', 'Low', 'Close', 'Volume']] < 0).any().any():
            return False
        
        # Check OHLC relationships
        if not ((df['High'] >= df['Low']).all() and
                (df['High'] >= df['Open']).all() and
                (df['High'] >= df['Close']).all() and
                (df['Low'] <= df['Open']).all() and
                (df['Low'] <= df['Close']).all()):
            return False
        
        return True

