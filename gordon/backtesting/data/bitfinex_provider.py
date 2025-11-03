"""
Bitfinex Historical Data Provider
==================================
Day 31: Bitfinex-specific historical data fetcher with intelligent caching.

Fetches extensive historical OHLCV data from Bitfinex API with:
- Intelligent caching to CSV files
- Chunked fetching to handle large date ranges
- Rate limit handling
- Automatic retry logic
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
from typing import Optional, Dict
from pathlib import Path
import logging

from .provider import DataProvider

logger = logging.getLogger(__name__)


class BitfinexDataProvider(DataProvider):
    """
    Bitfinex historical data provider with caching.
    
    Fetches OHLCV data from Bitfinex API with intelligent caching.
    """
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        cache_dir: str = "./bitfinex_data",
        fetch_limit: int = 5000,
        rate_limit_delay: float = 0.5,
        retry_attempts: int = 3
    ):
        """
        Initialize Bitfinex data provider.
        
        Args:
            api_key: Bitfinex API key (optional for public endpoints)
            api_secret: Bitfinex API secret (optional for public endpoints)
            cache_dir: Directory to cache data files
            fetch_limit: Max candles per request (Bitfinex limit: 5000)
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
        
        self.base_url = 'https://api.bitfinex.com'
        self.session = requests.Session()
        self.request_count = 0
        self.last_request_time = time.time()
    
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
    
    def _timeframe_to_interval(self, timeframe: str) -> str:
        """Convert timeframe to Bitfinex interval."""
        mapping = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '3h': '3h', '6h': '6h', '12h': '12h',
            '1d': '1D', '7d': '7D', '14d': '14D', '1M': '1M'
        }
        return mapping.get(timeframe.lower(), '1h')
    
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Bitfinex API (tBTCUSD format)."""
        symbol = symbol.upper().replace('USD', 'USD')
        if not symbol.startswith('t'):
            symbol = f"t{symbol}"
        return symbol
    
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
    
    def _rate_limit_check(self):
        """Check and enforce rate limits."""
        current_time = time.time()
        time_diff = current_time - self.last_request_time
        
        if time_diff < 60:
            if self.request_count >= 80:  # Buffer of 10 requests
                sleep_time = 60 - time_diff + 1
                logger.warning(f"Rate limit approaching. Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        else:
            self.request_count = 0
    
    def _fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start_timestamp_ms: int,
        end_timestamp_ms: Optional[int] = None,
        limit: int = 5000
    ) -> Optional[list]:
        """Fetch candle data from Bitfinex."""
        self._rate_limit_check()
        
        formatted_symbol = self._format_symbol(symbol)
        interval = self._timeframe_to_interval(timeframe)
        
        url = f"{self.base_url}/v2/candles/trade:{interval}:{formatted_symbol}/hist"
        
        params = {
            'start': start_timestamp_ms,
            'limit': min(limit, self.fetch_limit)
        }
        
        if end_timestamp_ms:
            params['end'] = end_timestamp_ms
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                self.request_count += 1
                time.sleep(self.rate_limit_delay)
                
                data = response.json()
                # Bitfinex returns data in reverse chronological order
                return list(reversed(data))
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error fetching candles: {e}")
                    if attempt == self.retry_attempts - 1:
                        return None
            except Exception as e:
                logger.error(f"Error fetching candles: {e}")
                if attempt == self.retry_attempts - 1:
                    return None
                time.sleep(5)
        
        return None
    
    def fetch(
        self,
        symbol: str = 'BTCUSD',
        timeframe: str = '1h',
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        limit: int = 2000
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Bitfinex.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 3h, 6h, 12h, 1D, 7D, 14D, 1M)
            start: Start date
            end: End date
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
            # Calculate start based on limit
            timeframe_hours = self._timeframe_to_hours(timeframe)
            start = end - datetime.timedelta(hours=limit * timeframe_hours)
        
        start_timestamp_ms = int(start.timestamp() * 1000)
        end_timestamp_ms = int(end.timestamp() * 1000)
        
        # Check if cache covers requested range
        if cached_data is not None and not cached_data.empty:
            cache_end = cached_data.index[-1]
            cache_end_ts = int(cache_end.timestamp() * 1000)
            
            if cache_end_ts >= end_timestamp_ms:
                filtered = cached_data[
                    (cached_data.index >= start) & (cached_data.index <= end)
                ]
                if not filtered.empty:
                    logger.info(f"Returning cached data for {symbol} ({timeframe})")
                    return filtered
        
        # Fetch new data
        logger.info(f"Fetching Bitfinex data for {symbol} ({timeframe}) from {start} to {end}")
        
        all_candles = []
        current_timestamp_ms = start_timestamp_ms
        
        while current_timestamp_ms < end_timestamp_ms and len(all_candles) < limit:
            chunk_data = self._fetch_candles(
                symbol=symbol,
                timeframe=timeframe,
                start_timestamp_ms=current_timestamp_ms,
                end_timestamp_ms=end_timestamp_ms,
                limit=self.fetch_limit
            )
            
            if not chunk_data:
                break
            
            all_candles.extend(chunk_data)
            
            if len(chunk_data) < self.fetch_limit:
                break
            
            # Update timestamp for next chunk
            last_candle_ts = chunk_data[-1][0]
            current_timestamp_ms = last_candle_ts + 1
            
            if len(all_candles) >= limit:
                break
        
        if not all_candles:
            logger.warning(f"No data fetched for {symbol}")
            return cached_data if cached_data is not None else pd.DataFrame()
        
        # Convert to DataFrame
        # Bitfinex format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
        df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'Close', 'High', 'Low', 'Volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        # Reorder columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
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
        
        # Merge with cached data
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
    
    def _timeframe_to_hours(self, timeframe: str) -> float:
        """Convert timeframe to hours."""
        amount = int("".join(filter(str.isdigit, timeframe)))
        unit = "".join(filter(str.isalpha, timeframe)).lower()
        
        multipliers = {'m': 1/60, 'h': 1, 'd': 24, 'w': 168}
        return amount * multipliers.get(unit, 1)
    
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

