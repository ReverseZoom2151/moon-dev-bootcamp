"""
Yahoo Finance Data Provider
============================
Day 32: Yahoo Finance data provider for stocks, forex, and futures.

Downloads and updates historical data from Yahoo Finance using yfinance:
- Hourly data (last 730 days)
- Daily data (extensive history)
- Automatic caching and updates
- Supports stocks, forex, and futures
"""

import pandas as pd
import datetime
from datetime import timedelta
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import logging
import time

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from .provider import DataProvider

logger = logging.getLogger(__name__)


class YahooFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider for stocks, forex, and futures.
    
    Supports hourly and daily data downloads with caching.
    """
    
    def __init__(
        self,
        cache_dir: str = "./yahoo_data",
        hourly_fetch_days: int = 728,
        default_start_date: str = "2000-01-01",
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize Yahoo Finance data provider.
        
        Args:
            cache_dir: Directory to cache data files
            hourly_fetch_days: Max days for hourly data (Yahoo limit: ~730)
            default_start_date: Default start date for daily data
            rate_limit_delay: Delay between requests (seconds)
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance library is required. Install with: pip install yfinance")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.hourly_fetch_days = hourly_fetch_days
        self.default_start_date = default_start_date
        self.rate_limit_delay = rate_limit_delay
    
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
    
    def _get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Get cache file path for symbol/timeframe."""
        symbol_safe = symbol.replace('=', '_').replace('/', '_')
        suffix = '1h' if 'h' in timeframe.lower() else '1d'
        cache_filename = f"{symbol_safe}_{suffix}.csv"
        return self.cache_dir / cache_filename
    
    def _load_from_cache(self, cache_path: Path, is_hourly: bool = False) -> Optional[pd.DataFrame]:
        """Load data from cache file."""
        if not cache_path.exists():
            return None
        
        try:
            index_col = 'Datetime' if is_hourly else 'Date'
            df = pd.read_csv(cache_path, index_col=index_col, parse_dates=True)
            
            # Handle timezone
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            logger.info(f"Loaded {len(df)} rows from cache: {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"Error loading cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path, is_hourly: bool = False):
        """Save data to cache file."""
        try:
            date_format = '%Y-%m-%d %H:%M:%S' if is_hourly else '%Y-%m-%d'
            df.to_csv(cache_path, date_format=date_format)
            logger.info(f"Saved {len(df)} rows to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache {cache_path}: {e}")
    
    def _download_yahoo_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """Download data from Yahoo Finance."""
        try:
            # Convert end_date to exclusive
            end_date_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            yf_end_date = end_date_dt + timedelta(days=1)
            
            logger.info(f"Downloading {symbol} ({interval}) from {start_date} to {end_date}...")
            
            data = yf.download(
                tickers=symbol,
                start=start_date,
                end=yf_end_date,
                interval=interval,
                progress=False,
                auto_adjust=False
            )
            
            if data is None or data.empty:
                logger.warning(f"No data downloaded for {symbol}")
                return None
            
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                data = data.loc[:, ~data.columns.duplicated()]
            
            # Handle timezone
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Clean data
            essential_cols = ['Open', 'High', 'Low', 'Close']
            missing = [col for col in essential_cols if col not in data.columns]
            if missing:
                logger.warning(f"Missing columns for {symbol}: {missing}")
                return None
            
            # Drop rows with NaN Close
            initial_rows = len(data)
            data.dropna(subset=['Close'], inplace=True)
            if len(data) < initial_rows:
                logger.debug(f"Removed {initial_rows - len(data)} rows with NaN Close")
            
            # Convert Volume to int
            if 'Volume' in data.columns:
                data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0).astype(int)
            
            # Ensure required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required:
                if col not in data.columns:
                    if col == 'Volume':
                        data[col] = 0
                    else:
                        logger.error(f"Required column {col} missing")
                        return None
            
            logger.info(f"Downloaded {len(data)} rows for {symbol}")
            return data[required]
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}", exc_info=True)
            return None
    
    def fetch(
        self,
        symbol: str = 'AAPL',
        timeframe: str = '1d',
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
        limit: int = 2000
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            symbol: Symbol (e.g., 'AAPL', 'EURUSD=X', 'ES=F')
            timeframe: Timeframe ('1h' for hourly, '1d' for daily)
            start: Start date
            end: End date
            limit: Number of candles (used if start/end not provided)
            
        Returns:
            DataFrame with OHLCV data
        """
        is_hourly = 'h' in timeframe.lower()
        cache_path = self._get_cache_path(symbol, timeframe)
        
        # Load cached data
        cached_data = self._load_from_cache(cache_path, is_hourly=is_hourly)
        
        # Determine date range
        if end is None:
            end = datetime.datetime.now()
        
        if start is None:
            if is_hourly:
                # Hourly: limited to ~730 days
                start = end - timedelta(days=self.hourly_fetch_days)
            else:
                # Daily: use default start or calculate from limit
                if limit:
                    start = end - timedelta(days=limit)
                else:
                    start = datetime.datetime.strptime(self.default_start_date, '%Y-%m-%d')
        else:
            # If hourly and start is too far back, adjust
            if is_hourly:
                max_start = end - timedelta(days=self.hourly_fetch_days)
                if start < max_start:
                    logger.warning(f"Hourly data limited to {self.hourly_fetch_days} days. Adjusting start date.")
                    start = max_start
        
        # Check if cache covers requested range
        if cached_data is not None and not cached_data.empty:
            cache_end = cached_data.index[-1]
            
            if cache_end >= end:
                # Cache covers range, return filtered
                filtered = cached_data[
                    (cached_data.index >= start) & (cached_data.index <= end)
                ]
                if not filtered.empty:
                    logger.info(f"Returning cached data for {symbol} ({timeframe})")
                    return filtered
                
                # Need to update cache
                start_date_str = (cache_end + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                # Need to fill gap
                start_date_str = (cache_end + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # No cache, fetch from start
            start_date_str = start.strftime('%Y-%m-%d')
        
        end_date_str = end.strftime('%Y-%m-%d')
        
        # Download new data
        interval = '1h' if is_hourly else '1d'
        time.sleep(self.rate_limit_delay)
        
        new_data = self._download_yahoo_data(
            symbol=symbol,
            start_date=start_date_str,
            end_date=end_date_str,
            interval=interval
        )
        
        if new_data is None or new_data.empty:
            # Return cached data if available
            if cached_data is not None:
                filtered = cached_data[
                    (cached_data.index >= start) & (cached_data.index <= end)
                ]
                return filtered
            return pd.DataFrame()
        
        # Merge with cached data
        if cached_data is not None and not cached_data.empty:
            combined = pd.concat([cached_data, new_data])
            df = combined[~combined.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
        else:
            df = new_data
        
        # Filter to requested range
        df = df[(df.index >= start) & (df.index <= end)]
        
        # Save to cache
        if not df.empty:
            self._save_to_cache(df, cache_path, is_hourly=is_hourly)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data integrity."""
        if df.empty:
            return False
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Check for negative values in price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        if (df[price_cols] < 0).any().any():
            return False
        
        # Check OHLC relationships
        if not ((df['High'] >= df['Low']).all() and
                (df['High'] >= df['Open']).all() and
                (df['High'] >= df['Close']).all() and
                (df['Low'] <= df['Open']).all() and
                (df['Low'] <= df['Close']).all()):
            return False
        
        return True
    
    def update_all_symbols(self, symbols: List[Tuple[str, str]]) -> Dict[str, int]:
        """
        Update data for multiple symbols.
        
        Args:
            symbols: List of (symbol, asset_type) tuples
            
        Returns:
            Dictionary mapping symbols to row counts
        """
        results = {}
        
        for symbol, asset_type in symbols:
            try:
                # Determine timeframe based on asset type or use daily
                timeframe = '1d'  # Default to daily
                
                df = self.fetch(symbol=symbol, timeframe=timeframe)
                results[symbol] = len(df)
                
                logger.info(f"Updated {symbol}: {len(df)} rows")
                
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                results[symbol] = -1
        
        return results

