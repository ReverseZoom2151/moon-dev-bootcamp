"""Unified data fetcher for backtesting frameworks"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict
from .provider import DataProvider

logger = logging.getLogger(__name__)


class MockDataProvider(DataProvider):
    """Mock data provider for testing and demonstrations"""
    
    @property
    def column_mapping(self) -> Dict[str, str]:
        """Return column mapping for frameworks"""
        return {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
    
    def fetch(self, symbol: str = 'BTCUSDT', timeframe: str = '1h',
              start: datetime = None, end: datetime = None,
              limit: int = 2000) -> pd.DataFrame:
        """
        Generate mock OHLCV data
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start: Start date (ignored for mock data)
            end: End date (ignored for mock data)
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Determine frequency
            if 'h' in timeframe.lower():
                freq = 'H'
                multiplier = int(timeframe.lower().replace('h', '') or '1')
                hours = limit * multiplier
            elif 'd' in timeframe.lower():
                freq = 'D'
                multiplier = int(timeframe.lower().replace('d', '') or '1')
                hours = limit * multiplier * 24
            else:
                freq = 'H'
                hours = limit * 24
            
            # Create date range
            date_range = pd.date_range(
                start=datetime.now() - timedelta(hours=hours),
                end=datetime.now(),
                freq=freq
            )[:limit]
            
            # Generate realistic price data with seed for reproducibility
            np.random.seed(42)
            
            # Start with initial price based on symbol
            if 'BTC' in symbol.upper():
                initial_price = 50000
                volatility = 500
            elif 'ETH' in symbol.upper():
                initial_price = 3000
                volatility = 50
            else:
                initial_price = 100
                volatility = 5
            
            # Generate price movement
            price_changes = np.random.randn(len(date_range)) * volatility
            prices = initial_price + np.cumsum(price_changes)
            
            # Create OHLCV data
            df = pd.DataFrame({
                'Open': prices + np.random.randn(len(date_range)) * (volatility * 0.2),
                'High': prices + abs(np.random.randn(len(date_range)) * (volatility * 0.4)),
                'Low': prices - abs(np.random.randn(len(date_range)) * (volatility * 0.4)),
                'Close': prices,
                'Volume': abs(np.random.randn(len(date_range)) * 1000000)
            }, index=date_range)
            
            # Ensure high >= close and low <= close
            df['High'] = df[['High', 'Close']].max(axis=1)
            df['Low'] = df[['Low', 'Close']].min(axis=1)
            
            logger.info(f"Generated {len(df)} mock candles for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock data: {e}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data integrity"""
        if df.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return False
        
        # Check for invalid values
        if (df['High'] < df['Close']).any() or (df['Low'] > df['Close']).any():
            logger.error("Invalid price relationships detected")
            return False
        
        if (df['Volume'] < 0).any():
            logger.error("Negative volume detected")
            return False
        
        return True


class DataFetcher:
    """Unified data fetcher for all backtesting frameworks"""
    
    def __init__(self, provider: DataProvider = None):
        """
        Initialize data fetcher
        
        Args:
            provider: Data provider instance (defaults to MockDataProvider)
        """
        self.provider = provider or MockDataProvider()
        logger.info(f"DataFetcher initialized with {self.provider.__class__.__name__}")
    
    def fetch(self, symbol: str = 'BTCUSDT', timeframe: str = '1h',
              start: datetime = None, end: datetime = None,
              limit: int = 2000) -> pd.DataFrame:
        """
        Fetch data from provider
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start: Start date
            end: End date
            limit: Candle limit
            
        Returns:
            OHLCV DataFrame
        """
        df = self.provider.fetch(symbol, timeframe, start, end, limit)
        
        if not self.provider.validate_data(df):
            logger.warning(f"Data validation failed for {symbol}")
            return pd.DataFrame()
        
        return df
    
    def fetch_for_backtrader(self, symbol: str = 'BTCUSDT', 
                            timeframe: str = '1d',
                            start: datetime = None,
                            end: datetime = None) -> pd.DataFrame:
        """
        Fetch data formatted for Backtrader (lowercase columns)
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            start: Start date
            end: End date
            
        Returns:
            DataFrame with lowercase OHLCV columns
        """
        df = self.fetch(symbol, timeframe, start, end)
        
        if df.empty:
            return df
        
        # Rename columns to lowercase for Backtrader
        return df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
    
    def fetch_for_backtesting_lib(self, symbol: str = 'ETHUSDT',
                                  timeframe: str = '1h',
                                  limit: int = 2000) -> pd.DataFrame:
        """
        Fetch data formatted for backtesting.py (capitalized columns)
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            limit: Number of candles
            
        Returns:
            DataFrame with capitalized OHLCV columns
        """
        return self.fetch(symbol, timeframe, limit=limit)
