"""Abstract data provider interface"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime
import pandas as pd


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, 
              start: Optional[datetime] = None, 
              end: Optional[datetime] = None,
              limit: int = 2000) -> pd.DataFrame:
        """
        Fetch OHLCV data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h', '1d')
            start: Start date (optional)
            end: End date (optional)
            limit: Number of candles to fetch (optional)
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data integrity"""
        pass
    
    @property
    @abstractmethod
    def column_mapping(self) -> Dict[str, str]:
        """Get column name mapping for framework compatibility"""
        pass
