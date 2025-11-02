"""Abstract data provider interface"""

from abc import ABC, abstractmethod
from typing import Dict
from datetime import datetime
import pandas as pd


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, 
              start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch OHLCV data
        
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
