"""
Data Utilities Module
=====================
Contains functions for data manipulation, cleaning, and processing.
Handles OHLCV data preparation, resampling, and transformations.
"""

import pandas as pd
import numpy as np
from typing import List


class DataUtils:
    """Utilities for data manipulation and processing."""

    @staticmethod
    def clean_ohlcv_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare OHLCV data.

        Args:
            data: Raw OHLCV DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Ensure column names are correct
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        data.columns = [col.lower() for col in data.columns]

        # Remove NaN values
        data = data.dropna()

        # Convert to numeric
        for col in required_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Remove zero volume rows
        data = data[data['volume'] > 0]

        # Sort by index (timestamp)
        data = data.sort_index()

        return data

    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe.

        Args:
            data: OHLCV DataFrame with datetime index
            timeframe: Target timeframe (e.g., '1h', '4h', '1d')

        Returns:
            Resampled DataFrame
        """
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        return data.resample(timeframe).agg(ohlc_dict).dropna()

    @staticmethod
    def calculate_returns(data: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate returns over multiple periods.

        Args:
            data: OHLCV DataFrame
            periods: List of periods to calculate returns for

        Returns:
            DataFrame with returns columns added
        """
        for period in periods:
            data[f'return_{period}'] = data['close'].pct_change(period) * 100

        return data

    @staticmethod
    def handle_numpy_nan(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle numpy NaN compatibility issues.

        This addresses the numpy 2.0 compatibility issue found in Day_20.
        """
        # Replace np.NaN with pd.NA or None
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df


# Create singleton instance
data_utils = DataUtils()
