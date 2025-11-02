"""
Simple Moving Average (SMA) Indicator
======================================
Calculates simple moving average for price data.
Exchange-agnostic implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


class SMA:
    """Simple Moving Average calculator."""

    @staticmethod
    def calculate(data: Union[pd.Series, np.ndarray, list],
                  period: int) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Simple Moving Average.

        Args:
            data: Price data (Series, array, or list)
            period: Number of periods for moving average

        Returns:
            SMA values with same type as input
        """
        if isinstance(data, pd.Series):
            return data.rolling(window=period).mean()
        elif isinstance(data, (np.ndarray, list)):
            data_array = np.array(data) if isinstance(data, list) else data
            sma = np.convolve(data_array, np.ones(period) / period, mode='same')
            # Set initial values to NaN where not enough data
            sma[:period - 1] = np.nan
            return sma
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    @staticmethod
    def calculate_df(df: pd.DataFrame, column: str = 'close',
                     period: int = 20, name: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate SMA for DataFrame and add as new column.

        Args:
            df: DataFrame with price data
            column: Column to calculate SMA on
            period: SMA period
            name: Name for the new SMA column

        Returns:
            DataFrame with SMA column added
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        sma_name = name or f'sma_{period}'
        df[sma_name] = SMA.calculate(df[column], period)
        return df

    @staticmethod
    def crossover(price: Union[pd.Series, np.ndarray],
                  sma: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Detect when price crosses over SMA.

        Args:
            price: Price data
            sma: SMA values

        Returns:
            Boolean series/array where True indicates crossover
        """
        if isinstance(price, pd.Series) and isinstance(sma, pd.Series):
            return (price > sma) & (price.shift(1) <= sma.shift(1))
        else:
            price_arr = np.array(price)
            sma_arr = np.array(sma)
            crossover = np.zeros(len(price_arr), dtype=bool)
            for i in range(1, len(price_arr)):
                if price_arr[i] > sma_arr[i] and price_arr[i-1] <= sma_arr[i-1]:
                    crossover[i] = True
            return crossover

    @staticmethod
    def crossunder(price: Union[pd.Series, np.ndarray],
                   sma: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Detect when price crosses under SMA.

        Args:
            price: Price data
            sma: SMA values

        Returns:
            Boolean series/array where True indicates crossunder
        """
        if isinstance(price, pd.Series) and isinstance(sma, pd.Series):
            return (price < sma) & (price.shift(1) >= sma.shift(1))
        else:
            price_arr = np.array(price)
            sma_arr = np.array(sma)
            crossunder = np.zeros(len(price_arr), dtype=bool)
            for i in range(1, len(price_arr)):
                if price_arr[i] < sma_arr[i] and price_arr[i-1] >= sma_arr[i-1]:
                    crossunder[i] = True
            return crossunder

    @staticmethod
    def generate_signals(df: pd.DataFrame, price_col: str = 'close',
                        period: int = 20) -> pd.DataFrame:
        """
        Generate trading signals based on SMA crossovers.

        Args:
            df: DataFrame with price data
            price_col: Column with price data
            period: SMA period

        Returns:
            DataFrame with signals added
        """
        # Calculate SMA
        df[f'sma_{period}'] = SMA.calculate(df[price_col], period)

        # Generate signals
        df['signal'] = 0
        df.loc[SMA.crossover(df[price_col], df[f'sma_{period}']), 'signal'] = 1  # Buy
        df.loc[SMA.crossunder(df[price_col], df[f'sma_{period}']), 'signal'] = -1  # Sell

        return df