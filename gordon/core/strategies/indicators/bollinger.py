"""
Bollinger Bands Indicator
==========================
Calculates Bollinger Bands for volatility analysis.
Exchange-agnostic implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


class BollingerBands:
    """Bollinger Bands calculator."""

    @staticmethod
    def calculate(data: Union[pd.Series, np.ndarray, list],
                  period: int = 20,
                  std_dev: float = 2.0) -> Tuple:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price data (typically close prices)
            period: Moving average period (default 20)
            std_dev: Number of standard deviations (default 2)

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if isinstance(data, pd.Series):
            middle_band = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)

        else:
            data_array = np.array(data)
            middle_band = np.zeros(len(data_array))
            upper_band = np.zeros(len(data_array))
            lower_band = np.zeros(len(data_array))

            for i in range(period - 1, len(data_array)):
                window = data_array[i - period + 1:i + 1]
                mean = np.mean(window)
                std = np.std(window)

                middle_band[i] = mean
                upper_band[i] = mean + (std * std_dev)
                lower_band[i] = mean - (std * std_dev)

            # Set initial values to NaN
            middle_band[:period - 1] = np.nan
            upper_band[:period - 1] = np.nan
            lower_band[:period - 1] = np.nan

        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_df(df: pd.DataFrame, column: str = 'close',
                     period: int = 20, std_dev: float = 2.0,
                     prefix: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for DataFrame and add as new columns.

        Args:
            df: DataFrame with price data
            column: Column to calculate bands on
            period: Moving average period
            std_dev: Number of standard deviations
            prefix: Prefix for column names

        Returns:
            DataFrame with Bollinger Bands columns added
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        prefix = prefix or f'bb_{period}'
        upper, middle, lower = BollingerBands.calculate(df[column], period, std_dev)

        df[f'{prefix}_upper'] = upper
        df[f'{prefix}_middle'] = middle
        df[f'{prefix}_lower'] = lower

        return df

    @staticmethod
    def bandwidth(upper_band: Union[pd.Series, np.ndarray],
                  lower_band: Union[pd.Series, np.ndarray],
                  middle_band: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Bollinger Bandwidth (volatility measure).

        Args:
            upper_band: Upper Bollinger Band
            lower_band: Lower Bollinger Band
            middle_band: Middle Band (SMA)

        Returns:
            Bandwidth values
        """
        if isinstance(upper_band, pd.Series):
            return ((upper_band - lower_band) / middle_band) * 100
        else:
            upper = np.array(upper_band)
            lower = np.array(lower_band)
            middle = np.array(middle_band)
            return ((upper - lower) / middle) * 100

    @staticmethod
    def percent_b(price: Union[pd.Series, np.ndarray],
                  upper_band: Union[pd.Series, np.ndarray],
                  lower_band: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Calculate %B indicator (position within bands).

        Args:
            price: Current price
            upper_band: Upper Bollinger Band
            lower_band: Lower Bollinger Band

        Returns:
            %B values (0 = lower band, 1 = upper band)
        """
        if isinstance(price, pd.Series):
            return (price - lower_band) / (upper_band - lower_band)
        else:
            price_arr = np.array(price)
            upper = np.array(upper_band)
            lower = np.array(lower_band)
            return (price_arr - lower) / (upper - lower)

    @staticmethod
    def squeeze(bandwidth: Union[pd.Series, np.ndarray],
                threshold: float = 2.0) -> Union[pd.Series, np.ndarray]:
        """
        Detect Bollinger Band squeeze (low volatility).

        Args:
            bandwidth: Bollinger Bandwidth values
            threshold: Squeeze threshold

        Returns:
            Boolean array where True indicates squeeze
        """
        if isinstance(bandwidth, pd.Series):
            return bandwidth < threshold
        else:
            return np.array(bandwidth) < threshold

    @staticmethod
    def generate_signals(df: pd.DataFrame, close_col: str = 'close',
                        period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.

        Args:
            df: DataFrame with price data
            close_col: Column with close prices
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            DataFrame with signals added
        """
        # Calculate Bollinger Bands
        upper, middle, lower = BollingerBands.calculate(df[close_col], period, std_dev)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower

        # Calculate %B
        df['percent_b'] = BollingerBands.percent_b(df[close_col], upper, lower)

        # Generate signals
        df['signal'] = 0

        # Buy when price touches lower band
        df.loc[df[close_col] <= df['bb_lower'], 'signal'] = 1

        # Sell when price touches upper band
        df.loc[df[close_col] >= df['bb_upper'], 'signal'] = -1

        return df