"""
Volume Weighted Average Price (VWAP) Indicator
===============================================
Calculates VWAP for intraday trading.
Exchange-agnostic implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


class VWAP:
    """Volume Weighted Average Price calculator."""

    @staticmethod
    def calculate(high: Union[pd.Series, np.ndarray],
                  low: Union[pd.Series, np.ndarray],
                  close: Union[pd.Series, np.ndarray],
                  volume: Union[pd.Series, np.ndarray],
                  reset_daily: bool = True) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Volume Weighted Average Price.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            reset_daily: Reset VWAP daily (for intraday)

        Returns:
            VWAP values
        """
        # Calculate typical price
        if isinstance(high, pd.Series):
            typical_price = (high + low + close) / 3
            cumulative_tpv = (typical_price * volume).cumsum()
            cumulative_volume = volume.cumsum()

            # Reset daily if index is datetime
            if reset_daily and isinstance(high.index, pd.DatetimeIndex):
                # Group by date and calculate VWAP for each day
                df_temp = pd.DataFrame({
                    'tpv': typical_price * volume,
                    'volume': volume
                })
                df_temp['date'] = df_temp.index.date

                # Calculate cumulative sums within each day
                df_temp['cum_tpv'] = df_temp.groupby('date')['tpv'].cumsum()
                df_temp['cum_vol'] = df_temp.groupby('date')['volume'].cumsum()

                vwap = df_temp['cum_tpv'] / df_temp['cum_vol']
            else:
                vwap = cumulative_tpv / cumulative_volume

            return vwap

        else:
            # Numpy implementation
            high_arr = np.array(high)
            low_arr = np.array(low)
            close_arr = np.array(close)
            volume_arr = np.array(volume)

            typical_price = (high_arr + low_arr + close_arr) / 3
            vwap = np.zeros(len(typical_price))

            cumulative_tpv = 0
            cumulative_volume = 0

            for i in range(len(typical_price)):
                cumulative_tpv += typical_price[i] * volume_arr[i]
                cumulative_volume += volume_arr[i]

                if cumulative_volume > 0:
                    vwap[i] = cumulative_tpv / cumulative_volume
                else:
                    vwap[i] = typical_price[i]

            return vwap

    @staticmethod
    def calculate_df(df: pd.DataFrame, high_col: str = 'high',
                     low_col: str = 'low', close_col: str = 'close',
                     volume_col: str = 'volume', name: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate VWAP for DataFrame and add as new column.

        Args:
            df: DataFrame with OHLCV data
            high_col: Column with high prices
            low_col: Column with low prices
            close_col: Column with close prices
            volume_col: Column with volume data
            name: Name for the new VWAP column

        Returns:
            DataFrame with VWAP column added
        """
        required_cols = [high_col, low_col, close_col, volume_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        vwap_name = name or 'vwap'
        df[vwap_name] = VWAP.calculate(
            df[high_col], df[low_col], df[close_col], df[volume_col]
        )
        return df

    @staticmethod
    def calculate_bands(vwap: Union[pd.Series, np.ndarray],
                       high: Union[pd.Series, np.ndarray],
                       low: Union[pd.Series, np.ndarray],
                       std_multiplier: float = 1.0) -> Tuple:
        """
        Calculate VWAP bands (upper and lower).

        Args:
            vwap: VWAP values
            high: High prices
            low: Low prices
            std_multiplier: Standard deviation multiplier for bands

        Returns:
            Tuple of (upper_band, lower_band)
        """
        if isinstance(vwap, pd.Series):
            # Calculate standard deviation
            typical_price = (high + low) / 2
            std = (typical_price - vwap).rolling(window=20).std()

            upper_band = vwap + (std * std_multiplier)
            lower_band = vwap - (std * std_multiplier)

        else:
            vwap_arr = np.array(vwap)
            high_arr = np.array(high)
            low_arr = np.array(low)

            typical_price = (high_arr + low_arr) / 2
            upper_band = np.zeros(len(vwap_arr))
            lower_band = np.zeros(len(vwap_arr))

            # Calculate rolling standard deviation
            window = 20
            for i in range(window, len(vwap_arr)):
                window_data = typical_price[i - window:i] - vwap_arr[i - window:i]
                std = np.std(window_data)
                upper_band[i] = vwap_arr[i] + (std * std_multiplier)
                lower_band[i] = vwap_arr[i] - (std * std_multiplier)

            # Fill initial values
            upper_band[:window] = vwap_arr[:window]
            lower_band[:window] = vwap_arr[:window]

        return upper_band, lower_band

    @staticmethod
    def get_deviation(price: Union[pd.Series, np.ndarray],
                     vwap: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Calculate price deviation from VWAP as percentage.

        Args:
            price: Current price
            vwap: VWAP values

        Returns:
            Deviation percentage
        """
        if isinstance(price, pd.Series) and isinstance(vwap, pd.Series):
            return ((price - vwap) / vwap) * 100
        else:
            price_arr = np.array(price)
            vwap_arr = np.array(vwap)
            return ((price_arr - vwap_arr) / vwap_arr) * 100

    @staticmethod
    def generate_signals(df: pd.DataFrame, close_col: str = 'close',
                        threshold: float = 1.0) -> pd.DataFrame:
        """
        Generate trading signals based on VWAP crossovers and deviations.

        Args:
            df: DataFrame with OHLCV data
            close_col: Column with close prices
            threshold: Deviation threshold for signals (%)

        Returns:
            DataFrame with signals added
        """
        # Calculate VWAP if not present
        if 'vwap' not in df.columns:
            df = VWAP.calculate_df(df)

        # Calculate deviation
        df['vwap_deviation'] = VWAP.get_deviation(df[close_col], df['vwap'])

        # Generate signals
        df['signal'] = 0

        # Buy when price is below VWAP by threshold
        df.loc[df['vwap_deviation'] < -threshold, 'signal'] = 1

        # Sell when price is above VWAP by threshold
        df.loc[df['vwap_deviation'] > threshold, 'signal'] = -1

        return df