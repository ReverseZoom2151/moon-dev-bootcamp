"""
Relative Strength Index (RSI) Indicator
========================================
Calculates RSI momentum oscillator.
Exchange-agnostic implementation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


class RSI:
    """Relative Strength Index calculator."""

    @staticmethod
    def calculate(data: Union[pd.Series, np.ndarray, list],
                  period: int = 14) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Relative Strength Index.

        Args:
            data: Price data (Series, array, or list)
            period: RSI period (default 14)

        Returns:
            RSI values (0-100 scale)
        """
        if isinstance(data, pd.Series):
            # Calculate price changes
            delta = data.diff()

            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gain/loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        else:
            # Convert to numpy array
            data_array = np.array(data)
            rsi_array = np.full(len(data_array), np.nan)

            if len(data_array) < period + 1:
                return rsi_array

            # Calculate price changes
            deltas = np.diff(data_array)

            # Initialize first average gain/loss
            seed = deltas[:period]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period

            # Calculate first RSI value
            if down != 0:
                rs = up / down
                rsi_array[period] = 100 - (100 / (1 + rs))
            else:
                rsi_array[period] = 100

            # Calculate remaining RSI values using smoothing
            for i in range(period + 1, len(data_array)):
                delta = deltas[i - 1]

                if delta > 0:
                    up_val = delta
                    down_val = 0
                else:
                    up_val = 0
                    down_val = -delta

                up = (up * (period - 1) + up_val) / period
                down = (down * (period - 1) + down_val) / period

                if down != 0:
                    rs = up / down
                    rsi_array[i] = 100 - (100 / (1 + rs))
                else:
                    rsi_array[i] = 100

            return rsi_array

    @staticmethod
    def calculate_df(df: pd.DataFrame, column: str = 'close',
                     period: int = 14, name: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate RSI for DataFrame and add as new column.

        Args:
            df: DataFrame with price data
            column: Column to calculate RSI on
            period: RSI period
            name: Name for the new RSI column

        Returns:
            DataFrame with RSI column added
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        rsi_name = name or f'rsi_{period}'
        df[rsi_name] = RSI.calculate(df[column], period)
        return df

    @staticmethod
    def get_signals(rsi_values: Union[pd.Series, np.ndarray],
                    oversold: float = 30,
                    overbought: float = 70) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Get buy and sell signals based on RSI levels.

        Args:
            rsi_values: RSI values
            oversold: Oversold threshold (default 30)
            overbought: Overbought threshold (default 70)

        Returns:
            Tuple of (buy_signals, sell_signals) as boolean arrays
        """
        if isinstance(rsi_values, pd.Series):
            buy_signals = rsi_values < oversold
            sell_signals = rsi_values > overbought
        else:
            rsi_array = np.array(rsi_values)
            buy_signals = rsi_array < oversold
            sell_signals = rsi_array > overbought

        return buy_signals, sell_signals

    @staticmethod
    def divergence(price: Union[pd.Series, np.ndarray],
                   rsi: Union[pd.Series, np.ndarray],
                   lookback: int = 14) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Detect bullish and bearish divergences.

        Args:
            price: Price data
            rsi: RSI values
            lookback: Number of periods to look back for divergence

        Returns:
            Tuple of (bullish_div, bearish_div) as boolean arrays
        """
        if isinstance(price, pd.Series) and isinstance(rsi, pd.Series):
            # Find price lows and highs
            price_lows = price.rolling(window=lookback).min() == price
            price_highs = price.rolling(window=lookback).max() == price

            # Find RSI lows and highs
            rsi_lows = rsi.rolling(window=lookback).min() == rsi
            rsi_highs = rsi.rolling(window=lookback).max() == rsi

            # Detect divergences
            bullish_div = price_lows & ~rsi_lows  # Price makes lower low, RSI doesn't
            bearish_div = price_highs & ~rsi_highs  # Price makes higher high, RSI doesn't

        else:
            # Numpy implementation
            price_arr = np.array(price)
            rsi_arr = np.array(rsi)
            bullish_div = np.zeros(len(price_arr), dtype=bool)
            bearish_div = np.zeros(len(price_arr), dtype=bool)

            for i in range(lookback, len(price_arr)):
                window_price = price_arr[i - lookback:i + 1]
                window_rsi = rsi_arr[i - lookback:i + 1]

                # Check for divergences at current position
                if price_arr[i] == np.min(window_price):
                    if rsi_arr[i] != np.min(window_rsi):
                        bullish_div[i] = True

                if price_arr[i] == np.max(window_price):
                    if rsi_arr[i] != np.max(window_rsi):
                        bearish_div[i] = True

        return bullish_div, bearish_div

    @staticmethod
    def generate_signals(df: pd.DataFrame, price_col: str = 'close',
                        period: int = 14, oversold: float = 30,
                        overbought: float = 70) -> pd.DataFrame:
        """
        Generate trading signals based on RSI levels.

        Args:
            df: DataFrame with price data
            price_col: Column with price data
            period: RSI period
            oversold: Oversold threshold
            overbought: Overbought threshold

        Returns:
            DataFrame with signals added
        """
        # Calculate RSI
        df[f'rsi_{period}'] = RSI.calculate(df[price_col], period)

        # Generate signals
        df['signal'] = 0
        df.loc[df[f'rsi_{period}'] < oversold, 'signal'] = 1  # Buy
        df.loc[df[f'rsi_{period}'] > overbought, 'signal'] = -1  # Sell

        return df