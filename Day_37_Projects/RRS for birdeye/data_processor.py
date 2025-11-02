# data_processor.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_returns_and_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates log returns and rolling volatility.

    Args:
        df: DataFrame with a 'close' column.
        window: The rolling window period for volatility calculation.

    Returns:
        DataFrame with added 'log_return' and 'volatility' columns.
    """
    if 'close' not in df.columns:
        logger.error("'close' column not found in DataFrame for return/volatility calculation.")
        # Return original df or raise error depending on desired behavior
        return df # Or raise ValueError("'close' column missing")

    # Ensure close is numeric and handle potential infinities from zero prices
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan) # Replace infinities if log(0) occurs

    # Calculate log returns, handling potential NaNs from coerce or initial shift
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return'] = df['log_return'].replace([np.inf, -np.inf], np.nan)

    # Calculate volatility, ensuring enough non-NaN values for the window
    if len(df.dropna(subset=['log_return'])) >= window:
        df['volatility'] = df['log_return'].rolling(window=window, min_periods=window).std()
    else:
        logger.warning(f"Not enough data points ({len(df.dropna(subset=['log_return']))}) for volatility window {window}. Setting volatility to NaN.")
        df['volatility'] = np.nan

    logger.debug("Calculated log returns and volatility.")
    return df

def calculate_volume_metrics(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates rolling average volume and volume ratio.

    Args:
        df: DataFrame with a 'volume' column.
        window: The rolling window period for average volume calculation.

    Returns:
        DataFrame with added 'average_volume' and 'volume_ratio' columns.
    """
    if 'volume' not in df.columns:
        logger.error("'volume' column not found in DataFrame for volume metrics calculation.")
        return df # Or raise ValueError("'volume' column missing")

    # Ensure volume is numeric
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # Calculate average volume
    if len(df.dropna(subset=['volume'])) >= window:
        df['average_volume'] = df['volume'].rolling(window=window, min_periods=window).mean()
    else:
        logger.warning(f"Not enough data points ({len(df.dropna(subset=['volume']))}) for average volume window {window}. Setting average_volume to NaN.")
        df['average_volume'] = np.nan

    # Calculate volume ratio, handle division by zero or NaN average volume
    df['volume_ratio'] = df['volume'] / df['average_volume']
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan) # Handle division by zero/very small numbers

    logger.debug("Calculated volume metrics.")
    return df
