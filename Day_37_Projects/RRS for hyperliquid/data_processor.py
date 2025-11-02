# data_processor.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def interval_to_minutes(interval: str) -> int:
    """Convert interval string (e.g., '1m', '5m', '1h', '1d') to minutes."""
    logger.debug(f"Converting interval '{interval}' to minutes.")
    try:
        if isinstance(interval, int):
             return interval # Assume already in minutes if int
        
        multiplier = int(interval[:-1])
        unit = interval[-1].lower()

        if unit == 'm':
            minutes = multiplier
        elif unit == 'h':
            minutes = multiplier * 60
        elif unit == 'd':
            minutes = multiplier * 1440 # 24 * 60
        else:
            raise ValueError(f"Unknown interval unit: {unit}")
        
        logger.debug(f"Converted {interval} to {minutes} minutes.")
        return minutes
    except (ValueError, TypeError, IndexError) as e:
        logger.error(f"Failed to parse interval string '{interval}': {e}. Defaulting to 15 minutes.")
        return 15 # Default to 15 min on parsing error

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
        return df

    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return'] = df['log_return'].replace([np.inf, -np.inf], np.nan)

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
        return df

    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    if len(df.dropna(subset=['volume'])) >= window:
        df['average_volume'] = df['volume'].rolling(window=window, min_periods=window).mean()
    else:
        logger.warning(f"Not enough data points ({len(df.dropna(subset=['volume']))}) for average volume window {window}. Setting average_volume to NaN.")
        df['average_volume'] = np.nan

    df['volume_ratio'] = df['volume'] / df['average_volume']
    df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan)

    logger.debug("Calculated volume metrics.")
    return df
