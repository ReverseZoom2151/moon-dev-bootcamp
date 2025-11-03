"""
Liquidation Data Preparation Utilities
======================================
Utilities for preparing liquidation data for backtesting.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional


def prepare_liquidation_data_for_backtest(
    data_path: str,
    symbol: Optional[str] = None,
    resample_frequency: str = '1T'
) -> pd.DataFrame:
    """
    Prepare liquidation data for backtesting.
    
    Converts raw liquidation CSV data into format suitable for backtesting.py library.
    
    Args:
        data_path: Path to liquidation CSV file
        symbol: Optional symbol filter (if CSV contains multiple symbols)
        resample_frequency: Resampling frequency (default: '1T' for 1 minute)
        
    Returns:
        DataFrame ready for backtesting with OHLC and volume columns
    """
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Filter by symbol if provided
    if symbol and 'symbol' in data.columns:
        data = data[data['symbol'] == symbol].copy()
    
    # Ensure required columns exist
    required_cols = ['LIQ_SIDE', 'price', 'usd_size']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data missing required columns. Found: {data.columns.tolist()}")
    
    # Create L LIQ and S LIQ volume columns
    data['l_liq_volume'] = np.where(data['LIQ_SIDE'] == 'L LIQ', data['usd_size'], 0)
    data['s_liq_volume'] = np.where(data['LIQ_SIDE'] == 'S LIQ', data['usd_size'], 0)
    
    # Resample data to specified frequency
    agg_funcs = {
        'price': 'mean',         # Mean price of liquidations in the period
        'l_liq_volume': 'sum',   # Sum of L LIQ volume in the period
        's_liq_volume': 'sum'    # Sum of S LIQ volume in the period
    }
    data_resampled = data.resample(resample_frequency).agg(agg_funcs)
    
    # Create OHLC columns required by backtesting.py
    # Simplified approach: Use the mean price for Open, High, Low, Close
    # This is a limitation as we don't have true OHLC data from liquidations
    data_resampled['Open'] = data_resampled['price']
    data_resampled['High'] = data_resampled['price']
    data_resampled['Low'] = data_resampled['price']
    data_resampled['Close'] = data_resampled['price']
    
    # Handle missing data after resampling
    price_columns = ['price', 'Open', 'High', 'Low', 'Close']
    data_resampled[price_columns] = data_resampled[price_columns].ffill()
    volume_columns = ['l_liq_volume', 's_liq_volume']
    data_resampled[volume_columns] = data_resampled[volume_columns].fillna(0)
    
    # Ensure sorted by index
    data_resampled.sort_index(inplace=True)
    
    # Remove rows with NaN in critical columns
    data_resampled.dropna(subset=['Close'], inplace=True)
    
    if data_resampled.empty or data_resampled['Close'].isnull().all():
        raise ValueError("No valid data available for backtesting after processing.")
    
    return data_resampled


def prepare_liquidation_data_for_sliq_strategy(
    data_path: str,
    symbol: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare liquidation data specifically for S LIQ strategy.
    
    Args:
        data_path: Path to liquidation CSV file
        symbol: Optional symbol filter
        
    Returns:
        DataFrame ready for LiquidationSLiqStrategy
    """
    return prepare_liquidation_data_for_backtest(
        data_path,
        symbol=symbol,
        resample_frequency='1T'
    )


def prepare_liquidation_data_for_lliq_strategy(
    data_path: str,
    symbol: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare liquidation data specifically for L LIQ strategy.
    
    Args:
        data_path: Path to liquidation CSV file
        symbol: Optional symbol filter
        
    Returns:
        DataFrame ready for LiquidationLLiqStrategy
    """
    return prepare_liquidation_data_for_backtest(
        data_path,
        symbol=symbol,
        resample_frequency='1T'
    )


def prepare_liquidation_data_for_short_strategy(
    data_path: str,
    symbol: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare liquidation data specifically for short liquidation strategies.
    
    Args:
        data_path: Path to liquidation CSV file
        symbol: Optional symbol filter
        
    Returns:
        DataFrame ready for LiquidationShortSLiqStrategy
    """
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Filter by symbol if provided
    if symbol and 'symbol' in data.columns:
        data = data[data['symbol'] == symbol].copy()
    
    # Ensure required columns exist
    required_cols = ['LIQ_SIDE', 'price', 'usd_size']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data missing required columns. Found: {data.columns.tolist()}")
    
    # Create L LIQ and S LIQ volume columns
    data['long_liquidations'] = np.where(data['LIQ_SIDE'] == 'L LIQ', data['usd_size'], 0)
    data['short_liquidations'] = np.where(data['LIQ_SIDE'] == 'S LIQ', data['usd_size'], 0)
    
    # Resample data to 1-minute frequency
    agg_funcs = {
        'price': 'mean',
        'long_liquidations': 'sum',
        'short_liquidations': 'sum'
    }
    data_resampled = data.resample('1T').agg(agg_funcs)
    
    # Create OHLC columns required by backtesting.py
    data_resampled['Open'] = data_resampled['price']
    data_resampled['High'] = data_resampled['price']
    data_resampled['Low'] = data_resampled['price']
    data_resampled['Close'] = data_resampled['price']
    
    # Handle missing data after resampling
    price_columns = ['price', 'Open', 'High', 'Low', 'Close']
    data_resampled[price_columns] = data_resampled[price_columns].ffill()
    volume_columns = ['long_liquidations', 'short_liquidations']
    data_resampled[volume_columns] = data_resampled[volume_columns].fillna(0)
    
    # Ensure sorted by index
    data_resampled.sort_index(inplace=True)
    
    # Remove rows with NaN in critical columns
    data_resampled.dropna(subset=['Close'], inplace=True)
    
    if data_resampled.empty or data_resampled['Close'].isnull().all():
        raise ValueError("No valid data available for backtesting after processing.")
    
    return data_resampled



