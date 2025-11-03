"""
Data Filtering Utilities for Backtesting
==========================================
Day 23: Utilities for filtering historical data by time periods, market hours, and seasons.

Features:
- Exclude summer months (May-September)
- Filter market vs non-market hours
- Filter first/last hour of trading
- Timezone-aware filtering
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from datetime import datetime, time

logger = logging.getLogger(__name__)


class DataFilter:
    """
    Data filtering utilities for backtesting.
    
    Provides methods to filter historical data by various time-based criteria.
    """

    @staticmethod
    def filter_exclude_months(
        df: pd.DataFrame,
        start_month: int = 5,
        end_month: int = 9
    ) -> pd.DataFrame:
        """
        Filter DataFrame to exclude a range of months.
        
        Args:
            df: DataFrame with datetime index
            start_month: Start month to exclude (1-12)
            end_month: End month to exclude (1-12)
            
        Returns:
            Filtered DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        logger.info(f"Filtering data to exclude months {start_month} through {end_month}")
        
        # Create boolean mask for rows *within* the exclude range
        exclude_mask = (df.index.month >= start_month) & (df.index.month <= end_month)
        
        # Keep rows where the mask is False (i.e., outside the exclude range)
        filtered_df = df[~exclude_mask].copy()
        
        logger.info(f"Filtering complete. Original shape: {df.shape}, Filtered shape: {filtered_df.shape}")
        return filtered_df

    @staticmethod
    def filter_market_hours(
        df: pd.DataFrame,
        open_time_str: str = '13:30',
        close_time_str: str = '20:00',
        timezone: str = 'UTC'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter DataFrame into market hours and non-market hours (weekdays only).
        
        Args:
            df: DataFrame with datetime index
            open_time_str: Market open time in HH:MM format (UTC)
            close_time_str: Market close time in HH:MM format (UTC)
            timezone: Timezone of the datetime index
            
        Returns:
            Tuple of (market_hours_df, non_market_hours_df)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Ensure timezone-aware
        if df.index.tz is None:
            logger.warning("Timezone-naive index detected. Assuming UTC.")
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert(timezone)
        
        logger.info(f"Filtering for market hours ({open_time_str}-{close_time_str}) and non-market hours, weekdays only")
        
        # Filter for weekdays first (Monday=0, Sunday=6)
        weekdays_df = df[df.index.dayofweek < 5].copy()
        
        # Filter for market hours: [open_time, close_time)
        market_hours_df = weekdays_df.between_time(open_time_str, close_time_str, inclusive='left')
        
        # Filter for non-market hours: [close_time, open_time) on weekdays
        open_time = datetime.strptime(open_time_str, '%H:%M').time()
        close_time = datetime.strptime(close_time_str, '%H:%M').time()
        
        non_market_hours_mask = (
            (weekdays_df.index.time < open_time) |
            (weekdays_df.index.time >= close_time)
        )
        non_market_hours_df = weekdays_df[non_market_hours_mask]
        
        logger.info(f"Filtering complete. Market hours shape: {market_hours_df.shape}, "
                   f"Non-market hours shape: {non_market_hours_df.shape}")
        
        return market_hours_df, non_market_hours_df

    @staticmethod
    def filter_first_last_hour(
        df: pd.DataFrame,
        market_open_str: str = '13:30',
        one_hour_after_open_str: str = '14:30',
        one_hour_before_close_str: str = '19:00',
        market_close_str: str = '20:00',
        timezone: str = 'UTC'
    ) -> pd.DataFrame:
        """
        Filter DataFrame for the first and last hour of trading, excluding weekends.
        
        Args:
            df: DataFrame with datetime index
            market_open_str: Market open time in HH:MM format
            one_hour_after_open_str: One hour after open in HH:MM format
            one_hour_before_close_str: One hour before close in HH:MM format
            market_close_str: Market close time in HH:MM format
            timezone: Timezone of the datetime index
            
        Returns:
            Filtered DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Ensure timezone-aware
        if df.index.tz is None:
            logger.warning("Timezone-naive index detected. Assuming UTC.")
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert(timezone)
        
        logger.info(f"Filtering for first hour ({market_open_str}-{one_hour_after_open_str}) "
                   f"and last hour ({one_hour_before_close_str}-{market_close_str}), weekdays only")
        
        # Filter data for first hour after open
        open_hours = df.between_time(market_open_str, one_hour_after_open_str)
        open_hours = open_hours[open_hours.index.dayofweek < 5]  # Exclude weekends
        
        # Filter data for last hour before close
        close_hours = df.between_time(one_hour_before_close_str, market_close_str)
        close_hours = close_hours[close_hours.index.dayofweek < 5]  # Exclude weekends
        
        # Concatenate the two filtered segments
        filtered_data = pd.concat([open_hours, close_hours]).sort_index()
        
        logger.info(f"Filtering complete. Result shape: {filtered_data.shape}")
        return filtered_data

    @staticmethod
    def ensure_timezone(df: pd.DataFrame, timezone: str = 'UTC') -> pd.DataFrame:
        """
        Ensure DataFrame index is timezone-aware and in specified timezone.
        
        Args:
            df: DataFrame with datetime index
            timezone: Target timezone
            
        Returns:
            DataFrame with timezone-aware index
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        if df.index.tz is None:
            logger.info(f"Localizing naive datetime index to {timezone}")
            df.index = df.index.tz_localize(timezone)
        else:
            logger.info(f"Converting timezone-aware datetime index to {timezone}")
            df.index = df.index.tz_convert(timezone)
        
        return df

