"""
RRS Calculator
==============
Unified Relative Rotation Strength calculator supporting all exchanges.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RRSCalculator:
    """
    Calculates Relative Rotation Strength (RRS) metrics.
    
    RRS compares a symbol's performance relative to a benchmark,
    combining returns, volatility, and volume factors.
    """
    
    def __init__(
        self,
        smoothing_window: int = 5,
        volatility_lookback: int = 20,
        volume_lookback: int = 10,
        min_data_points: int = 50
    ):
        """
        Initialize RRS calculator.
        
        Args:
            smoothing_window: Number of periods for RRS smoothing
            volatility_lookback: Periods for volatility calculation
            volume_lookback: Periods for volume analysis
            min_data_points: Minimum data points required for valid calculation
        """
        self.smoothing_window = smoothing_window
        self.volatility_lookback = volatility_lookback
        self.volume_lookback = volume_lookback
        self.min_data_points = min_data_points
    
    def calculate_rrs(
        self,
        symbol_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        symbol_name: str = "Unknown",
        benchmark_name: str = "BTC"
    ) -> pd.DataFrame:
        """
        Calculate RRS metrics for symbol vs benchmark.
        
        Args:
            symbol_df: DataFrame with 'timestamp', 'log_return', 'volatility', 'volume_ratio'
            benchmark_df: DataFrame with 'timestamp', 'log_return'
            symbol_name: Name of symbol for logging
            benchmark_name: Name of benchmark for logging
            
        Returns:
            DataFrame with RRS metrics added
        """
        required_symbol_cols = ['timestamp', 'log_return', 'volatility', 'volume_ratio']
        required_benchmark_cols = ['timestamp', 'log_return']
        
        if not all(col in symbol_df.columns for col in required_symbol_cols):
            logger.error(f"Symbol DataFrame missing required columns: {required_symbol_cols}")
            return pd.DataFrame()
        
        if not all(col in benchmark_df.columns for col in required_benchmark_cols):
            logger.error(f"Benchmark DataFrame missing required columns: {required_benchmark_cols}")
            return pd.DataFrame()
        
        logger.debug(f"Calculating RRS for {symbol_name} vs {benchmark_name}")
        
        # Create copies
        symbol_df = symbol_df.copy()
        benchmark_df = benchmark_df.copy()
        
        # Ensure timestamps are datetime
        symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
        benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
        
        # Set timestamp as index
        symbol_df = symbol_df.set_index('timestamp')
        benchmark_df = benchmark_df.set_index('timestamp')
        
        # Join DataFrames (inner join for overlapping periods)
        df_merged = symbol_df.join(
            benchmark_df[['log_return']],
            rsuffix='_benchmark',
            how='inner'
        )
        
        if df_merged.empty:
            logger.warning(f"No overlapping data between {symbol_name} and {benchmark_name}")
            return pd.DataFrame()
        
        # Check minimum data points
        if len(df_merged) < self.min_data_points:
            logger.warning(f"Insufficient data points: {len(df_merged)} < {self.min_data_points}")
            return pd.DataFrame()
        
        # Calculate differential returns
        df_merged['differential_return'] = (
            df_merged['log_return'] - df_merged['log_return_benchmark']
        )
        
        # Cumulative differential returns
        df_merged['cumulative_differential_return'] = df_merged['differential_return'].cumsum()
        
        # Rolling statistics for normalization
        rolling_window = min(self.smoothing_window * 4, len(df_merged) // 4)
        rolling_window = max(rolling_window, 5)
        
        rolling_mean = df_merged['differential_return'].rolling(
            window=rolling_window,
            min_periods=max(1, rolling_window // 2)
        ).mean()
        
        rolling_std = df_merged['differential_return'].rolling(
            window=rolling_window,
            min_periods=max(1, rolling_window // 2)
        ).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.fillna(1.0).replace(0, 1.0)
        
        # Normalized differential returns (z-score)
        df_merged['normalized_return'] = (
            (df_merged['differential_return'] - rolling_mean) / rolling_std
        )
        
        # Calculate raw RRS score
        volume_factor = np.log1p(df_merged['volume_ratio'])
        volatility_factor = 1 / (1 + df_merged['volatility'])
        
        df_merged['raw_rrs'] = (
            df_merged['normalized_return'] * 0.6 +  # 60% weight on returns
            volume_factor * 0.25 +                  # 25% weight on volume
            volatility_factor * 0.15                # 15% weight on volatility
        )
        
        # Smoothed RRS using exponential moving average
        smoothing_alpha = 2 / (self.smoothing_window + 1)
        df_merged['smoothed_rrs'] = df_merged['raw_rrs'].ewm(
            alpha=smoothing_alpha,
            min_periods=1
        ).mean()
        
        # RRS momentum
        df_merged['rrs_momentum'] = df_merged['smoothed_rrs'].diff(periods=3)
        
        # RRS percentile rank
        df_merged['rrs_percentile'] = df_merged['smoothed_rrs'].rolling(
            window=min(50, len(df_merged)),
            min_periods=10
        ).rank(pct=True) * 100
        
        # Additional strength indicators
        df_merged['outperformance_ratio'] = (
            (df_merged['differential_return'] > 0).rolling(
                window=rolling_window,
                min_periods=max(1, rolling_window // 2)
            ).mean()
        )
        
        # Relative strength trend
        df_merged['rrs_trend'] = df_merged['smoothed_rrs'].rolling(
            window=min(10, len(df_merged) // 2),
            min_periods=3
        ).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 2 else 0,
            raw=True
        )
        
        # Risk-adjusted RRS
        returns_std = df_merged['differential_return'].rolling(
            window=rolling_window,
            min_periods=max(1, rolling_window // 2)
        ).std()
        returns_std = returns_std.fillna(1.0).replace(0, 1.0)
        
        df_merged['risk_adjusted_rrs'] = (
            df_merged['cumulative_differential_return'] / returns_std
        )
        
        # Handle infinite values and NaNs
        df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        numeric_columns = [
            'differential_return', 'normalized_return', 'raw_rrs',
            'smoothed_rrs', 'rrs_momentum', 'rrs_percentile',
            'outperformance_ratio', 'rrs_trend', 'risk_adjusted_rrs'
        ]
        
        for col in numeric_columns:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0)
        
        # Reset index
        df_merged = df_merged.reset_index()
        
        # Add metadata
        df_merged['symbol_name'] = symbol_name
        df_merged['benchmark_name'] = benchmark_name
        
        logger.info(f"Successfully calculated RRS for {symbol_name} vs {benchmark_name}")
        
        return df_merged
    
    def rank_symbols_by_rrs(
        self,
        rrs_data_dict: Dict[str, pd.DataFrame],
        ranking_metric: str = 'smoothed_rrs'
    ) -> pd.DataFrame:
        """
        Rank multiple symbols by their RRS scores.
        
        Args:
            rrs_data_dict: Dict mapping symbol names to RRS DataFrames
            ranking_metric: Column to use for ranking
            
        Returns:
            DataFrame with symbol rankings
        """
        rankings = []
        
        for symbol, rrs_df in rrs_data_dict.items():
            if rrs_df.empty or ranking_metric not in rrs_df.columns:
                continue
            
            # Get latest values
            latest_data = rrs_df.iloc[-1]
            
            # Calculate summary statistics
            rrs_values = rrs_df[ranking_metric].dropna()
            if len(rrs_values) == 0:
                continue
            
            ranking_data = {
                'symbol': symbol,
                'current_rrs': latest_data[ranking_metric],
                'rrs_momentum': latest_data.get('rrs_momentum', 0),
                'rrs_trend': latest_data.get('rrs_trend', 0),
                'outperformance_ratio': latest_data.get('outperformance_ratio', 0),
                'risk_adjusted_rrs': latest_data.get('risk_adjusted_rrs', 0),
                'rrs_percentile': latest_data.get('rrs_percentile', 50),
                'volatility': latest_data.get('volatility', 0),
                'volume_ratio': latest_data.get('volume_ratio', 1),
                'data_points': len(rrs_df),
                'timestamp': latest_data['timestamp']
            }
            
            rankings.append(ranking_data)
        
        if not rankings:
            logger.warning("No valid rankings data available")
            return pd.DataFrame()
        
        # Create rankings DataFrame
        rankings_df = pd.DataFrame(rankings)
        
        # Sort by RRS score (descending)
        rankings_df = rankings_df.sort_values('current_rrs', ascending=False).reset_index(drop=True)
        
        # Add ranking position
        rankings_df['rank'] = range(1, len(rankings_df) + 1)
        
        # Add strength categories
        rankings_df['strength_category'] = pd.cut(
            rankings_df['current_rrs'],
            bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
            labels=['Very Weak', 'Weak', 'Neutral', 'Strong', 'Very Strong']
        )
        
        logger.info(f"Generated rankings for {len(rankings_df)} symbols")
        
        return rankings_df

