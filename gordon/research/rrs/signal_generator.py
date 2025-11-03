"""
RRS Signal Generator
====================
Generates trading signals based on RRS analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RRSSignalGenerator:
    """
    Generates trading signals from RRS rankings.
    
    Creates buy/sell signals based on RRS thresholds and momentum.
    """
    
    def __init__(
        self,
        strong_threshold: float = 1.0,
        weak_threshold: float = -1.0
    ):
        """
        Initialize signal generator.
        
        Args:
            strong_threshold: RRS threshold for strong buy signals
            weak_threshold: RRS threshold for strong sell signals
        """
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
    
    def generate_signals(
        self,
        rankings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals from RRS rankings.
        
        Args:
            rankings_df: DataFrame with symbol rankings
            
        Returns:
            DataFrame with trading signals added
        """
        if rankings_df.empty:
            return rankings_df
        
        signals_df = rankings_df.copy()
        
        # Generate primary signals
        conditions = [
            (signals_df['current_rrs'] >= self.strong_threshold) &
            (signals_df['rrs_momentum'] > 0),
            (signals_df['current_rrs'] >= self.strong_threshold) &
            (signals_df['rrs_momentum'] <= 0),
            (signals_df['current_rrs'] <= self.weak_threshold) &
            (signals_df['rrs_momentum'] < 0),
            (signals_df['current_rrs'] <= self.weak_threshold) &
            (signals_df['rrs_momentum'] >= 0),
        ]
        
        choices = ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']
        
        signals_df['primary_signal'] = np.select(
            conditions,
            choices,
            default='HOLD'
        )
        
        # Generate confidence scores
        signals_df['signal_confidence'] = (
            np.abs(signals_df['current_rrs']) * signals_df['outperformance_ratio']
        )
        
        # Risk assessment
        signals_df['risk_level'] = pd.cut(
            signals_df['volatility'],
            bins=[0, 0.02, 0.05, 0.1, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Volume confirmation
        signals_df['volume_confirmation'] = signals_df['volume_ratio'] > 1.2
        
        # Trend confirmation
        signals_df['trend_confirmation'] = np.where(
            signals_df['primary_signal'].isin(['STRONG_BUY', 'BUY']),
            signals_df['rrs_trend'] > 0,
            np.where(
                signals_df['primary_signal'].isin(['STRONG_SELL', 'SELL']),
                signals_df['rrs_trend'] < 0,
                True
            )
        )
        
        logger.info(f"Generated trading signals for {len(signals_df)} symbols")
        
        signal_counts = signals_df['primary_signal'].value_counts()
        logger.info(f"Signal distribution: {signal_counts.to_dict()}")
        
        return signals_df
    
    def get_top_signals(
        self,
        signals_df: pd.DataFrame,
        signal_type: str = 'STRONG_BUY',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top signals of a specific type.
        
        Args:
            signals_df: DataFrame with signals
            signal_type: Type of signal to filter
            top_n: Number of top signals to return
            
        Returns:
            DataFrame with top signals
        """
        filtered = signals_df[signals_df['primary_signal'] == signal_type]
        filtered = filtered.sort_values('signal_confidence', ascending=False)
        
        return filtered.head(top_n)
    
    def get_signals_by_rank(
        self,
        signals_df: pd.DataFrame,
        min_rank: int = 1,
        max_rank: int = 20
    ) -> pd.DataFrame:
        """
        Get signals within a rank range.
        
        Args:
            signals_df: DataFrame with signals
            min_rank: Minimum rank
            max_rank: Maximum rank
            
        Returns:
            DataFrame with filtered signals
        """
        return signals_df[
            (signals_df['rank'] >= min_rank) &
            (signals_df['rank'] <= max_rank)
        ]

