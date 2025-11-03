"""
Trader Classifier
=================
Classifies traders into tiers based on trade size and behavior.
"""

import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TraderClassifier:
    """
    Classifies traders into tiers for analysis.
    
    Categories:
    - Whale: Very large trades (>$1M)
    - Institutional: Large trades (>$50K)
    - Professional: Medium trades (>$10K)
    - Retail: Small trades (<$10K)
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize trader classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Tier thresholds (USD)
        self.whale_threshold = self.config.get('whale_threshold', 1000000.0)
        self.institutional_threshold = self.config.get('institutional_threshold', 50000.0)
        self.professional_threshold = self.config.get('professional_threshold', 10000.0)
    
    def classify_trade(self, trade_size_usd: float) -> str:
        """
        Classify a single trade by size.
        
        Args:
            trade_size_usd: Trade size in USD
            
        Returns:
            Trader tier string
        """
        if trade_size_usd >= self.whale_threshold:
            return "Whale"
        elif trade_size_usd >= self.institutional_threshold:
            return "Institutional"
        elif trade_size_usd >= self.professional_threshold:
            return "Professional"
        else:
            return "Retail"
    
    def classify_traders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all traders in a DataFrame.
        
        Args:
            df: DataFrame with trades
            
        Returns:
            DataFrame with trader classifications added
        """
        if df.empty:
            return df
        
        df_classified = df.copy()
        
        # Classify each trade
        df_classified['trader_tier'] = df_classified['usd_value'].apply(
            self.classify_trade
        )
        
        # Add professional grade flag
        df_classified['is_professional'] = df_classified['trader_tier'].isin([
            'Institutional', 'Whale'
        ])
        
        # Add market impact assessment
        df_classified['market_impact'] = df_classified['usd_value'].apply(
            lambda x: 'High' if x >= 100000 else 'Medium' if x >= 25000 else 'Low'
        )
        
        return df_classified
    
    def get_trader_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trader profiles with aggregated statistics.
        
        Args:
            df: DataFrame with classified trades
            
        Returns:
            DataFrame with trader profiles
        """
        if df.empty:
            return pd.DataFrame()
        
        # Group by trader
        profiles = df.groupby('trader').agg({
            'usd_value': ['sum', 'count', 'mean', 'max', 'min'],
            'timestamp': ['min', 'max'],
            'trader_tier': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Retail',
            'trade_direction': lambda x: (x == 'BUY').sum() / len(x) if len(x) > 0 else 0
        }).reset_index()
        
        profiles.columns = [
            'trader',
            'total_volume',
            'trade_count',
            'avg_trade_size',
            'max_trade_size',
            'min_trade_size',
            'first_trade_time',
            'last_trade_time',
            'dominant_tier',
            'buy_ratio'
        ]
        
        # Classify overall trader tier based on average trade size
        profiles['trader_classification'] = profiles['avg_trade_size'].apply(
            self.classify_trade
        )
        
        # Add activity duration
        profiles['first_trade_time'] = pd.to_datetime(profiles['first_trade_time'])
        profiles['last_trade_time'] = pd.to_datetime(profiles['last_trade_time'])
        profiles['activity_duration_days'] = (
            (profiles['last_trade_time'] - profiles['first_trade_time']).dt.total_seconds() / 86400
        )
        
        # Sort by total volume
        profiles = profiles.sort_values('total_volume', ascending=False).reset_index(drop=True)
        
        # Add ranking
        profiles['rank'] = range(1, len(profiles) + 1)
        
        return profiles
    
    def filter_by_tier(
        self,
        df: pd.DataFrame,
        tiers: list = None
    ) -> pd.DataFrame:
        """
        Filter trades by trader tier.
        
        Args:
            df: DataFrame with trades
            tiers: List of tiers to include (e.g., ['Institutional', 'Whale'])
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        if tiers is None:
            tiers = ['Institutional', 'Whale']
        
        return df[df['trader_tier'].isin(tiers)]
    
    def get_early_buyers(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Get top early buyers (first trades by timestamp).
        
        Args:
            df: DataFrame with trades
            top_n: Number of early buyers to return
            
        Returns:
            DataFrame with early buyers
        """
        if df.empty:
            return pd.DataFrame()
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Get unique traders in order of first trade
        early_buyers = []
        seen_traders = set()
        
        for _, row in df_sorted.iterrows():
            trader = row['trader']
            if trader not in seen_traders:
                early_buyers.append(row)
                seen_traders.add(trader)
            
            if len(early_buyers) >= top_n:
                break
        
        if early_buyers:
            return pd.DataFrame(early_buyers)
        else:
            return pd.DataFrame()

