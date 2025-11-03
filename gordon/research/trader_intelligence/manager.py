"""
Trader Intelligence Manager
===========================
Orchestrates early buyer analysis and trader classification.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .early_buyer_analyzer import EarlyBuyerAnalyzer
from .trader_classifier import TraderClassifier

logger = logging.getLogger(__name__)


class TraderIntelligenceManager:
    """
    Manages trader intelligence analysis.
    
    Coordinates early buyer analysis, trader classification,
    and integration with social signal trading.
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize trader intelligence manager.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Initialize components
        self.analyzer = EarlyBuyerAnalyzer(exchange_adapter, config)
        self.classifier = TraderClassifier(config.get('trader_classification', {}))
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', './trader_intelligence_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_symbol(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive trader analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (defaults to lookback_days ago)
            end_date: End date (defaults to now)
            lookback_days: Days to look back if dates not provided
            
        Returns:
            Dictionary with analysis results
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        
        if start_date is None:
            start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"Analyzing trader intelligence for {symbol}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Analyze early buyers
        trades_df = self.analyzer.analyze_early_buyers(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            sort_by='timestamp'
        )
        
        if trades_df.empty:
            logger.warning(f"No trades found for {symbol}")
            return {
                'trades': pd.DataFrame(),
                'classified_trades': pd.DataFrame(),
                'top_traders': pd.DataFrame(),
                'early_buyers': pd.DataFrame(),
                'trader_profiles': pd.DataFrame()
            }
        
        # Classify traders
        classified_df = self.classifier.classify_traders(trades_df)
        
        # Get top traders
        top_traders = self.analyzer.get_top_traders(
            classified_df,
            top_n=100,
            metric='total_volume'
        )
        
        # Get early buyers
        early_buyers = self.classifier.get_early_buyers(
            classified_df,
            top_n=50
        )
        
        # Generate trader profiles
        trader_profiles = self.classifier.get_trader_profiles(classified_df)
        
        # Save results
        self._save_results(symbol, {
            'trades': trades_df,
            'classified_trades': classified_df,
            'top_traders': top_traders,
            'early_buyers': early_buyers,
            'trader_profiles': trader_profiles
        })
        
        # Summary statistics
        logger.info(f"✅ Analysis Complete:")
        logger.info(f"  Total trades: {len(trades_df)}")
        logger.info(f"  Unique traders: {len(trader_profiles)}")
        logger.info(f"  Institutional/Whale trades: {len(classified_df[classified_df['is_professional']])}")
        logger.info(f"  Early buyers identified: {len(early_buyers)}")
        
        return {
            'trades': trades_df,
            'classified_trades': classified_df,
            'top_traders': top_traders,
            'early_buyers': early_buyers,
            'trader_profiles': trader_profiles
        }
    
    def get_accounts_to_follow(
        self,
        symbol: str,
        max_accounts: int = 20,
        min_institutional_trades: int = 3
    ) -> List[str]:
        """
        Get list of accounts to follow based on early buyer analysis.
        
        Args:
            symbol: Trading symbol to analyze
            max_accounts: Maximum accounts to return
            min_institutional_trades: Minimum institutional trades required
            
        Returns:
            List of trader addresses/IDs to follow
        """
        logger.info(f"Finding accounts to follow for {symbol}")
        
        # Analyze symbol
        results = self.analyze_symbol(symbol, lookback_days=7)
        
        trader_profiles = results.get('trader_profiles', pd.DataFrame())
        
        if trader_profiles.empty:
            logger.warning("No trader profiles found")
            return []
        
        # Filter for institutional/professional traders
        institutional = trader_profiles[
            trader_profiles['trader_classification'].isin(['Institutional', 'Whale'])
        ]
        
        # Filter by minimum trade count
        institutional = institutional[
            institutional['trade_count'] >= min_institutional_trades
        ]
        
        # Sort by total volume
        institutional = institutional.sort_values('total_volume', ascending=False)
        
        # Get account list
        accounts = institutional['trader'].head(max_accounts).tolist()
        
        logger.info(f"Found {len(accounts)} accounts to follow")
        
        return accounts
    
    def _save_results(self, symbol: str, results: Dict[str, pd.DataFrame]):
        """Save analysis results to CSV files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for result_type, df in results.items():
            if not df.empty:
                filename = f"{symbol}_{result_type}_{timestamp}.csv"
                self.analyzer.save_results(df, filename)

