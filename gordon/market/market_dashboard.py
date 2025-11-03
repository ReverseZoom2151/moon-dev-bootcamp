"""
Market Dashboard
================
Day 47: Unified market monitoring dashboard for Binance and Bitfinex.
"""

import pandas as pd
import numpy as np
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .exchange_clients import BinanceMarketClient, BitfinexMarketClient
from .display import MarketDisplay
from .funding_analysis import FundingRateAnalyzer

logger = logging.getLogger(__name__)


class MarketDashboard:
    """
    Unified market dashboard for monitoring tokens across exchanges.
    
    Features:
    - Trending tokens tracking
    - New listings detection
    - Volume leaders analysis
    - Consistent trending historical tracking
    - Possible gems identification
    - Funding rates analysis (Bitfinex)
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        exchange_name: str = 'binance',
        config: Optional[Dict] = None
    ):
        """
        Initialize market dashboard.
        
        Args:
            exchange_adapter: Exchange adapter instance
            exchange_name: Exchange name ('binance' or 'bitfinex')
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.exchange_name = exchange_name.lower()
        self.config = config or {}
        
        # Initialize exchange-specific client
        if self.exchange_name == 'binance':
            self.market_client = BinanceMarketClient(exchange_adapter, config)
        elif self.exchange_name == 'bitfinex':
            self.market_client = BitfinexMarketClient(exchange_adapter, config)
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")
        
        # Initialize display and funding analyzer
        self.display = MarketDisplay(config)
        self.funding_analyzer = FundingRateAnalyzer(
            exchange_adapter if self.exchange_name == 'bitfinex' else None,
            config
        )
        
        # Data directory
        self.output_dir = Path(self.config.get('output_dir', './market_dashboard_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Historical tracking
        self.history_file = self.output_dir / f'{self.exchange_name}_trending_history.csv'
        self.trending_history: Optional[pd.DataFrame] = None
    
    async def run_full_analysis(self, export_csv: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run full market analysis.
        
        Args:
            export_csv: Whether to export results to CSV
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info(f"Running market dashboard analysis for {self.exchange_name}")
        
        results = {}
        
        # Fetch trending tokens
        trending_df = await self.market_client.fetch_trending_tokens()
        results['trending'] = trending_df
        
        # Fetch new listings
        new_listings_df = await self.market_client.fetch_new_listings()
        results['new_listings'] = new_listings_df
        
        # Fetch volume leaders
        volume_df = await self.market_client.fetch_high_volume_tokens()
        results['volume_leaders'] = volume_df
        
        # Update trending history
        if not trending_df.empty:
            self.trending_history = self._update_trending_history(trending_df)
            results['consistent_trending'] = self.trending_history
        
        # Identify possible gems
        if not trending_df.empty:
            gems_df = self._identify_possible_gems(trending_df)
            results['possible_gems'] = gems_df
        
        # Fetch funding rates (Bitfinex only)
        if self.exchange_name == 'bitfinex' and self.config.get('include_funding_analysis', True):
            funding_df = await self.funding_analyzer.fetch_funding_rates()
            results['funding_rates'] = funding_df
            
            # Analyze arbitrage opportunities
            if not funding_df.empty:
                arbitrage_df = self.funding_analyzer.analyze_arbitrage_opportunities(funding_df)
                results['arbitrage_opportunities'] = arbitrage_df
        
        # Export to CSV if requested
        if export_csv:
            saved_files = self._export_to_csv(results)
            results['saved_files'] = saved_files
        
        return results
    
    def display_results(self, results: Dict[str, pd.DataFrame]):
        """
        Display all analysis results.
        
        Args:
            results: Dictionary with analysis results
        """
        exchange_display_name = self.exchange_name.upper()
        
        # Display trending tokens
        if 'trending' in results and not results['trending'].empty:
            self.display.display_trending_tokens(results['trending'], exchange_display_name)
        
        # Display new listings
        if 'new_listings' in results and not results['new_listings'].empty:
            self.display.display_new_listings(results['new_listings'], exchange_display_name)
        
        # Display volume leaders
        if 'volume_leaders' in results and not results['volume_leaders'].empty:
            self.display.display_volume_leaders(results['volume_leaders'], exchange_display_name)
        
        # Display consistent trending
        if 'consistent_trending' in results and results['consistent_trending'] is not None:
            if not results['consistent_trending'].empty:
                self.display.display_consistent_trending(results['consistent_trending'], exchange_display_name)
        
        # Display possible gems
        if 'possible_gems' in results and not results['possible_gems'].empty:
            self.display.display_possible_gems(results['possible_gems'], exchange_display_name)
        
        # Display funding rates (Bitfinex only)
        if self.exchange_name == 'bitfinex' and 'funding_rates' in results:
            if not results['funding_rates'].empty:
                self.funding_analyzer.display_funding_rates(results['funding_rates'])
    
    def _update_trending_history(self, trending_df: pd.DataFrame) -> pd.DataFrame:
        """
        Update historical tracking of trending tokens.
        
        Args:
            trending_df: DataFrame with current trending tokens
            
        Returns:
            Updated history DataFrame
        """
        if trending_df.empty:
            return pd.DataFrame()
        
        # Add timestamp
        df_with_timestamp = trending_df.copy()
        df_with_timestamp['timestamp'] = datetime.now()
        df_with_timestamp['date'] = datetime.now().strftime('%Y-%m-%d')
        df_with_timestamp['hour'] = datetime.now().strftime('%H')
        df_with_timestamp['exchange'] = self.exchange_name
        
        # Load existing history or create new
        if self.history_file.exists():
            try:
                history_df = pd.read_csv(self.history_file)
                if 'timestamp' in history_df.columns:
                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            except Exception as e:
                logger.warning(f"Error loading history file: {e}")
                history_df = pd.DataFrame()
        else:
            history_df = pd.DataFrame()
        
        # Combine with new data
        combined_df = pd.concat([history_df, df_with_timestamp], ignore_index=True)
        
        # Keep only recent data (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df[pd.to_datetime(combined_df['timestamp']) > cutoff_date]
        
        # Save updated history
        try:
            combined_df.to_csv(self.history_file, index=False)
            logger.info(f"Updated trending history with {len(df_with_timestamp)} tokens")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
        
        return combined_df
    
    def _identify_possible_gems(self, trending_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify possible gem tokens (low price, high volume potential).
        
        Args:
            trending_df: DataFrame with trending tokens
            
        Returns:
            DataFrame with possible gems
        """
        if trending_df.empty:
            return pd.DataFrame()
        
        gems_max_price = self.config.get('gems_max_price', 1.0)
        gems_min_volume = self.config.get('gems_min_volume', 100000)
        
        # Filter for gems
        gems_df = trending_df[
            (trending_df['price'].fillna(float('inf')) <= gems_max_price) & 
            (trending_df['volume24hUSD'].fillna(0) >= gems_min_volume)
        ].copy()
        
        return gems_df
    
    def _export_to_csv(self, results: Dict[str, pd.DataFrame]) -> Dict[str, Path]:
        """
        Export results to CSV files.
        
        Args:
            results: Dictionary with analysis results
            
        Returns:
            Dictionary with saved file paths
        """
        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for key, df in results.items():
            if key == 'saved_files' or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            
            try:
                filename = f'{self.exchange_name}_{key}_{timestamp}.csv'
                file_path = self.output_dir / filename
                df.to_csv(file_path, index=False, float_format='%.2f')
                saved_files[key] = file_path
                logger.info(f"Saved {key} to {file_path}")
            except Exception as e:
                logger.error(f"Error saving {key} to CSV: {e}")
        
        return saved_files
    
    def get_summary_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """
        Generate summary report.
        
        Args:
            results: Dictionary with analysis results
            
        Returns:
            Formatted summary report string
        """
        report_lines = [
            f"📊 MARKET DASHBOARD SUMMARY - {self.exchange_name.upper()}",
            "=" * 80,
            ""
        ]
        
        # Trending tokens summary
        if 'trending' in results and not results['trending'].empty:
            trending_df = results['trending']
            report_lines.append(f"🚀 Trending Tokens: {len(trending_df)}")
            if not trending_df.empty:
                avg_change = trending_df['price24hChangePercent'].mean()
                total_volume = trending_df['volume24hUSD'].sum()
                report_lines.append(f"   Average 24h Change: {avg_change:+.2f}%")
                report_lines.append(f"   Total Volume: ${total_volume:,.0f}")
            report_lines.append("")
        
        # New listings summary
        if 'new_listings' in results and not results['new_listings'].empty:
            new_listings_df = results['new_listings']
            report_lines.append(f"🌟 New Listings: {len(new_listings_df)}")
            report_lines.append("")
        
        # Volume leaders summary
        if 'volume_leaders' in results and not results['volume_leaders'].empty:
            volume_df = results['volume_leaders']
            report_lines.append(f"📊 Volume Leaders: {len(volume_df)}")
            if not volume_df.empty:
                top_volume = volume_df['volume24hUSD'].max()
                report_lines.append(f"   Top Volume: ${top_volume:,.0f}")
            report_lines.append("")
        
        # Possible gems summary
        if 'possible_gems' in results and not results['possible_gems'].empty:
            gems_df = results['possible_gems']
            report_lines.append(f"💎 Possible Gems: {len(gems_df)}")
            report_lines.append("")
        
        # Funding rates summary (Bitfinex only)
        if self.exchange_name == 'bitfinex' and 'funding_rates' in results:
            if not results['funding_rates'].empty:
                funding_df = results['funding_rates']
                report_lines.append(f"💰 Funding Rates: {len(funding_df)} tokens")
                if not funding_df.empty:
                    high_funding = funding_df[funding_df['funding_rate'].abs() > self.config.get('funding_rate_threshold', 0.01)]
                    if not high_funding.empty:
                        report_lines.append(f"   High Funding Rates (>1%): {len(high_funding)}")
                report_lines.append("")
        
        # Saved files
        if 'saved_files' in results:
            saved_files = results['saved_files']
            if saved_files:
                report_lines.append("💾 CSV Files Saved:")
                for key, path in saved_files.items():
                    report_lines.append(f"   • {key}: {path}")
        
        return "\n".join(report_lines)

