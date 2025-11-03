"""
Whale Tracking Manager
======================
Day 44 (Base) + Day 46 (Enhanced): Coordinates whale position tracking across exchanges.
Enhanced with multi-address tracking, liquidation risk analysis, and position aggregation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .whale_tracker import WhalePositionTracker
from .multi_address_tracker import MultiAddressWhaleTracker
from .trading_utils import EnhancedTradingUtils
from .position_sizing import PositionSizingHelper

logger = logging.getLogger(__name__)


class WhaleTrackingManager:
    """
    Manages whale position tracking across exchanges.
    
    Coordinates whale tracking, position analysis, and movement detection.
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize whale tracking manager.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Initialize components
        self.tracker = WhalePositionTracker(exchange_adapter, config)
        self.multi_address_tracker = MultiAddressWhaleTracker(exchange_adapter, config)
        self.utils = EnhancedTradingUtils(config)
        self.sizing = PositionSizingHelper(config.get('position_sizing', {}))
        
        # Cache for tracking movements
        self.previous_positions: Dict[str, pd.DataFrame] = {}
    
    async def track_whales(
        self,
        symbol: Optional[str] = None,
        min_value_usd: Optional[float] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Track whale positions and generate analysis.
        
        Args:
            symbol: Optional symbol to filter by
            min_value_usd: Minimum position value threshold
            
        Returns:
            Dictionary with whale tracking results
        """
        logger.info(f"Tracking whale positions for {symbol or 'all symbols'}")
        
        # Get current large positions
        current_positions = await self.tracker.get_large_positions(
            symbol=symbol,
            min_value_usd=min_value_usd
        )
        
        if current_positions.empty:
            logger.warning("No large positions found")
            return {
                'current_positions': pd.DataFrame(),
                'whale_positions': pd.DataFrame(),
                'top_positions': pd.DataFrame(),
                'movements': {},
                'summary': {}
            }
        
        # Filter for whales only
        whale_positions = self.tracker.get_whale_positions(current_positions)
        
        # Get top positions
        top_positions = self.tracker.get_top_positions(current_positions)
        
        # Analyze movements
        cache_key = symbol or 'all'
        previous_positions = self.previous_positions.get(cache_key, pd.DataFrame())
        movements = self.tracker.analyze_whale_movements(
            current_positions,
            previous_positions
        )
        
        # Update cache
        self.previous_positions[cache_key] = current_positions.copy()
        
        # Generate summary
        summary = self.tracker.get_whale_summary(whale_positions)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not current_positions.empty:
            filename = f"whale_positions_{symbol or 'all'}_{timestamp}.csv"
            self.tracker.save_positions(current_positions, filename)
        
        return {
            'current_positions': current_positions,
            'whale_positions': whale_positions,
            'top_positions': top_positions,
            'movements': movements,
            'summary': summary
        }
    
    def get_whale_summary_report(self, results: Dict) -> str:
        """
        Generate a formatted summary report.
        
        Args:
            results: Results dictionary from track_whales
            
        Returns:
            Formatted report string
        """
        summary = results.get('summary', {})
        whale_positions = results.get('whale_positions', pd.DataFrame())
        
        if whale_positions.empty:
            return "No whale positions found."
        
        report_lines = [
            "🐋 WHALE POSITION SUMMARY",
            "=" * 60,
            f"Total Whale Positions: {summary.get('total_positions', 0)}",
            f"Total Value: ${summary.get('total_value_usd', 0):,.2f}",
            f"Total Unrealized PnL: ${summary.get('total_unrealized_pnl', 0):,.2f}",
            f"Average Position Value: ${summary.get('avg_position_value', 0):,.2f}",
            f"Max Position Value: ${summary.get('max_position_value', 0):,.2f}",
            f"Whale Count: {summary.get('whale_count', 0)}",
            f"Institutional Count: {summary.get('institutional_count', 0)}",
            "",
            "Top Symbols by Value:",
        ]
        
        top_symbols = summary.get('top_symbols', {})
        for i, (symbol, value) in enumerate(list(top_symbols.items())[:10], 1):
            report_lines.append(f"  {i:2d}. {symbol:10s} ${value:>12,.2f}")
        
        # Add movements if available
        movements = results.get('movements', {})
        if movements:
            report_lines.append("")
            report_lines.append("Recent Movements:")
            if not movements.get('new_positions', pd.DataFrame()).empty:
                report_lines.append(f"  New Positions: {len(movements['new_positions'])}")
            if not movements.get('closed_positions', pd.DataFrame()).empty:
                report_lines.append(f"  Closed Positions: {len(movements['closed_positions'])}")
            if not movements.get('increased_positions', pd.DataFrame()).empty:
                report_lines.append(f"  Increased Positions: {len(movements['increased_positions'])}")
            if not movements.get('decreased_positions', pd.DataFrame()).empty:
                report_lines.append(f"  Decreased Positions: {len(movements['decreased_positions'])}")
        
        return "\n".join(report_lines)
    
    # ========== DAY 46: MULTI-ADDRESS TRACKING ==========
    
    async def track_multi_address_whales(
        self,
        addresses: Optional[List[str]] = None,
        symbol: Optional[str] = None,
        min_value_usd: Optional[float] = None,
        export_csv: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Track positions for multiple whale addresses.
        
        Args:
            addresses: List of addresses to track. If None, loads from file.
            symbol: Optional symbol to filter by
            min_value_usd: Minimum position value threshold
            export_csv: Whether to export results to CSV
            
        Returns:
            Dictionary with tracking results
        """
        logger.info("Starting multi-address whale tracking")
        
        # Load addresses if not provided
        if addresses is None:
            addresses = self.multi_address_tracker.load_whale_addresses()
        
        if not addresses:
            logger.warning("No whale addresses available")
            return {
                'positions': pd.DataFrame(),
                'aggregated': pd.DataFrame(),
                'liquidation_risk': pd.DataFrame(),
                'closest_long': pd.DataFrame(),
                'closest_short': pd.DataFrame()
            }
        
        # Fetch positions for all addresses
        positions_df = await self.multi_address_tracker.fetch_all_positions_parallel(
            addresses=addresses,
            symbol=symbol
        )
        
        if positions_df.empty:
            logger.warning("No positions found for tracked addresses")
            return {
                'positions': pd.DataFrame(),
                'aggregated': pd.DataFrame(),
                'liquidation_risk': pd.DataFrame(),
                'closest_long': pd.DataFrame(),
                'closest_short': pd.DataFrame()
            }
        
        # Filter by minimum value
        if min_value_usd:
            positions_df = positions_df[positions_df['position_value'] >= min_value_usd].copy()
        
        # Aggregate positions
        aggregated_df = self.multi_address_tracker.aggregate_positions(positions_df)
        
        # Get current prices for liquidation risk analysis
        current_prices = {}
        if not positions_df.empty and 'coin' in positions_df.columns:
            unique_coins = positions_df['coin'].unique()
            if self.exchange_adapter:
                for coin in unique_coins:
                    try:
                        ticker = await self.exchange_adapter.get_ticker(coin)
                        if ticker and 'last' in ticker:
                            current_prices[coin] = float(ticker['last'])
                    except Exception as e:
                        logger.debug(f"Could not get price for {coin}: {e}")
        
        # Calculate liquidation risk
        liquidation_risk_df = self.multi_address_tracker.calculate_liquidation_risk(
            positions_df,
            current_prices if current_prices else None
        )
        
        # Get positions closest to liquidation
        closest_long, closest_short = self.multi_address_tracker.get_positions_closest_to_liquidation(
            positions_df,
            current_prices=current_prices if current_prices else None
        )
        
        # Export to CSV if requested
        saved_files = {}
        if export_csv:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save all positions
            positions_file = self.multi_address_tracker.save_positions_to_csv(
                positions_df,
                f'multi_address_positions_{timestamp}.csv'
            )
            if positions_file:
                saved_files['positions'] = positions_file
            
            # Save aggregated positions
            agg_file = self.multi_address_tracker.save_aggregated_positions(
                positions_df,
                f'aggregated_positions_{timestamp}.csv'
            )
            if agg_file:
                saved_files['aggregated'] = agg_file
            
            # Save liquidation risk positions
            liq_files = self.multi_address_tracker.save_liquidation_risk_positions(
                positions_df,
                current_prices=current_prices if current_prices else None
            )
            saved_files.update(liq_files)
        
        return {
            'positions': positions_df,
            'aggregated': aggregated_df,
            'liquidation_risk': liquidation_risk_df,
            'closest_long': closest_long,
            'closest_short': closest_short,
            'saved_files': saved_files
        }
    
    async def get_liquidation_risk_report(
        self,
        positions_df: pd.DataFrame,
        threshold_pct: float = 3.0,
        top_n: int = 10
    ) -> str:
        """
        Generate liquidation risk report.
        
        Args:
            positions_df: DataFrame with positions
            threshold_pct: Maximum distance to liquidation (%)
            top_n: Number of top positions to show
            
        Returns:
            Formatted report string
        """
        if positions_df.empty:
            return "No positions available for liquidation risk analysis."
        
        # Get current prices asynchronously
        current_prices = {}
        if self.exchange_adapter:
            unique_coins = positions_df['coin'].unique()
            for coin in unique_coins:
                try:
                    ticker = await self.exchange_adapter.get_ticker(coin)
                    if ticker and 'last' in ticker:
                        current_prices[coin] = float(ticker['last'])
                except Exception as e:
                    logger.debug(f"Could not get price for {coin}: {e}")
        
        # Calculate liquidation risk
        risk_df = self.multi_address_tracker.calculate_liquidation_risk(
            positions_df,
            current_prices if current_prices else None
        )
        
        if risk_df.empty:
            return "No positions with liquidation prices found."
        
        # Filter by threshold
        risk_df = risk_df[risk_df['distance_to_liq_pct'] <= threshold_pct].copy()
        
        if risk_df.empty:
            return f"No positions within {threshold_pct}% of liquidation."
        
        # Split by direction
        longs = risk_df[risk_df['is_long']].head(top_n)
        shorts = risk_df[~risk_df['is_long']].head(top_n)
        
        report_lines = [
            f"💥 LIQUIDATION RISK ANALYSIS (within {threshold_pct}% of liquidation)",
            "=" * 80,
            "",
            f"🚀 TOP {len(longs)} LONG POSITIONS CLOSEST TO LIQUIDATION 📈",
            "-" * 80
        ]
        
        if not longs.empty:
            for i, (_, row) in enumerate(longs.iterrows(), 1):
                report_lines.append(
                    f"#{i:2d} {row['coin']:8s} | "
                    f"Value: ${row['position_value']:>12,.2f} | "
                    f"Entry: ${row['entry_price']:>10,.2f} | "
                    f"Liq: ${row['liquidation_price']:>10,.2f} | "
                    f"Distance: {row['distance_to_liq_pct']:>5.2f}% | "
                    f"Leverage: {row['leverage']:>4.1f}x"
                )
                report_lines.append(f"    Address: {row['address']}")
        else:
            report_lines.append("No long positions at risk.")
        
        report_lines.append("")
        report_lines.append(f"💥 TOP {len(shorts)} SHORT POSITIONS CLOSEST TO LIQUIDATION 📉")
        report_lines.append("-" * 80)
        
        if not shorts.empty:
            for i, (_, row) in enumerate(shorts.iterrows(), 1):
                report_lines.append(
                    f"#{i:2d} {row['coin']:8s} | "
                    f"Value: ${row['position_value']:>12,.2f} | "
                    f"Entry: ${row['entry_price']:>10,.2f} | "
                    f"Liq: ${row['liquidation_price']:>10,.2f} | "
                    f"Distance: {row['distance_to_liq_pct']:>5.2f}% | "
                    f"Leverage: {row['leverage']:>4.1f}x"
                )
                report_lines.append(f"    Address: {row['address']}")
        else:
            report_lines.append("No short positions at risk.")
        
        # Summary statistics
        total_long_value = longs['position_value'].sum() if not longs.empty else 0
        total_short_value = shorts['position_value'].sum() if not shorts.empty else 0
        
        report_lines.append("")
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total Long Value at Risk: ${total_long_value:,.2f}")
        report_lines.append(f"  Total Short Value at Risk: ${total_short_value:,.2f}")
        report_lines.append(f"  Total Positions at Risk: {len(risk_df)}")
        
        return "\n".join(report_lines)
    
    def get_aggregated_positions_report(
        self,
        positions_df: pd.DataFrame
    ) -> str:
        """
        Generate aggregated positions report.
        
        Args:
            positions_df: DataFrame with positions
            
        Returns:
            Formatted report string
        """
        if positions_df.empty:
            return "No positions available for aggregation."
        
        agg_df = self.multi_address_tracker.aggregate_positions(positions_df)
        
        if agg_df.empty:
            return "No aggregated positions found."
        
        report_lines = [
            "📊 AGGREGATED POSITIONS BY COIN & DIRECTION",
            "=" * 80,
            ""
        ]
        
        # Top long positions
        longs = agg_df[agg_df['is_long']].head(10)
        if not longs.empty:
            report_lines.append("🚀 TOP LONG POSITIONS (AGGREGATED):")
            report_lines.append("-" * 80)
            for i, (_, row) in enumerate(longs.iterrows(), 1):
                report_lines.append(
                    f"#{i:2d} {row['coin']:8s} | "
                    f"Total Value: ${row['total_value']:>12,.2f} | "
                    f"Traders: {row['num_traders']:>4d} | "
                    f"Avg Value: ${row['avg_value_per_trader']:>10,.2f} | "
                    f"Avg Leverage: {row['avg_leverage']:>5.1f}x | "
                    f"PnL: ${row['total_pnl']:>12,.2f}"
                )
            report_lines.append("")
        
        # Top short positions
        shorts = agg_df[~agg_df['is_long']].head(10)
        if not shorts.empty:
            report_lines.append("💥 TOP SHORT POSITIONS (AGGREGATED):")
            report_lines.append("-" * 80)
            for i, (_, row) in enumerate(shorts.iterrows(), 1):
                report_lines.append(
                    f"#{i:2d} {row['coin']:8s} | "
                    f"Total Value: ${row['total_value']:>12,.2f} | "
                    f"Traders: {row['num_traders']:>4d} | "
                    f"Avg Value: ${row['avg_value_per_trader']:>10,.2f} | "
                    f"Avg Leverage: {row['avg_leverage']:>5.1f}x | "
                    f"PnL: ${row['total_pnl']:>12,.2f}"
                )
        
        # Summary
        total_long_value = agg_df[agg_df['is_long']]['total_value'].sum()
        total_short_value = agg_df[~agg_df['is_long']]['total_value'].sum()
        total_long_pnl = agg_df[agg_df['is_long']]['total_pnl'].sum()
        total_short_pnl = agg_df[~agg_df['is_long']]['total_pnl'].sum()
        
        report_lines.append("")
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total Long Value: ${total_long_value:,.2f}")
        report_lines.append(f"  Total Short Value: ${total_short_value:,.2f}")
        report_lines.append(f"  Total Long PnL: ${total_long_pnl:,.2f}")
        report_lines.append(f"  Total Short PnL: ${total_short_pnl:,.2f}")
        report_lines.append(f"  Total Positions: {len(agg_df)}")
        
        return "\n".join(report_lines)

