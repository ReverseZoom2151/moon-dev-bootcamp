"""
Whale Tracking Manager
======================
Day 44: Coordinates whale position tracking across exchanges.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .whale_tracker import WhalePositionTracker
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

