"""
Whale Position Tracker
=======================
Day 44: Track and analyze large positions (whales) on exchanges.
Monitors significant holders and their position movements.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class WhalePositionTracker:
    """
    Track large positions (whales) on exchanges.
    
    Monitors significant holders and their position movements
    for social signal trading and copy trading opportunities.
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize whale position tracker.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Whale thresholds
        self.whale_threshold_usd = self.config.get('whale_threshold_usd', 100000)
        self.institutional_threshold_usd = self.config.get('institutional_threshold_usd', 250000)
        self.min_position_value = self.config.get('min_position_value', 25000)
        
        # Tracking settings
        self.top_n_positions = self.config.get('top_n_positions', 25)
        self.update_interval_minutes = self.config.get('update_interval_minutes', 30)
        
        # Output directory
        self.output_dir = Path(self.config.get('output_dir', './whale_tracking_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Position cache
        self.whale_positions_cache: Dict[str, Dict] = {}
        self.last_update: Optional[datetime] = None
    
    async def get_large_positions(
        self,
        symbol: Optional[str] = None,
        min_value_usd: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get large positions from exchange.
        
        Args:
            symbol: Optional symbol to filter by
            min_value_usd: Minimum position value threshold
            
        Returns:
            DataFrame with large positions
        """
        if min_value_usd is None:
            min_value_usd = self.min_position_value
        
        if not self.exchange_adapter:
            logger.error("Exchange adapter not available")
            return pd.DataFrame()
        
        logger.info(f"Fetching large positions (min: ${min_value_usd:,.2f})")
        
        try:
            # Get all positions from exchange
            if hasattr(self.exchange_adapter, 'get_all_positions'):
                positions = await self.exchange_adapter.get_all_positions()
            elif hasattr(self.exchange_adapter, 'fetch_positions'):
                positions = await self.exchange_adapter.fetch_positions()
            else:
                logger.warning("Exchange adapter doesn't support position fetching")
                return pd.DataFrame()
            
            # Filter and process positions
            large_positions = []
            for pos in positions:
                if not pos or pos.get('contracts', 0) == 0:
                    continue
                
                symbol_pos = pos.get('symbol', '')
                if symbol and symbol_pos != symbol:
                    continue
                
                # Calculate position value
                size = abs(float(pos.get('contracts', 0)))
                entry_price = float(pos.get('entryPrice', 0))
                current_price = float(pos.get('markPrice', 0) or pos.get('mark_price', 0) or entry_price)
                
                if current_price == 0:
                    continue
                
                position_value = size * current_price
                
                if position_value < min_value_usd:
                    continue
                
                # Calculate PnL
                unrealized_pnl = float(pos.get('unrealizedPnl', 0) or pos.get('unrealized_pnl', 0))
                pnl_percent = (unrealized_pnl / (size * entry_price) * 100) if entry_price > 0 else 0
                
                # Classify whale tier
                whale_tier = self._classify_whale_tier(position_value)
                
                large_positions.append({
                    'symbol': symbol_pos,
                    'size': size,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'position_value_usd': position_value,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percent': pnl_percent,
                    'leverage': pos.get('leverage', 1),
                    'side': pos.get('side', 'unknown'),
                    'whale_tier': whale_tier,
                    'timestamp': datetime.now()
                })
            
            if not large_positions:
                logger.info("No large positions found")
                return pd.DataFrame()
            
            df = pd.DataFrame(large_positions)
            df = df.sort_values('position_value_usd', ascending=False)
            
            logger.info(f"Found {len(df)} large positions")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching large positions: {e}")
            return pd.DataFrame()
    
    def _classify_whale_tier(self, position_value: float) -> str:
        """
        Classify whale tier based on position value.
        
        Args:
            position_value: Position value in USD
            
        Returns:
            Whale tier string
        """
        if position_value >= self.institutional_threshold_usd:
            return "Institutional"
        elif position_value >= self.whale_threshold_usd:
            return "Whale"
        elif position_value >= self.min_position_value:
            return "Large"
        else:
            return "Standard"
    
    def get_whale_positions(
        self,
        positions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter for whale-tier positions only.
        
        Args:
            positions_df: DataFrame with positions
            
        Returns:
            DataFrame with whale positions only
        """
        if positions_df.empty:
            return pd.DataFrame()
        
        whale_df = positions_df[
            positions_df['whale_tier'].isin(['Whale', 'Institutional'])
        ]
        
        return whale_df.sort_values('position_value_usd', ascending=False)
    
    def get_top_positions(
        self,
        positions_df: pd.DataFrame,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get top N positions by value.
        
        Args:
            positions_df: DataFrame with positions
            top_n: Number of top positions to return
            
        Returns:
            DataFrame with top positions
        """
        if positions_df.empty:
            return pd.DataFrame()
        
        if top_n is None:
            top_n = self.top_n_positions
        
        return positions_df.head(top_n)
    
    def analyze_whale_movements(
        self,
        current_positions: pd.DataFrame,
        previous_positions: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze whale position movements and changes.
        
        Args:
            current_positions: Current positions DataFrame
            previous_positions: Previous positions DataFrame (optional)
            
        Returns:
            Dictionary with movement analysis
        """
        if current_positions.empty:
            return {
                'new_positions': pd.DataFrame(),
                'closed_positions': pd.DataFrame(),
                'increased_positions': pd.DataFrame(),
                'decreased_positions': pd.DataFrame()
            }
        
        if previous_positions is None or previous_positions.empty:
            return {
                'new_positions': current_positions,
                'closed_positions': pd.DataFrame(),
                'increased_positions': pd.DataFrame(),
                'decreased_positions': pd.DataFrame()
            }
        
        # Find new positions
        current_symbols = set(current_positions['symbol'].unique())
        previous_symbols = set(previous_positions['symbol'].unique())
        new_symbols = current_symbols - previous_symbols
        new_positions = current_positions[current_positions['symbol'].isin(new_symbols)]
        
        # Find closed positions
        closed_symbols = previous_symbols - current_symbols
        closed_positions = previous_positions[previous_positions['symbol'].isin(closed_symbols)]
        
        # Find changed positions
        common_symbols = current_symbols & previous_symbols
        
        increased = []
        decreased = []
        
        for sym in common_symbols:
            curr_pos = current_positions[current_positions['symbol'] == sym].iloc[0]
            prev_pos = previous_positions[previous_positions['symbol'] == sym].iloc[0]
            
            curr_size = curr_pos['position_value_usd']
            prev_size = prev_pos['position_value_usd']
            
            if curr_size > prev_size * 1.1:  # 10% increase threshold
                increased.append(curr_pos)
            elif curr_size < prev_size * 0.9:  # 10% decrease threshold
                decreased.append(curr_pos)
        
        increased_df = pd.DataFrame(increased) if increased else pd.DataFrame()
        decreased_df = pd.DataFrame(decreased) if decreased else pd.DataFrame()
        
        return {
            'new_positions': new_positions,
            'closed_positions': closed_positions,
            'increased_positions': increased_df,
            'decreased_positions': decreased_df
        }
    
    def save_positions(self, positions_df: pd.DataFrame, filename: str):
        """Save positions to CSV."""
        try:
            filepath = self.output_dir / filename
            positions_df.to_csv(filepath, index=False)
            logger.info(f"Saved positions to {filepath}")
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def get_whale_summary(self, positions_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for whale positions.
        
        Args:
            positions_df: DataFrame with positions
            
        Returns:
            Dictionary with summary statistics
        """
        if positions_df.empty:
            return {}
        
        return {
            'total_positions': len(positions_df),
            'total_value_usd': positions_df['position_value_usd'].sum(),
            'total_unrealized_pnl': positions_df['unrealized_pnl'].sum(),
            'avg_position_value': positions_df['position_value_usd'].mean(),
            'max_position_value': positions_df['position_value_usd'].max(),
            'whale_count': len(positions_df[positions_df['whale_tier'] == 'Whale']),
            'institutional_count': len(positions_df[positions_df['whale_tier'] == 'Institutional']),
            'top_symbols': positions_df.groupby('symbol')['position_value_usd'].sum().sort_values(ascending=False).head(10).to_dict()
        }

