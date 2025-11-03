"""
Multi-Address Whale Position Tracker
====================================
Day 46: Enhanced whale tracking with multi-address support, liquidation risk analysis,
and position aggregation.

Features:
- Load whale addresses from file
- Parallel position fetching
- Position aggregation by coin and direction
- Liquidation risk analysis (distance to liquidation)
- CSV export/reporting
"""

import pandas as pd
import numpy as np
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class MultiAddressWhaleTracker:
    """
    Multi-address whale position tracker.
    
    Tracks positions for multiple whale addresses simultaneously,
    aggregates positions by coin and direction, and analyzes
    liquidation risks.
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize multi-address whale tracker.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Configuration
        self.min_position_value = self.config.get('min_position_value', 25000)
        self.top_n_positions = self.config.get('top_n_positions', 30)
        self.whale_threshold_usd = self.config.get('whale_threshold_usd', 100000)
        self.liquidation_risk_threshold = self.config.get('liquidation_risk_threshold', 3.0)  # 3% default
        
        # Parallel processing
        self.max_workers = self.config.get('max_workers', 10)
        self.api_request_delay = self.config.get('api_request_delay', 0.1)
        
        # Data directory
        self.output_dir = Path(self.config.get('output_dir', './whale_tracking_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Whale addresses file
        self.whale_addresses_file = Path(self.config.get('whale_addresses_file', 'whale_addresses.txt'))
        if not self.whale_addresses_file.is_absolute():
            self.whale_addresses_file = self.output_dir / self.whale_addresses_file
        
        # Cache
        self.whale_addresses: List[str] = []
        self.position_cache: Dict[str, pd.DataFrame] = {}
    
    def load_whale_addresses(self, addresses_file: Optional[Path] = None) -> List[str]:
        """
        Load whale addresses from file.
        
        Args:
            addresses_file: Optional path to addresses file. If None, uses configured file.
            
        Returns:
            List of whale addresses
        """
        file_path = addresses_file or self.whale_addresses_file
        
        try:
            if not file_path.exists():
                logger.warning(f"Whale addresses file not found: {file_path}")
                logger.info(f"Creating example file at {file_path}")
                # Create example file
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write("# Whale Addresses File\n")
                    f.write("# Add one address per line\n")
                    f.write("# Lines starting with # are ignored\n")
                    f.write("# Example addresses:\n")
                    f.write("# 0x1234567890abcdef1234567890abcdef12345678\n")
                return []
            
            with open(file_path, 'r') as f:
                addresses = [
                    line.strip() 
                    for line in f 
                    if line.strip() and not line.startswith('#')
                ]
            
            logger.info(f"Loaded {len(addresses)} whale addresses from {file_path}")
            self.whale_addresses = addresses
            return addresses
            
        except Exception as e:
            logger.error(f"Error loading whale addresses from {file_path}: {e}")
            return []
    
    async def fetch_positions_for_address(
        self,
        address: str,
        symbol: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch positions for a single address.
        
        Args:
            address: Wallet address or trader ID
            symbol: Optional symbol to filter by
            
        Returns:
            List of position dictionaries
        """
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return []
        
        try:
            # Add delay to respect rate limits
            await asyncio.sleep(self.api_request_delay)
            
            # For Binance/Bitfinex, we'd use trader_id instead of address
            # This is a placeholder - actual implementation depends on exchange
            if hasattr(self.exchange_adapter, 'get_positions_by_address'):
                positions = await self.exchange_adapter.get_positions_by_address(address)
            elif hasattr(self.exchange_adapter, 'get_positions_by_trader'):
                positions = await self.exchange_adapter.get_positions_by_trader(address)
            else:
                # Fallback: use Moon Dev API if available
                try:
                    from gordon.research.data_providers.moondev_api import MoonDevAPI
                    api = MoonDevAPI()
                    positions_df = api.get_positions_hlp()
                    if positions_df is not None and not positions_df.empty:
                        # Filter by address
                        addr_positions = positions_df[positions_df['address'] == address]
                        if not addr_positions.empty:
                            positions = addr_positions.to_dict('records')
                        else:
                            positions = []
                    else:
                        positions = []
                except Exception:
                    logger.warning(f"Could not fetch positions for address {address[:10]}...")
                    return []
            
            # Process and filter positions
            processed = []
            for pos in positions:
                if not pos:
                    continue
                
                # Get position data
                coin = pos.get('coin') or pos.get('symbol', '')
                if symbol and coin != symbol:
                    continue
                
                position_value = float(pos.get('position_value', 0) or pos.get('positionValue', 0))
                if position_value < self.min_position_value:
                    continue
                
                entry_price = float(pos.get('entry_price', 0) or pos.get('entryPx', 0))
                liquidation_price = float(pos.get('liquidation_price', 0) or pos.get('liquidationPx', 0))
                leverage = float(pos.get('leverage', 1) or pos.get('leverage', {}).get('value', 1))
                unrealized_pnl = float(pos.get('unrealized_pnl', 0) or pos.get('unrealizedPnl', 0))
                
                # Determine if long or short
                size = float(pos.get('size', 0) or pos.get('sz', 0))
                is_long = size > 0 or pos.get('is_long', False) or pos.get('side', '').upper() == 'LONG'
                
                processed.append({
                    'address': address,
                    'coin': coin,
                    'entry_price': entry_price,
                    'liquidation_price': liquidation_price,
                    'leverage': leverage,
                    'position_value': position_value,
                    'unrealized_pnl': unrealized_pnl,
                    'is_long': is_long,
                    'size': abs(size),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return processed
            
        except Exception as e:
            logger.error(f"Error fetching positions for address {address[:10]}...: {e}")
            return []
    
    async def fetch_all_positions_parallel(
        self,
        addresses: Optional[List[str]] = None,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch positions for all addresses in parallel.
        
        Args:
            addresses: List of addresses to fetch. If None, uses loaded addresses.
            symbol: Optional symbol to filter by
            
        Returns:
            DataFrame with all positions
        """
        if addresses is None:
            addresses = self.whale_addresses or self.load_whale_addresses()
        
        if not addresses:
            logger.warning("No whale addresses available")
            return pd.DataFrame()
        
        logger.info(f"Fetching positions for {len(addresses)} addresses (max workers: {self.max_workers})")
        
        all_positions = []
        
        # Process in batches to avoid overwhelming the API
        batch_size = self.max_workers * 2
        
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            
            tasks = [
                self.fetch_positions_for_address(addr, symbol)
                for addr in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in parallel fetch: {result}")
                    continue
                if result:
                    all_positions.extend(result)
            
            # Small delay between batches
            if i + batch_size < len(addresses):
                await asyncio.sleep(0.5)
        
        if not all_positions:
            logger.info("No positions found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_positions)
        logger.info(f"Fetched {len(df)} positions from {len(addresses)} addresses")
        
        return df
    
    def aggregate_positions(
        self,
        positions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate positions by coin and direction.
        
        Args:
            positions_df: DataFrame with positions
            
        Returns:
            Aggregated DataFrame
        """
        if positions_df.empty:
            return pd.DataFrame()
        
        # Group by coin and direction
        agg_df = positions_df.groupby(['coin', 'is_long']).agg({
            'position_value': 'sum',
            'unrealized_pnl': 'sum',
            'address': 'count',
            'leverage': 'mean',
            'liquidation_price': lambda x: np.nanmean(x) if not x.isnull().all() else np.nan
        }).reset_index()
        
        # Rename columns
        agg_df = agg_df.rename(columns={
            'position_value': 'total_value',
            'unrealized_pnl': 'total_pnl',
            'address': 'num_traders',
            'leverage': 'avg_leverage',
            'liquidation_price': 'avg_liquidation_price'
        })
        
        # Add direction column
        agg_df['direction'] = agg_df['is_long'].apply(lambda x: 'LONG' if x else 'SHORT')
        
        # Calculate average value per trader
        agg_df['avg_value_per_trader'] = agg_df['total_value'] / agg_df['num_traders']
        
        # Sort by total value
        agg_df = agg_df.sort_values('total_value', ascending=False)
        
        return agg_df
    
    def calculate_liquidation_risk(
        self,
        positions_df: pd.DataFrame,
        current_prices: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate liquidation risk (distance to liquidation) for positions.
        
        Args:
            positions_df: DataFrame with positions
            current_prices: Optional dict of current prices by coin
            
        Returns:
            DataFrame with liquidation risk metrics
        """
        if positions_df.empty:
            return pd.DataFrame()
        
        # Filter positions with liquidation prices
        risk_df = positions_df[positions_df['liquidation_price'] > 0].copy()
        if risk_df.empty:
            return pd.DataFrame()
        
        # Get current prices if not provided
        if current_prices is None:
            current_prices = {}
            unique_coins = risk_df['coin'].unique()
            
            # Try to get prices from exchange adapter
            if self.exchange_adapter:
                for coin in unique_coins:
                    try:
                        # This would need to be implemented per exchange
                        # For now, we'll mark price as needed
                        current_prices[coin] = 0.0
                    except Exception:
                        current_prices[coin] = 0.0
        
        # Calculate current price for each position
        risk_df['current_price'] = risk_df['coin'].map(current_prices).fillna(0)
        
        # Filter out positions without current prices
        risk_df = risk_df[risk_df['current_price'] > 0].copy()
        if risk_df.empty:
            return pd.DataFrame()
        
        # Correct position direction based on liquidation price
        risk_df['is_long_corrected'] = risk_df['liquidation_price'] < risk_df['entry_price']
        risk_df['is_long'] = risk_df['is_long_corrected']
        
        # Calculate distance to liquidation (%)
        risk_df['distance_to_liq_pct'] = np.where(
            risk_df['is_long'],
            abs((risk_df['current_price'] - risk_df['liquidation_price']) / risk_df['current_price'] * 100),
            abs((risk_df['liquidation_price'] - risk_df['current_price']) / risk_df['current_price'] * 100)
        )
        
        # Sort by distance to liquidation (closest first)
        risk_df = risk_df.sort_values('distance_to_liq_pct')
        
        return risk_df
    
    def get_positions_closest_to_liquidation(
        self,
        positions_df: pd.DataFrame,
        threshold_pct: Optional[float] = None,
        top_n: Optional[int] = None,
        current_prices: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get positions closest to liquidation.
        
        Args:
            positions_df: DataFrame with positions
            threshold_pct: Maximum distance to liquidation (default: config value)
            top_n: Number of top positions to return (default: config value)
            current_prices: Optional dict of current prices
            
        Returns:
            Tuple of (long_positions_df, short_positions_df)
        """
        if threshold_pct is None:
            threshold_pct = self.liquidation_risk_threshold
        
        if top_n is None:
            top_n = self.top_n_positions
        
        # Calculate liquidation risk
        risk_df = self.calculate_liquidation_risk(positions_df, current_prices)
        if risk_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter by threshold
        risk_df = risk_df[risk_df['distance_to_liq_pct'] <= threshold_pct].copy()
        
        # Split by direction
        longs = risk_df[risk_df['is_long']].head(top_n)
        shorts = risk_df[~risk_df['is_long']].head(top_n)
        
        return longs, shorts
    
    def save_positions_to_csv(
        self,
        positions_df: pd.DataFrame,
        filename: str = 'whale_positions.csv'
    ) -> Path:
        """
        Save positions to CSV file.
        
        Args:
            positions_df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if positions_df.empty:
            logger.warning("No positions to save")
            return None
        
        file_path = self.output_dir / filename
        positions_df.to_csv(file_path, index=False, float_format='%.2f')
        logger.info(f"Saved {len(positions_df)} positions to {file_path}")
        
        return file_path
    
    def save_aggregated_positions(
        self,
        positions_df: pd.DataFrame,
        filename: str = 'aggregated_positions.csv'
    ) -> Path:
        """
        Save aggregated positions to CSV.
        
        Args:
            positions_df: DataFrame with positions
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        agg_df = self.aggregate_positions(positions_df)
        
        if agg_df.empty:
            logger.warning("No aggregated positions to save")
            return None
        
        file_path = self.output_dir / filename
        agg_df.to_csv(file_path, index=False, float_format='%.2f')
        logger.info(f"Saved aggregated positions to {file_path}")
        
        return file_path
    
    def save_liquidation_risk_positions(
        self,
        positions_df: pd.DataFrame,
        current_prices: Optional[Dict[str, float]] = None,
        threshold_pct: Optional[float] = None
    ) -> Dict[str, Path]:
        """
        Save liquidation risk positions to CSV files.
        
        Args:
            positions_df: DataFrame with positions
            current_prices: Optional dict of current prices
            threshold_pct: Maximum distance to liquidation
            
        Returns:
            Dictionary with file paths for long, short, and combined CSVs
        """
        longs, shorts = self.get_positions_closest_to_liquidation(
            positions_df,
            threshold_pct=threshold_pct,
            current_prices=current_prices
        )
        
        saved_files = {}
        
        # Save long positions
        if not longs.empty:
            longs_file = self.output_dir / 'liquidation_risk_long.csv'
            longs.to_csv(longs_file, index=False, float_format='%.2f')
            saved_files['long'] = longs_file
            logger.info(f"Saved {len(longs)} long liquidation risk positions to {longs_file}")
        
        # Save short positions
        if not shorts.empty:
            shorts_file = self.output_dir / 'liquidation_risk_short.csv'
            shorts.to_csv(shorts_file, index=False, float_format='%.2f')
            saved_files['short'] = shorts_file
            logger.info(f"Saved {len(shorts)} short liquidation risk positions to {shorts_file}")
        
        # Save combined
        if not longs.empty or not shorts.empty:
            combined = pd.concat([longs, shorts], ignore_index=True)
            if not combined.empty:
                combined = combined.sort_values('distance_to_liq_pct')
                combined_file = self.output_dir / 'liquidation_risk_combined.csv'
                combined.to_csv(combined_file, index=False, float_format='%.2f')
                saved_files['combined'] = combined_file
                logger.info(f"Saved {len(combined)} combined liquidation risk positions to {combined_file}")
        
        return saved_files

