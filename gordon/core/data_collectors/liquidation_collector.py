"""
Liquidation Data Collector
==========================
Day 21: Batch collection of historical liquidation data for analysis and backtesting.

Features:
- Fetches historical liquidation data from exchanges
- Processes and categorizes liquidations (L LIQ vs S LIQ)
- Exports to CSV files for backtesting
- Creates aggregated totals across symbols
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class LiquidationCollector:
    """
    Liquidation data collector utility.
    
    Collects historical liquidation data from exchanges,
    processes it, and saves to CSV files for backtesting.
    """

    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        """
        Initialize liquidation collector.
        
        Args:
            exchange_adapter: Exchange adapter instance
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or self._get_default_config()
        
        # Output directory
        self.output_dir = Path(self.config.get('output_dir', './output_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filtering parameters
        self.min_usd_size = self.config.get('min_usd_size', 3000)
        
        logger.info(f"Liquidation Collector initialized. Output dir: {self.output_dir}")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'output_dir': './output_data',
            'min_usd_size': 3000,
            'aggregation_period': '5min',
            'default_symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'WIFUSDT']
        }

    async def fetch_liquidations(self, symbol: str, since: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch liquidation data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            since: Optional datetime to fetch from (defaults to 24 hours ago)
            
        Returns:
            DataFrame with liquidation data
        """
        if not self.exchange_adapter:
            logger.error("No exchange adapter configured")
            return pd.DataFrame()
        
        if since is None:
            since = datetime.now() - timedelta(days=1)
        
        try:
            since_ms = int(since.timestamp() * 1000)
            
            # Fetch liquidations using exchange adapter
            if hasattr(self.exchange_adapter, 'get_liquidations'):
                liquidations = await self.exchange_adapter.get_liquidations(symbol)
            elif hasattr(self.exchange_adapter, 'ccxt_client') and self.exchange_adapter.ccxt_client:
                # Use CCXT client directly
                liquidations = await self.exchange_adapter.ccxt_client.fetch_liquidations(symbol, since=since_ms)
            else:
                logger.error("Exchange adapter does not support liquidation fetching")
                return pd.DataFrame()
            
            if not liquidations:
                logger.warning(f"No liquidations found for {symbol}")
                return pd.DataFrame()
            
            # Process liquidation data
            data = []
            for liq in liquidations:
                # Handle different exchange formats
                if isinstance(liq, dict):
                    if 'info' in liq:
                        # CCXT format
                        info = liq['info']
                        data.append({
                            'symbol': symbol,
                            'side': info.get('side', liq.get('side', '')),
                            'order_type': info.get('type', 'LIMIT'),
                            'time_in_force': info.get('timeInForce', 'GTC'),
                            'original_quantity': float(info.get('origQty', liq.get('amount', 0))),
                            'price': float(info.get('price', liq.get('price', 0))),
                            'average_price': float(info.get('avgPrice', info.get('price', 0))),
                            'order_status': info.get('status', 'FILLED'),
                            'order_last_filled_quantity': float(info.get('lastFilled', 0)),
                            'order_filled_accumulated_quantity': float(info.get('cumQty', info.get('amount', 0))),
                            'order_trade_time': int(info.get('updateTime', liq.get('timestamp', 0))),
                            'usd_size': float(info.get('cumQty', info.get('amount', 0))) * float(info.get('avgPrice', info.get('price', 0)))
                        })
                    else:
                        # Direct format
                        data.append({
                            'symbol': symbol,
                            'side': liq.get('side', ''),
                            'order_type': liq.get('order_type', 'LIMIT'),
                            'time_in_force': liq.get('time_in_force', 'GTC'),
                            'original_quantity': float(liq.get('original_quantity', liq.get('quantity', 0))),
                            'price': float(liq.get('price', 0)),
                            'average_price': float(liq.get('average_price', liq.get('price', 0))),
                            'order_status': liq.get('order_status', 'FILLED'),
                            'order_last_filled_quantity': float(liq.get('order_last_filled_quantity', 0)),
                            'order_filled_accumulated_quantity': float(liq.get('order_filled_accumulated_quantity', liq.get('quantity', 0))),
                            'order_trade_time': int(liq.get('order_trade_time', liq.get('timestamp', 0))),
                            'usd_size': float(liq.get('usd_size', liq.get('usd_value', 0)))
                        })
            
            df = pd.DataFrame(data)
            
            if df.empty:
                return df
            
            logger.info(f"Fetched {len(df)} liquidations for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching liquidations for {symbol}: {e}")
            return pd.DataFrame()

    def process_liquidation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and categorize liquidation data.
        
        Args:
            df: Raw liquidation DataFrame
            
        Returns:
            Processed DataFrame with LIQ_SIDE categorization
        """
        if df.empty:
            return df
        
        # Categorize liquidations
        # L LIQ: Long liquidation (SELL side, large size)
        # S LIQ: Short liquidation (BUY side, large size)
        df['LIQ_SIDE'] = np.select(
            [
                (df['symbol'].str.len() <= 7) & (df['usd_size'] > self.min_usd_size) & (df['side'] == 'SELL'),
                (df['symbol'].str.len() <= 7) & (df['usd_size'] > self.min_usd_size) & (df['side'] == 'BUY')
            ],
            ['L LIQ', 'S LIQ'],
            default=None
        )
        
        # Filter to only categorized liquidations
        df = df[df['LIQ_SIDE'].notna()].copy()
        
        # Convert timestamp to datetime
        if 'order_trade_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['order_trade_time'], unit='ms')
        elif 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df.index)
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Remove USDT suffix from symbol names
        df['symbol'] = df['symbol'].str.replace('USDT', '')
        
        return df

    def save_to_csv(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Save liquidation data to CSV file.
        
        Args:
            df: DataFrame to save
            symbol: Symbol name (without USDT suffix)
            
        Returns:
            Path to saved file
        """
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return ""
        
        output_path = self.output_dir / f'{symbol}_liq_data.csv'
        
        # Ensure required columns exist
        required_cols = ['symbol', 'LIQ_SIDE', 'price', 'usd_size']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"DataFrame missing required columns. Found: {df.columns.tolist()}")
            return ""
        
        # Select and save columns
        df_to_save = df.reset_index()
        df_to_save.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df_to_save)} liquidations for {symbol} to {output_path}")
        return str(output_path)

    def create_aggregated_totals(self, all_liqs: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated totals across all symbols.
        
        Args:
            all_liqs: Combined DataFrame with all liquidations
            
        Returns:
            Aggregated DataFrame
        """
        if all_liqs.empty:
            return pd.DataFrame()
        
        aggregation_period = self.config.get('aggregation_period', '5min')
        
        # Separate L LIQ and S LIQ
        df_totals_L = all_liqs[all_liqs['LIQ_SIDE'] == 'L LIQ'].resample(aggregation_period).agg({
            'usd_size': 'sum',
            'price': 'mean'
        })
        df_totals_S = all_liqs[all_liqs['LIQ_SIDE'] == 'S LIQ'].resample(aggregation_period).agg({
            'usd_size': 'sum',
            'price': 'mean'
        })
        
        # Add LIQ_SIDE column
        df_totals_L['LIQ_SIDE'] = 'L LIQ'
        df_totals_S['LIQ_SIDE'] = 'S LIQ'
        
        # Combine
        df_totals = pd.concat([df_totals_L, df_totals_S])
        df_totals['symbol'] = 'All'
        
        # Reset index and reorder columns
        df_totals.reset_index(inplace=True)
        df_totals = df_totals[['datetime', 'symbol', 'LIQ_SIDE', 'price', 'usd_size']]
        
        return df_totals

    async def collect(self, symbols: List[str], since: Optional[datetime] = None) -> Dict[str, str]:
        """
        Collect liquidation data for multiple symbols.
        
        Args:
            symbols: List of symbols to collect
            since: Optional datetime to fetch from
            
        Returns:
            Dictionary mapping symbol to CSV file path
        """
        if since is None:
            since = datetime.now() - timedelta(days=1)
        
        all_liqs = pd.DataFrame()
        saved_files = {}
        
        logger.info(f"Collecting liquidation data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                # Fetch liquidations
                df = await self.fetch_liquidations(symbol, since)
                
                if df.empty:
                    logger.warning(f"No liquidations found for {symbol}")
                    continue
                
                # Process data
                df_processed = self.process_liquidation_data(df)
                
                if df_processed.empty:
                    logger.warning(f"No valid liquidations after processing for {symbol}")
                    continue
                
                # Save per symbol
                symbol_short = symbol.replace('USDT', '')
                saved_path = self.save_to_csv(df_processed, symbol_short)
                
                if saved_path:
                    saved_files[symbol] = saved_path
                
                # Add to combined DataFrame
                all_liqs = pd.concat([all_liqs, df_processed])
                
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                continue
        
        # Create and save aggregated totals
        if not all_liqs.empty:
            df_totals = self.create_aggregated_totals(all_liqs)
            if not df_totals.empty:
                totals_path = self.output_dir / 'total_liq_data.csv'
                df_totals.to_csv(totals_path, index=False)
                logger.info(f"Saved aggregated totals to {totals_path}")
                logger.info(f"Aggregated Totals Head:\n{df_totals.head()}")
        
        logger.info(f"Collection complete. Saved {len(saved_files)} symbol files")
        return saved_files


async def main():
    """Example usage."""
    import asyncio
    
    # This would be initialized with actual exchange adapter
    # collector = LiquidationCollector(exchange_adapter=exchange)
    # symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'WIFUSDT']
    # files = await collector.collect(symbols)
    # print(f"Saved files: {files}")
    pass


if __name__ == "__main__":
    asyncio.run(main())

