"""
ETH SMA-Based Strategy
======================
Day 26: Entry/exit strategy based on ETH price vs SMAs.

Strategy Logic:
- Entry: When ETH price > SMA20, enter positions in token batch
- Exit: When ETH price <= SMA41, close all positions
- Used for trend-following approach
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class EthSMAStrategy:
    """
    ETH SMA-based strategy for portfolio entry/exit.
    
    Uses ETH price relative to SMAs to determine when to enter/exit positions.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ETH SMA strategy.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sma_short = self.config.get('sma_short', 20)
        self.sma_long = self.config.get('sma_long', 41)
        self.timeframe = self.config.get('timeframe', '1d')
        self.limit = self.config.get('limit', 200)
        self.min_position_pct = self.config.get('min_position_pct', 0.1)  # 10% of target

    async def fetch_eth_data(
        self,
        exchange_adapter,
        symbol: str = 'ETHUSDT'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch ETH candle data with SMAs.
        
        Args:
            exchange_adapter: Exchange adapter instance
            symbol: ETH symbol
            
        Returns:
            DataFrame with OHLCV and SMAs
        """
        try:
            # Fetch OHLCV data
            ohlcv = await exchange_adapter.fetch_ohlcv(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=self.limit
            )
            
            if not ohlcv or len(ohlcv) == 0:
                logger.error(f"Failed to fetch ETH data for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Calculate SMAs
            df[f'SMA_{self.sma_short}'] = df['close'].rolling(window=self.sma_short).mean()
            df[f'SMA_{self.sma_long}'] = df['close'].rolling(window=self.sma_long).mean()
            
            # Calculate price vs SMA flags
            df[f'Price > SMA_{self.sma_short}'] = df['close'] > df[f'SMA_{self.sma_short}']
            df[f'Price > SMA_{self.sma_long}'] = df['close'] > df[f'SMA_{self.sma_long}']
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching ETH data: {e}")
            return None

    def analyze_signals(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze ETH SMA signals.
        
        Args:
            df: DataFrame with ETH data and SMAs
            
        Returns:
            Dictionary with signals and prices
        """
        if df is None or df.empty:
            return {
                'entry_signal': False,
                'exit_signal': False,
                'eth_price': None,
                'sma_short': None,
                'sma_long': None
            }
        
        # Get latest data
        latest = df.iloc[-1]
        
        eth_price = float(latest['close'])
        sma_short_val = float(latest[f'SMA_{self.sma_short}'])
        sma_long_val = float(latest[f'SMA_{self.sma_long}'])
        
        price_gt_sma_short = bool(latest[f'Price > SMA_{self.sma_short}'])
        price_gt_sma_long = bool(latest[f'Price > SMA_{self.sma_long}'])
        
        # Entry signal: Price > SMA20
        entry_signal = price_gt_sma_short
        
        # Exit signal: Price <= SMA41
        exit_signal = not price_gt_sma_long
        
        return {
            'entry_signal': entry_signal,
            'exit_signal': exit_signal,
            'eth_price': eth_price,
            'sma_short': sma_short_val,
            'sma_long': sma_long_val,
            'price_gt_sma_short': price_gt_sma_short,
            'price_gt_sma_long': price_gt_sma_long
        }

    def get_entry_symbols(
        self,
        positions: List[Dict],
        target_size: float,
        excluded_symbols: List[str]
    ) -> List[str]:
        """
        Get symbols that should enter positions.
        
        Args:
            positions: Current positions list
            target_size: Target position size in USD
            excluded_symbols: List of symbols to exclude
            
        Returns:
            List of symbols to enter
        """
        entry_symbols = []
        
        # Create position map
        position_map = {pos['symbol']: pos.get('usd_value', 0) for pos in positions}
        
        # Check each symbol in batch
        token_batch = self.config.get('token_batch', [])
        
        for symbol in token_batch:
            if symbol in excluded_symbols:
                continue
            
            current_value = position_map.get(symbol, 0)
            min_entry_value = target_size * self.min_position_pct
            
            # Enter if position is less than minimum threshold
            if current_value < min_entry_value:
                entry_symbols.append(symbol)
        
        return entry_symbols

    def get_exit_symbols(
        self,
        positions: List[Dict],
        excluded_symbols: List[str],
        min_value: float = 2.0
    ) -> List[str]:
        """
        Get symbols that should exit positions.
        
        Args:
            positions: Current positions list
            excluded_symbols: List of symbols to exclude
            min_value: Minimum position value to consider
            
        Returns:
            List of symbols to exit
        """
        exit_symbols = []
        
        for pos in positions:
            symbol = pos.get('symbol', '')
            usd_value = pos.get('usd_value', 0) or 0.0
            
            # Exit if not excluded and value above minimum
            if symbol not in excluded_symbols and usd_value > min_value:
                exit_symbols.append(symbol)
        
        return exit_symbols

