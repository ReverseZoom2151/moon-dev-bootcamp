"""
Historical Data Collection Manager
===================================
Day 31-32: Unified manager for downloading and updating historical data.

Provides high-level interface for batch data collection and updates.
"""

import logging
import asyncio
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta

from .provider_factory import DataProviderFactory
from .yahoo_provider import YahooFinanceProvider

logger = logging.getLogger(__name__)


class HistoricalDataManager:
    """
    Manager for downloading and updating historical data.
    
    Supports batch operations for multiple symbols and exchanges.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize historical data manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def download_binance_data(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        weeks: int = 52,
        api_key: str = "",
        api_secret: str = ""
    ) -> Dict[str, int]:
        """
        Download historical data from Binance for multiple symbols.
        
        Args:
            symbols: List of symbols to download
            timeframe: Timeframe
            weeks: Number of weeks of history
            api_key: Binance API key
            api_secret: Binance API secret
            
        Returns:
            Dictionary mapping symbols to row counts
        """
        provider = DataProviderFactory.create_provider(
            'binance',
            {
                'api_key': api_key,
                'api_secret': api_secret,
                'cache_dir': self.config.get('binance_cache_dir', './binance_data')
            }
        )
        
        results = {}
        end = datetime.now()
        start = end - timedelta(weeks=weeks)
        
        for symbol in symbols:
            try:
                df = provider.fetch(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end
                )
                results[symbol] = len(df)
                logger.info(f"Downloaded {len(df)} rows for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                results[symbol] = -1
        
        return results
    
    def download_bitfinex_data(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        weeks: int = 52,
        api_key: str = "",
        api_secret: str = ""
    ) -> Dict[str, int]:
        """
        Download historical data from Bitfinex for multiple symbols.
        
        Args:
            symbols: List of symbols to download
            timeframe: Timeframe
            weeks: Number of weeks of history
            api_key: Bitfinex API key
            api_secret: Bitfinex API secret
            
        Returns:
            Dictionary mapping symbols to row counts
        """
        provider = DataProviderFactory.create_provider(
            'bitfinex',
            {
                'api_key': api_key,
                'api_secret': api_secret,
                'cache_dir': self.config.get('bitfinex_cache_dir', './bitfinex_data')
            }
        )
        
        results = {}
        end = datetime.now()
        start = end - timedelta(weeks=weeks)
        
        for symbol in symbols:
            try:
                df = provider.fetch(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end
                )
                results[symbol] = len(df)
                logger.info(f"Downloaded {len(df)} rows for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                results[symbol] = -1
        
        return results
    
    def download_yahoo_data(
        self,
        symbols: List[Tuple[str, str]],
        timeframe: str = '1d',
        update_existing: bool = True
    ) -> Dict[str, int]:
        """
        Download historical data from Yahoo Finance.
        
        Args:
            symbols: List of (symbol, asset_type) tuples
            timeframe: Timeframe ('1h' or '1d')
            update_existing: Whether to update existing cached data
            
        Returns:
            Dictionary mapping symbols to row counts
        """
        provider = DataProviderFactory.create_provider(
            'yahoo',
            {
                'cache_dir': self.config.get('yahoo_cache_dir', './yahoo_data'),
                'hourly_fetch_days': self.config.get('hourly_fetch_days', 728),
                'default_start_date': self.config.get('default_start_date', '2000-01-01')
            }
        )
        
        results = {}
        
        for symbol, asset_type in symbols:
            try:
                if update_existing:
                    # Use update_all_symbols if available
                    if hasattr(provider, 'update_all_symbols'):
                        update_results = provider.update_all_symbols([(symbol, asset_type)])
                        results[symbol] = update_results.get(symbol, 0)
                    else:
                        # Fallback: fetch recent data
                        df = provider.fetch(symbol=symbol, timeframe=timeframe)
                        results[symbol] = len(df)
                else:
                    df = provider.fetch(symbol=symbol, timeframe=timeframe)
                    results[symbol] = len(df)
                
                logger.info(f"Downloaded/updated {results[symbol]} rows for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                results[symbol] = -1
        
        return results
    
    def update_all_data(
        self,
        binance_symbols: Optional[List[str]] = None,
        bitfinex_symbols: Optional[List[str]] = None,
        yahoo_symbols: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Update all data sources.
        
        Args:
            binance_symbols: Binance symbols to update
            bitfinex_symbols: Bitfinex symbols to update
            yahoo_symbols: Yahoo Finance symbols to update
            
        Returns:
            Dictionary with results for each exchange
        """
        results = {}
        
        if binance_symbols:
            logger.info("Updating Binance data...")
            results['binance'] = self.download_binance_data(binance_symbols)
        
        if bitfinex_symbols:
            logger.info("Updating Bitfinex data...")
            results['bitfinex'] = self.download_bitfinex_data(bitfinex_symbols)
        
        if yahoo_symbols:
            logger.info("Updating Yahoo Finance data...")
            results['yahoo'] = self.download_yahoo_data(yahoo_symbols, update_existing=True)
        
        return results

