"""
Data Provider Factory
=====================
Day 31-32: Factory for creating data providers.

Provides unified interface to create exchange-specific and Yahoo Finance data providers.
"""

import logging
from typing import Optional, Dict
from pathlib import Path

from .provider import DataProvider
from .binance_provider import BinanceDataProvider
from .bitfinex_provider import BitfinexDataProvider
from .yahoo_provider import YahooFinanceProvider

logger = logging.getLogger(__name__)


class DataProviderFactory:
    """Factory for creating data providers."""
    
    @staticmethod
    def create_provider(
        provider_type: str = "mock",
        config: Optional[Dict] = None
    ) -> DataProvider:
        """
        Create a data provider instance.
        
        Args:
            provider_type: Type of provider ('binance', 'bitfinex', 'yahoo', 'mock')
            config: Configuration dictionary
            
        Returns:
            DataProvider instance
        """
        config = config or {}
        
        if provider_type.lower() == "binance":
            return BinanceDataProvider(
                api_key=config.get('api_key', ''),
                api_secret=config.get('api_secret', ''),
                cache_dir=config.get('cache_dir', './binance_data'),
                fetch_limit=config.get('fetch_limit', 1000),
                rate_limit_delay=config.get('rate_limit_delay', 0.2),
                retry_attempts=config.get('retry_attempts', 3)
            )
        
        elif provider_type.lower() == "bitfinex":
            return BitfinexDataProvider(
                api_key=config.get('api_key', ''),
                api_secret=config.get('api_secret', ''),
                cache_dir=config.get('cache_dir', './bitfinex_data'),
                fetch_limit=config.get('fetch_limit', 5000),
                rate_limit_delay=config.get('rate_limit_delay', 0.5),
                retry_attempts=config.get('retry_attempts', 3)
            )
        
        elif provider_type.lower() == "yahoo":
            return YahooFinanceProvider(
                cache_dir=config.get('cache_dir', './yahoo_data'),
                hourly_fetch_days=config.get('hourly_fetch_days', 728),
                default_start_date=config.get('default_start_date', '2000-01-01'),
                rate_limit_delay=config.get('rate_limit_delay', 1.0)
            )
        
        elif provider_type.lower() == "mock":
            # Lazy import to avoid circular dependency
            from .fetcher import MockDataProvider
            return MockDataProvider()
        
        else:
            logger.warning(f"Unknown provider type: {provider_type}. Using mock provider.")
            # Lazy import to avoid circular dependency
            from .fetcher import MockDataProvider
            return MockDataProvider()
    
    @staticmethod
    def create_provider_from_config(config: Dict) -> DataProvider:
        """
        Create provider from configuration dictionary.
        
        Args:
            config: Configuration with 'type' and provider-specific settings
            
        Returns:
            DataProvider instance
        """
        provider_type = config.get('type', 'mock')
        provider_config = config.get('config', {})
        
        return DataProviderFactory.create_provider(provider_type, provider_config)
    
    @staticmethod
    def detect_provider_from_symbol(symbol: str) -> str:
        """
        Detect appropriate provider based on symbol format.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Provider type name
        """
        symbol_upper = symbol.upper()
        
        # Yahoo Finance symbols
        if '=X' in symbol or '=F' in symbol:  # Forex or futures
            return 'yahoo'
        elif symbol.isalpha() and len(symbol) <= 5:  # Stock ticker
            return 'yahoo'
        
        # Crypto exchanges
        elif 'USDT' in symbol or 'USD' in symbol:
            # Could be Binance or Bitfinex
            # Default to Binance for crypto
            return 'binance'
        
        # Default to mock
        return 'mock'

