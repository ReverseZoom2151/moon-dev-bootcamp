"""
Exchange Factory
================
Factory pattern for creating exchange instances.
"""

from typing import Dict, Any, Optional
import logging

from .base import BaseExchange
from .binance import BinanceAdapter
from .bitfinex import BitfinexAdapter
from .hyperliquid import HyperLiquidAdapter


class ExchangeFactory:
    """
    Factory class for creating exchange instances.

    This centralizes exchange creation and makes it easy to add new exchanges.
    """

    # Registry of available exchanges
    EXCHANGES = {
        'binance': BinanceAdapter,
        'bitfinex': BitfinexAdapter,
        'hyperliquid': HyperLiquidAdapter,
        'hyperliquid_testnet': HyperLiquidAdapter,
        'binance_testnet': BinanceAdapter,
        'bitfinex_paper': BitfinexAdapter
    }

    @classmethod
    def create_exchange(cls, exchange_name: str, credentials: Dict,
                       event_bus: Any) -> BaseExchange:
        """
        Create an exchange instance.

        Args:
            exchange_name: Name of the exchange
            credentials: Exchange credentials
            event_bus: Event bus instance

        Returns:
            Exchange instance

        Raises:
            ValueError: If exchange is not supported
        """
        # Normalize exchange name
        exchange_name = exchange_name.lower()

        # Check if exchange is supported
        if exchange_name not in cls.EXCHANGES:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        # Get exchange class
        exchange_class = cls.EXCHANGES[exchange_name]

        # Handle special cases
        if exchange_name == 'hyperliquid_testnet':
            credentials['is_mainnet'] = False
        elif exchange_name == 'binance_testnet':
            credentials['testnet'] = True
        elif exchange_name == 'bitfinex_paper':
            credentials['testnet'] = True

        # Create and return instance
        return exchange_class(credentials, event_bus)

    @classmethod
    def get_supported_exchanges(cls) -> list:
        """Get list of supported exchanges."""
        return list(cls.EXCHANGES.keys())

    @classmethod
    def is_supported(cls, exchange_name: str) -> bool:
        """Check if an exchange is supported."""
        return exchange_name.lower() in cls.EXCHANGES

    @classmethod
    def register_exchange(cls, name: str, exchange_class: type):
        """
        Register a new exchange adapter.

        Args:
            name: Exchange name
            exchange_class: Exchange adapter class
        """
        if not issubclass(exchange_class, BaseExchange):
            raise ValueError("Exchange class must inherit from BaseExchange")

        cls.EXCHANGES[name.lower()] = exchange_class
        logging.info(f"Registered exchange: {name}")

    @classmethod
    def create_multiple(cls, exchanges_config: Dict, event_bus: Any) -> Dict[str, BaseExchange]:
        """
        Create multiple exchange instances from configuration.

        Args:
            exchanges_config: Dictionary of exchange configurations
            event_bus: Event bus instance

        Returns:
            Dictionary of exchange instances
        """
        exchanges = {}

        for name, config in exchanges_config.items():
            if config.get('enabled', False):
                try:
                    exchange = cls.create_exchange(name, config, event_bus)
                    exchanges[name] = exchange
                    logging.info(f"Created exchange instance: {name}")
                except Exception as e:
                    logging.error(f"Failed to create exchange {name}: {e}")

        return exchanges