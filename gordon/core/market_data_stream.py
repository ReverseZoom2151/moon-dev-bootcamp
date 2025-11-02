"""
Market Data Streaming Module
============================
Exchange-agnostic real-time market data streaming.

REFACTORED: This module now serves as a backward-compatible wrapper
around the new modular streaming system in the streams/ package.

The new modular structure provides:
- streams/base_stream.py: Base class for all streams
- streams/liquidation_stream.py: Liquidation monitoring
- streams/trade_stream.py: Trade data streaming
- streams/funding_stream.py: Funding rate streaming
- streams/orderbook_stream.py: Order book streaming
- streams/ticker_stream.py: Ticker/price streaming
- streams/aggregator.py: Stream aggregation and combination

For new code, import directly from streams package:
    from exchange_orchestrator.core.streams import (
        LiquidationStream, TradeStream, StreamAggregator
    )

This module maintains backward compatibility with existing code.

Consolidates functionality from Day 2 projects:
- Day_2_Projects/binance_big_liqs.py (large liquidation tracking with visual output)
- Day_2_Projects/binance_huge_trades.py (aggregated trade detection with colors)
- Day_2_Projects/binance_funding.py (funding rate monitoring)
- Day_2_Projects/binance_liqs.py (basic liquidation stream)
- Day_2_Projects/binance_recent_trades.py (recent trade monitoring)
"""

from typing import Dict, Optional, Any
from enum import Enum
import logging

# Import from modular streams package
from .streams import StreamAggregator


class StreamType(Enum):
    """Types of market data streams."""
    LIQUIDATIONS = "liquidations"
    LARGE_TRADES = "large_trades"
    FUNDING_RATES = "funding"
    TRADES = "trades"
    ORDERBOOK = "orderbook"
    TICKER = "ticker"


class MarketDataStream:
    """
    Unified market data streaming handler.

    REFACTORED: Now delegates to the modular StreamAggregator.
    Maintains backward compatibility with existing code.

    Consolidates:
    - binance_liqs.py / binance_big_liqs.py (liquidation tracking)
    - binance_huge_trades.py / binance_recent_trades.py (large trade detection)
    - binance_funding.py (funding rate monitoring)
    """

    def __init__(self, event_bus: Any, config: Optional[Dict] = None):
        """
        Initialize the market data stream handler.

        Args:
            event_bus: Event bus for publishing events
            config: Optional configuration
        """
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Configure individual stream settings
        stream_config = {
            'liquidation': {
                'min_liquidation_usd': self.config.get('min_liquidation_usd', 100000),
                'enable_visual_output': self.config.get('enable_visual_output', False),
                'enable_csv_logging': self.config.get('enable_csv_logging', False),
                'liquidation_timezone': 'Europe/Bucharest',
                'csv_file': 'binance_bigliqs.csv'
            },
            'trade': {
                'min_trade_usd': self.config.get('min_trade_usd', 500000),
                'mega_trade_usd': self.config.get('mega_trade_usd', 3000000),
                'enable_visual_output': self.config.get('enable_visual_output', False),
                'trade_timezone': 'US/Eastern'
            },
            'funding': {
                'extreme_funding_threshold': self.config.get('extreme_funding_threshold', 50)
            },
            'aggregation_window': self.config.get('aggregation_window', 1),
            'min_aggregated_trade': self.config.get('min_aggregated_trade', 500000),
            'enable_visual_output': self.config.get('enable_visual_output', False)
        }

        # Initialize the new modular stream aggregator
        self.aggregator = StreamAggregator(event_bus, stream_config)

        # Provide backward-compatible access to individual components
        self.liquidation_stream = self.aggregator.liquidation_stream
        self.trade_stream = self.aggregator.trade_stream
        self.funding_stream = self.aggregator.funding_stream
        self.trade_aggregator = self.aggregator.trade_aggregator

        # Backward-compatible property accessors
        @property
        def recent_liquidations():
            return self.liquidation_stream.recent_liquidations

        @property
        def recent_large_trades():
            return self.trade_stream.recent_large_trades

        @property
        def funding_rates():
            return self.funding_stream.funding_rates

        @property
        def stats():
            # Combine stats from all streams for backward compatibility
            return {
                'total_liquidations': self.liquidation_stream.stats['total_liquidations'],
                'total_liquidation_volume': self.liquidation_stream.stats['total_liquidation_volume'],
                'total_large_trades': self.trade_stream.stats['total_large_trades'],
                'total_trade_volume': self.trade_stream.stats['total_trade_volume'],
                'long_liquidations': self.liquidation_stream.stats['long_liquidations'],
                'short_liquidations': self.liquidation_stream.stats['short_liquidations']
            }

        # Make properties accessible
        self.recent_liquidations = self.liquidation_stream.recent_liquidations
        self.recent_large_trades = self.trade_stream.recent_large_trades
        self.funding_rates = self.funding_stream.funding_rates

        # Day 2 specific configurations (kept for backward compatibility)
        self.DAY2_SYMBOLS = ['btcusdt', 'ethusdt', 'solusdt', 'bnbusdt', 'dogeusdt', 'wifusdt']
        self.min_liquidation_usd = self.config.get('min_liquidation_usd', 100000)
        self.min_trade_usd = self.config.get('min_trade_usd', 500000)
        self.mega_trade_usd = self.config.get('mega_trade_usd', 3000000)
        self.extreme_funding_threshold = self.config.get('extreme_funding_threshold', 50)

    async def process_liquidation(self, exchange: str, data: Dict):
        """
        Process liquidation data (from binance_liqs.py / binance_big_liqs.py).

        REFACTORED: Delegates to liquidation_stream module.

        Args:
            exchange: Exchange name
            data: Raw liquidation data
        """
        await self.liquidation_stream.process(exchange, data)

    async def process_trade(self, exchange: str, data: Dict):
        """
        Process trade data (from binance_recent_trades.py / binance_huge_trades.py).

        REFACTORED: Delegates to trade_stream module and aggregator.

        Args:
            exchange: Exchange name
            data: Raw trade data
        """
        await self.aggregator.process('trade', exchange, data)

    async def process_funding(self, exchange: str, data: Dict):
        """
        Process funding rate data (from binance_funding.py).

        REFACTORED: Delegates to funding_stream module.

        Args:
            exchange: Exchange name
            data: Raw funding data
        """
        await self.funding_stream.process(exchange, data)

    # Statistics and monitoring (delegated to aggregator)

    def get_statistics(self) -> Dict:
        """
        Get current statistics.

        REFACTORED: Delegates to aggregator.
        """
        return self.aggregator.get_statistics()

    def get_market_summary(self, symbol: Optional[str] = None) -> Dict:
        """
        Get market summary for a symbol or all symbols.

        REFACTORED: Delegates to aggregator.
        """
        return self.aggregator.get_market_summary(symbol)

    def clear_old_data(self, max_age_minutes: int = 60):
        """
        Clear data older than specified age.

        REFACTORED: Delegates to aggregator.
        """
        self.aggregator.clear_old_data(max_age_minutes)