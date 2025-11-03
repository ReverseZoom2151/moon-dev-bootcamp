"""
Streams Package
===============
Modular market data streaming system.

This package provides specialized stream handlers for different types
of market data, all inheriting from a common base class.

Stream Types:
- BaseStream: Abstract base class for all streams
- LiquidationStream: Monitors liquidation events
- TradeStream: Monitors trade events
- FundingStream: Monitors funding rates
- OrderBookStream: Monitors order book depth
- TickerStream: Monitors price/ticker data
- StreamAggregator: Combines multiple streams
- TradeAggregator: Aggregates trades in time windows

Usage:
    from gordon.core.streams import (
        LiquidationStream,
        TradeStream,
        FundingStream,
        StreamAggregator
    )

    # Initialize aggregator with all streams
    aggregator = StreamAggregator(event_bus, config)

    # Or use individual streams
    liquidation_stream = LiquidationStream(event_bus, config)
    await liquidation_stream.process('binance', liquidation_data)
"""

from .base_stream import BaseStream
from .liquidation_stream import LiquidationStream, LiquidationData
from .trade_stream import TradeStream, TradeData
from .funding_stream import FundingStream, FundingData
from .orderbook_stream import OrderBookStream, OrderBookData, OrderBookLevel
from .ticker_stream import TickerStream, TickerData
from .aggregator import StreamAggregator, TradeAggregator

__all__ = [
    # Base
    'BaseStream',

    # Streams
    'LiquidationStream',
    'TradeStream',
    'FundingStream',
    'OrderBookStream',
    'TickerStream',

    # Data classes
    'LiquidationData',
    'TradeData',
    'FundingData',
    'OrderBookData',
    'OrderBookLevel',
    'TickerData',

    # Aggregators
    'StreamAggregator',
    'TradeAggregator',
]

# Version info
__version__ = '1.0.0'
