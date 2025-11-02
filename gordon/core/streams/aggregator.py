"""
Stream Aggregator Module
=========================
Combines and aggregates data from multiple streams.

Includes Day 2's trade aggregation logic for detecting
coordinated large trades within time windows.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from termcolor import cprint
import logging


class TradeAggregator:
    """
    Trade aggregation logic from binance_huge_trades.py.
    Aggregates trades within time windows to detect large coordinated moves.

    Includes Day 2's exact aggregation logic and visual output.
    """

    def __init__(self, window_seconds: int = 1, min_size: float = 500000, enable_visual: bool = False):
        """
        Initialize trade aggregator.

        Args:
            window_seconds: Time window for aggregation
            min_size: Minimum USD size to report (Day 2: 500k)
            enable_visual: Enable Day 2 visual output
        """
        self.window_seconds = window_seconds
        self.min_size = min_size
        self.trade_buckets: Dict[tuple, float] = {}
        self.logger = logging.getLogger(__name__)
        self.enable_visual = enable_visual

    async def add_trade(self, symbol: str, timestamp: datetime, usd_size: float, is_sell: bool):
        """
        Add trade to aggregation bucket.

        Args:
            symbol: Trading symbol
            timestamp: Trade timestamp
            usd_size: Trade size in USD
            is_sell: True if sell, False if buy
        """
        # Round timestamp to window
        window = int(timestamp.timestamp() / self.window_seconds) * self.window_seconds
        trade_key = (symbol, window, is_sell)

        self.trade_buckets[trade_key] = self.trade_buckets.get(trade_key, 0) + usd_size

    async def check_aggregated_trades(self, event_bus: Any):
        """
        Check for significant aggregated trades.

        Day 2 style with typo 'deletetions' preserved for authenticity.

        Args:
            event_bus: Event bus for emitting events
        """
        current_window = int(datetime.now().timestamp() / self.window_seconds) * self.window_seconds

        deletetions = []  # Day 2 typo preserved for authenticity
        for trade_key, usd_size in self.trade_buckets.items():
            symbol, window, is_sell = trade_key

            # Process completed windows
            if window < current_window and usd_size >= self.min_size:
                # Day 2 visual output (if enabled)
                if self.enable_visual:
                    attrs = ['bold']
                    back_color = 'on_magenta' if is_sell else 'on_blue'
                    trade_type = "SELL" if is_sell else 'BUY'
                    time_str = datetime.fromtimestamp(window).strftime("%H:%M:%S")

                    # Day 2 blinking for huge trades (>$3M)
                    if usd_size > 3000000:
                        usd_size_m = usd_size / 1000000
                        cprint(
                            f"\033[5m{trade_type} {symbol} {time_str} ${usd_size_m:.2f}m\033[0m",
                            'white', back_color, attrs=attrs
                        )
                    else:
                        usd_size_m = usd_size / 1000000
                        cprint(
                            f"{trade_type} {symbol} {time_str} ${usd_size_m:.2f}m",
                            'white', back_color, attrs=attrs
                        )

                await event_bus.emit("aggregated_large_trade", {
                    'symbol': symbol,
                    'window': datetime.fromtimestamp(window),
                    'side': 'sell' if is_sell else 'buy',
                    'total_volume': usd_size,
                    'alert_level': 'high' if usd_size > self.min_size * 5 else 'medium'
                })

                self.logger.info(
                    f"AGGREGATED TRADE: {symbol} {'SELL' if is_sell else 'BUY'} "
                    f"${usd_size:,.0f} in {self.window_seconds}s"
                )
                deletetions.append(trade_key)

        # Clean up processed buckets
        for key in deletetions:
            del self.trade_buckets[key]

    def get_statistics(self) -> Dict:
        """Get aggregator statistics."""
        total_buckets = len(self.trade_buckets)
        total_volume = sum(self.trade_buckets.values())

        return {
            'active_buckets': total_buckets,
            'total_aggregated_volume': total_volume,
            'window_seconds': self.window_seconds,
            'min_size': self.min_size
        }


class StreamAggregator:
    """
    Aggregates data from multiple stream types.

    Provides unified access to all stream data and cross-stream analysis.
    """

    def __init__(self, event_bus: Any, config: Optional[Dict] = None):
        """
        Initialize stream aggregator.

        Args:
            event_bus: Event bus for publishing events
            config: Optional configuration
        """
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Import stream classes
        from .liquidation_stream import LiquidationStream
        from .trade_stream import TradeStream
        from .funding_stream import FundingStream
        from .orderbook_stream import OrderBookStream
        from .ticker_stream import TickerStream

        # Initialize all streams
        self.liquidation_stream = LiquidationStream(event_bus, config.get('liquidation', {}))
        self.trade_stream = TradeStream(event_bus, config.get('trade', {}))
        self.funding_stream = FundingStream(event_bus, config.get('funding', {}))
        self.orderbook_stream = OrderBookStream(event_bus, config.get('orderbook', {}))
        self.ticker_stream = TickerStream(event_bus, config.get('ticker', {}))

        # Trade aggregator
        self.trade_aggregator = TradeAggregator(
            window_seconds=config.get('aggregation_window', 1),
            min_size=config.get('min_aggregated_trade', 500000),
            enable_visual=config.get('enable_visual_output', False)
        )

        # Stream mapping
        self.streams = {
            'liquidation': self.liquidation_stream,
            'trade': self.trade_stream,
            'funding': self.funding_stream,
            'orderbook': self.orderbook_stream,
            'ticker': self.ticker_stream
        }

    async def process(self, stream_type: str, exchange: str, data: Dict) -> None:
        """
        Process data through appropriate stream.

        Args:
            stream_type: Type of stream (liquidation, trade, funding, etc.)
            exchange: Exchange name
            data: Raw data to process
        """
        stream = self.streams.get(stream_type)
        if stream:
            await stream.process(exchange, data)

            # Add trades to aggregator
            if stream_type == 'trade':
                # Extract trade info for aggregator
                # This is a simplified extraction - real implementation would
                # need to parse the data properly
                try:
                    from .trade_stream import TradeData
                    # Parse trade to get details for aggregation
                    if exchange.lower() == 'binance':
                        symbol = data.get('s', '').replace('USDT', '')
                        timestamp = datetime.fromtimestamp(int(data.get('T', 0)) / 1000)
                        usd_size = float(data.get('p', 0)) * float(data.get('q', 0))
                        is_sell = data.get('m', False)

                        await self.trade_aggregator.add_trade(symbol, timestamp, usd_size, is_sell)
                except Exception as e:
                    self.logger.error(f"Error adding trade to aggregator: {e}")
        else:
            self.logger.warning(f"Unknown stream type: {stream_type}")

    async def check_aggregated_trades(self):
        """Check and emit aggregated trade events."""
        await self.trade_aggregator.check_aggregated_trades(self.event_bus)

    def get_market_summary(self, symbol: Optional[str] = None) -> Dict:
        """
        Get comprehensive market summary across all streams.

        Args:
            symbol: Optional symbol to filter by

        Returns:
            Dictionary containing data from all streams
        """
        if symbol:
            # Symbol-specific summary
            return {
                'symbol': symbol,
                'funding': self.funding_stream.get_funding_rate(symbol).__dict__ if self.funding_stream.get_funding_rate(symbol) else None,
                'ticker': self.ticker_stream.get_ticker(symbol).__dict__ if self.ticker_stream.get_ticker(symbol) else None,
                'orderbook': self.orderbook_stream.get_order_book(symbol).__dict__ if self.orderbook_stream.get_order_book(symbol) else None,
                'recent_liquidations': [
                    liq.__dict__ for liq in self.liquidation_stream.recent_liquidations[-10:]
                    if liq.symbol == symbol
                ],
                'recent_trades': [
                    trade.__dict__ for trade in self.trade_stream.recent_large_trades[-10:]
                    if trade.symbol == symbol
                ]
            }
        else:
            # Overall market summary
            return {
                'liquidations': {
                    'total': self.liquidation_stream.stats['total_liquidations'],
                    'total_volume': self.liquidation_stream.stats['total_liquidation_volume'],
                    'long_short_ratio': self.liquidation_stream.stats['long_liquidations'] / max(self.liquidation_stream.stats['short_liquidations'], 1)
                },
                'trades': {
                    'total_large_trades': self.trade_stream.stats['total_large_trades'],
                    'total_volume': self.trade_stream.stats['total_trade_volume'],
                    'buy_sell_ratio': self.trade_stream.stats['buy_volume'] / max(self.trade_stream.stats['sell_volume'], 1)
                },
                'funding': {
                    'monitored_symbols': len(self.funding_stream.funding_rates),
                    'extreme_funding': [
                        f"{ex}:{sym}" for (ex, sym), funding in self.funding_stream.funding_rates.items()
                        if funding.is_extreme
                    ]
                },
                'ticker': {
                    'monitored_symbols': len(self.ticker_stream.tickers),
                    'volatile_symbols': len(self.ticker_stream.get_volatile_symbols())
                },
                'orderbook': {
                    'monitored_books': len(self.orderbook_stream.order_books)
                }
            }

    def get_statistics(self) -> Dict:
        """Get statistics from all streams."""
        return {
            'liquidation': self.liquidation_stream.get_statistics(),
            'trade': self.trade_stream.get_statistics(),
            'funding': self.funding_stream.get_statistics(),
            'orderbook': self.orderbook_stream.get_statistics(),
            'ticker': self.ticker_stream.get_statistics(),
            'aggregator': self.trade_aggregator.get_statistics()
        }

    def clear_old_data(self, max_age_minutes: int = 60):
        """Clear old data from all streams."""
        self.liquidation_stream.clear_old_data(max_age_minutes)
        self.trade_stream.clear_old_data(max_age_minutes)
