"""
Order Book Stream Module
=========================
Monitors and processes order book data across exchanges.

Tracks depth, spread, and imbalance in real-time.
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from .base_stream import BaseStream


@dataclass
class OrderBookLevel:
    """Single order book level (price and quantity)."""
    price: float
    quantity: float


@dataclass
class OrderBookData:
    """Order book snapshot data."""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime
    exchange: str

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 10000
        return 0.0

    def calculate_imbalance(self, depth: int = 5) -> float:
        """
        Calculate order book imbalance.

        Args:
            depth: Number of levels to consider

        Returns:
            Imbalance ratio (-1 to 1, positive = more bids)
        """
        bid_volume = sum(level.quantity for level in self.bids[:depth])
        ask_volume = sum(level.quantity for level in self.asks[:depth])
        total = bid_volume + ask_volume

        if total > 0:
            return (bid_volume - ask_volume) / total
        return 0.0


class OrderBookStream(BaseStream):
    """
    Order book stream handler.

    Monitors order book updates and detects significant changes
    in depth, spread, and imbalance.
    """

    def __init__(self, event_bus, config: Optional[Dict] = None):
        """
        Initialize order book stream.

        Args:
            event_bus: Event bus for publishing events
            config: Optional configuration
        """
        super().__init__(event_bus, config)

        # Configuration
        self.max_spread_bps = self.config.get('max_spread_bps', 100)  # 1%
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.7)

        # Data storage - keyed by (exchange, symbol)
        self.order_books: Dict[tuple, OrderBookData] = {}

        # Statistics
        self.stats.update({
            'total_updates': 0,
            'wide_spread_alerts': 0,
            'imbalance_alerts': 0
        })

    async def process(self, exchange: str, data: Dict) -> None:
        """
        Process order book data.

        Args:
            exchange: Exchange name
            data: Raw order book data
        """
        try:
            # Parse order book based on exchange
            if exchange.lower() == 'binance':
                order_book = self._parse_binance_orderbook(data)
            elif exchange.lower() == 'hyperliquid':
                order_book = self._parse_hyperliquid_orderbook(data)
            else:
                order_book = self._parse_generic_orderbook(data)

            # Store order book
            key = (order_book.exchange, order_book.symbol)
            self.order_books[key] = order_book

            # Update statistics
            self.stats['total_updates'] += 1

            # Emit basic update
            await self.emit_event("orderbook_update", {
                'symbol': order_book.symbol,
                'exchange': order_book.exchange,
                'spread': order_book.spread,
                'spread_bps': order_book.spread_bps,
                'mid_price': order_book.mid_price
            })

            # Check for wide spread
            if order_book.spread_bps > self.max_spread_bps:
                self.stats['wide_spread_alerts'] += 1
                await self._emit_wide_spread_alert(order_book)

            # Check for order book imbalance
            imbalance = order_book.calculate_imbalance()
            if abs(imbalance) > self.imbalance_threshold:
                self.stats['imbalance_alerts'] += 1
                await self._emit_imbalance_alert(order_book, imbalance)

            self.stats['total_processed'] += 1

        except Exception as e:
            self._handle_error(e, "processing order book")

    def _parse_binance_orderbook(self, data: Dict) -> OrderBookData:
        """Parse Binance order book data."""
        bids = [
            OrderBookLevel(price=float(bid[0]), quantity=float(bid[1]))
            for bid in data.get('b', data.get('bids', []))
        ]
        asks = [
            OrderBookLevel(price=float(ask[0]), quantity=float(ask[1]))
            for ask in data.get('a', data.get('asks', []))
        ]

        return OrderBookData(
            symbol=data.get('s', '').replace('USDT', ''),
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(int(data.get('E', data.get('T', 0))) / 1000),
            exchange='binance'
        )

    def _parse_hyperliquid_orderbook(self, data: Dict) -> OrderBookData:
        """Parse HyperLiquid order book data."""
        bids = [
            OrderBookLevel(price=float(bid['px']), quantity=float(bid['sz']))
            for bid in data.get('levels', {}).get('bids', [])
        ]
        asks = [
            OrderBookLevel(price=float(ask['px']), quantity=float(ask['sz']))
            for ask in data.get('levels', {}).get('asks', [])
        ]

        return OrderBookData(
            symbol=data.get('coin', ''),
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(data.get('time', 0)),
            exchange='hyperliquid'
        )

    def _parse_generic_orderbook(self, data: Dict) -> OrderBookData:
        """Parse generic order book data."""
        bids = [
            OrderBookLevel(price=float(bid[0]), quantity=float(bid[1]))
            for bid in data.get('bids', [])
        ]
        asks = [
            OrderBookLevel(price=float(ask[0]), quantity=float(ask[1]))
            for ask in data.get('asks', [])
        ]

        return OrderBookData(
            symbol=data.get('symbol', ''),
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(data.get('timestamp', 0)),
            exchange=data.get('exchange', 'unknown')
        )

    async def _emit_wide_spread_alert(self, order_book: OrderBookData) -> None:
        """Emit alert for wide spread."""
        await self.emit_event("wide_spread_detected", {
            'symbol': order_book.symbol,
            'exchange': order_book.exchange,
            'spread': order_book.spread,
            'spread_bps': order_book.spread_bps,
            'best_bid': order_book.bids[0].price if order_book.bids else 0,
            'best_ask': order_book.asks[0].price if order_book.asks else 0,
            'alert_level': 'warning'
        })
        self.logger.warning(
            f"WIDE SPREAD: {order_book.exchange} {order_book.symbol} "
            f"{order_book.spread_bps:.2f} bps"
        )

    async def _emit_imbalance_alert(self, order_book: OrderBookData, imbalance: float) -> None:
        """Emit alert for order book imbalance."""
        side = 'bid' if imbalance > 0 else 'ask'
        await self.emit_event("orderbook_imbalance", {
            'symbol': order_book.symbol,
            'exchange': order_book.exchange,
            'imbalance': imbalance,
            'side': side,
            'alert_level': 'info'
        })
        self.logger.info(
            f"ORDER BOOK IMBALANCE: {order_book.exchange} {order_book.symbol} "
            f"{abs(imbalance)*100:.1f}% {side}-heavy"
        )

    def get_order_book(self, symbol: str, exchange: Optional[str] = None) -> Optional[OrderBookData]:
        """
        Get current order book for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Optional specific exchange (if None, returns first match)

        Returns:
            OrderBookData if found, None otherwise
        """
        if exchange:
            key = (exchange, symbol)
            return self.order_books.get(key)
        else:
            # Return first match for symbol
            for (ex, sym), order_book in self.order_books.items():
                if sym == symbol:
                    return order_book
            return None

    def get_statistics(self) -> Dict:
        """Get order book stream statistics."""
        base_stats = super().get_statistics()

        return {
            **base_stats,
            'monitored_books': len(self.order_books)
        }
