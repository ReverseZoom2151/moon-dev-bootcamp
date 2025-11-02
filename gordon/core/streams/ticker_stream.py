"""
Ticker Stream Module
====================
Monitors and processes ticker/price data across exchanges.

Tracks price changes, volume, and 24-hour statistics.
"""

from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

from .base_stream import BaseStream


@dataclass
class TickerData:
    """Ticker/price data."""
    symbol: str
    last_price: float
    price_change_24h: float
    price_change_percent_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    quote_volume_24h: float
    timestamp: datetime
    exchange: str

    @property
    def is_volatile(self) -> bool:
        """Check if price change is volatile (>5%)."""
        return abs(self.price_change_percent_24h) > 5.0

    @property
    def is_highly_volatile(self) -> bool:
        """Check if price change is highly volatile (>10%)."""
        return abs(self.price_change_percent_24h) > 10.0


class TickerStream(BaseStream):
    """
    Ticker stream handler.

    Monitors ticker updates and detects significant price movements
    and volume changes.
    """

    def __init__(self, event_bus, config: Optional[Dict] = None):
        """
        Initialize ticker stream.

        Args:
            event_bus: Event bus for publishing events
            config: Optional configuration
        """
        super().__init__(event_bus, config)

        # Configuration
        self.volatility_threshold = self.config.get('volatility_threshold', 5.0)
        self.high_volatility_threshold = self.config.get('high_volatility_threshold', 10.0)
        self.volume_spike_threshold = self.config.get('volume_spike_threshold', 3.0)

        # Data storage - keyed by (exchange, symbol)
        self.tickers: Dict[tuple, TickerData] = {}
        self.previous_volumes: Dict[tuple, float] = {}

        # Statistics
        self.stats.update({
            'total_updates': 0,
            'volatility_alerts': 0,
            'volume_spike_alerts': 0
        })

    async def process(self, exchange: str, data: Dict) -> None:
        """
        Process ticker data.

        Args:
            exchange: Exchange name
            data: Raw ticker data
        """
        try:
            # Parse ticker based on exchange
            if exchange.lower() == 'binance':
                ticker = self._parse_binance_ticker(data)
            elif exchange.lower() == 'hyperliquid':
                ticker = self._parse_hyperliquid_ticker(data)
            else:
                ticker = self._parse_generic_ticker(data)

            # Store ticker
            key = (ticker.exchange, ticker.symbol)
            old_ticker = self.tickers.get(key)
            self.tickers[key] = ticker

            # Update statistics
            self.stats['total_updates'] += 1

            # Emit basic update
            await self.emit_event("ticker_update", {
                'symbol': ticker.symbol,
                'exchange': ticker.exchange,
                'price': ticker.last_price,
                'change_24h': ticker.price_change_percent_24h,
                'volume_24h': ticker.volume_24h
            })

            # Check for volatility
            if ticker.is_highly_volatile:
                self.stats['volatility_alerts'] += 1
                await self._emit_volatility_alert(ticker, 'high')
            elif ticker.is_volatile:
                self.stats['volatility_alerts'] += 1
                await self._emit_volatility_alert(ticker, 'moderate')

            # Check for volume spikes
            if old_ticker:
                await self._check_volume_spike(ticker, old_ticker)

            self.stats['total_processed'] += 1

        except Exception as e:
            self._handle_error(e, "processing ticker")

    def _parse_binance_ticker(self, data: Dict) -> TickerData:
        """Parse Binance ticker data."""
        return TickerData(
            symbol=data.get('s', '').replace('USDT', ''),
            last_price=float(data.get('c', 0)),
            price_change_24h=float(data.get('p', 0)),
            price_change_percent_24h=float(data.get('P', 0)),
            high_24h=float(data.get('h', 0)),
            low_24h=float(data.get('l', 0)),
            volume_24h=float(data.get('v', 0)),
            quote_volume_24h=float(data.get('q', 0)),
            timestamp=datetime.fromtimestamp(int(data.get('E', 0)) / 1000),
            exchange='binance'
        )

    def _parse_hyperliquid_ticker(self, data: Dict) -> TickerData:
        """Parse HyperLiquid ticker data."""
        return TickerData(
            symbol=data.get('coin', ''),
            last_price=float(data.get('px', 0)),
            price_change_24h=float(data.get('change_24h', 0)),
            price_change_percent_24h=float(data.get('change_pct_24h', 0)),
            high_24h=float(data.get('high_24h', 0)),
            low_24h=float(data.get('low_24h', 0)),
            volume_24h=float(data.get('volume_24h', 0)),
            quote_volume_24h=float(data.get('quote_volume_24h', 0)),
            timestamp=datetime.fromtimestamp(data.get('time', 0)),
            exchange='hyperliquid'
        )

    def _parse_generic_ticker(self, data: Dict) -> TickerData:
        """Parse generic ticker data."""
        return TickerData(
            symbol=data.get('symbol', ''),
            last_price=float(data.get('last_price', 0)),
            price_change_24h=float(data.get('price_change_24h', 0)),
            price_change_percent_24h=float(data.get('price_change_percent_24h', 0)),
            high_24h=float(data.get('high_24h', 0)),
            low_24h=float(data.get('low_24h', 0)),
            volume_24h=float(data.get('volume_24h', 0)),
            quote_volume_24h=float(data.get('quote_volume_24h', 0)),
            timestamp=datetime.fromtimestamp(data.get('timestamp', 0)),
            exchange=data.get('exchange', 'unknown')
        )

    async def _emit_volatility_alert(self, ticker: TickerData, level: str) -> None:
        """Emit alert for price volatility."""
        alert_level = 'warning' if level == 'high' else 'info'

        await self.emit_event("price_volatility", {
            'symbol': ticker.symbol,
            'exchange': ticker.exchange,
            'price': ticker.last_price,
            'change_24h': ticker.price_change_24h,
            'change_percent_24h': ticker.price_change_percent_24h,
            'volatility_level': level,
            'alert_level': alert_level
        })

        if level == 'high':
            self.logger.warning(
                f"HIGH VOLATILITY: {ticker.exchange} {ticker.symbol} "
                f"{ticker.price_change_percent_24h:+.2f}%"
            )
        else:
            self.logger.info(
                f"Volatility: {ticker.exchange} {ticker.symbol} "
                f"{ticker.price_change_percent_24h:+.2f}%"
            )

    async def _check_volume_spike(self, current: TickerData, previous: TickerData) -> None:
        """Check for volume spikes."""
        if previous.volume_24h > 0:
            volume_ratio = current.volume_24h / previous.volume_24h

            if volume_ratio > self.volume_spike_threshold:
                self.stats['volume_spike_alerts'] += 1

                await self.emit_event("volume_spike", {
                    'symbol': current.symbol,
                    'exchange': current.exchange,
                    'current_volume': current.volume_24h,
                    'previous_volume': previous.volume_24h,
                    'volume_ratio': volume_ratio,
                    'alert_level': 'info'
                })

                self.logger.info(
                    f"VOLUME SPIKE: {current.exchange} {current.symbol} "
                    f"{volume_ratio:.1f}x increase"
                )

    def get_ticker(self, symbol: str, exchange: Optional[str] = None) -> Optional[TickerData]:
        """
        Get current ticker for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Optional specific exchange (if None, returns first match)

        Returns:
            TickerData if found, None otherwise
        """
        if exchange:
            key = (exchange, symbol)
            return self.tickers.get(key)
        else:
            # Return first match for symbol
            for (ex, sym), ticker in self.tickers.items():
                if sym == symbol:
                    return ticker
            return None

    def get_volatile_symbols(self) -> Dict[tuple, TickerData]:
        """
        Get all symbols with volatile price movements.

        Returns:
            Dictionary of volatile tickers keyed by (exchange, symbol)
        """
        return {
            key: ticker for key, ticker in self.tickers.items()
            if ticker.is_volatile
        }

    def get_statistics(self) -> Dict:
        """Get ticker stream statistics."""
        base_stats = super().get_statistics()

        return {
            **base_stats,
            'monitored_symbols': len(self.tickers),
            'volatile_symbols': len(self.get_volatile_symbols())
        }
