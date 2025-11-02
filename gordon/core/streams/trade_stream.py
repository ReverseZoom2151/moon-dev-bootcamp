"""
Trade Stream Module
===================
Monitors and processes trade events across exchanges.

Consolidates functionality from Day 2 projects:
- binance_recent_trades.py (recent trade monitoring)
- binance_huge_trades.py (aggregated trade detection with colors)
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import pytz
from termcolor import cprint

from .base_stream import BaseStream


@dataclass
class TradeData:
    """Trade event data."""
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    usd_value: float
    timestamp: datetime
    exchange: str
    is_maker: bool = False

    @property
    def is_large(self) -> bool:
        """Check if this is a large trade."""
        return self.usd_value > 100000

    @property
    def is_whale(self) -> bool:
        """Check if this is a whale trade."""
        return self.usd_value > 500000


class TradeStream(BaseStream):
    """
    Trade stream handler.

    Monitors trade events across exchanges and emits alerts
    for large trades and unusual trading patterns.
    """

    def __init__(self, event_bus, config: Optional[Dict] = None):
        """
        Initialize trade stream.

        Args:
            event_bus: Event bus for publishing events
            config: Optional configuration
        """
        super().__init__(event_bus, config)

        # Configuration
        self.min_trade_usd = self.config.get('min_trade_usd', 500000)
        self.mega_trade_usd = self.config.get('mega_trade_usd', 3000000)
        self.enable_visual_output = self.config.get('enable_visual_output', False)

        # Day 2 timezone support
        self.TRADE_TIMEZONE = pytz.timezone(
            self.config.get('trade_timezone', 'US/Eastern')
        )

        # Data storage
        self.recent_large_trades: List[TradeData] = []

        # Statistics
        self.stats.update({
            'total_large_trades': 0,
            'total_trade_volume': 0,
            'buy_volume': 0,
            'sell_volume': 0
        })

    async def process(self, exchange: str, data: Dict) -> None:
        """
        Process trade data.

        Args:
            exchange: Exchange name
            data: Raw trade data
        """
        try:
            # Parse trade based on exchange
            if exchange.lower() == 'binance':
                trade = self._parse_binance_trade(data)
            elif exchange.lower() == 'hyperliquid':
                trade = self._parse_hyperliquid_trade(data)
            else:
                trade = self._parse_generic_trade(data)

            # Filter by size
            if trade.usd_value < self.min_trade_usd:
                return

            # Update statistics
            self._update_stats(trade)

            # Store trade
            self.recent_large_trades.append(trade)

            # Visual output (Day 2 style)
            if self.enable_visual_output:
                self._display_trade(trade)

            # Emit events based on size
            await self._emit_trade_events(trade)

            # Check for unusual activity
            await self._check_unusual_trading_activity(trade)

            self.stats['total_processed'] += 1

        except Exception as e:
            self._handle_error(e, "processing trade")

    def _parse_binance_trade(self, data: Dict) -> TradeData:
        """Parse Binance trade data."""
        return TradeData(
            symbol=data.get('s', '').replace('USDT', ''),
            side='sell' if data.get('m', False) else 'buy',
            price=float(data.get('p', 0)),
            quantity=float(data.get('q', 0)),
            usd_value=float(data.get('p', 0)) * float(data.get('q', 0)),
            timestamp=datetime.fromtimestamp(int(data.get('T', 0)) / 1000),
            exchange='binance',
            is_maker=data.get('m', False)
        )

    def _parse_hyperliquid_trade(self, data: Dict) -> TradeData:
        """Parse HyperLiquid trade data."""
        return TradeData(
            symbol=data.get('coin', ''),
            side=data.get('side', 'unknown'),
            price=float(data.get('px', 0)),
            quantity=float(data.get('sz', 0)),
            usd_value=float(data.get('sz', 0)) * float(data.get('px', 0)),
            timestamp=datetime.fromtimestamp(data.get('time', 0)),
            exchange='hyperliquid',
            is_maker=data.get('is_maker', False)
        )

    def _parse_generic_trade(self, data: Dict) -> TradeData:
        """Parse generic trade data."""
        return TradeData(
            symbol=data.get('symbol', ''),
            side=data.get('side', 'unknown'),
            price=float(data.get('price', 0)),
            quantity=float(data.get('quantity', 0)),
            usd_value=float(data.get('usd_value', 0)),
            timestamp=datetime.fromtimestamp(data.get('timestamp', 0)),
            exchange=data.get('exchange', 'unknown'),
            is_maker=data.get('is_maker', False)
        )

    def _update_stats(self, trade: TradeData) -> None:
        """Update statistics with new trade."""
        self.stats['total_large_trades'] += 1
        self.stats['total_trade_volume'] += trade.usd_value

        if trade.side == 'buy':
            self.stats['buy_volume'] += trade.usd_value
        else:
            self.stats['sell_volume'] += trade.usd_value

    async def _emit_trade_events(self, trade: TradeData) -> None:
        """Emit events based on trade size."""
        if trade.is_whale:
            await self.emit_event("whale_trade", {
                'trade': trade.__dict__,
                'alert_level': 'high'
            })
            self.logger.info(
                f"WHALE TRADE: {trade.symbol} {trade.side} ${trade.usd_value:,.0f}"
            )

        elif trade.is_large:
            await self.emit_event("large_trade", {
                'trade': trade.__dict__
            })

    async def _check_unusual_trading_activity(self, trade: TradeData) -> None:
        """Check for unusual trading patterns."""
        # Get recent trades for the same symbol
        recent_same_symbol = [
            t for t in self.recent_large_trades[-50:]
            if t.symbol == trade.symbol
        ]

        if len(recent_same_symbol) >= 10:
            # Calculate average trade size
            avg_size = sum(t.usd_value for t in recent_same_symbol[:-1]) / len(recent_same_symbol[:-1])

            # Check if current trade is significantly larger
            if trade.usd_value > avg_size * 5:
                await self.emit_event("unusual_trade_size", {
                    'trade': trade.__dict__,
                    'average_size': avg_size,
                    'size_multiple': trade.usd_value / avg_size,
                    'alert_level': 'high'
                })

                self.logger.info(
                    f"UNUSUAL TRADE SIZE: {trade.symbol} ${trade.usd_value:,.0f} "
                    f"({trade.usd_value/avg_size:.1f}x average)"
                )

            # Check for one-sided flow
            buy_volume = sum(t.usd_value for t in recent_same_symbol[-10:] if t.side == 'buy')
            sell_volume = sum(t.usd_value for t in recent_same_symbol[-10:] if t.side == 'sell')
            total_volume = buy_volume + sell_volume

            if total_volume > 0:
                buy_ratio = buy_volume / total_volume

                if buy_ratio > 0.8 or buy_ratio < 0.2:
                    await self.emit_event("one_sided_flow", {
                        'symbol': trade.symbol,
                        'buy_ratio': buy_ratio,
                        'buy_volume': buy_volume,
                        'sell_volume': sell_volume,
                        'alert_level': 'warning'
                    })

    def _display_trade(self, trade: TradeData) -> None:
        """Display trade with Day 2 visual formatting."""
        # Convert to configured timezone
        time_str = trade.timestamp.astimezone(self.TRADE_TIMEZONE).strftime('%H:%M:%S')

        attrs = ['bold']
        back_color = 'on_blue' if trade.side == 'buy' else 'on_magenta'
        trade_type = trade.side.upper()

        # Check for mega trades (Day 2 blinking)
        if trade.usd_value > self.mega_trade_usd:
            usd_size_m = trade.usd_value / 1000000
            # ANSI escape code for blinking
            cprint(
                f"\033[5m{trade_type} {trade.symbol[:4]} {time_str} ${usd_size_m:.2f}m\033[0m",
                'white', back_color, attrs=attrs
            )
        else:
            usd_size_m = trade.usd_value / 1000000
            cprint(
                f"{trade_type} {trade.symbol[:4]} {time_str} ${usd_size_m:.2f}m",
                'white', back_color, attrs=attrs
            )

    def clear_old_data(self, max_age_minutes: int = 60) -> None:
        """Clear trades older than specified age."""
        cutoff = datetime.now().timestamp() - (max_age_minutes * 60)
        self.recent_large_trades = [
            trade for trade in self.recent_large_trades
            if trade.timestamp.timestamp() > cutoff
        ]

    def get_statistics(self) -> Dict:
        """Get trade stream statistics."""
        base_stats = super().get_statistics()
        total_volume = self.stats['buy_volume'] + self.stats['sell_volume']
        buy_ratio = self.stats['buy_volume'] / max(total_volume, 1)

        return {
            **base_stats,
            'recent_large_trades': len(self.recent_large_trades),
            'buy_sell_ratio': buy_ratio
        }
