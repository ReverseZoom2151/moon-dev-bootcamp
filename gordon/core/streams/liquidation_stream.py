"""
Liquidation Stream Module
==========================
Monitors and processes liquidation events across exchanges.

Consolidates functionality from Day 2 projects:
- binance_liqs.py (basic liquidation stream)
- binance_big_liqs.py (large liquidation tracking with visual output)
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import csv
import os
import pytz
from termcolor import cprint

from .base_stream import BaseStream


@dataclass
class LiquidationData:
    """Liquidation event data."""
    symbol: str
    side: str  # 'long' or 'short'
    price: float
    quantity: float
    usd_value: float
    timestamp: datetime
    exchange: str

    @property
    def is_large(self) -> bool:
        """Check if this is a large liquidation."""
        return self.usd_value > 100000

    @property
    def is_huge(self) -> bool:
        """Check if this is a huge liquidation."""
        return self.usd_value > 1000000


class LiquidationStream(BaseStream):
    """
    Liquidation stream handler.

    Monitors liquidation events across exchanges and emits alerts
    for large liquidations and potential cascades.
    """

    def __init__(self, event_bus, config: Optional[Dict] = None):
        """
        Initialize liquidation stream.

        Args:
            event_bus: Event bus for publishing events
            config: Optional configuration
        """
        super().__init__(event_bus, config)

        # Configuration
        self.min_liquidation_usd = self.config.get('min_liquidation_usd', 100000)
        self.enable_visual_output = self.config.get('enable_visual_output', False)
        self.enable_csv_logging = self.config.get('enable_csv_logging', False)

        # Day 2 timezone support
        self.LIQUIDATION_TIMEZONE = pytz.timezone(
            self.config.get('liquidation_timezone', 'Europe/Bucharest')
        )

        # Data storage
        self.recent_liquidations: List[LiquidationData] = []

        # Statistics
        self.stats.update({
            'total_liquidations': 0,
            'total_liquidation_volume': 0,
            'long_liquidations': 0,
            'short_liquidations': 0
        })

        # CSV setup
        if self.enable_csv_logging:
            self.csv_file = self.config.get('csv_file', 'binance_bigliqs.csv')
            self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'symbol', 'side', 'order_type', 'time_in_force',
                    'original_quantity', 'price', 'average_price',
                    'order_status', 'order_last_filled_quantity',
                    'order_filled_accumulated_quantity', 'order_trade_time',
                    'usd_size'
                ])

    async def process(self, exchange: str, data: Dict) -> None:
        """
        Process liquidation data.

        Args:
            exchange: Exchange name
            data: Raw liquidation data
        """
        try:
            # Parse liquidation based on exchange
            if exchange.lower() == 'binance':
                liquidation = self._parse_binance_liquidation(data)
            elif exchange.lower() == 'hyperliquid':
                liquidation = self._parse_hyperliquid_liquidation(data)
            else:
                liquidation = self._parse_generic_liquidation(data)

            # Filter by size
            if liquidation.usd_value < self.min_liquidation_usd:
                return

            # Update statistics
            self._update_stats(liquidation)

            # Store liquidation
            self.recent_liquidations.append(liquidation)

            # Visual output (Day 2 style)
            if self.enable_visual_output:
                self._display_liquidation(liquidation)

            # CSV logging
            if self.enable_csv_logging:
                self._write_csv(liquidation)

            # Emit events based on size
            await self._emit_liquidation_events(liquidation)

            # Check for cascades
            await self._check_liquidation_cascade(liquidation)

            self.stats['total_processed'] += 1

        except Exception as e:
            self._handle_error(e, "processing liquidation")

    def _parse_binance_liquidation(self, data: Dict) -> LiquidationData:
        """Parse Binance liquidation data."""
        order_data = data.get('o', data)
        return LiquidationData(
            symbol=order_data['s'].replace('USDT', ''),
            side='long' if order_data['S'] == 'SELL' else 'short',
            price=float(order_data['p']),
            quantity=float(order_data['z']),
            usd_value=float(order_data['z']) * float(order_data['p']),
            timestamp=datetime.fromtimestamp(int(order_data['T']) / 1000),
            exchange='binance'
        )

    def _parse_hyperliquid_liquidation(self, data: Dict) -> LiquidationData:
        """Parse HyperLiquid liquidation data."""
        return LiquidationData(
            symbol=data.get('coin', ''),
            side='long' if data.get('side') == 'sell' else 'short',
            price=float(data.get('px', 0)),
            quantity=float(data.get('sz', 0)),
            usd_value=float(data.get('sz', 0)) * float(data.get('px', 0)),
            timestamp=datetime.fromtimestamp(data.get('time', 0)),
            exchange='hyperliquid'
        )

    def _parse_generic_liquidation(self, data: Dict) -> LiquidationData:
        """Parse generic liquidation data."""
        return LiquidationData(
            symbol=data.get('symbol', ''),
            side=data.get('side', 'unknown'),
            price=float(data.get('price', 0)),
            quantity=float(data.get('quantity', 0)),
            usd_value=float(data.get('usd_value', 0)),
            timestamp=datetime.fromtimestamp(data.get('timestamp', 0)),
            exchange=data.get('exchange', 'unknown')
        )

    def _update_stats(self, liquidation: LiquidationData) -> None:
        """Update statistics with new liquidation."""
        self.stats['total_liquidations'] += 1
        self.stats['total_liquidation_volume'] += liquidation.usd_value

        if liquidation.side == 'long':
            self.stats['long_liquidations'] += 1
        else:
            self.stats['short_liquidations'] += 1

    async def _emit_liquidation_events(self, liquidation: LiquidationData) -> None:
        """Emit events based on liquidation size."""
        if liquidation.is_huge:
            await self.emit_event("huge_liquidation", {
                'liquidation': liquidation.__dict__,
                'alert_level': 'critical'
            })
            self.logger.warning(
                f"HUGE LIQUIDATION: {liquidation.symbol} {liquidation.side} "
                f"${liquidation.usd_value:,.0f}"
            )

        elif liquidation.is_large:
            await self.emit_event("large_liquidation", {
                'liquidation': liquidation.__dict__,
                'alert_level': 'warning'
            })
            self.logger.info(
                f"Large liquidation: {liquidation.symbol} {liquidation.side} "
                f"${liquidation.usd_value:,.0f}"
            )

        else:
            await self.emit_event("liquidation", {
                'liquidation': liquidation.__dict__
            })

    async def _check_liquidation_cascade(self, liquidation: LiquidationData) -> None:
        """Check for potential liquidation cascades."""
        # Get recent liquidations for the same symbol
        recent_same_symbol = [
            liq for liq in self.recent_liquidations[-20:]
            if liq.symbol == liquidation.symbol
        ]

        if len(recent_same_symbol) >= 5:
            # Check if they're all in the same direction
            same_side = all(liq.side == liquidation.side for liq in recent_same_symbol[-5:])

            if same_side:
                total_volume = sum(liq.usd_value for liq in recent_same_symbol[-5:])

                await self.emit_event("liquidation_cascade_detected", {
                    'symbol': liquidation.symbol,
                    'side': liquidation.side,
                    'count': len(recent_same_symbol[-5:]),
                    'total_volume': total_volume,
                    'alert_level': 'critical'
                })

                self.logger.warning(
                    f"LIQUIDATION CASCADE: {liquidation.symbol} {liquidation.side} - "
                    f"{len(recent_same_symbol[-5:])} liquidations, ${total_volume:,.0f}"
                )

    def _display_liquidation(self, liquidation: LiquidationData) -> None:
        """Display liquidation with Day 2 visual formatting."""
        # Convert timestamp to configured timezone
        time_str = liquidation.timestamp.astimezone(self.LIQUIDATION_TIMEZONE).strftime('%H:%M:%S')

        # Day 2 formatting
        liq_type = 'L LIQ' if liquidation.side == 'long' else 'S LIQ'
        symbol_short = liquidation.symbol[:4]  # Truncate to 4 chars
        output = f"{liq_type} {symbol_short} {time_str} {liquidation.usd_value:.2f}"

        # Day 2 colors
        color = 'blue' if liquidation.side == 'long' else 'magenta'
        attrs = ['bold'] if liquidation.usd_value > 10000 else []

        cprint(output, 'white', f'on_{color}', attrs=attrs)
        print('')  # Add empty line

    def _write_csv(self, liquidation: LiquidationData) -> None:
        """Write liquidation to CSV (Day 2 format)."""
        row = [
            liquidation.symbol,
            'SELL' if liquidation.side == 'long' else 'BUY',
            'LIMIT',  # order_type
            'GTC',    # time_in_force
            liquidation.quantity,
            liquidation.price,
            liquidation.price,  # average_price
            'FILLED',  # order_status
            liquidation.quantity,  # last_filled_quantity
            liquidation.quantity,  # filled_accumulated_quantity
            int(liquidation.timestamp.timestamp() * 1000),  # trade_time
            liquidation.usd_value
        ]

        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def clear_old_data(self, max_age_minutes: int = 60) -> None:
        """Clear liquidations older than specified age."""
        cutoff = datetime.now().timestamp() - (max_age_minutes * 60)
        self.recent_liquidations = [
            liq for liq in self.recent_liquidations
            if liq.timestamp.timestamp() > cutoff
        ]

    def get_statistics(self) -> Dict:
        """Get liquidation stream statistics."""
        base_stats = super().get_statistics()
        liquidation_ratio = self.stats['long_liquidations'] / max(self.stats['short_liquidations'], 1)

        return {
            **base_stats,
            'recent_liquidations': len(self.recent_liquidations),
            'liquidation_ratio': liquidation_ratio
        }
