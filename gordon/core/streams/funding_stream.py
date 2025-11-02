"""
Funding Stream Module
=====================
Monitors and processes funding rate data across exchanges.

Consolidates functionality from Day 2 projects:
- binance_funding.py (funding rate monitoring)
"""

from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

from .base_stream import BaseStream


@dataclass
class FundingData:
    """Funding rate data."""
    symbol: str
    funding_rate: float
    mark_price: float
    index_price: float
    next_funding_time: datetime
    timestamp: datetime
    exchange: str

    @property
    def yearly_rate(self) -> float:
        """Calculate yearly funding rate."""
        return self.funding_rate * 3 * 365 * 100  # Assuming 8-hour funding periods

    @property
    def is_extreme(self) -> bool:
        """Check if funding is extreme."""
        yearly = self.yearly_rate
        return yearly > 50 or yearly < -50


class FundingStream(BaseStream):
    """
    Funding rate stream handler.

    Monitors funding rates across exchanges and emits alerts
    for extreme funding and potential arbitrage opportunities.
    """

    def __init__(self, event_bus, config: Optional[Dict] = None):
        """
        Initialize funding stream.

        Args:
            event_bus: Event bus for publishing events
            config: Optional configuration
        """
        super().__init__(event_bus, config)

        # Configuration
        self.extreme_funding_threshold = self.config.get('extreme_funding_threshold', 50)

        # Data storage - keyed by (exchange, symbol)
        self.funding_rates: Dict[tuple, FundingData] = {}

        # Statistics
        self.stats.update({
            'total_funding_updates': 0,
            'extreme_funding_count': 0,
            'arbitrage_opportunities': 0
        })

    async def process(self, exchange: str, data: Dict) -> None:
        """
        Process funding rate data.

        Args:
            exchange: Exchange name
            data: Raw funding data
        """
        try:
            # Parse funding based on exchange
            if exchange.lower() == 'binance':
                funding = self._parse_binance_funding(data)
            elif exchange.lower() == 'hyperliquid':
                funding = self._parse_hyperliquid_funding(data)
            else:
                funding = self._parse_generic_funding(data)

            # Store funding data
            key = (funding.exchange, funding.symbol)
            self.funding_rates[key] = funding

            # Update statistics
            self.stats['total_funding_updates'] += 1

            # Check for extreme funding
            if funding.is_extreme:
                self.stats['extreme_funding_count'] += 1
                await self._emit_extreme_funding_alert(funding)

            # Regular funding update
            await self.emit_event("funding_update", {
                'symbol': funding.symbol,
                'exchange': funding.exchange,
                'rate': funding.funding_rate,
                'yearly_rate': funding.yearly_rate
            })

            # Check for funding arbitrage opportunities
            await self._check_funding_arbitrage(funding)

            self.stats['total_processed'] += 1

        except Exception as e:
            self._handle_error(e, "processing funding")

    def _parse_binance_funding(self, data: Dict) -> FundingData:
        """Parse Binance funding data."""
        return FundingData(
            symbol=data.get('s', '').replace('USDT', ''),
            funding_rate=float(data.get('r', 0)),
            mark_price=float(data.get('p', 0)),
            index_price=float(data.get('i', 0)),
            next_funding_time=datetime.fromtimestamp(int(data.get('T', 0)) / 1000),
            timestamp=datetime.fromtimestamp(int(data.get('E', 0)) / 1000),
            exchange='binance'
        )

    def _parse_hyperliquid_funding(self, data: Dict) -> FundingData:
        """Parse HyperLiquid funding data."""
        return FundingData(
            symbol=data.get('coin', ''),
            funding_rate=float(data.get('funding', 0)),
            mark_price=float(data.get('mark_px', 0)),
            index_price=float(data.get('index_px', 0)),
            next_funding_time=datetime.fromtimestamp(data.get('next_funding', 0)),
            timestamp=datetime.now(),
            exchange='hyperliquid'
        )

    def _parse_generic_funding(self, data: Dict) -> FundingData:
        """Parse generic funding data."""
        return FundingData(
            symbol=data.get('symbol', ''),
            funding_rate=float(data.get('funding_rate', 0)),
            mark_price=float(data.get('mark_price', 0)),
            index_price=float(data.get('index_price', 0)),
            next_funding_time=datetime.fromtimestamp(data.get('next_funding_time', 0)),
            timestamp=datetime.fromtimestamp(data.get('timestamp', 0)),
            exchange=data.get('exchange', 'unknown')
        )

    async def _emit_extreme_funding_alert(self, funding: FundingData) -> None:
        """Emit alert for extreme funding rates."""
        await self.emit_event("extreme_funding", {
            'funding': funding.__dict__,
            'yearly_rate': funding.yearly_rate,
            'alert_level': 'warning'
        })
        self.logger.warning(
            f"EXTREME FUNDING: {funding.exchange} {funding.symbol} "
            f"{funding.yearly_rate:.2f}% yearly"
        )

    async def _check_funding_arbitrage(self, funding: FundingData) -> None:
        """Check for funding arbitrage opportunities across exchanges."""
        # Compare with same symbol on other exchanges
        for (exchange, symbol), other_funding in self.funding_rates.items():
            if symbol == funding.symbol and exchange != funding.exchange:
                rate_diff = abs(funding.funding_rate - other_funding.funding_rate)

                # If difference is significant (>0.1%)
                if rate_diff > 0.001:
                    self.stats['arbitrage_opportunities'] += 1

                    await self.emit_event("funding_arbitrage", {
                        'symbol': funding.symbol,
                        'exchange1': funding.exchange,
                        'rate1': funding.funding_rate,
                        'yearly_rate1': funding.yearly_rate,
                        'exchange2': other_funding.exchange,
                        'rate2': other_funding.funding_rate,
                        'yearly_rate2': other_funding.yearly_rate,
                        'difference': rate_diff,
                        'yearly_difference': rate_diff * 3 * 365 * 100
                    })

                    self.logger.info(
                        f"FUNDING ARBITRAGE: {funding.symbol} - "
                        f"{funding.exchange} {funding.yearly_rate:.2f}% vs "
                        f"{other_funding.exchange} {other_funding.yearly_rate:.2f}%"
                    )

    def get_funding_rate(self, symbol: str, exchange: Optional[str] = None) -> Optional[FundingData]:
        """
        Get current funding rate for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Optional specific exchange (if None, returns first match)

        Returns:
            FundingData if found, None otherwise
        """
        if exchange:
            key = (exchange, symbol)
            return self.funding_rates.get(key)
        else:
            # Return first match for symbol
            for (ex, sym), funding in self.funding_rates.items():
                if sym == symbol:
                    return funding
            return None

    def get_extreme_funding_rates(self) -> Dict[str, FundingData]:
        """
        Get all symbols with extreme funding rates.

        Returns:
            Dictionary of extreme funding rates keyed by (exchange, symbol)
        """
        return {
            key: funding for key, funding in self.funding_rates.items()
            if funding.is_extreme
        }

    def get_statistics(self) -> Dict:
        """Get funding stream statistics."""
        base_stats = super().get_statistics()

        return {
            **base_stats,
            'monitored_symbols': len(self.funding_rates),
            'extreme_funding_symbols': len(self.get_extreme_funding_rates())
        }
