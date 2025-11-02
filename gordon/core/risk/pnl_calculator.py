"""
PnL Calculator
==============
Calculates and tracks Profit & Loss metrics.
"""

from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from .base_manager import BaseRiskManager


class PnLCalculator(BaseRiskManager):
    """
    Handles PnL calculations and tracking:
    - Real-time PnL calculation
    - Daily/weekly/monthly PnL tracking
    - Position PnL monitoring
    - PnL-based exit triggers
    """

    def __init__(self, event_bus, config_manager, demo_mode: bool = False):
        """
        Initialize PnL calculator.

        Args:
            event_bus: Event bus for communication
            config_manager: Configuration manager
            demo_mode: Whether to run in demo mode
        """
        super().__init__(event_bus, config_manager, demo_mode)

        # PnL tracking
        self.daily_pnl = defaultdict(float)  # date -> pnl
        self.position_pnls = {}  # position_id -> pnl
        self.closed_trades = []

        # PnL targets and limits
        self.default_target_percent = self.risk_config.get("default_target_percent", 9)
        self.default_max_loss_percent = self.risk_config.get("default_max_loss_percent", -8)
        self.daily_loss_limit = self.risk_config.get("daily_loss_limit", 500)

    async def get_position_details(
        self,
        exchange: str,
        symbol: str
    ) -> Tuple[bool, float, float, float, bool]:
        """
        Get detailed position information including PnL.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Tuple of (has_position, size, entry_price, pnl_percent, is_long)
        """
        if self.demo_mode:
            # Return demo position data
            return True, 0.1, 42000, 5.2, True

        try:
            if exchange not in self.exchange_connections:
                self.logger.warning(f"No exchange connection for {exchange}")
                return False, 0, 0, 0, False

            conn = self.exchange_connections[exchange]

            # Fetch positions from exchange
            positions = await conn.fetch_positions([symbol])

            if not positions or positions[0]['contracts'] == 0:
                return False, 0, 0, 0, False

            pos = positions[0]
            size = pos['contracts']
            entry_price = pos.get('entryPrice', 0)
            unrealized_pnl = pos.get('unrealizedPnl', 0)

            # Calculate PnL percentage
            entry_value = entry_price * size if entry_price and size else 1
            pnl_percent = (unrealized_pnl / entry_value) * 100 if entry_value != 0 else 0

            is_long = pos.get('side', '').lower() == 'long'

            self.logger.info(
                f"Position {symbol} on {exchange}: Size={size}, "
                f"Entry=${entry_price:.2f}, PnL={pnl_percent:.2f}%"
            )

            return True, size, entry_price, pnl_percent, is_long

        except Exception as e:
            self.logger.error(f"Error getting position details for {symbol} on {exchange}: {e}")
            return False, 0, 0, 0, False

    def calculate_position_pnl(
        self,
        entry_price: float,
        current_price: float,
        size: float,
        is_long: bool
    ) -> Tuple[float, float]:
        """
        Calculate position PnL.

        Args:
            entry_price: Entry price
            current_price: Current market price
            size: Position size
            is_long: Whether position is long

        Returns:
            Tuple of (pnl_dollars, pnl_percent)
        """
        if is_long:
            pnl_dollars = (current_price - entry_price) * size
        else:
            pnl_dollars = (entry_price - current_price) * size

        entry_value = entry_price * size
        pnl_percent = (pnl_dollars / entry_value) * 100 if entry_value != 0 else 0

        return pnl_dollars, pnl_percent

    async def check_pnl_limits(
        self,
        exchange: str,
        symbol: str,
        target_percent: Optional[float] = None,
        max_loss_percent: Optional[float] = None
    ) -> bool:
        """
        Check if position PnL exceeds limits and needs closing.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            target_percent: Take profit target (uses default if None)
            max_loss_percent: Stop loss limit (uses default if None)

        Returns:
            True if position should be closed, False otherwise
        """
        target = target_percent or self.default_target_percent
        max_loss = max_loss_percent or self.default_max_loss_percent

        has_position, size, entry_price, pnl_percent, is_long = await self.get_position_details(
            exchange, symbol
        )

        if not has_position:
            return False

        # Check if PnL exceeds limits
        if pnl_percent >= target or pnl_percent <= max_loss:
            reason = "target_reached" if pnl_percent >= target else "stop_loss"

            self.logger.warning(
                f"PnL limit hit for {symbol}: {pnl_percent:.2f}% "
                f"(target={target}%, max_loss={max_loss}%) - {reason}"
            )

            # Emit kill switch event
            await self.emit_event("kill_switch_triggered", {
                "exchange": exchange,
                "symbol": symbol,
                "reason": reason,
                "pnl_percent": pnl_percent,
                "size": size,
                "is_long": is_long,
                "entry_price": entry_price
            })

            return True

        return False

    def update_daily_pnl(self, pnl: float, date: Optional[datetime] = None):
        """
        Update daily PnL tracking.

        Args:
            pnl: PnL to add
            date: Date for PnL (uses today if None)
        """
        target_date = (date or datetime.now()).date()
        self.daily_pnl[target_date] += pnl

        self.logger.info(
            f"Daily PnL updated for {target_date}: ${self.daily_pnl[target_date]:.2f}"
        )

    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """
        Get PnL for a specific date.

        Args:
            date: Date to query (uses today if None)

        Returns:
            PnL for the date
        """
        target_date = (date or datetime.now()).date()
        return self.daily_pnl[target_date]

    def is_daily_loss_limit_exceeded(self) -> bool:
        """
        Check if daily loss limit is exceeded.

        Returns:
            True if limit exceeded, False otherwise
        """
        today = datetime.now().date()
        return self.daily_pnl[today] <= -self.daily_loss_limit

    async def check_daily_loss_alert(self):
        """
        Check daily loss and emit alert if necessary.
        """
        today = datetime.now().date()
        current_loss = abs(self.daily_pnl[today])

        # Alert at 80% of limit
        if current_loss >= self.daily_loss_limit * 0.8:
            level = "critical" if current_loss >= self.daily_loss_limit else "warning"

            await self.emit_event("daily_loss_alert", {
                "level": level,
                "loss": self.daily_pnl[today],
                "limit": self.daily_loss_limit,
                "date": today.isoformat()
            })

            if level == "critical":
                self.logger.error(
                    f"Daily loss limit exceeded: ${current_loss:.2f} >= ${self.daily_loss_limit:.2f}"
                )
            else:
                self.logger.warning(
                    f"Approaching daily loss limit: ${current_loss:.2f} "
                    f"(limit: ${self.daily_loss_limit:.2f})"
                )

    def record_closed_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        size: float,
        is_long: bool,
        pnl: float
    ):
        """
        Record a closed trade for analysis.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            is_long: Whether position was long
            pnl: Realized PnL
        """
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "is_long": is_long,
            "pnl": pnl,
            "pnl_percent": (pnl / (entry_price * size)) * 100 if entry_price * size != 0 else 0
        }

        self.closed_trades.append(trade_record)

        # Keep only last 1000 trades
        if len(self.closed_trades) > 1000:
            self.closed_trades = self.closed_trades[-1000:]

        # Update daily PnL
        self.update_daily_pnl(pnl)

        self.logger.info(
            f"Trade closed: {symbol} - PnL: ${pnl:.2f} ({trade_record['pnl_percent']:.2f}%)"
        )

    def get_pnl_metrics(self) -> Dict:
        """
        Get comprehensive PnL metrics.

        Returns:
            Dictionary of PnL statistics
        """
        today = datetime.now().date()

        # Calculate weekly and monthly PnL
        week_start = today - timedelta(days=today.weekday())
        month_start = today.replace(day=1)

        weekly_pnl = sum(
            pnl for date, pnl in self.daily_pnl.items()
            if date >= week_start
        )

        monthly_pnl = sum(
            pnl for date, pnl in self.daily_pnl.items()
            if date >= month_start
        )

        # Calculate trade statistics
        if self.closed_trades:
            winning_trades = [t for t in self.closed_trades if t["pnl"] > 0]
            losing_trades = [t for t in self.closed_trades if t["pnl"] < 0]

            win_rate = len(winning_trades) / len(self.closed_trades) * 100
            avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0

        return {
            "daily_pnl": self.daily_pnl[today],
            "weekly_pnl": weekly_pnl,
            "monthly_pnl": monthly_pnl,
            "daily_loss_limit": self.daily_loss_limit,
            "total_closed_trades": len(self.closed_trades),
            "winning_trades": len([t for t in self.closed_trades if t["pnl"] > 0]),
            "losing_trades": len([t for t in self.closed_trades if t["pnl"] < 0]),
            "win_rate_percent": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }

    def reset_daily_pnl(self):
        """
        Reset daily PnL (typically called at start of new trading day).
        """
        yesterday = (datetime.now() - timedelta(days=1)).date()
        if yesterday in self.daily_pnl:
            self.logger.info(f"Yesterday's PnL: ${self.daily_pnl[yesterday]:.2f}")

        # Reset for today
        today = datetime.now().date()
        self.daily_pnl[today] = 0

        self.logger.info("Daily PnL reset for new trading day")

    def get_pnl_targets(self) -> Dict:
        """
        Get current PnL target and loss limit settings.

        Returns:
            Dictionary of PnL thresholds
        """
        return {
            "target_percent": self.default_target_percent,
            "max_loss_percent": self.default_max_loss_percent,
            "daily_loss_limit": self.daily_loss_limit
        }

    def update_pnl_targets(
        self,
        target_percent: Optional[float] = None,
        max_loss_percent: Optional[float] = None,
        daily_loss_limit: Optional[float] = None
    ):
        """
        Update PnL target and loss limit settings.

        Args:
            target_percent: New target profit percentage
            max_loss_percent: New max loss percentage
            daily_loss_limit: New daily loss limit
        """
        if target_percent is not None:
            self.default_target_percent = target_percent

        if max_loss_percent is not None:
            self.default_max_loss_percent = max_loss_percent

        if daily_loss_limit is not None:
            self.daily_loss_limit = daily_loss_limit

        self.logger.info(
            f"PnL targets updated: Target={self.default_target_percent}%, "
            f"MaxLoss={self.default_max_loss_percent}%, DailyLimit=${self.daily_loss_limit:.2f}"
        )
