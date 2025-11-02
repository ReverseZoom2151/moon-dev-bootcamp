"""
Drawdown Calculator
===================
Tracks and calculates drawdown metrics for risk management.
"""

from typing import Dict, Optional
from datetime import datetime
from .base_manager import BaseRiskManager


class DrawdownCalculator(BaseRiskManager):
    """
    Handles drawdown tracking and monitoring:
    - Current drawdown calculation
    - Maximum drawdown tracking
    - Drawdown alerts and thresholds
    - Peak balance tracking
    """

    def __init__(self, event_bus, config_manager, demo_mode: bool = False):
        """
        Initialize drawdown calculator.

        Args:
            event_bus: Event bus for communication
            config_manager: Configuration manager
            demo_mode: Whether to run in demo mode
        """
        super().__init__(event_bus, config_manager, demo_mode)

        # Drawdown parameters
        self.max_drawdown_percent = self.risk_config.get("max_drawdown_percent", 20)
        self.drawdown_warning_percent = self.max_drawdown_percent * 0.8

        # Tracking
        self.peak_balance = 0
        self.current_balance = 0
        self.max_drawdown_seen = 0
        self.drawdown_history = []

    def update_balance(self, balance: float):
        """
        Update current balance and recalculate drawdown.

        Args:
            balance: New account balance
        """
        self.current_balance = balance

        # Update peak if we've reached a new high
        if balance > self.peak_balance:
            old_peak = self.peak_balance
            self.peak_balance = balance
            self.logger.info(f"New peak balance: ${balance:.2f} (previous: ${old_peak:.2f})")

        # Calculate and log current drawdown
        current_dd = self.calculate_current_drawdown()
        if current_dd > 0:
            self.logger.debug(f"Current drawdown: {current_dd:.2f}%")

    def calculate_current_drawdown(self) -> float:
        """
        Calculate current drawdown percentage.

        Returns:
            Current drawdown as percentage (0-100)
        """
        if self.peak_balance <= 0:
            return 0

        drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100

        # Track maximum drawdown seen
        if drawdown > self.max_drawdown_seen:
            self.max_drawdown_seen = drawdown
            self.logger.info(f"New maximum drawdown: {drawdown:.2f}%")

        return drawdown

    def is_drawdown_exceeded(self) -> bool:
        """
        Check if current drawdown exceeds maximum allowed.

        Returns:
            True if drawdown limit exceeded, False otherwise
        """
        current_dd = self.calculate_current_drawdown()
        return current_dd > self.max_drawdown_percent

    def is_drawdown_warning(self) -> bool:
        """
        Check if current drawdown is in warning zone.

        Returns:
            True if drawdown warning threshold reached, False otherwise
        """
        current_dd = self.calculate_current_drawdown()
        return current_dd > self.drawdown_warning_percent

    async def check_drawdown_alerts(self):
        """
        Check drawdown levels and emit alerts if necessary.
        """
        if self.peak_balance <= 0:
            return

        current_dd = self.calculate_current_drawdown()

        # Critical alert - drawdown limit exceeded
        if current_dd > self.max_drawdown_percent:
            await self.emit_event("drawdown_alert", {
                "level": "critical",
                "drawdown": current_dd,
                "limit": self.max_drawdown_percent,
                "current_balance": self.current_balance,
                "peak_balance": self.peak_balance,
                "timestamp": datetime.now().isoformat()
            })

            self.logger.error(
                f"CRITICAL: Drawdown limit exceeded: {current_dd:.2f}% > {self.max_drawdown_percent}%"
            )

        # Warning alert - approaching drawdown limit
        elif current_dd > self.drawdown_warning_percent:
            await self.emit_event("drawdown_alert", {
                "level": "warning",
                "drawdown": current_dd,
                "limit": self.max_drawdown_percent,
                "current_balance": self.current_balance,
                "peak_balance": self.peak_balance,
                "timestamp": datetime.now().isoformat()
            })

            self.logger.warning(
                f"WARNING: Approaching drawdown limit: {current_dd:.2f}% "
                f"(limit: {self.max_drawdown_percent}%)"
            )

    def calculate_recovery_needed(self) -> float:
        """
        Calculate percentage gain needed to recover from current drawdown.

        Returns:
            Percentage gain needed to reach peak balance
        """
        if self.current_balance <= 0:
            return float('inf')

        recovery_percent = ((self.peak_balance - self.current_balance) / self.current_balance) * 100

        return recovery_percent

    def get_drawdown_metrics(self) -> Dict:
        """
        Get comprehensive drawdown metrics.

        Returns:
            Dictionary of drawdown statistics
        """
        current_dd = self.calculate_current_drawdown()
        recovery_needed = self.calculate_recovery_needed()

        return {
            "current_drawdown_percent": current_dd,
            "max_drawdown_percent": self.max_drawdown_percent,
            "max_drawdown_seen": self.max_drawdown_seen,
            "peak_balance": self.peak_balance,
            "current_balance": self.current_balance,
            "recovery_needed_percent": recovery_needed,
            "is_warning": current_dd > self.drawdown_warning_percent,
            "is_critical": current_dd > self.max_drawdown_percent,
            "drawdown_allowance": self.max_drawdown_percent - current_dd
        }

    def reset_peak_balance(self, new_peak: Optional[float] = None):
        """
        Reset peak balance (e.g., for new trading period).

        Args:
            new_peak: New peak balance (uses current balance if None)
        """
        old_peak = self.peak_balance
        self.peak_balance = new_peak or self.current_balance

        self.logger.info(
            f"Peak balance reset: ${self.peak_balance:.2f} (previous: ${old_peak:.2f})"
        )

    def add_to_history(self):
        """
        Record current drawdown in history for analysis.
        """
        current_dd = self.calculate_current_drawdown()

        self.drawdown_history.append({
            "timestamp": datetime.now().isoformat(),
            "drawdown_percent": current_dd,
            "balance": self.current_balance,
            "peak": self.peak_balance
        })

        # Keep only last 1000 entries
        if len(self.drawdown_history) > 1000:
            self.drawdown_history = self.drawdown_history[-1000:]

    def get_drawdown_statistics(self) -> Dict:
        """
        Get statistical analysis of drawdown history.

        Returns:
            Dictionary of drawdown statistics
        """
        if not self.drawdown_history:
            return {
                "count": 0,
                "avg_drawdown": 0,
                "max_drawdown": 0,
                "min_drawdown": 0
            }

        drawdowns = [entry["drawdown_percent"] for entry in self.drawdown_history]

        return {
            "count": len(drawdowns),
            "avg_drawdown": sum(drawdowns) / len(drawdowns),
            "max_drawdown": max(drawdowns),
            "min_drawdown": min(drawdowns),
            "current_drawdown": drawdowns[-1] if drawdowns else 0
        }

    def update_max_drawdown_limit(self, new_limit: float):
        """
        Update maximum drawdown limit.

        Args:
            new_limit: New maximum drawdown percentage
        """
        old_limit = self.max_drawdown_percent
        self.max_drawdown_percent = new_limit
        self.drawdown_warning_percent = new_limit * 0.8

        self.logger.info(
            f"Drawdown limit updated: {new_limit:.2f}% (previous: {old_limit:.2f}%)"
        )
