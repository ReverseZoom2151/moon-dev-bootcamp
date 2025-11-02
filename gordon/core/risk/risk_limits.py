"""
Risk Limits
===========
Manages risk limits and thresholds for trading operations.
"""

import asyncio
from typing import Dict, Optional
from datetime import datetime
from .base_manager import BaseRiskManager


class RiskLimits(BaseRiskManager):
    """
    Handles risk limit enforcement:
    - Maximum positions
    - Position size limits
    - Kill switches
    - Trading permissions
    - Emergency stops
    """

    def __init__(self, event_bus, config_manager, demo_mode: bool = False):
        """
        Initialize risk limits manager.

        Args:
            event_bus: Event bus for communication
            config_manager: Configuration manager
            demo_mode: Whether to run in demo mode
        """
        super().__init__(event_bus, config_manager, demo_mode)

        # Risk limits
        self.max_positions = self.risk_config.get("max_positions", 5)
        self.max_position_size = self.risk_config.get("max_position_size", 10000)
        self.max_risk_per_position = self.risk_config.get("max_risk_per_position", 1000)

        # Trading state
        self.is_trading_allowed = True
        self.current_positions = 0

        # Kill switch tracking
        self.kill_switches_active = {}  # symbol -> reason

    async def check_trade_allowed(
        self,
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None
    ) -> bool:
        """
        Check if a trade is allowed based on all risk limits.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: Buy or sell
            amount: Trade amount
            price: Trade price

        Returns:
            Whether trade is allowed
        """
        # Check if trading is globally allowed
        if not self.is_trading_allowed:
            self.logger.warning("Trading is globally disabled")
            return False

        # Check if symbol has kill switch active
        if symbol in self.kill_switches_active:
            self.logger.warning(
                f"Kill switch active for {symbol}: {self.kill_switches_active[symbol]}"
            )
            return False

        # Check position count
        if self.current_positions >= self.max_positions:
            self.logger.warning(
                f"Max positions reached: {self.current_positions}/{self.max_positions}"
            )
            await self.emit_event("risk_limit_reached", {
                "type": "max_positions",
                "current": self.current_positions,
                "limit": self.max_positions
            })
            return False

        # Check position size
        if price:
            position_value = amount * price
            if position_value > self.max_position_size:
                self.logger.warning(
                    f"Position too large: ${position_value:.2f} > ${self.max_position_size:.2f}"
                )
                await self.emit_event("risk_limit_reached", {
                    "type": "position_size",
                    "size": position_value,
                    "limit": self.max_position_size
                })
                return False

        return True

    async def position_size_kill_switch(self, exchange: str):
        """
        Check positions and trigger kill switch if any exceed risk limits.

        Args:
            exchange: Exchange name
        """
        if self.demo_mode:
            self.logger.info("[DEMO] Position size kill switch check")
            return

        try:
            if exchange not in self.exchange_connections:
                return

            conn = self.exchange_connections[exchange]
            positions = await conn.fetch_positions()

            for pos in positions:
                notional = pos.get('notional', 0)
                if abs(notional) > self.max_risk_per_position:
                    symbol = pos['symbol']
                    self.logger.warning(
                        f"Position {symbol} exceeds max risk: "
                        f"${abs(notional):.2f} > ${self.max_risk_per_position:.2f}"
                    )

                    # Activate kill switch for this symbol
                    self.kill_switches_active[symbol] = "size_limit"

                    # Emit kill switch event
                    await self.emit_event("kill_switch_triggered", {
                        "exchange": exchange,
                        "symbol": symbol,
                        "reason": "size_limit",
                        "notional": notional,
                        "limit": self.max_risk_per_position
                    })

        except Exception as e:
            self.logger.error(f"Error in position size kill switch: {e}")

    async def execute_kill_switch(self, exchange: str, symbol: str, reason: str = "manual"):
        """
        Execute kill switch to close position immediately.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            reason: Reason for kill switch
        """
        if self.demo_mode:
            self.logger.info(f"[DEMO] Kill switch executed for {symbol}: {reason}")
            return True

        try:
            # Activate kill switch
            self.kill_switches_active[symbol] = reason

            # Emit order to close position
            await self.emit_event("close_position_emergency", {
                "exchange": exchange,
                "symbol": symbol,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            self.logger.warning(f"Kill switch executed for {symbol}: {reason}")
            return True

        except Exception as e:
            self.logger.error(f"Error executing kill switch for {symbol}: {e}")
            return False

    def clear_kill_switch(self, symbol: str):
        """
        Clear kill switch for a symbol.

        Args:
            symbol: Trading symbol
        """
        if symbol in self.kill_switches_active:
            reason = self.kill_switches_active.pop(symbol)
            self.logger.info(f"Kill switch cleared for {symbol} (was: {reason})")

    def disable_trading(self, reason: str = "manual"):
        """
        Disable all trading globally.

        Args:
            reason: Reason for disabling trading
        """
        self.is_trading_allowed = False
        self.logger.warning(f"Trading globally disabled: {reason}")

    def enable_trading(self):
        """
        Re-enable trading globally.
        """
        self.is_trading_allowed = True
        self.logger.info("Trading globally enabled")

    def increment_position_count(self):
        """
        Increment current position count.
        """
        self.current_positions += 1
        self.logger.debug(f"Position count: {self.current_positions}/{self.max_positions}")

    def decrement_position_count(self):
        """
        Decrement current position count.
        """
        self.current_positions = max(0, self.current_positions - 1)
        self.logger.debug(f"Position count: {self.current_positions}/{self.max_positions}")

    def get_risk_limits(self) -> Dict:
        """
        Get current risk limits and trading state.

        Returns:
            Dictionary of risk limits
        """
        return {
            "max_positions": self.max_positions,
            "current_positions": self.current_positions,
            "max_position_size": self.max_position_size,
            "max_risk_per_position": self.max_risk_per_position,
            "is_trading_allowed": self.is_trading_allowed,
            "active_kill_switches": len(self.kill_switches_active),
            "kill_switches": self.kill_switches_active.copy()
        }

    def update_risk_limits(
        self,
        max_positions: Optional[int] = None,
        max_position_size: Optional[float] = None,
        max_risk_per_position: Optional[float] = None
    ):
        """
        Update risk limit parameters.

        Args:
            max_positions: New maximum positions
            max_position_size: New maximum position size
            max_risk_per_position: New maximum risk per position
        """
        if max_positions is not None:
            self.max_positions = max_positions

        if max_position_size is not None:
            self.max_position_size = max_position_size

        if max_risk_per_position is not None:
            self.max_risk_per_position = max_risk_per_position

        self.logger.info(
            f"Risk limits updated: MaxPositions={self.max_positions}, "
            f"MaxSize=${self.max_position_size:.2f}, "
            f"MaxRisk=${self.max_risk_per_position:.2f}"
        )

    async def start_monitoring(self, exchanges: list, interval: int = 60):
        """
        Start continuous monitoring of risk limits.

        Args:
            exchanges: List of exchanges to monitor
            interval: Check interval in seconds
        """
        self.logger.info(f"Starting risk limits monitoring (interval: {interval}s)")

        while self.is_trading_allowed:
            try:
                for exchange in exchanges:
                    # Check position sizes
                    await self.position_size_kill_switch(exchange)

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in risk limits monitoring: {e}")
                await asyncio.sleep(interval)

    async def execute_gradual_close(
        self,
        exchange: str,
        symbol: str,
        total_size: float,
        chunks: int = 5,
        delay_seconds: int = 5
    ) -> bool:
        """
        Close position gradually in chunks.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            total_size: Total position size to close
            chunks: Number of chunks to split the close into
            delay_seconds: Delay between chunks

        Returns:
            Success status
        """
        if self.demo_mode:
            self.logger.info(f"[DEMO] Gradual close of {total_size} {symbol} in {chunks} chunks")
            return True

        try:
            chunk_size = total_size / chunks

            for i in range(chunks):
                self.logger.info(
                    f"Closing chunk {i+1}/{chunks}: {chunk_size} of {symbol}"
                )

                # Emit close order event
                await self.emit_event("close_position_chunk", {
                    "exchange": exchange,
                    "symbol": symbol,
                    "size": chunk_size,
                    "chunk": i + 1,
                    "total_chunks": chunks
                })

                if i < chunks - 1:
                    await asyncio.sleep(delay_seconds)

            self.logger.info(f"Gradual close completed for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error in gradual close for {symbol}: {e}")
            return False

    def get_active_kill_switches(self) -> Dict[str, str]:
        """
        Get all active kill switches.

        Returns:
            Dictionary mapping symbols to reasons
        """
        return self.kill_switches_active.copy()

    def clear_all_kill_switches(self):
        """
        Clear all active kill switches.
        """
        count = len(self.kill_switches_active)
        self.kill_switches_active.clear()
        self.logger.info(f"Cleared {count} kill switches")
