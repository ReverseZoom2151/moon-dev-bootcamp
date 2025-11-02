"""
Leverage Manager
================
Manages leverage limits and controls across trading operations.
"""

from typing import Dict, Optional
from .base_manager import BaseRiskManager


class LeverageManager(BaseRiskManager):
    """
    Handles leverage management:
    - Setting and enforcing leverage limits
    - Calculating effective leverage
    - Dynamic leverage adjustment based on market conditions
    """

    def __init__(self, event_bus, config_manager, demo_mode: bool = False):
        """
        Initialize leverage manager.

        Args:
            event_bus: Event bus for communication
            config_manager: Configuration manager
            demo_mode: Whether to run in demo mode
        """
        super().__init__(event_bus, config_manager, demo_mode)

        # Leverage parameters
        self.max_leverage = self.risk_config.get("max_leverage", 10)
        self.default_leverage = self.risk_config.get("default_leverage", 5)
        self.min_leverage = self.risk_config.get("min_leverage", 1)

        # Track leverage by exchange and symbol
        self.leverage_settings = {}

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price based on risk parameters.

        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            volatility: Optional volatility measure for adjustment

        Returns:
            Stop loss price
        """
        # Default stop loss percentage
        stop_loss_percent = self.risk_config.get("stop_loss_percent", 5)

        # Adjust for volatility if provided
        if volatility:
            # Higher volatility = wider stop (up to 10%)
            stop_loss_percent = min(stop_loss_percent * (1 + volatility), 10)

        if side.lower() == "buy":
            stop_loss = entry_price * (1 - stop_loss_percent / 100)
        else:
            stop_loss = entry_price * (1 + stop_loss_percent / 100)

        self.logger.info(
            f"Stop loss calculated: ${stop_loss:.2f} "
            f"(Entry: ${entry_price:.2f}, Side: {side}, Stop %: {stop_loss_percent:.2f}%)"
        )

        return stop_loss

    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take profit price based on risk/reward ratio.

        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            risk_reward_ratio: Risk/reward ratio (default 2:1)

        Returns:
            Take profit price
        """
        take_profit_percent = self.risk_config.get("take_profit_percent", 10)

        # Adjust based on risk/reward ratio
        take_profit_percent = take_profit_percent * risk_reward_ratio / 2

        if side.lower() == "buy":
            take_profit = entry_price * (1 + take_profit_percent / 100)
        else:
            take_profit = entry_price * (1 - take_profit_percent / 100)

        self.logger.info(
            f"Take profit calculated: ${take_profit:.2f} "
            f"(Entry: ${entry_price:.2f}, Side: {side}, Target %: {take_profit_percent:.2f}%)"
        )

        return take_profit

    def calculate_effective_leverage(
        self,
        position_size: float,
        balance: float
    ) -> float:
        """
        Calculate effective leverage of a position.

        Args:
            position_size: Position size in dollars
            balance: Account balance

        Returns:
            Effective leverage ratio
        """
        if balance <= 0:
            return 0

        leverage = position_size / balance

        self.logger.debug(
            f"Effective leverage: {leverage:.2f}x "
            f"(Position: ${position_size:.2f}, Balance: ${balance:.2f})"
        )

        return leverage

    def check_leverage_allowed(
        self,
        position_size: float,
        balance: float
    ) -> bool:
        """
        Check if position leverage is within allowed limits.

        Args:
            position_size: Position size in dollars
            balance: Account balance

        Returns:
            True if leverage is allowed, False otherwise
        """
        effective_leverage = self.calculate_effective_leverage(position_size, balance)

        if effective_leverage > self.max_leverage:
            self.logger.warning(
                f"Leverage too high: {effective_leverage:.2f}x > {self.max_leverage}x"
            )
            return False

        return True

    def get_max_position_size_for_leverage(
        self,
        balance: float,
        leverage: Optional[float] = None
    ) -> float:
        """
        Calculate maximum position size for given leverage.

        Args:
            balance: Account balance
            leverage: Desired leverage (uses max if not specified)

        Returns:
            Maximum position size
        """
        target_leverage = leverage or self.max_leverage
        max_size = balance * target_leverage

        self.logger.debug(
            f"Max position size for {target_leverage}x leverage: ${max_size:.2f}"
        )

        return max_size

    def adjust_leverage_for_volatility(
        self,
        base_leverage: float,
        volatility: float
    ) -> float:
        """
        Adjust leverage based on market volatility.

        Higher volatility should result in lower leverage.

        Args:
            base_leverage: Base leverage setting
            volatility: Volatility measure (0-1 scale)

        Returns:
            Adjusted leverage
        """
        # Reduce leverage as volatility increases
        adjustment_factor = 1 / (1 + volatility)
        adjusted_leverage = base_leverage * adjustment_factor

        # Ensure it stays within bounds
        adjusted_leverage = max(self.min_leverage, adjusted_leverage)
        adjusted_leverage = min(self.max_leverage, adjusted_leverage)

        self.logger.info(
            f"Leverage adjusted for volatility: {adjusted_leverage:.2f}x "
            f"(Base: {base_leverage:.2f}x, Volatility: {volatility:.2%})"
        )

        return adjusted_leverage

    def set_leverage_for_symbol(
        self,
        exchange: str,
        symbol: str,
        leverage: float
    ):
        """
        Set leverage for a specific exchange/symbol pair.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            leverage: Desired leverage
        """
        key = f"{exchange}:{symbol}"

        # Validate leverage is within bounds
        if leverage < self.min_leverage or leverage > self.max_leverage:
            self.logger.warning(
                f"Leverage {leverage}x outside bounds [{self.min_leverage}-{self.max_leverage}], "
                f"clamping to limits"
            )
            leverage = max(self.min_leverage, min(self.max_leverage, leverage))

        self.leverage_settings[key] = leverage
        self.logger.info(f"Leverage set for {key}: {leverage}x")

    def get_leverage_for_symbol(
        self,
        exchange: str,
        symbol: str
    ) -> float:
        """
        Get leverage setting for a specific exchange/symbol pair.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Leverage setting (or default if not set)
        """
        key = f"{exchange}:{symbol}"
        return self.leverage_settings.get(key, self.default_leverage)

    def get_leverage_settings(self) -> Dict:
        """
        Get all current leverage settings.

        Returns:
            Dictionary of leverage parameters and settings
        """
        return {
            "max_leverage": self.max_leverage,
            "default_leverage": self.default_leverage,
            "min_leverage": self.min_leverage,
            "symbol_settings": self.leverage_settings.copy()
        }

    def update_leverage_limits(self, max_leverage: float, default_leverage: Optional[float] = None):
        """
        Update global leverage limits.

        Args:
            max_leverage: New maximum leverage
            default_leverage: New default leverage (optional)
        """
        self.max_leverage = max_leverage

        if default_leverage is not None:
            self.default_leverage = min(default_leverage, max_leverage)

        self.logger.info(
            f"Leverage limits updated: Max={self.max_leverage}x, Default={self.default_leverage}x"
        )
