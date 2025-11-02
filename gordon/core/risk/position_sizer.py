"""
Position Sizer
==============
Calculates optimal position sizes based on risk management rules.
"""

from typing import Optional
from .base_manager import BaseRiskManager


class PositionSizer(BaseRiskManager):
    """
    Handles position sizing calculations using various methods:
    - Kelly criterion-based sizing
    - Fixed percentage risk
    - Volatility-adjusted sizing
    """

    def __init__(self, event_bus, config_manager, demo_mode: bool = False):
        """
        Initialize position sizer.

        Args:
            event_bus: Event bus for communication
            config_manager: Configuration manager
            demo_mode: Whether to run in demo mode
        """
        super().__init__(event_bus, config_manager, demo_mode)

        # Position sizing parameters
        self.risk_per_trade_percent = self.risk_config.get("risk_per_trade_percent", 2)
        self.max_position_size = self.risk_config.get("max_position_size", 10000)
        self.max_leverage = self.risk_config.get("max_leverage", 10)
        self.max_risk_per_position = self.risk_config.get("max_risk_per_position", 1000)

    def calculate_position_size(self, balance: float, stop_loss_percent: float) -> float:
        """
        Calculate position size based on risk management rules.

        Uses Kelly criterion-based sizing with leverage and absolute limits.

        Args:
            balance: Account balance
            stop_loss_percent: Stop loss percentage

        Returns:
            Position size in dollars
        """
        # Calculate risk amount based on percentage of balance
        risk_amount = balance * (self.risk_per_trade_percent / 100)

        # Calculate position size to match risk amount with stop loss
        position_size = risk_amount / (stop_loss_percent / 100)

        # Apply leverage limit
        max_with_leverage = balance * self.max_leverage
        position_size = min(position_size, max_with_leverage)

        # Apply absolute position size limit
        position_size = min(position_size, self.max_position_size)

        # Apply absolute risk limit
        position_size = min(position_size, self.max_risk_per_position)

        self.logger.info(
            f"Position size calculated: ${position_size:.2f} "
            f"(Balance: ${balance:.2f}, Risk: {self.risk_per_trade_percent}%, "
            f"Stop Loss: {stop_loss_percent}%)"
        )

        return position_size

    def calculate_position_size_with_volatility(
        self,
        balance: float,
        stop_loss_percent: float,
        volatility: float
    ) -> float:
        """
        Calculate position size adjusted for volatility.

        Higher volatility results in smaller position sizes.

        Args:
            balance: Account balance
            stop_loss_percent: Stop loss percentage
            volatility: Volatility measure (0-1 scale)

        Returns:
            Volatility-adjusted position size
        """
        base_size = self.calculate_position_size(balance, stop_loss_percent)

        # Reduce size based on volatility (higher volatility = smaller size)
        volatility_adjustment = 1 / (1 + volatility)
        adjusted_size = base_size * volatility_adjustment

        self.logger.info(
            f"Volatility-adjusted position size: ${adjusted_size:.2f} "
            f"(Base: ${base_size:.2f}, Volatility: {volatility:.2%})"
        )

        return adjusted_size

    def calculate_quantity_from_size(
        self,
        position_size: float,
        price: float,
        precision: int = 8
    ) -> float:
        """
        Convert position size in dollars to quantity of asset.

        Args:
            position_size: Position size in dollars
            price: Current price
            precision: Decimal precision for quantity

        Returns:
            Quantity of asset to trade
        """
        if price <= 0:
            self.logger.error("Invalid price for quantity calculation")
            return 0

        quantity = position_size / price
        quantity = round(quantity, precision)

        self.logger.debug(
            f"Quantity calculated: {quantity} "
            f"(Size: ${position_size:.2f}, Price: ${price:.2f})"
        )

        return quantity

    def check_position_size_allowed(
        self,
        position_value: float,
        current_exposure: float = 0
    ) -> bool:
        """
        Check if a position size is within allowed limits.

        Args:
            position_value: Value of the position
            current_exposure: Current total exposure

        Returns:
            True if position is allowed, False otherwise
        """
        # Check absolute position size
        if position_value > self.max_position_size:
            self.logger.warning(
                f"Position too large: ${position_value:.2f} > ${self.max_position_size:.2f}"
            )
            return False

        # Check risk limit
        if position_value > self.max_risk_per_position:
            self.logger.warning(
                f"Position exceeds max risk: ${position_value:.2f} > ${self.max_risk_per_position:.2f}"
            )
            return False

        return True

    def get_sizing_parameters(self) -> dict:
        """
        Get current position sizing parameters.

        Returns:
            Dictionary of sizing parameters
        """
        return {
            "risk_per_trade_percent": self.risk_per_trade_percent,
            "max_position_size": self.max_position_size,
            "max_leverage": self.max_leverage,
            "max_risk_per_position": self.max_risk_per_position
        }

    def update_sizing_parameters(self, params: dict):
        """
        Update position sizing parameters.

        Args:
            params: Dictionary of parameters to update
        """
        if "risk_per_trade_percent" in params:
            self.risk_per_trade_percent = params["risk_per_trade_percent"]

        if "max_position_size" in params:
            self.max_position_size = params["max_position_size"]

        if "max_leverage" in params:
            self.max_leverage = params["max_leverage"]

        if "max_risk_per_position" in params:
            self.max_risk_per_position = params["max_risk_per_position"]

        self.logger.info(f"Position sizing parameters updated: {params}")
