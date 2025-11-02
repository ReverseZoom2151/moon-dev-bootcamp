"""
Decimal Precision Mixin
========================
Provides decimal precision and rounding utilities for order amounts and prices.

Common patterns identified:
- All exchanges require proper rounding to avoid precision errors
- Price and amount precision varies by symbol
- Need to format values according to exchange requirements

This mixin provides methods to handle decimal precision consistently.
"""

from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Dict, Optional, Union


class DecimalPrecisionMixin:
    """
    Mixin class that provides decimal precision handling.

    This mixin adds methods to properly round and format prices and amounts
    according to exchange requirements.

    Usage:
    ------
        class MyExchange(DecimalPrecisionMixin, BaseExchange):
            async def place_order(self, symbol, side, amount, price):
                # Get precision info
                info = self.get_symbol_precision(symbol)

                # Round values appropriately
                amount = self.round_amount(amount, symbol)
                price = self.round_price(price, symbol)

                # Place order with rounded values...

    Attributes:
    -----------
        symbols_info: Dictionary containing symbol information including precision
    """

    def get_symbol_precision(self, symbol: str) -> Dict:
        """
        Get precision information for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with precision information:
            - price_precision: Number of decimal places for price
            - amount_precision: Number of decimal places for amount
            - min_amount: Minimum order amount
            - min_cost: Minimum order cost (amount * price)

        Raises:
            KeyError: If symbol info not found
        """
        if not hasattr(self, 'symbols_info'):
            return {
                'price_precision': 8,
                'amount_precision': 8,
                'min_amount': 0.0,
                'min_cost': 0.0
            }

        symbol_info = self.symbols_info.get(symbol, {})

        return {
            'price_precision': symbol_info.get('price_precision', 8),
            'amount_precision': symbol_info.get('amount_precision', 8),
            'min_amount': symbol_info.get('min_amount', 0.0),
            'min_cost': symbol_info.get('min_cost', 0.0)
        }

    def round_price(
        self,
        price: Union[float, Decimal],
        symbol: str,
        rounding: str = ROUND_HALF_UP
    ) -> float:
        """
        Round a price to the correct precision for the symbol.

        Args:
            price: Price to round
            symbol: Trading symbol
            rounding: Rounding method (ROUND_DOWN, ROUND_UP, ROUND_HALF_UP)

        Returns:
            Rounded price as float

        Examples:
        ---------
            # Symbol with 2 decimal price precision
            round_price(1.23456, 'BTC/USDT') -> 1.23

            # Symbol with 8 decimal price precision
            round_price(0.123456789, 'ETH/BTC') -> 0.12345679
        """
        precision_info = self.get_symbol_precision(symbol)
        precision = precision_info['price_precision']

        # Convert to Decimal for accurate rounding
        dec_price = Decimal(str(price))

        # Create quantizer based on precision
        quantizer = Decimal('0.1') ** precision

        # Round
        rounded = dec_price.quantize(quantizer, rounding=rounding)

        return float(rounded)

    def round_amount(
        self,
        amount: Union[float, Decimal],
        symbol: str,
        rounding: str = ROUND_DOWN
    ) -> float:
        """
        Round an amount to the correct precision for the symbol.

        Args:
            amount: Amount to round
            symbol: Trading symbol
            rounding: Rounding method (default ROUND_DOWN to avoid exceeding balance)

        Returns:
            Rounded amount as float

        Examples:
        ---------
            # Symbol with 3 decimal amount precision
            round_amount(0.123456, 'BTC/USDT') -> 0.123

            # Symbol with 0 decimal amount precision
            round_amount(1.5, 'SOMETHING/USDT') -> 1.0
        """
        precision_info = self.get_symbol_precision(symbol)
        precision = precision_info['amount_precision']

        # Convert to Decimal for accurate rounding
        dec_amount = Decimal(str(amount))

        # Create quantizer based on precision
        quantizer = Decimal('0.1') ** precision

        # Round
        rounded = dec_amount.quantize(quantizer, rounding=rounding)

        return float(rounded)

    def format_price(self, price: Union[float, Decimal], symbol: str) -> str:
        """
        Format a price as a string with correct precision.

        Args:
            price: Price to format
            symbol: Trading symbol

        Returns:
            Formatted price string

        Examples:
        ---------
            format_price(1.23456, 'BTC/USDT') -> '1.23'
            format_price(0.00012345, 'LOW/BTC') -> '0.00012345'
        """
        rounded_price = self.round_price(price, symbol)
        precision_info = self.get_symbol_precision(symbol)
        precision = precision_info['price_precision']

        return f"{rounded_price:.{precision}f}"

    def format_amount(self, amount: Union[float, Decimal], symbol: str) -> str:
        """
        Format an amount as a string with correct precision.

        Args:
            amount: Amount to format
            symbol: Trading symbol

        Returns:
            Formatted amount string

        Examples:
        ---------
            format_amount(0.123456, 'BTC/USDT') -> '0.123'
            format_amount(1.5, 'SOMETHING/USDT') -> '1'
        """
        rounded_amount = self.round_amount(amount, symbol)
        precision_info = self.get_symbol_precision(symbol)
        precision = precision_info['amount_precision']

        return f"{rounded_amount:.{precision}f}"

    def validate_min_amount(self, amount: float, symbol: str) -> bool:
        """
        Check if amount meets the minimum requirement.

        Args:
            amount: Amount to validate
            symbol: Trading symbol

        Returns:
            True if amount meets minimum, False otherwise
        """
        precision_info = self.get_symbol_precision(symbol)
        min_amount = precision_info['min_amount']

        return amount >= min_amount

    def validate_min_cost(self, amount: float, price: float, symbol: str) -> bool:
        """
        Check if order cost (amount * price) meets minimum requirement.

        Args:
            amount: Order amount
            price: Order price
            symbol: Trading symbol

        Returns:
            True if cost meets minimum, False otherwise
        """
        precision_info = self.get_symbol_precision(symbol)
        min_cost = precision_info['min_cost']

        cost = amount * price
        return cost >= min_cost

    def adjust_amount_to_min(self, amount: float, symbol: str) -> float:
        """
        Adjust amount to meet minimum if it's slightly below.

        Args:
            amount: Original amount
            symbol: Trading symbol

        Returns:
            Adjusted amount (either original if valid, or minimum)
        """
        precision_info = self.get_symbol_precision(symbol)
        min_amount = precision_info['min_amount']

        if amount < min_amount:
            return min_amount

        return amount

    def calculate_max_amount_from_cost(
        self,
        cost: float,
        price: float,
        symbol: str
    ) -> float:
        """
        Calculate maximum amount that can be bought with given cost.

        Args:
            cost: Total cost available
            price: Price per unit
            symbol: Trading symbol

        Returns:
            Maximum amount rounded to correct precision

        Examples:
        ---------
            # $1000 at $50000 per BTC
            calculate_max_amount_from_cost(1000, 50000, 'BTC/USDT') -> 0.02
        """
        if price <= 0:
            return 0.0

        raw_amount = cost / price
        return self.round_amount(raw_amount, symbol, rounding=ROUND_DOWN)

    def calculate_cost(
        self,
        amount: float,
        price: float,
        symbol: str,
        include_fees: bool = False,
        fee_rate: float = 0.001
    ) -> float:
        """
        Calculate total cost for an order.

        Args:
            amount: Order amount
            price: Order price
            symbol: Trading symbol
            include_fees: Whether to include trading fees
            fee_rate: Fee rate (default 0.1%)

        Returns:
            Total cost including fees if requested
        """
        base_cost = amount * price

        if include_fees:
            fee = base_cost * fee_rate
            return base_cost + fee

        return base_cost

    def step_size_round(
        self,
        value: Union[float, Decimal],
        step_size: Union[float, Decimal],
        rounding: str = ROUND_DOWN
    ) -> float:
        """
        Round a value to the nearest step size.

        Some exchanges use step sizes instead of decimal precision.

        Args:
            value: Value to round
            step_size: Step size (e.g., 0.01, 0.1, 1.0)
            rounding: Rounding method

        Returns:
            Rounded value

        Examples:
        ---------
            step_size_round(1.234, 0.1) -> 1.2
            step_size_round(1.234, 0.5) -> 1.0
        """
        if step_size == 0:
            return float(value)

        dec_value = Decimal(str(value))
        dec_step = Decimal(str(step_size))

        # Calculate number of steps
        steps = dec_value / dec_step

        # Round to nearest step
        rounded_steps = steps.quantize(Decimal('1'), rounding=rounding)

        # Multiply back
        result = rounded_steps * dec_step

        return float(result)

    def get_decimal_places(self, value: Union[float, Decimal, str]) -> int:
        """
        Get number of decimal places in a value.

        Args:
            value: Value to check

        Returns:
            Number of decimal places

        Examples:
        ---------
            get_decimal_places(1.23) -> 2
            get_decimal_places(0.00001) -> 5
            get_decimal_places(1.0) -> 0
        """
        dec_value = Decimal(str(value))
        return -dec_value.as_tuple().exponent if dec_value.as_tuple().exponent < 0 else 0
