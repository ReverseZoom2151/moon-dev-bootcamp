"""
Position Sizing Helpers
=======================
Day 44: Position sizing calculation utilities.
Helpers for leverage, position size, and risk-based sizing.
"""

import logging
from typing import Dict, Optional, Tuple
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)


class PositionSizingHelper:
    """
    Position sizing calculation helpers.
    
    Provides utilities for calculating position sizes based on:
    - Account balance percentage
    - Fixed USD amount
    - Risk percentage
    - Leverage
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize position sizing helper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default settings
        self.default_leverage = self.config.get('default_leverage', 1)
        self.max_leverage = self.config.get('max_leverage', 10)
        self.default_risk_percent = self.config.get('default_risk_percent', 0.02)  # 2%
        self.default_position_percent = self.config.get('default_position_percent', 0.95)  # 95% of balance
    
    def calculate_size_from_balance_percent(
        self,
        account_balance: float,
        price: float,
        leverage: int = 1,
        balance_percent: float = 0.95,
        size_decimals: int = 4
    ) -> Tuple[int, float]:
        """
        Calculate position size based on account balance percentage.
        
        Args:
            account_balance: Total account balance in USD
            price: Current price of asset
            leverage: Leverage multiplier
            balance_percent: Percentage of balance to use (0.95 = 95%)
            size_decimals: Decimal precision for size
            
        Returns:
            Tuple of (leverage, size)
        """
        try:
            # Calculate usable balance
            usable_balance = account_balance * balance_percent
            
            # Calculate size: (balance / price) * leverage
            size = (usable_balance / price) * leverage
            
            # Round to appropriate precision
            size = round(size, size_decimals)
            
            logger.info(f"Position size calculation:")
            logger.info(f"  Account balance: ${account_balance:,.2f}")
            logger.info(f"  Usable balance ({balance_percent*100}%): ${usable_balance:,.2f}")
            logger.info(f"  Price: ${price:,.2f}")
            logger.info(f"  Leverage: {leverage}x")
            logger.info(f"  Calculated size: {size}")
            
            return leverage, size
            
        except Exception as e:
            logger.error(f"Error calculating size from balance percent: {e}")
            return 1, 0.0
    
    def calculate_size_from_usd_amount(
        self,
        usd_amount: float,
        price: float,
        leverage: int = 1,
        size_decimals: int = 4
    ) -> Tuple[int, float]:
        """
        Calculate position size based on fixed USD amount.
        
        Args:
            usd_amount: Fixed USD amount to use
            price: Current price of asset
            leverage: Leverage multiplier
            size_decimals: Decimal precision for size
            
        Returns:
            Tuple of (leverage, size)
        """
        try:
            # Calculate size: (usd_amount / price) * leverage
            size = (usd_amount / price) * leverage
            
            # Round to appropriate precision
            size = round(size, size_decimals)
            
            logger.info(f"Position size calculation (USD amount):")
            logger.info(f"  USD amount: ${usd_amount:,.2f}")
            logger.info(f"  Price: ${price:,.2f}")
            logger.info(f"  Leverage: {leverage}x")
            logger.info(f"  Calculated size: {size}")
            
            return leverage, size
            
        except Exception as e:
            logger.error(f"Error calculating size from USD amount: {e}")
            return 1, 0.0
    
    def calculate_size_from_risk_percent(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        risk_percent: float = 0.02,
        leverage: int = 1,
        size_decimals: int = 4
    ) -> Tuple[int, float]:
        """
        Calculate position size based on risk percentage.
        
        Args:
            account_balance: Total account balance in USD
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_percent: Risk percentage per trade (0.02 = 2%)
            leverage: Leverage multiplier
            size_decimals: Decimal precision for size
            
        Returns:
            Tuple of (leverage, size)
        """
        try:
            # Calculate risk amount
            risk_amount = account_balance * risk_percent
            
            # Calculate price difference
            price_diff = abs(entry_price - stop_loss_price)
            
            if price_diff == 0:
                logger.warning("Entry price equals stop loss price")
                return 1, 0.0
            
            # Calculate size: risk_amount / price_diff
            # Then apply leverage
            base_size = risk_amount / price_diff
            size = base_size * leverage
            
            # Round to appropriate precision
            size = round(size, size_decimals)
            
            logger.info(f"Position size calculation (risk-based):")
            logger.info(f"  Account balance: ${account_balance:,.2f}")
            logger.info(f"  Risk amount ({risk_percent*100}%): ${risk_amount:,.2f}")
            logger.info(f"  Entry price: ${entry_price:,.2f}")
            logger.info(f"  Stop loss: ${stop_loss_price:,.2f}")
            logger.info(f"  Price difference: ${price_diff:,.2f}")
            logger.info(f"  Leverage: {leverage}x")
            logger.info(f"  Calculated size: {size}")
            
            return leverage, size
            
        except Exception as e:
            logger.error(f"Error calculating size from risk percent: {e}")
            return 1, 0.0
    
    def calculate_size_from_atr(
        self,
        account_balance: float,
        current_price: float,
        atr_value: float,
        risk_percent: float = 0.02,
        atr_multiplier: float = 2.0,
        leverage: int = 1,
        size_decimals: int = 4
    ) -> Tuple[int, float]:
        """
        Calculate position size using ATR for stop loss.
        
        Args:
            account_balance: Total account balance in USD
            current_price: Current price of asset
            atr_value: Current ATR value
            risk_percent: Risk percentage per trade
            atr_multiplier: Multiplier for ATR stop loss
            leverage: Leverage multiplier
            size_decimals: Decimal precision for size
            
        Returns:
            Tuple of (leverage, size)
        """
        try:
            # Calculate stop loss based on ATR
            stop_loss_distance = atr_value * atr_multiplier
            stop_loss_price = current_price - stop_loss_distance
            
            # Use risk-based calculation
            return self.calculate_size_from_risk_percent(
                account_balance=account_balance,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                risk_percent=risk_percent,
                leverage=leverage,
                size_decimals=size_decimals
            )
            
        except Exception as e:
            logger.error(f"Error calculating size from ATR: {e}")
            return 1, 0.0
    
    def validate_position_size(
        self,
        size: float,
        min_size: float = 0.001,
        max_size: Optional[float] = None,
        account_balance: Optional[float] = None,
        price: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate position size against limits.
        
        Args:
            size: Position size to validate
            min_size: Minimum position size
            max_size: Maximum position size (optional)
            account_balance: Account balance for max validation (optional)
            price: Current price for max validation (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check minimum size
            if size < min_size:
                return False, f"Position size {size} below minimum {min_size}"
            
            # Check maximum size if provided
            if max_size and size > max_size:
                return False, f"Position size {size} above maximum {max_size}"
            
            # Check against account balance if provided
            if account_balance and price:
                position_value = size * price
                if position_value > account_balance * 1.1:  # 10% buffer
                    return False, f"Position value ${position_value:,.2f} exceeds account balance ${account_balance:,.2f}"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating position size: {e}")
            return False, str(e)
    
    def adjust_leverage(
        self,
        symbol: str,
        leverage: int,
        exchange_adapter=None
    ) -> bool:
        """
        Adjust leverage for a symbol.
        
        Args:
            symbol: Trading symbol
            leverage: Desired leverage
            exchange_adapter: Exchange adapter instance
            
        Returns:
            True if successful
        """
        try:
            # Validate leverage
            if leverage < 1 or leverage > self.max_leverage:
                logger.warning(f"Invalid leverage {leverage}, must be between 1 and {self.max_leverage}")
                return False
            
            # Set leverage via exchange adapter
            if exchange_adapter and hasattr(exchange_adapter, 'set_leverage'):
                result = exchange_adapter.set_leverage(symbol, leverage)
                logger.info(f"Set leverage to {leverage}x for {symbol}")
                return result
            else:
                logger.warning("Exchange adapter doesn't support leverage setting")
                return False
                
        except Exception as e:
            logger.error(f"Error adjusting leverage: {e}")
            return False
    
    def calculate_position_value(
        self,
        size: float,
        price: float,
        leverage: int = 1
    ) -> float:
        """
        Calculate position value in USD.
        
        Args:
            size: Position size
            price: Current price
            leverage: Leverage multiplier
            
        Returns:
            Position value in USD
        """
        try:
            # With leverage, position value is size * price
            # But margin requirement is size * price / leverage
            position_value = size * price
            margin_required = position_value / leverage if leverage > 0 else position_value
            
            return position_value
            
        except Exception as e:
            logger.error(f"Error calculating position value: {e}")
            return 0.0

