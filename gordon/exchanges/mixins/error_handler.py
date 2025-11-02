"""
Error Handler Mixin
===================
Provides standardized error handling patterns for exchange operations.

Common patterns identified across all exchanges:
- Try-except wrapping of API calls
- Logging errors with context
- Emitting error events
- Returning safe default values

This mixin provides reusable error handling methods to reduce code duplication.
"""

import logging
from typing import Any, Callable, Optional, TypeVar, Dict
from functools import wraps
import asyncio


T = TypeVar('T')


class ErrorHandlerMixin:
    """
    Mixin class that provides standardized error handling.

    This mixin adds error handling utilities to exchange adapters for
    consistent error management across different operations.

    Usage:
    ------
        class MyExchange(ErrorHandlerMixin, BaseExchange):
            async def place_order(self, symbol, side, amount):
                return await self.handle_api_error(
                    self._place_order_impl,
                    "place_order",
                    default={},
                    symbol=symbol,
                    side=side,
                    amount=amount
                )

            async def _place_order_impl(self, symbol, side, amount):
                # Actual implementation
                pass

    Attributes:
    -----------
        logger: Logger instance for error logging
        event_bus: Event bus for emitting error events
    """

    async def handle_api_error(
        self,
        func: Callable,
        operation: str,
        default: Any = None,
        emit_event: bool = True,
        **kwargs
    ) -> Any:
        """
        Execute a function with standardized error handling.

        Args:
            func: The async function to execute
            operation: Name of the operation (for logging)
            default: Default value to return on error
            emit_event: Whether to emit an error event
            **kwargs: Arguments to pass to the function

        Returns:
            Function result or default value on error
        """
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(**kwargs)
            else:
                return func(**kwargs)
        except Exception as e:
            return await self._handle_exception(e, operation, default, emit_event, kwargs)

    async def _handle_exception(
        self,
        exception: Exception,
        operation: str,
        default: Any,
        emit_event: bool,
        context: Dict
    ) -> Any:
        """
        Internal method to handle exceptions.

        Args:
            exception: The exception that occurred
            operation: Name of the operation
            default: Default value to return
            emit_event: Whether to emit an error event
            context: Context information about the operation

        Returns:
            Default value
        """
        # Get logger
        logger = getattr(self, 'logger', logging.getLogger(self.__class__.__name__))

        # Format error message with context
        error_msg = self._format_error_message(operation, exception, context)
        logger.error(error_msg)

        # Emit error event if requested and event_bus available
        if emit_event and hasattr(self, 'event_bus') and self.event_bus:
            try:
                await self.emit_event(f"{operation}_failed", {
                    "error": str(exception),
                    "error_type": type(exception).__name__,
                    "context": context
                })
            except Exception as emit_error:
                logger.error(f"Failed to emit error event: {emit_error}")

        return default

    def _format_error_message(self, operation: str, exception: Exception, context: Dict) -> str:
        """
        Format a detailed error message.

        Args:
            operation: Name of the operation
            exception: The exception that occurred
            context: Context information

        Returns:
            Formatted error message
        """
        # Build context string
        context_str = ", ".join(f"{k}={v}" for k, v in context.items() if k != 'self')

        # Format the message
        msg = f"Failed to {operation}"
        if context_str:
            msg += f" ({context_str})"
        msg += f": {type(exception).__name__}: {str(exception)}"

        return msg

    def api_error_handler(self, operation: str, default: Any = None, emit_event: bool = True):
        """
        Decorator for automatic error handling of API methods.

        This decorator wraps async methods with standardized error handling.

        Args:
            operation: Name of the operation
            default: Default value to return on error
            emit_event: Whether to emit error events

        Usage:
        ------
            @api_error_handler("place_order", default={})
            async def place_order(self, symbol, side, amount):
                # Implementation
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract self from args if present
                self_arg = args[0] if args and hasattr(args[0], 'handle_api_error') else None

                if self_arg:
                    # Build context from args and kwargs
                    context = kwargs.copy()
                    return await self_arg.handle_api_error(
                        lambda **kw: func(*args, **kw),
                        operation,
                        default,
                        emit_event,
                        **context
                    )
                else:
                    # Fallback to direct execution
                    return await func(*args, **kwargs)

            return wrapper
        return decorator

    async def retry_on_error(
        self,
        func: Callable,
        operation: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        default: Any = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic on failure.

        Args:
            func: The async function to execute
            operation: Name of the operation
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            backoff_multiplier: Multiplier for exponential backoff
            default: Default value to return if all retries fail
            **kwargs: Arguments to pass to the function

        Returns:
            Function result or default value after max retries
        """
        logger = getattr(self, 'logger', logging.getLogger(self.__class__.__name__))
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                else:
                    return func(**kwargs)
            except Exception as e:
                last_exception = e

                if attempt < max_retries:
                    wait_time = retry_delay * (backoff_multiplier ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {operation} after error: {e}. "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} retries failed for {operation}: {e}")

        # All retries failed, handle final exception
        return await self._handle_exception(last_exception, operation, default, True, kwargs)

    def validate_parameters(self, **validations) -> None:
        """
        Validate parameters and raise ValueError if invalid.

        Args:
            **validations: Dictionary of parameter_name: (value, condition, error_message)

        Raises:
            ValueError: If any validation fails

        Usage:
        ------
            self.validate_parameters(
                symbol=(symbol, lambda s: s and isinstance(s, str), "Symbol must be a non-empty string"),
                amount=(amount, lambda a: a > 0, "Amount must be positive"),
                price=(price, lambda p: p is None or p > 0, "Price must be positive or None")
            )
        """
        for param_name, (value, condition, error_msg) in validations.items():
            if not condition(value):
                raise ValueError(f"Invalid {param_name}: {error_msg}")
