"""
Exchange Mixins
===============
Reusable components for exchange adapters to eliminate code duplication.

This package contains mixins that provide common functionality across
different exchange implementations:

- RateLimiterMixin: Rate limiting for API calls
- ErrorHandlerMixin: Standardized error handling patterns
- SymbolNormalizerMixin: Symbol format normalization
- DecimalPrecisionMixin: Precision and rounding utilities
- OrderManagerMixin: Common order operations and management

Usage:
------
Mix these into exchange adapter classes to inherit their functionality:

    class MyExchangeAdapter(RateLimiterMixin, SymbolNormalizerMixin, BaseExchange):
        def __init__(self, credentials, event_bus):
            super().__init__(credentials, event_bus)
            self.init_rate_limiter(max_requests=1200, time_window=60)
"""

from .rate_limiter import RateLimiterMixin
from .error_handler import ErrorHandlerMixin
from .symbol_normalizer import SymbolNormalizerMixin
from .decimal_precision import DecimalPrecisionMixin
from .order_manager import OrderManagerMixin

__all__ = [
    'RateLimiterMixin',
    'ErrorHandlerMixin',
    'SymbolNormalizerMixin',
    'DecimalPrecisionMixin',
    'OrderManagerMixin',
]
