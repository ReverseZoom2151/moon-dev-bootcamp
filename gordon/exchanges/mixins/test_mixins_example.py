"""
Example Tests for Exchange Mixins
==================================
This file demonstrates how to test mixins independently and in combination.

Run with: pytest test_mixins_example.py -v
"""

import pytest
import asyncio
from decimal import Decimal


# Import mixins
import sys
sys.path.insert(0, '..')

from rate_limiter import RateLimiterMixin
from error_handler import ErrorHandlerMixin
from symbol_normalizer import SymbolNormalizerMixin
from decimal_precision import DecimalPrecisionMixin
from order_manager import OrderManagerMixin


# ============================================================================
# Test RateLimiterMixin
# ============================================================================

class MockExchangeWithRateLimiter(RateLimiterMixin):
    """Mock exchange for testing rate limiter."""

    def __init__(self):
        self.init_rate_limiter(max_requests=5, time_window=1)
        self.request_count = 0

    async def make_request(self):
        await self.rate_limit_acquire()
        self.request_count += 1
        return self.request_count


@pytest.mark.asyncio
async def test_rate_limiter_basic():
    """Test basic rate limiting functionality."""
    exchange = MockExchangeWithRateLimiter()

    # Should allow up to 5 requests immediately
    for i in range(5):
        result = await exchange.make_request()
        assert result == i + 1

    assert exchange.request_count == 5


@pytest.mark.asyncio
async def test_rate_limiter_blocking():
    """Test that rate limiter blocks when limit is reached."""
    exchange = MockExchangeWithRateLimiter()

    import time
    start_time = time.time()

    # Make 6 requests (limit is 5)
    for i in range(6):
        await exchange.make_request()

    elapsed = time.time() - start_time

    # Should have taken at least 1 second (the time window)
    assert elapsed >= 1.0
    assert exchange.request_count == 6


def test_rate_limiter_stats():
    """Test rate limiter statistics."""
    exchange = MockExchangeWithRateLimiter()

    stats = exchange.get_rate_limit_stats()

    assert stats['max_requests'] == 5
    assert stats['time_window'] == 1
    assert stats['current_requests'] == 0
    assert stats['requests_remaining'] == 5


# ============================================================================
# Test ErrorHandlerMixin
# ============================================================================

class MockExchangeWithErrorHandler(ErrorHandlerMixin):
    """Mock exchange for testing error handler."""

    def __init__(self):
        self.logger = None  # Would normally be set by BaseExchange
        self.event_bus = None
        self.error_count = 0

    async def emit_event(self, event_type, data):
        """Mock event emission."""
        pass

    async def failing_operation(self):
        """Operation that always fails."""
        raise ValueError("This operation always fails")

    async def successful_operation(self, value):
        """Operation that succeeds."""
        return value * 2


@pytest.mark.asyncio
async def test_error_handler_catches_exceptions():
    """Test that error handler catches exceptions and returns default."""
    exchange = MockExchangeWithErrorHandler()

    result = await exchange.handle_api_error(
        exchange.failing_operation,
        "test_operation",
        default="default_value"
    )

    assert result == "default_value"


@pytest.mark.asyncio
async def test_error_handler_allows_success():
    """Test that error handler allows successful operations."""
    exchange = MockExchangeWithErrorHandler()

    result = await exchange.handle_api_error(
        exchange.successful_operation,
        "test_operation",
        default=0,
        value=5
    )

    assert result == 10


@pytest.mark.asyncio
async def test_retry_on_error():
    """Test retry logic with exponential backoff."""
    exchange = MockExchangeWithErrorHandler()

    call_count = 0

    async def eventually_succeeds():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Not yet")
        return "success"

    result = await exchange.retry_on_error(
        eventually_succeeds,
        "test_operation",
        max_retries=3,
        retry_delay=0.1,
        backoff_multiplier=2.0
    )

    assert result == "success"
    assert call_count == 3


# ============================================================================
# Test SymbolNormalizerMixin
# ============================================================================

class MockExchangeWithSymbolNormalizer(SymbolNormalizerMixin):
    """Mock exchange for testing symbol normalizer."""

    def __init__(self, default_quote='USDT', separator='/', prefix='', suffix=''):
        self.init_symbol_normalizer(default_quote, separator, prefix, suffix)


def test_symbol_normalizer_binance_format():
    """Test Binance-style symbol normalization (slash separator)."""
    exchange = MockExchangeWithSymbolNormalizer(default_quote='USDT', separator='/')

    # Should add /USDT to bare symbols
    assert exchange.normalize_symbol('BTC') == 'BTC/USDT'

    # Should leave already formatted symbols alone
    assert exchange.normalize_symbol('BTC/USDT') == 'BTC/USDT'

    # Should handle other quote currencies
    assert exchange.normalize_symbol('ETH/BTC') == 'ETH/BTC'


def test_symbol_normalizer_bitfinex_format():
    """Test Bitfinex-style symbol normalization (prefix, no separator)."""
    exchange = MockExchangeWithSymbolNormalizer(
        default_quote='UST',
        separator='',
        prefix='t'
    )

    # Should add 't' prefix and UST quote
    assert exchange.normalize_symbol('BTC') == 'tBTCUST'

    # Should handle already formatted
    assert exchange.normalize_symbol('BTC/USDT') == 'tBTCUSDT'


def test_symbol_normalizer_hyperliquid_format():
    """Test HyperLiquid-style symbol normalization (dash separator)."""
    exchange = MockExchangeWithSymbolNormalizer(default_quote='USD', separator='-')

    # Should add -USD to bare symbols
    assert exchange.normalize_symbol('BTC') == 'BTC-USD'

    # Should convert from other formats
    assert exchange.normalize_symbol('BTC/USDT') == 'BTC-USDT'


def test_parse_symbol():
    """Test symbol parsing into base and quote."""
    exchange = MockExchangeWithSymbolNormalizer()

    # Parse slash format
    base, quote = exchange.parse_symbol('BTC/USDT')
    assert base == 'BTC'
    assert quote == 'USDT'

    # Parse dash format
    base, quote = exchange.parse_symbol('BTC-USD')
    assert base == 'BTC'
    assert quote == 'USD'

    # Parse bare symbol
    base, quote = exchange.parse_symbol('BTC')
    assert base == 'BTC'
    assert quote is None


def test_extract_currencies():
    """Test extracting base and quote currencies."""
    exchange = MockExchangeWithSymbolNormalizer(default_quote='USDT')

    assert exchange.extract_base_currency('BTC/USDT') == 'BTC'
    assert exchange.extract_quote_currency('BTC/USDT') == 'USDT'

    # Should use default for bare symbols
    assert exchange.extract_quote_currency('ETH') == 'USDT'


# ============================================================================
# Test DecimalPrecisionMixin
# ============================================================================

class MockExchangeWithDecimalPrecision(DecimalPrecisionMixin):
    """Mock exchange for testing decimal precision."""

    def __init__(self):
        self.symbols_info = {
            'BTC/USDT': {
                'price_precision': 2,
                'amount_precision': 6,
                'min_amount': 0.001,
                'min_cost': 10.0
            },
            'ETH/USDT': {
                'price_precision': 2,
                'amount_precision': 4,
                'min_amount': 0.01,
                'min_cost': 5.0
            }
        }


def test_round_price():
    """Test price rounding to correct precision."""
    exchange = MockExchangeWithDecimalPrecision()

    # BTC has 2 decimal price precision
    assert exchange.round_price(50123.456789, 'BTC/USDT') == 50123.46

    # ETH also has 2 decimal price precision
    assert exchange.round_price(3456.789, 'ETH/USDT') == 3456.79


def test_round_amount():
    """Test amount rounding to correct precision."""
    exchange = MockExchangeWithDecimalPrecision()

    # BTC has 6 decimal amount precision
    assert exchange.round_amount(0.123456789, 'BTC/USDT') == 0.123456

    # ETH has 4 decimal amount precision
    assert exchange.round_amount(1.23456789, 'ETH/USDT') == 1.2345


def test_format_price():
    """Test price formatting as string."""
    exchange = MockExchangeWithDecimalPrecision()

    formatted = exchange.format_price(50123.4, 'BTC/USDT')
    assert formatted == '50123.40'
    assert isinstance(formatted, str)


def test_validate_min_amount():
    """Test minimum amount validation."""
    exchange = MockExchangeWithDecimalPrecision()

    # BTC min amount is 0.001
    assert exchange.validate_min_amount(0.001, 'BTC/USDT') == True
    assert exchange.validate_min_amount(0.0001, 'BTC/USDT') == False

    # ETH min amount is 0.01
    assert exchange.validate_min_amount(0.01, 'ETH/USDT') == True
    assert exchange.validate_min_amount(0.001, 'ETH/USDT') == False


def test_validate_min_cost():
    """Test minimum cost validation."""
    exchange = MockExchangeWithDecimalPrecision()

    # BTC min cost is 10.0
    assert exchange.validate_min_cost(0.001, 50000, 'BTC/USDT') == True  # 50 > 10
    assert exchange.validate_min_cost(0.0001, 50000, 'BTC/USDT') == False  # 5 < 10


def test_calculate_max_amount_from_cost():
    """Test calculating max amount from available cost."""
    exchange = MockExchangeWithDecimalPrecision()

    # $1000 at $50000 per BTC = 0.02 BTC
    max_amount = exchange.calculate_max_amount_from_cost(1000, 50000, 'BTC/USDT')
    assert max_amount == 0.02

    # Should round down to amount precision
    max_amount = exchange.calculate_max_amount_from_cost(1111, 50000, 'BTC/USDT')
    assert max_amount == 0.022220  # Rounded to 6 decimals


def test_step_size_round():
    """Test rounding to step sizes."""
    exchange = MockExchangeWithDecimalPrecision()

    # Round to 0.1 steps
    assert exchange.step_size_round(1.234, 0.1) == 1.2

    # Round to 0.5 steps
    assert exchange.step_size_round(1.234, 0.5) == 1.0

    # Round to 1.0 steps
    assert exchange.step_size_round(1.7, 1.0) == 1.0


# ============================================================================
# Test OrderManagerMixin
# ============================================================================

class MockExchangeWithOrderManager(OrderManagerMixin):
    """Mock exchange for testing order manager."""

    def __init__(self):
        import logging
        self.logger = logging.getLogger('MockExchange')
        self.orders = []
        self.positions = {}

    async def place_order(self, symbol, side, amount, order_type='market', price=None, **kwargs):
        """Mock order placement."""
        order = {
            'id': f'order_{len(self.orders)}',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'type': order_type,
            'price': price,
            'status': 'open',
            **kwargs
        }
        self.orders.append(order)
        return order

    async def get_ticker(self, symbol):
        """Mock ticker."""
        return {'last': 50000}

    async def get_position(self, symbol):
        """Mock position."""
        return self.positions.get(symbol, {})


@pytest.mark.asyncio
async def test_place_bracket_order():
    """Test placing bracket orders."""
    exchange = MockExchangeWithOrderManager()

    result = await exchange.place_bracket_order(
        symbol='BTC/USDT',
        side='buy',
        amount=0.1,
        stop_loss=45000,
        take_profit=55000
    )

    assert result['status'] == 'success'
    assert 'main_order' in result
    assert 'stop_loss' in result
    assert 'take_profit' in result

    # Should have placed 3 orders
    assert len(exchange.orders) == 3


@pytest.mark.asyncio
async def test_place_scaled_entry():
    """Test placing scaled entry orders."""
    exchange = MockExchangeWithOrderManager()

    orders = await exchange.place_scaled_entry(
        symbol='BTC/USDT',
        side='buy',
        total_amount=1.0,
        num_orders=5,
        price_range=(48000, 52000)
    )

    # Should place 5 orders
    assert len(orders) == 5

    # Each order should be 0.2 BTC
    for order in orders:
        assert order['amount'] == 0.2

    # Prices should be evenly distributed
    prices = [order['price'] for order in orders]
    assert prices[0] == 48000
    assert prices[-1] == 52000


@pytest.mark.asyncio
async def test_execute_twap():
    """Test TWAP execution."""
    exchange = MockExchangeWithOrderManager()

    # Execute over very short duration for testing
    orders = await exchange.execute_twap(
        symbol='BTC/USDT',
        side='buy',
        total_amount=1.0,
        duration_minutes=0.01,  # ~0.6 seconds
        num_slices=3
    )

    # Should place 3 orders
    assert len(orders) == 3

    # Each order should be roughly 0.333 BTC
    for order in orders:
        assert abs(order['amount'] - 0.333) < 0.01


@pytest.mark.asyncio
async def test_smart_close_position():
    """Test smart position closing."""
    exchange = MockExchangeWithOrderManager()

    # Set up a mock position
    exchange.positions['BTC/USDT'] = {
        'symbol': 'BTC/USDT',
        'amount': 1.0,
        'side': 'long'
    }

    # Close 50% of position
    result = await exchange.smart_close_position(
        symbol='BTC/USDT',
        partial_percent=50,
        use_limit=True,
        limit_offset_percent=0.1
    )

    # Should place a sell order for 0.5 BTC
    assert result['side'] == 'sell'
    assert result['amount'] == 0.5
    assert result['type'] == 'limit'


def test_format_order_status():
    """Test order status formatting."""
    exchange = MockExchangeWithOrderManager()

    order = {
        'id': '12345',
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'amount': 0.1,
        'price': 50000,
        'status': 'filled'
    }

    formatted = exchange.format_order_status(order)
    assert 'Order #12345' in formatted
    assert 'BUY' in formatted
    assert '0.1' in formatted
    assert 'BTC/USDT' in formatted
    assert '50000' in formatted
    assert 'FILLED' in formatted


def test_format_position_status():
    """Test position status formatting."""
    exchange = MockExchangeWithOrderManager()

    position = {
        'symbol': 'BTC/USDT',
        'amount': 0.5,
        'entry_price': 50000,
        'pnl': 500.00,
        'pnl_percent': 1.00
    }

    formatted = exchange.format_position_status(position)
    assert 'LONG' in formatted
    assert '0.5' in formatted
    assert 'BTC/USDT' in formatted
    assert '50000' in formatted
    assert '+500.00' in formatted
    assert '+1.00%' in formatted


# ============================================================================
# Test Multiple Mixins Together
# ============================================================================

class CompleteExchange(
    RateLimiterMixin,
    ErrorHandlerMixin,
    SymbolNormalizerMixin,
    DecimalPrecisionMixin,
    OrderManagerMixin
):
    """Mock exchange using all mixins together."""

    def __init__(self):
        import logging
        self.logger = logging.getLogger('CompleteExchange')
        self.event_bus = None

        # Initialize all mixins
        self.init_rate_limiter(max_requests=10, time_window=1)
        self.init_symbol_normalizer(default_quote='USDT', separator='/')

        self.symbols_info = {
            'BTC/USDT': {
                'price_precision': 2,
                'amount_precision': 6,
                'min_amount': 0.001,
                'min_cost': 10.0
            }
        }

        self.orders = []

    async def emit_event(self, event_type, data):
        """Mock event emission."""
        pass

    async def place_order(self, symbol, side, amount, order_type='market', price=None, **kwargs):
        """Mock order placement with all mixin features."""
        order = {
            'id': f'order_{len(self.orders)}',
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'type': order_type,
            'price': price,
            'status': 'open'
        }
        self.orders.append(order)
        return order


@pytest.mark.asyncio
async def test_complete_exchange_integration():
    """Test all mixins working together."""
    exchange = CompleteExchange()

    # Test with error handling
    result = await exchange.handle_api_error(
        exchange._place_order_with_all_mixins,
        "place_order",
        default={},
        symbol='BTC',  # Bare symbol
        side='buy',
        amount=0.123456789,  # Needs rounding
        price=50123.456  # Needs rounding
    )

    assert result is not None
    assert result['symbol'] == 'BTC/USDT'  # Normalized
    assert result['amount'] == 0.123456  # Rounded to 6 decimals
    assert result['price'] == 50123.46  # Rounded to 2 decimals

    async def _place_order_with_all_mixins(self, symbol, side, amount, price):
        """Internal method using all mixins."""
        # Rate limiting
        await self.rate_limit_acquire()

        # Symbol normalization
        symbol = self.normalize_symbol(symbol)

        # Precision handling
        amount = self.round_amount(amount, symbol)
        price = self.round_price(price, symbol)

        # Place order
        return await self.place_order(symbol, side, amount, price=price)

    # Bind the method
    exchange._place_order_with_all_mixins = _place_order_with_all_mixins.__get__(
        exchange, CompleteExchange
    )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
