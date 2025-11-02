"""
Rate Limiter Mixin
==================
Provides rate limiting functionality for API calls to prevent hitting exchange limits.

Common pattern identified across all exchanges:
- Binance: 1200 requests/minute
- Bitfinex: 60 requests/minute
- HyperLiquid: Custom rate limiting

This mixin provides a reusable rate limiter that can be configured per exchange.
"""

import asyncio
import time
from typing import Optional


class RateLimiterMixin:
    """
    Mixin class that provides rate limiting functionality.

    This mixin adds rate limiting capabilities to exchange adapters to ensure
    API calls respect exchange-specific rate limits.

    Usage:
    ------
        class MyExchange(RateLimiterMixin, BaseExchange):
            def __init__(self, credentials, event_bus):
                super().__init__(credentials, event_bus)
                self.init_rate_limiter(max_requests=1200, time_window=60)

            async def some_api_call(self):
                await self.rate_limit_acquire()
                # Make API call...

    Attributes:
    -----------
        _rate_limiter: Internal RateLimiter instance
    """

    def init_rate_limiter(self, max_requests: int, time_window: int):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds for the rate limit
        """
        self._rate_limiter = _RateLimiter(max_requests, time_window)

    async def rate_limit_acquire(self):
        """
        Acquire permission to make an API call, waiting if necessary.

        This method will block if the rate limit has been reached until
        it's safe to make another request.
        """
        if hasattr(self, '_rate_limiter'):
            await self._rate_limiter.acquire()

    def get_rate_limit_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with rate limit statistics including:
            - current_requests: Number of requests in current window
            - max_requests: Maximum allowed requests
            - time_window: Time window in seconds
            - requests_remaining: Remaining requests in current window
        """
        if hasattr(self, '_rate_limiter'):
            return self._rate_limiter.get_stats()
        return {}


class _RateLimiter:
    """
    Internal rate limiter implementation.

    This class implements a sliding window rate limiter that tracks
    requests over a time window and enforces limits.
    """

    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """
        Wait if necessary to respect rate limits.

        This method implements a sliding window algorithm:
        1. Remove requests outside the current time window
        2. Check if at rate limit
        3. If at limit, calculate and wait for the required time
        4. Record the new request
        """
        async with self._lock:
            now = time.time()

            # Remove old requests outside the time window
            self.requests = [req for req in self.requests if now - req < self.time_window]

            # Check if we're at the limit
            if len(self.requests) >= self.max_requests:
                # Calculate wait time based on oldest request
                oldest_request = self.requests[0]
                wait_time = self.time_window - (now - oldest_request)

                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Re-clean the list after waiting
                    now = time.time()
                    self.requests = [req for req in self.requests if now - req < self.time_window]

            # Record this request
            self.requests.append(time.time())

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current statistics
        """
        now = time.time()
        # Clean old requests
        self.requests = [req for req in self.requests if now - req < self.time_window]

        return {
            'current_requests': len(self.requests),
            'max_requests': self.max_requests,
            'time_window': self.time_window,
            'requests_remaining': max(0, self.max_requests - len(self.requests))
        }
