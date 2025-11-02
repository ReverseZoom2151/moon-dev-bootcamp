"""
Unified Cache Manager
=====================
Consolidates all caching functionality previously scattered across strategy_manager.py.
Supports TTL and LRU eviction strategies.
"""

import time
import logging
from typing import Any, Optional, Dict, Callable, Tuple
from collections import OrderedDict
from threading import Lock
import hashlib
import json


class CacheManager:
    """
    Unified cache manager with TTL and LRU support.
    Replaces multiple separate caching systems from strategy_manager.py.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of items in cache (LRU eviction)
            default_ttl: Default time-to-live in seconds
        """
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _make_key(self, *args, **kwargs) -> str:
        """
        Create cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create a string representation of all arguments
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)

        # Use hash for consistent key length
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            # Check if key exists
            if key not in self.cache:
                self.misses += 1
                return None

            # Check TTL
            if key in self.timestamps:
                elapsed = time.time() - self.timestamps[key]
                if elapsed > self.default_ttl:
                    # Expired - remove from cache
                    del self.cache[key]
                    del self.timestamps[key]
                    self.misses += 1
                    return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self.lock:
            # Remove oldest if at capacity
            if key not in self.cache and len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                if oldest in self.timestamps:
                    del self.timestamps[oldest]
                self.evictions += 1

            # Add/update cache
            self.cache[key] = value
            self.cache.move_to_end(key)
            self.timestamps[key] = time.time()

            # Store custom TTL if provided
            if ttl is not None and ttl != self.default_ttl:
                # Note: For simplicity, using default_ttl for all items
                # Could extend to support per-item TTL if needed
                pass

    def get_or_compute(self, key: str, compute_func: Callable,
                      ttl: Optional[int] = None) -> Any:
        """
        Get value from cache or compute if not present.

        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Time-to-live in seconds

        Returns:
            Cached or computed value
        """
        # Try to get from cache
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        value = compute_func()

        # Store in cache
        self.set(key, value, ttl)

        return value

    def get_or_compute_async(self, *args, compute_func: Callable,
                            ttl: Optional[int] = None, **kwargs) -> Any:
        """
        Async version of get_or_compute with automatic key generation.

        Args:
            *args: Arguments for key generation
            compute_func: Async function to compute value
            ttl: Time-to-live in seconds
            **kwargs: Keyword arguments for key generation

        Returns:
            Cached or computed value
        """
        import asyncio

        # Generate key from arguments
        key = self._make_key(*args, **kwargs)

        # Try to get from cache
        value = self.get(key)
        if value is not None:
            return value

        # Compute value (handle async)
        if asyncio.iscoroutinefunction(compute_func):
            # If we're in async context, await it
            loop = asyncio.get_event_loop()
            value = loop.run_until_complete(compute_func(*args, **kwargs))
        else:
            value = compute_func(*args, **kwargs)

        # Store in cache
        self.set(key, value, ttl)

        return value

    def invalidate(self, key: str) -> bool:
        """
        Remove item from cache.

        Args:
            key: Cache key

        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.timestamps:
                    del self.timestamps[key]
                return True
            return False

    def invalidate_pattern(self, pattern: str):
        """
        Remove all items matching pattern from cache.

        Args:
            pattern: Pattern to match (simple string matching)
        """
        with self.lock:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.timestamps:
                    del self.timestamps[key]

    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.logger.info("Cache cleared")

    def cleanup_expired(self):
        """Remove all expired items from cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = []

            for key, timestamp in self.timestamps.items():
                if current_time - timestamp > self.default_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                if key in self.cache:
                    del self.cache[key]
                del self.timestamps[key]

            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired items")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': f"{hit_rate:.1f}%",
                'total_requests': total_requests
            }

    def __repr__(self):
        """String representation."""
        stats = self.get_stats()
        return (f"CacheManager(size={stats['size']}/{stats['max_size']}, "
                f"hit_rate={stats['hit_rate']})")


class StrategyCache(CacheManager):
    """
    Specialized cache for strategy data with predefined key patterns.
    Replaces the multiple cache dictionaries from strategy_manager.py.
    """

    def __init__(self, strategy_name: str, max_size: int = 100, default_ttl: int = 300):
        """
        Initialize strategy-specific cache.

        Args:
            strategy_name: Name of the strategy
            max_size: Maximum cache size
            default_ttl: Default TTL in seconds
        """
        super().__init__(max_size, default_ttl)
        self.strategy_name = strategy_name

    def cache_ohlcv(self, exchange: str, symbol: str, timeframe: str,
                   data: Any, ttl: Optional[int] = None) -> str:
        """
        Cache OHLCV data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe
            data: OHLCV data to cache
            ttl: Time-to-live

        Returns:
            Cache key used
        """
        key = f"{self.strategy_name}:ohlcv:{exchange}:{symbol}:{timeframe}"
        self.set(key, data, ttl)
        return key

    def get_ohlcv(self, exchange: str, symbol: str, timeframe: str) -> Optional[Any]:
        """
        Get cached OHLCV data.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Cached data or None
        """
        key = f"{self.strategy_name}:ohlcv:{exchange}:{symbol}:{timeframe}"
        return self.get(key)

    def cache_indicator(self, indicator: str, symbol: str, period: int,
                       value: Any, ttl: Optional[int] = None) -> str:
        """
        Cache indicator value.

        Args:
            indicator: Indicator name (sma, rsi, etc.)
            symbol: Trading symbol
            period: Indicator period
            value: Indicator value
            ttl: Time-to-live

        Returns:
            Cache key used
        """
        key = f"{self.strategy_name}:{indicator}:{symbol}:{period}"
        self.set(key, value, ttl)
        return key

    def get_indicator(self, indicator: str, symbol: str, period: int) -> Optional[Any]:
        """
        Get cached indicator value.

        Args:
            indicator: Indicator name
            symbol: Trading symbol
            period: Indicator period

        Returns:
            Cached value or None
        """
        key = f"{self.strategy_name}:{indicator}:{symbol}:{period}"
        return self.get(key)

    def cache_signal(self, symbol: str, signal: Any, ttl: Optional[int] = None) -> str:
        """
        Cache trading signal.

        Args:
            symbol: Trading symbol
            signal: Trading signal
            ttl: Time-to-live

        Returns:
            Cache key used
        """
        key = f"{self.strategy_name}:signal:{symbol}"
        self.set(key, signal, ttl)
        return key

    def get_signal(self, symbol: str) -> Optional[Any]:
        """
        Get cached trading signal.

        Args:
            symbol: Trading symbol

        Returns:
            Cached signal or None
        """
        key = f"{self.strategy_name}:signal:{symbol}"
        return self.get(key)

    def invalidate_symbol(self, symbol: str):
        """
        Invalidate all cached data for a symbol.

        Args:
            symbol: Trading symbol
        """
        self.invalidate_pattern(f"{self.strategy_name}:*:{symbol}")


# Global cache instance for shared data
global_cache = CacheManager(max_size=5000, default_ttl=600)