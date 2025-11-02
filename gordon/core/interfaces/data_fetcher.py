"""
Data Fetcher Interface
=======================
Abstract interface for data fetching to enable testing and multiple implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd


class DataFetcherInterface(ABC):
    """
    Abstract interface for fetching market data.
    Allows for live, paper, and backtest implementations.
    """

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str,
                         limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 15m, 1h, 1d, etc.)
            limit: Number of candles to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        """
        Fetch current ticker data.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with keys: bid, ask, last, volume
        """
        pass

    @abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, List]:
        """
        Fetch order book data.

        Args:
            symbol: Trading symbol
            limit: Depth of order book

        Returns:
            Dict with keys: bids, asks (each is list of [price, amount])
        """
        pass

    @abstractmethod
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Fetch recent trades.

        Args:
            symbol: Trading symbol
            limit: Number of trades

        Returns:
            List of trade dictionaries
        """
        pass

    @abstractmethod
    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Fetch current funding rate (for perpetual futures).

        Args:
            symbol: Trading symbol

        Returns:
            Funding rate or None if not applicable
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.

        Returns:
            List of symbol strings
        """
        pass

    @abstractmethod
    def get_available_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes.

        Returns:
            List of timeframe strings
        """
        pass