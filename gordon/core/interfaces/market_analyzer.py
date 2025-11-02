"""
Market Analyzer Interface
==========================
Abstract interface for market analysis and metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


class MarketAnalyzerInterface(ABC):
    """
    Abstract interface for market analysis.
    Provides consistent market metrics and analysis across implementations.
    """

    @abstractmethod
    async def analyze_volatility(self, symbol: str, period: int = 20) -> Dict[str, float]:
        """
        Analyze market volatility.

        Args:
            symbol: Trading symbol
            period: Analysis period

        Returns:
            Dict with volatility metrics (std_dev, atr, etc.)
        """
        pass

    @abstractmethod
    async def analyze_liquidity(self, symbol: str) -> Dict[str, float]:
        """
        Analyze market liquidity.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with liquidity metrics (spread, depth, etc.)
        """
        pass

    @abstractmethod
    async def analyze_trend(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market trend.

        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe

        Returns:
            Dict with trend metrics (direction, strength, etc.)
        """
        pass

    @abstractmethod
    async def detect_support_resistance(self, symbol: str,
                                       lookback: int = 100) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels.

        Args:
            symbol: Trading symbol
            lookback: Number of periods to analyze

        Returns:
            Dict with support and resistance levels
        """
        pass

    @abstractmethod
    async def calculate_correlation(self, symbol1: str, symbol2: str,
                                  period: int = 30) -> float:
        """
        Calculate correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            period: Correlation period

        Returns:
            Correlation coefficient
        """
        pass

    @abstractmethod
    async def get_market_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market profile.

        Args:
            symbol: Trading symbol

        Returns:
            Market profile with volume distribution, POC, etc.
        """
        pass

    @abstractmethod
    async def detect_market_regime(self, symbol: str) -> str:
        """
        Detect current market regime.

        Args:
            symbol: Trading symbol

        Returns:
            Market regime (trending, ranging, volatile, etc.)
        """
        pass