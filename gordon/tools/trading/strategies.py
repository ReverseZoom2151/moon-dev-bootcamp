"""
Trading Strategy Tools for Gordon
==================================
Execute various trading strategies on live markets.
"""

from typing import Dict, Any, Optional
from langchain_core.tools import tool
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from gordon.core.strategy_manager import StrategyManager


@tool
def run_sma_strategy(
    symbol: str,
    exchange: str = "binance",
    short_period: int = 10,
    long_period: int = 30,
    position_size: float = 0.01,
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute a Simple Moving Average (SMA) crossover strategy.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        exchange: Exchange to trade on
        short_period: Short SMA period
        long_period: Long SMA period
        position_size: Size of position to take
        dry_run: If True, simulate trades without executing

    Returns:
        Strategy execution results including signals and trades
    """
    try:
        manager = StrategyManager()

        # Configure strategy
        config = {
            'symbol': symbol,
            'exchange': exchange,
            'short_period': short_period,
            'long_period': long_period,
            'position_size': position_size,
            'dry_run': dry_run
        }

        # Run strategy
        result = manager.execute_strategy('sma_crossover', config)

        return {
            'status': 'success',
            'strategy': 'SMA Crossover',
            'symbol': symbol,
            'signal': result.get('signal'),
            'trade_executed': result.get('trade_executed', False),
            'details': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def run_rsi_strategy(
    symbol: str,
    exchange: str = "binance",
    rsi_period: int = 14,
    oversold_level: float = 30,
    overbought_level: float = 70,
    position_size: float = 0.01,
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute an RSI (Relative Strength Index) strategy.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        exchange: Exchange to trade on
        rsi_period: Period for RSI calculation
        oversold_level: RSI level to consider oversold
        overbought_level: RSI level to consider overbought
        position_size: Size of position to take
        dry_run: If True, simulate trades without executing

    Returns:
        Strategy execution results
    """
    try:
        manager = StrategyManager()

        config = {
            'symbol': symbol,
            'exchange': exchange,
            'rsi_period': rsi_period,
            'oversold': oversold_level,
            'overbought': overbought_level,
            'position_size': position_size,
            'dry_run': dry_run
        }

        result = manager.execute_strategy('rsi', config)

        return {
            'status': 'success',
            'strategy': 'RSI',
            'symbol': symbol,
            'rsi_value': result.get('rsi_value'),
            'signal': result.get('signal'),
            'trade_executed': result.get('trade_executed', False),
            'details': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def run_vwap_strategy(
    symbol: str,
    exchange: str = "binance",
    lookback_period: int = 20,
    std_dev_multiplier: float = 2.0,
    position_size: float = 0.01,
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute a VWAP (Volume Weighted Average Price) strategy.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        exchange: Exchange to trade on
        lookback_period: Period for VWAP calculation
        std_dev_multiplier: Standard deviation multiplier for bands
        position_size: Size of position to take
        dry_run: If True, simulate trades without executing

    Returns:
        Strategy execution results
    """
    try:
        manager = StrategyManager()

        config = {
            'symbol': symbol,
            'exchange': exchange,
            'lookback_period': lookback_period,
            'std_dev_multiplier': std_dev_multiplier,
            'position_size': position_size,
            'dry_run': dry_run
        }

        result = manager.execute_strategy('vwap', config)

        return {
            'status': 'success',
            'strategy': 'VWAP',
            'symbol': symbol,
            'vwap': result.get('vwap'),
            'current_price': result.get('price'),
            'signal': result.get('signal'),
            'trade_executed': result.get('trade_executed', False),
            'details': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def run_mean_reversion_strategy(
    symbol: str,
    exchange: str = "binance",
    lookback_period: int = 20,
    z_score_threshold: float = 2.0,
    position_size: float = 0.01,
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute a Mean Reversion strategy.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        exchange: Exchange to trade on
        lookback_period: Period for calculating mean
        z_score_threshold: Z-score threshold for entry
        position_size: Size of position to take
        dry_run: If True, simulate trades without executing

    Returns:
        Strategy execution results
    """
    try:
        manager = StrategyManager()

        config = {
            'symbol': symbol,
            'exchange': exchange,
            'lookback_period': lookback_period,
            'z_score_threshold': z_score_threshold,
            'position_size': position_size,
            'dry_run': dry_run
        }

        result = manager.execute_strategy('mean_reversion', config)

        return {
            'status': 'success',
            'strategy': 'Mean Reversion',
            'symbol': symbol,
            'z_score': result.get('z_score'),
            'signal': result.get('signal'),
            'trade_executed': result.get('trade_executed', False),
            'details': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


@tool
def run_bollinger_strategy(
    symbol: str,
    exchange: str = "binance",
    bb_period: int = 20,
    std_dev: float = 2.0,
    position_size: float = 0.01,
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute a Bollinger Bands strategy.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        exchange: Exchange to trade on
        bb_period: Period for Bollinger Bands
        std_dev: Standard deviation multiplier
        position_size: Size of position to take
        dry_run: If True, simulate trades without executing

    Returns:
        Strategy execution results
    """
    try:
        manager = StrategyManager()

        config = {
            'symbol': symbol,
            'exchange': exchange,
            'bb_period': bb_period,
            'std_dev': std_dev,
            'position_size': position_size,
            'dry_run': dry_run
        }

        result = manager.execute_strategy('bollinger_bands', config)

        return {
            'status': 'success',
            'strategy': 'Bollinger Bands',
            'symbol': symbol,
            'upper_band': result.get('upper_band'),
            'middle_band': result.get('middle_band'),
            'lower_band': result.get('lower_band'),
            'current_price': result.get('price'),
            'signal': result.get('signal'),
            'trade_executed': result.get('trade_executed', False),
            'details': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }