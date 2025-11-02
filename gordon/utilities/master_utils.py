"""
Master Utilities Module
=======================
Consolidates all utility functions from nice_funcs.py files across Days 2-56.
Single source of truth for all trading utilities.

This module now imports from focused utility modules for better organization:
- data_utils: Data manipulation and formatting
- math_utils: Mathematical calculations and indicators
- time_utils: Time and date utilities
- exchange_utils: Exchange-specific utilities
- signal_utils: Trading signal and pattern detection
- format_utils: Formatting and display utilities

All original functions remain accessible through this module for backward compatibility.
"""

import pandas as pd
from typing import List, Dict, Tuple
import logging

# Import from focused utility modules
from .data_utils import DataUtils, data_utils
from .exchange_utils import ExchangeUtils
from .signal_utils import SignalUtils
from .format_utils import FormatUtils
from .exchange_utils import ExchangeUtils, exchange_utils
from .math_utils import MathUtils, math_utils
from .signal_utils import SignalUtils, signal_utils
from .format_utils import FormatUtils, format_utils
from .time_utils import time_utils


class MasterUtils:
    """
    Master utilities class consolidating all nice_funcs.py functionality.

    This class brings together utilities from:
    - Day_10: Basic trading functions
    - Day_20: Mean reversion utilities
    - Day_25: Easy buy/sell functions
    - Day_45: Market making utilities
    - Day_56: Advanced Polymarket functions
    And many more...

    Now organized into focused modules for better maintainability.
    """

    def __init__(self):
        """Initialize the master utilities."""
        self.logger = logging.getLogger(__name__)

        # Initialize utility instances
        self.data_utils = data_utils
        self.math_utils = math_utils
        self.time_utils = time_utils
        self.exchange_utils = exchange_utils
        self.signal_utils = signal_utils
        self.format_utils = format_utils

    # ===========================================
    # TECHNICAL INDICATORS (from math_utils)
    # ===========================================

    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        return MathUtils.calculate_sma(data, period)

    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return MathUtils.calculate_ema(data, period)

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        return MathUtils.calculate_rsi(data, period)

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20,
                                 std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        return MathUtils.calculate_bollinger_bands(data, period, std_dev)

    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        return MathUtils.calculate_vwap(data)

    @staticmethod
    def calculate_vwma(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Weighted Moving Average."""
        return MathUtils.calculate_vwma(data, period)

    @staticmethod
    def calculate_stochastic_rsi(data: pd.DataFrame, rsi_period: int = 14,
                                stoch_period: int = 14, smooth_k: int = 3,
                                smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic RSI.

        Returns:
            Tuple of (stoch_rsi_k, stoch_rsi_d)
        """
        return MathUtils.calculate_stochastic_rsi(data, rsi_period, stoch_period, smooth_k, smooth_d)

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        return MathUtils.calculate_atr(data, period)

    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26,
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD.

        Returns:
            Tuple of (macd, signal, histogram)
        """
        return MathUtils.calculate_macd(data, fast, slow, signal)

    # ===========================================
    # SUPPLY & DEMAND ZONES (from math_utils)
    # ===========================================

    @staticmethod
    def supply_demand_zones_hl(data: pd.DataFrame, lookback: int = 10,
                               threshold_percent: float = 0.5) -> Tuple[List[float], List[float]]:
        """
        Identify supply and demand zones (HyperLiquid version).

        Returns:
            Tuple of (supply_zones, demand_zones)
        """
        return MathUtils.supply_demand_zones_hl(data, lookback, threshold_percent)

    # ===========================================
    # POSITION & RISK MANAGEMENT (from exchange_utils)
    # ===========================================

    @staticmethod
    def calculate_position_size(balance: float, risk_percent: float = 1.0,
                              stop_loss_percent: float = 2.0,
                              leverage: int = 1) -> float:
        """
        Calculate position size based on risk management.

        Args:
            balance: Account balance
            risk_percent: Risk per trade (%)
            stop_loss_percent: Stop loss distance (%)
            leverage: Leverage to use

        Returns:
            Position size
        """
        return ExchangeUtils.calculate_position_size(balance, risk_percent, stop_loss_percent, leverage)

    @staticmethod
    def calculate_stop_loss(entry_price: float, side: str,
                          atr_value: float, multiplier: float = 2.0) -> float:
        """
        Calculate stop loss price using ATR.

        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            atr_value: ATR value
            multiplier: ATR multiplier

        Returns:
            Stop loss price
        """
        return ExchangeUtils.calculate_stop_loss(entry_price, side, atr_value, multiplier)

    @staticmethod
    def calculate_take_profit(entry_price: float, side: str,
                            risk_reward_ratio: float = 2.0,
                            stop_loss_price: float = None) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            risk_reward_ratio: Risk/reward ratio
            stop_loss_price: Stop loss price

        Returns:
            Take profit price
        """
        return ExchangeUtils.calculate_take_profit(entry_price, side, risk_reward_ratio, stop_loss_price)

    @staticmethod
    def calculate_pnl(entry_price: float, exit_price: float, amount: float,
                     side: str, leverage: int = 1, fees: float = 0.001) -> Dict:
        """
        Calculate PnL for a trade.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            amount: Trade amount
            side: 'buy' or 'sell'
            leverage: Leverage used
            fees: Fee percentage

        Returns:
            Dictionary with PnL details
        """
        return ExchangeUtils.calculate_pnl(entry_price, exit_price, amount, side, leverage, fees)

    # ===========================================
    # MARKET ANALYSIS (from math_utils and signal_utils)
    # ===========================================

    @staticmethod
    def detect_trend(data: pd.DataFrame, period: int = 20) -> str:
        """
        Detect market trend.

        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        return SignalUtils.detect_trend(data, period)

    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20,
                                 num_levels: int = 3) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels.

        Returns:
            Dictionary with support and resistance levels
        """
        return MathUtils.detect_support_resistance(data, window, num_levels)

    @staticmethod
    def calculate_volatility(data: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate price volatility.

        Returns:
            Volatility percentage
        """
        return MathUtils.calculate_volatility(data, period)

    # ===========================================
    # PATTERN DETECTION (from signal_utils)
    # ===========================================

    @staticmethod
    def detect_engulfing(data: pd.DataFrame) -> str:
        """
        Detect engulfing candle patterns.

        Returns:
            'bullish_engulfing', 'bearish_engulfing', or 'none'
        """
        return SignalUtils.detect_engulfing(data)

    @staticmethod
    def detect_gap(data: pd.DataFrame, min_gap_percent: float = 0.5) -> str:
        """
        Detect gap patterns.

        Returns:
            'gap_up', 'gap_down', or 'none'
        """
        return SignalUtils.detect_gap(data, min_gap_percent)

    @staticmethod
    def detect_breakout(data: pd.DataFrame, period: int = 20,
                       volume_multiplier: float = 1.5) -> str:
        """
        Detect breakout patterns.

        Returns:
            'resistance_breakout', 'support_breakout', or 'none'
        """
        return SignalUtils.detect_breakout(data, period, volume_multiplier)

    # ===========================================
    # DATA PROCESSING (from data_utils)
    # ===========================================

    @staticmethod
    def clean_ohlcv_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare OHLCV data.

        Args:
            data: Raw OHLCV DataFrame

        Returns:
            Cleaned DataFrame
        """
        return DataUtils.clean_ohlcv_data(data)

    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to different timeframe.

        Args:
            data: OHLCV DataFrame with datetime index
            timeframe: Target timeframe (e.g., '1h', '4h', '1d')

        Returns:
            Resampled DataFrame
        """
        return DataUtils.resample_ohlcv(data, timeframe)

    @staticmethod
    def calculate_returns(data: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate returns over multiple periods.

        Args:
            data: OHLCV DataFrame
            periods: List of periods to calculate returns for

        Returns:
            DataFrame with returns columns added
        """
        return DataUtils.calculate_returns(data, periods)

    # ===========================================
    # NUMPY COMPATIBILITY (from data_utils)
    # ===========================================

    @staticmethod
    def handle_numpy_nan(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle numpy NaN compatibility issues.

        This addresses the numpy 2.0 compatibility issue found in Day_20.
        """
        return DataUtils.handle_numpy_nan(df)

    # ===========================================
    # LIQUIDATION ANALYSIS (from exchange_utils)
    # ===========================================

    @staticmethod
    def calculate_liquidation_levels(price: float, leverage: int,
                                   side: str = "long") -> Dict[str, float]:
        """
        Calculate liquidation levels.

        Args:
            price: Current price
            leverage: Leverage used
            side: 'long' or 'short'

        Returns:
            Dictionary with liquidation levels
        """
        return ExchangeUtils.calculate_liquidation_levels(price, leverage, side)

    @staticmethod
    def find_liquidation_clusters(orderbook: Dict, price: float,
                                threshold_percent: float = 5.0) -> List[Dict]:
        """
        Find potential liquidation clusters in orderbook.

        Args:
            orderbook: Orderbook data
            price: Current price
            threshold_percent: Distance threshold

        Returns:
            List of liquidation clusters
        """
        return ExchangeUtils.find_liquidation_clusters(orderbook, price, threshold_percent)

    # ===========================================
    # RRS (RELATIVE ROTATION STRENGTH) (from math_utils)
    # ===========================================

    @staticmethod
    def calculate_rrs(symbol_data: pd.DataFrame, market_data: pd.DataFrame,
                     period: int = 14) -> pd.Series:
        """
        Calculate Relative Rotation Strength.

        Args:
            symbol_data: Symbol OHLCV data
            market_data: Market benchmark data
            period: Calculation period

        Returns:
            RRS values
        """
        return MathUtils.calculate_rrs(symbol_data, market_data, period)

    # ===========================================
    # MARKET MAKING (from exchange_utils)
    # ===========================================

    @staticmethod
    def calculate_spread(ask: float, bid: float) -> float:
        """Calculate bid-ask spread percentage."""
        return ExchangeUtils.calculate_spread(ask, bid)

    @staticmethod
    def calculate_fair_value(orderbook: Dict, depth: int = 5) -> float:
        """
        Calculate fair value from orderbook.

        Args:
            orderbook: Orderbook data
            depth: Depth to consider

        Returns:
            Fair value price
        """
        return ExchangeUtils.calculate_fair_value(orderbook, depth)

    @staticmethod
    def calculate_order_imbalance(orderbook: Dict, depth: int = 5) -> float:
        """
        Calculate order book imbalance.

        Returns:
            Imbalance ratio (positive = more buying pressure)
        """
        return ExchangeUtils.calculate_order_imbalance(orderbook, depth)

    # ===========================================
    # SMART ORDER ROUTING (from exchange_utils)
    # ===========================================

    @staticmethod
    def calculate_order_slices(total_amount: float, max_slice_size: float = None,
                             num_slices: int = None) -> List[float]:
        """
        Calculate order slices for large orders.

        Args:
            total_amount: Total order amount
            max_slice_size: Maximum size per slice
            num_slices: Number of slices (alternative to max_slice_size)

        Returns:
            List of slice amounts
        """
        return ExchangeUtils.calculate_order_slices(total_amount, max_slice_size, num_slices)

    # ===========================================
    # PORTFOLIO ANALYTICS (from math_utils)
    # ===========================================

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        return MathUtils.calculate_sharpe_ratio(returns, risk_free_rate)

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Dictionary with drawdown metrics
        """
        return MathUtils.calculate_max_drawdown(equity_curve)

    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate win rate statistics.

        Args:
            trades: List of trade dictionaries with 'pnl' key

        Returns:
            Dictionary with win rate statistics
        """
        return MathUtils.calculate_win_rate(trades)

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for position sizing.

        Args:
            win_rate: Win rate percentage (0-100)
            avg_win: Average winning amount
            avg_loss: Average losing amount (positive value)

        Returns:
            Kelly percentage (0-100)
        """
        return MathUtils.kelly_criterion(win_rate, avg_win, avg_loss)

    # ===========================================
    # LOGGING AND MONITORING (from format_utils)
    # ===========================================

    @staticmethod
    def log_trade(trade_data: Dict, filepath: str = "trades.json"):
        """
        Log trade to file.

        Args:
            trade_data: Trade information
            filepath: Path to log file
        """
        return FormatUtils.log_trade(trade_data, filepath)

    @staticmethod
    def format_number(value: float, decimals: int = 2) -> str:
        """Format number for display."""
        return FormatUtils.format_number(value, decimals)


# Create singleton instance
master_utils = MasterUtils()
