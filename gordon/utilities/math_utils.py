"""
Math Utilities Module
=====================
Contains mathematical calculations, technical indicators, and statistical functions.
Includes moving averages, RSI, MACD, Bollinger Bands, and other technical indicators.
"""

import pandas as pd
import numpy as np
import ta
from typing import Tuple, List, Dict


class MathUtils:
    """Utilities for mathematical calculations and technical indicators."""

    # ===========================================
    # TECHNICAL INDICATORS
    # ===========================================

    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data['close'].rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        return ta.momentum.RSIIndicator(data['close'], window=period).rsi()

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20,
                                 std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        cumulative_tpv = (typical_price * data['volume']).cumsum()
        cumulative_volume = data['volume'].cumsum()
        return cumulative_tpv / cumulative_volume

    @staticmethod
    def calculate_vwma(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Weighted Moving Average."""
        vwma = (data['close'] * data['volume']).rolling(window=period).sum() / \
               data['volume'].rolling(window=period).sum()
        return vwma

    @staticmethod
    def calculate_stochastic_rsi(data: pd.DataFrame, rsi_period: int = 14,
                                stoch_period: int = 14, smooth_k: int = 3,
                                smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic RSI.

        Returns:
            Tuple of (stoch_rsi_k, stoch_rsi_d)
        """
        rsi = ta.momentum.RSIIndicator(data['close'], window=rsi_period).rsi()
        stoch_rsi = ta.momentum.StochRSIIndicator(
            data['close'],
            window=rsi_period,
            smooth1=smooth_k,
            smooth2=smooth_d
        )
        return stoch_rsi.stochrsi_k(), stoch_rsi.stochrsi_d()

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        return ta.volatility.AverageTrueRange(
            data['high'], data['low'], data['close'], window=period
        ).average_true_range()

    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26,
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD.

        Returns:
            Tuple of (macd, signal, histogram)
        """
        macd_indicator = ta.trend.MACD(data['close'], window_slow=slow,
                                      window_fast=fast, window_sign=signal)
        return (macd_indicator.macd(),
                macd_indicator.macd_signal(),
                macd_indicator.macd_diff())

    # ===========================================
    # SUPPLY & DEMAND ZONES
    # ===========================================

    @staticmethod
    def supply_demand_zones_hl(data: pd.DataFrame, lookback: int = 10,
                               threshold_percent: float = 0.5) -> Tuple[List[float], List[float]]:
        """
        Identify supply and demand zones (HyperLiquid version).

        Returns:
            Tuple of (supply_zones, demand_zones)
        """
        supply_zones = []
        demand_zones = []

        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        volumes = data['volume'].values

        for i in range(lookback, len(data)):
            # Check for supply zone (resistance)
            if highs[i] == max(highs[i-lookback:i+1]):
                if volumes[i] > np.mean(volumes[i-lookback:i]):
                    supply_zones.append(highs[i])

            # Check for demand zone (support)
            if lows[i] == min(lows[i-lookback:i+1]):
                if volumes[i] > np.mean(volumes[i-lookback:i]):
                    demand_zones.append(lows[i])

        # Remove duplicates and sort
        supply_zones = sorted(list(set(supply_zones)), reverse=True)
        demand_zones = sorted(list(set(demand_zones)))

        return supply_zones[:5], demand_zones[:5]  # Return top 5 zones

    # ===========================================
    # MARKET ANALYSIS
    # ===========================================

    @staticmethod
    def detect_support_resistance(data: pd.DataFrame, window: int = 20,
                                 num_levels: int = 3) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels.

        Returns:
            Dictionary with support and resistance levels
        """
        highs = data['high'].rolling(window=window).max()
        lows = data['low'].rolling(window=window).min()

        # Find peaks and troughs
        resistance_levels = []
        support_levels = []

        for i in range(window, len(data) - window):
            if highs.iloc[i] == data['high'].iloc[i-window:i+window].max():
                resistance_levels.append(highs.iloc[i])
            if lows.iloc[i] == data['low'].iloc[i-window:i+window].min():
                support_levels.append(lows.iloc[i])

        # Get unique levels and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:num_levels]
        support_levels = sorted(list(set(support_levels)))[:num_levels]

        return {
            "resistance": resistance_levels,
            "support": support_levels
        }

    @staticmethod
    def calculate_volatility(data: pd.DataFrame, period: int = 20) -> float:
        """
        Calculate price volatility.

        Returns:
            Volatility percentage
        """
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100
        return volatility.iloc[-1]

    # ===========================================
    # RRS (RELATIVE ROTATION STRENGTH)
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
        # Calculate relative performance
        symbol_returns = symbol_data['close'].pct_change()
        market_returns = market_data['close'].pct_change()

        relative_performance = (1 + symbol_returns) / (1 + market_returns) - 1

        # Calculate RRS
        rrs = relative_performance.rolling(window=period).mean() * 100

        return rrs

    # ===========================================
    # PORTFOLIO ANALYTICS
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
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Dictionary with drawdown metrics
        """
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # Find recovery
        if max_dd < 0:
            recovery_idx = equity_curve[max_dd_idx:][equity_curve[max_dd_idx:] >= cummax[max_dd_idx]].index
            if len(recovery_idx) > 0:
                recovery_time = recovery_idx[0] - max_dd_idx
            else:
                recovery_time = None
        else:
            recovery_time = None

        return {
            "max_drawdown": round(max_dd * 100, 2),
            "max_drawdown_date": max_dd_idx,
            "recovery_time": recovery_time
        }

    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate win rate statistics.

        Args:
            trades: List of trade dictionaries with 'pnl' key

        Returns:
            Dictionary with win rate statistics
        """
        if not trades:
            return {"win_rate": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0}

        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else 0

        return {
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses)
        }

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
        if avg_loss == 0:
            return 0

        p = win_rate / 100  # Probability of win
        q = 1 - p  # Probability of loss
        b = avg_win / avg_loss  # Win/loss ratio

        kelly = (p * b - q) / b
        kelly = max(0, min(kelly, 0.25))  # Cap at 25% for safety

        return round(kelly * 100, 2)


# Create singleton instance
math_utils = MathUtils()
