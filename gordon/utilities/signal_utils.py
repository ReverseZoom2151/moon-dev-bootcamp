"""
Signal Utilities Module
=======================
Contains trading signal generation and pattern detection utilities.
Includes trend detection, pattern recognition, and breakout identification.
"""

import pandas as pd
import numpy as np


class SignalUtils:
    """Utilities for trading signals and pattern detection."""

    # ===========================================
    # TREND DETECTION
    # ===========================================

    @staticmethod
    def detect_trend(data: pd.DataFrame, period: int = 20) -> str:
        """
        Detect market trend.

        Returns:
            'uptrend', 'downtrend', or 'sideways'
        """
        sma = data['close'].rolling(window=period).mean()
        current_price = data['close'].iloc[-1]
        sma_current = sma.iloc[-1]

        # Calculate slope
        sma_slope = (sma.iloc[-1] - sma.iloc[-5]) / sma.iloc[-5] * 100

        if current_price > sma_current and sma_slope > 0.5:
            return "uptrend"
        elif current_price < sma_current and sma_slope < -0.5:
            return "downtrend"
        else:
            return "sideways"

    # ===========================================
    # PATTERN DETECTION
    # ===========================================

    @staticmethod
    def detect_engulfing(data: pd.DataFrame) -> str:
        """
        Detect engulfing candle patterns.

        Returns:
            'bullish_engulfing', 'bearish_engulfing', or 'none'
        """
        if len(data) < 2:
            return "none"

        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]

        # Bullish engulfing
        if (prev_candle['close'] < prev_candle['open'] and  # Previous is red
            curr_candle['close'] > curr_candle['open'] and  # Current is green
            curr_candle['open'] <= prev_candle['close'] and  # Opens below prev close
            curr_candle['close'] >= prev_candle['open']):  # Closes above prev open
            return "bullish_engulfing"

        # Bearish engulfing
        if (prev_candle['close'] > prev_candle['open'] and  # Previous is green
            curr_candle['close'] < curr_candle['open'] and  # Current is red
            curr_candle['open'] >= prev_candle['close'] and  # Opens above prev close
            curr_candle['close'] <= prev_candle['open']):  # Closes below prev open
            return "bearish_engulfing"

        return "none"

    @staticmethod
    def detect_gap(data: pd.DataFrame, min_gap_percent: float = 0.5) -> str:
        """
        Detect gap patterns.

        Returns:
            'gap_up', 'gap_down', or 'none'
        """
        if len(data) < 2:
            return "none"

        prev_candle = data.iloc[-2]
        curr_candle = data.iloc[-1]

        gap_percent = ((curr_candle['open'] - prev_candle['close']) /
                      prev_candle['close']) * 100

        if gap_percent > min_gap_percent:
            return "gap_up"
        elif gap_percent < -min_gap_percent:
            return "gap_down"

        return "none"

    @staticmethod
    def detect_breakout(data: pd.DataFrame, period: int = 20,
                       volume_multiplier: float = 1.5) -> str:
        """
        Detect breakout patterns.

        Returns:
            'resistance_breakout', 'support_breakout', or 'none'
        """
        high_period = data['high'].rolling(window=period).max()
        low_period = data['low'].rolling(window=period).min()
        avg_volume = data['volume'].rolling(window=period).mean()

        current = data.iloc[-1]
        current_high_period = high_period.iloc[-2]  # Previous period high
        current_low_period = low_period.iloc[-2]  # Previous period low
        current_avg_volume = avg_volume.iloc[-1]

        # Check for resistance breakout
        if (current['close'] > current_high_period and
            current['volume'] > current_avg_volume * volume_multiplier):
            return "resistance_breakout"

        # Check for support breakout
        if (current['close'] < current_low_period and
            current['volume'] > current_avg_volume * volume_multiplier):
            return "support_breakout"

        return "none"

    @staticmethod
    def detect_doji(data: pd.DataFrame, threshold_percent: float = 0.1) -> bool:
        """
        Detect doji candlestick pattern.

        Args:
            data: OHLCV DataFrame
            threshold_percent: Body size threshold as % of range

        Returns:
            True if doji detected
        """
        if len(data) < 1:
            return False

        candle = data.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        range_val = candle['high'] - candle['low']

        if range_val == 0:
            return False

        body_percent = (body / range_val) * 100
        return body_percent <= threshold_percent

    @staticmethod
    def detect_hammer(data: pd.DataFrame) -> bool:
        """
        Detect hammer candlestick pattern.

        Returns:
            True if hammer detected
        """
        if len(data) < 1:
            return False

        candle = data.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])

        # Hammer: long lower shadow, small body, minimal upper shadow
        return (lower_shadow > body * 2 and
                upper_shadow < body * 0.5 and
                body > 0)

    @staticmethod
    def detect_shooting_star(data: pd.DataFrame) -> bool:
        """
        Detect shooting star candlestick pattern.

        Returns:
            True if shooting star detected
        """
        if len(data) < 1:
            return False

        candle = data.iloc[-1]
        body = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])

        # Shooting star: long upper shadow, small body, minimal lower shadow
        return (upper_shadow > body * 2 and
                lower_shadow < body * 0.5 and
                body > 0)

    # ===========================================
    # SIGNAL GENERATION
    # ===========================================

    @staticmethod
    def generate_ma_crossover_signal(data: pd.DataFrame,
                                    fast_period: int = 10,
                                    slow_period: int = 20) -> str:
        """
        Generate moving average crossover signal.

        Returns:
            'buy', 'sell', or 'neutral'
        """
        if len(data) < slow_period + 1:
            return "neutral"

        fast_ma = data['close'].rolling(window=fast_period).mean()
        slow_ma = data['close'].rolling(window=slow_period).mean()

        # Current and previous values
        fast_curr = fast_ma.iloc[-1]
        fast_prev = fast_ma.iloc[-2]
        slow_curr = slow_ma.iloc[-1]
        slow_prev = slow_ma.iloc[-2]

        # Check for crossover
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            return "buy"
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            return "sell"

        return "neutral"

    @staticmethod
    def generate_rsi_signal(data: pd.DataFrame,
                          period: int = 14,
                          oversold: float = 30,
                          overbought: float = 70) -> str:
        """
        Generate RSI-based signal.

        Returns:
            'buy', 'sell', or 'neutral'
        """
        if len(data) < period + 1:
            return "neutral"

        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]
        previous_rsi = rsi.iloc[-2]

        # Generate signals
        if previous_rsi <= oversold and current_rsi > oversold:
            return "buy"
        elif previous_rsi >= overbought and current_rsi < overbought:
            return "sell"

        return "neutral"

    @staticmethod
    def generate_bollinger_signal(data: pd.DataFrame,
                                 period: int = 20,
                                 std_dev: int = 2) -> str:
        """
        Generate Bollinger Bands signal.

        Returns:
            'buy', 'sell', or 'neutral'
        """
        if len(data) < period:
            return "neutral"

        middle = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        current_price = data['close'].iloc[-1]
        lower_band = lower.iloc[-1]
        upper_band = upper.iloc[-1]

        # Generate signals
        if current_price <= lower_band:
            return "buy"
        elif current_price >= upper_band:
            return "sell"

        return "neutral"

    @staticmethod
    def generate_composite_signal(data: pd.DataFrame) -> dict:
        """
        Generate composite signal from multiple indicators.

        Returns:
            Dictionary with signal breakdown
        """
        signals = {
            "ma_crossover": SignalUtils.generate_ma_crossover_signal(data),
            "rsi": SignalUtils.generate_rsi_signal(data),
            "bollinger": SignalUtils.generate_bollinger_signal(data),
            "trend": SignalUtils.detect_trend(data),
            "engulfing": SignalUtils.detect_engulfing(data),
            "breakout": SignalUtils.detect_breakout(data)
        }

        # Count buy/sell signals
        buy_count = sum(1 for v in signals.values() if v in ['buy', 'uptrend', 'bullish_engulfing', 'resistance_breakout'])
        sell_count = sum(1 for v in signals.values() if v in ['sell', 'downtrend', 'bearish_engulfing', 'support_breakout'])

        # Determine overall signal
        if buy_count > sell_count + 1:
            overall = "strong_buy"
        elif buy_count > sell_count:
            overall = "buy"
        elif sell_count > buy_count + 1:
            overall = "strong_sell"
        elif sell_count > buy_count:
            overall = "sell"
        else:
            overall = "neutral"

        signals["overall"] = overall
        signals["buy_signals"] = buy_count
        signals["sell_signals"] = sell_count

        return signals


# Create singleton instance
signal_utils = SignalUtils()
