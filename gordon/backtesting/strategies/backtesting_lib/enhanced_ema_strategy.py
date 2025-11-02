"""
Enhanced EMA Strategy with Multiple Indicators (Day 17)
=======================================================
Comprehensive strategy using multiple technical indicators.

Features:
- EMA crossovers (fast, slow, trend)
- StochRSI for momentum
- Volume confirmation
- OBV for trend strength
- ATR-based position sizing and stops
- Partial profit taking
"""

import pandas as pd
import pandas_ta as ta
from backtesting.lib import crossover, TrailingStrategy


class EnhancedEMAStrategy(TrailingStrategy):
    """
    Enhanced EMA Strategy with Multiple Indicators (Day 17).

    Comprehensive strategy using:
    - EMA crossovers (fast, slow, trend)
    - StochRSI for momentum
    - Volume confirmation
    - OBV for trend strength
    - ATR-based position sizing and stops
    - Partial profit taking
    """

    # Default parameters (optimizable)
    ema_fast_period = 9
    ema_slow_period = 18
    ema_trend_period = 200
    stochrsi_rsi_len = 14
    atr_period = 14
    volume_ma_period = 20
    risk_per_trade = 0.02  # 2% risk per trade
    atr_sl_multiplier = 2.0
    partial_profit_factor = 1.5

    def init(self):
        """Initialize all indicators."""
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # EMAs
        self.ema_fast = self.I(ta.ema, close, self.ema_fast_period)
        self.ema_slow = self.I(ta.ema, close, self.ema_slow_period)
        self.ema_trend = self.I(ta.ema, close, self.ema_trend_period)

        # StochRSI
        stochrsi = ta.stochrsi(
            close,
            length=self.stochrsi_rsi_len,
            rsi_length=self.stochrsi_rsi_len,
            k=3,
            d=3
        )
        self.stoch_k = self.I(lambda: stochrsi['STOCHRSIk_14_14_3_3'].values)
        self.stoch_d = self.I(lambda: stochrsi['STOCHRSId_14_14_3_3'].values)

        # ATR for position sizing and stops
        self.atr = self.I(ta.atr, high, low, close, self.atr_period)

        # Volume indicators
        self.volume_ma = self.I(ta.sma, volume, self.volume_ma_period)
        self.obv = self.I(ta.obv, close, volume)

        # Set trailing stop
        self.set_trailing_sl(self.atr_sl_multiplier)

        # Position tracking
        if not hasattr(self, 'position_entry_bar'):
            self.position_entry_bar = None
        if not hasattr(self, 'position_entry_price'):
            self.position_entry_price = None

    def next(self):
        """Execute strategy logic with advanced conditions."""
        price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # Wait for indicators to stabilize
        if len(self.data) < max(self.ema_trend_period, self.volume_ma_period) + 10:
            return

        # Dynamic position sizing based on ATR
        equity = self.equity
        position_size = (equity * self.risk_per_trade) / (atr_val * self.atr_sl_multiplier) if atr_val > 0 else 0
        position_size = min(max(position_size, 0.01), 0.5)
        position_size = round(position_size, 2)

        # Trend filters
        bullish_trend = price > self.ema_trend[-1]
        bearish_trend = price < self.ema_trend[-1]

        # Volume confirmation
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1]
        obv_trend = self.obv[-1] > self.obv[-2]

        # EMA crossovers
        ema_cross_up = crossover(self.ema_fast, self.ema_slow)
        ema_cross_down = crossover(self.ema_slow, self.ema_fast)

        # StochRSI signals
        stoch_bullish = crossover(self.stoch_k, self.stoch_d) and self.stoch_d[-1] < 40
        stoch_bearish = crossover(self.stoch_d, self.stoch_k) and self.stoch_d[-1] > 60

        # Entry logic
        if not self.position:
            # Long entry
            if all([bullish_trend, ema_cross_up, stoch_bullish, volume_ok, obv_trend]):
                sl = price - atr_val * self.atr_sl_multiplier
                tp = price + atr_val * self.partial_profit_factor
                self.buy(size=position_size, sl=sl, tp=tp)
                self.position_entry_price = price
                self.position_entry_bar = len(self.data)

            # Short entry
            elif all([bearish_trend, ema_cross_down, stoch_bearish, volume_ok, not obv_trend]):
                sl = price + atr_val * self.atr_sl_multiplier
                tp = price - atr_val * self.partial_profit_factor
                self.sell(size=position_size, sl=sl, tp=tp)
                self.position_entry_price = price
                self.position_entry_bar = len(self.data)

        # Position management
        elif self.position:
            # Partial profit taking
            if self.position.pl_pct >= 1.5:
                self.position.close(0.5)  # Close 50% of position

            # Time-based exit for stagnant positions
            current_duration = len(self.data) - self.position_entry_bar if self.position_entry_bar else 0
            if current_duration > 5 and abs(self.position.pl_pct) < 0.5:
                self.position.close()
                self.position_entry_bar = None
