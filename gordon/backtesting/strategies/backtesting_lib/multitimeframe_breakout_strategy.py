"""
Multi-timeframe Breakout Strategy with Bollinger Bands (Day 18)
================================================================
Combines daily resistance levels with hourly Bollinger Band breakouts.

Features:
- Uses daily timeframe for resistance calculation
- Uses hourly timeframe for entry signals
- Entry: Price breaks above daily resistance + BB upper band + volume confirmation
- Risk management: ATR-based stops
- Position sizing: Risk-based (2% per trade)
"""

import numpy as np
from backtesting import Strategy
from backtesting.test import SMA


def BBANDS_CUSTOM(data, period=20, std_dev=2):
    """Custom Bollinger Bands calculation for Day 18 strategy."""
    middle_band = SMA(data, period)
    std = np.std(data[-period:] if len(data) >= period else data)
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    return upper_band, middle_band, lower_band


class MultiTimeframeBreakoutStrategy(Strategy):
    """
    Multi-timeframe Breakout Strategy with Bollinger Bands (Day 18).

    Combines daily resistance levels with hourly Bollinger Band breakouts:
    - Uses daily timeframe for resistance calculation
    - Uses hourly timeframe for entry signals
    - Entry: Price breaks above daily resistance + BB upper band + volume confirmation
    - Risk management: ATR-based stops
    - Position sizing: Risk-based (2% per trade)
    """

    # Strategy parameters
    atr_period = 14
    tp_percent = 5  # Take profit percentage
    sl_atr_mult = 1.5  # Stop loss ATR multiplier
    volume_factor = 1.5  # Volume confirmation factor
    volume_period = 20
    bb_period = 20
    bb_std = 2

    # Optimization ranges
    atr_period_range = [10, 14, 20]
    tp_percent_range = [3, 5, 8]
    sl_atr_mult_range = [1.0, 2.0]

    def __init__(self, *args, daily_resistance=None, **kwargs):
        """Initialize with daily resistance levels."""
        super().__init__(*args, **kwargs)
        self.daily_resistance_data = daily_resistance

    def init(self):
        """Initialize indicators."""
        # ATR for volatility
        self.atr = self.I(SMA, abs(self.data.High - self.data.Low), self.atr_period)

        # Volume moving average
        self.volume_ma = self.I(SMA, self.data.Volume, self.volume_period)

        # Bollinger Bands
        def bb_upper(data):
            return BBANDS_CUSTOM(data, self.bb_period, self.bb_std)[0]
        def bb_mid(data):
            return BBANDS_CUSTOM(data, self.bb_period, self.bb_std)[1]
        def bb_lower(data):
            return BBANDS_CUSTOM(data, self.bb_period, self.bb_std)[2]

        self.bb_upper = self.I(bb_upper, self.data.Close)
        self.bb_mid = self.I(bb_mid, self.data.Close)
        self.bb_lower = self.I(bb_lower, self.data.Close)

        # Trade tracking
        self.trade_count = 0
        self.win_count = 0

    def get_daily_resistance(self, current_time):
        """Get daily resistance for current time."""
        if self.daily_resistance_data is None:
            # If no daily data provided, use current high as resistance
            return self.data.High[-20:].max() if len(self.data) >= 20 else self.data.High[-1]

        # Convert timestamp to date for matching
        current_date = current_time.date() if hasattr(current_time, 'date') else current_time

        # Find corresponding daily resistance
        try:
            # Look for exact date match
            if current_date in self.daily_resistance_data.index:
                return self.daily_resistance_data[current_date]

            # Use previous day's resistance
            prev_dates = self.daily_resistance_data.index[self.daily_resistance_data.index < current_date]
            if len(prev_dates) > 0:
                return self.daily_resistance_data[prev_dates[-1]]

            # Use first available resistance
            return self.daily_resistance_data.iloc[0]
        except:
            # Fallback to current high
            return self.data.High[-20:].max() if len(self.data) >= 20 else self.data.High[-1]

    def next(self):
        """Execute strategy logic."""
        # Get current values
        current_time = self.data.index[-1]
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        # Get daily resistance
        daily_resistance = self.get_daily_resistance(current_time)

        # Check for valid values
        if (np.isnan(daily_resistance) or np.isnan(current_close) or
            np.isnan(self.bb_upper[-1]) or np.isnan(self.volume_ma[-1]) or
            not np.isfinite(daily_resistance) or not np.isfinite(current_close)):
            return

        # Breakout conditions
        breakout_conditions = (
            current_close > daily_resistance and  # Above daily resistance
            current_close > self.bb_upper[-1] and  # Above Bollinger upper band
            current_volume > self.volume_ma[-1] * self.volume_factor  # Volume confirmation
        )

        # Entry logic
        if not self.position and breakout_conditions:
            entry_price = current_close

            # Calculate stops
            stop_loss = max(0, entry_price - self.atr[-1] * self.sl_atr_mult)
            take_profit = entry_price * (1 + self.tp_percent / 100)

            # Risk-based position sizing
            risk_amount = 0.02 * self.equity  # 2% risk per trade
            price_risk = entry_price - stop_loss

            if price_risk > 0:
                size = min(risk_amount / price_risk, 0.5)  # Cap at 50% of equity
                self.buy(size=size, sl=stop_loss, tp=take_profit)
                self.trade_count += 1
