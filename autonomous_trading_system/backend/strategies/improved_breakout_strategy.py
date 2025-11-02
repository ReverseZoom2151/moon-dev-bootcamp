import numpy as np
import pandas as pd
import logging
from backtesting import Strategy
from backtesting.test import SMA

logger = logging.getLogger(__name__)

# Custom Bollinger Bands implementation

def BBANDS(data, period=20, std_dev=2):
    """Calculate Bollinger Bands for the data"""
    middle_band = SMA(data, period)
    std = np.std(data[-period:] if len(data) >= period else data)
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    return upper_band, middle_band, lower_band


class ImprovedBreakoutStrategy(Strategy):
    """
    Improved Breakout Strategy based on daily resistance breakout with filters
    """
    # Strategy parameters
    atr_period = 14
    tp_percent = 5
    sl_atr_mult = 1.5
    volume_factor = 1.5
    volume_period = 20
    bb_period = 20
    bb_std = 2

    # Parameter optimization ranges
    atr_period_range = [10, 14, 20]
    tp_percent_range = [3, 5, 8]
    sl_atr_mult_range = [1.0, 2.0]

    # Will be set by backtesting service
    daily_resistance: pd.Series = None

    def init(self):
        if self.__class__.daily_resistance is None:
            raise ValueError("daily_resistance not set for ImprovedBreakoutStrategy")
        # Indicators
        self.atr = self.I(SMA, abs(self.data.High - self.data.Low), self.atr_period)
        self.volume_ma = self.I(SMA, self.data.Volume, self.volume_period)
        self.bb_upper = self.I(lambda d: BBANDS(d, self.bb_period, self.bb_std)[0], self.data.Close)
        self.bb_mid = self.I(lambda d: BBANDS(d, self.bb_period, self.bb_std)[1], self.data.Close)
        self.bb_lower = self.I(lambda d: BBANDS(d, self.bb_period, self.bb_std)[2], self.data.Close)
        self.trade_count = 0
        self.win_count = 0

    def next(self):
        # Determine current date
        current_time = self.data.index[-1]
        current_date = current_time.date()
        # Fetch daily resistance
        dr = self.__class__.daily_resistance
        try:
            idx = dr.index[dr.index.date == current_date][0]
            daily_res = dr.loc[idx]
        except Exception:
            prev = dr.index[dr.index.date < current_date]
            if len(prev) > 0:
                daily_res = dr.loc[prev[-1]]
            else:
                daily_res = dr.iloc[0]
        # Current metrics and debug logging
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        upper_bb = self.bb_upper[-1]
        vol_ma = self.volume_ma[-1]
        atr_val = self.atr[-1]
        # Debug info
        logger.info(f"Timestamp: {current_time}")
        logger.info(f"Daily Resistance: {daily_res}")
        logger.info(f"Current Close: {current_close}")
        logger.info(f"BB Upper: {upper_bb}")
        logger.info(f"Volume MA: {vol_ma}")
        # Validate data
        if any([pd.isna(x) for x in [daily_res, current_close, upper_bb, vol_ma, atr_val]]):
            return
        # Breakout conditions
        breakout = (
            current_close > daily_res
            and current_close > upper_bb
            and current_volume > vol_ma * self.volume_factor
        )
        # Entry
        if not self.position and breakout:
            entry_price = current_close
            stop_loss = entry_price - atr_val * self.sl_atr_mult
            take_profit = entry_price * (1 + self.tp_percent / 100)
            risk_amount = 0.02 * self._broker.equity
            price_risk = entry_price - stop_loss
            if price_risk <= 0:
                return
            size = risk_amount / price_risk
            self.buy(size=size, sl=stop_loss, tp=take_profit)
            self.trade_count += 1 