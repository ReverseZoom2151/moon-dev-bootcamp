"""
StochRSI + Bollinger Bands Strategy (Day 16)
=============================================
Combines momentum and volatility indicators.

Entry: Price above lower BB + StochRSI K crosses above D
Exit: Stop loss or take profit
Position Sizing: Fixed percentage of equity
"""

import pandas as pd
# Import external backtesting package, avoiding conflict with gordon.backtesting
from ...utils.backtesting_import import get_backtesting_strategy, get_backtesting_crossover

Strategy = get_backtesting_strategy()
crossover = get_backtesting_crossover()

if Strategy is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")

from ta.momentum import StochRSIIndicator
from ta.volatility import BollingerBands


class StochRSIBollingerStrategy(Strategy):
    """
    StochRSI + Bollinger Bands Strategy (Day 16).

    Combines momentum and volatility indicators:
    - Entry: Price above lower BB + StochRSI K crosses above D
    - Exit: Stop loss or take profit
    - Position Sizing: Fixed percentage of equity
    """

    # Strategy parameters
    rsi_window = 14
    stochrsi_smooth1 = 3
    stochrsi_smooth2 = 3
    bbands_length = 20
    bbands_std = 2
    position_size = 0.05  # 5% of equity
    stop_loss_factor = 0.85  # 15% stop loss
    take_profit_factor = 1.40  # 40% take profit

    def init(self):
        """Initialize technical indicators."""
        close_data = self.data.Close

        # Bollinger Bands
        self.lower_band, self.mid_band, self.upper_band = self.I(
            self._calculate_bbands,
            close_data,
            window=self.bbands_length,
            window_dev=self.bbands_std
        )

        # StochRSI
        self.stoch_rsi_k, self.stoch_rsi_d = self.I(
            self._calculate_stochrsi,
            close_data,
            window=self.rsi_window,
            smooth1=self.stochrsi_smooth1,
            smooth2=self.stochrsi_smooth2
        )

    def _calculate_bbands(self, close_series, window, window_dev):
        """Calculate Bollinger Bands."""
        indicator_bb = BollingerBands(
            close=pd.Series(close_series),
            window=window,
            window_dev=window_dev
        )
        return (
            indicator_bb.bollinger_lband().to_numpy(),
            indicator_bb.bollinger_mavg().to_numpy(),
            indicator_bb.bollinger_hband().to_numpy()
        )

    def _calculate_stochrsi(self, close_series, window, smooth1, smooth2):
        """Calculate StochRSI."""
        indicator_stochrsi = StochRSIIndicator(
            close=pd.Series(close_series),
            window=window,
            smooth1=smooth1,
            smooth2=smooth2
        )
        return (
            indicator_stochrsi.stochrsi_k().to_numpy(),
            indicator_stochrsi.stochrsi_d().to_numpy()
        )

    def next(self):
        """Execute strategy logic."""
        # Wait for enough data
        if len(self.data) < max(self.bbands_length, self.rsi_window):
            return

        # Check entry conditions
        price_above_lower = self.data.Close[-1] > self.lower_band[-1]
        stoch_cross = crossover(self.stoch_rsi_k, self.stoch_rsi_d)

        if price_above_lower and stoch_cross:
            if not pd.isna(self.data.Close[-1]) and self.data.Close[-1] > 0:
                # Calculate stop loss and take profit
                sl = self.data.Close[-1] * self.stop_loss_factor
                tp = self.data.Close[-1] * self.take_profit_factor

                # Enter position
                self.buy(size=self.position_size, sl=sl, tp=tp)
