"""
SMA-based Mean Reversion Strategy (Day 20)
==========================================
Trading strategy designed for ranging markets with reversion to mean behavior.

Trading Logic:
- Buy when price drops below SMA by buy_pct percentage
- Sell when price rises above SMA by sell_pct percentage
- Optional stop loss and take profit levels
- Designed for ranging markets with reversion to mean behavior
"""

import numpy as np
# Import external backtesting package, avoiding conflict with gordon.backtesting
from ...utils.backtesting_import import get_backtesting_strategy, get_backtesting_sma

Strategy = get_backtesting_strategy()
SMA = get_backtesting_sma()

if Strategy is None or SMA is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")


class MeanReversionStrategy(Strategy):
    """
    SMA-based Mean Reversion Strategy (Day 20).

    Trading logic:
    - Buy when price drops below SMA by buy_pct percentage
    - Sell when price rises above SMA by sell_pct percentage
    - Optional stop loss and take profit levels
    - Designed for ranging markets with reversion to mean behavior
    """

    # Strategy parameters
    sma_period = 14
    buy_pct = 10.0    # Buy when price is X% below SMA
    sell_pct = 15.0   # Sell when price is X% above SMA
    stop_loss = 0.0   # Optional stop loss percentage (0 = disabled)
    take_profit = 0.0 # Optional take profit percentage (0 = disabled)

    # Optimization ranges
    sma_period_range = [10, 14, 20, 30]
    buy_pct_range = [5, 10, 15, 20, 25]
    sell_pct_range = [5, 10, 15, 20, 25]

    def init(self):
        """Initialize indicators."""
        self.sma = self.I(SMA, self.data.Close, self.sma_period)

        # Trade tracking
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.position_entry_bar = None

    def next(self):
        """Execute strategy logic."""
        # Calculate thresholds
        buy_threshold = self.sma[-1] * (1 - self.buy_pct / 100)
        sell_threshold = self.sma[-1] * (1 + self.sell_pct / 100)
        price = self.data.Close[-1]

        # Skip if SMA not ready
        if np.isnan(self.sma[-1]) or not np.isfinite(self.sma[-1]):
            return

        # Entry logic - buy when price is below SMA by buy_pct
        if not self.position and price < buy_threshold:
            # Risk-based position sizing
            risk_amount = 0.02 * self.equity  # 2% risk per trade

            # Calculate stop loss if enabled
            if self.stop_loss > 0:
                sl_price = price * (1 - self.stop_loss / 100)
                price_risk = price - sl_price
                if price_risk > 0:
                    size = min(risk_amount / price_risk, 0.5)
                else:
                    size = 0.5
            else:
                sl_price = None
                size = 0.5  # Use 50% of equity if no stop loss

            # Calculate take profit if enabled
            if self.take_profit > 0:
                tp_price = price * (1 + self.take_profit / 100)
            else:
                tp_price = None

            # Place order
            self.buy(size=size, sl=sl_price, tp=tp_price)
            self.trade_count += 1
            self.position_entry_bar = len(self.data)

        # Exit logic - sell when price is above SMA by sell_pct
        elif self.position and price > sell_threshold:
            # Track win/loss
            try:
                pnl_pct = (price / self.position.price - 1) * 100
            except AttributeError:
                pnl_pct = 0

            if pnl_pct > 0:
                self.win_count += 1
            else:
                self.loss_count += 1

            self.position.close()
            self.position_entry_bar = None
