"""
Liquidation-Based Strategy: Long on S LIQ
=========================================
Day 21: Trading strategy that enters long positions when short liquidations exceed threshold.

Strategy Logic:
- Enter long when S LIQ (Short Liquidations) volume exceeds threshold within time window
- Stop loss and take profit based on percentages
- Designed to capitalize on potential price reversals after short liquidations
"""

import numpy as np
import pandas as pd
# Import external backtesting package, avoiding conflict with gordon.backtesting
from ...utils.backtesting_import import get_backtesting_strategy, get_backtesting_sma

Strategy = get_backtesting_strategy()
SMA = get_backtesting_sma()

if Strategy is None or SMA is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")


class LiquidationSLiqStrategy(Strategy):
    """
    Long on S LIQ (Short Liquidations) Strategy.
    
    Enters long positions when short liquidations exceed threshold,
    indicating potential upward price movement.
    """

    # Strategy parameters
    s_liq_entry_thresh = 100000  # S LIQ volume threshold to trigger entry
    entry_time_window_mins = 20  # Lookback window for S LIQ entry signal (minutes)
    take_profit = 0.02  # Take profit percentage (e.g., 0.02 = 2%)
    stop_loss = 0.01  # Stop loss percentage (e.g., 0.01 = 1%)

    # Optimization ranges
    s_liq_entry_thresh_range = range(10000, 500000, 10000)
    entry_time_window_mins_range = range(5, 60, 5)
    take_profit_range = [i / 1000 for i in range(5, 31, 5)]  # 0.5% to 3.0% in 0.5% steps
    stop_loss_range = [i / 1000 for i in range(5, 31, 5)]  # 0.5% to 3.0% in 0.5% steps

    def init(self):
        """Initialize indicators."""
        # Pre-calculate or access the required data columns
        # Ensure s_liq_volume column exists in data
        if 's_liq_volume' not in self.data.columns:
            raise ValueError("Data must contain 's_liq_volume' column")
        
        self.s_liq_volume = self.data.s_liq_volume

    def next(self):
        """Execute strategy logic."""
        current_time = self.data.index[-1]
        
        # Only check for entry if not already in a position
        if not self.position:
            entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_s_liquidations = self.s_liq_volume[entry_start_idx:].sum()
            
            # Enter long if S LIQ exceeds threshold
            if recent_s_liquidations >= self.s_liq_entry_thresh:
                current_price = self.data.Close[-1]
                sl_price = current_price * (1 - self.stop_loss)
                tp_price = current_price * (1 + self.take_profit)
                self.buy(sl=sl_price, tp=tp_price)

