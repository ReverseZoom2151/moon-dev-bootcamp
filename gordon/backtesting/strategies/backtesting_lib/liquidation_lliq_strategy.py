"""
Liquidation-Based Strategy: Long on L LIQ
=========================================
Day 21: Trading strategy that enters long positions on L LIQ and exits on S LIQ.

Strategy Logic:
- Enter long when L LIQ (Long Liquidations) volume exceeds threshold
- Exit when S LIQ (Short Liquidations) volume exceeds threshold
- More sophisticated entry/exit logic than S LIQ-only strategy
"""

import numpy as np
import pandas as pd
# Import external backtesting package, avoiding conflict with gordon.backtesting
from ...utils.backtesting_import import get_backtesting_strategy, get_backtesting_sma

Strategy = get_backtesting_strategy()
SMA = get_backtesting_sma()

if Strategy is None or SMA is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")


class LiquidationLLiqStrategy(Strategy):
    """
    Long on L LIQ (Long Liquidations) Strategy.
    
    Enters long positions when long liquidations exceed threshold,
    and exits when short liquidations exceed threshold.
    """

    # Strategy parameters
    l_liq_entry_thresh = 100000  # L LIQ volume threshold to trigger entry
    entry_time_window_mins = 5  # Lookback window for L LIQ entry signal (minutes)
    
    s_liq_closure_thresh = 50000  # S LIQ volume threshold to trigger exit
    exit_time_window_mins = 5  # Lookback window for S LIQ exit signal (minutes)
    
    take_profit = 0.02  # Take profit percentage (e.g., 0.02 = 2%)
    stop_loss = 0.01  # Stop loss percentage (e.g., 0.01 = 1%)

    # Optimization ranges
    l_liq_entry_thresh_range = range(10000, 500000, 10000)
    entry_time_window_mins_range = range(1, 11, 1)
    s_liq_closure_thresh_range = range(10000, 500000, 10000)
    exit_time_window_mins_range = range(1, 11, 1)
    take_profit_range = [i / 100 for i in range(1, 5, 1)]  # 1% to 4%
    stop_loss_range = [i / 100 for i in range(1, 5, 1)]  # 1% to 4%

    def init(self):
        """Initialize indicators."""
        # Ensure required columns exist
        if 'l_liq_volume' not in self.data.columns:
            raise ValueError("Data must contain 'l_liq_volume' column")
        if 's_liq_volume' not in self.data.columns:
            raise ValueError("Data must contain 's_liq_volume' column")
        
        # Pre-calculate or access the required data columns
        self.l_liq_volume = self.data.l_liq_volume
        self.s_liq_volume = self.data.s_liq_volume

    def next(self):
        """Execute strategy logic."""
        current_time = self.data.index[-1]
        current_price = self.data.Close[-1]

        # Entry Logic - Only check for entry if not already in a position
        if not self.position:
            entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_l_liquidations = self.l_liq_volume[entry_start_idx:].sum()

            # Enter long if L LIQ exceeds threshold
            if recent_l_liquidations >= self.l_liq_entry_thresh:
                sl_price = current_price * (1 - self.stop_loss)
                tp_price = current_price * (1 + self.take_profit)
                self.buy(sl=sl_price, tp=tp_price)

        # Exit Logic - Based on S LIQ threshold, in addition to TP/SL
        elif self.position:
            exit_start_time = current_time - pd.Timedelta(minutes=self.exit_time_window_mins)
            exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
            recent_s_liquidations = self.s_liq_volume[exit_start_idx:].sum()

            # Close position if recent S LIQ exceeds threshold
            if recent_s_liquidations >= self.s_liq_closure_thresh:
                self.position.close()

