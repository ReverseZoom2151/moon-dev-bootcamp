"""
Liquidation-Based Strategy: Short on S LIQ
===========================================
Day 22: Trading strategy that enters SHORT positions when short liquidations exceed threshold.

Strategy Logic:
- Enter SHORT when S LIQ (Short Liquidations) volume exceeds threshold within time window
- Exit when L LIQ (Long Liquidations) volume exceeds threshold
- Stop loss and take profit based on percentages
- Designed to capitalize on potential downward price movement after short liquidations
"""

import numpy as np
import pandas as pd
# Import external backtesting package, avoiding conflict with gordon.backtesting
from ...utils.backtesting_import import get_backtesting_strategy

Strategy = get_backtesting_strategy()

if Strategy is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")


class LiquidationShortSLiqStrategy(Strategy):
    """
    Short on S LIQ (Short Liquidations) Strategy.
    
    Enters SHORT positions when short liquidations exceed threshold,
    indicating potential downward price movement.
    """

    # Strategy parameters
    short_liquidation_thresh = 100000  # S LIQ volume threshold to trigger SHORT entry
    entry_time_window_mins = 5  # Lookback window for S LIQ entry signal (minutes)
    long_liquidation_closure_thresh = 50000  # L LIQ volume threshold to trigger SHORT exit
    exit_time_window_mins = 5  # Lookback window for L LIQ exit signal (minutes)
    take_profit_pct = 0.02  # Take profit percentage for shorts (e.g., 0.02 = 2% down)
    stop_loss_pct = 0.01  # Stop loss percentage for shorts (e.g., 0.01 = 1% up)

    # Optimization ranges
    short_liquidation_thresh_range = range(10000, 500000, 10000)
    entry_time_window_mins_range = range(1, 11, 1)
    long_liquidation_closure_thresh_range = range(10000, 500000, 10000)
    exit_time_window_mins_range = range(1, 11, 1)
    take_profit_pct_range = [i / 100 for i in range(1, 5, 1)]  # 1% to 4%
    stop_loss_pct_range = [i / 100 for i in range(1, 5, 1)]  # 1% to 4%

    def init(self):
        """Initialize indicators."""
        # Ensure required columns exist
        if 'short_liquidations' not in self.data.columns:
            raise ValueError("Data must contain 'short_liquidations' column")
        if 'long_liquidations' not in self.data.columns:
            raise ValueError("Data must contain 'long_liquidations' column")
        
        self.short_liquidations = self.data.short_liquidations
        self.long_liquidations = self.data.long_liquidations

    def next(self):
        """Execute strategy logic."""
        current_time = self.data.index[-1]
        
        # Entry Logic - Only check for entry if not already in a position
        if not self.position:
            entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_short_liquidations = self.short_liquidations[entry_start_idx:].sum()
            
            # Enter SHORT if S LIQ exceeds threshold
            if recent_short_liquidations >= self.short_liquidation_thresh:
                current_price = self.data.Close[-1]
                sl_price = current_price * (1 + self.stop_loss_pct)  # Stop loss above for shorts
                tp_price = current_price * (1 - self.take_profit_pct)  # Take profit below for shorts
                self.sell(sl=sl_price, tp=tp_price)

        # Exit Logic - Based on L LIQ threshold, in addition to TP/SL
        elif self.position.is_short:  # Only check for L LIQ exit if in a short position
            exit_start_time = current_time - pd.Timedelta(minutes=self.exit_time_window_mins)
            exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
            recent_long_liquidations = self.long_liquidations[exit_start_idx:].sum()
            
            # Close position if recent L LIQ exceeds threshold
            if recent_long_liquidations >= self.long_liquidation_closure_thresh:
                self.position.close()

