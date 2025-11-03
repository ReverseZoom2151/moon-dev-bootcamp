"""
Liquidation Strategy with Entry Delay (Alpha Decay Testing)
===========================================================
Day 22: Extension of liquidation strategy that introduces entry delays to test alpha decay.

Alpha Decay: Measures how strategy performance degrades as entry delays increase.
This helps determine the optimal execution speed and slippage tolerance.
"""

import numpy as np
import pandas as pd
# Import external backtesting package, avoiding conflict with gordon.backtesting
from ...utils.backtesting_import import get_backtesting_strategy

Strategy = get_backtesting_strategy()

if Strategy is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")

from .liquidation_short_sliq_strategy import LiquidationShortSLiqStrategy


class DelayedLiquidationShortStrategy(LiquidationShortSLiqStrategy):
    """
    Delayed Liquidation Short Strategy.
    
    Extends LiquidationShortSLiqStrategy to introduce a delay between signal
    and actual trade entry. Used for alpha decay analysis.
    """

    delay_minutes = 0  # Entry delay in minutes

    def init(self):
        """Initialize strategy with delay tracking."""
        super().init()
        self.trade_entry_bar = -1  # Track the bar index of entry

    def next(self):
        """Execute strategy logic with delay handling."""
        current_bar_idx = len(self.data.Close) - 1
        
        # If we are already in a position OR have just entered on this bar,
        # let the base class handle exits/management. Delay only affects entry.
        if self.position or self.trade_entry_bar == current_bar_idx:
            super().next()
            return
        
        # Delay Logic for Entry
        # Check if a potential SHORT entry signal occurred within the delay window.
        potential_entry_signal = False
        last_potential_entry_bar = -1
        
        # Look back through the delay window to find signals
        for lookback in range(1, self.delay_minutes + 1):
            if current_bar_idx - lookback < 0:
                break  # Bounds check
            
            prev_time = self.data.index[current_bar_idx - lookback]
            entry_start_time = prev_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            
            # Check the short entry condition on the past bar
            past_short_liquidations = self.short_liquidations[
                entry_start_idx:current_bar_idx - lookback + 1
            ].sum()
            
            if past_short_liquidations >= self.short_liquidation_thresh:
                potential_entry_signal = True
                last_potential_entry_bar = current_bar_idx - lookback
                break  # Found the most recent signal within the delay window
        
        # If a signal occurred within the delay window, and we haven't entered yet
        if potential_entry_signal and self.trade_entry_bar < last_potential_entry_bar:
            # Calculate bars passed since the signal should have occurred
            bars_since_signal = current_bar_idx - last_potential_entry_bar
            
            # Check if the delay period has passed
            if bars_since_signal >= self.delay_minutes:
                # Delay is over, execute the SHORT entry logic NOW using current price
                if not self.position:  # Double check we aren't in a position
                    current_price = self.data.Close[-1]
                    sl_price = current_price * (1 + self.stop_loss_pct)
                    tp_price = current_price * (1 - self.take_profit_pct)
                    self.sell(sl=sl_price, tp=tp_price)
                    self.trade_entry_bar = current_bar_idx  # Mark entry bar as NOW
                    # We entered due to delay, so return here and don't run super().next() for this bar
                    return
        
        # If no potential signal was found in the lookback, or the delay hasn't passed,
        # or we already processed an entry for the signal,
        # run the base logic normally to check for a *new* signal on the *current* bar.
        super().next()

