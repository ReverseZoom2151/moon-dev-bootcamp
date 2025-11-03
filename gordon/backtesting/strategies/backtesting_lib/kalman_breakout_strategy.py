"""
Kalman Filter Breakout/Reversal Strategy
=========================================
Day 23: Trading strategy using Kalman Filter for price smoothing and mean reversion signals.

Strategy Logic:
- Uses Kalman Filter to smooth closing prices
- Enters SHORT when price breaks ABOVE the filtered mean (anticipating reversal)
- Enters LONG when price breaks BELOW the filtered mean (anticipating reversal)
- Exits when price crosses back over the mean
"""

import numpy as np
# Import external backtesting package, avoiding conflict with gordon.backtesting
from ...utils.backtesting_import import get_backtesting_strategy

Strategy = get_backtesting_strategy()

if Strategy is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")

try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    KalmanFilter = None


class KalmanBreakoutReversalStrategy(Strategy):
    """
    Kalman Filter Breakout/Reversal Strategy.
    
    Uses a Kalman Filter on closing prices to identify mean reversion opportunities.
    Enters trades betting on reversal when price breaks the filtered mean.
    """
    
    # Parameters to be optimized
    window = 50  # Window parameter (used indirectly by Kalman filter)
    take_profit = 0.05  # 5% take profit
    stop_loss = 0.03  # 3% stop loss
    
    # Optimization ranges
    window_range = range(20, 100, 10)
    take_profit_range = [i / 100 for i in range(1, 11, 1)]  # 1% to 10%
    stop_loss_range = [i / 100 for i in range(1, 11, 1)]  # 1% to 10%

    def init(self):
        """Initialize the Kalman Filter."""
        if not KALMAN_AVAILABLE:
            raise ImportError(
                "pykalman package not found. Install with: pip install pykalman"
            )
        
        # Simple Kalman Filter setup for price smoothing
        self.kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,  # Measurement noise
            transition_covariance=0.01  # Process noise
        )
        
        # Apply the Kalman filter to the closing prices
        try:
            self.filtered_state_means, _ = self.kf.filter(self.data.Close)
        except ValueError:
            # Handle cases with insufficient data for filtering
            self.filtered_state_means = np.full_like(self.data.Close, np.nan)

    def next(self):
        """Define the trading logic for each bar."""
        # Ensure enough data and valid filter output
        if len(self.data.Close) < 2 or np.isnan(self.filtered_state_means[-1]):
            return
        
        filtered_mean = self.filtered_state_means[-1]
        current_close = self.data.Close[-1]
        
        # Short Entry
        # If price breaks significantly above the filtered mean, anticipate reversal (short)
        if not self.position.is_short and current_close > filtered_mean:
            if not self.position:  # Check again to ensure no position was opened
                self.sell(
                    sl=current_close * (1 + self.stop_loss),
                    tp=current_close * (1 - self.take_profit)
                )
        
        # Short Exit
        # Close short if price reverts below the filtered mean
        elif self.position.is_short and current_close < filtered_mean:
            self.position.close(reason="Price reverted below KF mean")
        
        # Long Entry
        # If price breaks significantly below the filtered mean, anticipate reversal (long)
        if not self.position.is_long and current_close < filtered_mean:
            if not self.position:  # Check again
                self.buy(
                    sl=current_close * (1 - self.stop_loss),
                    tp=current_close * (1 + self.take_profit)
                )
        
        # Long Exit
        # Close long if price reverts above the filtered mean
        elif self.position.is_long and current_close > filtered_mean:
            self.position.close(reason="Price reverted above KF mean")

