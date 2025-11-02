"""
Moving Average (MA) Reversal Strategy V2
---
This strategy is based on the Day 49 `2xmareversal.py` script.

It uses two Simple Moving Averages (SMA) to generate trading signals.

Entry Rules:
- Long Position: Enter when price is above both moving averages.
- Short Position: Enter when price is above the fast MA but below the slow MA.

Risk Management:
- Uses percentage-based take profit and stop loss.
- Short positions have an additional exit rule: close if the price moves above the slow MA.
"""

from backtesting import Strategy
from backtesting.lib import crossover
import talib

class MAReversalStrategyV2(Strategy):
    """
    Implementation of the 2x MA Reversal Strategy.
    """
    # Default parameters (can be overridden during backtest)
    ma_fast = 20
    ma_slow = 40
    take_profit = 0.10  # 10%
    stop_loss = 0.10    # 10%

    def init(self):
        """
        Initialize the strategy indicators.
        """
        # Calculate moving averages using TA-Lib
        self.sma_fast = self.I(talib.SMA, self.data.Close, self.ma_fast)
        self.sma_slow = self.I(talib.SMA, self.data.Close, self.ma_slow)

    def next(self):
        """
        Define the trading logic for the next iteration (bar).
        """
        price = self.data.Close[-1]
        
        # If we are already in a position, manage it
        if self.position:
            # Additional exit rule for short positions
            if self.position.is_short and price > self.sma_slow[-1]:
                self.position.close()
                return

        # If no position is open, check for new entry signals
        if not self.position:
            # Check for short setup: price > fast MA AND price < slow MA
            if price > self.sma_fast[-1] and price < self.sma_slow[-1]:
                self.sell(
                    sl=price * (1 + self.stop_loss),
                    tp=price * (1 - self.take_profit)
                )
            
            # Check for long setup: price > fast MA AND price > slow MA
            elif price > self.sma_fast[-1] and price > self.sma_slow[-1]:
                self.buy(
                    sl=price * (1 - self.stop_loss),
                    tp=price * (1 + self.take_profit)
                )
