"""
SMA Crossover Strategy with Stop Loss and Trailing Stop (Day 13)
================================================================
Uses simple moving average crossover signals with risk management.

Entry: Price crosses above SMA
Exit: Stop loss or trailing stop triggered
Risk Management: Fixed stop loss and dynamic trailing stop
"""

import logging
import backtrader as bt

logger = logging.getLogger(__name__)


class SmaCrossStrategy(bt.SignalStrategy):
    """
    SMA Crossover Strategy with Stop Loss and Trailing Stop (Day 13).

    Uses simple moving average crossover signals with risk management:
    - Entry: Price crosses above SMA
    - Exit: Stop loss or trailing stop triggered
    - Risk Management: Fixed stop loss and dynamic trailing stop
    """

    params = (
        ('sma_period', 20),
        ('stop_loss', 0.02),  # 2% stop loss
        ('trailing_stop', 0.01),  # 1% trailing stop
    )

    def __init__(self):
        """Initialize indicators and signals."""
        self.sma = bt.ind.SMA(period=self.params.sma_period)
        self.price = self.data
        self.crossover = bt.ind.CrossOver(self.price, self.sma)

        # Add signal for long positions
        self.signal_add(bt.SIGNAL_LONG, self.crossover)

        # Track stop levels
        self.stop_loss = None
        self.trailing_stop = None
        self.order = None

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, '
                    f'Comm: {order.executed.comm:.2f}'
                )
                self.stop_loss = order.executed.price * (1 - self.params.stop_loss)
                self.trailing_stop = order.executed.price * (1 - self.params.trailing_stop)
            else:
                logger.info(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, '
                    f'Comm: {order.executed.comm:.2f}'
                )
                self.stop_loss = None
                self.trailing_stop = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        """Execute strategy logic on each bar."""
        if self.order:
            return

        # Check trailing stop
        if self.position:
            if self.data.close[0] > self.trailing_stop:
                self.trailing_stop = self.data.close[0] * (1 - self.params.trailing_stop)
            elif self.data.close[0] < self.trailing_stop:
                self.order = self.sell()
                logger.info(f'TRAILING STOP TRIGGERED at {self.data.close[0]:.2f}')

        # Check stop loss
        if self.position and self.data.close[0] < self.stop_loss:
            self.order = self.sell()
            logger.info(f'STOP LOSS TRIGGERED at {self.data.close[0]:.2f}')
