import logging, os
import backtrader as bt
import backtrader.analyzers as btanalyzers

logger = logging.getLogger(__name__)

class SmaCross(bt.SignalStrategy):
    params = (
        ('sma_period', 20),
        ('stop_loss', 0.02),
        ('trailing_stop', 0.01),
    )

    def __init__(self):
        self.sma = bt.ind.SMA(period=self.params.sma_period)
        self.price = self.data
        self.crossover = bt.ind.CrossOver(self.price, self.sma)
        self.signal_add(bt.SIGNAL_LONG, self.crossover)
        self.stop_loss = None
        self.trailing_stop = None
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f'BUY EXECUTED at {order.executed.price:.2f}')
                self.stop_loss = order.executed.price * (1 - self.params.stop_loss)
                self.trailing_stop = order.executed.price * (1 - self.params.trailing_stop)
            else:
                logger.info(f'SELL EXECUTED at {order.executed.price:.2f}')
                self.stop_loss = None
                self.trailing_stop = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning('Order Cancelled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.order:
            return
        if self.position:
            if self.data.close[0] > self.trailing_stop:
                self.trailing_stop = self.data.close[0] * (1 - self.params.trailing_stop)
            elif self.data.close[0] < self.trailing_stop:
                self.order = self.sell()
                logger.info(f'TRAILING STOP at {self.data.close[0]:.2f}')
            elif self.data.close[0] < self.stop_loss:
                self.order = self.sell()
                logger.info(f'STOP LOSS at {self.data.close[0]:.2f}')


def run_backtrader(data_file: str, initial_cash: float = 1000000, commission: float = 0.001) -> dict:
    """Run backtrader backtest with SmaCross strategy on CSV file"""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = bt.feeds.YahooFinanceCSVData(dataname=data_file)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.adddata(data)
    cerebro.addsizer(bt.sizers.AllInSizer, percents=95)
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', None)
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.total.closed if hasattr(trades, 'total') else None
    total_return = (final_value / initial_cash - 1) if initial_cash else None
    return {
        'strategy': 'SmaCross',
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': drawdown,
        'total_trades': total_trades
    } 