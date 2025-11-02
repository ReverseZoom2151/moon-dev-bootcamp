######## BACKTESTING with backtrader 2024 (Bitfinex Version)
import sys, pandas as pd, logging, os, backtrader as bt, backtrader.analyzers as btanalyzers
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.bitfinex_nice_funcs import create_exchange

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmaCross(bt.SignalStrategy):
    params = (('sma_period', 20), ('stop_loss', 0.02), ('trailing_stop', 0.01),)
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
                logger.info(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.stop_loss = order.executed.price * (1 - self.params.stop_loss)
                self.trailing_stop = order.executed.price * (1 - self.params.trailing_stop)
            else:
                logger.info(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.stop_loss = None
                self.trailing_stop = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning('Order Canceled/Margin/Rejected')
        self.order = None
    def next(self):
        if self.order:
            return
        if self.position:
            if self.data.close[0] > self.trailing_stop:
                self.trailing_stop = self.data.close[0] * (1 - self.params.trailing_stop)
            elif self.data.close[0] < self.trailing_stop:
                self.order = self.sell()
                logger.info(f'TRAILING STOP TRIGGERED at {self.data.close[0]:.2f}')
        elif self.position and self.data.close[0] < self.stop_loss:
            self.order = self.sell()
            logger.info(f'STOP LOSS TRIGGERED at {self.data.close[0]:.2f}')

def fetch_bitfinex_data(symbol='BTC:USDT', timeframe='1d', fromdate=datetime(2017,1,6), todate=datetime(2022,5,4)):
    exchange = create_exchange()
    since = int(fromdate.timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[df.index <= todate]
    return df

def run_backtest(initial_cash=1000000, commission=0.001):
    try:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SmaCross)
        df = fetch_bitfinex_data()
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        cerebro.broker.set_cash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.addsizer(bt.sizers.AllInSizer, percents=95)
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(btanalyzers.Transactions, _name='tx')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        logger.info("Starting backtest...")
        results = cerebro.run()
        if not results:
            raise ValueError("No results returned from backtest")
        strategy = results[0]
        print("\n=== Backtest Results ===")
        print(f"Initial Portfolio Value: ${initial_cash:,.2f}")
        print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        print(f"Total Return: {((cerebro.broker.getvalue() / initial_cash - 1) * 100):.2f}%")
        try:
            sharpe = strategy.analyzers.sharpe.get_analysis()
            drawdown = strategy.analyzers.drawdown.get_analysis()
            transactions = strategy.analyzers.tx.get_analysis()
            print(f"\nSharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
            print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A')}%")
            print(f"Number of Transactions: {len(transactions)}")
        except Exception as e:
            logger.error(f"Error processing analyzer results: {str(e)}")
            print("\nError processing some analyzer results. Check logs for details.")
        return cerebro, results
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        cerebro, results = run_backtest()
        cerebro.plot()
    except Exception as e:
        logger.error(f"Failed to execute backtest: {str(e)}")
        print("\nBacktest failed. Check the logs above for details.") 