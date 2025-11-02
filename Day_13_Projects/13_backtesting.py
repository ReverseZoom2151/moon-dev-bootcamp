######## BACKTESTING with backtrader 2024
import logging, os, backtrader as bt, backtrader.analyzers as btanalyzers
from datetime import datetime 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmaCross(bt.SignalStrategy):
    """
    A strategy that generates buy signals when price crosses above the SMA
    and sell signals when price crosses below the SMA.
    """
    params = (
        ('sma_period', 20),
        ('stop_loss', 0.02),  # 2% stop loss
        ('trailing_stop', 0.01),  # 1% trailing stop
    )

    def __init__(self):
        # Initialize indicators
        self.sma = bt.ind.SMA(period=self.params.sma_period)
        self.price = self.data
        self.crossover = bt.ind.CrossOver(self.price, self.sma)
        
        # Add signals
        self.signal_add(bt.SIGNAL_LONG, self.crossover)
        
        # Initialize stop loss and trailing stop
        self.stop_loss = None
        self.trailing_stop = None
        
        # Track order status
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                # Set stop loss and trailing stop
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
        # Check if we have an open order
        if self.order:
            return

        # Update trailing stop if we have a position
        if self.position:
            if self.data.close[0] > self.trailing_stop:
                self.trailing_stop = self.data.close[0] * (1 - self.params.trailing_stop)
            elif self.data.close[0] < self.trailing_stop:
                self.order = self.sell()
                logger.info(f'TRAILING STOP TRIGGERED at {self.data.close[0]:.2f}')

        # Check stop loss if we have a position
        elif self.position and self.data.close[0] < self.stop_loss:
            self.order = self.sell()
            logger.info(f'STOP LOSS TRIGGERED at {self.data.close[0]:.2f}')

def run_backtest(data_file, initial_cash=1000000, commission=0.001):
    """
    Run the backtest with the specified parameters.
    
    Args:
        data_file (str): Path to the CSV data file
        initial_cash (float): Initial cash amount
        commission (float): Commission rate
    
    Returns:
        tuple: (cerebro instance, backtest results)
    """
    try:
        # Initialize cerebro engine
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SmaCross)

        # Load and validate data
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Read and validate CSV data
        try:
            data = bt.feeds.YahooFinanceCSVData(
                dataname=data_file,
                fromdate=datetime(2017, 1, 6),
                todate=datetime(2022, 5, 4),
                reverse=False
            )
            logger.info(f"Successfully loaded data from {data_file}")
        except Exception as e:
            logger.error(f"Error loading data file: {str(e)}")
            raise

        # Configure broker
        cerebro.broker.set_cash(initial_cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.adddata(data)
        cerebro.addsizer(bt.sizers.AllInSizer, percents=95)

        # Add analyzers
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(btanalyzers.Transactions, _name='tx')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')

        # Run backtest
        logger.info("Starting backtest...")
        results = cerebro.run()
        
        if not results:
            raise ValueError("No results returned from backtest")

        # Get the first strategy instance
        strategy = results[0]
        
        # Print results
        print("\n=== Backtest Results ===")
        print(f"Initial Portfolio Value: ${initial_cash:,.2f}")
        print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        print(f"Total Return: {((cerebro.broker.getvalue() / initial_cash - 1) * 100):.2f}%")
        
        # Print analyzer results with error handling
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
        # Check if data file exists before running
        data_file = 'Bitcoin-price-USD.csv'
        if not os.path.exists(data_file):
            logger.error(f"Data file '{data_file}' not found in current directory")
            print(f"Please ensure the file '{data_file}' exists in the current directory")
            exit(1)
            
        cerebro, results = run_backtest(data_file)
        cerebro.plot()
    except Exception as e:
        logger.error(f"Failed to execute backtest: {str(e)}")
        print("\nBacktest failed. Check the logs above for details.")