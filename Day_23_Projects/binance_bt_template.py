import os
import pandas as pd
import warnings
import sys
from backtesting import Backtest, Strategy
from pykalman import KalmanFilter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.binance_nice_funcs import create_exchange

warnings.filterwarnings('ignore')

INITIAL_CASH = 100000
COMMISSION_RATE = 0.002
OPTIMIZATION_ITERATIONS = 50
OUTPUT_FILENAME = 'binance_backtest_results.txt'

class KalmanBreakoutReversal(Strategy):
    window = 50
    take_profit = 0.05
    stop_loss = 0.03
    def init(self):
        self.kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], initial_state_mean=0, initial_state_covariance=1, observation_covariance=1, transition_covariance=0.01)
        try:
            self.filtered_state_means, _ = self.kf.filter(self.data.Close)
        except ValueError:
            self.filtered_state_means = np.full_like(self.data.Close, np.nan)
    def next(self):
        if len(self.data.Close) < 2 or np.isnan(self.filtered_state_means[-1]):
            return
        filtered_mean = self.filtered_state_means[-1]
        current_close = self.data.Close[-1]
        if not self.position.is_short and current_close > filtered_mean:
            if not self.position:
                self.sell(sl=current_close * (1 + self.stop_loss), tp=current_close * (1 - self.take_profit))
        elif self.position.is_short and current_close < filtered_mean:
            self.position.close(reason='Price reverted below KF mean')
        if not self.position.is_long and current_close < filtered_mean:
            if not self.position:
                self.buy(sl=current_close * (1 - self.stop_loss), tp=current_close * (1 + self.take_profit))
        elif self.position.is_long and current_close > filtered_mean:
            self.position.close(reason='Price reverted above KF mean')

def fetch_ohlcv(exchange, symbol, timeframe='1d', limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_backtest(data, strategy_class, cash, commission, **strategy_params):
    bt = Backtest(data, strategy_class, cash=cash, commission=commission)
    try:
        stats = bt.run(**strategy_params)
        return stats
    except Exception as e:
        print(f'Error running backtest: {e}')
        return None

def run_optimization(data, strategy_class, cash, commission, params_to_optimize, maximize, n_iter):
    bt = Backtest(data, strategy_class, cash=cash, commission=commission)
    try:
        results = bt.optimize(**params_to_optimize, maximize=maximize, method='skopt', n_iter=n_iter)
        return results
    except ImportError:
        print('Error: scikit-optimize not found. Install it: pip install scikit-optimize')
        return None
    except Exception as e:
        print(f'Error during optimization: {e}')
        return None

def main():
    exchange = create_exchange()
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # Example symbols
    output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILENAME)
    with open(output_file_path, 'w') as file:
        file.write('--- Binance Backtest Results ---\n\n')
        for symbol in symbols:
            print(f'Fetching data for {symbol}...')
            data = fetch_ohlcv(exchange, symbol)
            if data is not None and not data.empty:
                stats_default = run_backtest(data, KalmanBreakoutReversal, INITIAL_CASH, COMMISSION_RATE)
                if stats_default is not None:
                    result_str = f'Default Results for {symbol}:\n{stats_default}\n{'-' * 80}\n'
                    print(result_str)
                    file.write(result_str)
                params_to_optimize = {
                    'window': (20, 100),
                    'take_profit': (0.01, 0.10),
                    'stop_loss': (0.01, 0.10),
                }
                optimization_results = run_optimization(data, KalmanBreakoutReversal, INITIAL_CASH, COMMISSION_RATE, params_to_optimize, 'Equity Final [$]', OPTIMIZATION_ITERATIONS)
                if optimization_results:
                    result_str = f'Optimized Results for {symbol}:\n{optimization_results}\n'
                    if optimization_results._strategy:
                        best_params = optimization_results._strategy
                        result_str += f'Best Parameters for {symbol}:\n  Window: {getattr(best_params, 'window', 'N/A')}\n  Take Profit: {getattr(best_params, 'take_profit', 'N/A')}\n  Stop Loss: {getattr(best_params, 'stop_loss', 'N/A')}\n'
                    result_str += f'{'=' * 80}\n'
                    print(result_str)
                    file.write(result_str)
    print(f'\nProcessing complete. Results saved to {output_file_path}')

if __name__ == '__main__':
    import numpy as np
    main() 