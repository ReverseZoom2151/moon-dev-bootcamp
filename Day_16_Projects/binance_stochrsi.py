import pandas as pd
import warnings
import sys, os
from backtesting import Backtest, Strategy
from ta.momentum import StochRSIIndicator
from ta.volatility import BollingerBands
from backtesting.lib import crossover

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Day_12_Projects.binance_nice_funcs import create_exchange

warnings.filterwarnings("ignore")

PRICE_SCALING_FACTOR = 1000
STOP_LOSS_FACTOR = 0.85
TAKE_PROFIT_FACTOR = 1.40
INITIAL_CASH = 1_000_000
POSITION_SIZE = 0.05

def calculate_bbands(close_series, window, window_dev):
    indicator_bb = BollingerBands(close=pd.Series(close_series), window=window, window_dev=window_dev)
    return (
        indicator_bb.bollinger_lband().to_numpy(),
        indicator_bb.bollinger_mavg().to_numpy(),
        indicator_bb.bollinger_hband().to_numpy()
    )

def calculate_stochrsi(close_series, window, smooth1, smooth2):
    indicator_stochrsi = StochRSIIndicator(close=pd.Series(close_series), window=window, smooth1=smooth1, smooth2=smooth2)
    return (
        indicator_stochrsi.stochrsi_k().to_numpy(),
        indicator_stochrsi.stochrsi_d().to_numpy()
    )

class Strat(Strategy):
    rsi_window = 14
    stochrsi_smooth1 = 3
    stochrsi_smooth2 = 3 
    bbands_length = 20
    bbands_std = 2
    def init(self):
        close_data = self.data.Close
        self.lower_band, self.mid_band, self.upper_band = self.I(
            calculate_bbands,
            close_data, window=self.bbands_length, window_dev=self.bbands_std
        )
        self.stoch_rsi_k, self.stoch_rsi_d = self.I(
            calculate_stochrsi,
            close_data, window=self.rsi_window, smooth1=self.stochrsi_smooth1, smooth2=self.stochrsi_smooth2
        )
    def next(self):
        lower = self.lower_band
        if len(self.data) < max(self.bbands_length, self.rsi_window):
             return
        if (self.data.Close[-1] > lower[-1] and crossover(self.stoch_rsi_k, self.stoch_rsi_d)):
            if not pd.isna(self.data.Close[-1]) and self.data.Close[-1] > 0:
                 self.buy(size=POSITION_SIZE, sl=self.data.Close[-1] * STOP_LOSS_FACTOR, tp=self.data.Close[-1] * TAKE_PROFIT_FACTOR)

def fetch_data(symbol='ETHUSDT', timeframe='1h', limit=2000):
    print(f"Attempting to fetch data for {symbol} ({timeframe})...")
    exchange = create_exchange()
    since = exchange.milliseconds() - (limit * exchange.parse_timeframe(timeframe) * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not ohlcv:
        print(f"Warning: No OHLCV data returned for {symbol} {timeframe}.")
        return pd.DataFrame()
    print(f"Successfully fetched {len(ohlcv)} candles for {symbol}.")
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
         df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    return df

def run_backtest():
    data_df = fetch_data()
    if data_df.empty:
         print("\nCould not fetch data. Please check API keys and exchange connectivity.")
         return
    print("\nSample Data (Last 5 rows):")
    print(data_df.tail())
    try:
        backtester = Backtest(data_df, Strat, cash=INITIAL_CASH, commission=.001)
        stats = backtester.run()
        backtester.plot()
        print("\nBacktest Stats:")
        print(stats)
    except Exception as backtest_error:
         print(f"\nAn error occurred during backtesting: {backtest_error}")
         import traceback
         traceback.print_exc()

if __name__ == "__main__":
    run_backtest() 