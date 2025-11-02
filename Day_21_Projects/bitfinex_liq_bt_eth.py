import numpy as np
import pandas as pd
import warnings
import os
from backtesting import Backtest, Strategy

warnings.filterwarnings('ignore')

class LongOnSLiqStrategy(Strategy):
    s_liq_entry_thresh = 100000
    entry_time_window_mins = 20
    take_profit = 0.02
    stop_loss = 0.01
    def init(self):
        self.s_liq_volume = self.data.s_liq_volume
    def next(self):
        current_time = self.data.index[-1]
        if not self.position:
            entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_s_liquidations = self.s_liq_volume[entry_start_idx:].sum()
            if recent_s_liquidations >= self.s_liq_entry_thresh:
                sl_price = self.data.Close[-1] * (1 - self.stop_loss)
                tp_price = self.data.Close[-1] * (1 + self.take_profit)
                self.buy(sl=sl_price, tp=tp_price)

symbol = "ETH:USDT"
output_dir = './output_data'
data_path = os.path.join(output_dir, f'{symbol.replace(":USDT", "")}_liq_data.csv')
try:
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    exit()
data = data[['LIQ_SIDE', 'price', 'usd_size']]
s_liq_data = data[data['LIQ_SIDE'] == 'S LIQ'].copy()
ohlc_agg = {'price': 'mean', 'usd_size': 'sum'}
data_resampled = s_liq_data.resample('T').agg(ohlc_agg)
data_resampled['Open'] = data_resampled['price']
data_resampled['High'] = data_resampled['price']
data_resampled['Low'] = data_resampled['price']
data_resampled['Close'] = data_resampled['price']
data_resampled.rename(columns={'usd_size': 's_liq_volume'}, inplace=True)
price_columns = ['price', 'Open', 'High', 'Low', 'Close']
data_resampled[price_columns] = data_resampled[price_columns].ffill()
data_resampled['s_liq_volume'].fillna(0, inplace=True)
data_resampled.sort_index(inplace=True)
data_resampled.dropna(subset=['Close'], inplace=True)
print(f"--- Data Processed for {symbol} ---")
print("Processed Data Head:")
print(data_resampled.head())
print("\nProcessed Data Info:")
print(data_resampled.info())
print("\nCheck for NaNs:")
print(data_resampled.isnull().sum())
if data_resampled.empty or data_resampled['Close'].isnull().all():
    print("Error: No valid data available for backtesting after processing.")
    exit()
print(f"\n--- Running Backtest & Optimization for {symbol} ---")
bt = Backtest(data_resampled, LongOnSLiqStrategy, cash=100000, commission=0.002)
optimization_results = bt.optimize(
    s_liq_entry_thresh=range(10000, 500000, 10000),
    entry_time_window_mins=range(5, 60, 5),
    take_profit=[i / 1000 for i in range(5, 31, 5)],
    stop_loss=[i / 1000 for i in range(5, 31, 5)],
    maximize='Equity Final [$]',
)
print("\n--- Optimization Results ---")
print(optimization_results)
print("\n--- Best Parameters Found ---")
if optimization_results._strategy:
    best_params = optimization_results._strategy.__dict__
    print(f"S Liq Entry Threshold: {best_params.get('s_liq_entry_thresh', 'N/A')}")
    print(f"Entry Time Window (mins): {best_params.get('entry_time_window_mins', 'N/A')}")
    print(f"Take Profit: {best_params.get('take_profit', 'N/A')}")
    print(f"Stop Loss: {best_params.get('stop_loss', 'N/A')}")
else:
    print("Optimization did not identify a best strategy.")
print("\n--- Plotting Best Strategy Performance ---")
try:
    bt.plot()
except Exception as e:
    print(f"Could not generate plot: {e}") 