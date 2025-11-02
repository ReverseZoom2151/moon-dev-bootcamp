import numpy as np
from backtesting import Backtest, Strategy
import pandas as pd
import warnings
import os
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

class LongOnLLiqStrategy(Strategy):
    # --- Strategy Parameters (to be optimized) ---
    l_liq_entry_thresh = 100000    # L LIQ volume threshold to trigger entry
    entry_time_window_mins = 5     # Lookback window for L LIQ entry signal (minutes)
    
    s_liq_closure_thresh = 50000   # S LIQ volume threshold to trigger exit
    exit_time_window_mins = 5      # Lookback window for S LIQ exit signal (minutes)
    
    take_profit = 0.02             # Take profit percentage (e.g., 0.02 = 2%)
    stop_loss = 0.01               # Stop loss percentage (e.g., 0.01 = 1%)

    def init(self):
        # Pre-calculate or access the required data columns
        self.l_liq_volume = self.data.l_liq_volume
        self.s_liq_volume = self.data.s_liq_volume

    def next(self):
        current_time = self.data.index[-1]

        # --- Entry Logic --- 
        if not self.position: # Only check for entry if not already in a position
            entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_l_liquidations = self.l_liq_volume[entry_start_idx:].sum()

            # Enter long if L LIQ exceeds threshold
            if recent_l_liquidations >= self.l_liq_entry_thresh:
                sl_price = self.data.Close[-1] * (1 - self.stop_loss)
                tp_price = self.data.Close[-1] * (1 + self.take_profit)
                self.buy(sl=sl_price, tp=tp_price)

        # --- Exit Logic (based on S LIQ threshold, in addition to TP/SL) --- 
        elif self.position: # Only check for S LIQ exit if in a position
            exit_start_time = current_time - pd.Timedelta(minutes=self.exit_time_window_mins)
            exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
            recent_s_liquidations = self.s_liq_volume[exit_start_idx:].sum()

            # Close position if recent S LIQ exceeds threshold
            if recent_s_liquidations >= self.s_liq_closure_thresh:
                self.position.close(reason="S Liq Threshold Hit")

# --- Configuration ---
symbol = "SOL"  # Define the symbol to process (e.g., "SOL", "WIF", "ETH")
output_dir = './output_data'
data_path = os.path.join(output_dir, f'{symbol}_liq_data.csv')

# --- Data Loading and Preprocessing ---
try:
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    print(f"Please ensure 'cutitup.py' has been run successfully and {symbol}_liq_data.csv exists in {output_dir}.")
    exit()

# Keep only necessary columns initially
if not all(col in data.columns for col in ['LIQ_SIDE', 'price', 'usd_size']):
    print(f"Error: Input CSV {data_path} missing required columns ('LIQ_SIDE', 'price', 'usd_size').")
    exit()
data = data[['LIQ_SIDE', 'price', 'usd_size']]

# --- Feature Engineering --- 

# Create L LIQ and S LIQ volume columns using vectorized operations
data['l_liq_volume'] = np.where(data['LIQ_SIDE'] == 'L LIQ', data['usd_size'], 0)
data['s_liq_volume'] = np.where(data['LIQ_SIDE'] == 'S LIQ', data['usd_size'], 0)

# Resample data to 1-minute frequency for backtesting
agg_funcs = {
    'price': 'mean',         # Mean price of liquidations in the minute
    'l_liq_volume': 'sum',   # Sum of L LIQ volume in the minute
    's_liq_volume': 'sum'    # Sum of S LIQ volume in the minute
}
data_resampled = data.resample('T').agg(agg_funcs)

# Create OHLC columns required by backtesting.py
data_resampled['Open'] = data_resampled['price']
data_resampled['High'] = data_resampled['price']
data_resampled['Low'] = data_resampled['price']
data_resampled['Close'] = data_resampled['price']

# Handle missing data after resampling
price_columns = ['price', 'Open', 'High', 'Low', 'Close']
data_resampled[price_columns] = data_resampled[price_columns].ffill()
volume_columns = ['l_liq_volume', 's_liq_volume']
data_resampled[volume_columns] = data_resampled[volume_columns].fillna(0)

# Keep price column for reference, drop if not needed
# data_resampled.drop(columns=['price'], inplace=True)

# Ensure the DataFrame is sorted by index
data_resampled.sort_index(inplace=True)

# Remove rows with NaN in critical columns (like Close) before backtesting
data_resampled.dropna(subset=['Close'], inplace=True)

# --- Data Verification ---
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

# --- Backtesting and Optimization ---
print(f"\n--- Running Backtest & Optimization for {symbol} ---")
bt = Backtest(data_resampled, LongOnLLiqStrategy, cash=100000, commission=0.002)

# Define optimization ranges (adjust these based on data and expectations)
optimization_results = bt.optimize(
    l_liq_entry_thresh=range(10000, 500000, 10000),      # L LIQ threshold for entry
    entry_time_window_mins=range(1, 11, 1),              # Entry lookback window
    s_liq_closure_thresh=range(10000, 500000, 10000),    # S LIQ threshold for exit
    exit_time_window_mins=range(1, 11, 1),               # Exit lookback window
    take_profit=[i / 1000 for i in range(5, 31, 5)],     # TP: 0.5% to 3.0% in 0.5% steps
    stop_loss=[i / 1000 for i in range(5, 31, 5)],       # SL: 0.5% to 3.0% in 0.5% steps
    maximize='Equity Final [$]',
    # Example constraint: ensure TP > SL
    # constraint=lambda p: p.take_profit > p.stop_loss 
)

# --- Results ---
print("\n--- Optimization Results ---")
print(optimization_results)

print("\n--- Best Parameters Found ---")
if optimization_results._strategy:
    best_params = optimization_results._strategy.__dict__
    print(f"L Liq Entry Threshold: {best_params.get('l_liq_entry_thresh', 'N/A')}")
    print(f"Entry Time Window (mins): {best_params.get('entry_time_window_mins', 'N/A')}")
    print(f"S Liq Closure Threshold: {best_params.get('s_liq_closure_thresh', 'N/A')}")
    print(f"Exit Time Window (mins): {best_params.get('exit_time_window_mins', 'N/A')}")
    print(f"Take Profit: {best_params.get('take_profit', 'N/A')}")
    print(f"Stop Loss: {best_params.get('stop_loss', 'N/A')}")
else:
    print("Optimization did not identify a best strategy.")

# Optional: Plot the results of the best strategy
print("\n--- Plotting Best Strategy Performance ---")
try:
    bt.plot()
except Exception as e:
    print(f"Could not generate plot: {e}")