import numpy as np
from backtesting import Backtest, Strategy
import pandas as pd
import warnings
import os
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

class LiquidationStrategy(Strategy):
    # Optimization parameters with defaults
    liquidation_thresh = 100000  # Threshold for S LIQ volume
    time_window_mins = 20      # Lookback window in minutes
    take_profit = 0.02         # Take profit percentage
    stop_loss = 0.01           # Stop loss percentage

    def init(self):
        # Use the precalculated S LIQ volume column
        self.s_liq_volume = self.data.s_liq_volume

    def next(self):
        current_time = self.data.index[-1]
        start_time = current_time - pd.Timedelta(minutes=self.time_window_mins)

        # Slice the s_liq_volume series within the time window
        # Note: self.data.index should be sorted for searchsorted to work correctly
        start_idx = np.searchsorted(self.data.index, start_time, side='left')
        recent_s_liquidations = self.s_liq_volume[start_idx:].sum()

        # Entry condition: Buy if recent S LIQ volume exceeds threshold and not already in a position
        if recent_s_liquidations >= self.liquidation_thresh and not self.position:
            # Calculate TP/SL based on the current closing price
            sl_price = self.data.Close[-1] * (1 - self.stop_loss)
            tp_price = self.data.Close[-1] * (1 + self.take_profit)
            self.buy(sl=sl_price, tp=tp_price)

# --- Data Loading and Preprocessing ---

# Define symbol and directories
symbol = "BTC"
output_dir = './output_data'
data_path = os.path.join(output_dir, f'{symbol}_liq_data.csv')

# Load the data
try:
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")
    print("Please ensure 'cutitup.py' has been run successfully and the file exists.")
    exit()

# Keep only necessary columns initially
data = data[['symbol', 'LIQ_SIDE', 'price', 'usd_size']]

# --- Feature Engineering for Strategy --- 

# Filter for Short Liquidations (S LIQ) as these are the trigger for our BUY strategy
s_liq_data = data[data['LIQ_SIDE'] == 'S LIQ'].copy()

# Resample data to 1-minute frequency for backtesting
# Aggregate S LIQ volume and calculate mean price within each minute
# This assumes OHLC can be represented by the mean liquidation price in that minute
ohlc_agg = {
    'price': 'mean', # Use mean price of liquidations in the minute
    'usd_size': 'sum'   # Sum the volume of S LIQs in the minute
}
data_resampled = s_liq_data.resample('T').agg(ohlc_agg)

# Create OHLC columns required by backtesting.py
# Simplified approach: Use the mean price for Open, High, Low, Close
# This is a limitation as we don't have true OHLC data
data_resampled['Open'] = data_resampled['price']
data_resampled['High'] = data_resampled['price']
data_resampled['Low'] = data_resampled['price']
data_resampled['Close'] = data_resampled['price']

# Rename aggregated usd_size to represent S LIQ volume
data_resampled.rename(columns={'usd_size': 's_liq_volume'}, inplace=True)

# Handle missing data after resampling:
# Forward fill price-related columns to maintain price continuity
price_columns = ['price', 'Open', 'High', 'Low', 'Close']
data_resampled[price_columns] = data_resampled[price_columns].ffill()
# Fill missing liquidation volumes with 0, as no S LIQ occurred in those minutes
data_resampled['s_liq_volume'].fillna(0, inplace=True)

# Drop the original mean price column if no longer needed (OHLC columns exist)
# data_resampled.drop(columns=['price'], inplace=True)
# Keep price column for potential reference or plotting if needed

# Ensure the DataFrame is sorted by the index (should be by default after resample)
data_resampled.sort_index(inplace=True)

# Remove rows with NaN in critical columns (e.g., Close) before backtesting
data_resampled.dropna(subset=['Close'], inplace=True)

# Verify the processed data
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

# Create and configure the backtest
bt = Backtest(data_resampled, LiquidationStrategy, cash=100000, commission=0.002)

# Optimization parameters
# Reduced range for threshold based on potential S LIQ values
optimization_results = bt.optimize(
    liquidation_thresh=range(5000, 500000, 5000),  # Adjusted range for S LIQ
    time_window_mins=range(5, 60, 5),            # Time window range
    take_profit=[i / 100 for i in range(1, 5, 1)],    # TP range 1% to 4%
    stop_loss=[i / 100 for i in range(1, 5, 1)],      # SL range 1% to 4%
    maximize='Equity Final [$]',
    # constraint=lambda param: param.take_profit > param.stop_loss # Optional constraint
)

# Print the optimization results overview
print("\nOptimization Results:")
print(optimization_results)

# Print the best optimized parameters
print("\nBest Parameters Found:")
if optimization_results._strategy:
    print("Liquidation Threshold:", optimization_results._strategy.liquidation_thresh)
    print("Time Window (minutes):", optimization_results._strategy.time_window_mins)
    print("Take Profit:", optimization_results._strategy.take_profit)
    print("Stop Loss:", optimization_results._strategy.stop_loss)
else:
    print("Optimization did not yield a best strategy.")

# Optional: Plot the results of the best strategy
try:
    bt.plot()
except Exception as e:
    print(f"\nCould not generate plot: {e}")