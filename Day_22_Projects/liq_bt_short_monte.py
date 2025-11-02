import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import warnings
import os
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

class ShortOnSLiqStrategy(Strategy):
    """ 
    A strategy that enters a SHORT position based on recent Short Liquidation (S LIQ) volume 
    and exits based on recent Long Liquidation (L LIQ) volume, or Take Profit/Stop Loss.
    """
    # --- Strategy Parameters (Defaults - can be optimized later) ---
    short_liquidation_thresh = 100000  # S LIQ volume threshold to trigger SHORT entry
    entry_time_window_mins = 5       # Lookback window for S LIQ entry signal (minutes)
    long_liquidation_closure_thresh = 50000 # L LIQ volume threshold to trigger SHORT exit
    exit_time_window_mins = 5        # Lookback window for L LIQ exit signal (minutes)
    take_profit_pct = 0.02           # Take profit percentage for shorts (e.g., 0.02 = 2% down)
    stop_loss_pct = 0.01             # Stop loss percentage for shorts (e.g., 0.01 = 1% up)

    def init(self):
        """Initialize strategy indicators and state."""
        self.short_liquidations = self.data.short_liquidations
        self.long_liquidations = self.data.long_liquidations

    def next(self):
        """Define the logic executed on each bar."""
        # --- Entry Logic (Short Only) --- 
        if not self.position: # Only check for entry if not already in a position
            entry_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_short_liquidations = self.short_liquidations[entry_start_idx:].sum()

            # Enter SHORT if S LIQ volume exceeds threshold
            if recent_short_liquidations >= self.short_liquidation_thresh:
                sl_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
                tp_price = self.data.Close[-1] * (1 - self.take_profit_pct)
                self.sell(sl=sl_price, tp=tp_price)

        # --- Exit Logic (based on L LIQ threshold, in addition to TP/SL) ---
        elif self.position.is_short: # Only check for L LIQ exit if in a short position
            exit_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.exit_time_window_mins)
            exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
            recent_long_liquidations = self.long_liquidations[exit_start_idx:].sum()
            
            # Close position if recent L LIQ exceeds threshold
            if recent_long_liquidations >= self.long_liquidation_closure_thresh:
                self.position.close(reason="L Liq Threshold Hit")


def main():
    """Main function to load data, preprocess, and run a single backtest."""
    # --- Configuration ---
    symbol = "SOL"  # Define the symbol to process
    
    # Construct path relative to the script's parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from script_dir (to ATC Bootcamp Projects) then into Day_21_Projects/output_data
    data_dir_relative_to_parent = os.path.join('..', 'Day_21_Projects', 'output_data') 
    data_path = os.path.abspath(os.path.join(script_dir, data_dir_relative_to_parent, f'{symbol}_liq_data.csv'))

    # --- Data Loading ---
    print(f"Loading data for {symbol} from: {data_path}")
    try:
        data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_path}'")
        print(f"Please ensure 'cutitup.py' has run successfully creating {symbol}_liq_data.csv.")
        return

    # --- Data Preprocessing & Feature Engineering ---
    print("Preprocessing data...")
    required_cols = ['LIQ_SIDE', 'price', 'usd_size']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: Input CSV '{data_path}' missing required columns: {required_cols}.")
        return
    data = data[required_cols].copy()

    # Create L LIQ and S LIQ volume columns using vectorized operations
    data['long_liquidations'] = np.where(data['LIQ_SIDE'] == 'L LIQ', data['usd_size'], 0)
    data['short_liquidations'] = np.where(data['LIQ_SIDE'] == 'S LIQ', data['usd_size'], 0)

    # Resample data to 1-minute frequency
    agg_funcs = {
        'price': 'mean',
        'long_liquidations': 'sum',
        'short_liquidations': 'sum'
    }
    data_resampled = data.resample('min').agg(agg_funcs)

    # --- OHLC Creation ---
    # Synthesize OHLC columns from mean liquidation price in the interval.
    data_resampled['Open'] = data_resampled['price']
    data_resampled['High'] = data_resampled['price']
    data_resampled['Low'] = data_resampled['price']
    data_resampled['Close'] = data_resampled['price']

    # Handle missing data after resampling
    price_columns = ['price', 'Open', 'High', 'Low', 'Close']
    data_resampled[price_columns] = data_resampled[price_columns].ffill()
    volume_columns = ['long_liquidations', 'short_liquidations']
    data_resampled[volume_columns] = data_resampled[volume_columns].fillna(0)

    # Ensure the DataFrame is sorted by index and has no NaNs in Close
    data_resampled.sort_index(inplace=True)
    data_resampled.dropna(subset=['Close'], inplace=True)
    
    print("Data preprocessing complete.")
    if data_resampled.empty:
        print("Error: No data available for backtesting after processing.")
        return
    print(f"Data shape for backtest: {data_resampled.shape}")

    # --- Run Single Backtest ---
    print(f"\n--- Running Backtest for {symbol} ---")
    # Uses default parameters defined in the Strategy class
    bt = Backtest(data_resampled, ShortOnSLiqStrategy, cash=100000, commission=0.002)
    
    try:
        stats = bt.run()
        print("\n--- Backtest Results ---")
        print(stats)
        
        # Plot results
        print("\n--- Plotting Performance ---")
        bt.plot()
        
    except Exception as e:
        print(f"Error during backtest run: {e}")

# --- Main execution guard ---
if __name__ == '__main__':
    main()
