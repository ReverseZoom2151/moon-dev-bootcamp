import numpy as np
from backtesting import Backtest, Strategy
import pandas as pd
import warnings
import os
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

class LongOnSLiqStrategy(Strategy): # Renamed strategy
    # --- Strategy Parameters (to be optimized) ---
    s_liq_entry_thresh = 100000    # S LIQ volume threshold for entry
    entry_time_window_mins = 20    # Lookback window for S LIQ entry signal (minutes)
    take_profit = 0.02             # Take profit percentage
    stop_loss = 0.01               # Stop loss percentage

    def init(self):
        # Use the precalculated S LIQ volume column
        self.s_liq_volume = self.data.s_liq_volume

    def next(self):
        current_time = self.data.index[-1]

        # --- Entry Logic --- 
        if not self.position: # Only check for entry if not already in a position
            entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            # Ensure index is sorted for searchsorted
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_s_liquidations = self.s_liq_volume[entry_start_idx:].sum()

            # Enter long if S LIQ volume exceeds threshold
            if recent_s_liquidations >= self.s_liq_entry_thresh:
                sl_price = self.data.Close[-1] * (1 - self.stop_loss)
                tp_price = self.data.Close[-1] * (1 + self.take_profit)
                self.buy(sl=sl_price, tp=tp_price)

        # Note: No additional exit logic based on L LIQ in this strategy, only TP/SL.

# --- Main Execution Block ---
def run_backtest():
    # --- Configuration ---
    symbol = "WIF"  # Define the symbol to process
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
    required_cols = ['LIQ_SIDE', 'price', 'usd_size']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: Input CSV {data_path} missing required columns {required_cols}.")
        exit()
    data = data[required_cols]

    # --- Feature Engineering for Strategy --- 

    # Filter for Short Liquidations (S LIQ) as these trigger our LONG entry strategy
    s_liq_data = data[data['LIQ_SIDE'] == 'S LIQ'].copy()

    # Resample data to 1-minute frequency for backtesting
    # Using 'min' instead of deprecated 'T'
    ohlc_agg = {
        'price': 'mean',   # Mean price of S LIQs in the minute
        'usd_size': 'sum' # Sum the volume of S LIQs in the minute
    }
    data_resampled = s_liq_data.resample('min').agg(ohlc_agg)

    # Create OHLC columns required by backtesting.py using mean price
    data_resampled['Open'] = data_resampled['price']
    data_resampled['High'] = data_resampled['price']
    data_resampled['Low'] = data_resampled['price']
    data_resampled['Close'] = data_resampled['price']

    # Rename aggregated usd_size to represent S LIQ volume
    data_resampled.rename(columns={'usd_size': 's_liq_volume'}, inplace=True)

    # Handle missing data after resampling:
    price_columns = ['price', 'Open', 'High', 'Low', 'Close']
    data_resampled[price_columns] = data_resampled[price_columns].ffill()
    data_resampled['s_liq_volume'].fillna(0, inplace=True)

    # Ensure the DataFrame is sorted by index
    data_resampled.sort_index(inplace=True)

    # Remove rows with NaN in critical columns (e.g., Close) before backtesting
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
    print(f"\n--- Running Backtest & Manual Parameter Search for {symbol} ---")
    print("--- Testing Parameters Sequentially to Avoid Windows Handle Limits ---")
    
    # Define parameter ranges for manual search
    s_liq_thresholds = range(10000, 200000, 20000)    # 10k to 190k by 20k steps
    time_windows = range(10, 61, 10)                  # 10 to 60 by 10 steps
    take_profits = [0.01, 0.02, 0.03, 0.04, 0.05]     # 1% to 5% by 1% steps
    stop_losses = [0.01, 0.02, 0.03, 0.04, 0.05]      # 1% to 5% by 1% steps
    
    # Initialize tracking variables
    best_equity = 0
    best_params = None
    best_stats = None
    total_combinations = len(s_liq_thresholds) * len(time_windows) * len(take_profits) * len(stop_losses)
    current_combo = 0
    
    # Manual grid search
    for threshold in s_liq_thresholds:
        for window in time_windows:
            for tp in take_profits:
                for sl in stop_losses:
                    # Skip invalid combinations (Optional constraint: ensure TP > SL)
                    if tp <= sl:
                        continue
                        
                    current_combo += 1
                    print(f"\nTesting combination {current_combo}/{total_combinations}: "
                          f"threshold={threshold}, window={window}, tp={tp:.2f}, sl={sl:.2f}")
                    
                    # Create strategy instance with current parameters
                    class CurrentStrategy(LongOnSLiqStrategy):
                        s_liq_entry_thresh = threshold
                        entry_time_window_mins = window
                        take_profit = tp
                        stop_loss = sl
                    
                    # Run backtest with current parameters
                    bt = Backtest(data_resampled, CurrentStrategy, cash=100000, commission=0.002)
                    stats = bt.run()
                    
                    # Track the best result
                    final_equity = stats['Equity Final [$]']
                    print(f"Final Equity: ${final_equity:.2f}")
                    
                    if final_equity > best_equity:
                        best_equity = final_equity
                        best_params = {
                            's_liq_entry_thresh': threshold,
                            'entry_time_window_mins': window, 
                            'take_profit': tp,
                            'stop_loss': sl
                        }
                        best_stats = stats
                        print("✓ New best parameters found!")
    
    # --- Results ---
    print("\n--- Best Parameters Found ---")
    if best_params:
        print(f"S Liq Entry Threshold: {best_params.get('s_liq_entry_thresh', 'N/A')}")
        print(f"Entry Time Window (mins): {best_params.get('entry_time_window_mins', 'N/A')}")
        print(f"Take Profit: {best_params.get('take_profit', 'N/A')}")
        print(f"Stop Loss: {best_params.get('stop_loss', 'N/A')}")
        print(f"Final Equity: ${best_equity:.2f}")
        
        # Print more details about the best strategy
        print("\n--- Best Strategy Performance ---")
        print(f"Return: {best_stats['Return [%]']:.2f}%")
        print(f"Max Drawdown: {best_stats['Max. Drawdown [%]']:.2f}%")
        print(f"Sharpe Ratio: {best_stats['Sharpe Ratio']:.2f}")
        print(f"Win Rate: {best_stats['Win Rate [%]']:.2f}%")
        print(f"# Trades: {best_stats['# Trades']}")
        
        # Optional: Plot the best strategy
        print("\n--- Plotting Best Strategy Performance ---")
        try:
            # Re-run the backtest with best parameters for plotting
            class BestStrategy(LongOnSLiqStrategy):
                s_liq_entry_thresh = best_params['s_liq_entry_thresh']
                entry_time_window_mins = best_params['entry_time_window_mins']
                take_profit = best_params['take_profit']
                stop_loss = best_params['stop_loss']
                
            final_bt = Backtest(data_resampled, BestStrategy, cash=100000, commission=0.002)
            final_bt.run()
            final_bt.plot()
        except Exception as e:
            print(f"Could not generate plot: {e}")
    else:
        print("No valid parameters found.")

if __name__ == '__main__':
    run_backtest()