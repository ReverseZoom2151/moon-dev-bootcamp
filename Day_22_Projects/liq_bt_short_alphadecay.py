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
    """ 
    A strategy that enters a SHORT position based on recent Short Liquidation (S LIQ) volume 
    and exits based on recent Long Liquidation (L LIQ) volume, or Take Profit/Stop Loss.
    
    Note: This base strategy does not implement LONG entry logic.
    """
    # --- Strategy Parameters (Defaults) ---
    # These are NOT optimized in this script, just used for the decay test.
    short_liquidation_thresh = 100000  # S LIQ volume threshold to trigger SHORT entry
    entry_time_window_mins = 5       # Lookback window for S LIQ entry signal (minutes)
    long_liquidation_closure_thresh = 50000 # L LIQ volume threshold to trigger SHORT exit
    exit_time_window_mins = 5        # Lookback window for L LIQ exit signal (minutes)
    take_profit_pct = 0.02           # Take profit percentage for shorts (e.g., 0.02 = 2% down)
    stop_loss_pct = 0.01             # Stop loss percentage for shorts (e.g., 0.01 = 1% up)

    def init(self):
        """Initialize strategy indicators and state."""
        # Pre-calculate or access the required data columns
        self.short_liquidations = self.data.short_liquidations
        self.long_liquidations = self.data.long_liquidations
        self.trade_entry_bar = -1 # Track the bar index of entry

    def next(self):
        """Define the logic executed on each bar."""
        current_bar_idx = len(self.data.Close) - 1

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
                self.trade_entry_bar = current_bar_idx # Record entry bar index

        # --- Exit Logic (based on L LIQ threshold, in addition to TP/SL) ---
        elif self.position.is_short: # Only check for L LIQ exit if in a short position
            # Use a fixed window for exit signal check, independent of entry bar
            exit_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.exit_time_window_mins)
            exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
            recent_long_liquidations = self.long_liquidations[exit_start_idx:].sum()
            
            # Close position if recent L LIQ exceeds threshold
            if recent_long_liquidations >= self.long_liquidation_closure_thresh:
                self.position.close(reason="L Liq Threshold Hit")

class DelayedLiquidationStrategy(LiquidationStrategy):
    """ 
    Extends LiquidationStrategy to introduce a delay between the signal 
    and the actual trade entry.
    """
    # This class variable is modified by the test loop. 
    # Be cautious if adapting this for parallel execution, as modifying class 
    # variables directly is not thread-safe or process-safe. 
    # For this sequential test, it's a simpler approach.
    delay_minutes = 0  

    def next(self):
        """Applies delay logic before executing the base strategy's next()."""
        current_bar_idx = len(self.data.Close) - 1
        
        # If we are already in a position OR have just entered on this bar, 
        # let the base class handle exits/management. Delay only affects entry.
        if self.position or self.trade_entry_bar == current_bar_idx:
             super().next()
             return
             
        # --- Delay Logic for Entry ---
        # Check if a potential SHORT entry signal occurred within the delay window.
        # This simulates looking back to see if a signal *should* have triggered earlier.
        # Note: This is a simplified check for the purpose of this alpha decay test.
        # A more robust implementation might store signal states explicitly.
        potential_entry_signal = False
        last_potential_entry_bar = -1
        
        for lookback in range(1, self.delay_minutes + 1):
            if current_bar_idx - lookback < 0: break # Bounds check
            
            prev_time = self.data.index[current_bar_idx - lookback]
            entry_start_time = prev_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            
            # Check the short entry condition on the past bar
            past_short_liquidations = self.short_liquidations[entry_start_idx : current_bar_idx - lookback + 1].sum()
            
            if past_short_liquidations >= self.short_liquidation_thresh:
                potential_entry_signal = True
                last_potential_entry_bar = current_bar_idx - lookback
                break # Found the most recent signal within the delay window

        # If a signal occurred within the delay window, but we haven't entered yet
        # (self.trade_entry_bar < last_potential_entry_bar implies the last entry 
        # was before this potential signal, or no entry yet)
        if potential_entry_signal and self.trade_entry_bar < last_potential_entry_bar:
            # Calculate bars passed since the signal should have occurred
            bars_since_signal = current_bar_idx - last_potential_entry_bar
            
            # Check if the delay period has passed
            if bars_since_signal >= self.delay_minutes:
                 # Delay is over, execute the SHORT entry logic NOW using current price
                 if not self.position: # Double check we aren't in a position
                    sl_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
                    tp_price = self.data.Close[-1] * (1 - self.take_profit_pct)
                    self.sell(sl=sl_price, tp=tp_price)
                    self.trade_entry_bar = current_bar_idx # Mark entry bar as NOW
                    # We entered due to delay, so return here and don't run super().next() for this bar
                    return 
        
        # If no potential signal was found in the lookback, or the delay hasn't passed, 
        # or we already processed an entry for the signal, 
        # run the base logic normally to check for a *new* signal on the *current* bar.
        super().next()


def run_alpha_decay_test(data, strategy_class, delays):
    """ 
    Runs backtests for a given strategy class with varying entry delays.

    Args:
        data (pd.DataFrame): The prepared OHLCV data with liquidation features.
        strategy_class (Type[Strategy]): The strategy class (must have a `delay_minutes` attribute).
        delays (list[int]): A list of delay values (in minutes) to test.
    """
    for delay in delays:
        print(f"\n--- Running backtest with {delay}-minute delay ---")
        
        # --- IMPORTANT --- 
        # Modifies the class variable directly. This is simple for sequential tests 
        # but not suitable for parallel execution. An alternative would be 
        # creating strategy instances with parameters, if the library supports it easily,
        # or dynamic subclass creation.
        strategy_class.delay_minutes = delay

        # Create and configure the backtest
        bt = Backtest(data, strategy_class, cash=100000, commission=0.002)

        # Run the backtest and print the results
        try:
            stats = bt.run()
            print(stats)
            # Optional: Plotting for each delay can be noisy
            # bt.plot(filename=f'alphadecay_delay_{delay}min.html', open_browser=False)
        except Exception as e:
            print(f"Error during backtest run with delay {delay}: {e}")
        print("\n" + "="*50)


def main():
    """Main function to load data, preprocess, and run the alpha decay test."""
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
        print(f"Please ensure 'cutitup.py' has run successfully creating {symbol}_liq_data.csv in the correct output directory.")
        return # Exit main function gracefully

    # --- Data Preprocessing & Feature Engineering ---
    print("Preprocessing data...")
    required_cols = ['LIQ_SIDE', 'price', 'usd_size']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: Input CSV '{data_path}' missing required columns: {required_cols}.")
        return
    data = data[required_cols].copy() # Use copy to avoid SettingWithCopyWarning

    # Create L LIQ and S LIQ volume columns using vectorized operations
    data['long_liquidations'] = np.where(data['LIQ_SIDE'] == 'L LIQ', data['usd_size'], 0)
    data['short_liquidations'] = np.where(data['LIQ_SIDE'] == 'S LIQ', data['usd_size'], 0)

    # Resample data to 1-minute frequency for backtesting
    agg_funcs = {
        'price': 'mean',
        'long_liquidations': 'sum',
        'short_liquidations': 'sum'
    }
    data_resampled = data.resample('min').agg(agg_funcs)

    # --- OHLC Creation --- 
    # Synthesize OHLC columns from mean liquidation price in the interval.
    # This is a simplification necessary because the input data lacks true OHLC.
    # Backtesting results should be interpreted with this limitation in mind.
    data_resampled['Open'] = data_resampled['price']
    data_resampled['High'] = data_resampled['price']
    data_resampled['Low'] = data_resampled['price']
    data_resampled['Close'] = data_resampled['price']

    # Handle missing data after resampling
    price_columns = ['price', 'Open', 'High', 'Low', 'Close']
    data_resampled[price_columns] = data_resampled[price_columns].ffill()
    volume_columns = ['long_liquidations', 'short_liquidations']
    data_resampled[volume_columns] = data_resampled[volume_columns].fillna(0)

    # Ensure the DataFrame is sorted by index
    data_resampled.sort_index(inplace=True)

    # Remove rows with NaN in critical columns (like Close) before backtesting
    data_resampled.dropna(subset=['Close'], inplace=True)
    
    # Drop original price column if needed (optional)
    # data_resampled.drop(columns=['price'], inplace=True)
    
    print("Data preprocessing complete.")
    if data_resampled.empty:
        print("Error: No data available for backtesting after processing.")
        return
    print(f"Data shape for backtest: {data_resampled.shape}")


    # --- Run Alpha Decay Test ---
    # Define delay intervals in minutes
    delays = [0, 1, 2, 5, 10, 15, 30, 60]

    # Run the alpha decay test using the prepared data
    run_alpha_decay_test(data_resampled, DelayedLiquidationStrategy, delays)


if __name__ == '__main__':
    main()

