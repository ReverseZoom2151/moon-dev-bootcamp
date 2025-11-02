import os
import pandas as pd
from backtesting import Backtest, Strategy
from pykalman import KalmanFilter
import warnings
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_FOLDER_NAME = 'test_data'  # Folder containing CSV files relative to script location
OUTPUT_FILENAME = 'backtest_results.txt'
INITIAL_CASH = 100000
COMMISSION_RATE = 0.002
OPTIMIZATION_ITERATIONS = 50 # Number of iterations for skopt

# --- Strategy Definition ---
class KalmanBreakoutReversal(Strategy):
    """
    A strategy that uses a Kalman Filter on closing prices.
    It enters trades betting on a reversal when price breaks the filtered mean.
    - Enters SHORT when price breaks ABOVE the mean.
    - Enters LONG when price breaks BELOW the mean.
    - Exits when price crosses back over the mean.
    """
    # Parameters to be optimized
    window = 50         # Default window (used indirectly by Kalman filter? Check usage)
    take_profit = 0.05  # 5%
    stop_loss = 0.03    # 3%

    def init(self):
        """Initialize the Kalman Filter."""
        # Simple Kalman Filter setup for price smoothing
        self.kf = KalmanFilter(transition_matrices=[1],
                               observation_matrices=[1],
                               initial_state_mean=0,
                               initial_state_covariance=1,
                               observation_covariance=1,      # Measurement noise
                               transition_covariance=0.01)   # Process noise

        # Apply the Kalman filter to the closing prices
        try:
            self.filtered_state_means, _ = self.kf.filter(self.data.Close)
        except ValueError:
            # Handle cases with insufficient data for filtering
            self.filtered_state_means = np.full_like(self.data.Close, np.nan)

    def next(self):
        """Define the trading logic for each bar."""
        # Ensure enough data and valid filter output
        if len(self.data.Close) < 2 or np.isnan(self.filtered_state_means[-1]):
            return

        filtered_mean = self.filtered_state_means[-1]
        current_close = self.data.Close[-1]
        
        # --- Short Entry --- 
        # If price breaks significantly above the filtered mean, anticipate reversal (short)
        if not self.position.is_short and current_close > filtered_mean: 
             # Optional: Add threshold condition, e.g., current_close > filtered_mean * 1.001 
             if not self.position: # Check again to ensure no position was opened (e.g., long closed)
                 self.sell(sl=current_close * (1 + self.stop_loss), 
                           tp=current_close * (1 - self.take_profit))

        # --- Short Exit --- 
        # Close short if price reverts below the filtered mean
        elif self.position.is_short and current_close < filtered_mean:
            self.position.close(reason="Price reverted below KF mean")

        # --- Long Entry --- 
        # If price breaks significantly below the filtered mean, anticipate reversal (long)
        if not self.position.is_long and current_close < filtered_mean: 
             # Optional: Add threshold condition, e.g., current_close < filtered_mean * 0.999
             if not self.position: # Check again
                 self.buy(sl=current_close * (1 - self.stop_loss), 
                          tp=current_close * (1 + self.take_profit))

        # --- Long Exit --- 
        # Close long if price reverts above the filtered mean
        elif self.position.is_long and current_close > filtered_mean:
            self.position.close(reason="Price reverted above KF mean")

# --- Helper Functions ---
def load_prepare_data(file_path):
    """Loads CSV data, sets index, assigns standard columns."""
    try:
        data = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
        # Assuming fixed column order if no header: Open, High, Low, Close, Volume
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Basic validation
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Missing required OHLCV columns")
        if data.isnull().values.any():
             print(f"Warning: NaNs found in {os.path.basename(file_path)}, attempting forward fill.")
             data.ffill(inplace=True)
             data.bfill(inplace=True) # Backfill any remaining NaNs at the start
        data.dropna(inplace=True) # Drop any rows that still have NaNs
        return data
    except Exception as e:
        print(f"Error loading or preparing data from {file_path}: {e}")
        return None

def run_backtest(data, strategy_class, cash, commission, **strategy_params):
    """Runs a single backtest and returns statistics."""
    bt = Backtest(data, strategy_class, cash=cash, commission=commission)
    try:
        stats = bt.run(**strategy_params)
        return stats
    except Exception as e:
        print(f"Error running backtest: {e}")
        return None

def run_optimization(data, strategy_class, cash, commission, params_to_optimize, maximize, n_iter):
    """Runs optimization using skopt and returns results."""
    bt = Backtest(data, strategy_class, cash=cash, commission=commission)
    try:
        results = bt.optimize(**params_to_optimize,
                              maximize=maximize,
                              method='skopt',
                              n_iter=n_iter)
        return results
    except ImportError:
         print("Error: scikit-optimize not found. Please install it: pip install scikit-optimize")
         return None
    except Exception as e:
        print(f"Error during optimization: {e}")
        return None

# --- Main Execution Logic ---
def main():
    """Finds data files, runs backtests and optimizations, writes results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(script_dir, DATA_FOLDER_NAME)
    output_file_path = os.path.join(script_dir, OUTPUT_FILENAME)

    if not os.path.isdir(data_folder_path):
        print(f"Error: Data folder not found at {data_folder_path}")
        return

    data_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]
    if not data_files:
        print(f"Error: No CSV files found in {data_folder_path}")
        return

    print(f"Found {len(data_files)} CSV files in {data_folder_path}")
    print(f"Results will be saved to: {output_file_path}")

    # Open the output file
    with open(output_file_path, 'w') as file:
        
        # --- Initial Backtest with Defaults ---
        print("\n--- Running Initial Backtests (Default Parameters) ---")
        file.write("--- Initial Backtests (Default Parameters) ---\n\n")
        for data_file in data_files:
            print(f"Processing {data_file}...")
            file_path = os.path.join(data_folder_path, data_file)
            data = load_prepare_data(file_path)
            
            if data is not None and not data.empty:
                stats_default = run_backtest(data, KalmanBreakoutReversal, 
                                             cash=INITIAL_CASH, commission=COMMISSION_RATE)
                if stats_default:
                    result_str = f"Default Results for {data_file}:\n{stats_default}\n{'-' * 80}\n"
                    print(result_str)
                    file.write(result_str)
                else:
                    print(f"Skipping default backtest for {data_file} due to errors.")
            else:
                 print(f"Skipping {data_file} due to data loading/preparation error.")
        
        # --- Optimization ---
        print("\n--- Running Optimizations (skopt) ---")
        file.write("\n--- Optimizations (skopt) ---\n\n")
        for data_file in data_files:
            print(f"Optimizing for {data_file}...")
            file_path = os.path.join(data_folder_path, data_file)
            data = load_prepare_data(file_path)
            
            if data is not None and not data.empty:
                # Define parameter ranges for skopt
                params_to_optimize = {
                    'window': (20, 100),  # Example range for window (adjust as needed)
                    'take_profit': (0.01, 0.10), # TP range 1% to 10%
                    'stop_loss': (0.01, 0.10),   # SL range 1% to 10%
                }
                
                optimization_results = run_optimization(data, KalmanBreakoutReversal, 
                                                      cash=INITIAL_CASH, 
                                                      commission=COMMISSION_RATE,
                                                      params_to_optimize=params_to_optimize,
                                                      maximize='Equity Final [$]',
                                                      n_iter=OPTIMIZATION_ITERATIONS)
                
                if optimization_results:
                    result_str = f"Optimized Results for {data_file}:\n{optimization_results}\n"
                    # Safely access best strategy parameters
                    if optimization_results._strategy:
                         best_params = optimization_results._strategy
                         result_str += f"Best Parameters for {data_file}:\n"
                         result_str += f"  Window: {getattr(best_params, 'window', 'N/A')}\n"
                         result_str += f"  Take Profit: {getattr(best_params, 'take_profit', 'N/A')}\n"
                         result_str += f"  Stop Loss: {getattr(best_params, 'stop_loss', 'N/A')}\n"
                    else:
                        result_str += f"Best parameters could not be determined for {data_file}.\n"
                    result_str += f"{'=' * 80}\n"
                    print(result_str)
                    file.write(result_str)
                else:
                    print(f"Skipping optimization for {data_file} due to errors.")
            else:
                 print(f"Skipping {data_file} due to data loading/preparation error.")
                 
    print(f"\nProcessing complete. Results saved to {output_file_path}")

# --- Main execution guard ---
if __name__ == '__main__':
    # Add necessary imports if not already at top level
    import numpy as np 
    main()