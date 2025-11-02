import pandas as pd
import talib as ta
from backtesting import Backtest
from backtesting.lib import crossover, TrailingStrategy
import itertools # For generating parameter combinations
# Import multiprocessing for freeze_support if needed (usually not strictly required unless building executables)
# import multiprocessing

# No longer need the custom function, TA-Lib has STOCHRSI
# def stoch_rsi_custom(...): ...

# --- Strategy Definition ---
class EnhancedEmaStrategy(TrailingStrategy):
    # == Core Strategy Parameters ==
    ema_fast_period = 9
    ema_slow_period = 18
    ema_trend_period = 200  # Long-term trend filter
    stochrsi_rsi_len = 14
    atr_period = 14
    volume_ma_period = 20  # Volume moving average period
    
    # Risk Management
    risk_per_trade = 0.02  # 2% of equity per trade
    atr_sl_multiplier = 2.0
    partial_profit_factor = 1.5  # Take partial profit at 1.5x ATR
    
    def init(self):
        # Price indicators
        self.ema_fast = self.I(ta.EMA, self.data.Close, self.ema_fast_period)
        self.ema_slow = self.I(ta.EMA, self.data.Close, self.ema_slow_period)
        self.ema_trend = self.I(ta.EMA, self.data.Close, self.ema_trend_period)
        
        # Momentum indicator
        self.stoch_k, self.stoch_d = self.I(
            ta.STOCHRSI, self.data.Close, self.stochrsi_rsi_len, 3, 3, 0)
        
        # Volatility indicator
        self.atr = self.I(ta.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        
        # Volume analysis
        self.volume_ma = self.I(ta.SMA, self.data.Volume, self.volume_ma_period)
        self.obv = self.I(ta.OBV, self.data.Close, self.data.Volume)
        
        # Initialize trailing stop
        self.set_trailing_sl(self.atr_sl_multiplier)

        # Add position duration tracking at the class level
        if not hasattr(self, 'position_entry_bar'):
            self.position_entry_bar = None
        if not hasattr(self, 'position_entry_price'):
            self.position_entry_price = None

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        
        # Skip if we don't have enough data
        if len(self.data) < max(self.ema_trend_period, self.volume_ma_period) + 10:
            return

        # Calculate dynamic position size
        equity = self.equity
        position_size = (equity * self.risk_per_trade) / (atr_val * self.atr_sl_multiplier)
        
        # Add validation and constraints
        position_size = min(position_size, 0.5)  # Never risk more than 50% of equity
        position_size = max(position_size, 0.01) # Minimum 1% position size
        position_size = round(position_size, 2)

        # Add ATR validation
        if atr_val <= 0.0001:  # Prevent division by near-zero values
            return

        # Trend filter conditions
        bullish_trend = price > self.ema_trend[-1]
        bearish_trend = price < self.ema_trend[-1]
        
        # Volume conditions
        volume_ok = self.data.Volume[-1] > self.volume_ma[-1]
        obv_trend = self.obv[-1] > self.obv[-2]  # OBV rising

        # Entry/Exit signals
        ema_cross_up = crossover(self.ema_fast, self.ema_slow)
        ema_cross_down = crossover(self.ema_slow, self.ema_fast)
        stoch_bullish = crossover(self.stoch_k, self.stoch_d) and self.stoch_d[-1] < 40
        stoch_bearish = crossover(self.stoch_d, self.stoch_k) and self.stoch_d[-1] > 60

        # === Entry Logic with Filters ===
        if not self.position:
            # Long entry with all filters
            if all([bullish_trend, ema_cross_up, stoch_bullish, volume_ok, obv_trend]):
                sl = price - atr_val * self.atr_sl_multiplier
                tp = price + atr_val * self.partial_profit_factor
                self.buy(size=position_size, sl=sl, tp=tp)
                self.position_entry_price = price
                
            # Short entry with all filters
            elif all([bearish_trend, ema_cross_down, stoch_bearish, volume_ok, not obv_trend]):
                sl = price + atr_val * self.atr_sl_multiplier
                tp = price - atr_val * self.partial_profit_factor
                self.sell(size=position_size, sl=sl, tp=tp)
                self.position_entry_price = price

        # === Advanced Exit Management ===
        elif self.position:
            # Track position duration
            current_duration = len(self.data) - self.position_entry_bar if self.position_entry_bar else 0
            
            # Partial profit taking
            if self.position.pl_pct >= 1.5:  # Take 50% off at 1.5x ATR
                # Close 50% of the position as partial profit taking.
                self.position.close(0.5)
                # Note: Updating SL to breakeven is not supported in this version since `position.orders`
                # is not available. The remaining position's stop-loss remains unchanged.

            # Time-based exit (close after 5 bars if not working)
            if current_duration > 5 and abs(self.position.pl_pct) < 0.5:
                self.position.close()
                self.position_entry_bar = None

        # Update position entry bar when opening new position
        if not self.position and (ema_cross_up or ema_cross_down):
            self.position_entry_bar = len(self.data)

# --- Data Loading and Preprocessing Function ---
def load_data(filename, factor=1000.0):
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename)
    print("Data loaded successfully.")

    # Adjust column names if needed
    if 'datetime' in data.columns:
        data.rename(columns={'datetime': 'Datetime', 'open': 'Open', 'high': 'High',
                           'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

    # Verify required columns exist
    required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"CSV {filename} missing required columns. Found: {data.columns}. Required: {required_cols}")

    # Scale price data
    data['Open'] = data['Open'] / factor
    data['High'] = data['High'] / factor
    data['Low'] = data['Low'] / factor
    data['Close'] = data['Close'] / factor

    # Convert Datetime and set index
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    data.dropna(inplace=True)
    data = data.sort_index()

    print(f"Data range: {data.index.min()} to {data.index.max()}")
    print(f"Number of rows: {len(data)}")
    return data

# --- Main Execution Block ---
if __name__ == '__main__':
    # Optional: If creating frozen executables, uncomment the next line
    # multiprocessing.freeze_support()

    # --- Configuration ---
    timeframes = {
        '1h': 'storage_SOL-USD1h100.csv',
        '4h': 'storage_SOL-USD4h100.csv', # Assumed filename
        '1d': 'storage_SOL-USD1d100.csv'  # Assumed filename
        # Add more timeframes and filenames here
    }
    initial_cash = 10000
    commission = 0.002
    data_scale_factor = 1000.0

    # --- Updated Optimization Parameters ---
    opt_params = {
        'ema_fast_period': [7, 9, 12],
        'ema_slow_period': [15, 18, 21],
        'ema_trend_period': [150, 200, 250],
        'stochrsi_rsi_len': [10, 14, 18],
        'atr_sl_multiplier': [1.5, 2.0, 2.5],
        'risk_per_trade': [0.01, 0.02, 0.03],
        'partial_profit_factor': [1.0, 1.5, 2.0],
        'volume_ma_period': [15, 20, 25]
    }

    overall_best_return = -float('inf')
    overall_best_params = None
    overall_best_timeframe = None
    all_results = {}

    # --- Loop Through Timeframes ---
    for tf_str, filename in timeframes.items():
        print(f"\n\n===== Processing Timeframe: {tf_str} =====")
        try:
            data = load_data(filename, factor=data_scale_factor)
            backtest = Backtest(data, EnhancedEmaStrategy, cash=initial_cash, commission=commission)

            # --- Run Basic Backtest with Defaults ---
            print(f"\n--- Running basic backtest ({tf_str}) ---")
            stats_default = backtest.run() # Uses default params from class definition
            print(stats_default)
            all_results[tf_str] = {'default': stats_default, 'optimized': None, 'best_params': None}

            # --- Manual Grid Search Optimization ---
            print(f"\n--- Running manual grid search ({tf_str}) ---")

            # Generate all combinations
            keys, values = zip(*opt_params.items())
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            timeframe_best_return = -float('inf')
            timeframe_best_params = None
            timeframe_best_stats = None

            # Correctly calculate total combinations to test *after* applying constraints
            valid_combinations = 0
            combinations_to_test = []
            for params in param_combinations:
                 if params['ema_fast_period'] < params['ema_slow_period']:
                     combinations_to_test.append(params)
                     valid_combinations += 1

            total_combinations_to_run = valid_combinations # Use count of valid combos
            current = 0

            for params in combinations_to_test: # Iterate only through valid combos
                current += 1
                print(f"Testing {tf_str} combination {current}/{total_combinations_to_run}: {params}")

                try:
                    result = backtest.run(**params) # Pass current param combination
                    returns = result['Return [%]']

                    if returns > timeframe_best_return:
                        timeframe_best_return = returns
                        timeframe_best_params = params
                        timeframe_best_stats = result

                        # Check if this is the best overall across timeframes
                        if returns > overall_best_return:
                            overall_best_return = returns
                            overall_best_params = params
                            overall_best_timeframe = tf_str

                except Exception as e:
                    print(f"Error with parameters on {tf_str}: {params}\nError details: {e}")

            # Store and print results for this timeframe
            print(f"\n--- Optimization Results for {tf_str} ---")
            if timeframe_best_params:
                print(f"Best Return: {timeframe_best_return:.2f}%")
                print(f"Best Params: {timeframe_best_params}")
                all_results[tf_str]['optimized'] = timeframe_best_stats
                all_results[tf_str]['best_params'] = timeframe_best_params

                # Optionally plot the best result for this timeframe
                # print(f"\n--- Plotting best result for {tf_str} ---")
                # backtest.run(**timeframe_best_params) # Re-run to set internal state for plot
                # backtest.plot(plot_volume=False, plot_equity=True, title=f"Best Result {tf_str}")
            else:
                print("No profitable combination found during optimization.")

        except FileNotFoundError:
            print(f"Error: Data file '{filename}' not found for timeframe {tf_str}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred processing timeframe {tf_str}: {e}")
            import traceback
            traceback.print_exc()

    # --- Final Summary ---
    print("\n\n===== Overall Results Summary =====")
    if overall_best_timeframe:
        print(f"Best Overall Return: {overall_best_return:.2f}%")
        print(f"Best Timeframe: {overall_best_timeframe}")
        print(f"Best Parameters: {overall_best_params}")
        print("\n--- Stats for Overall Best Run ---")
        print(all_results[overall_best_timeframe]['optimized'])

        # Plot the overall best result - removing the title argument
        print("\n--- Plotting Overall Best Result ---")
        try:
            best_tf_data = load_data(timeframes[overall_best_timeframe], factor=data_scale_factor)
            best_tf_backtest = Backtest(best_tf_data, EnhancedEmaStrategy, cash=initial_cash, commission=commission)
            best_tf_backtest.run(**overall_best_params)
            best_tf_backtest.plot(plot_volume=False, plot_equity=True)  # Removed 'title' parameter
        except Exception as e:
            print(f"Could not plot overall best result: {e}")

    else:
        print("No profitable strategy found across any timeframe during optimization.")
