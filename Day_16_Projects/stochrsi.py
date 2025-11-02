import pandas as pd
import warnings 
import ccxt
import sys, os
# import numpy as np # No longer explicitly needed here
from backtesting import Backtest, Strategy
from ta.momentum import StochRSIIndicator
from ta.volatility import BollingerBands
from backtesting.lib import crossover 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import key as xP_KEY, secret as xP_SECRET

# filter all warnings
warnings.filterwarnings("ignore")

# --- Constants ---
PRICE_SCALING_FACTOR = 1000
STOP_LOSS_FACTOR = 0.85
TAKE_PROFIT_FACTOR = 1.40
INITIAL_CASH = 1_000_000
POSITION_SIZE = 0.05

# --- Indicator Helper Functions ---
def calculate_bbands(close_series, window, window_dev):
    """Calculates Bollinger Bands using the 'ta' library."""
    indicator_bb = BollingerBands(close=pd.Series(close_series), window=window, window_dev=window_dev)
    # Return bands as a tuple of numpy arrays, which self.I expects for multi-output indicators
    return (
        indicator_bb.bollinger_lband().to_numpy(),
        indicator_bb.bollinger_mavg().to_numpy(),
        indicator_bb.bollinger_hband().to_numpy()
    )

def calculate_stochrsi(close_series, window, smooth1, smooth2):
    """Calculates StochRSI using the 'ta' library."""
    indicator_stochrsi = StochRSIIndicator(close=pd.Series(close_series), window=window, smooth1=smooth1, smooth2=smooth2)
    # Return K and D lines as a tuple of numpy arrays
    return (
        indicator_stochrsi.stochrsi_k().to_numpy(),
        indicator_stochrsi.stochrsi_d().to_numpy()
    )


# --- Strategy Definition ---
class Strat(Strategy):
    rsi_window = 14
    stochrsi_smooth1 = 3
    stochrsi_smooth2 = 3 
    bbands_length = 20
    # stochrsi_length = 14 # This parameter seems unused by ta.StochRSIIndicator which uses rsi_window (window)
    bbands_std = 2

    def init(self):
        # Ensure data passed to ta library is a pandas Series
        close_data = self.data.Close # backtesting.py provides numpy arrays here

        # Calculate Bollinger Bands using the helper function via self.I
        self.lower_band, self.mid_band, self.upper_band = self.I(
            calculate_bbands,
            close_data, # Pass the raw close data
            window=self.bbands_length,
            window_dev=self.bbands_std
        )
        
        # Calculate StochRSI using the helper function via self.I
        self.stoch_rsi_k, self.stoch_rsi_d = self.I(
            calculate_stochrsi,
            close_data, # Pass the raw close data
            window=self.rsi_window, # Use rsi_window as the 'window' parameter for ta.StochRSIIndicator
            smooth1=self.stochrsi_smooth1,
            smooth2=self.stochrsi_smooth2
        )

    def next(self):
        lower = self.lower_band

        # Check if enough data is available (indicator values might be NaN initially)
        if len(self.data) < max(self.bbands_length, self.rsi_window): # Adjust wait period
             return

        if (
            self.data.Close[-1] > lower[-1] 
            and crossover(self.stoch_rsi_k, self.stoch_rsi_d)
        ):
            # Check for valid price data before placing order
            if not pd.isna(self.data.Close[-1]) and self.data.Close[-1] > 0:
                 self.buy(size=POSITION_SIZE, sl=self.data.Close[-1] * STOP_LOSS_FACTOR, tp=self.data.Close[-1] * TAKE_PROFIT_FACTOR)


# --- Data Fetching (remains the same) ---
def fetch_data(symbol, timeframe, limit=2000):
    print(f"Attempting to fetch data for {symbol} ({timeframe})...")
    actual_symbol_used = symbol # Store original symbol for later use
    try:
        exchange = ccxt.phemex({
            'apiKey': xP_KEY,
            'secret': xP_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

        try:
            markets = exchange.load_markets()
            # print("Available markets on Phemex:") # Optional: reduce verbosity
            eth_markets = [s for s in markets.keys() if 'ETH' in s and markets[s].get('active')]
            btc_markets = [s for s in markets.keys() if 'BTC' in s and markets[s].get('active')]
            # print(f"Active ETH markets: {eth_markets}") # Optional: reduce verbosity
            # print(f"Active BTC markets: {btc_markets}") # Optional: reduce verbosity

            if symbol not in markets or not markets[symbol].get('active'):
                print(f"Symbol {symbol} not found or inactive. Looking for alternatives...")
                found_alt = False
                if 'ETH' in symbol:
                    # Prioritize USDT, then USD pairs
                    for alt_suffix in ['USDT', 'USD']:
                         potential_alt = f'ETH/{alt_suffix}'
                         if potential_alt in markets and markets[potential_alt].get('active'):
                              print(f"Found active alternative: {potential_alt}")
                              actual_symbol_used = potential_alt
                              found_alt = True
                              break
                elif 'BTC' in symbol:
                     for alt_suffix in ['USDT', 'USD']:
                         potential_alt = f'BTC/{alt_suffix}'
                         if potential_alt in markets and markets[potential_alt].get('active'):
                              print(f"Found active alternative: {potential_alt}")
                              actual_symbol_used = potential_alt
                              found_alt = True
                              break
                if not found_alt:
                     print(f"No suitable active alternative found for {symbol}.")
                     return pd.DataFrame(), symbol # Return original symbol if no alt found

        except Exception as market_load_error:
            print(f"Could not load markets: {market_load_error}")
            # Continue trying the original symbol if market loading fails
            actual_symbol_used = symbol

        timeframe_duration_in_ms = exchange.parse_timeframe(timeframe) * 1000
        since = exchange.milliseconds() - (limit * timeframe_duration_in_ms)
        
        print(f"Attempting to fetch OHLCV for symbol: {actual_symbol_used}")
        ohlcv = exchange.fetch_ohlcv(actual_symbol_used, timeframe, since=since, limit=limit)

        if not ohlcv:
            print(f"Warning: No OHLCV data returned for {actual_symbol_used} {timeframe}.")
            return pd.DataFrame(), actual_symbol_used
        print(f"Successfully fetched {len(ohlcv)} candles for {actual_symbol_used}.")

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Ensure numeric types after fetching
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True) # Drop rows with NaN values resulting from coercion
        return df, actual_symbol_used # Return the symbol that was actually used

    except ccxt.NetworkError as e:
        print(f"CCXT NetworkError fetching data: {e}")
    except ccxt.AuthenticationError as e:
        print(f"CCXT AuthenticationError: {e}. Please check your API Key and Secret.")
    except ccxt.ExchangeError as e:
        print(f"CCXT ExchangeError fetching data: {e}")
        if hasattr(e, 'args') and len(e.args) > 0:
            print(f"Error Details: {e.args[0]}")
    except Exception as e:
        print(f"An unexpected error occurred during data fetch: {e}")
        import traceback
        traceback.print_exc()
    return pd.DataFrame(), symbol # Return original symbol on error


# --- Main Execution ---
def run_backtest():
    successful_symbol = None
    # Try multiple symbols in order
    for symbol_to_try in ['ETHUSD', 'ETH/USD', 'ETH/USDT', 'ETHUSDT']: # Use a different variable name
        print(f"\nTrying symbol: {symbol_to_try}")
        data_df, actual_symbol_fetched = fetch_data(symbol_to_try, '1h') # Get the actual symbol used
        if not data_df.empty:
            print(f"Success fetching data with actual symbol: {actual_symbol_fetched}")
            successful_symbol = actual_symbol_fetched # Store the successful symbol
            break # Exit loop once data is fetched

    if successful_symbol is None or data_df.empty:
         print("\nCould not fetch data with any of the attempted symbols. Please check:")
         print("1. API key and secret are correct in Day_4_Projects/key_file.py")
         print("2. API key has permissions to access market data")
         print("3. Phemex services are available")
         print("4. Check active symbols on Phemex (e.g., ETH/USDT, BTC/USDT)")
         return

    # --- Data Scaling ---
    # Important: Check if scaling is appropriate for the fetched symbol (e.g., ETH/USDT vs ETHUSD)
    # Phemex ETH/USDT prices are usually direct (e.g., 3500.50), not scaled like ETHUSD (e.g., 35005000)
    # Let's assume USDT pairs DON'T need scaling, but USD pairs might.
    if successful_symbol.endswith('USD') and not successful_symbol.endswith('USDT'):
         print(f"Applying price scaling factor {PRICE_SCALING_FACTOR} for {successful_symbol}")
         data_df[['Open', 'High', 'Low', 'Close']] /= PRICE_SCALING_FACTOR
         # Volume scaling might also differ, adjust if needed
         data_df['Volume'] *= PRICE_SCALING_FACTOR
    else:
         print(f"Skipping price scaling for {successful_symbol}")


    print("\nSample Data (Last 5 rows):")
    print(data_df.tail())

    # --- Run Backtest ---
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
