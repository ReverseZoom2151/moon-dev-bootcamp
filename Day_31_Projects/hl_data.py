import dontshare as d
import pandas as pd
import datetime
import os
import ccxt
from math import ceil  
import time # Added for potential delays

# --- Configuration ---
# Use variables for easier modification
SYMBOL = 'UNI/USD'  # Example symbol, ensure it's available on Coinbase
TIMEFRAME = '1h'    # Example timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
WEEKS_TO_FETCH = 100 # Number of weeks of historical data
CACHE_DIR = '.'     # Directory to save cached data files (current directory)
FETCH_LIMIT = 200   # Number of candles to fetch per API call (check exchange limits)

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Helper Functions ---

def timeframe_to_sec(timeframe: str) -> int:
    """Converts CCXT timeframe string to seconds."""
    amount = int("".join(filter(str.isdigit, timeframe)))
    unit = "".join(filter(str.isalpha, timeframe))

    if unit == 'm':
        return amount * 60
    elif unit == 'h':
        return amount * 60 * 60
    elif unit == 'd':
        return amount * 24 * 60 * 60
    elif unit == 'w': # CCXT might use 'w' for weeks, though less common for OHLCV
        return amount * 7 * 24 * 60 * 60
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

def get_historical_data(symbol: str, timeframe: str, weeks: int) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for a given symbol, timeframe, and number of weeks.
    Caches results to a CSV file.
    """
    # Construct cache filename
    cache_filename = os.path.join(CACHE_DIR, f'{symbol.replace("/", "_")}-{timeframe}-{weeks}wks-data.csv')

    # Check if cached file exists
    if os.path.exists(cache_filename):
        print(f"Loading cached data from: {cache_filename}")
        try:
            # Explicitly parse dates when loading from CSV
            return pd.read_csv(cache_filename, index_col='datetime', parse_dates=True)
        except Exception as e:
            print(f"Error loading cache file {cache_filename}: {e}. Fetching new data.")

    print(f"No cache found or cache error. Fetching new data for {symbol} ({timeframe}, {weeks} weeks)...")

    # Initialize Coinbase exchange
    try:
        coinbase = ccxt.coinbase({
            'apiKey': d.api_key,
            'secret': d.api_secret,
            'enableRateLimit': True, # Enable built-in rate limiting
        })
        # Optional: Check if the market exists
        markets = coinbase.load_markets()
        if symbol not in markets:
             raise ValueError(f"Symbol {symbol} not found on Coinbase.")

    except AttributeError:
         raise ImportError("Failed to import api_key or api_secret from dontshare.py")
    except ccxt.AuthenticationError:
         raise ValueError("Coinbase authentication failed. Check API key and secret.")
    except ccxt.ExchangeError as e:
         raise ConnectionError(f"Failed to connect to Coinbase: {e}")

    # Calculate required parameters
    granularity_sec = timeframe_to_sec(timeframe)
    total_time_sec = weeks * 7 * 24 * 60 * 60
    # Calculate runs needed based on FETCH_LIMIT candles per run
    run_times = ceil(total_time_sec / (granularity_sec * FETCH_LIMIT))

    print(f"Calculated granularity: {granularity_sec}s")
    print(f"Total time to fetch: {total_time_sec}s")
    print(f"Required API calls (runs): {run_times}")

    # Fetch data in chunks
    all_ohlcv = []
    now = datetime.datetime.now(datetime.timezone.utc)
    since_timestamp_ms = int((now - datetime.timedelta(weeks=weeks)).timestamp() * 1000)

    print(f"Starting fetch from: {datetime.datetime.fromtimestamp(since_timestamp_ms / 1000, tz=datetime.timezone.utc)}")

    current_timestamp_ms = since_timestamp_ms
    for i in range(run_times): # Fetch backwards using 'since' is generally more reliable with pagination
        print(f"Fetching run {i+1}/{run_times} starting from timestamp {current_timestamp_ms}...")
        try:
            # Fetch OHLCV data
            ohlcv = coinbase.fetch_ohlcv(symbol, timeframe, since=current_timestamp_ms, limit=FETCH_LIMIT)

            if not ohlcv:
                print(f"No more data returned for {symbol} at timestamp {current_timestamp_ms}. Stopping fetch.")
                break # Stop if no data is returned

            all_ohlcv.extend(ohlcv)

            # Update the timestamp for the next iteration to the timestamp of the *last* candle + 1 interval
            last_candle_ts = ohlcv[-1][0]
            current_timestamp_ms = last_candle_ts + (granularity_sec * 1000)

            # Optional: Add a small delay to respect rate limits further if needed
            # time.sleep(coinbase.rateLimit / 1000)

        except ccxt.RateLimitExceeded as e:
            print(f"Rate limit exceeded: {e}. Waiting...")
            time.sleep(60) # Wait a minute before retrying this chunk
            # Re-try the same chunk by not updating current_timestamp_ms here, or implement backoff
            # For simplicity here, we might lose a chunk on retry failure
        except ccxt.NetworkError as e:
            print(f"Network error fetching chunk {i+1}: {e}. Retrying after delay...")
            time.sleep(10)
             # Consider retrying the same chunk
        except ccxt.ExchangeError as e:
            print(f"Exchange error fetching chunk {i+1}: {e}. Skipping chunk.")
            # Decide how to handle: skip chunk or stop? Skipping might leave gaps.
            # Updating timestamp to avoid infinite loop on persistent error for a range
            current_timestamp_ms += (granularity_sec * 1000 * FETCH_LIMIT)
        except Exception as e:
             print(f"Unexpected error during fetch run {i+1}: {e}. Stopping fetch.")
             break # Stop on unexpected errors


    if not all_ohlcv:
        print("Error: No data fetched.")
        return pd.DataFrame() # Return empty DataFrame

    # --- Process Fetched Data ---
    print(f"Fetched total {len(all_ohlcv)} candles.")

    # Convert to DataFrame
    dataframe = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # Convert timestamp to datetime and set as index
    dataframe['datetime'] = pd.to_datetime(dataframe['timestamp'], unit='ms', utc=True)
    dataframe.set_index('datetime', inplace=True)
    dataframe.drop('timestamp', axis=1, inplace=True) # Drop the original timestamp column

    # Ensure data types are correct (redundant if fetch_ohlcv guarantees floats, but safe)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
         dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    dataframe.dropna(inplace=True) # Drop rows if conversion failed

    # Remove duplicate timestamps (important!)
    initial_rows = len(dataframe)
    dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
    print(f"Removed {initial_rows - len(dataframe)} duplicate entries based on timestamp.")

    # Sort by datetime index
    dataframe.sort_index(inplace=True)

    # Select final columns (already done by DataFrame creation)
    # dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']] # Ensure order if needed

    # Save to CSV
    try:
        dataframe.to_csv(cache_filename)
        print(f"Data saved successfully to: {cache_filename}")
    except IOError as e:
        print(f"Error saving data to cache file {cache_filename}: {e}")

    return dataframe


# --- Main Execution ---

if __name__ == "__main__":
    print(f"--- Fetching Historical Data for {SYMBOL} --- ")
    print(f"Timeframe: {TIMEFRAME}, Weeks: {WEEKS_TO_FETCH}")

    try:
        # Check if dontshare exists and has keys
        if not (hasattr(d, 'api_key') and hasattr(d, 'api_secret')):
             raise ImportError("dontshare.py must contain 'api_key' and 'api_secret'")

        historical_data = get_historical_data(SYMBOL, TIMEFRAME, WEEKS_TO_FETCH)

        if not historical_data.empty:
            print("\n--- Data Summary ---")
            print(f"Shape: {historical_data.shape}")
            print(f"Start Date: {historical_data.index.min()}")
            print(f"End Date: {historical_data.index.max()}")
            print("\n--- First 5 Rows ---")
            print(historical_data.head())
            print("\n--- Last 5 Rows ---")
            print(historical_data.tail())
        else:
            print("\nNo data was fetched or loaded.")

    except (ImportError, ValueError, ConnectionError) as e:
         print(f"\nError: {e}")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
         import traceback
         traceback.print_exc()

    print("\n--- Script Finished ---")
