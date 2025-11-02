import pandas as pd
import os
import warnings
import datetime # Import datetime for time objects
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore') # Suppress potential warnings

# --- Configuration ---
# Assumes the script is run from Day_23_Projects and data is in a sibling folder
DATA_FOLDER_NAME = 'test_data'
INPUT_FILENAME = 'BTC-5m-520wks-data.csv' # Update if needed
MARKET_HOURS_SUFFIX = '-mkt-open'
NON_MARKET_HOURS_SUFFIX = '-mkt-closed'

# Market hours in UTC (adjust if your data or needs differ)
MARKET_OPEN_UTC_STR = '13:30'  # 9:30 AM EDT in UTC
MARKET_CLOSE_UTC_STR = '20:00' # 4:00 PM EDT in UTC

# --- Functions ---

def load_data(file_path):
    """Loads CSV data, parses datetime, localizes to UTC, and sets index."""
    print(f"Loading data from: {file_path}")
    try:
        data = pd.read_csv(file_path, parse_dates=['datetime'])
        # Assume datetime is naive UTC, localize it
        if data['datetime'].dt.tz is None:
            print("Localizing naive datetime index to UTC...")
            data['datetime'] = data['datetime'].dt.tz_localize('UTC')
        else:
            # If already timezone-aware, convert to UTC just in case
            print("Converting timezone-aware datetime index to UTC...")
            data['datetime'] = data['datetime'].dt.tz_convert('UTC')
            
        data.set_index('datetime', inplace=True)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        return None

def filter_market_non_market_hours(df, open_time_str, close_time_str):
    """Filters the DataFrame into market hours and non-market hours (weekdays only)."""
    print(f"Filtering for market hours ({open_time_str}-{close_time_str} UTC) and non-market hours, weekdays only...")
    try:
        # Ensure index is UTC DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz.zone != 'UTC':
             raise ValueError("DataFrame index must be a UTC-localized DatetimeIndex.")
             
        # Convert time strings to time objects for comparison
        open_time = datetime.datetime.strptime(open_time_str, '%H:%M').time()
        close_time = datetime.datetime.strptime(close_time_str, '%H:%M').time()

        # Filter for weekdays first
        weekdays_df = df[df.index.dayofweek < 5].copy() # Monday=0, Sunday=6

        # Filter for market hours: [open_time, close_time)
        # between_time is inclusive by default, use inclusive='left' for [) behavior
        market_hours_df = weekdays_df.between_time(open_time_str, close_time_str, inclusive='left')

        # Filter for non-market hours: [close_time, open_time) on weekdays
        # Original logic: time < open_time OR time >= close_time
        non_market_hours_mask = (weekdays_df.index.time < open_time) | \
                                (weekdays_df.index.time >= close_time)
        non_market_hours_df = weekdays_df[non_market_hours_mask]

        print(f"Filtering complete. Market hours shape: {market_hours_df.shape}, Non-market hours shape: {non_market_hours_df.shape}")
        return market_hours_df, non_market_hours_df
    except Exception as e:
        print(f"Error during filtering: {e}")
        return None, None

def save_data(df, output_path):
    """Saves the DataFrame to a CSV file."""
    try:
        df.to_csv(output_path)
        print(f"Data saved successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data to '{output_path}': {e}")
        return False

# --- Main Execution Logic ---
def main():
    """Orchestrates the loading, filtering, and saving process."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(script_dir, DATA_FOLDER_NAME)
    input_file_path = os.path.join(data_folder_path, INPUT_FILENAME)
    
    # Check if data folder exists
    if not os.path.isdir(data_folder_path):
        print(f"Error: Data folder '{DATA_FOLDER_NAME}' not found in script directory '{script_dir}'")
        print(f"Expected path: {data_folder_path}")
        return

    # Load data
    data = load_data(input_file_path)
    if data is None:
        return

    # Filter data
    market_hours_data, non_market_hours_data = filter_market_non_market_hours(
        data, MARKET_OPEN_UTC_STR, MARKET_CLOSE_UTC_STR
    )
    
    if market_hours_data is None or non_market_hours_data is None:
        print("Filtering failed. No files saved.")
        return
        
    # Construct output paths
    base, ext = os.path.splitext(input_file_path)
    output_path_market = f"{base}{MARKET_HOURS_SUFFIX}{ext}"
    output_path_non_market = f"{base}{NON_MARKET_HOURS_SUFFIX}{ext}"

    # Save filtered data
    if not market_hours_data.empty:
        save_data(market_hours_data, output_path_market)
    else:
        print("Market hours data is empty. Skipping save.")
        
    if not non_market_hours_data.empty:
        save_data(non_market_hours_data, output_path_non_market)
    else:
        print("Non-market hours data is empty. Skipping save.")

# --- Main execution guard ---
if __name__ == '__main__':
    main()