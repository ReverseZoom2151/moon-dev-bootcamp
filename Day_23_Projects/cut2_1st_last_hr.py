import pandas as pd
import os
import warnings
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore') # Suppress potential warnings

# --- Configuration ---
# Assumes the script is run from Day_23_Projects and data is in a sibling folder
# Adjust DATA_FOLDER_NAME and INPUT_FILENAME as needed
DATA_FOLDER_NAME = 'test_data'
INPUT_FILENAME = 'BTC-1m-10wks-data.csv' 
OUTPUT_SUFFIX = '-1stlasthr'

# Trading hours in UTC (adjust if your data or needs differ)
NYSE_OPEN_UTC = '13:30'  # 9:30 AM EDT in UTC
ONE_HOUR_AFTER_OPEN_UTC = '14:30'  # 10:30 AM EDT in UTC
ONE_HOUR_BEFORE_CLOSE_UTC = '19:00'  # 3:00 PM EDT in UTC
NYSE_CLOSE_UTC = '20:00'  # 4:00 PM EDT in UTC

# --- Functions ---

def load_data(file_path):
    """Loads CSV data, parses datetime, and sets index."""
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

def filter_trading_hours(df, start_time_first_hour, end_time_first_hour, 
                           start_time_last_hour, end_time_last_hour):
    """Filters the DataFrame for the first and last hour of trading, excluding weekends."""
    print(f"Filtering for {start_time_first_hour}-{end_time_first_hour} and {start_time_last_hour}-{end_time_last_hour} UTC, weekdays only...")
    try:
        # Ensure index is UTC DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz.zone != 'UTC':
             raise ValueError("DataFrame index must be a UTC-localized DatetimeIndex.")
             
        # Filter data for 1 hour after open
        open_hours = df.between_time(start_time_first_hour, end_time_first_hour)
        open_hours = open_hours[open_hours.index.dayofweek < 5]  # Exclude weekends (Mon=0, Sun=6)

        # Filter data for 1 hour before close
        close_hours = df.between_time(start_time_last_hour, end_time_last_hour)
        close_hours = close_hours[close_hours.index.dayofweek < 5]  # Exclude weekends

        # Concatenate the two filtered segments
        filtered_data = pd.concat([open_hours, close_hours]).sort_index()
        print(f"Filtering complete. Result shape: {filtered_data.shape}")
        return filtered_data
    except Exception as e:
        print(f"Error during filtering: {e}")
        return None

def save_filtered_data(df, input_path, suffix):
    """Saves the filtered DataFrame to a new CSV file."""
    try:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}{suffix}{ext}"
        df.to_csv(output_path)
        print(f"Filtered data saved successfully to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving filtered data to '{output_path}': {e}")
        return None

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
    filtered_data = filter_trading_hours(data, 
                                         start_time_first_hour=NYSE_OPEN_UTC, 
                                         end_time_first_hour=ONE_HOUR_AFTER_OPEN_UTC, 
                                         start_time_last_hour=ONE_HOUR_BEFORE_CLOSE_UTC, 
                                         end_time_last_hour=NYSE_CLOSE_UTC)
    
    if filtered_data is None or filtered_data.empty:
        print("Filtering resulted in empty data or an error occurred. No file saved.")
        return
        
    # Save filtered data
    save_filtered_data(filtered_data, input_file_path, OUTPUT_SUFFIX)

# --- Main execution guard ---
if __name__ == '__main__':
    main()