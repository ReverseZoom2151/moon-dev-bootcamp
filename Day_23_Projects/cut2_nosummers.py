import pandas as pd
import os
import warnings
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore') # Suppress potential warnings

# --- Configuration ---
# Assumes the script is run from Day_23_Projects and data is in a sibling folder
DATA_FOLDER_NAME = 'test_data'
INPUT_FILENAME = 'BTC-1m-10wks-data.csv' # Update if needed
OUTPUT_SUFFIX = '-nosummers'

# Months to exclude (inclusive range)
EXCLUDE_START_MONTH = 5  # May
EXCLUDE_END_MONTH = 9    # September

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
            # If already timezone-aware, convert to UTC
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

def filter_exclude_months(df, start_month, end_month):
    """Filters the DataFrame to exclude a range of months."""
    print(f"Filtering data to exclude months {start_month} through {end_month}...")
    try:
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
             raise ValueError("DataFrame index must be a DatetimeIndex.")
             
        # Create a boolean mask for rows *within* the exclude range
        exclude_mask = (df.index.month >= start_month) & (df.index.month <= end_month)
        
        # Keep rows where the mask is False (i.e., outside the exclude range)
        filtered_df = df[~exclude_mask]
        
        print(f"Filtering complete. Result shape: {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        print(f"Error during filtering: {e}")
        return None

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
    filtered_data = filter_exclude_months(data, EXCLUDE_START_MONTH, EXCLUDE_END_MONTH)
    
    if filtered_data is None or filtered_data.empty:
        print("Filtering resulted in empty data or an error occurred. No file saved.")
        return
        
    # Construct output path
    base, ext = os.path.splitext(input_file_path)
    output_path = f"{base}{OUTPUT_SUFFIX}{ext}"

    # Save filtered data
    save_data(filtered_data, output_path)

# --- Main execution guard ---
if __name__ == '__main__':
    main()