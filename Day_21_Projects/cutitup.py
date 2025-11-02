import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# File path (relative to the script location)
file_path = './WIF_liq_data.csv'
output_dir = './output_data'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file into a DataFrame, using the header from the file
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Input file not found at {file_path}")
    exit()

# Ensure required columns exist
required_columns = [
    "symbol", "side", "order_type", "time_in_force",
    "original_quantity", "price", "average_price", "order_status",
    "order_last_filled_quantity", "order_filled_accumulated_quantity",
    "order_trade_time", "usd_size"
]
if not all(col in df.columns for col in required_columns):
    print(f"Error: Input CSV missing required columns. Found: {df.columns}. Required: {required_columns}")
    exit()


# Function to determine liquidation type (Vectorized)
valid_symbol = (df['symbol'].astype(str).str.len() == 3) | (df['symbol'] == "1000PEPE")
large_size = df['usd_size'] > 3000
is_sell = df['side'] == "SELL"
is_buy = df['side'] == "BUY"

conditions = [
    valid_symbol & large_size & is_sell,
    valid_symbol & large_size & is_buy
]
choices = ["L LIQ", "S LIQ"]

df['LIQ_SIDE'] = np.select(conditions, choices, default=None)


# Filter out rows where LIQ_SIDE is None
df = df[df['LIQ_SIDE'].notna()].copy() # Use .copy() to avoid potential SettingWithCopyWarning

# Convert epoch to datetime
df['datetime'] = pd.to_datetime(df['order_trade_time'], unit='ms')
# Removed unused formatting line: df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))

# Set `datetime` as index
df.set_index('datetime', inplace=True)

# Adjust symbol handling: Ensure symbol is treated as string before slicing
df['symbol'] = df['symbol'].astype(str).apply(lambda x: x if x == "1000PEPE" else x[:3])


# List of symbols to process
symbols = ["BTC", "ETH", "SOL", "WIF", "1000PEPE"]

# Filter and save the DataFrame for each symbol
for symbol in symbols:
    # Filter DataFrame by symbol
    # Ensure comparison is robust, e.g., handle case sensitivity if needed
    filtered_df = df[df['symbol'].str.upper() == symbol.upper()] # Case-insensitive match

    if not filtered_df.empty:
        # Save the filtered DataFrame to a new CSV file in the output directory
        output_path = os.path.join(output_dir, f'{symbol}_liq_data.csv')
        filtered_df.to_csv(output_path)
        print(f"Saved data for {symbol} to {output_path}")
    else:
        print(f"No data found for symbol {symbol} in the input file.")


# Prepare DataFrame for resampling totals
df_all = df.reset_index() # Reset index to make 'datetime' a column for resampling setup
df_all['datetime'] = pd.to_datetime(df_all['datetime']) # Ensure it's datetime type
df_all.set_index('datetime', inplace=True) # Set 'datetime' back as index for resampling

# Separate resample for L LIQ and S LIQ
# Use observed=True to avoid creating entries for intervals with no data
df_totals_L = df_all[df_all['LIQ_SIDE'] == 'L LIQ'].resample('5min').agg({'usd_size': 'sum', 'price': 'mean'})
df_totals_S = df_all[df_all['LIQ_SIDE'] == 'S LIQ'].resample('5min').agg({'usd_size': 'sum', 'price': 'mean'})


# Add LIQ_SIDE column back
df_totals_L['LIQ_SIDE'] = 'L LIQ'
df_totals_S['LIQ_SIDE'] = 'S LIQ'

# Combine the dataframes
df_totals = pd.concat([df_totals_L, df_totals_S])

# Add symbol column set to 'All'
df_totals['symbol'] = 'All'

# Reset index to have 'datetime' as a column
df_totals.reset_index(inplace=True)

# Reorder the DataFrame columns
df_totals = df_totals[['datetime', 'symbol', 'LIQ_SIDE', 'price', 'usd_size']]

# Save the aggregated totals DataFrame
output_totals_path = os.path.join(output_dir, 'total_liq_data.csv')
df_totals.to_csv(output_totals_path, index=False)
print(f"Saved aggregated totals to {output_totals_path}")

# Ensure the 'totals' DataFrame is generated properly
print("Aggregated Totals Head:")
print(df_totals.head())