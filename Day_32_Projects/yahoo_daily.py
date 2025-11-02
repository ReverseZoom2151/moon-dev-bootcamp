#!/usr/bin/env python3
"""
Script to download and update daily historical stock, forex, and futures data
from Yahoo Finance using the yfinance library.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import logging
from typing import List, Tuple, Optional

# --- Configuration ---
CONFIG = {
    "SAVE_DIR": './data',
    "LOG_FILE": 'data_download.log',
    "SYMBOLS": [
        ('AAPL', 'stock'), ('GOOGL', 'stock'), ('MSFT', 'stock'), ('AMZN', 'stock'),
        ('EURUSD=X', 'forex'), ('GBPUSD=X', 'forex'), ('USDJPY=X', 'forex'), ('AUDUSD=X', 'forex'),
        ('ES=F', 'future'), ('NQ=F', 'future'), ('YM=F', 'future'), ('RTY=F', 'future'),
        ('GC=F', 'future'), ('SI=F', 'future'), ('CL=F', 'future'), ('NG=F', 'future')
    ],
    "DEFAULT_START_DATE": "2000-01-01", # Start date if no existing data
    "RATE_LIMIT_DELAY": 1 # Seconds between downloads
}

# --- Setup ---
save_dir = CONFIG["SAVE_DIR"]
log_path = os.path.join(save_dir, CONFIG["LOG_FILE"])

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# --- Core Functions ---

def load_existing_data(filepath: str) -> Optional[pd.DataFrame]:
    """Loads existing data from a CSV file, handling potential errors."""
    try:
        existing_data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        # Ensure index is timezone-naive DatetimeIndex
        if isinstance(existing_data.index, pd.DatetimeIndex):
            if existing_data.index.tz is not None:
                 existing_data.index = existing_data.index.tz_localize(None)
            logger.info(f"Loaded {len(existing_data)} rows from {filepath}. Last date: {existing_data.index[-1].date()}")
            return existing_data
        else:
            logger.warning(f"Index in {filepath} is not a DatetimeIndex after loading. Ignoring file.")
            return None
    except FileNotFoundError:
        logger.info(f"No existing file found at {filepath}.")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Existing file {filepath} is empty. Starting fresh.")
        os.remove(filepath) # Remove empty file
        return None
    except Exception as e:
        logger.error(f"Error loading existing data from {filepath}: {e}. Starting fresh.")
        return None # Treat as if file doesn't exist

def download_new_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Downloads data from yfinance for a given period, handling MultiIndex columns."""
    logger.info(f"Attempting download for {symbol} from {start_date} to {end_date}...")
    data = None
    try:
        # Convert end_date string to datetime object before adding timedelta
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        yf_end_date = end_date_dt + timedelta(days=1) # yfinance end date is exclusive

        # yfinance interval is '1d' for daily data
        data = yf.download(
            tickers=symbol,
            start=start_date, # start can be string or datetime
            end=yf_end_date,  # end should be datetime here
            interval='1d',
            progress=False, # Disable progress bar for cleaner logs
            auto_adjust=False # Keep OHLC separate, don't auto adjust
        )

        if data is None or data.empty:
            logger.warning(f"No data downloaded for {symbol} in the specified period.")
            return None

        # --- Handle Potential MultiIndex Columns ---
        if isinstance(data.columns, pd.MultiIndex):
            logger.debug(f"MultiIndex columns detected for {symbol}. Flattening...")
            # Flatten MultiIndex: take the first level (e.g., 'Open', 'Close')
            data.columns = data.columns.get_level_values(0)
            # Remove duplicate columns if any resulted from flattening
            data = data.loc[:,~data.columns.duplicated()]
            logger.debug(f"Flattened columns for {symbol}: {data.columns.tolist()}")
        # --- End MultiIndex Handling ---

        # Log columns *after* potential flattening
        logger.debug(f"Columns after processing for {symbol}: {data.columns.tolist()}")

        # Ensure index is timezone-naive DatetimeIndex
        if data.index.tz is not None:
             data.index = data.index.tz_localize(None)

        # --- More Robust Data Cleaning (using potentially flattened columns) ---
        # Define essential columns expected
        essential_cols = ['Open', 'High', 'Low', 'Close']
        volume_col = 'Volume'

        # Check for essential columns
        missing_essential = [col for col in essential_cols if col not in data.columns]
        if missing_essential:
             logger.warning(f"Essential columns missing for {symbol}: {missing_essential}. Skipping row removal.")
             # Decide if you want to return None or proceed without these columns
             # return None # Option: Skip symbol if essential data missing
        else:
             # Drop rows where essential columns are NaN (especially 'Close')
             initial_rows = len(data)
             data.dropna(subset=['Close'], inplace=True) # Primarily focus on Close
             if len(data) < initial_rows:
                 logger.debug(f"Removed {initial_rows - len(data)} rows with NaN 'Close' for {symbol}.")

        # Ensure Volume is integer if present
        if volume_col in data.columns:
             try:
                 # Convert to numeric first (handles non-numeric errors), fill NaNs with 0, then convert to int
                 data[volume_col] = pd.to_numeric(data[volume_col], errors='coerce').fillna(0).astype(int)
             except Exception as vol_e:
                  logger.warning(f"Could not convert Volume column to int for {symbol}: {vol_e}")
                  # Decide how to handle: drop volume, keep as float, etc.
                  # data.drop(columns=[volume_col], inplace=True) # Option: Drop Volume column
        else:
            logger.warning(f"Volume column missing for {symbol}.")
        # --- End Cleaning ---

        if data.empty:
             logger.warning(f"Data for {symbol} became empty after cleaning.")
             return None

        logger.info(f"Successfully downloaded and cleaned {len(data)} new rows for {symbol}.")
        return data

    except KeyError as ke:
        # Catch KeyError specifically if it still occurs (e.g., during cleaning)
        logger.error(f"KeyError processing data for {symbol}: {ke}. Columns: {data.columns.tolist() if data is not None else 'N/A'}")
        return None
    except Exception as e:
        # Catching generic Exception as yfinance errors can vary
        logger.error(f"Error during download/processing for {symbol}: {e}", exc_info=True) # Log full traceback
        return None

def process_and_save(symbol: str, asset_type: str,
                     existing_data: Optional[pd.DataFrame],
                     new_data: Optional[pd.DataFrame]) -> int:
    """Merges new data with existing, removes duplicates, sorts, and saves."""
    filepath = os.path.join(CONFIG["SAVE_DIR"], f"{symbol.replace('=', '_')}_{asset_type}_1d.csv")
    final_data = existing_data

    if new_data is not None and not new_data.empty:
        if existing_data is not None and not existing_data.empty:
             logger.info(f"Merging {len(existing_data)} existing rows with {len(new_data)} new rows for {symbol}.")
             # Use concat and drop duplicates based on index
             combined = pd.concat([existing_data, new_data])
             final_data = combined[~combined.index.duplicated(keep='last')] # Keep last entry for a date
        else:
             logger.info(f"No existing data for {symbol}, using {len(new_data)} downloaded rows.")
             final_data = new_data
    elif existing_data is not None:
         logger.info(f"No new data downloaded for {symbol}. Keeping existing {len(existing_data)} rows.")
    else:
         logger.info(f"No existing or new data available for {symbol}.")
         return 0 # No rows to save

    if final_data is not None and not final_data.empty:
        try:
            # Sort by date index before saving
            final_data.sort_index(inplace=True)
            # Save to CSV
            final_data.to_csv(filepath, date_format='%Y-%m-%d')
            logger.info(f"Successfully saved {len(final_data)} total rows to {filepath}")
            return len(final_data)
        except IOError as e:
            logger.error(f"Error saving data for {symbol} to {filepath}: {e}")
            return len(final_data) # Return count even if save failed, as data exists in memory
        except Exception as e:
            logger.error(f"Unexpected error during saving for {symbol}: {e}")
            return len(final_data)
    else:
         return 0 # Should not happen if logic is correct, but safe fallback

def download_and_update_symbol(symbol_info: Tuple[str, str]) -> int:
    """Handles the full download and update process for a single symbol."""
    symbol, asset_type = symbol_info
    logger.info(f"--- Processing symbol: {symbol} ({asset_type}) ---")
    filepath = os.path.join(CONFIG["SAVE_DIR"], f"{symbol.replace('=', '_')}_{asset_type}_1d.csv")

    # 1. Load existing data
    existing_data = load_existing_data(filepath)

    # 2. Determine start date for download
    if existing_data is not None and not existing_data.empty:
        start_date_dt = existing_data.index[-1].date() + timedelta(days=1)
        start_date_str = start_date_dt.strftime('%Y-%m-%d')
    else:
        start_date_str = CONFIG["DEFAULT_START_DATE"]
        start_date_dt = datetime.strptime(start_date_str, '%Y-%m-%d').date()

    # 3. Determine end date (yesterday)
    end_date_dt = datetime.now().date() - timedelta(days=1)
    end_date_str = end_date_dt.strftime('%Y-%m-%d')

    # 4. Check if download is needed
    new_data = None
    if start_date_dt <= end_date_dt:
        # Add rate limiting delay before download
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
        new_data = download_new_data(symbol, start_date_str, end_date_str)
    else:
        logger.info(f"Data for {symbol} is already up-to-date (last date: {existing_data.index[-1].date()}).")

    # 5. Process and save
    total_rows = process_and_save(symbol, asset_type, existing_data, new_data)
    logger.info(f"--- Finished processing {symbol}. Total rows: {total_rows} ---")
    return total_rows


def download_all_daily_data(symbols: List[Tuple[str, str]]) -> None:
    """Iterates through symbols and downloads/updates data for each."""
    logger.info("=== Starting all daily data download/update process ===")
    total_rows_map = {}
    for symbol_info in symbols:
        try:
            rows = download_and_update_symbol(symbol_info)
            total_rows_map[symbol_info[0]] = rows
        except Exception as e:
            logger.error(f"Critical error processing symbol {symbol_info[0]}: {e}", exc_info=True)
            total_rows_map[symbol_info[0]] = -1 # Indicate error

    logger.info("=== All daily data download/update attempts completed ===")
    logger.info(f"Final row counts: {total_rows_map}")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Script started.")
    download_all_daily_data(CONFIG["SYMBOLS"])
    logger.info("Script finished.")