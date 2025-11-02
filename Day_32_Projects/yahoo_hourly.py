#!/usr/bin/env python3
"""
Script to download and update hourly historical stock and forex data
from Yahoo Finance using the yfinance library.

Note: Yahoo Finance typically limits hourly data downloads to the last 730 days.
This script downloads/updates data within that allowed period.
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
    "LOG_FILE": 'data_download_hourly.log', # Separate log file for hourly
    # Symbols excluding futures (hourly data for futures can be complex due to sessions)
    "SYMBOLS": [
        ('AAPL', 'stock'), ('GOOGL', 'stock'), ('MSFT', 'stock'), ('AMZN', 'stock'),
        ('EURUSD=X', 'forex'), ('GBPUSD=X', 'forex'), ('USDJPY=X', 'forex'), ('AUDUSD=X', 'forex')
    ],
    "HOURLY_FETCH_DAYS": 728, # Reduced from 730 to be safer
    "RATE_LIMIT_DELAY": 1     # Seconds between downloads
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

def load_existing_hourly_data(filepath: str) -> Optional[pd.DataFrame]:
    """Loads existing hourly data from a CSV file, handling potential errors."""
    try:
        # Hourly data usually has 'Datetime' index from yfinance
        existing_data = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)
        if isinstance(existing_data.index, pd.DatetimeIndex):
            if existing_data.index.tz is not None:
                 existing_data.index = existing_data.index.tz_localize(None)
            logger.info(f"Loaded {len(existing_data)} rows from {filepath}. Last datetime: {existing_data.index[-1]}")
            return existing_data
        else:
            logger.warning(f"Index in {filepath} is not a DatetimeIndex after loading. Ignoring file.")
            return None
    except FileNotFoundError:
        logger.info(f"No existing file found at {filepath}.")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Existing file {filepath} is empty. Starting fresh.")
        os.remove(filepath)
        return None
    except Exception as e:
        logger.error(f"Error loading existing hourly data from {filepath}: {e}. Starting fresh.")
        return None

def download_new_hourly_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Downloads hourly data from yfinance, handling MultiIndex columns."""
    logger.info(f"Attempting hourly download for {symbol} from {start_date} to {end_date}...")
    data = None # Initialize
    try:
        # Convert end_date string to date object and add 1 day to make it exclusive for yf
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        yf_end_date = end_date_dt + timedelta(days=1)

        data = yf.download(
            tickers=symbol,
            start=start_date,
            end=yf_end_date,
            interval='1h',
            progress=False,
            auto_adjust=False
        )

        if data is None or data.empty:
            logger.warning(f"No hourly data downloaded for {symbol} in the specified period.")
            return None

        # --- Handle Potential MultiIndex Columns ---
        if isinstance(data.columns, pd.MultiIndex):
            logger.debug(f"MultiIndex columns detected for {symbol} (hourly). Flattening...")
            data.columns = data.columns.get_level_values(0)
            data = data.loc[:,~data.columns.duplicated()]
            logger.debug(f"Flattened columns for {symbol} (hourly): {data.columns.tolist()}")
        # --- End MultiIndex Handling ---

        logger.debug(f"Columns after processing for {symbol} (hourly): {data.columns.tolist()}")

        # Ensure index is timezone-naive DatetimeIndex
        if data.index.tz is not None:
             data.index = data.index.tz_localize(None)

        # --- Cleaning (using potentially flattened columns) ---
        essential_cols = ['Open', 'High', 'Low', 'Close']
        volume_col = 'Volume'

        missing_essential = [col for col in essential_cols if col not in data.columns]
        if missing_essential:
             logger.warning(f"Essential columns missing for {symbol} (hourly): {missing_essential}. Skipping row removal.")
        else:
             initial_rows = len(data)
             data.dropna(subset=['Close'], inplace=True)
             if len(data) < initial_rows:
                 logger.debug(f"Removed {initial_rows - len(data)} hourly rows with NaN 'Close' for {symbol}.")

        if volume_col in data.columns:
             try:
                 data[volume_col] = pd.to_numeric(data[volume_col], errors='coerce').fillna(0).astype(int)
             except Exception as vol_e:
                  logger.warning(f"Could not convert hourly Volume column to int for {symbol}: {vol_e}")
        else:
            logger.warning(f"Volume column missing for {symbol} (hourly).")

        if data.empty:
             logger.warning(f"Hourly data for {symbol} became empty after cleaning.")
             return None
        # --- End Cleaning ---

        logger.info(f"Successfully downloaded and cleaned {len(data)} new hourly rows for {symbol}.")
        return data

    except KeyError as ke:
        logger.error(f"KeyError processing hourly data for {symbol}: {ke}. Columns: {data.columns.tolist() if data is not None else 'N/A'}")
        return None
    except Exception as e:
        logger.error(f"Error during hourly download/processing for {symbol}: {e}", exc_info=True)
        return None

def process_and_save_hourly(symbol: str, asset_type: str,
                            existing_data: Optional[pd.DataFrame],
                            new_data: Optional[pd.DataFrame]) -> int:
    """Merges new hourly data with existing, removes duplicates, sorts, and saves."""
    # Use different filename suffix for hourly
    filepath = os.path.join(CONFIG["SAVE_DIR"], f"{symbol.replace('=', '_')}_{asset_type}_1h.csv")
    final_data = existing_data

    if new_data is not None and not new_data.empty:
        if existing_data is not None and not existing_data.empty:
             logger.info(f"Merging {len(existing_data)} existing hourly rows with {len(new_data)} new rows for {symbol}.")
             combined = pd.concat([existing_data, new_data])
             final_data = combined[~combined.index.duplicated(keep='last')]
        else:
             logger.info(f"No existing hourly data for {symbol}, using {len(new_data)} downloaded rows.")
             final_data = new_data
    elif existing_data is not None:
         logger.info(f"No new hourly data downloaded for {symbol}. Keeping existing {len(existing_data)} rows.")
    else:
         logger.info(f"No existing or new hourly data available for {symbol}.")
         return 0

    if final_data is not None and not final_data.empty:
        try:
            final_data.sort_index(inplace=True)
            # Save with appropriate datetime format
            final_data.to_csv(filepath, date_format='%Y-%m-%d %H:%M:%S')
            logger.info(f"Successfully saved {len(final_data)} total hourly rows to {filepath}")
            return len(final_data)
        except IOError as e:
            logger.error(f"Error saving hourly data for {symbol} to {filepath}: {e}")
            return len(final_data)
        except Exception as e:
            logger.error(f"Unexpected error during hourly saving for {symbol}: {e}")
            return len(final_data)
    else:
         return 0

def download_and_update_hourly_symbol(symbol_info: Tuple[str, str]) -> int:
    """Handles the full download and update process for a single symbol (hourly)."""
    symbol, asset_type = symbol_info
    logger.info(f"--- Processing HOURLY symbol: {symbol} ({asset_type}) ---")
    filepath = os.path.join(CONFIG["SAVE_DIR"], f"{symbol.replace('=', '_')}_{asset_type}_1h.csv")

    # 1. Load existing data
    existing_data = load_existing_hourly_data(filepath)

    # 2. Determine date range for download
    max_hist_days = CONFIG["HOURLY_FETCH_DAYS"]
    end_date_dt = datetime.now() # Use datetime object here initially
    earliest_start_dt = end_date_dt - timedelta(days=max_hist_days)

    if existing_data is not None and not existing_data.empty:
        last_timestamp = existing_data.index[-1]
        start_date_dt = last_timestamp + timedelta(hours=1)
        start_date_dt = max(start_date_dt, earliest_start_dt)
    else:
        start_date_dt = earliest_start_dt

    # Convert to DATE strings only for the yfinance download call
    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    end_date_str = end_date_dt.strftime('%Y-%m-%d') # Use only the date part

    # 4. Check if download is needed
    new_data = None
    # Compare dates only, not times, to decide if a download attempt is needed
    if start_date_dt.date() <= end_date_dt.date():
        time.sleep(CONFIG["RATE_LIMIT_DELAY"])
        # Pass DATE strings to the download function
        new_data = download_new_hourly_data(symbol, start_date_str, end_date_str)
    elif existing_data is not None:
        logger.info(f"Hourly data for {symbol} seems up-to-date (last timestamp: {existing_data.index[-1]}).")
    else:
         logger.info(f"Could not determine need for download for {symbol} (no existing data and start >= end).")

    # 5. Process and save
    total_rows = process_and_save_hourly(symbol, asset_type, existing_data, new_data)
    logger.info(f"--- Finished processing HOURLY {symbol}. Total rows: {total_rows} ---")
    return total_rows


def download_all_hourly_data(symbols: List[Tuple[str, str]]) -> None:
    """Iterates through symbols and downloads/updates hourly data for each."""
    logger.info("=== Starting all HOURLY data download/update process ===")
    total_rows_map = {}
    for symbol_info in symbols:
        try:
            rows = download_and_update_hourly_symbol(symbol_info)
            total_rows_map[symbol_info[0]] = rows
        except Exception as e:
            logger.error(f"Critical error processing HOURLY symbol {symbol_info[0]}: {e}", exc_info=True)
            total_rows_map[symbol_info[0]] = -1

    logger.info("=== All HOURLY data download/update attempts completed ===")
    logger.info(f"Final hourly row counts: {total_rows_map}")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Hourly data download script started.")
    download_all_hourly_data(CONFIG["SYMBOLS"])
    logger.info("Hourly data download script finished.")
