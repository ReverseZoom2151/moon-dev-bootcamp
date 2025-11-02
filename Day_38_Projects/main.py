'''
THIS IS THE FILE WHERE I FIND NEW PPL TO FOLLOW
ALL I DO IS PUT THE TOKEN ADDRESS IN BELOW AND IT LOOKS FOR THE EARLY BUYERS
THEN SAVES TO TO A CSV

To run:
1. put your birdeye api key in dontshare.py
2. put the token address you wanna see early buyers of in the TOKEN_ADDRESS variable
3. make an output folder and put the path in OUTPUT_FOLDER   
4. run the file


2qEHjDLDLbuBgRYvsxhc5D6uDWAivNFZGan56P1tpump - pnut
9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump - fart
GJAFwWjJ3vnTsrQVabjBVK2TYB1YtRCQXRDfDgUnpump - act
'''


import dontshare as d # dontshare.py is where i put my birdeye api key
import requests
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# --- Configuration ---

@dataclass(frozen=True)
class Config:
    """Script configuration."""
    TOKEN_ADDRESS: str = "Df6yfrKC8kZE3KNkrHERKzAetSxbrWeniQfyJY4Jpump" # Example: pnut
    # Dates in MM-DD-YYYY format
    START_DATE_STR: str = "11-20-2020"
    END_DATE_STR: str = "11-27-2027" # Set far in the future to get all trades up to now if needed
    MIN_TRADE_SIZE_USD: float = 3000.0
    MAX_TRADE_SIZE_USD: float = 100000000.0
    OUTPUT_DIR: Path = Path("output_data") # Relative path for output
    SORT_TYPE: str = "asc" # "asc" for earliest first, "desc" for latest first
    API_BASE_URL: str = "https://public-api.birdeye.so"
    EXPLORER_BASE_URL: str = "https://gmgn.ai/sol/address/"
    # API fetch settings
    FETCH_LIMIT: int = 50
    MAX_OFFSET: int = 100000 # Safety limit for API calls
    RETRY_DELAY_SECONDS: int = 5
    MAX_CONSECUTIVE_ERRORS: int = 3
    MAX_EMPTY_BATCHES: int = 3
    REQUEST_DELAY_SECONDS: float = 0.1 # Delay between API calls

# Initialize config
CONFIG = Config()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Attempt to get API key from environment variable first, then dontshare.py
API_KEY = os.getenv("BIRDEYE_API_KEY")
if not API_KEY:
    try:
        API_KEY = d.birdeye_api_key
        logger.info("Using API key from dontshare.py")
    except (ImportError, AttributeError):
        logger.error("Birdeye API key not found in environment variable BIRDEYE_API_KEY or dontshare.py")
        exit(1) # Exit if no API key is found
else:
    logger.info("Using API key from environment variable")

# --- Helper Functions ---

def parse_date_range(start_str: str, end_str: str) -> Tuple[datetime, datetime]:
    """Parses date strings into timezone-aware UTC datetime objects."""
    try:
        # Assume dates are naive, treat as UTC start/end of day
        start_dt = datetime.strptime(start_str, "%m-%d-%Y").replace(tzinfo=timezone.utc)
        # End date is inclusive, so go to the end of that day
        end_dt = (datetime.strptime(end_str, "%m-%d-%Y") + timedelta(days=1) - timedelta(seconds=1)).replace(tzinfo=timezone.utc)
        return start_dt, end_dt
    except ValueError as e:
        logger.error(f"Error parsing dates: {e}. Please use MM-DD-YYYY format.")
        raise

def make_api_request(url: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Makes a GET request to the Birdeye API with basic error handling."""
    try:
        response = requests.get(url, headers=headers, timeout=30) # Add timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        logger.warning("Request timed out.")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - Status: {response.status_code}")
        logger.error(f"Response body: {response.text[:500]}...") # Log part of the error response
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred: {req_err}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response: {response.text[:500]}...")
    return None

def process_trade(trade: Dict[str, Any], start_dt: datetime, end_dt: datetime) -> Optional[Dict[str, Any]]:
    """Processes a single trade, calculating USD value and checking filters."""
    try:
        trade_time_ts = trade.get('blockUnixTime')
        if trade_time_ts is None:
            logger.warning("Trade missing 'blockUnixTime'. Skipping.")
            return None
        trade_time = datetime.fromtimestamp(trade_time_ts, tz=timezone.utc)

        # Check date range first (efficiency)
        if not (start_dt <= trade_time <= end_dt):
            # logger.debug(f"Trade time {trade_time} outside of date range {start_dt} - {end_dt}")
            return None # Signal to stop if sorting ascending, or just skip if descending

        quote = trade.get('quote', {})
        base = trade.get('base', {})
        owner = trade.get('owner')
        tx_hash = trade.get('txHash')

        if not all([quote, base, owner, tx_hash]):
             logger.warning(f"Trade missing essential fields (quote, base, owner, txHash) at {trade_time}. Skipping. Data: {trade}")
             return None

        # Calculate trade size
        ui_amount_str = quote.get('uiAmountString') # Use string for precision
        nearest_price_str = quote.get('nearestPrice')

        if ui_amount_str is None or nearest_price_str is None:
            logger.warning(f"Trade missing 'uiAmountString' or 'nearestPrice' in quote data at {trade_time}. Skipping. Quote: {quote}")
            return None

        try:
            ui_amount = float(ui_amount_str)
            nearest_price = float(nearest_price_str)
            trade_size_usd = abs(ui_amount * nearest_price) # Use absolute value
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating trade size for trade at {trade_time}: {e}. Quote data: {quote}. Skipping.")
            return None

        # logger.debug(f"Processing trade: {trade_time} | Size: ${trade_size_usd:.2f}")

        # Check trade size filter
        if not (CONFIG.MIN_TRADE_SIZE_USD <= trade_size_usd <= CONFIG.MAX_TRADE_SIZE_USD):
            # logger.debug(f"Trade size ${trade_size_usd:.2f} outside of filter range.")
            return None

        # If all checks pass, format the output
        owner_link = f"{CONFIG.EXPLORER_BASE_URL}{owner}"
        return {
            'Timestamp': trade_time.strftime('%Y-%m-%d %H:%M:%S %Z'), # Standard format
            'Owner': owner,
            'Owner Link': owner_link,
            'From Symbol': quote.get('symbol', 'Unknown'),
            'From Amount': ui_amount,
            'To Symbol': base.get('symbol', 'Unknown'),
            'To Amount': base.get('uiAmountString', 'Unknown'), # Use string
            'USD Value': trade_size_usd,
            'Tx Hash': tx_hash
        }

    except Exception as e:
        logger.exception(f"Unexpected error processing trade: {trade}. Error: {e}")
        return None


# --- Main Application Logic ---

def fetch_and_process_trades(token_address: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetches trades, processes them based on filters, and returns a DataFrame."""
    all_processed_trades: List[Dict[str, Any]] = []
    offset = 0
    total_trades_fetched = 0
    consecutive_errors = 0
    consecutive_empty_batches = 0

    headers = {"accept": "application/json", "X-API-KEY": API_KEY}

    logger.info(f"Starting trade fetch for {token_address}...")

    while offset <= CONFIG.MAX_OFFSET:
        url = (
            f"{CONFIG.API_BASE_URL}/defi/txs/token?address={token_address}"
            f"&offset={offset}&limit={CONFIG.FETCH_LIMIT}&tx_type=swap&sort_type={CONFIG.SORT_TYPE}"
        )
        logger.info(f"Fetching trades from offset {offset}...")

        data = make_api_request(url, headers)

        if not data:
            consecutive_errors += 1
            logger.warning(f"API request failed or returned no data. Error count: {consecutive_errors}/{CONFIG.MAX_CONSECUTIVE_ERRORS}")
            if consecutive_errors >= CONFIG.MAX_CONSECUTIVE_ERRORS:
                logger.error(f"Reached maximum consecutive errors ({CONFIG.MAX_CONSECUTIVE_ERRORS}). Stopping.")
                break
            time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            # Do not increment offset on error, retry the same offset implicitly on next loop
            continue
        else:
            consecutive_errors = 0 # Reset error count on success

        trades = data.get('data', {}).get('items', [])
        total_trades_fetched += len(trades)

        if not trades:
            consecutive_empty_batches += 1
            logger.info(f"No trades in this batch. Empty batch count: {consecutive_empty_batches}/{CONFIG.MAX_EMPTY_BATCHES}")
            if consecutive_empty_batches >= CONFIG.MAX_EMPTY_BATCHES:
                logger.info(f"Reached maximum consecutive empty batches ({CONFIG.MAX_EMPTY_BATCHES}). Assuming end of relevant data.")
                break
            # Still increment offset even if batch is empty, to move past potentially empty ranges
            offset += CONFIG.FETCH_LIMIT
            time.sleep(CONFIG.REQUEST_DELAY_SECONDS)
            continue
        else:
            consecutive_empty_batches = 0 # Reset empty count

        logger.info(f"Processing {len(trades)} trades from batch...")
        batch_processed_count = 0
        stop_processing = False
        for trade in trades:
            processed = process_trade(trade, start_dt, end_dt)
            if processed:
                all_processed_trades.append(processed)
                batch_processed_count += 1
            elif processed is None and CONFIG.SORT_TYPE == 'asc':
                # If process_trade returns None (e.g., out of date range) and we are sorting ascending,
                # we can potentially stop fetching earlier if we assume trades are ordered by time.
                # Check the time of the *first* trade in the batch for this optimization.
                try:
                    first_trade_time = datetime.fromtimestamp(trades[0].get('blockUnixTime'), tz=timezone.utc)
                    if first_trade_time > end_dt:
                        logger.info(f"First trade in batch ({first_trade_time}) is after end date ({end_dt}). Stopping fetch early (asc sort).")
                        stop_processing = True
                        break # Stop processing this batch
                except Exception:
                    pass # Ignore errors here, just won't optimize

        logger.info(f"Added {batch_processed_count} trades from this batch.")

        if stop_processing:
            break # Stop fetching more batches

        offset += CONFIG.FETCH_LIMIT
        if offset > CONFIG.MAX_OFFSET:
            logger.warning(f"Reached maximum offset limit ({CONFIG.MAX_OFFSET}). Stopping.")
            break

        time.sleep(CONFIG.REQUEST_DELAY_SECONDS) # Wait before next request

    logger.info(f"Finished fetching. Total trades fetched: {total_trades_fetched}. Total trades matching filters: {len(all_processed_trades)}.")
    return pd.DataFrame(all_processed_trades)


def save_trades(df: pd.DataFrame, output_path: Path):
    """Saves the DataFrame to a CSV file."""
    if not df.empty:
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved {len(df)} trades to {output_path}")
        except IOError as e:
            logger.error(f"Error saving trades to {output_path}: {e}")
    else:
        logger.info("No trades found matching the specified criteria. No CSV file created.")

# --- Main Execution ---

if __name__ == "__main__":
    start_timer = time.time()

    logger.info(f"--- Starting Early Buyer Search for {CONFIG.TOKEN_ADDRESS} ---")
    logger.info(f"Date Range: {CONFIG.START_DATE_STR} to {CONFIG.END_DATE_STR}")
    logger.info(f"Trade Size Filter (USD): ${CONFIG.MIN_TRADE_SIZE_USD:,.2f} - ${CONFIG.MAX_TRADE_SIZE_USD:,.2f}")
    logger.info(f"Output Directory: {CONFIG.OUTPUT_DIR.resolve()}")
    logger.info(f"Sort Order: {CONFIG.SORT_TYPE}")

    if not API_KEY:
        # Already logged error, just exit cleanly
        exit(1)

    try:
        start_datetime, end_datetime = parse_date_range(CONFIG.START_DATE_STR, CONFIG.END_DATE_STR)
        logger.info(f"Parsed UTC Date Range: {start_datetime} to {end_datetime}")

        trades_df = fetch_and_process_trades(CONFIG.TOKEN_ADDRESS, start_datetime, end_datetime)

        output_file = CONFIG.OUTPUT_DIR / f"{CONFIG.TOKEN_ADDRESS}.csv"
        save_trades(trades_df, output_file)

    except Exception as e:
        logger.exception("An unexpected error occurred during the main execution.")

    end_timer = time.time()
    duration = end_timer - start_timer
    logger.info(f"--- Script finished in {duration:.2f} seconds ---")