# data_fetcher.py
import requests
from datetime import datetime
import pandas as pd
import logging
from typing import Dict, List, Optional
import json

# Local imports
from config import BIRDEYE_API_KEY

# Setup logger for this module
logger = logging.getLogger(__name__)

def fetch_data(address: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Fetches OHLCV data from the Birdeye API for a given token address and time range.

    Args:
        address: The token mint address on Solana.
        timeframe: The timeframe string (e.g., '1m', '5m', '1H', '1D').
        start_time: The start datetime object (UTC).
        end_time: The end datetime object (UTC).

    Returns:
        A pandas DataFrame containing the OHLCV data, or an empty DataFrame if fetching fails
        or no data is available.
    """
    if not BIRDEYE_API_KEY:
        logger.error("Birdeye API key is not configured. Cannot fetch data.")
        return pd.DataFrame()

    time_from_ts = int(start_time.timestamp())
    time_to_ts = int(end_time.timestamp())

    url = f"https://public-api.birdeye.so/defi/ohlcv?address={address}&type={timeframe}&time_from={time_from_ts}&time_to={time_to_ts}"
    headers = {"X-API-KEY": BIRDEYE_API_KEY}

    logger.info(f"Fetching Birdeye data for {address} [{timeframe}]")
    logger.debug(f"Request URL: {url}")
    logger.debug(f"Time range (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f"Time range (Unix): {time_from_ts} to {time_to_ts}")

    try:
        response = requests.get(url, headers=headers, timeout=30) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        json_response = response.json()
        items: Optional[List[Dict]] = json_response.get('data', {}).get('items')

        if not items:
            logger.warning(f"No data items received from Birdeye for {address} in the specified range.")
            return pd.DataFrame()

        # Process received data
        processed_data = []
        required_keys = {'unixTime', 'o', 'h', 'l', 'c', 'v'}
        for item in items:
            if not required_keys.issubset(item.keys()):
                logger.warning(f"Skipping incomplete data item: {item}")
                continue
            # Ensure numeric types are correct, handle potential None or string values
            try:
                processed_data.append({
                    'timestamp': datetime.utcfromtimestamp(int(item['unixTime'])),
                    'open': float(item['o']),
                    'high': float(item['h']),
                    'low': float(item['l']),
                    'close': float(item['c']),
                    'volume': float(item['v'])
                })
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing data item {item}: {e}. Skipping item.")
                continue

        if not processed_data:
            logger.warning(f"No valid data items could be processed for {address}.")
            return pd.DataFrame()

        df = pd.DataFrame(processed_data)
        df = df.sort_values(by='timestamp').reset_index(drop=True)

        # Padding: Ensure minimum length for indicator calculations (e.g., rolling windows)
        # Some indicators require a certain number of data points to compute.
        # Here, we ensure at least 40 rows by duplicating the earliest available data.
        # Consider the implications of this padding method based on your analysis needs.
        min_rows_required = 40 # Example minimum for certain TA calculations
        if 0 < len(df) < min_rows_required:
            rows_to_add = min_rows_required - len(df)
            logger.debug(f"Padding data for {address}: Have {len(df)}, need {min_rows_required}. Adding {rows_to_add} rows.")
            # Use pd.concat instead of list multiplication for clarity
            first_row_df = df.iloc[0:1]
            padding_df = pd.concat([first_row_df] * rows_to_add, ignore_index=True)
            df = pd.concat([padding_df, df], ignore_index=True)

        logger.info(f"Successfully retrieved and processed {len(df)} rows of data for {address}.")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed for {address}: {e}")
        # Consider adding retries with backoff here for production use
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response for {address}: {e}. Response text: {response.text[:200]}...")
    except Exception as e:
        logger.error(f"An unexpected error occurred during data fetching for {address}: {e}", exc_info=True)

    return pd.DataFrame() # Return empty DataFrame on failure
 