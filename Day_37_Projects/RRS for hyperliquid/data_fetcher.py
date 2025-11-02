# data_fetcher.py
import requests
from datetime import datetime, timedelta
import pandas as pd
import logging
import time
from typing import List, Dict, Optional
import json # Added for JSONDecodeError handling

# Local imports
import config # Use config directly
from data_processor import interval_to_minutes

logger = logging.getLogger(__name__)

def parse_ohlcv_data(snapshot_data: List[Dict], symbol: str) -> pd.DataFrame:
    """Parses the OHLCV snapshot data list from Hyperliquid into a DataFrame."""
    if not snapshot_data:
        logger.warning(f"No snapshot data provided for parsing ({symbol}).")
        return pd.DataFrame()

    logger.debug(f"Parsing {len(snapshot_data)} candle snapshots for {symbol}")
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = []
    required_keys = {'t', 'o', 'h', 'l', 'c', 'v'}

    for snapshot in snapshot_data:
        if not required_keys.issubset(snapshot.keys()):
            logger.warning(f"Skipping incomplete snapshot item for {symbol}: {snapshot}")
            continue
        try:
            # Convert timestamp from ms to datetime object (UTC)
            timestamp = datetime.utcfromtimestamp(int(snapshot['t'] / 1000))
            data.append([
                timestamp,
                float(snapshot['o']),
                float(snapshot['h']),
                float(snapshot['l']),
                float(snapshot['c']),
                float(snapshot['v'])
            ])
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error processing snapshot item for {symbol}: {snapshot}. Error: {e}. Skipping item.")
            continue

    if not data:
         logger.warning(f"No valid snapshots could be processed for {symbol}.")
         return pd.DataFrame()

    df = pd.DataFrame(data, columns=columns)
    # Sort by timestamp just in case API doesn't guarantee order
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    logger.debug(f"Parsed DataFrame shape for {symbol}: {df.shape}")
    return df

def get_ohlcv_chunk(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict]]:
    """Fetches a single chunk of OHLCV data from the Hyperliquid API."""
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000)
        }
    }

    logger.debug(f"Requesting chunk for {symbol} ({interval})")
    logger.debug(f" Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.debug(f" Payload: {payload}")

    max_retries = 3
    retry_delay = 5 # seconds
    response = None # Initialize response to None
    for attempt in range(max_retries):
        try:
            response = requests.post(config.API_URL, headers=config.API_HEADERS, json=payload, timeout=20)
            logger.debug(f"Response Status Code: {response.status_code}")
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            # Check if response content type is JSON
            if 'application/json' not in response.headers.get('Content-Type', ''):
                logger.error(f"Unexpected content type received: {response.headers.get('Content-Type')}. Response text: {response.text[:200]}...")
                # Treat as failure, maybe retry or return None
                raise requests.exceptions.RequestException("Non-JSON response received")

            data = response.json()
            # Check if data is a list (expected format for candleSnapshot)
            if isinstance(data, list):
                 logger.debug(f"Received {len(data)} data points in chunk.")
                 return data
            else:
                # Handle unexpected JSON structure (e.g., error message)
                logger.error(f"Unexpected JSON structure received for {symbol}: {data}")
                return None # Or raise an exception

        except requests.exceptions.Timeout:
             logger.warning(f"Request timeout on attempt {attempt + 1}/{max_retries} for {symbol}. Retrying in {retry_delay}s...")
             time.sleep(retry_delay)
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed on attempt {attempt + 1}/{max_retries} for {symbol}: {e}")
            if response is not None:
                 logger.error(f"Response text: {response.text[:200]}...")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Max retries reached. Failed to fetch chunk.")
                return None # Failed after retries
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode JSON response for {symbol}: {e}. Response text: {response.text[:200] if response else 'N/A'}...")
             return None # JSON error is critical
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching chunk for {symbol}: {e}", exc_info=True)
            return None # Unexpected error
            
    return None # Should technically be unreachable if max_retries > 0

def fetch_data(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """Fetches historical OHLCV data for a symbol within a time range, handling pagination."""
    logger.info(f"Starting data fetch for {symbol} ({interval}) from {start_time} to {end_time}")
    all_dataframes: List[pd.DataFrame] = []
    
    # Calculate chunk size based on interval and API limit
    try:
        interval_minutes = interval_to_minutes(interval)
        if interval_minutes <= 0:
             raise ValueError("Interval minutes must be positive")
        # Calculate time delta for one chunk (max records * interval)
        chunk_delta = timedelta(minutes=interval_minutes * config.MAX_CALL_LIMIT)
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid interval format '{interval}' or MAX_CALL_LIMIT: {e}. Cannot determine chunk size.")
        return pd.DataFrame()

    current_chunk_end_time = end_time
    iteration = 0

    while current_chunk_end_time > start_time:
        iteration += 1
        # Calculate start time for the current chunk
        current_chunk_start_time = max(start_time, current_chunk_end_time - chunk_delta)
        logger.debug(f"Iteration {iteration}: Fetching chunk from {current_chunk_start_time} to {current_chunk_end_time}")
        
        # Fetch one chunk of data
        raw_data_chunk = get_ohlcv_chunk(symbol, interval, current_chunk_start_time, current_chunk_end_time)
        
        if raw_data_chunk is not None:
            df_chunk = parse_ohlcv_data(raw_data_chunk, symbol)
            if not df_chunk.empty:
                all_dataframes.append(df_chunk)
                logger.debug(f" Added chunk with {len(df_chunk)} rows. Total DFs: {len(all_dataframes)}")
            else:
                 logger.warning(f"Parsed chunk for {symbol} was empty. Raw data length: {len(raw_data_chunk)}")
        else:
            # If get_ohlcv_chunk returns None, it means fetching failed after retries
            logger.error(f"Failed to fetch data chunk for {symbol} for range {current_chunk_start_time} to {current_chunk_end_time}. Stopping fetch.")
            # Decide whether to return partial data or fail completely
            # break # Option 1: Stop fetching more chunks
            pass # Option 2: Continue trying older chunks (might be problematic)
            # For now, let's break to avoid potential infinite loops if API is down
            break 

        # Move to the next time window (before the current chunk's start)
        # Subtracting interval duration to avoid overlap issues if API includes start/end times differently
        current_chunk_end_time = current_chunk_start_time - timedelta(minutes=interval_minutes)
        
        # Small delay to be polite to the API
        time.sleep(0.5)

    if not all_dataframes:
        logger.warning(f"No data successfully fetched for {symbol} in the entire range.")
        return pd.DataFrame()

    # Concatenate all collected DataFrames
    try:
        full_df = pd.concat(all_dataframes)
        # Drop duplicates based on timestamp (essential after chunking)
        full_df = full_df.drop_duplicates(subset='timestamp')
        # Ensure final sort by timestamp
        full_df = full_df.sort_values(by='timestamp').reset_index(drop=True)
        logger.info(f"Successfully fetched and combined data for {symbol}. Final shape: {full_df.shape}")
        return full_df
    except Exception as e:
        logger.error(f"Error concatenating or processing final DataFrame for {symbol}: {e}", exc_info=True)
        return pd.DataFrame() # Return empty on final processing error
