# main.py
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# --- Local Imports --- #
import config
from data_fetcher import fetch_data
from data_processor import calculate_returns_and_volatility, calculate_volume_metrics
from rrs_calculator import calculate_rrs
# top_rrs is not typically used directly by the RRS calculation main script
# from top_rrs import process_results_files 

# --- Logging Setup --- #
log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
        # Optional: Add FileHandler for persistent logs
        # logging.FileHandler('hyperliquid_rrs_analyzer.log')
    ]
)

logger = logging.getLogger(__name__) # Get logger for this module

# --- Helper Functions (Similar to Birdeye version) --- #

def ensure_directories_exist():
    """Creates the data and results directories if they don't exist."""
    try:
        Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured data directory exists: {config.DATA_DIR}")
        logger.info(f"Ensured results directory exists: {config.RESULTS_DIR}")
    except OSError as e:
        logger.error(f"Error creating directories: {e}", exc_info=True)
        sys.exit(1)

def save_dataframe(df: pd.DataFrame, directory: str, filename: str):
    """Saves a DataFrame to a CSV file with error handling."""
    if df.empty:
        logger.debug(f"Skipping save for empty DataFrame: {filename}")
        return
    
    file_path = Path(directory) / filename
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"💾 Data saved successfully to: {file_path}")
    except OSError as e:
        logger.error(f"Error saving file {file_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving {file_path}: {e}", exc_info=True)

def fetch_and_process_symbol_data(
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime
) -> Optional[pd.DataFrame]:
    """Fetches and processes data for a single symbol."""
    logger.info(f"--- Processing symbol: {symbol} ---")
    df = fetch_data(symbol, timeframe, start_time, end_time)
    
    if df.empty:
        logger.warning(f"No data fetched for {symbol}. Skipping.")
        return None
    
    actual_start = df['timestamp'].min()
    actual_end = df['timestamp'].max()
    logger.debug(f"Symbol {symbol} actual data range: {actual_start} to {actual_end}")

    df = calculate_returns_and_volatility(df)
    df = calculate_volume_metrics(df)

    essential_cols = ['log_return', 'volatility', 'volume_ratio']
    initial_len = len(df)
    df.dropna(subset=essential_cols, inplace=True)
    if len(df) < initial_len:
        logger.debug(f"Dropped {initial_len - len(df)} rows with NaNs in essential columns for {symbol}")

    if df.empty:
        logger.warning(f"No valid processed data for {symbol} after cleaning. Skipping.")
        return None

    logger.debug(f"Finished processing for {symbol}. Shape: {df.shape}")
    return df

def update_historical_summary(run_summary: pd.DataFrame):
    """Appends the current run summary to the historical CSV file."""
    historical_file = Path(config.RESULTS_DIR) / "historical_runs.csv"
    try:
        if historical_file.exists():
            historical_df = pd.read_csv(historical_file)
            historical_df = pd.concat([historical_df, run_summary], ignore_index=True)
        else:
            historical_df = run_summary
        
        historical_df.to_csv(historical_file, index=False)
        logger.info(f"🌟 Historical Database updated! Total runs: {len(historical_df)}. Location: {historical_file}")
    except pd.errors.ParserError as e:
         logger.error(f"Error reading historical file {historical_file}. It might be corrupted: {e}")
    except OSError as e:
        logger.error(f"Error writing to historical file {historical_file}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while updating {historical_file}: {e}", exc_info=True)

# --- Main Processing Logic --- #

def process_timeframe(timeframe: str, lookback_days: int, benchmark_symbol: str):
    """Processes RRS calculation for a specific timeframe."""
    logger.info(f"===== Starting processing for timeframe: {timeframe} (Lookback: {lookback_days} days) =====" )
    
    # Validate benchmark symbol is in the list
    if benchmark_symbol not in config.SYMBOLS:
        logger.error(f"Benchmark symbol '{benchmark_symbol}' not found in config.SYMBOLS list. Add it or choose another.")
        return
    
    # Get symbols excluding the benchmark
    symbols_to_process = [s for s in config.SYMBOLS if s != benchmark_symbol]
    if not symbols_to_process:
        logger.warning("No symbols to process (excluding benchmark).")
        return

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    logger.info(f"Time range (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Benchmark Data --- #
    logger.info(f"--- Processing benchmark: {benchmark_symbol} ---")
    benchmark_df_processed = fetch_and_process_symbol_data(
        benchmark_symbol, timeframe, start_time, end_time
    )
    if benchmark_df_processed is None or benchmark_df_processed.empty:
        logger.error(f"Failed to fetch or process benchmark data for {benchmark_symbol}. Cannot proceed for timeframe {timeframe}.")
        return
    
    # Keep only necessary columns for benchmark
    benchmark_df_for_rrs = benchmark_df_processed[['timestamp', 'log_return']].copy()
    logger.debug(f"Processed benchmark data shape for RRS calc: {benchmark_df_for_rrs.shape}")
    
    # --- Symbol Processing --- #
    rrs_results_list: List[Dict] = []
    for symbol_name in symbols_to_process:
        symbol_df_processed = fetch_and_process_symbol_data(
            symbol_name, timeframe, start_time, end_time
        )
        if symbol_df_processed is None or symbol_df_processed.empty:
            continue # Skip if fetching or processing failed

        # Calculate RRS
        rrs_df = calculate_rrs(symbol_df_processed, benchmark_df_for_rrs)
        if rrs_df.empty:
            logger.warning(f"No RRS data calculated for {symbol_name}.")
            continue

        # Save processed data with RRS for the symbol
        save_dataframe(rrs_df, config.DATA_DIR, f"{symbol_name}_{timeframe}_{lookback_days}d_processed_rrs.csv")

        # Get the latest smoothed RRS value
        try:
            rrs_df = rrs_df.sort_values(by='timestamp')
            latest_rrs = rrs_df.iloc[-1]['smoothed_rrs']
            if pd.isna(latest_rrs):
                 logger.warning(f"Latest smoothed_rrs for {symbol_name} is NaN. Skipping ranking.")
                 continue
            rrs_results_list.append({'symbol': symbol_name, 'rrs': latest_rrs})
            logger.debug(f"Latest RRS for {symbol_name}: {latest_rrs:.4f}")
        except (IndexError, KeyError) as e:
            logger.error(f"Error getting latest RRS for {symbol_name}: {e}")
        except Exception as e:
             logger.error(f"Unexpected error getting latest RRS for {symbol_name}: {e}", exc_info=True)

    # --- Finalize Timeframe Results --- #
    if not rrs_results_list:
        logger.warning(f"No symbols had valid RRS scores for timeframe {timeframe}.")
        return

    final_rrs_df = pd.DataFrame(rrs_results_list)
    final_rrs_df.sort_values('rrs', ascending=False, inplace=True)
    logger.info(f"--- RRS Ranking for Timeframe: {timeframe} ---")
    logger.info('\n' + final_rrs_df.to_string())

    # Save current run results
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{timeframe}_{lookback_days}d_{benchmark_symbol}_{timestamp_str}.csv"
    save_dataframe(final_rrs_df, config.RESULTS_DIR, filename)

    # Prepare and update historical summary
    try:
        top_performer = final_rrs_df.iloc[0]
        bottom_performer = final_rrs_df.iloc[-1]
        run_summary_data = {
            'run_timestamp_utc': [datetime.utcnow()],
            'timeframe': [timeframe],
            'lookback_days': [lookback_days],
            'benchmark_symbol': [benchmark_symbol],
            'num_symbols_analyzed': [len(symbols_to_process)],
            'num_symbols_ranked': [len(final_rrs_df)],
            'top_performer_symbol': [top_performer['symbol']],
            'top_performer_rrs': [top_performer['rrs']],
            'bottom_performer_symbol': [bottom_performer['symbol']],
            'bottom_performer_rrs': [bottom_performer['rrs']]
        }
        run_summary_df = pd.DataFrame(run_summary_data)
        update_historical_summary(run_summary_df)

        # Log summary of current run
        logger.info(f"--- Run Summary for {timeframe} ---")
        logger.info(f"Benchmark: {benchmark_symbol}")
        logger.info(f"Symbols Analyzed: {len(symbols_to_process)}, Symbols Ranked: {len(final_rrs_df)}")
        logger.info(f"🏆 Top Performer: {top_performer['symbol']} (RRS: {top_performer['rrs']:.4f})")
        logger.info(f"📉 Bottom Performer: {bottom_performer['symbol']} (RRS: {bottom_performer['rrs']:.4f})")
    except IndexError:
        logger.error(f"Could not generate run summary for {timeframe} - likely no symbols were ranked.")
    except Exception as e:
        logger.error(f"Error generating or saving run summary for {timeframe}: {e}", exc_info=True)
        
    logger.info(f"===== Finished processing for timeframe: {timeframe} =====" )

def main():
    """Main function to orchestrate the RRS analysis across all timeframes."""
    logger.info("🚀 Starting Hyperliquid RRS Analyzer...")
    ensure_directories_exist() # Create data/results dirs if needed

    benchmark_symbol_name = 'BTC' # Standard benchmark
    lookback = config.LOOKBACK_DAYS
    timeframes_to_run = config.TIMEFRAMES

    logger.info(f"Configuration: Benchmark='{benchmark_symbol_name}', Lookback={lookback} days")
    logger.info(f"Timeframes to analyze: {timeframes_to_run}")
    logger.info(f"Number of symbols configured: {len(config.SYMBOLS)}")

    for tf in timeframes_to_run:
        process_timeframe(tf, lookback, benchmark_symbol_name)

    logger.info("🏁 Hyperliquid RRS Analyzer Finished.")
    # Note: Does not automatically generate the combined Top_RRS.csv like the Birdeye version.
    # Run top_rrs.py separately if needed.

if __name__ == '__main__':
    main()
