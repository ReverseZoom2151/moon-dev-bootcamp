# main.py
import pandas as pd
from datetime import datetime, timedelta
import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# --- Local Imports --- #
# Assuming config.py is in the same directory
import config
from data_fetcher import fetch_data
from data_processor import calculate_returns_and_volatility, calculate_volume_metrics
from rrs_calculator import calculate_rrs
from top_rrs import process_results_files

# --- Logging Setup --- #
# Determine log level from environment variable or default to INFO
log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
        # Optional: Add FileHandler for persistent logs
        # logging.FileHandler('birdeye_rrs_analyzer.log')
    ]
)

logger = logging.getLogger(__name__) # Get logger for this module

# --- Helper Functions --- #

def ensure_directories_exist():
    """Creates the data and results directories if they don't exist."""
    try:
        Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured data directory exists: {config.DATA_DIR}")
        logger.info(f"Ensured results directory exists: {config.RESULTS_DIR}")
    except OSError as e:
        logger.error(f"Error creating directories: {e}", exc_info=True)
        sys.exit(1) # Exit if directories can't be created

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
    symbol_name: str,
    symbol_address: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime
) -> Optional[pd.DataFrame]:
    """Fetches and processes data for a single symbol."""
    logger.info(f"--- Processing symbol: {symbol_name} ({symbol_address}) ---")
    df = fetch_data(symbol_address, timeframe, start_time, end_time)
    
    if df.empty:
        logger.warning(f"No data fetched for {symbol_name}. Skipping.")
        return None
    
    # Log actual data range received
    actual_start = df['timestamp'].min()
    actual_end = df['timestamp'].max()
    logger.debug(f"Symbol {symbol_name} actual data range: {actual_start} to {actual_end}")

    # Process data: Calculate returns, volatility, and volume metrics
    df = calculate_returns_and_volatility(df)
    df = calculate_volume_metrics(df)

    # Drop rows with NaNs in essential columns created during processing
    essential_cols = ['log_return', 'volatility', 'volume_ratio']
    initial_len = len(df)
    df.dropna(subset=essential_cols, inplace=True)
    if len(df) < initial_len:
        logger.debug(f"Dropped {initial_len - len(df)} rows with NaNs in essential columns for {symbol_name}")

    if df.empty:
        logger.warning(f"No valid processed data (returns, vol, volume) for {symbol_name} after cleaning. Skipping.")
        return None

    logger.debug(f"Finished processing for {symbol_name}. Shape: {df.shape}")
    return df

def update_historical_summary(run_summary: pd.DataFrame):
    """Appends the current run summary to the historical CSV file."""
    historical_file = Path(config.RESULTS_DIR) / "historical_runs.csv"
    try:
        if historical_file.exists():
            historical_df = pd.read_csv(historical_file)
            # Use concat for potentially better performance and future compatibility
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
    
    # Validate benchmark symbol
    if benchmark_symbol not in config.SYMBOLS:
        logger.error(f"Benchmark symbol '{benchmark_symbol}' not found in config.SYMBOLS. Cannot proceed.")
        return
    benchmark_address = config.SYMBOLS[benchmark_symbol]
    
    # Get symbols excluding the benchmark
    symbols_to_process = {s: addr for s, addr in config.SYMBOLS.items() if s != benchmark_symbol}
    if not symbols_to_process:
        logger.warning("No symbols to process (excluding benchmark).")
        return

    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    logger.info(f"Time range (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch and process benchmark data
    logger.info(f"--- Processing benchmark: {benchmark_symbol} ({benchmark_address}) ---")
    benchmark_df_raw = fetch_data(benchmark_address, timeframe, start_time, end_time)
    
    if benchmark_df_raw.empty:
        logger.error(f"Failed to fetch benchmark data for {benchmark_symbol}. Cannot proceed with timeframe {timeframe}.")
        return
    
    # Save raw benchmark data (optional)
    save_dataframe(benchmark_df_raw, config.DATA_DIR, f"{benchmark_symbol}_{timeframe}_{lookback_days}d_raw.csv")

    benchmark_df_processed = calculate_returns_and_volatility(benchmark_df_raw)
    benchmark_df_processed = benchmark_df_processed[['timestamp', 'log_return']].dropna()
    
    if benchmark_df_processed.empty:
        logger.error(f"Benchmark data for {benchmark_symbol} became empty after processing returns. Cannot proceed.")
        return
    logger.debug(f"Processed benchmark data shape: {benchmark_df_processed.shape}")

    # Initialize list to store RRS results for this timeframe
    rrs_results_list: List[Dict] = []

    # Process each symbol
    for symbol_name, symbol_address in symbols_to_process.items():
        symbol_df_processed = fetch_and_process_symbol_data(
            symbol_name, symbol_address, timeframe, start_time, end_time
        )
        
        if symbol_df_processed is None or symbol_df_processed.empty:
            continue # Skip if fetching or basic processing failed

        # Calculate RRS
        rrs_df = calculate_rrs(symbol_df_processed, benchmark_df_processed)
        
        if rrs_df.empty:
            logger.warning(f"No RRS data calculated for {symbol_name} (likely no overlapping data with benchmark or calculation issue).")
            continue

        # Save the processed data with RRS values for the symbol
        save_dataframe(rrs_df, config.DATA_DIR, f"{symbol_name}_{timeframe}_{lookback_days}d_processed.csv")

        # Get the latest smoothed RRS value
        try:
            # Ensure data is sorted by timestamp if not already guaranteed
            rrs_df = rrs_df.sort_values(by='timestamp')
            latest_rrs = rrs_df.iloc[-1]['smoothed_rrs']
            if pd.isna(latest_rrs):
                 logger.warning(f"Latest smoothed_rrs for {symbol_name} is NaN. Skipping ranking.")
                 continue
            rrs_results_list.append({'symbol': symbol_name, 'rrs': latest_rrs})
            logger.debug(f"Latest RRS for {symbol_name}: {latest_rrs:.4f}")
        except (IndexError, KeyError) as e:
            logger.error(f"Error getting latest RRS for {symbol_name}: {e}")
            continue
        except Exception as e:
             logger.error(f"Unexpected error getting latest RRS for {symbol_name}: {e}", exc_info=True)
             continue

    # --- Finalize Timeframe Results --- #
    if not rrs_results_list:
        logger.warning(f"No symbols had valid RRS scores for timeframe {timeframe}.")
        return # Nothing to save or summarize for this timeframe

    # Rank tokens based on RRS
    final_rrs_df = pd.DataFrame(rrs_results_list)
    final_rrs_df.sort_values('rrs', ascending=False, inplace=True)
    logger.info(f"--- RRS Ranking for Timeframe: {timeframe} ---")
    logger.info('\n' + final_rrs_df.to_string())

    # Save current run results
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{timeframe}_{lookback_days}d_{benchmark_symbol}_{timestamp_str}.csv"
    save_dataframe(final_rrs_df, config.RESULTS_DIR, filename)

    # Prepare and update historical summary
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
    logger.info(f"===== Finished processing for timeframe: {timeframe} =====" )

def main():
    """Main function to orchestrate the RRS analysis across all timeframes."""
    logger.info("🚀 Starting Birdeye RRS Analyzer...")
    ensure_directories_exist() # Create data/results dirs if needed

    # --- Configuration --- #
    # Benchmark symbol name (must exist as a key in config.SYMBOLS)
    benchmark_symbol_name = 'SOL'
    lookback = config.LOOKBACK_DAYS
    timeframes_to_run = config.TIMEFRAMES

    logger.info(f"Configuration: Benchmark='{benchmark_symbol_name}', Lookback={lookback} days")
    logger.info(f"Timeframes to analyze: {timeframes_to_run}")
    logger.info(f"Number of symbols configured (incl. benchmark): {len(config.SYMBOLS)}")

    # --- Run Analysis for Each Timeframe --- #
    for tf in timeframes_to_run:
        process_timeframe(tf, lookback, benchmark_symbol_name)

    # --- Generate Consolidated Report --- #
    logger.info("===== Generating Final Top RRS Report across all timeframes =====")
    try:
        top_performers_all = process_results_files(config.RESULTS_DIR)

        if not top_performers_all.empty:
            # Log summary statistics from the consolidated report
            logger.info("--- Overall RRS Score Distribution (Top Performers) ---")
            logger.info(f"Max RRS: {top_performers_all['rrs'].max():.4f}")
            logger.info(f"Avg RRS: {top_performers_all['rrs'].mean():.4f}")
            logger.info(f"Min RRS: {top_performers_all['rrs'].min():.4f}")
            logger.info(f"Count: {len(top_performers_all)}")
        else:
            logger.warning("Consolidated Top RRS report is empty.")

    except Exception as e:
         logger.error(f"Failed to generate or process the final Top RRS report: {e}", exc_info=True)

    # --- Interpretation Guide --- #
    logger.info("--- RRS Score Interpretation Guide ---")
    logger.info(" > 2.0 : Exceptional Strength 🚀")
    logger.info("1.0-2.0: Strong Strength 💪")
    logger.info("0.5-1.0: Moderate Strength 📈")
    logger.info("0.0-0.5: Weak Strength 😐")
    logger.info(" < 0.0 : Underperforming 📉")
    logger.info("🏁 Birdeye RRS Analyzer Finished.")

if __name__ == '__main__':
    main()
