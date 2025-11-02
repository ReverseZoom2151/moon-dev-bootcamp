# binance_rrs_main.py
import pandas as pd
import os
import logging
import sys
import binance_rrs_config as config
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from binance_rrs_data_fetcher import fetch_data, get_top_volume_symbols
from binance_rrs_data_processor import calculate_returns_and_volatility, calculate_volume_metrics, calculate_technical_indicators
from binance_rrs_calculator import calculate_rrs, rank_symbols_by_rrs, generate_trading_signals

# --- Logging Setup --- #
log_level_str = os.environ.get('LOG_LEVEL', config.LOG_LEVEL).upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('binance_rrs_analyzer.log')
    ]
)

logger = logging.getLogger(__name__)

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
        sys.exit(1)

def save_dataframe(df: pd.DataFrame, directory: str, filename: str):
    """Saves a DataFrame to CSV with error handling."""
    if df.empty:
        logger.warning(f"Empty DataFrame, not saving to {filename}")
        return
    
    try:
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")
        
        if config.EXPORT_TO_JSON:
            json_filepath = filepath.replace('.csv', '.json')
            df.to_json(json_filepath, orient='records', date_format='iso', indent=2)
            logger.info(f"Also saved JSON to {json_filepath}")
            
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {filename}: {e}")

def fetch_and_process_symbol_data(symbol_name: str, symbol_code: str, timeframe: str, 
                                start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
    """Fetches and processes data for a single symbol."""
    try:
        logger.info(f"📊 Processing {symbol_name} ({symbol_code}) - {timeframe}")
        
        # Fetch raw data
        raw_df = fetch_data(symbol_code, timeframe, start_time, end_time)
        if raw_df.empty:
            logger.warning(f"No data fetched for {symbol_name}")
            return None
        
        logger.info(f"Fetched {len(raw_df)} data points for {symbol_name}")
        
        # Process returns and volatility
        processed_df = calculate_returns_and_volatility(raw_df)
        
        # Calculate volume metrics
        processed_df = calculate_volume_metrics(processed_df)
        
        # Calculate technical indicators
        processed_df = calculate_technical_indicators(processed_df)
        
        # Save processed data if enabled
        if config.SAVE_RAW_DATA:
            filename = config.DATA_FILE_PATTERN.format(symbol=symbol_name, timeframe=timeframe)
            save_dataframe(processed_df, config.DATA_DIR, filename)
        
        logger.info(f"Successfully processed {symbol_name} with {len(processed_df.columns)} features")
        return processed_df
        
    except Exception as e:
        logger.error(f"Failed to process {symbol_name}: {e}", exc_info=True)
        return None

def update_historical_summary(run_summary: pd.DataFrame):
    """Appends current run summary to historical CSV."""
    historical_file = os.path.join(config.RESULTS_DIR, config.HISTORICAL_SUMMARY_FILE)
    
    try:
        if os.path.exists(historical_file):
            historical_df = pd.read_csv(historical_file)
            updated_df = pd.concat([historical_df, run_summary], ignore_index=True)
        else:
            updated_df = run_summary
        
        updated_df.to_csv(historical_file, index=False)
        logger.info(f"Updated historical summary with {len(run_summary)} new entries")
        
    except Exception as e:
        logger.error(f"Failed to update historical summary: {e}")

def process_timeframe(timeframe: str, lookback_days: int, benchmark_symbol: str = config.DEFAULT_BENCHMARK):
    """Processes RRS calculation for a specific timeframe."""
    logger.info(f"🎯 === Processing Timeframe: {timeframe} ({lookback_days} days) ===")
    
    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    
    logger.info(f"Time range: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    # Get benchmark data first
    benchmark_code = config.SYMBOLS.get(benchmark_symbol)
    if not benchmark_code:
        logger.error(f"Benchmark symbol {benchmark_symbol} not found in config")
        return
    
    logger.info(f"📈 Fetching benchmark data: {benchmark_symbol} ({benchmark_code})")
    benchmark_df = fetch_and_process_symbol_data(benchmark_symbol, benchmark_code, timeframe, start_time, end_time)
    
    if benchmark_df is None or benchmark_df.empty:
        logger.error(f"Failed to fetch benchmark data for {benchmark_symbol}")
        return
    
    # Determine symbols to analyze
    if config.SYMBOLS:
        # Use configured symbols
        symbols_to_analyze = [(name, code) for name, code in config.SYMBOLS.items() 
                            if name != benchmark_symbol]
        logger.info(f"Using {len(symbols_to_analyze)} configured symbols")
    else:
        # Auto-discover top volume symbols
        top_symbols = get_top_volume_symbols(50)
        symbols_to_analyze = [(symbol.replace('USDT', ''), symbol) for symbol in top_symbols 
                            if symbol != benchmark_code]
        logger.info(f"Auto-discovered {len(symbols_to_analyze)} top volume symbols")
    
    # Process symbols and calculate RRS
    rrs_results = {}
    successful_symbols = 0
    
    for symbol_name, symbol_code in symbols_to_analyze:
        try:
            # Fetch and process symbol data
            symbol_df = fetch_and_process_symbol_data(symbol_name, symbol_code, timeframe, start_time, end_time)
            
            if symbol_df is not None and not symbol_df.empty:
                # Calculate RRS
                rrs_df = calculate_rrs(symbol_df, benchmark_df, symbol_name, benchmark_symbol)
                
                if not rrs_df.empty:
                    rrs_results[symbol_name] = rrs_df
                    successful_symbols += 1
                    logger.info(f"✅ Completed RRS calculation for {symbol_name}")
                else:
                    logger.warning(f"⚠️ Empty RRS result for {symbol_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {symbol_name}: {e}")
            continue
    
    logger.info(f"📊 Successfully calculated RRS for {successful_symbols} symbols")
    
    if not rrs_results:
        logger.warning("No RRS results generated")
        return
    
    # Generate rankings
    logger.info("🏆 Generating symbol rankings...")
    rankings_df = rank_symbols_by_rrs(rrs_results, 'smoothed_rrs')
    
    if not rankings_df.empty:
        # Generate trading signals
        signals_df = generate_trading_signals(rankings_df)
        
        # Save results
        results_filename = config.RESULTS_FILE_PATTERN.format(timeframe=timeframe)
        save_dataframe(signals_df, config.RESULTS_DIR, results_filename)
        
        # Create summary
        summary_data = []
        for _, row in signals_df.head(config.TOP_N_SYMBOLS).iterrows():
            summary_data.append({
                'run_timestamp': datetime.utcnow(),
                'timeframe': timeframe,
                'symbol': row['symbol'],
                'rank': row['rank'],
                'rrs_score': row['current_rrs'],
                'signal': row['primary_signal'],
                'confidence': row['signal_confidence'],
                'risk_level': row['risk_level']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = config.SUMMARY_FILE_PATTERN.format(timeframe=timeframe)
        save_dataframe(summary_df, config.RESULTS_DIR, summary_filename)
        
        # Update historical summary
        update_historical_summary(summary_df)
        
        # Log top performers
        logger.info(f"🚀 Top 5 Performers ({timeframe}):")
        for i, (_, row) in enumerate(signals_df.head(5).iterrows(), 1):
            logger.info(f"  {i}. {row['symbol']}: RRS={row['current_rrs']:.3f} | Signal={row['primary_signal']} | Risk={row['risk_level']}")
        
        # Log signals summary
        signal_counts = signals_df['primary_signal'].value_counts()
        logger.info(f"📋 Signal Distribution: {dict(signal_counts)}")
    
    logger.info(f"✅ Completed {timeframe} analysis")

def main():
    """Main function to orchestrate Binance RRS analysis across all timeframes."""
    logger.info("🚀 Starting Binance RRS Analysis System")
    logger.info(f"📊 Configuration: {len(config.SYMBOLS)} symbols, {len(config.TIMEFRAMES)} timeframes")
    logger.info(f"🎯 Benchmark: {config.DEFAULT_BENCHMARK}")
    
    try:
        # Ensure directories exist
        ensure_directories_exist()
        
        # Process each timeframe
        for timeframe, lookback_days in config.TIMEFRAMES.items():
            try:
                process_timeframe(timeframe, lookback_days, config.DEFAULT_BENCHMARK)
            except Exception as e:
                logger.error(f"Failed to process timeframe {timeframe}: {e}", exc_info=True)
                continue
        
        logger.info("✅ Binance RRS Analysis completed successfully!")
        logger.info(f"📂 Results saved to: {config.RESULTS_DIR}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Analysis interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in main analysis: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
