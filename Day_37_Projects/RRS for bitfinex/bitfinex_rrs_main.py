# bitfinex_rrs_main.py
import pandas as pd
import os
import logging
import sys
import bitfinex_rrs_config as config
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from bitfinex_rrs_data_fetcher import fetch_data, get_professional_market_overview
from bitfinex_rrs_data_processor import (calculate_professional_returns_and_volatility, 
                                        calculate_professional_volume_metrics, 
                                        calculate_professional_technical_indicators,
                                        calculate_professional_market_regime)
from bitfinex_rrs_calculator import (calculate_professional_rrs, 
                                    rank_symbols_by_professional_rrs, 
                                    generate_professional_trading_signals)

# --- Professional Logging Setup --- #
log_level_str = os.environ.get('LOG_LEVEL', config.LOG_LEVEL).upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bitfinex_professional_rrs_analyzer.log')
    ]
)

logger = logging.getLogger(__name__)

# --- Professional Helper Functions --- #

def ensure_professional_directories_exist():
    """Creates the professional data and results directories if they don't exist."""
    try:
        Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured professional data directory exists: {config.DATA_DIR}")
        logger.info(f"Ensured professional results directory exists: {config.RESULTS_DIR}")
    except OSError as e:
        logger.error(f"Error creating professional directories: {e}", exc_info=True)
        sys.exit(1)

def save_professional_dataframe(df: pd.DataFrame, directory: str, filename: str):
    """Saves a DataFrame to CSV with professional error handling."""
    if df.empty:
        logger.warning(f"Empty professional DataFrame, not saving to {filename}")
        return
    
    try:
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} professional records to {filepath}")
        
        if config.EXPORT_TO_JSON:
            json_filepath = filepath.replace('.csv', '.json')
            df.to_json(json_filepath, orient='records', date_format='iso', indent=2)
            logger.info(f"Also saved professional JSON to {json_filepath}")
            
        if config.EXPORT_TO_EXCEL:
            excel_filepath = filepath.replace('.csv', '.xlsx')
            df.to_excel(excel_filepath, index=False, sheet_name='Professional_RRS')
            logger.info(f"Also saved professional Excel to {excel_filepath}")
            
    except Exception as e:
        logger.error(f"Failed to save professional DataFrame to {filename}: {e}")

def fetch_and_process_professional_symbol_data(symbol_name: str, symbol_code: str, timeframe: str, 
                                             start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
    """Fetches and processes data for a single symbol using professional methods."""
    try:
        logger.info(f"📊 Processing professional data for {symbol_name} ({symbol_code}) - {timeframe}")
        
        # Fetch raw professional data
        raw_df = fetch_data(symbol_code, timeframe, start_time, end_time)
        if raw_df.empty:
            logger.warning(f"No professional data fetched for {symbol_name}")
            return None
        
        logger.info(f"Fetched {len(raw_df)} professional data points for {symbol_name}")
        
        # Professional processing pipeline
        processed_df = calculate_professional_returns_and_volatility(raw_df)
        processed_df = calculate_professional_volume_metrics(processed_df)
        processed_df = calculate_professional_technical_indicators(processed_df)
        processed_df = calculate_professional_market_regime(processed_df)
        
        # Save professional processed data if enabled
        if config.SAVE_RAW_DATA:
            filename = config.DATA_FILE_PATTERN.format(symbol=symbol_name, timeframe=timeframe)
            save_professional_dataframe(processed_df, config.DATA_DIR, filename)
        
        logger.info(f"Successfully processed professional data for {symbol_name} with {len(processed_df.columns)} features")
        return processed_df
        
    except Exception as e:
        logger.error(f"Failed to process professional data for {symbol_name}: {e}", exc_info=True)
        return None

def update_professional_historical_summary(run_summary: pd.DataFrame):
    """Appends current run summary to professional historical CSV."""
    historical_file = os.path.join(config.RESULTS_DIR, config.HISTORICAL_SUMMARY_FILE)
    
    try:
        if os.path.exists(historical_file):
            historical_df = pd.read_csv(historical_file)
            updated_df = pd.concat([historical_df, run_summary], ignore_index=True)
        else:
            updated_df = run_summary
        
        updated_df.to_csv(historical_file, index=False)
        logger.info(f"Updated professional historical summary with {len(run_summary)} new entries")
        
    except Exception as e:
        logger.error(f"Failed to update professional historical summary: {e}")

def process_professional_timeframe(timeframe: str, lookback_days: int, benchmark_symbol: str = config.DEFAULT_BENCHMARK):
    """Processes professional RRS calculation for a specific timeframe."""
    logger.info(f"🎯 === Processing Professional Timeframe: {timeframe} ({lookback_days} days) ===")
    
    # Calculate time range
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    
    logger.info(f"Professional time range: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    # Get professional benchmark data first
    benchmark_code = config.SYMBOLS.get(benchmark_symbol)
    if not benchmark_code:
        logger.error(f"Professional benchmark symbol {benchmark_symbol} not found in config")
        return
    
    logger.info(f"📈 Fetching professional benchmark data: {benchmark_symbol} ({benchmark_code})")
    benchmark_df = fetch_and_process_professional_symbol_data(benchmark_symbol, benchmark_code, 
                                                             timeframe, start_time, end_time)
    
    if benchmark_df is None or benchmark_df.empty:
        logger.error(f"Failed to fetch professional benchmark data for {benchmark_symbol}")
        return
    
    # Determine symbols for professional analysis
    if config.SYMBOLS:
        # Use configured professional symbols
        symbols_to_analyze = [(name, code) for name, code in config.SYMBOLS.items() 
                            if name != benchmark_symbol]
        logger.info(f"Using {len(symbols_to_analyze)} configured professional symbols")
    else:
        # Auto-discover top professional volume symbols
        top_symbols = get_professional_market_overview(50)
        symbols_to_analyze = [(symbol_data['symbol'].replace('usd', ''), symbol_data['symbol']) 
                            for symbol_data in top_symbols 
                            if symbol_data['symbol'] != benchmark_code.lower()]
        logger.info(f"Auto-discovered {len(symbols_to_analyze)} top professional volume symbols")
    
    # Process professional symbols and calculate RRS
    professional_rrs_results = {}
    successful_symbols = 0
    
    for symbol_name, symbol_code in symbols_to_analyze:
        try:
            # Fetch and process professional symbol data
            symbol_df = fetch_and_process_professional_symbol_data(symbol_name, symbol_code, 
                                                                 timeframe, start_time, end_time)
            
            if symbol_df is not None and not symbol_df.empty:
                # Calculate professional RRS
                rrs_df = calculate_professional_rrs(symbol_df, benchmark_df, symbol_name, benchmark_symbol)
                
                if not rrs_df.empty:
                    professional_rrs_results[symbol_name] = rrs_df
                    successful_symbols += 1
                    logger.info(f"✅ Completed professional RRS calculation for {symbol_name}")
                else:
                    logger.warning(f"⚠️ Empty professional RRS result for {symbol_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process professional symbol {symbol_name}: {e}")
            continue
    
    logger.info(f"📊 Successfully calculated professional RRS for {successful_symbols} symbols")
    
    if not professional_rrs_results:
        logger.warning("No professional RRS results generated")
        return
    
    # Generate professional rankings
    logger.info("🏆 Generating professional symbol rankings...")
    rankings_df = rank_symbols_by_professional_rrs(professional_rrs_results, 'smoothed_rrs')
    
    if not rankings_df.empty:
        # Generate professional trading signals
        signals_df = generate_professional_trading_signals(rankings_df)
        
        # Save professional results
        results_filename = config.RESULTS_FILE_PATTERN.format(timeframe=timeframe)
        save_professional_dataframe(signals_df, config.RESULTS_DIR, results_filename)
        
        # Create professional summary
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
                'risk_level': row['risk_level'],
                'professional_grade': row.get('professional_grade', 'Standard'),
                'position_size_suggestion': row.get('position_size_suggestion', 1.0),
                'beta': row.get('beta', 1.0),
                'alpha': row.get('alpha', 0.0),
                'information_ratio': row.get('information_ratio', 0.0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = config.SUMMARY_FILE_PATTERN.format(timeframe=timeframe)
        save_professional_dataframe(summary_df, config.RESULTS_DIR, summary_filename)
        
        # Update professional historical summary
        update_professional_historical_summary(summary_df)
        
        # Log top professional performers
        logger.info(f"🚀 Top 5 Professional Performers ({timeframe}):")
        for i, (_, row) in enumerate(signals_df.head(5).iterrows(), 1):
            logger.info(f"  {i}. {row['symbol']}: RRS={row['current_rrs']:.3f} | "
                       f"Signal={row['primary_signal']} | Grade={row.get('professional_grade', 'Standard')} | "
                       f"Risk={row['risk_level']}")
        
        # Log professional signals summary
        signal_counts = signals_df['primary_signal'].value_counts()
        grade_counts = signals_df.get('professional_grade', pd.Series()).value_counts()
        logger.info(f"📋 Professional Signal Distribution: {dict(signal_counts)}")
        logger.info(f"🏛️ Professional Grade Distribution: {dict(grade_counts)}")
    
    logger.info(f"✅ Completed professional {timeframe} analysis")

def generate_professional_market_report():
    """Generate comprehensive professional market analysis report."""
    try:
        logger.info("📈 Generating professional market analysis report...")
        
        # Get professional market overview
        market_data = get_professional_market_overview(20)
        if not market_data:
            logger.warning("No professional market data available for report")
            return
        
        # Create professional market summary
        market_df = pd.DataFrame(market_data)
        market_summary = {
            'report_timestamp': datetime.utcnow(),
            'total_professional_pairs': len(market_data),
            'avg_volume_usd': market_df['volume_usd'].mean() if 'volume_usd' in market_df.columns else 0,
            'top_performer': market_df.iloc[0]['symbol'] if len(market_df) > 0 else 'N/A',
            'market_sentiment': 'Bullish' if market_df['daily_change_perc'].mean() > 0 else 'Bearish'
        }
        
        # Save professional market report
        report_df = pd.DataFrame([market_summary])
        save_professional_dataframe(report_df, config.RESULTS_DIR, 'bitfinex_market_report.csv')
        
        logger.info("✅ Professional market report generated")
        
    except Exception as e:
        logger.error(f"Failed to generate professional market report: {e}")

def main():
    """Main function to orchestrate professional Bitfinex RRS analysis across all timeframes."""
    logger.info("🚀 Starting Bitfinex Professional RRS Analysis System")
    logger.info(f"🏛️ Professional Configuration: {len(config.SYMBOLS)} symbols, {len(config.TIMEFRAMES)} timeframes")
    logger.info(f"🎯 Professional Benchmark: {config.DEFAULT_BENCHMARK}")
    logger.info(f"🔧 Professional Features: Margin Analysis, Risk Management, Institutional Metrics")
    
    try:
        # Ensure professional directories exist
        ensure_professional_directories_exist()
        
        # Generate professional market report
        generate_professional_market_report()
        
        # Process each professional timeframe
        for timeframe, lookback_days in config.TIMEFRAMES.items():
            try:
                process_professional_timeframe(timeframe, lookback_days, config.DEFAULT_BENCHMARK)
            except Exception as e:
                logger.error(f"Failed to process professional timeframe {timeframe}: {e}", exc_info=True)
                continue
        
        logger.info("✅ Bitfinex Professional RRS Analysis completed successfully!")
        logger.info(f"🏛️ Professional results saved to: {config.RESULTS_DIR}")
        logger.info("📊 Professional features included: Beta, Alpha, Information Ratio, Risk Metrics")
        
    except KeyboardInterrupt:
        logger.info("🛑 Professional analysis interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in professional main analysis: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
