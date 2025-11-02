# binance_top_rrs.py
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from binance_rrs_config import RESULTS_DIR

# Setup logger for this module
logger = logging.getLogger(__name__)

def process_binance_results_files(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """Processes individual Binance RRS result CSV files to find top performers across timeframes.

    Reads all CSV files (except specified exclusions) from the Binance results directory,
    extracts the top performers based on RRS score from each timeframe analysis,
    compiles them into a single DataFrame, sorts by RRS score, and saves
    the consolidated list to 'Binance_Top_RRS.csv'.

    Args:
        results_dir: The path to the directory containing the Binance result CSV files.
                     Defaults to RESULTS_DIR from binance_rrs_config.py.

    Returns:
        A pandas DataFrame containing the top performers across all processed files,
        sorted by RRS score (descending).
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        logger.error(f"Binance results directory not found: {results_dir}")
        return pd.DataFrame()  # Return empty if directory doesn't exist

    all_top_performers: List[Dict] = []
    files_processed_count = 0
    excluded_files = ['binance_historical_summary.csv', 'Binance_Top_RRS.csv']  # Files to skip

    logger.info(f"Processing Binance RRS result files in: {results_dir}")

    # Process each CSV file in the results directory
    for file_path in results_path.glob('*.csv'):
        if file_path.name in excluded_files:
            logger.debug(f"Skipping excluded file: {file_path.name}")
            continue

        # Only process Binance-specific result files
        if not file_path.name.startswith('binance_'):
            logger.debug(f"Skipping non-Binance file: {file_path.name}")
            continue

        logger.debug(f"Processing Binance file: {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            files_processed_count += 1

            # Check if required columns exist
            required_columns = ['symbol', 'current_rrs', 'primary_signal', 'signal_confidence']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns {missing_columns} in {file_path.name}")
                continue

            # Handle different possible RRS column names
            rrs_column = None
            possible_rrs_columns = ['current_rrs', 'rrs_score', 'rrs', 'smoothed_rrs']
            for col in possible_rrs_columns:
                if col in df.columns:
                    rrs_column = col
                    break
            
            if rrs_column is None:
                logger.warning(f"No RRS score column found in {file_path.name}")
                continue

            # Sort by RRS score and get top 5 performers from this file
            df_sorted = df.sort_values(by=rrs_column, ascending=False)
            top_performers = df_sorted.head(5)

            # Extract timeframe from filename (e.g., binance_rrs_results_1h.csv -> 1h)
            timeframe = "unknown"
            if "_results_" in file_path.name:
                try:
                    timeframe = file_path.name.split("_results_")[1].replace(".csv", "")
                except IndexError:
                    timeframe = "unknown"

            # Add top performers to the list
            for _, row in top_performers.iterrows():
                performer_data = {
                    'symbol': row['symbol'],
                    'timeframe': timeframe,
                    'rrs_score': row[rrs_column],
                    'signal': row.get('primary_signal', 'N/A'),
                    'confidence': row.get('signal_confidence', 0),
                    'risk_level': row.get('risk_level', 'Medium'),
                    'rank': row.get('rank', 0),
                    'source_file': file_path.name,
                    'last_price': row.get('last_price', 0),
                    'volume_rank': row.get('volume_rank', 0),
                    'volatility': row.get('volatility', 0),
                    'momentum_score': row.get('momentum_score', 0),
                    'trend_strength': row.get('trend_strength', 0)
                }
                all_top_performers.append(performer_data)

            logger.info(f"Extracted {len(top_performers)} top performers from {file_path.name}")

        except pd.errors.EmptyDataError:
            logger.warning(f"Empty file encountered: {file_path.name}")
        except pd.errors.ParserError as e:
            logger.error(f"Parse error in file {file_path.name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path.name}: {e}")

    logger.info(f"Processed {files_processed_count} Binance files")

    if not all_top_performers:
        logger.warning("No top performers found across all processed files")
        return pd.DataFrame()

    # Create consolidated DataFrame
    consolidated_df = pd.DataFrame(all_top_performers)
    
    # Sort by RRS score (descending) and reset index
    consolidated_df = consolidated_df.sort_values(by='rrs_score', ascending=False).reset_index(drop=True)
    
    # Add overall ranking
    consolidated_df['overall_rank'] = range(1, len(consolidated_df) + 1)
    
    # Calculate additional metrics
    consolidated_df['exchange'] = 'Binance'
    consolidated_df['analysis_timestamp'] = datetime.utcnow()
    
    # Save to CSV
    output_file = results_path / 'Binance_Top_RRS.csv'
    try:
        consolidated_df.to_csv(output_file, index=False)
        logger.info(f"Saved consolidated Binance top performers to: {output_file}")
        logger.info(f"Total top performers: {len(consolidated_df)}")
        
        # Log summary statistics
        if len(consolidated_df) > 0:
            logger.info("=== BINANCE TOP RRS SUMMARY ===")
            logger.info(f"Best Overall RRS Score: {consolidated_df['rrs_score'].max():.4f}")
            logger.info(f"Worst RRS Score: {consolidated_df['rrs_score'].min():.4f}")
            logger.info(f"Average RRS Score: {consolidated_df['rrs_score'].mean():.4f}")
            
            # Signal distribution
            signal_counts = consolidated_df['signal'].value_counts()
            logger.info(f"Signal Distribution: {dict(signal_counts)}")
            
            # Top 10 overall performers
            logger.info("TOP 10 BINANCE PERFORMERS:")
            for i, (_, row) in enumerate(consolidated_df.head(10).iterrows(), 1):
                logger.info(f"  {i:2d}. {row['symbol']:8s} | {row['timeframe']:3s} | RRS: {row['rrs_score']:6.3f} | Signal: {row['signal']:4s} | Risk: {row['risk_level']}")
                
    except Exception as e:
        logger.error(f"Failed to save consolidated results: {e}")

    return consolidated_df

def analyze_binance_performance_trends(df: pd.DataFrame) -> Dict:
    """Analyzes performance trends across Binance timeframes and symbols."""
    if df.empty:
        return {}
    
    analysis = {}
    
    try:
        # Timeframe performance analysis
        timeframe_stats = df.groupby('timeframe')['rrs_score'].agg(['count', 'mean', 'max', 'min']).round(4)
        analysis['timeframe_performance'] = timeframe_stats.to_dict('index')
        
        # Symbol frequency analysis (which symbols appear most often in top lists)
        symbol_frequency = df['symbol'].value_counts()
        analysis['most_consistent_performers'] = symbol_frequency.head(10).to_dict()
        
        # Signal strength analysis
        signal_performance = df.groupby('signal')['rrs_score'].agg(['count', 'mean']).round(4)
        analysis['signal_effectiveness'] = signal_performance.to_dict('index')
        
        # Risk level distribution
        risk_distribution = df['risk_level'].value_counts().to_dict()
        analysis['risk_distribution'] = risk_distribution
        
        logger.info("Binance performance trends analysis completed")
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {e}")
    
    return analysis

def export_binance_top_performers_json(df: pd.DataFrame, results_dir: str = RESULTS_DIR):
    """Exports top Binance performers to JSON format for API consumption."""
    if df.empty:
        return
    
    try:
        # Create JSON-friendly format
        json_data = {
            'metadata': {
                'exchange': 'Binance',
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'total_performers': len(df),
                'data_source': 'Binance RRS Analysis System'
            },
            'top_performers': df.head(25).to_dict('records'),  # Top 25 for JSON
            'performance_trends': analyze_binance_performance_trends(df)
        }
        
        output_file = Path(results_dir) / 'Binance_Top_RRS.json'
        
        import json
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"Exported Binance top performers JSON to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to export JSON: {e}")

def main():
    """Main function to process Binance RRS results and generate top performers list."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('binance_top_rrs.log')
        ]
    )
    
    logger.info("🚀 Starting Binance Top RRS Analysis")
    
    try:
        # Process results files
        top_performers_df = process_binance_results_files()
        
        if not top_performers_df.empty:
            # Export to JSON as well
            export_binance_top_performers_json(top_performers_df)
            
            logger.info("✅ Binance Top RRS analysis completed successfully")
            return top_performers_df
        else:
            logger.warning("⚠️ No top performers found")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ Fatal error in Binance top RRS analysis: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
