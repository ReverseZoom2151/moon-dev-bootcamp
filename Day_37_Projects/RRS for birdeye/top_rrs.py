import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict

# Local imports
from config import RESULTS_DIR

# Setup logger for this module
logger = logging.getLogger(__name__)


def process_results_files(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """Processes individual RRS result CSV files to find top performers across timeframes.

    Reads all CSV files (except specified exclusions) from the results directory,
    extracts the top 3 performers based on 'rrs' score from each,
    compiles them into a single DataFrame, sorts by RRS score, and saves
    the consolidated list to 'Top_RRS.csv'.

    Args:
        results_dir: The path to the directory containing the result CSV files.
                     Defaults to RESULTS_DIR from config.py.

    Returns:
        A pandas DataFrame containing the top performers across all processed files,
        sorted by RRS score (descending).
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame() # Return empty if directory doesn't exist

    all_top_performers: List[Dict] = []
    files_processed_count = 0
    excluded_files = ['historical_runs.csv', 'Top_RRS.csv'] # Files to skip

    logger.info(f"Processing RRS result files in: {results_dir}")

    # Process each CSV file in the results directory
    for file_path in results_path.glob('*.csv'):
        if file_path.name in excluded_files:
            logger.debug(f"Skipping excluded file: {file_path.name}")
            continue

        logger.debug(f"Processing file: {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            files_processed_count += 1

            # Check required columns
            if 'symbol' not in df.columns or 'rrs' not in df.columns:
                logger.warning(f"Skipping file {file_path.name}: Missing required columns 'symbol' or 'rrs'.")
                continue

            # Ensure rrs is numeric
            df['rrs'] = pd.to_numeric(df['rrs'], errors='coerce')
            df.dropna(subset=['rrs'], inplace=True)

            if df.empty:
                 logger.warning(f"No valid RRS data found in file {file_path.name} after cleaning. Skipping.")
                 continue

            # Sort by RRS and get top 3
            top_3 = df.nlargest(3, 'rrs')

            # Extract timeframe from filename (assuming format like '1H_90d_...')
            try:
                # Split by '_', take the first part as timeframe
                timeframe = file_path.stem.split('_')[0]
            except IndexError:
                logger.warning(f"Could not extract timeframe from filename {file_path.name}. Using 'Unknown'.")
                timeframe = 'Unknown'

            # Add file info to the results
            for _, row in top_3.iterrows():
                all_top_performers.append({
                    'timeframe': timeframe,
                    'symbol': row['symbol'],
                    'rrs': row['rrs'],
                    'source_file': file_path.name # Keep track of source
                })

        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty file: {file_path.name}")
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}", exc_info=True)

    if not all_top_performers:
        logger.warning(f"No top performers found after processing {files_processed_count} files.")
        return pd.DataFrame()

    # Create DataFrame from all results
    results_df = pd.DataFrame(all_top_performers)

    # Sort by RRS score (descending)
    results_df = results_df.sort_values('rrs', ascending=False).reset_index(drop=True)

    # Save to CSV
    output_file = results_path / 'Top_RRS.csv'
    try:
        results_df.to_csv(output_file, index=False)
        logger.info(f"🌟 Top RRS Scores Across All Timeframes ({len(results_df)} entries):")
        # Use logger.info for results, consider level based on verbosity needs
        # Use to_string for better formatting in logs compared to raw print
        logger.info('\n' + results_df.to_string())
        logger.info(f"💾 Top RRS results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save Top_RRS.csv to {output_file}: {e}", exc_info=True)

    return results_df

if __name__ == '__main__':
    # Configure basic logging when run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    process_results_files()
