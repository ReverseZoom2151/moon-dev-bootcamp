#!/usr/bin/env python3
"""
Lists available technical indicators from pandas_ta and talib libraries
and saves them to JSON files.
"""

import os
import json
import logging
from typing import List, Dict, Any

# --- Configuration ---
# Get the absolute path of the directory where the script resides
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to the script directory
ml_results_dir = os.path.join(script_dir, "ml", "results")

CONFIG = {
    # --- Paths ---
    "RESULTS_DIR_PATH": ml_results_dir, # Absolute path
    # Filenames within RESULTS_DIR_PATH
    "PANDAS_TA_FILENAME": "pandas_ta_indicators.json",
    "TALIB_FILENAME": "talib_indicators.json"
}

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Functions ---

def list_pandas_ta_indicators() -> List[str]:
    """
    Retrieves all available indicator names from pandas_ta.
    Note: Relies on internal pandas_ta structure, might need update if library changes.
    """
    logger.info("Attempting to list indicators from pandas_ta...")
    try:
        import pandas_ta as ta
        # Access the categories dictionary (common way, but check pandas_ta docs for stability)
        # categories = ta.Category # This might be deprecated or internal
        # indicators = []
        # for category_name, indicator_list in categories.items():
        #     indicators.extend(indicator_list)

        # Safer approach: Use ta.indicators() if available and suits the need
        # Or inspect the ta module members
        indicators = [item for item in dir(ta) if callable(getattr(ta, item)) and not item.startswith('_') and item.islower()]
        # This might include non-indicator functions, needs careful filtering or use a known method.
        # Falling back to a known, potentially less stable method if the above is too broad:
        if not indicators or len(indicators) < 50: # Basic sanity check
            logger.warning("Using ta.Category lookup, potentially less stable.")
            indicators = list(ta.Category.keys()) # Often categories are also indicator names
            # This is a heuristic, pandas_ta structure varies.
            # A truly robust method might need parsing help(ta) or specific API calls.

        logger.info(f"Found {len(indicators)} potential pandas_ta indicators.")
        return sorted(list(set(indicators)))
    except ImportError:
        logger.error("pandas_ta library not found. Please install it.")
        return []
    except Exception as e:
        logger.error(f"Error listing pandas_ta indicators: {e}", exc_info=True)
        return []

def list_talib_indicators() -> List[str]:
    """
    Retrieves all available function names from the talib library.
    """
    logger.info("Listing indicators from talib...")
    try:
        import talib
        indicators = talib.get_functions()
        logger.info(f"Found {len(indicators)} talib indicators.")
        return sorted(indicators)
    except ImportError:
        logger.error("TA-Lib library not found. Please ensure it and its wrapper are installed correctly.")
        return []
    except Exception as e:
        logger.error(f"Error listing talib indicators: {e}", exc_info=True)
        return []

def save_to_json(data: List[str], filepath: str) -> None:
    """
    Saves the provided list data to a JSON file.
    Ensures the output directory exists.
    """
    if not data:
        logger.warning(f"No data provided to save to {filepath}. Skipping.")
        return
    try:
        output_dir = os.path.dirname(filepath)
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved {len(data)} items to {filepath}")
    except IOError as e:
        logger.error(f"Error writing JSON to {filepath}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error saving JSON to {filepath}: {e}", exc_info=True)

# --- Main Orchestration ---

def main(config: Dict[str, Any]) -> None:
    """Main function to list indicators and save them."""
    logger.info("=== Indicator Listing Script Started ===")

    results_dir = config["RESULTS_DIR_PATH"]
    pandas_ta_filepath = os.path.join(results_dir, config["PANDAS_TA_FILENAME"])
    talib_filepath = os.path.join(results_dir, config["TALIB_FILENAME"])

    # 1. List and Save pandas_ta Indicators
    pandas_ta_indicators = list_pandas_ta_indicators()
    save_to_json(pandas_ta_indicators, pandas_ta_filepath)

    # 2. List and Save talib Indicators
    talib_indicators = list_talib_indicators()
    save_to_json(talib_indicators, talib_filepath)

    logger.info("=== Indicator Listing Complete ===")

if __name__ == "__main__":
    main(CONFIG)
