import os
import json
import logging
from typing import List, Dict, Any
from backend.core.config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndicatorListingService:
    def __init__(self):
        self.results_dir = config.get('RESULTS_DIR_PATH', './ml/results')
        self.pandas_ta_filepath = os.path.join(self.results_dir, config.get('PANDAS_TA_FILENAME', 'pandas_ta_indicators.json'))
        self.talib_filepath = os.path.join(self.results_dir, config.get('TALIB_FILENAME', 'talib_indicators.json'))
        os.makedirs(self.results_dir, exist_ok=True)

    def list_pandas_ta_indicators(self) -> List[str]:
        """
        Retrieves all available indicator names from pandas_ta.
        Note: Relies on internal pandas_ta structure, might need update if library changes.
        """
        logger.info("Attempting to list indicators from pandas_ta...")
        try:
            import pandas_ta as ta
            indicators = [item for item in dir(ta) if callable(getattr(ta, item)) and not item.startswith('_') and item.islower()]
            if not indicators or len(indicators) < 50:  # Basic sanity check
                logger.warning("Using ta.Category lookup, potentially less stable.")
                indicators = list(ta.Category.keys())
            logger.info(f"Found {len(indicators)} potential pandas_ta indicators.")
            return sorted(list(set(indicators)))
        except ImportError:
            logger.error("pandas_ta library not found. Please install it.")
            return []
        except Exception as e:
            logger.error(f"Error listing pandas_ta indicators: {e}", exc_info=True)
            return []

    def list_talib_indicators(self) -> List[str]:
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

    def save_to_json(self, data: List[str], filepath: str) -> None:
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

    async def list_and_save_indicators(self) -> None:
        """Main function to list indicators and save them."""
        logger.info("=== Indicator Listing Service Started ===")

        # List and Save pandas_ta Indicators
        pandas_ta_indicators = self.list_pandas_ta_indicators()
        self.save_to_json(pandas_ta_indicators, self.pandas_ta_filepath)

        # List and Save talib Indicators
        talib_indicators = self.list_talib_indicators()
        self.save_to_json(talib_indicators, self.talib_filepath)

        logger.info("=== Indicator Listing Complete ===") 