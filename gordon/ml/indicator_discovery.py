"""
Indicator Discovery System
==========================
Day 33: Lists and catalogs available technical indicators from pandas_ta and talib.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import json

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class IndicatorDiscovery:
    """
    Discovers and catalogs available technical indicators.
    
    Lists indicators from pandas_ta and talib libraries.
    """
    
    def __init__(self, results_dir: str = './ml_results'):
        """
        Initialize indicator discovery.
        
        Args:
            results_dir: Directory to save indicator lists
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def list_pandas_ta_indicators(self) -> List[str]:
        """List available pandas_ta indicators."""
        if not PANDAS_TA_AVAILABLE:
            logger.warning("pandas_ta not available")
            return []
        
        try:
            # Get all callable attributes that are indicators
            indicators = [
                item for item in dir(ta)
                if callable(getattr(ta, item, None))
                and not item.startswith('_')
                and item.islower()
            ]
            
            # Filter out non-indicator functions
            known_non_indicators = {'utils', 'config', 'version', 'help'}
            indicators = [ind for ind in indicators if ind not in known_non_indicators]
            
            logger.info(f"Found {len(indicators)} pandas_ta indicators")
            return sorted(list(set(indicators)))
            
        except Exception as e:
            logger.error(f"Error listing pandas_ta indicators: {e}")
            return []
    
    def list_talib_indicators(self) -> List[str]:
        """List available talib indicators."""
        if not TALIB_AVAILABLE:
            logger.warning("talib not available")
            return []
        
        try:
            indicators = talib.get_functions()
            logger.info(f"Found {len(indicators)} talib indicators")
            return sorted(indicators)
            
        except Exception as e:
            logger.error(f"Error listing talib indicators: {e}")
            return []
    
    def save_indicator_lists(self) -> Dict[str, Path]:
        """Save indicator lists to JSON files."""
        saved_files = {}
        
        # Pandas TA
        pandas_ta_indicators = self.list_pandas_ta_indicators()
        if pandas_ta_indicators:
            pandas_ta_file = self.results_dir / 'pandas_ta_indicators.json'
            with open(pandas_ta_file, 'w') as f:
                json.dump(pandas_ta_indicators, f, indent=2)
            saved_files['pandas_ta'] = pandas_ta_file
            logger.info(f"Saved {len(pandas_ta_indicators)} pandas_ta indicators")
        
        # Talib
        talib_indicators = self.list_talib_indicators()
        if talib_indicators:
            talib_file = self.results_dir / 'talib_indicators.json'
            with open(talib_file, 'w') as f:
                json.dump(talib_indicators, f, indent=2)
            saved_files['talib'] = talib_file
            logger.info(f"Saved {len(talib_indicators)} talib indicators")
        
        return saved_files
    
    def load_indicator_lists(self) -> Dict[str, List[str]]:
        """Load indicator lists from JSON files."""
        indicators = {}
        
        pandas_ta_file = self.results_dir / 'pandas_ta_indicators.json'
        if pandas_ta_file.exists():
            try:
                with open(pandas_ta_file, 'r') as f:
                    indicators['pandas_ta'] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading pandas_ta indicators: {e}")
        
        talib_file = self.results_dir / 'talib_indicators.json'
        if talib_file.exists():
            try:
                with open(talib_file, 'r') as f:
                    indicators['talib'] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading talib indicators: {e}")
        
        return indicators

