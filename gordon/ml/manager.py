"""
ML Integration Module
=====================
Day 33: Integration of ML indicator evaluation into Gordon's backtesting system.
"""

import logging
from typing import Dict, Optional
import pandas as pd

from .indicator_evaluator import MLIndicatorEvaluator
from .indicator_looper import IndicatorLooper
from .indicator_ranker import IndicatorRanker
from .indicator_discovery import IndicatorDiscovery

logger = logging.getLogger(__name__)


class MLIndicatorManager:
    """
    Manager for ML-based indicator evaluation and selection.
    
    Provides high-level interface for discovering, evaluating, and ranking indicators.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ML indicator manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        ml_config = self.config.get('ml', {})
        
        eval_config = ml_config.get('indicator_evaluation', {})
        self.evaluator = MLIndicatorEvaluator(eval_config, eval_config.get('results_dir', './ml_results'))
        
        loop_config = ml_config.get('indicator_looping', {})
        self.looper = IndicatorLooper(loop_config, loop_config.get('results_dir', './ml_results'))
        
        rank_config = ml_config.get('indicator_ranking', {})
        self.ranker = IndicatorRanker(rank_config.get('results_dir', './ml_results'))
        
        self.discovery = IndicatorDiscovery(eval_config.get('results_dir', './ml_results'))
    
    def discover_indicators(self) -> Dict[str, list]:
        """Discover and save available indicators."""
        logger.info("Discovering available indicators...")
        saved_files = self.discovery.save_indicator_lists()
        return self.discovery.load_indicator_lists()
    
    def evaluate_indicator_set(
        self,
        df: pd.DataFrame,
        pandas_ta_indicators: list = None,
        talib_indicators: list = None
    ) -> Dict:
        """
        Evaluate a specific set of indicators.
        
        Args:
            df: OHLCV DataFrame
            pandas_ta_indicators: List of pandas_ta indicator configs
            talib_indicators: List of talib indicator names
            
        Returns:
            Evaluation results
        """
        indicator_configs = {}
        
        if pandas_ta_indicators:
            indicator_configs['pandas_ta'] = pandas_ta_indicators
        
        if talib_indicators:
            indicator_configs['talib'] = talib_indicators
        
        return self.evaluator.evaluate_indicators(df, indicator_configs)
    
    def run_indicator_looping(self, df: pd.DataFrame) -> list:
        """
        Run multiple generations of indicator evaluation.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of generation results
        """
        return self.looper.run_all_generations(df)
    
    def get_top_indicators(self, top_n: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Get top indicators ranked by different metrics.
        
        Args:
            top_n: Number of top indicators to return
            
        Returns:
            Dictionary with rankings by importance, MSE, and R2
        """
        return self.ranker.get_top_indicators(top_n=top_n)
    
    def get_best_indicators_for_model(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get best indicators for a specific model.
        
        Args:
            model_name: Model name (e.g., 'RandomForestRegressor')
            top_n: Number of top indicators
            
        Returns:
            DataFrame with top indicators
        """
        return self.ranker.get_best_indicators_for_model(model_name, top_n)

