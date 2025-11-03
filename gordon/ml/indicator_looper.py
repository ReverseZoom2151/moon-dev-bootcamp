"""
ML Indicator Looping System
============================
Day 33: Loops through multiple indicator combinations to find best performers.
"""

import pandas as pd
import numpy as np
import logging
import random
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from gordon.ml.indicator_evaluator import MLIndicatorEvaluator
from gordon.ml.indicator_discovery import IndicatorDiscovery

logger = logging.getLogger(__name__)


class IndicatorLooper:
    """
    Loops through multiple indicator combinations to find best performers.
    
    Tests random combinations of indicators across multiple generations.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        results_dir: str = './ml_results'
    ):
        """
        Initialize indicator looper.
        
        Args:
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        self.config = config or self._default_config()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = MLIndicatorEvaluator(config, str(self.results_dir))
        self.discovery = IndicatorDiscovery(str(self.results_dir))
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'generations': 10,
            'indicators_per_lib': 3,
            'random_seed': 42
        }
    
    def load_indicator_lists(self) -> Dict[str, List[str]]:
        """Load available indicator lists."""
        # Try to load saved lists
        indicators = self.discovery.load_indicator_lists()
        
        # If not found, discover and save
        if not indicators:
            logger.info("Discovering indicators...")
            self.discovery.save_indicator_lists()
            indicators = self.discovery.load_indicator_lists()
        
        return indicators
    
    def select_random_indicators(
        self,
        pandas_ta_list: List[str],
        talib_list: List[str],
        num_per_lib: int = 3
    ) -> Dict[str, List[Dict]]:
        """Select random indicators from each library."""
        selected = {
            'pandas_ta': [],
            'talib': []
        }
        
        # Select pandas_ta indicators
        if pandas_ta_list:
            selected_pandas = random.sample(
                pandas_ta_list,
                min(num_per_lib, len(pandas_ta_list))
            )
            for ind_name in selected_pandas:
                selected['pandas_ta'].append({'kind': ind_name})
        
        # Select talib indicators
        if talib_list:
            selected_talib = random.sample(
                talib_list,
                min(num_per_lib, len(talib_list))
            )
            selected['talib'] = selected_talib
        
        return selected
    
    def run_generation(
        self,
        generation: int,
        df: pd.DataFrame,
        indicator_lists: Dict[str, List[str]]
    ) -> Dict[str, any]:
        """Run a single generation of indicator evaluation."""
        logger.info(f"Running generation {generation}...")
        
        # Select random indicators
        selected = self.select_random_indicators(
            indicator_lists.get('pandas_ta', []),
            indicator_lists.get('talib', []),
            self.config.get('indicators_per_lib', 3)
        )
        
        logger.info(f"Selected indicators: {selected}")
        
        # Evaluate indicators
        results = self.evaluator.evaluate_indicators(df, selected)
        
        # Add generation info
        results['generation'] = generation
        results['selected_indicators'] = selected
        
        return results
    
    def run_all_generations(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """
        Run multiple generations of indicator evaluation.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of generation results
        """
        logger.info("Starting indicator looping...")
        
        # Set random seed
        random.seed(self.config.get('random_seed', 42))
        
        # Load indicator lists
        indicator_lists = self.load_indicator_lists()
        
        generations = self.config.get('generations', 10)
        all_results = []
        
        for gen in range(1, generations + 1):
            try:
                result = self.run_generation(gen, df, indicator_lists)
                all_results.append(result)
                
                # Save incremental results
                self._save_incremental_results(result, gen)
                
            except Exception as e:
                logger.error(f"Error in generation {gen}: {e}")
        
        # Save final summary
        self._save_summary(all_results)
        
        logger.info(f"Completed {len(all_results)} generations")
        return all_results
    
    def _save_incremental_results(self, result: Dict, generation: int):
        """Save results incrementally."""
        if 'results' in result:
            # Append to performance CSV
            perf_data = []
            for model_result in result['results']:
                if 'error' not in model_result:
                    perf_data.append({
                        'Generation': generation,
                        'Model': model_result['model'],
                        'MSE': model_result.get('mse', 0),
                        'R2': model_result.get('r2', 0),
                        'NumFeatures': model_result.get('num_features', 0),
                        'Indicators': json.dumps(result.get('selected_indicators', {}))
                    })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                perf_file = self.results_dir / 'model_performance.csv'
                
                if perf_file.exists():
                    perf_df.to_csv(perf_file, mode='a', header=False, index=False)
                else:
                    perf_df.to_csv(perf_file, index=False)
                
                # Append to importance CSV
                importance_data = []
                for model_result in result['results']:
                    if 'importances' in model_result:
                        for feature, importance in zip(
                            model_result['features'],
                            model_result['importances']
                        ):
                            importance_data.append({
                                'Generation': generation,
                                'Model': model_result['model'],
                                'Feature': feature,
                                'Importance': importance,
                                'MSE': model_result.get('mse', 0),
                                'R2': model_result.get('r2', 0)
                            })
                
                if importance_data:
                    imp_df = pd.DataFrame(importance_data)
                    imp_file = self.results_dir / 'feature_importance.csv'
                    
                    if imp_file.exists():
                        imp_df.to_csv(imp_file, mode='a', header=False, index=False)
                    else:
                        imp_df.to_csv(imp_file, index=False)
    
    def _save_summary(self, all_results: List[Dict]):
        """Save summary of all generations."""
        summary = {
            'total_generations': len(all_results),
            'config': self.config,
            'best_results': []
        }
        
        # Find best results by R2
        best_by_model = {}
        for result in all_results:
            if 'results' in result:
                for model_result in result['results']:
                    if 'error' not in model_result:
                        model_name = model_result['model']
                        r2 = model_result.get('r2', 0)
                        
                        if model_name not in best_by_model or r2 > best_by_model[model_name]['r2']:
                            best_by_model[model_name] = {
                                **model_result,
                                'generation': result.get('generation', 0)
                            }
        
        summary['best_results'] = list(best_by_model.values())
        
        # Save summary
        summary_file = self.results_dir / 'looping_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved summary to {summary_file}")

