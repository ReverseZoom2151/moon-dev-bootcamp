"""
Indicator Ranking and Selection System
=======================================
Day 33: Analyzes feature importance results and ranks indicators by performance.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class IndicatorRanker:
    """
    Ranks and selects best indicators based on ML evaluation results.
    """
    
    def __init__(self, results_dir: str = './ml_results'):
        """
        Initialize indicator ranker.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
    
    def load_importance_data(self, filename: str = 'feature_importance.csv') -> Optional[pd.DataFrame]:
        """Load feature importance data."""
        filepath = self.results_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Importance file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            required_cols = {'Model', 'Feature', 'Importance', 'MSE', 'R2'}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                logger.error(f"Missing required columns: {missing}")
                return None
            
            logger.info(f"Loaded {len(df)} importance records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading importance data: {e}")
            return None
    
    def filter_overfit_features(
        self,
        df: pd.DataFrame,
        r2_threshold: float = 0.95
    ) -> pd.DataFrame:
        """Filter out features with suspiciously high R2 (potential overfitting)."""
        initial_count = len(df)
        filtered = df[df['R2'] <= r2_threshold].copy()
        removed = initial_count - len(filtered)
        
        if removed > 0:
            logger.info(f"Filtered out {removed} features with R2 > {r2_threshold}")
        
        return filtered
    
    def rank_by_importance(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:
        """Rank features by importance score."""
        if 'Importance' not in df.columns:
            logger.error("Importance column not found")
            return pd.DataFrame()
        
        # Group by feature and average importance across models
        feature_importance = df.groupby('Feature')['Importance'].agg(['mean', 'std', 'count']).reset_index()
        feature_importance.columns = ['Feature', 'AvgImportance', 'StdImportance', 'NumModels']
        
        # Sort by average importance
        ranked = feature_importance.sort_values('AvgImportance', ascending=False).head(top_n)
        
        return ranked
    
    def rank_by_mse(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:
        """Rank features by MSE (lower is better)."""
        if 'MSE' not in df.columns:
            logger.error("MSE column not found")
            return pd.DataFrame()
        
        # Group by feature and average MSE
        feature_mse = df.groupby('Feature')['MSE'].agg(['mean', 'std', 'count']).reset_index()
        feature_mse.columns = ['Feature', 'AvgMSE', 'StdMSE', 'NumModels']
        
        # Sort by average MSE (ascending - lower is better)
        ranked = feature_mse.sort_values('AvgMSE', ascending=True).head(top_n)
        
        return ranked
    
    def rank_by_r2(
        self,
        df: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:
        """Rank features by R2 score (higher is better)."""
        if 'R2' not in df.columns:
            logger.error("R2 column not found")
            return pd.DataFrame()
        
        # Group by feature and average R2
        feature_r2 = df.groupby('Feature')['R2'].agg(['mean', 'std', 'count']).reset_index()
        feature_r2.columns = ['Feature', 'AvgR2', 'StdR2', 'NumModels']
        
        # Sort by average R2 (descending - higher is better)
        ranked = feature_r2.sort_values('AvgR2', ascending=False).head(top_n)
        
        return ranked
    
    def get_top_indicators(
        self,
        r2_threshold: float = 0.95,
        top_n: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """
        Get top indicators ranked by different metrics.
        
        Args:
            r2_threshold: Filter threshold for R2 (to remove overfit)
            top_n: Number of top indicators to return
            
        Returns:
            Dictionary with rankings by different metrics
        """
        df = self.load_importance_data()
        if df is None:
            return {}
        
        # Filter overfit features
        filtered = self.filter_overfit_features(df, r2_threshold)
        
        if filtered.empty:
            logger.warning("No features remaining after filtering")
            return {}
        
        # Rank by different metrics
        rankings = {
            'importance': self.rank_by_importance(filtered, top_n),
            'mse': self.rank_by_mse(filtered, top_n),
            'r2': self.rank_by_r2(filtered, top_n)
        }
        
        # Save rankings
        for metric, ranking_df in rankings.items():
            if not ranking_df.empty:
                filename = f"top_{top_n}_{metric}.csv"
                filepath = self.results_dir / filename
                ranking_df.to_csv(filepath, index=False)
                logger.info(f"Saved {metric} ranking to {filepath}")
        
        return rankings
    
    def get_best_indicators_for_model(
        self,
        model_name: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """Get best indicators for a specific model."""
        df = self.load_importance_data()
        if df is None:
            return pd.DataFrame()
        
        # Filter by model
        model_df = df[df['Model'] == model_name].copy()
        
        if model_df.empty:
            logger.warning(f"No data found for model: {model_name}")
            return pd.DataFrame()
        
        # Filter overfit
        filtered = self.filter_overfit_features(model_df)
        
        # Sort by importance
        sorted_df = filtered.sort_values('Importance', ascending=False).head(top_n)
        
        return sorted_df[['Feature', 'Importance', 'MSE', 'R2']]

