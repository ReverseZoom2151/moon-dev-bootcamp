"""
ML Indicator Evaluation System
===============================
Day 33: Machine learning-based indicator evaluation and feature importance analysis.

Trains ML models to predict price movements using technical indicators,
analyzes feature importance, and identifies the best performing indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import warnings

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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

warnings.filterwarnings('ignore')


class MLIndicatorEvaluator:
    """
    Evaluates technical indicators using machine learning models.
    
    Trains ML models to predict price movements and analyzes which
    indicators are most predictive.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        results_dir: str = './ml_results'
    ):
        """
        Initialize ML indicator evaluator.
        
        Args:
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.config = config or self._default_config()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model definitions
        self.models = self._get_models()
        
        logger.info(f"MLIndicatorEvaluator initialized with {len(self.models)} models")
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'target_column': 'Close',
            'target_shift': -1,  # Predict next period
            'test_size': 0.2,
            'shuffle_split': False,  # Time series data
            'calc_permutation_importance': False,
            'permutation_n_repeats': 5,
            'permutation_sample_size': 1000,
            'use_models': ['LinearRegression', 'RandomForestRegressor', 'XGBRegressor']
        }
    
    def _get_models(self) -> Dict[str, Any]:
        """Get configured ML models."""
        models = {}
        use_models = self.config.get('use_models', [])
        
        if 'LinearRegression' in use_models:
            models['LinearRegression'] = LinearRegression()
        
        if 'RandomForestRegressor' in use_models:
            models['RandomForestRegressor'] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        if 'XGBRegressor' in use_models and XGBOOST_AVAILABLE:
            models['XGBRegressor'] = XGBRegressor(
                n_estimators=100,
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )
        
        return models
    
    def calculate_indicators(
        self,
        df: pd.DataFrame,
        indicator_configs: Dict[str, List[Dict]]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculate technical indicators.
        
        Args:
            df: OHLCV DataFrame
            indicator_configs: Dictionary with 'pandas_ta' and 'talib' indicator lists
            
        Returns:
            Tuple of (DataFrame with indicators, list of feature names)
        """
        df_out = df.copy()
        features_added = []
        
        # Ensure lowercase columns for talib
        col_map = {col: col.lower() for col in df_out.columns}
        df_lower = df_out.rename(columns=col_map)
        
        # Pandas TA indicators
        if PANDAS_TA_AVAILABLE and 'pandas_ta' in indicator_configs:
            for indi_config in indicator_configs['pandas_ta']:
                try:
                    kind = indi_config.pop('kind')
                    result = df_out.ta(kind=kind, append=True, **indi_config)
                    if result is not None:
                        # Extract feature names
                        if hasattr(result, 'columns'):
                            features_added.extend([col for col in result.columns if col not in df_out.columns])
                    indi_config['kind'] = kind  # Restore for next iteration
                except Exception as e:
                    logger.warning(f"Could not calculate pandas_ta indicator '{kind}': {e}")
        
        # Talib indicators
        if TALIB_AVAILABLE and 'talib' in indicator_configs:
            required_cols = {'open', 'high', 'low', 'close', 'volume'}
            if required_cols.issubset(df_lower.columns):
                for indi_name in indicator_configs['talib']:
                    try:
                        func = getattr(talib, indi_name)
                        if 'volume' in str(func.__code__.co_varnames):
                            result = func(
                                df_lower['open'],
                                df_lower['high'],
                                df_lower['low'],
                                df_lower['close'],
                                df_lower['volume']
                            )
                        else:
                            result = func(
                                df_lower['open'],
                                df_lower['high'],
                                df_lower['low'],
                                df_lower['close']
                            )
                        
                        if isinstance(result, tuple):
                            for i, arr in enumerate(result):
                                col_name = f"{indi_name}_{i}"
                                df_out[col_name] = arr
                                features_added.append(col_name)
                        else:
                            df_out[indi_name] = result
                            features_added.append(indi_name)
                    except Exception as e:
                        logger.warning(f"Could not calculate talib indicator '{indi_name}': {e}")
        
        return df_out, features_added
    
    def prepare_target(
        self,
        df: pd.DataFrame,
        target_col: str,
        shift: int
    ) -> Tuple[pd.DataFrame, str]:
        """Prepare target variable by shifting."""
        target_name = f"{target_col}_future_{abs(shift)}"
        df[target_name] = df[target_col].shift(shift)
        
        # Remove rows with NaN target
        initial_rows = len(df)
        df = df.dropna(subset=[target_name])
        removed = initial_rows - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} rows due to target shifting")
        
        return df, target_name
    
    def get_feature_importance(
        self,
        model_name: str,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features: List[str]
    ) -> Optional[np.ndarray]:
        """Extract feature importance from model."""
        if model_name in ['RandomForestRegressor', 'XGBRegressor']:
            try:
                return model.feature_importances_
            except AttributeError:
                pass
        
        if model_name == 'LinearRegression':
            try:
                return np.abs(model.coef_)
            except AttributeError:
                pass
        
        if self.config.get('calc_permutation_importance', False):
            try:
                sample_size = min(
                    self.config.get('permutation_sample_size', 1000),
                    len(X_test)
                )
                if sample_size < len(X_test):
                    X_sample = X_test.sample(n=sample_size, random_state=42)
                    y_sample = y_test.loc[X_sample.index]
                else:
                    X_sample, y_sample = X_test, y_test
                
                perm_importance = permutation_importance(
                    model,
                    X_sample,
                    y_sample,
                    n_repeats=self.config.get('permutation_n_repeats', 5),
                    random_state=42,
                    scoring='r2',
                    n_jobs=-1
                )
                return perm_importance.importances_mean
            except Exception as e:
                logger.error(f"Error calculating permutation importance: {e}")
        
        return None
    
    def train_evaluate_model(
        self,
        name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features: List[str]
    ) -> Dict[str, Any]:
        """Train and evaluate a single model."""
        logger.info(f"Training {name}...")
        
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            importances = self.get_feature_importance(
                name, model, X_test, y_test, features
            )
            
            result = {
                'model': name,
                'mse': mse,
                'r2': r2,
                'features': features,
                'num_features': len(features)
            }
            
            if importances is not None:
                result['importances'] = importances.tolist()
            
            logger.info(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            return {
                'model': name,
                'error': str(e)
            }
    
    def evaluate_indicators(
        self,
        df: pd.DataFrame,
        indicator_configs: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Evaluate indicator combinations using ML models.
        
        Args:
            df: OHLCV DataFrame
            indicator_configs: Dictionary with indicator configurations
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting indicator evaluation...")
        
        # Calculate indicators
        df_with_indicators, features = self.calculate_indicators(df, indicator_configs)
        
        if not features:
            logger.warning("No features calculated")
            return {'error': 'No features calculated'}
        
        # Prepare target
        target_col = self.config.get('target_column', 'Close')
        shift = self.config.get('target_shift', -1)
        df_with_target, target_name = self.prepare_target(
            df_with_indicators,
            target_col,
            shift
        )
        
        # Prepare features and target
        feature_df = df_with_target[features].select_dtypes(include=[np.number])
        feature_df = feature_df.dropna()
        
        if feature_df.empty:
            logger.error("No valid features after processing")
            return {'error': 'No valid features'}
        
        # Align target with features
        target_series = df_with_target.loc[feature_df.index, target_name]
        
        # Split data
        test_size = self.config.get('test_size', 0.2)
        shuffle = self.config.get('shuffle_split', False)
        
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df,
            target_series,
            test_size=test_size,
            shuffle=shuffle
        )
        
        # Train and evaluate models
        results = []
        for name, model in self.models.items():
            result = self.train_evaluate_model(
                name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                list(feature_df.columns)
            )
            results.append(result)
        
        return {
            'results': results,
            'features': list(feature_df.columns),
            'num_features': len(feature_df.columns),
            'target': target_name
        }
    
    def save_results(self, evaluation_results: Dict[str, Any], filename: str = None):
        """Save evaluation results to files."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"evaluation_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Saved results to {filepath}")
        
        # Save performance CSV
        if 'results' in evaluation_results:
            perf_data = []
            for result in evaluation_results['results']:
                if 'error' not in result:
                    perf_data.append({
                        'Model': result['model'],
                        'MSE': result.get('mse', 0),
                        'R2': result.get('r2', 0),
                        'NumFeatures': result.get('num_features', 0)
                    })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                perf_file = self.results_dir / 'model_performance.csv'
                perf_df.to_csv(perf_file, index=False)
                logger.info(f"Saved performance data to {perf_file}")
        
        # Save feature importance CSV
        if 'results' in evaluation_results:
            importance_data = []
            for result in evaluation_results['results']:
                if 'importances' in result:
                    for feature, importance in zip(
                        result['features'],
                        result['importances']
                    ):
                        importance_data.append({
                            'Model': result['model'],
                            'Feature': feature,
                            'Importance': importance,
                            'MSE': result.get('mse', 0),
                            'R2': result.get('r2', 0)
                        })
            
            if importance_data:
                imp_df = pd.DataFrame(importance_data)
                imp_file = self.results_dir / 'feature_importance.csv'
                imp_df.to_csv(imp_file, index=False)
                logger.info(f"Saved importance data to {imp_file}")
        
        return filepath

