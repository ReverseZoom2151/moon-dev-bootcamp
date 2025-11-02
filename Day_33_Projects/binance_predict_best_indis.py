#!/usr/bin/env python3
"""
Binance Best Indicators Prediction Model

Trains and evaluates ML models to predict next period's closing price using
the top-performing technical indicators identified from Binance analysis.
Focuses on spot trading performance with direction accuracy optimization.
"""

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import time
import json
import warnings
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

try:
    from Day_26_Projects.binance_config import PRIMARY_SYMBOL
except ImportError:
    PRIMARY_SYMBOL = "BTCUSDT"

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, os.pardir, "Day_31_Projects", "binance_data")
ml_results_dir = os.path.join(script_dir, "binance_ml", "results")

CONFIG = {
    # --- Paths ---
    "DATA_PATH": os.path.join(data_dir, f'{PRIMARY_SYMBOL}_historical_data.csv'),
    "BACKUP_DATA_PATH": os.path.join(data_dir, 'BTCUSDT_1h_historical.csv'),
    "RESULTS_DIR_PATH": ml_results_dir,
    "TOP_FEATURES_DIR": ml_results_dir,  # Where top indicator files are stored
    
    # Output files
    "PERFORMANCE_FILENAME": "binance_best_model_performance.csv",
    "PREDICTIONS_FILENAME": "binance_best_predictions.csv",
    "ANALYSIS_FILENAME": "binance_prediction_analysis.json",
    
    # --- Exchange Specific ---
    "EXCHANGE": "Binance",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "TRADING_FOCUS": "spot_trading",
    
    # --- Data & Features ---
    "TARGET_COLUMN": "close",
    "TARGET_SHIFT": -1,  # Predict next period
    
    # --- Model Training ---
    "TEST_SIZE": 0.25,
    "VALIDATION_SIZE": 0.15,
    "SHUFFLE_SPLIT": False,  # Time series
    
    # Enhanced models for Binance
    "MODELS": {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=150, 
            max_depth=10,
            min_samples_split=5,
            random_state=42, 
            n_jobs=-1
        ),
        'GradientBoostingRegressor': GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'XGBRegressor': XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
    },
    
    # --- Feature Analysis ---
    "USE_TOP_FEATURES": True,
    "MAX_FEATURES": 25,  # Limit for focused analysis
    "FEATURE_SELECTION_METHOD": "importance",  # or "r2" or "direction_accuracy"
    
    # --- Analysis ---
    "CALC_PERMUTATION_IMPORTANCE": True,
    "PERMUTATION_N_REPEATS": 3,
    "CREATE_PREDICTIONS_PLOT": True,
    "CREATE_IMPORTANCE_PLOT": True,
    "PLOT_SAMPLE_SIZE": 500,
}

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [Binance Prediction] %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(filepath: str, backup_path: str = None) -> Optional[pd.DataFrame]:
    """Load Binance data with enhanced validation."""
    paths = [filepath]
    if backup_path:
        paths.append(backup_path)
    
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                # Handle datetime
                for col in ['datetime', 'timestamp', 'time']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                        break
                
                # Validate columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    logger.info(f"✅ Loaded Binance data: {df.shape}")
                    logger.info(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
                    return df
                    
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
    
    logger.error("❌ No valid Binance data found")
    return None

def load_top_features(results_dir: str, method: str = "importance") -> Optional[List[str]]:
    """Load top features from analysis results."""
    
    # Try to find top features files
    feature_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith("binance_") and method in filename and filename.endswith(".csv"):
            feature_files.append(filename)
    
    if not feature_files:
        logger.warning(f"⚠️  No top features files found with method '{method}'")
        logger.info("💡 Please run binance_top_indis_from_loop.py first")
        return None
    
    all_features = set()
    
    for file in feature_files:
        try:
            filepath = os.path.join(results_dir, file)
            df = pd.read_csv(filepath)
            
            if 'Feature' in df.columns:
                # Get top features from this file
                top_features = df['Feature'].head(15).tolist()  # Top 15 from each model
                all_features.update(top_features)
                logger.info(f"📊 Loaded {len(top_features)} features from {file}")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not load features from {file}: {e}")
    
    final_features = list(all_features)
    logger.info(f"✅ Combined {len(final_features)} unique top features")
    
    return final_features

def calculate_best_indicators(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate the best-performing indicators."""
    
    logger.info(f"🔧 Calculating {len(feature_list)} top indicators...")
    df_out = df.copy()
    features_added = []
    
    # Parse and calculate indicators from feature names
    for feature_name in feature_list:
        try:
            # Parse feature name to understand indicator type
            if feature_name.startswith('ta_'):
                # pandas_ta indicator
                indicator_base = feature_name.replace('ta_', '').split('_')[0]
                
                if indicator_base == 'rsi':
                    # Extract period if specified
                    if 'rsi_' in feature_name and len(feature_name.split('_')) > 2:
                        period = int(feature_name.split('_')[2])
                        result = ta.rsi(df_out['close'], length=period)
                    else:
                        result = ta.rsi(df_out['close'])
                    
                    if isinstance(result, pd.Series) and not result.empty:
                        df_out[feature_name] = result
                        features_added.append(feature_name)
                
                elif indicator_base == 'ema':
                    if 'ema_' in feature_name and len(feature_name.split('_')) > 2:
                        period = int(feature_name.split('_')[2])
                        result = ta.ema(df_out['close'], length=period)
                    else:
                        result = ta.ema(df_out['close'])
                    
                    if isinstance(result, pd.Series) and not result.empty:
                        df_out[feature_name] = result
                        features_added.append(feature_name)
                
                elif indicator_base == 'sma':
                    if 'sma_' in feature_name and len(feature_name.split('_')) > 2:
                        period = int(feature_name.split('_')[2])
                        result = ta.sma(df_out['close'], length=period)
                    else:
                        result = ta.sma(df_out['close'])
                    
                    if isinstance(result, pd.Series) and not result.empty:
                        df_out[feature_name] = result
                        features_added.append(feature_name)
                
                elif indicator_base == 'macd':
                    result = ta.macd(df_out['close'])
                    if isinstance(result, pd.DataFrame):
                        # Map specific MACD components
                        if 'macd_12' in feature_name.lower():
                            df_out[feature_name] = result.iloc[:, 0]  # MACD line
                            features_added.append(feature_name)
                
                elif hasattr(ta, indicator_base):
                    # Generic pandas_ta indicator
                    func = getattr(ta, indicator_base)
                    result = func(df_out['close'])
                    if isinstance(result, pd.Series) and not result.empty:
                        df_out[feature_name] = result
                        features_added.append(feature_name)
            
            elif feature_name.startswith('talib_'):
                # TA-Lib indicator
                indicator_name = feature_name.replace('talib_', '').upper().split('_')[0]
                
                if hasattr(talib_abstract, indicator_name):
                    inputs = {
                        'open': df_out['open'],
                        'high': df_out['high'],
                        'low': df_out['low'],
                        'close': df_out['close'],
                        'volume': df_out['volume']
                    }
                    
                    func = talib_abstract.Function(indicator_name)
                    
                    # Handle period specification
                    if '_' in feature_name and feature_name.split('_')[-1].isdigit():
                        period = int(feature_name.split('_')[-1])
                        result = func(inputs, timeperiod=period)
                    else:
                        result = func(inputs)
                    
                    if isinstance(result, pd.Series) and not result.empty:
                        df_out[feature_name] = result
                        features_added.append(feature_name)
        
        except Exception as e:
            logger.warning(f"⚠️  Could not calculate {feature_name}: {e}")
    
    logger.info(f"✅ Successfully calculated {len(features_added)} indicators")
    return df_out, features_added

def prepare_features_target(df: pd.DataFrame, feature_cols: List[str], target_col: str, shift: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for Binance prediction."""
    
    logger.info("🎯 Preparing Binance prediction dataset...")
    df_out = df.copy()
    
    # Create target
    target_name = f"{target_col}_future_{abs(shift)}"
    df_out[target_name] = df_out[target_col].shift(shift)
    
    # Select columns
    cols_needed = feature_cols + [target_name]
    missing_cols = [col for col in cols_needed if col not in df_out.columns]
    
    if missing_cols:
        logger.warning(f"⚠️  Missing features: {missing_cols}")
        # Remove missing features
        feature_cols = [f for f in feature_cols if f in df_out.columns]
        cols_needed = feature_cols + [target_name]
    
    df_prepared = df_out[cols_needed].copy()
    
    # Clean data
    df_prepared.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_rows = len(df_prepared)
    df_prepared.dropna(inplace=True)
    
    logger.info(f"📊 Data preparation: {initial_rows} → {len(df_prepared)} rows")
    
    if df_prepared.empty:
        raise ValueError("No data remaining after cleaning")
    
    X = df_prepared[feature_cols]
    y = df_prepared[target_name]
    
    return X, y

def get_enhanced_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate enhanced metrics for trading."""
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
    # Trading-specific metrics
    try:
        # Direction accuracy
        actual_returns = y_true.pct_change().dropna()
        pred_returns = pd.Series(y_pred).pct_change().dropna()
        
        if len(actual_returns) == len(pred_returns):
            direction_accuracy = (np.sign(actual_returns) == np.sign(pred_returns)).mean()
            metrics['direction_accuracy'] = direction_accuracy
        
        # Profit potential (simplified)
        price_changes = y_true.diff().dropna()
        pred_changes = pd.Series(y_pred).diff().dropna()
        
        if len(price_changes) == len(pred_changes):
            correct_direction = (np.sign(price_changes) == np.sign(pred_changes))
            profit_potential = np.abs(price_changes[correct_direction]).sum()
            total_potential = np.abs(price_changes).sum()
            metrics['profit_capture'] = profit_potential / total_potential if total_potential > 0 else 0
        
    except Exception as e:
        logger.debug(f"Error calculating trading metrics: {e}")
        metrics['direction_accuracy'] = np.nan
        metrics['profit_capture'] = np.nan
    
    return metrics

def train_evaluate_model(name: str, model: Any, X_train, y_train, X_val, y_val, X_test, y_test, config: Dict) -> Dict:
    """Train and evaluate model with comprehensive metrics."""
    
    logger.info(f"🤖 Training {name}...")
    
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_metrics = get_enhanced_metrics(y_train, y_train_pred)
        val_metrics = get_enhanced_metrics(y_val, y_val_pred)
        test_metrics = get_enhanced_metrics(y_test, y_test_pred)
        
        logger.info(f"✅ {name}: Test R²={test_metrics['r2']:.4f}, Direction={test_metrics.get('direction_accuracy', 0):.3f}")
        
        # Feature importance
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        elif config.get("CALC_PERMUTATION_IMPORTANCE"):
            try:
                perm_imp = permutation_importance(model, X_test, y_test, n_repeats=config["PERMUTATION_N_REPEATS"], random_state=42)
                importance = perm_imp.importances_mean
            except:
                pass
        
        return {
            'model_name': name,
            'train_time': train_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'predictions': {
                'y_test': y_test.values,
                'y_pred': y_test_pred
            },
            'feature_importance': importance,
            'feature_names': X_test.columns.tolist()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to train {name}: {e}")
        return None

def create_plots(results: List[Dict], config: Dict) -> None:
    """Create visualization plots."""
    
    if not config.get("CREATE_PREDICTIONS_PLOT") and not config.get("CREATE_IMPORTANCE_PLOT"):
        return
    
    results_dir = config["RESULTS_DIR_PATH"]
    os.makedirs(results_dir, exist_ok=True)
    
    # Find best model
    best_model = None
    best_r2 = -np.inf
    
    for result in results:
        if result and result['test_metrics']['r2'] > best_r2:
            best_r2 = result['test_metrics']['r2']
            best_model = result
    
    if not best_model:
        return
    
    try:
        # Predictions plot
        if config.get("CREATE_PREDICTIONS_PLOT"):
            plt.figure(figsize=(12, 6))
            
            y_test = best_model['predictions']['y_test']
            y_pred = best_model['predictions']['y_pred']
            
            # Limit sample size for plot
            sample_size = min(len(y_test), config.get("PLOT_SAMPLE_SIZE", 500))
            indices = np.random.choice(len(y_test), sample_size, replace=False)
            
            plt.scatter(y_test[indices], y_pred[indices], alpha=0.6, s=20)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title(f'Binance {best_model["model_name"]} - Actual vs Predicted\nR² = {best_model["test_metrics"]["r2"]:.4f}')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(results_dir, f"binance_predictions_{best_model['model_name']}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Saved predictions plot: {plot_path}")
        
        # Feature importance plot
        if config.get("CREATE_IMPORTANCE_PLOT") and best_model['feature_importance'] is not None:
            plt.figure(figsize=(10, 8))
            
            importance = best_model['feature_importance']
            feature_names = best_model['feature_names']
            
            # Sort by importance
            sorted_indices = np.argsort(importance)[-15:]  # Top 15
            
            plt.barh(range(len(sorted_indices)), importance[sorted_indices])
            plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Binance {best_model["model_name"]} - Top Feature Importance')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(results_dir, f"binance_importance_{best_model['model_name']}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Saved importance plot: {plot_path}")
            
    except Exception as e:
        logger.warning(f"⚠️  Could not create plots: {e}")

def save_results(results: List[Dict], config: Dict) -> None:
    """Save comprehensive results."""
    
    results_dir = config["RESULTS_DIR_PATH"]
    os.makedirs(results_dir, exist_ok=True)
    
    # Performance summary
    performance_data = []
    predictions_data = []
    
    for result in results:
        if not result:
            continue
        
        perf_row = {
            'Model': result['model_name'],
            'Exchange': config['EXCHANGE'],
            'Symbol': config['PRIMARY_SYMBOL'],
            'TrainTime': result['train_time'],
            
            # Test metrics
            'Test_R2': result['test_metrics']['r2'],
            'Test_MSE': result['test_metrics']['mse'],
            'Test_MAE': result['test_metrics']['mae'],
            'Test_RMSE': result['test_metrics']['rmse'],
            'Test_Direction_Accuracy': result['test_metrics'].get('direction_accuracy', np.nan),
            'Test_Profit_Capture': result['test_metrics'].get('profit_capture', np.nan),
            
            # Validation metrics
            'Val_R2': result['val_metrics']['r2'],
            'Val_MAE': result['val_metrics']['mae'],
            
            # Overfitting check
            'Overfitting_R2': result['train_metrics']['r2'] - result['test_metrics']['r2']
        }
        performance_data.append(perf_row)
        
        # Predictions (sample)
        y_test = result['predictions']['y_test']
        y_pred = result['predictions']['y_pred']
        
        sample_size = min(100, len(y_test))  # Save sample
        sample_indices = np.linspace(0, len(y_test)-1, sample_size, dtype=int)
        
        for i in sample_indices:
            pred_row = {
                'Model': result['model_name'],
                'Index': i,
                'Actual': y_test[i],
                'Predicted': y_pred[i],
                'Error': abs(y_test[i] - y_pred[i])
            }
            predictions_data.append(pred_row)
    
    # Save performance
    perf_df = pd.DataFrame(performance_data)
    perf_path = os.path.join(results_dir, config["PERFORMANCE_FILENAME"])
    perf_df.to_csv(perf_path, index=False)
    logger.info(f"💾 Saved performance: {perf_path}")
    
    # Save predictions sample
    pred_df = pd.DataFrame(predictions_data)
    pred_path = os.path.join(results_dir, config["PREDICTIONS_FILENAME"])
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"💾 Saved predictions: {pred_path}")
    
    # Analysis summary
    analysis = {
        "binance_prediction_analysis": {
            "exchange": config["EXCHANGE"],
            "symbol": config["PRIMARY_SYMBOL"],
            "analysis_time": datetime.now().isoformat(),
            "models_trained": len(performance_data),
            "best_model": perf_df.loc[perf_df['Test_R2'].idxmax(), 'Model'] if len(perf_df) > 0 else None,
            "average_direction_accuracy": perf_df['Test_Direction_Accuracy'].mean() if len(perf_df) > 0 else 0,
            "feature_count": len(results[0]['feature_names']) if results and results[0] else 0
        }
    }
    
    analysis_path = os.path.join(results_dir, config["ANALYSIS_FILENAME"])
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"💾 Saved analysis: {analysis_path}")

def main(config: Dict) -> None:
    """Main prediction pipeline for Binance."""
    
    logger.info("🟠" + "="*60 + "🟠")
    logger.info("🚀  BINANCE BEST INDICATORS PREDICTION  🚀")
    logger.info("🟠" + "="*60 + "🟠")
    
    # Load data
    df = load_data(config["DATA_PATH"], config.get("BACKUP_DATA_PATH"))
    if df is None:
        logger.error("❌ Cannot proceed without data")
        return
    
    # Load top features
    top_features = load_top_features(config["TOP_FEATURES_DIR"], config["FEATURE_SELECTION_METHOD"])
    if not top_features:
        logger.warning("⚠️  No top features found, using default indicators")
        top_features = ['ta_rsi', 'ta_ema', 'ta_macd', 'talib_sma', 'talib_atr']
    
    # Limit features
    max_features = config.get("MAX_FEATURES", 25)
    if len(top_features) > max_features:
        top_features = top_features[:max_features]
        logger.info(f"🎯 Limited to top {max_features} features")
    
    # Calculate indicators
    df_with_indicators, calculated_features = calculate_best_indicators(df, top_features)
    
    if not calculated_features:
        logger.error("❌ No indicators calculated")
        return
    
    logger.info(f"✅ Using {len(calculated_features)} features for prediction")
    
    # Prepare data
    X, y = prepare_features_target(df_with_indicators, calculated_features, 
                                  config["TARGET_COLUMN"], config["TARGET_SHIFT"])
    
    if len(X) < 100:
        logger.error("❌ Insufficient data for training")
        return
    
    # Split data
    test_size = config["TEST_SIZE"]
    val_size = config.get("VALIDATION_SIZE", 0.15)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, shuffle=config["SHUFFLE_SPLIT"], random_state=42)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, shuffle=config["SHUFFLE_SPLIT"], random_state=42)
    
    logger.info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Train models
    results = []
    models = config["MODELS"]
    
    for name, model in tqdm(models.items(), desc="Training models"):
        result = train_evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test, config)
        if result:
            results.append(result)
    
    if not results:
        logger.error("❌ No models trained successfully")
        return
    
    # Create visualizations
    create_plots(results, config)
    
    # Save results
    save_results(results, config)
    
    # Summary
    best_result = max(results, key=lambda r: r['test_metrics']['r2'])
    
    logger.info("\n" + "="*60)
    logger.info("📊 BINANCE PREDICTION SUMMARY")
    logger.info("="*60)
    logger.info(f"💰 Symbol: {config['PRIMARY_SYMBOL']}")
    logger.info(f"🎯 Features used: {len(calculated_features)}")
    logger.info(f"🤖 Models trained: {len(results)}")
    logger.info(f"🏆 Best model: {best_result['model_name']}")
    logger.info(f"📈 Best R²: {best_result['test_metrics']['r2']:.4f}")
    logger.info(f"🎯 Best direction accuracy: {best_result['test_metrics'].get('direction_accuracy', 0):.3f}")
    
    logger.info("\n✅ Binance prediction analysis completed!")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CONFIG["DATA_PATH"] = os.path.join(script_dir, os.pardir, "Day_31_Projects", "binance_data", f'{PRIMARY_SYMBOL}_historical_data.csv')
        CONFIG["RESULTS_DIR_PATH"] = os.path.join(script_dir, "binance_ml", "results")
        CONFIG["TOP_FEATURES_DIR"] = os.path.join(script_dir, "binance_ml", "results")
        main(CONFIG)
    except Exception as e:
        logger.error(f"❌ Prediction pipeline failed: {e}")
