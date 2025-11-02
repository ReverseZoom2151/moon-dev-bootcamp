#!/usr/bin/env python3
"""
Bitfinex Professional Best Indicators Prediction Model

Advanced ML prediction system for Bitfinex using top-performing technical indicators.
Focuses on professional trading with enhanced validation, institutional-grade metrics,
and sophisticated risk assessment for margin and derivatives trading.
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from Day_26_Projects.bitfinex_config import PRIMARY_SYMBOL
except ImportError:
    PRIMARY_SYMBOL = "btcusd"

# --- Professional Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, os.pardir, "Day_31_Projects", "bitfinex_data")
ml_results_dir = os.path.join(script_dir, "bitfinex_ml", "results")

CONFIG = {
    # --- Paths ---
    "DATA_PATH": os.path.join(data_dir, f'{PRIMARY_SYMBOL}_historical_data.csv'),
    "BACKUP_DATA_PATH": os.path.join(data_dir, 'btcusd_1h_historical.csv'),
    "RESULTS_DIR_PATH": ml_results_dir,
    "TOP_FEATURES_DIR": ml_results_dir,
    
    # Professional output files
    "PERFORMANCE_FILENAME": "bitfinex_professional_model_performance.csv",
    "PREDICTIONS_FILENAME": "bitfinex_professional_predictions.csv",
    "ANALYSIS_FILENAME": "bitfinex_professional_prediction_analysis.json",
    
    # --- Exchange Specific ---
    "EXCHANGE": "Bitfinex",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "TRADING_FOCUS": "professional_trading",
    
    # --- Data & Features ---
    "TARGET_COLUMN": "close",
    "TARGET_SHIFT": -1,  # Predict next period
    
    # --- Model Training ---
    "TEST_SIZE": 0.2,
    "VALIDATION_SIZE": 0.15,
    "SHUFFLE_SPLIT": False,  # Time series
    
    # --- Scaling ---
    "USE_SCALING": True,
    
    # Professional model suite
    "MODELS": {},  # Populated below
    
    # --- Feature Analysis ---
    "USE_TOP_FEATURES": True,
    "MAX_FEATURES": 35,  # Professional: more features
    "FEATURE_SELECTION_METHOD": "importance",
    
    # --- Advanced Analysis ---
    "CALC_PERMUTATION_IMPORTANCE": True,
    "PERMUTATION_N_REPEATS": 5,
    "CREATE_PREDICTIONS_PLOT": True,
    "CREATE_IMPORTANCE_PLOT": True,
    "PLOT_SAMPLE_SIZE": 1000,
    
    # --- Professional Risk Metrics ---
    "CALCULATE_SHARPE_RATIO": True,
    "CALCULATE_MAX_DRAWDOWN": True,
    "CALCULATE_VAR": True,
    "VAR_CONFIDENCE": 0.95,
}

def init_professional_models():
    """Initialize professional model suite."""
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=2000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=3,
            random_state=42, n_jobs=-1
        ),
        'ExtraTreesRegressor': ExtraTreesRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        'GradientBoostingRegressor': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.08, max_depth=8,
            subsample=0.8, random_state=42
        ),
        'XGBRegressor': XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0, n_jobs=-1
        )
    }
    
    if HAS_LIGHTGBM:
        models['LGBMRegressor'] = LGBMRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=-1, n_jobs=-1
        )
    
    return models

CONFIG["MODELS"] = init_professional_models()

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [Bitfinex Professional] %(message)s'
)
logger = logging.getLogger(__name__)

def load_professional_data(filepath: str, backup_path: str = None) -> Optional[pd.DataFrame]:
    """Load Bitfinex data with professional validation."""
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
                    
                    # Professional data quality checks
                    initial_len = len(df)
                    
                    # Remove extreme outliers
                    for col in ['open', 'high', 'low', 'close']:
                        Q1 = df[col].quantile(0.01)
                        Q3 = df[col].quantile(0.99)
                        IQR = Q3 - Q1
                        df = df[(df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)]
                    
                    # Remove zero volume
                    df = df[df['volume'] > 0]
                    
                    logger.info(f"🏛️  Loaded Bitfinex data: {initial_len} → {len(df)} rows")
                    logger.info(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
                    return df.sort_index()
                    
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
    
    logger.error("❌ No valid Bitfinex data found")
    return None

def load_professional_features(results_dir: str, method: str = "importance") -> Optional[List[str]]:
    """Load top features with professional filtering."""
    
    if not os.path.exists(results_dir):
        logger.warning(f"⚠️  Results directory not found: {results_dir}")
        return None
    
    feature_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith("bitfinex_") and method in filename and filename.endswith(".csv"):
            feature_files.append(filename)
    
    if not feature_files:
        logger.warning(f"⚠️  No professional features files found")
        return None
    
    all_features = set()
    feature_scores = {}
    
    for file in feature_files:
        try:
            filepath = os.path.join(results_dir, file)
            df = pd.read_csv(filepath)
            
            if 'Feature' in df.columns:
                for idx, row in df.head(20).iterrows():
                    feature = row['Feature']
                    
                    # Weight by multiple metrics
                    score = 0
                    if 'Importance' in row:
                        score += float(row.get('Importance', 0)) * 0.4
                    if 'R2_Score' in row:
                        score += float(row.get('R2_Score', 0)) * 0.3
                    if 'Direction_Accuracy' in row:
                        score += float(row.get('Direction_Accuracy', 0)) * 0.3
                    
                    if feature not in feature_scores or score > feature_scores[feature]:
                        feature_scores[feature] = score
                    
                    all_features.add(feature)
                
                logger.info(f"🏛️  Loaded features from {file}")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not load {file}: {e}")
    
    if not feature_scores:
        return None
    
    # Sort by scores
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    final_features = [f[0] for f in sorted_features]
    
    logger.info(f"✅ Professional features: {len(final_features)}")
    return final_features

def calculate_professional_indicators(df: pd.DataFrame, feature_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate indicators with professional precision using pandas_ta."""
    
    logger.info(f"🔧 Calculating {len(feature_list)} professional indicators...")
    df_out = df.copy()
    features_added = []
    
    for feature_name in feature_list:
        try:
            if feature_name.startswith('ta_'):
                indicator_base = feature_name.replace('ta_', '').split('_')[0]
                
                if indicator_base == 'rsi':
                    if len(feature_name.split('_')) > 2:
                        period = int(feature_name.split('_')[2])
                        result = ta.rsi(df_out['close'], length=period)
                    else:
                        result = ta.rsi(df_out['close'])
                    
                    if isinstance(result, pd.Series) and not result.empty:
                        df_out[feature_name] = result
                        features_added.append(feature_name)
                
                elif indicator_base in ['ema', 'sma']:
                    if len(feature_name.split('_')) > 2:
                        period = int(feature_name.split('_')[2])
                        if indicator_base == 'ema':
                            result = ta.ema(df_out['close'], length=period)
                        else:
                            result = ta.sma(df_out['close'], length=period)
                    else:
                        if indicator_base == 'ema':
                            result = ta.ema(df_out['close'])
                        else:
                            result = ta.sma(df_out['close'])
                    
                    if isinstance(result, pd.Series) and not result.empty:
                        df_out[feature_name] = result
                        features_added.append(feature_name)
                
                elif indicator_base == 'macd':
                    result = ta.macd(df_out['close'])
                    if isinstance(result, pd.DataFrame):
                        if 'macd_12' in feature_name.lower():
                            df_out[feature_name] = result.iloc[:, 0]
                            features_added.append(feature_name)
                
                elif indicator_base == 'atr':
                    if len(feature_name.split('_')) > 2:
                        period = int(feature_name.split('_')[2])
                        result = ta.atr(df_out['high'], df_out['low'], df_out['close'], length=period)
                    else:
                        result = ta.atr(df_out['high'], df_out['low'], df_out['close'])
                    
                    if isinstance(result, pd.Series) and not result.empty:
                        df_out[feature_name] = result
                        features_added.append(feature_name)
                
                elif indicator_base == 'adx':
                    if len(feature_name.split('_')) > 2:
                        period = int(feature_name.split('_')[2])
                        result = ta.adx(df_out['high'], df_out['low'], df_out['close'], length=period)
                    else:
                        result = ta.adx(df_out['high'], df_out['low'], df_out['close'])
                    
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        df_out[feature_name] = result.iloc[:, 0]  # ADX line
                        features_added.append(feature_name)
                
                elif hasattr(ta, indicator_base):
                    func = getattr(ta, indicator_base)
                    try:
                        result = func(df_out['close'])
                        if isinstance(result, pd.Series) and not result.empty:
                            df_out[feature_name] = result
                            features_added.append(feature_name)
                        elif isinstance(result, pd.DataFrame):
                            df_out[feature_name] = result.iloc[:, 0]
                            features_added.append(feature_name)
                    except:
                        pass
        
        except Exception as e:
            logger.debug(f"Could not calculate {feature_name}: {e}")
    
    logger.info(f"✅ Professional calculation: {len(features_added)} indicators")
    return df_out, features_added

def prepare_professional_data(df: pd.DataFrame, feature_cols: List[str], target_col: str, shift: int, config: Dict) -> Tuple:
    """Prepare features and target for professional analysis."""
    
    logger.info("🎯 Preparing professional prediction dataset...")
    df_out = df.copy()
    
    # Create target
    target_name = f"{target_col}_future_{abs(shift)}"
    df_out[target_name] = df_out[target_col].shift(shift)
    
    # Select and validate features
    missing_cols = [col for col in feature_cols if col not in df_out.columns]
    if missing_cols:
        logger.warning(f"⚠️  Missing features: {missing_cols[:5]}...")
        feature_cols = [f for f in feature_cols if f in df_out.columns]
    
    # Prepare dataset
    all_cols = feature_cols + [target_name]
    df_prepared = df_out[all_cols].copy()
    
    # Professional data cleaning
    df_prepared.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_rows = len(df_prepared)
    df_prepared.dropna(inplace=True)
    
    logger.info(f"📊 Professional preparation: {initial_rows} → {len(df_prepared)} rows")
    
    if df_prepared.empty:
        raise ValueError("No data remaining after professional cleaning")
    
    X = df_prepared[feature_cols]
    y = df_prepared[target_name]
    
    # Scaling if configured
    if config.get("USE_SCALING"):
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        return X_scaled, y, scaler
    
    return X, y, None

def get_professional_metrics(y_true: pd.Series, y_pred: np.ndarray, config: Dict) -> Dict[str, float]:
    """Calculate professional trading metrics."""
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
    try:
        # Direction accuracy
        actual_returns = y_true.pct_change().dropna()
        pred_returns = pd.Series(y_pred, index=y_true.index).pct_change().dropna()
        
        if len(actual_returns) == len(pred_returns) and len(actual_returns) > 0:
            direction_accuracy = (np.sign(actual_returns) == np.sign(pred_returns)).mean()
            metrics['direction_accuracy'] = direction_accuracy
        
        # Professional risk metrics
        if config.get("CALCULATE_SHARPE_RATIO"):
            returns = actual_returns.dropna()
            if len(returns) > 30:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                metrics['sharpe_ratio'] = sharpe_ratio
        
        if config.get("CALCULATE_MAX_DRAWDOWN"):
            cumulative = (1 + actual_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
        
        if config.get("CALCULATE_VAR"):
            confidence = config.get("VAR_CONFIDENCE", 0.95)
            returns = actual_returns.dropna()
            if len(returns) > 100:
                var = np.percentile(returns, (1 - confidence) * 100)
                metrics['var_95'] = var
        
    except Exception as e:
        logger.debug(f"Error calculating professional metrics: {e}")
        for key in ['direction_accuracy', 'sharpe_ratio', 'max_drawdown', 'var_95']:
            if key not in metrics:
                metrics[key] = np.nan
    
    return metrics

def train_evaluate_professional_model(name: str, model: Any, X_train, y_train, X_val, y_val, 
                                     X_test, y_test, config: Dict) -> Optional[Dict]:
    """Train and evaluate model with professional metrics."""
    
    logger.info(f"🏛️  Training professional {name}...")
    
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Professional metrics
        train_metrics = get_professional_metrics(y_train, y_train_pred, config)
        val_metrics = get_professional_metrics(y_val, y_val_pred, config)
        test_metrics = get_professional_metrics(y_test, y_test_pred, config)
        
        logger.info(f"✅ {name}: R²={test_metrics['r2']:.4f}, Dir={test_metrics.get('direction_accuracy', 0):.3f}")
        
        # Feature importance
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        elif config.get("CALC_PERMUTATION_IMPORTANCE"):
            try:
                perm_imp = permutation_importance(
                    model, X_test, y_test, 
                    n_repeats=config["PERMUTATION_N_REPEATS"], 
                    random_state=42
                )
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

def save_professional_results(results: List[Dict], config: Dict) -> None:
    """Save comprehensive professional results."""
    
    results_dir = config["RESULTS_DIR_PATH"]
    os.makedirs(results_dir, exist_ok=True)
    
    # Performance summary
    performance_data = []
    
    for result in results:
        if not result:
            continue
        
        perf_row = {
            'Model': result['model_name'],
            'Exchange': config['EXCHANGE'],
            'Symbol': config['PRIMARY_SYMBOL'],
            'TrainTime': result['train_time'],
            'Test_R2': result['test_metrics']['r2'],
            'Test_MSE': result['test_metrics']['mse'],
            'Test_MAE': result['test_metrics']['mae'],
            'Test_RMSE': result['test_metrics']['rmse'],
            'Test_Direction_Accuracy': result['test_metrics'].get('direction_accuracy', np.nan),
            'Test_Sharpe_Ratio': result['test_metrics'].get('sharpe_ratio', np.nan),
            'Test_Max_Drawdown': result['test_metrics'].get('max_drawdown', np.nan),
            'Test_VaR_95': result['test_metrics'].get('var_95', np.nan),
            'Val_R2': result['val_metrics']['r2'],
            'Val_MAE': result['val_metrics']['mae'],
            'Overfitting_R2': result['train_metrics']['r2'] - result['test_metrics']['r2']
        }
        performance_data.append(perf_row)
    
    # Save performance
    perf_df = pd.DataFrame(performance_data)
    perf_path = os.path.join(results_dir, config["PERFORMANCE_FILENAME"])
    perf_df.to_csv(perf_path, index=False)
    logger.info(f"💾 Saved professional performance: {perf_path}")
    
    # Analysis summary
    analysis = {
        "bitfinex_professional_prediction_analysis": {
            "exchange": config["EXCHANGE"],
            "symbol": config["PRIMARY_SYMBOL"],
            "trading_focus": config["TRADING_FOCUS"],
            "analysis_time": datetime.now().isoformat(),
            "models_trained": len(performance_data),
            "best_model": perf_df.loc[perf_df['Test_R2'].idxmax(), 'Model'] if len(perf_df) > 0 else None,
            "average_direction_accuracy": perf_df['Test_Direction_Accuracy'].mean() if len(perf_df) > 0 else 0,
            "average_sharpe_ratio": perf_df['Test_Sharpe_Ratio'].mean() if len(perf_df) > 0 else 0,
            "feature_count": len(results[0]['feature_names']) if results and results[0] else 0
        }
    }
    
    analysis_path = os.path.join(results_dir, config["ANALYSIS_FILENAME"])
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"💾 Saved professional analysis: {analysis_path}")

def main(config: Dict) -> None:
    """Main professional prediction pipeline for Bitfinex."""
    
    logger.info("🟣" + "="*60 + "🟣")
    logger.info("🏛️   BITFINEX PROFESSIONAL PREDICTION SYSTEM   🏛️")
    logger.info("🟣" + "="*60 + "🟣")
    
    # Load data
    df = load_professional_data(config["DATA_PATH"], config.get("BACKUP_DATA_PATH"))
    if df is None:
        logger.error("❌ Cannot proceed without professional data")
        return
    
    # Load top features
    top_features = load_professional_features(config["TOP_FEATURES_DIR"], config["FEATURE_SELECTION_METHOD"])
    if not top_features:
        logger.warning("⚠️  No top features found, using default indicators")
        top_features = ['ta_rsi', 'ta_ema', 'ta_macd', 'ta_atr', 'ta_adx']
    
    # Limit features
    max_features = config.get("MAX_FEATURES", 35)
    if len(top_features) > max_features:
        top_features = top_features[:max_features]
        logger.info(f"🎯 Limited to top {max_features} features")
    
    # Calculate indicators
    df_with_indicators, calculated_features = calculate_professional_indicators(df, top_features)
    
    if not calculated_features:
        logger.error("❌ No indicators calculated")
        return
    
    logger.info(f"✅ Using {len(calculated_features)} features for prediction")
    
    # Prepare data
    try:
        X, y, scaler = prepare_professional_data(df_with_indicators, calculated_features, 
                                               config["TARGET_COLUMN"], config["TARGET_SHIFT"], config)
    except ValueError as e:
        logger.error(f"❌ Data preparation failed: {e}")
        return
    
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
    
    for name, model in tqdm(models.items(), desc="Training professional models"):
        result = train_evaluate_professional_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test, config)
        if result:
            results.append(result)
    
    if not results:
        logger.error("❌ No models trained successfully")
        return
    
    # Save results
    save_professional_results(results, config)
    
    # Summary
    best_result = max(results, key=lambda r: r['test_metrics']['r2'])
    
    logger.info("\n" + "="*60)
    logger.info("📊 BITFINEX PROFESSIONAL PREDICTION SUMMARY")
    logger.info("="*60)
    logger.info(f"💰 Symbol: {config['PRIMARY_SYMBOL']}")
    logger.info(f"🎯 Features used: {len(calculated_features)}")
    logger.info(f"🤖 Models trained: {len(results)}")
    logger.info(f"🏆 Best model: {best_result['model_name']}")
    logger.info(f"📈 Best R²: {best_result['test_metrics']['r2']:.4f}")
    logger.info(f"🎯 Best direction accuracy: {best_result['test_metrics'].get('direction_accuracy', 0):.3f}")
    logger.info(f"📊 Best Sharpe ratio: {best_result['test_metrics'].get('sharpe_ratio', 0):.3f}")
    
    logger.info("\n✅ Bitfinex professional prediction analysis completed!")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CONFIG["DATA_PATH"] = os.path.join(script_dir, os.pardir, "Day_31_Projects", "bitfinex_data", f'{PRIMARY_SYMBOL}_historical_data.csv')
        CONFIG["RESULTS_DIR_PATH"] = os.path.join(script_dir, "bitfinex_ml", "results")
        CONFIG["TOP_FEATURES_DIR"] = os.path.join(script_dir, "bitfinex_ml", "results")
        main(CONFIG)
    except Exception as e:
        logger.error(f"❌ Prediction pipeline failed: {e}")
