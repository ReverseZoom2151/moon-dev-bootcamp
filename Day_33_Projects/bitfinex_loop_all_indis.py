#!/usr/bin/env python3
"""
Bitfinex Professional ML Model Evaluation Pipeline

Evaluates ML models across multiple generations using random subsets
of technical indicators to predict Bitfinex price movements with
professional trading focus including margin and derivatives analysis.
"""

import os
import json
import random
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import warnings
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

try:
    from Day_26_Projects.bitfinex_config import PRIMARY_SYMBOL
except ImportError:
    PRIMARY_SYMBOL = "btcusd"

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, os.pardir, "Day_31_Projects", "bitfinex_data")
ml_results_dir = os.path.join(script_dir, "bitfinex_ml", "results")

CONFIG = {
    "DATA_PATH": os.path.join(data_dir, f'{PRIMARY_SYMBOL}_historical_data.csv'),
    "BACKUP_DATA_PATH": os.path.join(data_dir, 'btcusd_1h_historical.csv'),
    "RESULTS_BASE_DIR_PATH": ml_results_dir,
    "PANDAS_TA_INDICATORS_PATH": os.path.join(ml_results_dir, 'bitfinex_pandas_ta_indicators.json'),
    "PROFESSIONAL_INDICATORS_PATH": os.path.join(ml_results_dir, 'bitfinex_professional_indicators.json'),
    
    "PERFORMANCE_FILENAME": "bitfinex_model_performance.csv",
    "IMPORTANCE_FILENAME": "bitfinex_feature_importance.csv",
    
    "EXCHANGE": "Bitfinex",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "TRADING_TIER": "professional",
    
    "GENERATIONS": 20,
    "INDICATORS_PER_LIB": 5,
    "TARGET_COLUMN": "close",
    "TARGET_SHIFT": -1,
    "TEST_SIZE": 0.2,
    "SHUFFLE_SPLIT": False,
    
    "MODELS": {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=3000),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=150, random_state=42),
        'XGBRegressor': XGBRegressor(n_estimators=150, random_state=42, verbosity=0, n_jobs=-1),
        'LGBMRegressor': LGBMRegressor(n_estimators=150, random_state=42, verbosity=-1, n_jobs=-1),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    },
    
    "CALC_PERMUTATION_IMPORTANCE": True,
    "PERMUTATION_N_REPEATS": 5,
    "PERMUTATION_SAMPLE_SIZE": 2000,
}

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Bitfinex Pro] %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str, backup_path: str = None) -> Optional[pd.DataFrame]:
    """Load Bitfinex professional data."""
    paths = [filepath]
    if backup_path:
        paths.append(backup_path)
    
    for path in paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                for col in ['datetime', 'timestamp', 'time']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                        break
                
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    # Professional data cleaning
                    df = df[df['volume'] > 0]  # Remove zero volume
                    df['price_change'] = df['close'].pct_change().abs()
                    df = df[df['price_change'] <= 0.5]  # Remove extreme moves
                    df.drop('price_change', axis=1, inplace=True)
                    
                    logger.info(f"✅ Loaded Bitfinex professional data: {df.shape}")
                    return df
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
    
    logger.error("❌ No Bitfinex data found")
    return None

def load_indicator_lists(pandas_path: str, professional_path: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Load professional indicators."""
    pandas_ta_indicators, professional_indicators = [], []
    
    for path, name in [(pandas_path, 'pandas_ta'), (professional_path, 'professional')]:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    indicators = data
                elif isinstance(data, dict) and 'indicators' in data:
                    indicators = data['indicators']
                else:
                    continue
                
                if name == 'pandas_ta':
                    pandas_ta_indicators = indicators
                else:
                    professional_indicators = indicators
                
                logger.info(f"✅ Loaded {len(indicators)} {name} indicators")
        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
    
    return pandas_ta_indicators or None, professional_indicators or None

def prepare_target(df: pd.DataFrame, target_col: str, shift: int) -> Tuple[pd.DataFrame, str]:
    """Prepare professional target."""
    target_name = f"{target_col}_future_{abs(shift)}"
    df[target_name] = df[target_col].shift(shift)
    df = df.iloc[:-abs(shift)]
    logger.info(f"🎯 Professional target: {target_name}")
    return df, target_name

def calculate_indicators(df: pd.DataFrame, pandas_ta_list: List[str], professional_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate professional indicators."""
    df_out = df.copy()
    features_added = []
    
    # pandas_ta with professional parameters
    for indicator in pandas_ta_list:
        try:
            if hasattr(ta, indicator):
                func = getattr(ta, indicator)
                if indicator == 'rsi':
                    for period in [14, 21]:
                        result = func(df_out['close'], length=period)
                        if isinstance(result, pd.Series):
                            col_name = f"ta_rsi_{period}"
                            df_out[col_name] = result
                            features_added.append(col_name)
                elif indicator == 'ema':
                    for period in [12, 26, 50]:
                        result = func(df_out['close'], length=period)
                        if isinstance(result, pd.Series):
                            col_name = f"ta_ema_{period}"
                            df_out[col_name] = result
                            features_added.append(col_name)
                else:
                    result = func(df_out['close'])
                    if isinstance(result, pd.Series):
                        col_name = f"ta_{indicator}"
                        df_out[col_name] = result
                        features_added.append(col_name)
        except Exception as e:
            logger.warning(f"Failed pandas_ta {indicator}: {e}")
    
    # Professional pandas_ta indicators (replacing talib)
    if professional_list:
        logger.info(f"🏆 Calculating {len(professional_list)} professional pandas_ta indicators...")
        
        for indicator in professional_list:
            try:
                indicator_name = indicator.lower()
                col_name = f"prof_{indicator_name}"
                
                # Map professional indicators to pandas_ta equivalents with enhanced parameters
                if indicator_name == 'rsi':
                    df_out[col_name] = ta.rsi(df_out['close'], length=14)
                elif indicator_name == 'macd':
                    macd_result = ta.macd(df_out['close'])
                    if isinstance(macd_result, pd.DataFrame):
                        df_out[f"{col_name}_line"] = macd_result.iloc[:, 0]
                        df_out[f"{col_name}_histogram"] = macd_result.iloc[:, 2] if macd_result.shape[1] > 2 else None
                        features_added.extend([f"{col_name}_line", f"{col_name}_histogram"])
                elif indicator_name == 'bbands':
                    bbands_result = ta.bbands(df_out['close'], length=20)
                    if isinstance(bbands_result, pd.DataFrame):
                        df_out[f"{col_name}_lower"] = bbands_result.iloc[:, 0]
                        df_out[f"{col_name}_middle"] = bbands_result.iloc[:, 1] 
                        df_out[f"{col_name}_upper"] = bbands_result.iloc[:, 2]
                        features_added.extend([f"{col_name}_lower", f"{col_name}_middle", f"{col_name}_upper"])
                elif indicator_name == 'atr':
                    df_out[col_name] = ta.atr(df_out['high'], df_out['low'], df_out['close'], length=14)
                elif indicator_name == 'adx':
                    adx_result = ta.adx(df_out['high'], df_out['low'], df_out['close'], length=14)
                    if isinstance(adx_result, pd.DataFrame):
                        df_out[col_name] = adx_result.iloc[:, 0]
                elif indicator_name == 'stochrsi':
                    stochrsi_result = ta.stochrsi(df_out['close'])
                    if isinstance(stochrsi_result, pd.DataFrame):
                        df_out[f"{col_name}_k"] = stochrsi_result.iloc[:, 0]
                        df_out[f"{col_name}_d"] = stochrsi_result.iloc[:, 1]
                        features_added.extend([f"{col_name}_k", f"{col_name}_d"])
                elif indicator_name == 'vwap':
                    df_out[col_name] = ta.vwap(df_out['high'], df_out['low'], df_out['close'], df_out['volume'])
                elif indicator_name == 'supertrend':
                    supertrend_result = ta.supertrend(df_out['high'], df_out['low'], df_out['close'])
                    if isinstance(supertrend_result, pd.DataFrame):
                        df_out[col_name] = supertrend_result.iloc[:, 0]
                elif indicator_name in ['sma', 'ema', 'wma', 'dema', 'tema']:
                    if indicator_name == 'sma':
                        df_out[col_name] = ta.sma(df_out['close'], length=20)
                    elif indicator_name == 'ema':
                        df_out[col_name] = ta.ema(df_out['close'], length=20)
                    elif indicator_name == 'wma':
                        df_out[col_name] = ta.wma(df_out['close'], length=20)
                    elif indicator_name == 'dema':
                        df_out[col_name] = ta.dema(df_out['close'], length=20)
                    elif indicator_name == 'tema':
                        df_out[col_name] = ta.tema(df_out['close'], length=20)
                else:
                    # Generic professional pandas_ta calculation
                    if hasattr(ta, indicator_name):
                        func = getattr(ta, indicator_name)
                        try:
                            if indicator_name in ['mfi', 'cmf', 'eom']:
                                result = func(df_out['high'], df_out['low'], df_out['close'], df_out['volume'])
                            else:
                                result = func(df_out['close'])
                            
                            if isinstance(result, pd.Series):
                                df_out[col_name] = result
                            elif isinstance(result, pd.DataFrame):
                                df_out[col_name] = result.iloc[:, 0]
                        except:
                            result = func(df_out['close'])
                            if isinstance(result, (pd.Series, pd.DataFrame)):
                                df_out[col_name] = result.iloc[:, 0] if isinstance(result, pd.DataFrame) else result
                
                if col_name not in features_added and col_name in df_out.columns:
                    features_added.append(col_name)
                    
            except Exception as e:
                logger.warning(f"Failed professional indicator {indicator}: {e}")
    
    return df_out, features_added

def prepare_features_target(df: pd.DataFrame, features: List[str], target_name: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Professional data preparation."""
    cols_needed = features + [target_name]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        return None
    
    df_clean = df[cols_needed].copy()
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)
    
    if df_clean.empty:
        return None
    
    final_features = [f for f in features if f in df_clean.columns]
    if not final_features:
        return None
    
    return df_clean[final_features], df_clean[target_name]

def get_professional_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Professional trading metrics."""
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
    try:
        actual_dir = np.sign(y_true.diff().dropna())
        pred_dir = np.sign(pd.Series(y_pred).diff().dropna())
        if len(actual_dir) == len(pred_dir):
            metrics['direction_accuracy'] = (actual_dir == pred_dir).mean()
        else:
            metrics['direction_accuracy'] = np.nan
    except:
        metrics['direction_accuracy'] = np.nan
    
    return metrics

def get_feature_importance(model_name: str, model: Any, X: pd.DataFrame, y: pd.Series, config: Dict) -> Optional[np.ndarray]:
    """Professional importance calculation."""
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_)
    elif config.get("CALC_PERMUTATION_IMPORTANCE"):
        try:
            perm_imp = permutation_importance(model, X, y, n_repeats=config["PERMUTATION_N_REPEATS"], random_state=42)
            return perm_imp.importances_mean
        except:
            return None
    return None

def train_evaluate_model(gen: int, name: str, model: Any, X_train, y_train, X_test, y_test,
                        features: List[str], indicators: List[str], config: Dict, perf_path: str, imp_path: str) -> None:
    """Professional model training."""
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        metrics = get_professional_metrics(y_test, y_pred)
        
        logger.info(f"✅ {name} Gen {gen}: R²={metrics['r2']:.4f}, Dir={metrics['direction_accuracy']:.3f}")
        
        # Save performance
        perf_data = pd.DataFrame([{
            'Generation': gen, 'Model': name, 'Exchange': config['EXCHANGE'],
            'Symbol': config['PRIMARY_SYMBOL'], 'TrainingTier': config['TRADING_TIER'],
            'TrainTime': train_time, 'NumFeatures': len(features),
            'Indicators': ','.join(indicators),
            'Test_R2': metrics['r2'], 'Test_MSE': metrics['mse'], 'Test_MAE': metrics['mae'],
            'Test_RMSE': metrics['rmse'], 'Direction_Accuracy': metrics['direction_accuracy']
        }])
        perf_data.to_csv(perf_path, mode='a', header=not os.path.exists(perf_path), index=False)
        
        # Save importance
        importance = get_feature_importance(name, model, X_test, y_test, config)
        if importance is not None and len(importance) == len(features):
            imp_data = pd.DataFrame({
                'Generation': gen, 'Model': name, 'Feature': features,
                'Importance': importance, 'R2': metrics['r2'], 'Direction_Accuracy': metrics['direction_accuracy']
            })
            imp_data.to_csv(imp_path, mode='a', header=not os.path.exists(imp_path), index=False)
            
    except Exception as e:
        logger.error(f"❌ Failed {name} Gen {gen}: {e}")

def run_generation(gen: int, df_base: pd.DataFrame, target_name: str, pandas_ta_list: List[str], 
                  professional_list: List[str], models: Dict, config: Dict, perf_path: str, imp_path: str) -> None:
    """Professional generation execution."""
    logger.info(f"🏛️  === Professional Generation {gen}/{config['GENERATIONS']} ===")
    
    # Professional indicator selection
    num_select = config["INDICATORS_PER_LIB"]
    high_value_pandas = ['rsi', 'ema', 'macd', 'bbands', 'atr', 'stoch']
    high_value_professional = ['RSI', 'EMA', 'MACD', 'BBANDS', 'ATR', 'STOCH', 'VWAP', 'SUPERTREND']
    
    # Prioritize professional indicators
    selected_pandas = []
    if pandas_ta_list:
        priority = [ind for ind in high_value_pandas if ind in pandas_ta_list]
        others = [ind for ind in pandas_ta_list if ind not in high_value_pandas]
        selected_pandas = random.sample(priority, min(2, len(priority))) + random.sample(others, min(num_select-2, len(others)))
    
    selected_professional = []
    if professional_list:
        priority = [ind for ind in high_value_professional if ind in professional_list]
        others = [ind for ind in professional_list if ind not in high_value_professional]
        selected_professional = random.sample(priority, min(3, len(priority))) + random.sample(others, min(num_select-3, len(others)))
    
    logger.info(f"🎯 Professional: {len(selected_pandas)} pandas_ta, {len(selected_professional)} professional")
    
    # Calculate indicators
    df_gen, features = calculate_indicators(df_base, selected_pandas, selected_professional)
    if not features:
        logger.warning("No features, skipping")
        return
    
    # Prepare data
    prep_result = prepare_features_target(df_gen, features, target_name)
    if prep_result is None:
        logger.warning("Data prep failed, skipping")
        return
    
    X, y = prep_result
    if len(X) < 100:
        logger.warning("Insufficient professional data, skipping")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], shuffle=config["SHUFFLE_SPLIT"], random_state=42)
    
    # Train models
    for name, model in models.items():
        train_evaluate_model(gen, name, model, X_train, y_train, X_test, y_test,
                           X.columns.tolist(), selected_pandas + selected_talib, config, perf_path, imp_path)

def main(config: Dict) -> None:
    """Main professional pipeline."""
    logger.info("🔵" + "="*65 + "🔵")
    logger.info("🏛️   BITFINEX PROFESSIONAL ML EVALUATION PIPELINE   🏛️")
    logger.info("🔵" + "="*65 + "🔵")
    
    # Setup
    results_dir = config["RESULTS_BASE_DIR_PATH"]
    os.makedirs(results_dir, exist_ok=True)
    perf_path = os.path.join(results_dir, config["PERFORMANCE_FILENAME"])
    imp_path = os.path.join(results_dir, config["IMPORTANCE_FILENAME"])
    
    # Load data
    df = load_data(config["DATA_PATH"], config.get("BACKUP_DATA_PATH"))
    if df is None:
        return
    
    # Load indicators
    pandas_ta_list, professional_list = load_indicator_lists(config["PANDAS_TA_INDICATORS_PATH"], config["PROFESSIONAL_INDICATORS_PATH"])
    if not pandas_ta_list or not professional_list:
        logger.error("❌ Failed to load professional indicators")
        return
    
    # Prepare target
    df_with_target, target_name = prepare_target(df, config["TARGET_COLUMN"], config["TARGET_SHIFT"])
    
    # Run professional generations
    logger.info(f"🔄 Starting {config['GENERATIONS']} professional generations...")
    for gen in tqdm(range(1, config["GENERATIONS"] + 1), desc="Professional Gens"):
        run_generation(gen, df_with_target.copy(), target_name, pandas_ta_list, professional_list,
                      config["MODELS"], config, perf_path, imp_path)
    
    logger.info("✅ Bitfinex professional ML evaluation completed!")
    logger.info(f"📁 Professional results: {results_dir}")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CONFIG["DATA_PATH"] = os.path.join(script_dir, os.pardir, "Day_31_Projects", "bitfinex_data", f'{PRIMARY_SYMBOL}_historical_data.csv')
        CONFIG["RESULTS_BASE_DIR_PATH"] = os.path.join(script_dir, "bitfinex_ml", "results")
        CONFIG["PANDAS_TA_INDICATORS_PATH"] = os.path.join(script_dir, "bitfinex_ml", "results", 'bitfinex_pandas_ta_indicators.json')
        CONFIG["PROFESSIONAL_INDICATORS_PATH"] = os.path.join(script_dir, "bitfinex_ml", "results", 'bitfinex_professional_indicators.json')
        main(CONFIG)
    except Exception as e:
        logger.error(f"❌ Professional pipeline failed: {e}")
