#!/usr/bin/env python3
"""
Binance ML Model Evaluation Pipeline

Evaluates ML models across multiple generations using random subsets
of technical indicators to predict Binance price movements.
"""

import os
import json
import random
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple, Any, Optional

try:
    from Day_26_Projects.binance_config import PRIMARY_SYMBOL
except ImportError:
    PRIMARY_SYMBOL = "BTCUSDT"

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, os.pardir, "Day_31_Projects", "binance_data")
ml_results_dir = os.path.join(script_dir, "binance_ml", "results")

CONFIG = {
    "DATA_PATH": os.path.join(data_dir, f'{PRIMARY_SYMBOL}_historical_data.csv'),
    "BACKUP_DATA_PATH": os.path.join(data_dir, 'BTCUSDT_1h_historical.csv'),
    "RESULTS_BASE_DIR_PATH": ml_results_dir,
    "PANDAS_TA_INDICATORS_PATH": os.path.join(ml_results_dir, 'binance_pandas_ta_indicators.json'),
    "EXTENDED_INDICATORS_PATH": os.path.join(ml_results_dir, 'binance_extended_indicators.json'),
    
    "PERFORMANCE_FILENAME": "binance_model_performance.csv",
    "IMPORTANCE_FILENAME": "binance_feature_importance.csv",
    
    "EXCHANGE": "Binance",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    
    "GENERATIONS": 15,
    "INDICATORS_PER_LIB": 4,
    "TARGET_COLUMN": "close",
    "TARGET_SHIFT": -1,
    "TEST_SIZE": 0.25,
    "SHUFFLE_SPLIT": False,
    
    "MODELS": {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1)
    },
    
    "CALC_PERMUTATION_IMPORTANCE": True,
    "PERMUTATION_N_REPEATS": 3,
    "PERMUTATION_SAMPLE_SIZE": 1500,
}

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [Binance ML] %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath: str, backup_path: str = None) -> Optional[pd.DataFrame]:
    """Load Binance data with fallback."""
    paths_to_try = [filepath]
    if backup_path:
        paths_to_try.append(backup_path)
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                datetime_cols = ['datetime', 'timestamp', 'time']
                for col in datetime_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                        break
                
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    logger.info(f"✅ Loaded Binance data: {df.shape}")
                    return df
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
    
    logger.error("❌ No valid data files found")
    return None

def load_indicator_lists(pandas_path: str, extended_path: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Load indicator lists from JSON files."""
    pandas_ta_indicators, extended_indicators = [], []
    
    # Load pandas_ta
    try:
        for path, name in [(pandas_path, 'pandas_ta'), (extended_path, 'extended')]:
            with open(path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                if name == 'pandas_ta':
                    pandas_ta_indicators = data
                else:
                    extended_indicators = data
            elif isinstance(data, dict) and 'indicators' in data:
                if name == 'pandas_ta':
                    pandas_ta_indicators = data['indicators']
                else:
                    extended_indicators = data['indicators']
            logger.info(f"✅ Loaded {len(pandas_ta_indicators)} pandas_ta indicators")
    except Exception as e:
        logger.error(f"Error loading pandas_ta: {e}")
    
    return pandas_ta_indicators or None, extended_indicators or None

def prepare_target(df: pd.DataFrame, target_col: str, shift: int) -> Tuple[pd.DataFrame, str]:
    """Create future target column."""
    target_name = f"{target_col}_future_{abs(shift)}"
    df[target_name] = df[target_col].shift(shift)
    df = df.iloc[:-abs(shift)]
    logger.info(f"🎯 Target prepared: {target_name}")
    return df, target_name

def calculate_indicators(df: pd.DataFrame, pandas_ta_list: List[str], extended_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate selected technical indicators."""
    df_out = df.copy()
    features_added = []
    
    # pandas_ta indicators
    for indicator in pandas_ta_list:
        try:
            if hasattr(ta, indicator):
                func = getattr(ta, indicator)
                result = func(df_out['close'])
                if isinstance(result, pd.Series):
                    col_name = f"ta_{indicator}"
                    df_out[col_name] = result
                    features_added.append(col_name)
        except Exception as e:
            logger.warning(f"Failed pandas_ta {indicator}: {e}")
    
    # Extended pandas_ta indicators (replacing talib)
    if extended_list:
        logger.info(f"🔧 Calculating {len(extended_list)} extended pandas_ta indicators...")
        
        for indicator in extended_list:
            try:
                indicator_name = indicator.lower()
                col_name = f"ta_{indicator_name}"
                
                # Map common indicators to pandas_ta equivalents
                if indicator_name == 'rsi':
                    df_out[col_name] = ta.rsi(df['close'])
                elif indicator_name == 'macd':
                    macd_result = ta.macd(df['close'])
                    if isinstance(macd_result, pd.DataFrame):
                        df_out[f"{col_name}_line"] = macd_result.iloc[:, 0]
                        features_added.append(f"{col_name}_line")
                elif indicator_name == 'bbands':
                    bbands_result = ta.bbands(df['close'])
                    if isinstance(bbands_result, pd.DataFrame):
                        df_out[f"{col_name}_lower"] = bbands_result.iloc[:, 0]
                        df_out[f"{col_name}_middle"] = bbands_result.iloc[:, 1] 
                        df_out[f"{col_name}_upper"] = bbands_result.iloc[:, 2]
                        features_added.extend([f"{col_name}_lower", f"{col_name}_middle", f"{col_name}_upper"])
                elif indicator_name == 'atr':
                    df_out[col_name] = ta.atr(df['high'], df['low'], df['close'])
                elif indicator_name == 'adx':
                    adx_result = ta.adx(df['high'], df['low'], df['close'])
                    if isinstance(adx_result, pd.DataFrame):
                        df_out[col_name] = adx_result.iloc[:, 0]
                elif indicator_name in ['sma', 'ema', 'wma']:
                    if indicator_name == 'sma':
                        df_out[col_name] = ta.sma(df['close'])
                    elif indicator_name == 'ema':
                        df_out[col_name] = ta.ema(df['close'])
                    elif indicator_name == 'wma':
                        df_out[col_name] = ta.wma(df['close'])
                elif indicator_name == 'stoch':
                    stoch_result = ta.stoch(df['high'], df['low'], df['close'])
                    if isinstance(stoch_result, pd.DataFrame):
                        df_out[f"{col_name}_k"] = stoch_result.iloc[:, 0]
                        df_out[f"{col_name}_d"] = stoch_result.iloc[:, 1]
                        features_added.extend([f"{col_name}_k", f"{col_name}_d"])
                else:
                    # Generic pandas_ta calculation
                    if hasattr(ta, indicator_name):
                        func = getattr(ta, indicator_name)
                        result = func(df['close'])
                        if isinstance(result, pd.Series):
                            df_out[col_name] = result
                        elif isinstance(result, pd.DataFrame):
                            df_out[col_name] = result.iloc[:, 0]
                
                if col_name not in features_added and col_name in df_out.columns:
                    features_added.append(col_name)
                    
            except Exception as e:
                logger.warning(f"Failed extended indicator {indicator}: {e}")
    
    return df_out, features_added

def prepare_features_target(df: pd.DataFrame, features: List[str], target_name: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Prepare features and target for modeling."""
    cols_needed = features + [target_name]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return None
    
    df_clean = df[cols_needed].copy()
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)
    
    if df_clean.empty or len([f for f in features if f in df_clean.columns]) == 0:
        return None
    
    X = df_clean[[f for f in features if f in df_clean.columns]]
    y = df_clean[target_name]
    return X, y

def get_feature_importance(model_name: str, model: Any, X: pd.DataFrame, y: pd.Series, config: Dict) -> Optional[np.ndarray]:
    """Calculate feature importance."""
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
    """Train and evaluate a single model."""
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"✅ {name} Gen {gen}: R²={r2:.4f}, MAE={mae:.4f}")
        
        # Save performance
        perf_data = pd.DataFrame([{
            'Generation': gen, 'Model': name, 'Exchange': config['EXCHANGE'],
            'Symbol': config['PRIMARY_SYMBOL'], 'MSE': mse, 'R2': r2, 'MAE': mae,
            'TrainTime': train_time, 'NumFeatures': len(features),
            'Indicators': ','.join(indicators)
        }])
        perf_data.to_csv(perf_path, mode='a', header=not os.path.exists(perf_path), index=False)
        
        # Save importance
        importance = get_feature_importance(name, model, X_test, y_test, config)
        if importance is not None and len(importance) == len(features):
            imp_data = pd.DataFrame({
                'Generation': gen, 'Model': name, 'Feature': features,
                'Importance': importance, 'R2': r2, 'MAE': mae
            })
            imp_data.to_csv(imp_path, mode='a', header=not os.path.exists(imp_path), index=False)
            
    except Exception as e:
        logger.error(f"❌ Failed {name} Gen {gen}: {e}")

def run_generation(gen: int, df_base: pd.DataFrame, target_name: str, pandas_ta_list: List[str], 
                  extended_list: List[str], models: Dict, config: Dict, perf_path: str, imp_path: str) -> None:
    """Run a single generation."""
    logger.info(f"=== Generation {gen}/{config['GENERATIONS']} ===")
    
    # Select indicators
    num_select = config["INDICATORS_PER_LIB"]
    selected_pandas = random.sample(pandas_ta_list, min(num_select, len(pandas_ta_list)))
    selected_talib = random.sample(talib_list, min(num_select, len(talib_list)))
    all_indicators = selected_pandas + selected_talib
    
    logger.info(f"📊 Selected: {len(selected_pandas)} pandas_ta, {len(selected_talib)} talib")
    
    # Calculate indicators
    df_gen, features = calculate_indicators(df_base, selected_pandas, selected_extended)
    if not features:
        logger.warning("No features calculated, skipping generation")
        return
    
    # Prepare data
    prep_result = prepare_features_target(df_gen, features, target_name)
    if prep_result is None:
        logger.warning("Data preparation failed, skipping generation")
        return
    
    X, y = prep_result
    if len(X) < 50:
        logger.warning("Insufficient data after cleaning, skipping generation")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["TEST_SIZE"], shuffle=config["SHUFFLE_SPLIT"], random_state=42)
    
    # Train models
    for name, model in models.items():
        train_evaluate_model(gen, name, model, X_train, y_train, X_test, y_test, 
                           X.columns.tolist(), all_indicators, config, perf_path, imp_path)

def main(config: Dict) -> None:
    """Main execution pipeline."""
    logger.info("🟠" + "="*60 + "🟠")
    logger.info("🚀  BINANCE ML INDICATOR EVALUATION PIPELINE  🚀")
    logger.info("🟠" + "="*60 + "🟠")
    
    # Setup paths
    results_dir = config["RESULTS_BASE_DIR_PATH"]
    os.makedirs(results_dir, exist_ok=True)
    perf_path = os.path.join(results_dir, config["PERFORMANCE_FILENAME"])
    imp_path = os.path.join(results_dir, config["IMPORTANCE_FILENAME"])
    
    # Load data
    df = load_data(config["DATA_PATH"], config.get("BACKUP_DATA_PATH"))
    if df is None:
        logger.error("❌ No data available")
        return
    
    # Load indicators
    pandas_ta_list, extended_list = load_indicator_lists(config["PANDAS_TA_INDICATORS_PATH"], config["EXTENDED_INDICATORS_PATH"])
    if not pandas_ta_list or not extended_list:
        logger.error("❌ Failed to load indicator lists")
        return
    
    # Prepare target
    df_with_target, target_name = prepare_target(df, config["TARGET_COLUMN"], config["TARGET_SHIFT"])
    
    # Run generations
    logger.info(f"🔄 Starting {config['GENERATIONS']} generations...")
    for gen in tqdm(range(1, config["GENERATIONS"] + 1), desc="Generations"):
        run_generation(gen, df_with_target.copy(), target_name, pandas_ta_list, extended_list, 
                      config["MODELS"], config, perf_path, imp_path)
    
    logger.info("✅ Binance ML evaluation completed!")
    logger.info(f"📁 Results saved in: {results_dir}")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CONFIG["DATA_PATH"] = os.path.join(script_dir, os.pardir, "Day_31_Projects", "binance_data", f'{PRIMARY_SYMBOL}_historical_data.csv')
        CONFIG["RESULTS_BASE_DIR_PATH"] = os.path.join(script_dir, "binance_ml", "results")
        CONFIG["PANDAS_TA_INDICATORS_PATH"] = os.path.join(script_dir, "binance_ml", "results", 'binance_pandas_ta_indicators.json')
        CONFIG["EXTENDED_INDICATORS_PATH"] = os.path.join(script_dir, "binance_ml", "results", 'binance_extended_indicators.json')
        main(CONFIG)
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
