#!/usr/bin/env python3
"""
Evaluates ML models across multiple generations, each using a random subset
of technical indicators (Talib & Pandas TA) to predict future BTC price movement.
Results (performance and feature importance) are saved incrementally.
"""

import os
import json
import random
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
from talib import abstract as talib_abstract # Renamed to avoid clash
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
import warnings
from tqdm import tqdm
from sklearn.inspection import permutation_importance
import logging
from typing import Dict, List, Tuple, Any, Optional

# --- Configuration ---
# Get the absolute path of the directory where the script resides
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define base paths relative to the script directory
data_dir = os.path.join(script_dir, os.pardir, "Day_18_Projects", "data") # Assuming data is in Day_18
ml_results_dir = os.path.join(script_dir, "ml", "results")

CONFIG = {
    # --- Paths ---
    "DATA_PATH": os.path.join(data_dir, 'BTC-5m-30wks-data.csv'),
    "RESULTS_BASE_DIR_PATH": ml_results_dir,
    "PANDAS_TA_INDICATORS_PATH": os.path.join(ml_results_dir, 'pandas_ta_indicators.json'),
    "TALIB_INDICATORS_PATH": os.path.join(ml_results_dir, 'talib_indicators.json'),
    # Filenames within RESULTS_BASE_DIR_PATH
    "PERFORMANCE_FILENAME": "model_performance.csv",
    "IMPORTANCE_FILENAME": "feature_importance.csv", # Note: Long format

    # --- Generation Settings ---
    "GENERATIONS": 10, # Reduced for quicker testing, adjust as needed
    "INDICATORS_PER_LIB": 3, # Number of indicators to select from each library per generation

    # --- Data & Features ---
    "TARGET_COLUMN": "close",
    "TARGET_SHIFT": -1, # Predict next period's close

    # --- Model Training ---
    "TEST_SIZE": 0.2,
    "SHUFFLE_SPLIT": False, # Time series data
    "MODELS": {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1),
        # SVR and MLP can be slow, especially with permutation importance
        # 'SVR': SVR(),
        # 'MLPRegressor': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
    },

    # --- Feature Importance ---
    "CALC_PERMUTATION_IMPORTANCE": False, # Set True for SVR/MLP if included
    "PERMUTATION_N_REPEATS": 5, # Lower for speed if enabled
    "PERMUTATION_SAMPLE_SIZE": 1000, # Limit sample size for permutation importance speed
}

# --- Setup ---
warnings.filterwarnings('ignore') # Suppress warnings (use cautiously)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions ---

def append_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """Appends a DataFrame to a CSV file, creating file/header if needed."""
    try:
        file_exists = os.path.exists(filepath)
        df.to_csv(filepath, mode='a', header=not file_exists, index=False)
        # logger.debug(f"Appended data to {filepath}") # Can be noisy
    except Exception as e:
        logger.error(f"Error appending DataFrame to {filepath}: {e}", exc_info=True)

def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Saves a DataFrame to a CSV file, creating directories if needed."""
    try:
        output_dir = os.path.dirname(filepath)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"DataFrame saved successfully to: {filepath}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filepath}: {e}", exc_info=True)

# --- Core Functions ---

def load_data(filepath: str) -> Optional[pd.DataFrame]:
    """Loads data from CSV, parsing datetime."""
    if not os.path.exists(filepath):
        logger.error(f"Data file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        logger.info(f"Data loaded from {filepath}. Shape: {df.shape}")
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
         logger.error(f"Error loading data: {e}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}", exc_info=True)
        return None

def load_indicator_lists(pandas_path: str, talib_path: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Loads indicator names from JSON files."""
    pandas_ta_indicators, talib_indicators = None, None
    try:
        with open(pandas_path, 'r') as f:
            pandas_ta_indicators = json.load(f)
        logger.info(f"Loaded {len(pandas_ta_indicators)} pandas_ta indicators from {pandas_path}")
    except Exception as e:
        logger.error(f"Error loading pandas_ta indicators from {pandas_path}: {e}", exc_info=True)

    try:
        with open(talib_path, 'r') as f:
            talib_indicators = json.load(f)
        logger.info(f"Loaded {len(talib_indicators)} talib indicators from {talib_path}")
    except Exception as e:
        logger.error(f"Error loading talib indicators from {talib_path}: {e}", exc_info=True)

    return pandas_ta_indicators, talib_indicators

def prepare_target(df: pd.DataFrame, target_col: str, shift: int) -> Tuple[pd.DataFrame, str]:
    """Creates the future target column and removes initial NaN rows."""
    logger.info(f"Preparing target column '{target_col}' shifted by {shift}")
    target_name = f"{target_col}_future_{abs(shift)}"
    df[target_name] = df[target_col].shift(shift)
    initial_rows = len(df)
    df = df.iloc[:-abs(shift)] # Drop last rows with NaNs from target shift
    logger.info(f"Dropped {initial_rows - len(df)} rows due to target shifting.")
    return df, target_name

def calculate_selected_indicators(df: pd.DataFrame,
                                selected_pandas_ta: List[str],
                                selected_talib: List[str]
                                ) -> Tuple[pd.DataFrame, List[str]]:
    """Calculates selected indicators for the given DataFrame generation."""
    df_out = df.copy()
    features_added = []
    logger.debug(f"Calculating pandas_ta: {selected_pandas_ta}")
    logger.debug(f"Calculating talib: {selected_talib}")

    # Pandas TA
    if selected_pandas_ta:
        strategy = ta.Strategy(
            name="Selected_PandasTA",
            ta=[{'kind': ind} for ind in selected_pandas_ta]
        )
        try:
            df_out.ta.strategy(strategy, append=True)
            # Infer added columns (may need refinement based on pandas_ta naming)
            added_cols = [col for col in df_out.columns if col not in df.columns]
            features_added.extend(added_cols)
            logger.debug(f"pandas_ta added columns: {added_cols}")
        except Exception as e:
            logger.warning(f"Error applying pandas_ta strategy: {e}")

    # Talib
    required_talib_cols = {'open', 'high', 'low', 'close', 'volume'}
    df_cols_lower = {col.lower() for col in df_out.columns}
    if not required_talib_cols.issubset(df_cols_lower):
        logger.warning(f"Skipping Talib indicators: Missing required columns {required_talib_cols - df_cols_lower}")
    else:
        col_map = {c.lower(): c for c in df_out.columns}
        inputs = {k: df_out[col_map[k]] for k in required_talib_cols if k in col_map}
        for indi_name in selected_talib:
            try:
                func = talib_abstract.Function(indi_name)
                # Talib abstract API expects a dict of numpy arrays or pd.Series
                # Ensure inputs match the specific requirements of the indi_name func if needed
                output = func(inputs) # Basic call, assumes standard OHLCV input needs
                if isinstance(output, pd.Series):
                    col_name = indi_name # Or a more specific name if needed
                    df_out[col_name] = output
                    features_added.append(col_name)
                    logger.debug(f"Talib added Series: {col_name}")
                elif isinstance(output, pd.DataFrame):
                    # Add all columns from the DataFrame output
                    df_out = df_out.join(output)
                    features_added.extend(output.columns.tolist())
                    logger.debug(f"Talib added DataFrame columns: {output.columns.tolist()}")
                elif isinstance(output, np.ndarray):
                     col_name = indi_name # Or a more specific name if needed
                     df_out[col_name] = output
                     features_added.append(col_name)
                     logger.debug(f"Talib added ndarray: {col_name}")
                else:
                    logger.warning(f"Talib indicator '{indi_name}' produced unexpected output type: {type(output)}")

            except Exception as e:
                logger.warning(f"Could not calculate talib indicator '{indi_name}': {e}")

    final_features = sorted(list(set(f for f in features_added if f in df_out.columns)))
    logger.debug(f"Features added in this generation: {final_features}")
    return df_out, final_features

def prepare_features_target_gen(df_gen: pd.DataFrame, feature_cols: List[str], target_name: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Prepares features (X) and target (y) for a generation, drops NaNs."""
    logger.debug("Preparing features and target for generation...")
    cols_to_keep = feature_cols + [target_name]
    missing_cols = [col for col in cols_to_keep if col not in df_gen.columns]
    if missing_cols:
        logger.error(f"Columns missing after indicator calculation: {missing_cols}")
        return None

    df_prepared = df_gen[cols_to_keep].copy()

    initial_rows = len(df_prepared)
    df_prepared.dropna(inplace=True)
    rows_dropped = initial_rows - len(df_prepared)
    if rows_dropped > 0:
        logger.debug(f"Dropped {rows_dropped} rows with NaN values from indicators.")

    if df_prepared.empty:
        logger.warning("DataFrame is empty after preparing features and target for this generation.")
        return None
    if not feature_cols:
         logger.warning("No feature columns remain after processing. Skipping generation.")
         return None
    # Ensure only existing feature columns are used
    final_feature_cols = [f for f in feature_cols if f in df_prepared.columns]
    if not final_feature_cols:
        logger.warning("No usable feature columns found after NaN drop. Skipping generation.")
        return None


    X = df_prepared[final_feature_cols]
    y = df_prepared[target_name]
    logger.debug(f"Final shape for modeling: {X.shape}, {y.shape}")
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, shuffle: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into training and testing sets."""
    logger.debug(f"Splitting data: test_size={test_size}, shuffle={shuffle}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    logger.debug(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def get_models(config: Dict) -> Dict[str, Any]:
    """Returns dictionary of ML models defined in config."""
    return config.get("MODELS", {})

def get_feature_importance(model_name: str, model: Any, X: pd.DataFrame, y: pd.Series, features: List[str], config: Dict) -> Optional[np.ndarray]:
    """Extracts or calculates feature importance for a given model."""
    importance_values = None
    if model_name in ['RandomForestRegressor', 'XGBRegressor']:
        try: importance_values = model.feature_importances_
        except AttributeError: pass # Handled below
    elif model_name == 'LinearRegression':
        try: importance_values = np.abs(model.coef_)
        except AttributeError: pass # Handled below

    # Calculate Permutation Importance if needed or if direct importance failed
    if importance_values is None and config.get("CALC_PERMUTATION_IMPORTANCE", False):
        logger.info(f"Calculating permutation importance for {model_name}...")
        try:
            sample_size = min(config.get("PERMUTATION_SAMPLE_SIZE", 1000), len(X))
            if sample_size < len(X):
                logger.debug(f"Sampling {sample_size} rows for permutation importance.")
                X_sample = X.sample(n=sample_size, random_state=42)
                y_sample = y.loc[X_sample.index]
            else:
                X_sample, y_sample = X, y

            perm_importance = permutation_importance(
                model, X_sample, y_sample,
                n_repeats=config.get("PERMUTATION_N_REPEATS", 5),
                random_state=42, scoring='r2', n_jobs=-1
            )
            importance_values = perm_importance.importances_mean
        except Exception as e:
             logger.error(f"Error calculating permutation importance for {model_name}: {e}", exc_info=True)
             return None # Return None on error
    elif importance_values is None:
         logger.info(f"Feature importance calculation not available or configured for {model_name}.")
         return None # Return None if not calculable/configured

    # Ensure importance array length matches features
    if importance_values is not None and len(importance_values) != len(features):
        logger.error(f"Feature importance length mismatch for {model_name}. Expected {len(features)}, got {len(importance_values)}")
        return None

    return importance_values


def train_evaluate_save(gen: int, name: str, model: Any,
                        X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        features: List[str], selected_indicators: List[str],
                        config: Dict, perf_filepath: str, imp_filepath: str) -> None:
    """Trains a single model, evaluates, gets importance, and saves results."""
    logger.info(f"--- Training {name} (Gen: {gen}) ---")
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"{name} trained in {train_time:.2f}s")

        logger.info(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

        # Save Performance Metrics
        perf_data = {
            'Generation': gen, 'Model': name, 'MSE': mse, 'R2': r2,
            'NumFeatures': len(features),
            'Indicators': ','.join(selected_indicators)
        }
        append_to_csv(pd.DataFrame([perf_data]), perf_filepath)

        # Get and Save Feature Importance
        importances = get_feature_importance(name, model, X_test, y_test, features, config)

        if importances is not None:
            # Ensure features and importances align if get_feature_importance succeeded
            if len(importances) == len(features):
                 imp_data = pd.DataFrame({
                     'Generation': gen,
                     'Model': name,
                     'Feature': features,
                     'Importance': importances,
                     'MSE': mse, # Include metrics for context
                     'R2': r2
                 })
                 append_to_csv(imp_data, imp_filepath)
            else:
                 logger.error(f"Feature/Importance length mismatch for {name} in Gen {gen}. Skipping importance save.")

    except Exception as e:
        logger.error(f"Failed to train/evaluate/save model {name} in Gen {gen}: {e}", exc_info=True)
        # Optionally save failed state
        perf_data = {
            'Generation': gen, 'Model': name, 'MSE': float('nan'), 'R2': float('nan'),
            'NumFeatures': len(features), 'Indicators': ','.join(selected_indicators)
        }
        append_to_csv(pd.DataFrame([perf_data]), perf_filepath)


def run_generation(gen: int, df_base: pd.DataFrame, target_name: str,
                   pandas_ta_list: List[str], talib_list: List[str],
                   models: Dict[str, Any], config: Dict,
                   perf_filepath: str, imp_filepath: str) -> None:
    """Runs a single generation of indicator selection, training, and evaluation."""
    logger.info(f"=== Starting Generation {gen}/{config['GENERATIONS']} ===")

    # 1. Select Indicators
    num_to_select = config["INDICATORS_PER_LIB"]
    selected_pandas_ta = random.sample(pandas_ta_list, min(num_to_select, len(pandas_ta_list)))
    selected_talib = random.sample(talib_list, min(num_to_select, len(talib_list)))
    selected_indicators_all = selected_pandas_ta + selected_talib
    logger.info(f"Selected pandas_ta: {selected_pandas_ta}")
    logger.info(f"Selected talib: {selected_talib}")

    # 2. Calculate Indicators for this generation
    df_gen, features_added = calculate_selected_indicators(df_base, selected_pandas_ta, selected_talib)
    if not features_added:
        logger.warning(f"No features were successfully calculated in Gen {gen}. Skipping.")
        return

    # 3. Prepare Features & Target (Handles NaN drop)
    prep_result = prepare_features_target_gen(df_gen, features_added, target_name)
    if prep_result is None:
        logger.warning(f"Feature/Target preparation failed for Gen {gen}. Skipping.")
        return
    X, y = prep_result
    final_features = X.columns.tolist() # Use actual columns after NaN drop

    # 4. Split Data
    if len(X) < 10: # Arbitrary small number check
         logger.warning(f"Insufficient data ({len(X)} rows) after processing for Gen {gen}. Skipping.")
         return
    X_train, X_test, y_train, y_test = split_data(X, y, config["TEST_SIZE"], config["SHUFFLE_SPLIT"])


    # 5. Train, Evaluate, and Save each model
    for name, model_instance in models.items():
        train_evaluate_save(
            gen, name, model_instance,
            X_train, y_train, X_test, y_test,
            final_features, selected_indicators_all,
            config, perf_filepath, imp_filepath
        )

    logger.info(f"=== Completed Generation {gen} ===")


# --- Main Orchestration ---

def main(config: Dict) -> None:
    """Main function to run the ML evaluation pipeline across generations."""
    logger.info("=== ML Model Evaluation Script (Generational) Started ===")

    results_dir = config["RESULTS_BASE_DIR_PATH"]
    os.makedirs(results_dir, exist_ok=True)
    perf_filepath = os.path.join(results_dir, config["PERFORMANCE_FILENAME"])
    imp_filepath = os.path.join(results_dir, config["IMPORTANCE_FILENAME"])
    logger.info(f"Results will be saved to directory: {results_dir}")
    logger.info(f"Performance data: {perf_filepath}")
    logger.info(f"Importance data: {imp_filepath}")

    # Clean previous results files if they exist (optional)
    # if os.path.exists(perf_filepath): os.remove(perf_filepath)
    # if os.path.exists(imp_filepath): os.remove(imp_filepath)

    # 1. Load Base Data
    df_base = load_data(config["DATA_PATH"])
    if df_base is None: return

    # 2. Load Indicator Lists
    pandas_ta_list, talib_list = load_indicator_lists(
        config["PANDAS_TA_INDICATORS_PATH"], config["TALIB_INDICATORS_PATH"]
    )
    if pandas_ta_list is None or talib_list is None:
        logger.error("Failed to load one or both indicator lists. Exiting.")
        return

    # 3. Prepare Base Target Variable (do this once)
    df_base_with_target, target_name = prepare_target(df_base, config["TARGET_COLUMN"], config["TARGET_SHIFT"])

    # 4. Get Models
    models = get_models(config)
    if not models:
        logger.error("No models defined in configuration. Exiting.")
        return

    # 5. Run Generations
    for gen in tqdm(range(1, config["GENERATIONS"] + 1), desc="Overall Generations"):
        run_generation(
            gen, df_base_with_target.copy(), target_name, # Pass copy to avoid mutation
            pandas_ta_list, talib_list, models, config,
            perf_filepath, imp_filepath
        )

    logger.info("=== ML Model Evaluation (Generational) Complete ===")
    logger.info(f"Final results saved in: {results_dir}")

if __name__ == "__main__":
    # Note: Using __file__ assumes the script is run directly.
    try:
        # Recalculate paths within the main block to ensure context
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, os.pardir, "Day_18_Projects", "data") # Adjust if data path differs
        ml_results_dir = os.path.join(script_dir, "ml", "results")
        # Ensure the CONFIG reflects the calculated paths
        CONFIG["DATA_PATH"] = os.path.join(data_dir, 'BTC-5m-30wks-data.csv')
        CONFIG["RESULTS_BASE_DIR_PATH"] = ml_results_dir
        CONFIG["PANDAS_TA_INDICATORS_PATH"] = os.path.join(ml_results_dir, 'pandas_ta_indicators.json')
        CONFIG["TALIB_INDICATORS_PATH"] = os.path.join(ml_results_dir, 'talib_indicators.json')
        main(CONFIG)
    except NameError:
        logger.error("Could not determine script directory automatically using __file__.")
        logger.error("Please ensure CONFIG paths are correct or run as a script.")
        # Optionally attempt to run with potentially hardcoded paths
        # if all(k in CONFIG and CONFIG[k] for k in ["DATA_PATH", "RESULTS_BASE_DIR_PATH", ...]): # Add other path checks
        #    logger.warning("Attempting to proceed with paths defined in CONFIG.")
        #    main(CONFIG)
        # else:
        #    logger.error("Cannot proceed without valid paths.")
