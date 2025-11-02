import os
import pandas as pd
import numpy as np
import talib
import time
import logging
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple, Any, Optional

# Fallback for settings if import fails
try:
    from backend.core.config import settings
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR_PATH = os.path.join(BASE_DIR, 'ml', 'results')
    settings = type('Settings', (), {'RESULTS_DIR_PATH': RESULTS_DIR_PATH})()

# --- Configuration ---
CONFIG = {
    # --- Data & Features ---
    'TARGET_COLUMN': 'close',
    'TARGET_SHIFT': -1,  # Predict next period's close
    'PANDAS_TA_INDICATORS': [
        {'kind': 'rsi'},
        {'kind': 'macd'},  # Will extract MACD line
        {'kind': 'ema', 'length': 20}  # Example with parameter
    ],
    'TALIB_INDICATORS': ['ADX', 'CCI', 'ROC'],  # Function names in talib

    # --- Model Training ---
    'TEST_SIZE': 0.2,
    'SHUFFLE_SPLIT': False,  # Time series data, usually False
    'MODELS': {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1),
    },

    # --- Feature Importance ---
    'CALC_PERMUTATION_IMPORTANCE': False,
    'PERMUTATION_N_REPEATS': 5,  # Lower repeats for speed if enabled

    # --- Plotting ---
    'PLOT_BEST_PREDICTIONS': False,
    'PLOT_FEATURE_IMPORTANCE': False,
    'PLOT_SAMPLE_SIZE': 1000  # Limit plot size
}

# --- Setup ---
warnings.filterwarnings('ignore')  # Suppress warnings globally (use cautiously)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions ---

def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Saves a DataFrame to a CSV file, creating directories if needed."""
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(filepath)
        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrame
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
        # Optional: Set datetime as index if preferred
        # df.set_index('datetime', inplace=True)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
         logger.error(f"Error loading data: {e}")
         return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}", exc_info=True)
        return None

def calculate_indicators(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    """Calculates technical indicators using pandas_ta and talib."""
    logger.info("Calculating technical indicators...")
    features_added = []
    df_out = df.copy()

    # Pandas TA indicators
    for indi_config in config.get("PANDAS_TA_INDICATORS", []):
        try:
            kind = indi_config.pop("kind")  # Extract kind, rest are params
            logger.debug(f"Calculating pandas_ta: {kind} with params {indi_config}")
            # Use df.ta accessor
            df_out.ta(kind=kind, append=True, **indi_config)
            # Figure out column names pandas_ta added (can be complex)
            # Simple approach: assume common names or manually list them
            if kind == "rsi": features_added.append("RSI_14")  # Default name
            if kind == "macd": features_added.extend(["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"])
            if kind == "ema": features_added.append(f"EMA_{indi_config.get('length', 10)}")  # Example naming
        except Exception as e:
            logger.warning(f"Could not calculate pandas_ta indicator '{kind}': {e}")
            indi_config["kind"] = kind  # Put kind back for logging

    # Talib indicators
    # Ensure required columns (lowercase) exist for talib
    required_talib_cols = {'open', 'high', 'low', 'close', 'volume'}
    df_cols_lower = {col.lower() for col in df_out.columns}
    if not required_talib_cols.issubset(df_cols_lower):
         logger.warning(f"Skipping Talib indicators: Missing required columns {required_talib_cols - df_cols_lower}")
    else:
         # Map to actual case for talib functions
         col_map = {c.lower(): c for c in df_out.columns}
         for indi_name in config.get("TALIB_INDICATORS", []):
            try:
                logger.debug(f"Calculating talib: {indi_name}")
                func = getattr(talib, indi_name)
                # Assuming standard OHLCV inputs, adjust if needed
                if indi_name in ['ADX', 'ADXR', 'CCI', 'ATR', 'NATR']:  # Require HLC
                     df_out[indi_name.lower()] = func(df_out[col_map['high']], df_out[col_map['low']], df_out[col_map['close']])
                elif indi_name in ['ROC', 'RSI', 'SMA', 'EMA', 'MACD']:  # Require Close
                     result = func(df_out[col_map['close']])
                     # Handle multi-output indicators like MACD
                     if isinstance(result, tuple):
                          # Add specific logic if you need MACD signal/hist, e.g.
                          if indi_name == 'MACD':
                               df_out['macd_line'] = result[0]
                               features_added.append('macd_line')
                          # else add all outputs with indexed names?
                     else:
                          df_out[indi_name.lower()] = result
                          features_added.append(indi_name.lower())
                # Add more conditions based on inputs required by other talib funcs
                else:
                     logger.warning(f"Talib indicator '{indi_name}' calculation logic not fully implemented.")
                     continue  # Skip adding to features_added if not implemented

            except Exception as e:
                logger.warning(f"Could not calculate talib indicator '{indi_name}': {e}")

    # Clean up feature list (remove duplicates, ensure they exist)
    final_features = sorted(list(set(f for f in features_added if f in df_out.columns)))
    logger.info(f"Indicators calculated. Features added: {final_features}")
    logger.debug(f"DataFrame columns after indicators: {df_out.columns.tolist()}")
    return df_out, final_features

def prepare_features_target(df: pd.DataFrame, feature_cols: List[str], target_col: str, shift: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares features (X) and target (y), shifts target, and drops NaNs."""
    logger.info("Preparing features and target...")
    df_out = df.copy()

    # Create lagged target
    target_name = f"{target_col}_future_{abs(shift)}"
    df_out[target_name] = df_out[target_col].shift(shift)

    # Select feature columns + new target column
    cols_to_keep = feature_cols + [target_name]
    missing_cols = [col for col in cols_to_keep if col not in df_out.columns]
    if missing_cols:
        raise ValueError(f"Columns missing after indicator calculation: {missing_cols}")

    df_prepared = df_out[cols_to_keep].copy()

    # Drop rows with NaN values (from indicators or target shift)
    initial_rows = len(df_prepared)
    df_prepared.dropna(inplace=True)
    logger.info(f"Dropped {initial_rows - len(df_prepared)} rows with NaN values.")
    logger.info(f"Final shape for modeling: {df_prepared.shape}")

    if df_prepared.empty:
         raise ValueError("DataFrame is empty after preparing features and target.")

    X = df_prepared[feature_cols]
    y = df_prepared[target_name]
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, shuffle: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into training and testing sets."""
    logger.info(f"Splitting data: test_size={test_size}, shuffle={shuffle}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def get_models(config: Dict) -> Dict[str, Any]:
    """Returns dictionary of ML models defined in config."""
    # Could add more complex model instantiation here if needed
    return config.get("MODELS", {})

def get_feature_importance(model_name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series, features: List[str], config: Dict) -> Optional[np.ndarray]:
    """Extracts or calculates feature importance for a given model."""
    if model_name in ['RandomForestRegressor', 'XGBRegressor']:
        try:
            return model.feature_importances_
        except AttributeError:
             logger.warning(f"Could not get feature_importances_ for {model_name}")
             return None
    elif model_name == 'LinearRegression':
        try:
            return np.abs(model.coef_)
        except AttributeError:
            logger.warning(f"Could not get coef_ for {model_name}")
            return None
    elif config.get("CALC_PERMUTATION_IMPORTANCE", False):
        # For SVR, MLPRegressor, or others if specified
        logger.info(f"Calculating permutation importance for {model_name}...")
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test,
                n_repeats=config.get("PERMUTATION_N_REPEATS", 5),
                random_state=42, scoring='r2', n_jobs=-1
            )
            return perm_importance.importances_mean
        except Exception as e:
             logger.error(f"Error calculating permutation importance for {model_name}: {e}")
             return None
    else:
        logger.info(f"Feature importance calculation not configured for {model_name}.")
        return None

def train_evaluate_model(name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, features: List[str], config: Dict) -> Tuple[Dict, pd.Series, Optional[np.ndarray]]:
    """Trains a single model, evaluates, and gets feature importance."""
    logger.info(f"--- Training {name} ---")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"{name} trained in {train_time:.2f}s")

    logger.info(f"Evaluating {name}...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {'MSE': mse, 'R2': r2}
    logger.info(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

    importances = get_feature_importance(name, model, X_test, y_test, features, config)
    importance_series = pd.Series(importances, index=features) if importances is not None else pd.Series(dtype=float)

    return metrics, importance_series, y_pred

def plot_actual_vs_predicted(y_test: pd.Series, y_pred: np.ndarray, model_name: str, output_dir: str, prefix: str = "actual_vs_predicted", limit: Optional[int] = 1000):
    """Plots actual vs predicted values."""
    logger.info(f"Plotting Actual vs Predicted for {model_name}...")
    try:
        plt.figure(figsize=(14, 7))
        plot_slice = slice(0, limit) if limit else slice(None)
        # Ensure index alignment if y_test has datetime index
        idx = y_test.index[plot_slice] if isinstance(y_test.index, pd.DatetimeIndex) else np.arange(len(y_test.values[plot_slice]))

        plt.plot(idx, y_test.values[plot_slice], label='Actual', color='blue', alpha=0.7)
        plt.plot(idx, y_pred[plot_slice], label=f'Predicted ({model_name})', color='red', alpha=0.7)
        plt.legend()
        plt.title(f'Actual vs Predicted - {model_name} ({f"First {limit}" if limit else "All"} Samples)')
        plt.xlabel('Time/Sample Index')
        plt.ylabel(y_test.name or 'Target Value')
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'{prefix}_{model_name}.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Actual vs Predicted plot saved to: {plot_file}")
    except Exception as e:
        logger.error(f"Error plotting actual vs predicted for {model_name}: {e}", exc_info=True)

def plot_feature_importance(importance_series: pd.Series, model_name: str, output_dir: str, prefix: str = "feature_importance"):
    """Plots feature importances."""
    if importance_series.empty:
        logger.warning(f"No importance data to plot for {model_name}")
        return
    logger.info(f"Plotting Feature Importance for {model_name}...")
    try:
        importance_series = importance_series.sort_values(ascending=True)
        plt.figure(figsize=(10, max(6, len(importance_series) * 0.4)))  # Adjust height
        plt.barh(importance_series.index, importance_series.values)
        plt.xlabel('Importance Score')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'{prefix}_{model_name}.png')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Feature importance plot saved to: {plot_file}")
    except Exception as e:
         logger.error(f"Error plotting feature importance for {model_name}: {e}", exc_info=True)

# --- Main Orchestration ---

class IndicatorPredictionService:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.config = CONFIG
        self.output_dir = settings.RESULTS_DIR_PATH
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.output_dir}")

    def run_prediction(self) -> Dict[str, Any]:
        """Main function to run the ML evaluation pipeline."""
        logger.info("=== ML Model Evaluation Script Started ===")

        # 1. Load Data
        df = load_data(self.data_path)
        if df is None:
            return {'status': 'error', 'message': 'Data loading failed'}

        # 2. Calculate Indicators
        df, features = calculate_indicators(df, self.config)
        if not features:
            logger.error("No features were generated from indicators. Exiting.")
            return {'status': 'error', 'message': 'No features generated'}

        # 3. Prepare Features & Target
        try:
            X, y = prepare_features_target(df, features, self.config["TARGET_COLUMN"], self.config["TARGET_SHIFT"])
        except ValueError as e:
            logger.error(f"Error preparing features/target: {e}. Exiting.")
            return {'status': 'error', 'message': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error preparing features/target: {e}", exc_info=True)
            return {'status': 'error', 'message': 'Unexpected error'}

        # 4. Split Data
        X_train, X_test, y_train, y_test = split_data(X, y, self.config["TEST_SIZE"], self.config["SHUFFLE_SPLIT"])

        # 5. Train and Evaluate Models
        models = get_models(self.config)
        all_metrics = {}
        all_importances = {}
        predictions = {}

        for name, model_instance in tqdm(models.items(), desc="Training Models"):
            try:
                metrics, importance, y_pred = train_evaluate_model(
                    name, model_instance, X_train, y_train, X_test, y_test, features, self.config
                )
                all_metrics[name] = metrics
                all_importances[name] = importance  # This is a pd.Series
                predictions[name] = y_pred
            except Exception as e:
                logger.error(f"Failed to train/evaluate model {name}: {e}", exc_info=True)
                all_metrics[name] = {'MSE': float('nan'), 'R2': float('nan')}
                all_importances[name] = pd.Series(dtype=float)  # Empty series on error

        # 6. Save Results
        # Performance Metrics
        perf_df = pd.DataFrame.from_dict(all_metrics, orient='index').reset_index()
        perf_df = perf_df.rename(columns={'index': 'Model'})
        perf_filepath = os.path.join(self.output_dir, "model_performance.csv")
        save_dataframe(perf_df, perf_filepath)

        # Feature Importances (Wide Format)
        # Check if there's any importance data before creating DataFrame
        valid_importances = {name: imp for name, imp in all_importances.items() if not imp.empty}
        if valid_importances:
            imp_df = pd.DataFrame(valid_importances)
            imp_df.index.name = 'Feature'
            imp_filepath = os.path.join(self.output_dir, "feature_importance.csv")
            # Reset index to make 'Feature' a column before saving
            save_dataframe(imp_df.reset_index(), imp_filepath)
        else:
             logger.warning("No valid feature importance data generated to save.")

        # 7. Plotting
        if not perf_df.empty and perf_df['R2'].notna().any():
            # Plot best model predictions
            best_model_name = perf_df.loc[perf_df['R2'].idxmax()]['Model']  # Find best R2
            if self.config.get("PLOT_BEST_PREDICTIONS", False) and best_model_name in predictions:
                plot_actual_vs_predicted(
                    y_test, predictions[best_model_name], best_model_name,
                    self.output_dir, limit=self.config.get("PLOT_SAMPLE_SIZE")
                )

            # Plot feature importances
            if self.config.get("PLOT_FEATURE_IMPORTANCE", False):
                for name, importance_series in all_importances.items():
                     plot_feature_importance(importance_series, name, self.output_dir)
        else:
             logger.warning("Skipping plotting due to missing performance results.")

        logger.info("=== Summary of Model Performance ===")
        try:
             logger.info(perf_df.to_string(index=False))
        except Exception:
            pass  # Avoid error if perf_df is empty

        logger.info("=== ML Model Evaluation Complete ===")
        return {'status': 'success', 'message': 'Model evaluation completed', 'models_evaluated': list(all_metrics.keys())}