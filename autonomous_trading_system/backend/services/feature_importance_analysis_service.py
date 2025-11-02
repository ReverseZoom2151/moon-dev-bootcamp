import pandas as pd
import os
import logging
from typing import Dict, Optional, Any

# Fallback for settings if import fails
try:
    from backend.core.config import settings
except ImportError:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_DIR_PATH = os.path.join(BASE_DIR, 'ml', 'results')
    settings = type('Settings', (), {'RESULTS_DIR_PATH': RESULTS_DIR_PATH})()

# --- Configuration ---
CONFIG = {
    # --- Filtering ---
    'R2_FILTER_THRESHOLD': 0.95,  # Features with R2 score *above* this are filtered out

    # --- Top N Features ---
    'TOP_N_FEATURES': 50,
    'METRICS_TO_SORT': {
        # Metric Name: (Ascending Sort Order?, Output Suffix)
        'Importance': (False, 'top_50_importance'),
        'MSE': (True, 'top_50_mse'),
        'R2': (False, 'top_50_r2')
    }
}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Functions ---

def load_importance_data(filepath: str) -> Optional[pd.DataFrame]:
    """Loads the feature importance data from a CSV file."""
    if not os.path.exists(filepath):
        logger.error(f"Input file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
        # Basic validation
        required_cols = {'Model', 'Importance', 'MSE', 'R2'}
        if not required_cols.issubset(df.columns):
             missing = required_cols - set(df.columns)
             logger.error(f"Missing required columns in {filepath}: {missing}")
             return None
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Input file is empty: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading input file {filepath}: {e}", exc_info=True)
        return None

def filter_features(df: pd.DataFrame, r2_threshold: float) -> pd.DataFrame:
    """Filters out features with R2 scores above the specified threshold."""
    initial_count = len(df)
    # Keep rows where R2 is less than or equal to the threshold
    filtered_df = df[df['R2'] <= r2_threshold].copy()
    removed_count = initial_count - len(filtered_df)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} features with R2 > {r2_threshold}")
    return filtered_df

def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Saves a DataFrame to a CSV file, ensuring the directory exists."""
    try:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to: {filepath}")
    except IOError as e:
        logger.error(f"Error saving file {filepath}: {e}", exc_info=True)
    except Exception as e:
         logger.error(f"Unexpected error saving file {filepath}: {e}", exc_info=True)

def process_model_features(model_name: str, model_data: pd.DataFrame, config: Dict, results_dir_path: str) -> None:
    """Filters, sorts, and saves top features for a single model based on configured metrics."""
    logger.info(f"--- Processing model: {model_name} ({len(model_data)} features initially) ---")

    # 1. Filter potentially overfit features
    filtered_data = filter_features(model_data, config["R2_FILTER_THRESHOLD"])

    if filtered_data.empty:
        logger.warning(f"No features remaining for model '{model_name}' after R2 filtering. Skipping.")
        return

    # 2. Sort and save based on specified metrics
    top_n = config["TOP_N_FEATURES"]
    metrics_dict = config["METRICS_TO_SORT"]

    for metric, (ascending, suffix) in metrics_dict.items():
        if metric not in filtered_data.columns:
             logger.warning(f"Metric '{metric}' not found for model '{model_name}'. Skipping sort.")
             continue

        logger.info(f"Sorting '{model_name}' by '{metric}' (ascending={ascending})...")
        # Sort and get top N
        sorted_df = filtered_data.sort_values(metric, ascending=ascending).head(top_n)

        # Construct filename and save
        output_filename = f"{model_name}_{suffix}.csv"
        output_filepath = os.path.join(results_dir_path, output_filename)
        save_dataframe(sorted_df, output_filepath)

    logger.info(f"--- Finished processing model: {model_name} ---")

# --- Main Class ---

class FeatureImportanceAnalysisService:
    def __init__(self, importance_filepath: str):
        self.importance_filepath = importance_filepath
        self.config = CONFIG
        self.results_dir_path = settings.RESULTS_DIR_PATH
        os.makedirs(self.results_dir_path, exist_ok=True)
        logger.info(f"Using results directory: {self.results_dir_path}")

    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Main function to load, process, and save top features for all models."""
        logger.info("--- Starting Feature Importance Analysis ---")

        # Load data
        df = load_importance_data(self.importance_filepath)
        if df is None:
            logger.error("Analysis stopped due to data loading errors.")
            return {'status': 'error', 'message': 'Data loading failed'}

        # Get unique models and process each
        models = df['Model'].unique()
        logger.info(f"Found {len(models)} unique models to process: {models.tolist()}")

        for model in models:
            model_df = df[df['Model'] == model].copy()  # Work on a copy for safety
            process_model_features(model, model_df, self.config, self.results_dir_path)

        logger.info("--- Feature Importance Analysis Finished ---")
        return {'status': 'success', 'message': 'Feature importance analysis completed', 'models_processed': models.tolist()} 