#!/usr/bin/env python3
"""
Analyzes feature importance results from a CSV file, filters potentially
overfit features, and saves the top N features based on different metrics
(Importance, MSE, R2) for each model.
"""

import pandas as pd
import os
import logging
from typing import Dict, Optional

# --- Configuration ---
# Get the absolute path of the directory where the script resides
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to the script directory
ml_results_dir = os.path.join(script_dir, "ml", "results")

CONFIG = {
    # --- Paths ---
    # Define the absolute path for results to avoid ambiguity
    "RESULTS_DIR_PATH": ml_results_dir,
    "IMPORTANCE_FILENAME": "feature_importance.csv", # Filename within RESULTS_DIR_PATH

    # --- Filtering ---
    "R2_FILTER_THRESHOLD": 0.95, # Features with R2 score *above* this are filtered out

    # --- Top N Features ---
    "TOP_N_FEATURES": 50,
    "METRICS_TO_SORT": {
        # Metric Name: (Ascending Sort Order?, Output Suffix)
        "Importance": (False, "top_50_importance"),
        "MSE": (True, "top_50_mse"),
        "R2": (False, "top_50_r2")
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


def process_model_features(model_name: str, model_data: pd.DataFrame, config: Dict) -> None:
    """Filters, sorts, and saves top features for a single model based on configured metrics."""
    logger.info(f"--- Processing model: {model_name} ({len(model_data)} features initially) ---")

    # 1. Filter potentially overfit features
    filtered_data = filter_features(model_data, config["R2_FILTER_THRESHOLD"])

    if filtered_data.empty:
        logger.warning(f"No features remaining for model '{model_name}' after R2 filtering. Skipping.")
        return

    # 2. Sort and save based on specified metrics
    results_dir_path = config["RESULTS_DIR_PATH"] # Use the absolute path from config
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


def analyze_feature_importance(config: Dict) -> None:
    """Main function to load, process, and save top features for all models."""
    logger.info("--- Starting Feature Importance Analysis ---")

    # Construct input file path using the absolute path from config
    results_dir_path = config["RESULTS_DIR_PATH"]
    input_filepath = os.path.join(results_dir_path, config["IMPORTANCE_FILENAME"])
    logger.info(f"Using input importance file: {input_filepath}")

    # Load data
    df = load_importance_data(input_filepath)
    if df is None:
        logger.error("Analysis stopped due to data loading errors.")
        return

    # Ensure output directory exists (although save_dataframe also checks)
    os.makedirs(results_dir_path, exist_ok=True)

    # Get unique models and process each
    models = df['Model'].unique()
    logger.info(f"Found {len(models)} unique models to process: {models.tolist()}")

    for model in models:
        model_df = df[df['Model'] == model].copy() # Work on a copy for safety
        process_model_features(model, model_df, config)

    logger.info("--- Feature Importance Analysis Finished ---")

# --- Main Execution ---
if __name__ == "__main__":
    # Note: Using __file__ assumes the script is run directly.
    # If imported, this might behave differently. Consider passing paths explicitly if needed.
    try:
        # Recalculate paths within the main block to ensure context
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ml_results_dir = os.path.join(script_dir, "ml", "results")
        # Ensure the CONFIG reflects the calculated path
        CONFIG["RESULTS_DIR_PATH"] = ml_results_dir
        analyze_feature_importance(CONFIG)
    except NameError:
        # Handle cases where __file__ is not defined (e.g., interactive session)
        logger.error("Could not determine script directory automatically using __file__.")
        logger.error("Please ensure CONFIG['RESULTS_DIR_PATH'] is set correctly or run as a script.")
        # Optionally, try a fallback or exit
        # if "RESULTS_DIR_PATH" in CONFIG and os.path.exists(CONFIG["RESULTS_DIR_PATH"]):
        #    logger.info(f"Attempting to proceed with configured path: {CONFIG['RESULTS_DIR_PATH']}")
        #    analyze_feature_importance(CONFIG)
        # else:
        #    logger.error("Cannot proceed without a valid results directory path.")
