# rrs_calculator.py
import pandas as pd
import logging
import numpy as np # Import numpy for infinity handling

logger = logging.getLogger(__name__)

def calculate_rrs(symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Relative Rotation Strength (RRS) metrics.

    Aligns the symbol data with the benchmark data on timestamp and calculates
    differential return, normalized return, raw RRS, and smoothed RRS.

    Args:
        symbol_df: DataFrame for the symbol, must include 'timestamp', 'log_return',
                   'volatility', and 'volume_ratio' columns.
        benchmark_df: DataFrame for the benchmark, must include 'timestamp' and
                      'log_return' columns.

    Returns:
        DataFrame with RRS metrics added, or an empty DataFrame if calculation fails
        or no overlapping data exists.
    """
    required_symbol_cols = ['timestamp', 'log_return', 'volatility', 'volume_ratio']
    required_benchmark_cols = ['timestamp', 'log_return']

    if not all(col in symbol_df.columns for col in required_symbol_cols):
        logger.error(f"Symbol DataFrame missing required columns. Need: {required_symbol_cols}, Have: {symbol_df.columns.tolist()}")
        return pd.DataFrame()
    if not all(col in benchmark_df.columns for col in required_benchmark_cols):
        logger.error(f"Benchmark DataFrame missing required columns. Need: {required_benchmark_cols}, Have: {benchmark_df.columns.tolist()}")
        return pd.DataFrame()

    logger.debug(f"Calculating RRS. Input symbol df shape: {symbol_df.shape}, Benchmark df shape: {benchmark_df.shape}")

    # Ensure timestamps are datetime objects
    symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
    benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])

    # Align timestamps by setting them as index
    symbol_df = symbol_df.set_index('timestamp')
    benchmark_df = benchmark_df.set_index('timestamp')

    logger.debug(f"After setting index - symbol df shape: {symbol_df.shape}, benchmark df shape: {benchmark_df.shape}")

    # --- Join DataFrames ---
    # Use inner join to keep only timestamps present in BOTH dataframes
    # This ensures we compare the symbol and benchmark over the exact same periods.
    # Select only the benchmark's log return to avoid column name conflicts.
    df_merged = symbol_df.join(benchmark_df[['log_return']], rsuffix='_benchmark', how='inner')
    logger.debug(f"Shape after inner join: {df_merged.shape}")

    if df_merged.empty:
        logger.warning("No overlapping time periods found between symbol and benchmark after inner join.")
        return pd.DataFrame()

    # --- Handle Missing Data --- 
    # Check for NaNs introduced by calculations or join
    nan_counts = df_merged.isna().sum()
    if nan_counts.sum() > 0:
        logger.debug(f"NaN counts after join:\n{nan_counts[nan_counts > 0]}")
        # Forward fill first to carry forward last known value, then backfill for any remaining at the start
        # Consider the implications: This assumes the previous value is the best estimate for a missing point.
        df_merged = df_merged.ffill().bfill()
        logger.debug("NaN values filled using ffill then bfill.")

    # --- Calculate RRS Components --- 
    # Ensure necessary columns are numeric after potential filling
    cols_to_check = ['log_return', 'log_return_benchmark', 'volatility', 'volume_ratio']
    for col in cols_to_check:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    df_merged.dropna(subset=cols_to_check, inplace=True) # Drop rows if essential cols are still NaN
    if df_merged.empty:
        logger.warning("DataFrame empty after handling NaNs in essential RRS columns.")
        return pd.DataFrame()

    # Differential Return: Symbol's return relative to benchmark
    df_merged['differential_return'] = df_merged['log_return'] - df_merged['log_return_benchmark']

    # Normalized Return: Differential return adjusted for symbol's volatility
    # Handle potential division by zero if volatility is 0
    df_merged['normalized_return'] = (df_merged['differential_return'] /
                                     df_merged['volatility'].replace(0, np.nan))

    # Raw RRS: Normalized return weighted by volume activity
    df_merged['rrs'] = df_merged['normalized_return'] * df_merged['volume_ratio']

    # Smoothed RRS: Exponential moving average of raw RRS for trend identification
    # Using span=14 is common, adjust=False aligns with typical TA library behavior.
    df_merged['smoothed_rrs'] = df_merged['rrs'].ewm(span=14, adjust=False).mean()

    # Replace any infinities that might have resulted from divisions
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan)

    # Drop any final rows with NaN in critical RRS columns (optional, depends on need)
    final_nan_counts = df_merged[['differential_return', 'normalized_return', 'rrs', 'smoothed_rrs']].isna().sum()
    if final_nan_counts.sum() > 0:
        logger.debug(f"NaN counts in final RRS columns:\n{final_nan_counts[final_nan_counts > 0]}")
        # df_merged = df_merged.dropna(subset=['smoothed_rrs']) # Optional: Keep only rows with valid smoothed_rrs

    logger.debug(f"Final calculated df shape: {df_merged.shape}. Columns: {df_merged.columns.tolist()}")

    if df_merged.empty:
        logger.warning("Resulting DataFrame is empty after all RRS calculations!")
        return pd.DataFrame()

    df_merged.reset_index(inplace=True) # Convert timestamp index back to column
    return df_merged
