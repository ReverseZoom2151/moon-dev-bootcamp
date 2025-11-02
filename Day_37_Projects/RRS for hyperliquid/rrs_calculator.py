# rrs_calculator.py
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

def calculate_rrs(symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Relative Rotation Strength (RRS) metrics for Hyperliquid data.

    Args:
        symbol_df: Processed DataFrame for the symbol (requires columns:
                   'timestamp', 'log_return', 'volatility', 'volume_ratio').
        benchmark_df: Processed DataFrame for the benchmark (requires columns:
                      'timestamp', 'log_return').

    Returns:
        DataFrame with RRS metrics added, or original symbol_df if error/empty.
    """
    required_symbol_cols = ['timestamp', 'log_return', 'volatility', 'volume_ratio']
    required_benchmark_cols = ['timestamp', 'log_return']

    # --- Input Validation ---
    if not all(col in symbol_df.columns for col in required_symbol_cols):
        logger.error(f"Symbol DataFrame missing required columns for RRS calc. Need: {required_symbol_cols}")
        return symbol_df # Return original to indicate failure
    if not all(col in benchmark_df.columns for col in required_benchmark_cols):
        logger.error(f"Benchmark DataFrame missing required columns for RRS calc. Need: {required_benchmark_cols}")
        return symbol_df
    if symbol_df.empty or benchmark_df.empty:
         logger.warning("Input Symbol or Benchmark DataFrame is empty for RRS calc.")
         return symbol_df

    logger.debug(f"Calculating RRS. Input symbol df shape: {symbol_df.shape}, Benchmark df shape: {benchmark_df.shape}")

    try:
        # --- Data Alignment (Timestamp Index) --- 
        # Ensure timestamps are datetime objects first
        symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
        benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])

        df = symbol_df.set_index('timestamp')
        bench_df = benchmark_df.set_index('timestamp')

        # --- Join on Timestamp (Inner Join) --- 
        # Keep only overlapping periods
        df_merged = df.join(bench_df[['log_return']], rsuffix='_benchmark', how='inner')
        logger.debug(f"Shape after inner join: {df_merged.shape}")

        if df_merged.empty:
            logger.warning("No overlapping data after joining symbol and benchmark.")
            # Return empty DF consistent with other modules upon failure
            return pd.DataFrame(columns=symbol_df.columns.tolist() + ['differential_return', 'normalized_return', 'rrs', 'smoothed_rrs'])

        # --- Handle Potential NaNs (introduced by calcs or join) --- 
        initial_len = len(df_merged)
        # Simple ffill/bfill for now, consider more robust imputation if needed
        df_merged = df_merged.ffill().bfill()
        if len(df_merged.dropna()) < initial_len:
            logger.debug("Filled NaNs using ffill/bfill.")
        
        # Drop rows if essential columns are still NaN after filling
        essential_cols = ['log_return', 'log_return_benchmark', 'volatility', 'volume_ratio']
        df_merged.dropna(subset=essential_cols, inplace=True)
        if df_merged.empty:
             logger.warning("DataFrame empty after dropping NaNs in essential columns.")
             return pd.DataFrame(columns=symbol_df.columns.tolist() + ['differential_return', 'normalized_return', 'rrs', 'smoothed_rrs'])

        # --- RRS Calculations --- 
        df_merged['differential_return'] = df_merged['log_return'] - df_merged['log_return_benchmark']
        
        # Avoid division by zero in volatility
        df_merged['normalized_return'] = (df_merged['differential_return'] /
                                         df_merged['volatility'].replace(0, np.nan))
        
        df_merged['rrs'] = df_merged['normalized_return'] * df_merged['volume_ratio']
        df_merged['smoothed_rrs'] = df_merged['rrs'].ewm(span=14, adjust=False).mean()

        # Replace infinities
        df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
        # Optionally drop final NaNs if smoothed_rrs must be valid
        # df_merged.dropna(subset=['smoothed_rrs'], inplace=True)
        
        df_merged.reset_index(inplace=True) # Return timestamp to column
        logger.debug(f"Finished RRS calculation. Final shape: {df_merged.shape}")
        return df_merged

    except Exception as e:
        logger.error(f"Error calculating RRS: {e}", exc_info=True)
        # Return original df or empty df on error?
        # Returning empty is consistent with other modules
        return pd.DataFrame(columns=symbol_df.columns.tolist() + ['differential_return', 'normalized_return', 'rrs', 'smoothed_rrs'])
