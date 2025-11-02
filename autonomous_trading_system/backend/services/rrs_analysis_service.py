# rrs_analysis_service.py
"""
RRS (Relative Rotation Strength) Analysis Service
Implements Day 37 Projects RRS analysis functionality using Birdeye API
"""

import pandas as pd
import numpy as np
import logging
import json
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class RRSDataFetcher:
    """Handles data fetching from Birdeye API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://public-api.birdeye.so/defi/ohlcv"
        
    async def fetch_data(self, address: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Fetches OHLCV data from the Birdeye API for a given token address and time range.
        """
        if not self.api_key:
            logger.error("Birdeye API key is not configured. Cannot fetch data.")
            return pd.DataFrame()

        time_from_ts = int(start_time.timestamp())
        time_to_ts = int(end_time.timestamp())

        url = f"{self.base_url}?address={address}&type={timeframe}&time_from={time_from_ts}&time_to={time_to_ts}"
        headers = {"X-API-KEY": self.api_key}

        logger.info(f"Fetching Birdeye data for {address} [{timeframe}]")
        logger.debug(f"Request URL: {url}")
        logger.debug(f"Time range (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=settings.RRS_REQUEST_TIMEOUT) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    
                    items: Optional[List[Dict]] = json_response.get('data', {}).get('items')

                    if not items:
                        logger.warning(f"No data items received from Birdeye for {address} in the specified range.")
                        return pd.DataFrame()

                    # Process received data
                    processed_data = []
                    required_keys = {'unixTime', 'o', 'h', 'l', 'c', 'v'}
                    for item in items:
                        if not required_keys.issubset(item.keys()):
                            logger.warning(f"Skipping incomplete data item: {item}")
                            continue
                        try:
                            processed_data.append({
                                'timestamp': datetime.utcfromtimestamp(int(item['unixTime'])),
                                'open': float(item['o']),
                                'high': float(item['h']),
                                'low': float(item['l']),
                                'close': float(item['c']),
                                'volume': float(item['v'])
                            })
                        except (ValueError, TypeError, KeyError) as e:
                            logger.warning(f"Error processing data item {item}: {e}. Skipping item.")
                            continue

                    if not processed_data:
                        logger.warning(f"No valid data items could be processed for {address}.")
                        return pd.DataFrame()

                    df = pd.DataFrame(processed_data)
                    df = df.sort_values(by='timestamp').reset_index(drop=True)

                    # Padding: Ensure minimum length for indicator calculations
                    min_rows_required = settings.RRS_MIN_DATA_POINTS
                    if 0 < len(df) < min_rows_required:
                        rows_to_add = min_rows_required - len(df)
                        logger.debug(f"Padding data for {address}: Have {len(df)}, need {min_rows_required}. Adding {rows_to_add} rows.")
                        first_row_df = df.iloc[0:1]
                        padding_df = pd.concat([first_row_df] * rows_to_add, ignore_index=True)
                        df = pd.concat([padding_df, df], ignore_index=True)

                    logger.info(f"Successfully retrieved and processed {len(df)} rows of data for {address}.")
                    return df

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed for {address}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response for {address}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during data fetching for {address}: {e}", exc_info=True)

        return pd.DataFrame()

class RRSDataProcessor:
    """Handles data processing for RRS calculations"""
    
    @staticmethod
    def calculate_returns_and_volatility(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """Calculates log returns and rolling volatility."""
        if window is None:
            window = settings.RRS_VOLATILITY_WINDOW
            
        if 'close' not in df.columns:
            logger.error("'close' column not found in DataFrame for return/volatility calculation.")
            return df

        # Ensure close is numeric and handle potential infinities from zero prices
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)

        # Calculate log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return'] = df['log_return'].replace([np.inf, -np.inf], np.nan)

        # Calculate volatility
        if len(df.dropna(subset=['log_return'])) >= window:
            df['volatility'] = df['log_return'].rolling(window=window, min_periods=window).std()
        else:
            logger.warning(f"Not enough data points ({len(df.dropna(subset=['log_return']))}) for volatility window {window}. Setting volatility to NaN.")
            df['volatility'] = np.nan

        logger.debug("Calculated log returns and volatility.")
        return df

    @staticmethod
    def calculate_volume_metrics(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """Calculates rolling average volume and volume ratio."""
        if window is None:
            window = settings.RRS_VOLUME_WINDOW
            
        if 'volume' not in df.columns:
            logger.error("'volume' column not found in DataFrame for volume metrics calculation.")
            return df

        # Ensure volume is numeric
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # Calculate average volume
        if len(df.dropna(subset=['volume'])) >= window:
            df['average_volume'] = df['volume'].rolling(window=window, min_periods=window).mean()
        else:
            logger.warning(f"Not enough data points ({len(df.dropna(subset=['volume']))}) for average volume window {window}. Setting average_volume to NaN.")
            df['average_volume'] = np.nan

        # Calculate volume ratio
        df['volume_ratio'] = df['volume'] / df['average_volume']
        df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan)

        logger.debug("Calculated volume metrics.")
        return df

class RRSCalculator:
    """Handles RRS calculations"""
    
    @staticmethod
    def calculate_rrs(symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame, smoothing_span: int = None) -> pd.DataFrame:
        """Calculates Relative Rotation Strength (RRS) metrics."""
        if smoothing_span is None:
            smoothing_span = settings.RRS_SMOOTHING_SPAN
            
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

        # Join DataFrames
        df_merged = symbol_df.join(benchmark_df[['log_return']], rsuffix='_benchmark', how='inner')
        logger.debug(f"Shape after inner join: {df_merged.shape}")

        if df_merged.empty:
            logger.warning("No overlapping time periods found between symbol and benchmark after inner join.")
            return pd.DataFrame()

        # Handle missing data
        nan_counts = df_merged.isna().sum()
        if nan_counts.sum() > 0:
            logger.debug(f"NaN counts after join:\n{nan_counts[nan_counts > 0]}")
            df_merged = df_merged.ffill().bfill()
            logger.debug("NaN values filled using ffill then bfill.")

        # Ensure necessary columns are numeric
        cols_to_check = ['log_return', 'log_return_benchmark', 'volatility', 'volume_ratio']
        for col in cols_to_check:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

        df_merged.dropna(subset=cols_to_check, inplace=True)
        if df_merged.empty:
            logger.warning("DataFrame empty after handling NaNs in essential RRS columns.")
            return pd.DataFrame()

        # Calculate RRS Components
        # Differential Return: Symbol's return relative to benchmark
        df_merged['differential_return'] = df_merged['log_return'] - df_merged['log_return_benchmark']

        # Normalized Return: Differential return adjusted for symbol's volatility
        df_merged['normalized_return'] = (df_merged['differential_return'] / 
                                         df_merged['volatility'].replace(0, np.nan))

        # Raw RRS: Normalized return weighted by volume activity
        df_merged['rrs'] = df_merged['normalized_return'] * df_merged['volume_ratio']

        # Smoothed RRS: Exponential moving average of raw RRS
        df_merged['smoothed_rrs'] = df_merged['rrs'].ewm(span=smoothing_span, adjust=False).mean()

        # Replace any infinities
        df_merged = df_merged.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Final calculated df shape: {df_merged.shape}. Columns: {df_merged.columns.tolist()}")

        if df_merged.empty:
            logger.warning("Resulting DataFrame is empty after all RRS calculations!")
            return pd.DataFrame()

        df_merged.reset_index(inplace=True)
        return df_merged

class RRSTopAnalyzer:
    """Handles top RRS analysis across timeframes"""
    
    @staticmethod
    def process_results_files(results_dir: str) -> pd.DataFrame:
        """Processes individual RRS result CSV files to find top performers across timeframes."""
        results_path = Path(results_dir)
        if not results_path.is_dir():
            logger.error(f"Results directory not found: {results_dir}")
            return pd.DataFrame()

        all_top_performers: List[Dict] = []
        files_processed_count = 0
        excluded_files = ['historical_runs.csv', 'Top_RRS.csv']

        logger.info(f"Processing RRS result files in: {results_dir}")

        # Process each CSV file in the results directory
        for file_path in results_path.glob('*.csv'):
            if file_path.name in excluded_files:
                logger.debug(f"Skipping excluded file: {file_path.name}")
                continue

            logger.debug(f"Processing file: {file_path.name}")
            try:
                df = pd.read_csv(file_path)
                files_processed_count += 1

                # Check required columns
                if 'symbol' not in df.columns or 'rrs' not in df.columns:
                    logger.warning(f"Skipping file {file_path.name}: Missing required columns 'symbol' or 'rrs'.")
                    continue

                # Ensure rrs is numeric
                df['rrs'] = pd.to_numeric(df['rrs'], errors='coerce')
                df.dropna(subset=['rrs'], inplace=True)

                if df.empty:
                    logger.warning(f"No valid RRS data found in file {file_path.name} after cleaning. Skipping.")
                    continue

                # Sort by RRS and get top 3
                top_3 = df.nlargest(3, 'rrs')

                # Extract timeframe from filename
                try:
                    timeframe = file_path.stem.split('_')[0]
                except IndexError:
                    logger.warning(f"Could not extract timeframe from filename {file_path.name}. Using 'Unknown'.")
                    timeframe = 'Unknown'

                # Add file info to the results
                for _, row in top_3.iterrows():
                    all_top_performers.append({
                        'timeframe': timeframe,
                        'symbol': row['symbol'],
                        'rrs': row['rrs'],
                        'source_file': file_path.name
                    })

            except pd.errors.EmptyDataError:
                logger.warning(f"Skipping empty file: {file_path.name}")
            except Exception as e:
                logger.error(f"Error processing file {file_path.name}: {e}", exc_info=True)

        if not all_top_performers:
            logger.warning(f"No top performers found after processing {files_processed_count} files.")
            return pd.DataFrame()

        # Create DataFrame from all results
        results_df = pd.DataFrame(all_top_performers)
        results_df = results_df.sort_values('rrs', ascending=False).reset_index(drop=True)

        # Save to CSV
        output_file = results_path / 'Top_RRS.csv'
        try:
            results_df.to_csv(output_file, index=False)
            logger.info(f"🌟 Top RRS Scores Across All Timeframes ({len(results_df)} entries):")
            logger.info('\n' + results_df.to_string())
            logger.info(f"💾 Top RRS results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save Top_RRS.csv to {output_file}: {e}", exc_info=True)

        return results_df

class RRSAnalysisService:
    """Main RRS Analysis Service"""
    
    def __init__(self):
        self.api_key = settings.RRS_BIRDEYE_API_KEY or settings.BIRDEYE_KEY
        self.data_fetcher = RRSDataFetcher(self.api_key) if self.api_key else None
        self.data_processor = RRSDataProcessor()
        self.calculator = RRSCalculator()
        self.top_analyzer = RRSTopAnalyzer()
        self.is_running = False
        
        # Ensure directories exist
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        """Creates the data and results directories if they don't exist."""
        try:
            Path(settings.RRS_DATA_DIR).mkdir(parents=True, exist_ok=True)
            Path(settings.RRS_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured RRS directories exist: {settings.RRS_DATA_DIR}, {settings.RRS_RESULTS_DIR}")
        except OSError as e:
            logger.error(f"Error creating RRS directories: {e}", exc_info=True)
    
    def _save_dataframe(self, df: pd.DataFrame, directory: str, filename: str):
        """Saves a DataFrame to a CSV file with error handling."""
        if df.empty:
            logger.debug(f"Skipping save for empty DataFrame: {filename}")
            return
        
        file_path = Path(directory) / filename
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"💾 Data saved successfully to: {file_path}")
        except OSError as e:
            logger.error(f"Error saving file {file_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving {file_path}: {e}", exc_info=True)
    
    async def fetch_and_process_symbol_data(
        self,
        symbol_name: str,
        symbol_address: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetches and processes data for a single symbol."""
        if not self.data_fetcher:
            logger.error("Data fetcher not initialized - missing API key")
            return None
            
        logger.info(f"--- Processing symbol: {symbol_name} ({symbol_address}) ---")
        df = await self.data_fetcher.fetch_data(symbol_address, timeframe, start_time, end_time)
        
        if df.empty:
            logger.warning(f"No data fetched for {symbol_name}. Skipping.")
            return None
        
        # Log actual data range received
        actual_start = df['timestamp'].min()
        actual_end = df['timestamp'].max()
        logger.debug(f"Symbol {symbol_name} actual data range: {actual_start} to {actual_end}")

        # Process data: Calculate returns, volatility, and volume metrics
        df = self.data_processor.calculate_returns_and_volatility(df)
        df = self.data_processor.calculate_volume_metrics(df)

        # Drop rows with NaNs in essential columns
        essential_cols = ['log_return', 'volatility', 'volume_ratio']
        initial_len = len(df)
        df.dropna(subset=essential_cols, inplace=True)
        if len(df) < initial_len:
            logger.debug(f"Dropped {initial_len - len(df)} rows with NaNs in essential columns for {symbol_name}")

        if df.empty:
            logger.warning(f"No valid processed data for {symbol_name} after cleaning. Skipping.")
            return None

        logger.debug(f"Finished processing for {symbol_name}. Shape: {df.shape}")
        return df
    
    def _update_historical_summary(self, run_summary: pd.DataFrame):
        """Appends the current run summary to the historical CSV file."""
        if not settings.RRS_GENERATE_HISTORICAL_SUMMARY:
            return
            
        historical_file = Path(settings.RRS_RESULTS_DIR) / "historical_runs.csv"
        try:
            if historical_file.exists():
                historical_df = pd.read_csv(historical_file)
                historical_df = pd.concat([historical_df, run_summary], ignore_index=True)
            else:
                historical_df = run_summary
            
            historical_df.to_csv(historical_file, index=False)
            logger.info(f"🌟 Historical Database updated! Total runs: {len(historical_df)}. Location: {historical_file}")
        except Exception as e:
            logger.error(f"Error updating historical file {historical_file}: {e}", exc_info=True)
    
    async def process_timeframe(self, timeframe: str, lookback_days: int, benchmark_symbol: str):
        """Processes RRS calculation for a specific timeframe."""
        logger.info(f"===== Starting processing for timeframe: {timeframe} (Lookback: {lookback_days} days) =====")
        
        # Validate benchmark symbol
        if benchmark_symbol not in settings.RRS_SYMBOLS:
            logger.error(f"Benchmark symbol '{benchmark_symbol}' not found in RRS_SYMBOLS. Cannot proceed.")
            return
        benchmark_address = settings.RRS_SYMBOLS[benchmark_symbol]
        
        # Get symbols excluding the benchmark
        symbols_to_process = {s: addr for s, addr in settings.RRS_SYMBOLS.items() if s != benchmark_symbol}
        if not symbols_to_process:
            logger.warning("No symbols to process (excluding benchmark).")
            return

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)
        logger.info(f"Time range (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Fetch and process benchmark data
        logger.info(f"--- Processing benchmark: {benchmark_symbol} ({benchmark_address}) ---")
        benchmark_df_raw = await self.data_fetcher.fetch_data(benchmark_address, timeframe, start_time, end_time)
        
        if benchmark_df_raw.empty:
            logger.error(f"Failed to fetch benchmark data for {benchmark_symbol}. Cannot proceed with timeframe {timeframe}.")
            return
        
        # Save raw benchmark data if enabled
        if settings.RRS_SAVE_RAW_DATA:
            self._save_dataframe(benchmark_df_raw, settings.RRS_DATA_DIR, f"{benchmark_symbol}_{timeframe}_{lookback_days}d_raw.csv")

        benchmark_df_processed = self.data_processor.calculate_returns_and_volatility(benchmark_df_raw)
        benchmark_df_processed = benchmark_df_processed[['timestamp', 'log_return']].dropna()
        
        if benchmark_df_processed.empty:
            logger.error(f"Benchmark data for {benchmark_symbol} became empty after processing returns. Cannot proceed.")
            return
        logger.debug(f"Processed benchmark data shape: {benchmark_df_processed.shape}")

        # Initialize list to store RRS results for this timeframe
        rrs_results_list: List[Dict] = []

        # Process each symbol
        for symbol_name, symbol_address in symbols_to_process.items():
            symbol_df_processed = await self.fetch_and_process_symbol_data(
                symbol_name, symbol_address, timeframe, start_time, end_time
            )
            
            if symbol_df_processed is None or symbol_df_processed.empty:
                continue

            # Calculate RRS
            rrs_df = self.calculator.calculate_rrs(symbol_df_processed, benchmark_df_processed)
            
            if rrs_df.empty:
                logger.warning(f"No RRS data calculated for {symbol_name}.")
                continue

            # Save the processed data with RRS values if enabled
            if settings.RRS_SAVE_PROCESSED_DATA:
                self._save_dataframe(rrs_df, settings.RRS_DATA_DIR, f"{symbol_name}_{timeframe}_{lookback_days}d_processed.csv")

            # Get the latest smoothed RRS value
            try:
                rrs_df = rrs_df.sort_values(by='timestamp')
                latest_rrs = rrs_df.iloc[-1]['smoothed_rrs']
                if pd.isna(latest_rrs):
                    logger.warning(f"Latest smoothed_rrs for {symbol_name} is NaN. Skipping ranking.")
                    continue
                rrs_results_list.append({'symbol': symbol_name, 'rrs': latest_rrs})
                logger.debug(f"Latest RRS for {symbol_name}: {latest_rrs:.4f}")
            except (IndexError, KeyError) as e:
                logger.error(f"Error getting latest RRS for {symbol_name}: {e}")
                continue

        # Finalize Timeframe Results
        if not rrs_results_list:
            logger.warning(f"No symbols had valid RRS scores for timeframe {timeframe}.")
            return

        # Rank tokens based on RRS
        final_rrs_df = pd.DataFrame(rrs_results_list)
        final_rrs_df.sort_values('rrs', ascending=False, inplace=True)
        logger.info(f"--- RRS Ranking for Timeframe: {timeframe} ---")
        logger.info('\n' + final_rrs_df.to_string())

        # Save current run results
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timeframe}_{lookback_days}d_{benchmark_symbol}_{timestamp_str}.csv"
        self._save_dataframe(final_rrs_df, settings.RRS_RESULTS_DIR, filename)

        # Prepare and update historical summary
        if settings.RRS_GENERATE_HISTORICAL_SUMMARY and len(final_rrs_df) > 0:
            top_performer = final_rrs_df.iloc[0]
            bottom_performer = final_rrs_df.iloc[-1]
            run_summary_data = {
                'run_timestamp_utc': [datetime.utcnow()],
                'timeframe': [timeframe],
                'lookback_days': [lookback_days],
                'benchmark_symbol': [benchmark_symbol],
                'num_symbols_analyzed': [len(symbols_to_process)],
                'num_symbols_ranked': [len(final_rrs_df)],
                'top_performer_symbol': [top_performer['symbol']],
                'top_performer_rrs': [top_performer['rrs']],
                'bottom_performer_symbol': [bottom_performer['symbol']],
                'bottom_performer_rrs': [bottom_performer['rrs']]
            }
            run_summary_df = pd.DataFrame(run_summary_data)
            self._update_historical_summary(run_summary_df)

        # Log summary
        logger.info(f"--- Run Summary for {timeframe} ---")
        logger.info(f"Benchmark: {benchmark_symbol}")
        logger.info(f"Symbols Analyzed: {len(symbols_to_process)}, Symbols Ranked: {len(final_rrs_df)}")
        if len(final_rrs_df) > 0:
            top_performer = final_rrs_df.iloc[0]
            bottom_performer = final_rrs_df.iloc[-1]
            logger.info(f"🏆 Top Performer: {top_performer['symbol']} (RRS: {top_performer['rrs']:.4f})")
            logger.info(f"📉 Bottom Performer: {bottom_performer['symbol']} (RRS: {bottom_performer['rrs']:.4f})")
        logger.info(f"===== Finished processing for timeframe: {timeframe} =====")
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Main function to orchestrate the RRS analysis across all timeframes."""
        if not settings.ENABLE_RRS_ANALYSIS:
            return {"status": "disabled", "message": "RRS Analysis is disabled in configuration"}
            
        if not self.api_key:
            return {"status": "error", "message": "Birdeye API key not configured"}
        
        logger.info("🚀 Starting RRS Analysis...")
        self.is_running = True
        
        try:
            # Configuration
            benchmark_symbol_name = settings.RRS_BENCHMARK_SYMBOL
            lookback = settings.RRS_LOOKBACK_DAYS
            timeframes_to_run = settings.RRS_TIMEFRAMES

            logger.info(f"Configuration: Benchmark='{benchmark_symbol_name}', Lookback={lookback} days")
            logger.info(f"Timeframes to analyze: {timeframes_to_run}")
            logger.info(f"Number of symbols configured (incl. benchmark): {len(settings.RRS_SYMBOLS)}")

            # Run Analysis for Each Timeframe
            for tf in timeframes_to_run:
                await self.process_timeframe(tf, lookback, benchmark_symbol_name)

            # Generate Consolidated Report
            logger.info("===== Generating Final Top RRS Report across all timeframes =====")
            try:
                top_performers_all = self.top_analyzer.process_results_files(settings.RRS_RESULTS_DIR)

                if not top_performers_all.empty:
                    logger.info("--- Overall RRS Score Distribution (Top Performers) ---")
                    logger.info(f"Max RRS: {top_performers_all['rrs'].max():.4f}")
                    logger.info(f"Avg RRS: {top_performers_all['rrs'].mean():.4f}")
                    logger.info(f"Min RRS: {top_performers_all['rrs'].min():.4f}")
                    logger.info(f"Count: {len(top_performers_all)}")
                else:
                    logger.warning("Consolidated Top RRS report is empty.")

            except Exception as e:
                logger.error(f"Failed to generate final Top RRS report: {e}", exc_info=True)

            # Interpretation Guide
            logger.info("--- RRS Score Interpretation Guide ---")
            logger.info(f" > {settings.RRS_EXCEPTIONAL_THRESHOLD:.1f} : Exceptional Strength 🚀")
            logger.info(f"{settings.RRS_STRONG_THRESHOLD:.1f}-{settings.RRS_EXCEPTIONAL_THRESHOLD:.1f}: Strong Strength 💪")
            logger.info(f"{settings.RRS_MODERATE_THRESHOLD:.1f}-{settings.RRS_STRONG_THRESHOLD:.1f}: Moderate Strength 📈")
            logger.info(f"{settings.RRS_WEAK_THRESHOLD:.1f}-{settings.RRS_MODERATE_THRESHOLD:.1f}: Weak Strength 😐")
            logger.info(f" < {settings.RRS_WEAK_THRESHOLD:.1f} : Underperforming 📉")
            logger.info("🏁 RRS Analysis Finished.")
            
            return {
                "status": "completed",
                "message": "RRS analysis completed successfully",
                "timeframes_processed": timeframes_to_run,
                "benchmark_symbol": benchmark_symbol_name,
                "lookback_days": lookback,
                "symbols_count": len(settings.RRS_SYMBOLS)
            }
            
        except Exception as e:
            logger.error(f"Error during RRS analysis: {e}", exc_info=True)
            return {"status": "error", "message": f"RRS analysis failed: {str(e)}"}
        finally:
            self.is_running = False
    
    async def get_latest_rrs_rankings(self, timeframe: str = None) -> Dict[str, Any]:
        """Get the latest RRS rankings for a specific timeframe or all timeframes."""
        try:
            results_path = Path(settings.RRS_RESULTS_DIR)
            if not results_path.exists():
                return {"status": "error", "message": "Results directory not found"}
            
            # Get the most recent results file(s)
            if timeframe:
                pattern = f"{timeframe}_*.csv"
            else:
                pattern = "*.csv"
            
            files = list(results_path.glob(pattern))
            files = [f for f in files if f.name not in ['historical_runs.csv', 'Top_RRS.csv']]
            
            if not files:
                return {"status": "error", "message": f"No results files found for pattern: {pattern}"}
            
            # Sort by modification time and get the most recent
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if timeframe:
                # Return specific timeframe results
                latest_file = files[0]
                df = pd.read_csv(latest_file)
                return {
                    "status": "success",
                    "timeframe": timeframe,
                    "file": latest_file.name,
                    "rankings": df.to_dict('records')
                }
            else:
                # Return consolidated top performers
                top_rrs_file = results_path / 'Top_RRS.csv'
                if top_rrs_file.exists():
                    df = pd.read_csv(top_rrs_file)
                    return {
                        "status": "success",
                        "type": "consolidated",
                        "rankings": df.to_dict('records')
                    }
                else:
                    return {"status": "error", "message": "Top_RRS.csv not found"}
                    
        except Exception as e:
            logger.error(f"Error getting RRS rankings: {e}", exc_info=True)
            return {"status": "error", "message": f"Failed to get rankings: {str(e)}"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the RRS analysis service."""
        return {
            "service": "RRS Analysis Service",
            "enabled": settings.ENABLE_RRS_ANALYSIS,
            "running": self.is_running,
            "api_key_configured": bool(self.api_key),
            "symbols_count": len(settings.RRS_SYMBOLS),
            "timeframes": settings.RRS_TIMEFRAMES,
            "benchmark_symbol": settings.RRS_BENCHMARK_SYMBOL,
            "lookback_days": settings.RRS_LOOKBACK_DAYS,
            "data_dir": settings.RRS_DATA_DIR,
            "results_dir": settings.RRS_RESULTS_DIR,
            "interpretation_thresholds": {
                "exceptional": settings.RRS_EXCEPTIONAL_THRESHOLD,
                "strong": settings.RRS_STRONG_THRESHOLD,
                "moderate": settings.RRS_MODERATE_THRESHOLD,
                "weak": settings.RRS_WEAK_THRESHOLD
            }
        }
    
    def interpret_rrs_score(self, score: float) -> str:
        """Interpret an RRS score based on configured thresholds."""
        if score > settings.RRS_EXCEPTIONAL_THRESHOLD:
            return "Exceptional Strength 🚀"
        elif score > settings.RRS_STRONG_THRESHOLD:
            return "Strong Strength 💪"
        elif score > settings.RRS_MODERATE_THRESHOLD:
            return "Moderate Strength 📈"
        elif score > settings.RRS_WEAK_THRESHOLD:
            return "Weak Strength 😐"
        else:
            return "Underperforming 📉"

# Global service instance
rrs_service = RRSAnalysisService()
