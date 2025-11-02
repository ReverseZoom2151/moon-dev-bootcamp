# hyperliquid_rrs_analysis_service.py
"""
Hyperliquid RRS (Relative Rotation Strength) Analysis Service
Implements Day 37 RRS for Hyperliquid project functionality

This service provides comprehensive RRS analysis using Hyperliquid API data,
including data fetching, processing, RRS calculation, and top performer analysis.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import aiohttp
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class TradingConfig:
    """Trading configuration for RRS bot"""
    USDC_SIZE: float = 10.0
    LEVERAGE: int = 3
    SLEEP_SECONDS: int = 30
    RRS_CACHE_TIMEOUT: int = 30
    TP: float = 10.0
    SL: float = -10.0
    AUTO_ADJUST_LEVERAGE: bool = True
    LIMIT_ORDER_BUFFER: float = 0.001


class HyperliquidRRSDataFetcher:
    """Handles data fetching from Hyperliquid API"""
    
    def __init__(self):
        self.api_url = settings.HYPERLIQUID_RRS_API_URL
        self.api_headers = settings.HYPERLIQUID_RRS_API_HEADERS
        self.timeout = settings.HYPERLIQUID_RRS_REQUEST_TIMEOUT
        self.max_retries = settings.HYPERLIQUID_RRS_RETRY_ATTEMPTS
        self.retry_delay = settings.HYPERLIQUID_RRS_RETRY_DELAY
        self.max_call_limit = settings.HYPERLIQUID_RRS_MAX_CALL_LIMIT
    
    def interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        try:
            if isinstance(interval, int):
                return interval
            
            multiplier = int(interval[:-1])
            unit = interval[-1].lower()
            
            if unit == 'm':
                minutes = multiplier
            elif unit == 'h':
                minutes = multiplier * 60
            elif unit == 'd':
                minutes = multiplier * 1440
            else:
                raise ValueError(f"Unknown interval unit: {unit}")
            
            return minutes
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Failed to parse interval '{interval}': {e}")
            return 15  # Default to 15 minutes
    
    def parse_ohlcv_data(self, snapshot_data: List[Dict], symbol: str) -> pd.DataFrame:
        """Parse OHLCV snapshot data from Hyperliquid into DataFrame"""
        if not snapshot_data:
            logger.warning(f"No snapshot data provided for {symbol}")
            return pd.DataFrame()
        
        logger.debug(f"Parsing {len(snapshot_data)} candle snapshots for {symbol}")
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        required_keys = {'t', 'o', 'h', 'l', 'c', 'v'}
        
        for snapshot in snapshot_data:
            if not required_keys.issubset(snapshot.keys()):
                logger.warning(f"Skipping incomplete snapshot for {symbol}: {snapshot}")
                continue
            
            try:
                timestamp = datetime.utcfromtimestamp(int(snapshot['t'] / 1000))
                data.append([
                    timestamp,
                    float(snapshot['o']),
                    float(snapshot['h']),
                    float(snapshot['l']),
                    float(snapshot['c']),
                    float(snapshot['v'])
                ])
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing snapshot for {symbol}: {e}")
                continue
        
        if not data:
            logger.warning(f"No valid snapshots processed for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=columns)
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        logger.debug(f"Parsed DataFrame shape for {symbol}: {df.shape}")
        return df
    
    async def get_ohlcv_chunk(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> Optional[List[Dict]]:
        """Fetch a single chunk of OHLCV data from Hyperliquid API"""
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000)
            }
        }
        
        logger.debug(f"Requesting chunk for {symbol} ({interval})")
        logger.debug(f"Time range: {start_time} to {end_time}")
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        headers=self.api_headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        
                        if 'application/json' not in response.headers.get('Content-Type', ''):
                            logger.error(f"Unexpected content type: {response.headers.get('Content-Type')}")
                            raise aiohttp.ClientError("Non-JSON response received")
                        
                        data = await response.json()
                        
                        if isinstance(data, list):
                            logger.debug(f"Received {len(data)} data points")
                            return data
                        else:
                            logger.error(f"Unexpected JSON structure for {symbol}: {data}")
                            return None
                            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries} for {symbol}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}/{self.max_retries} for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        logger.error(f"Failed to fetch data for {symbol} after {self.max_retries} attempts")
        return None
    
    async def fetch_data(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch historical OHLCV data with pagination handling"""
        logger.info(f"Fetching data for {symbol} ({interval}) from {start_time} to {end_time}")
        all_dataframes: List[pd.DataFrame] = []
        
        try:
            interval_minutes = self.interval_to_minutes(interval)
            if interval_minutes <= 0:
                raise ValueError("Interval minutes must be positive")
            
            chunk_delta = timedelta(minutes=interval_minutes * self.max_call_limit)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid interval '{interval}': {e}")
            return pd.DataFrame()
        
        current_chunk_end_time = end_time
        iteration = 0
        
        while current_chunk_end_time > start_time:
            iteration += 1
            current_chunk_start_time = max(start_time, current_chunk_end_time - chunk_delta)
            
            logger.debug(f"Iteration {iteration}: Fetching chunk from {current_chunk_start_time} to {current_chunk_end_time}")
            
            raw_data_chunk = await self.get_ohlcv_chunk(symbol, interval, current_chunk_start_time, current_chunk_end_time)
            
            if raw_data_chunk is not None:
                df_chunk = self.parse_ohlcv_data(raw_data_chunk, symbol)
                if not df_chunk.empty:
                    all_dataframes.append(df_chunk)
                    logger.debug(f"Added chunk with {len(df_chunk)} rows")
                else:
                    logger.warning(f"Empty chunk for {symbol}")
            else:
                logger.error(f"Failed to fetch chunk for {symbol}")
                break
            
            current_chunk_end_time = current_chunk_start_time - timedelta(minutes=interval_minutes)
            await asyncio.sleep(0.5)  # Be polite to the API
        
        if not all_dataframes:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()
        
        try:
            full_df = pd.concat(all_dataframes)
            full_df = full_df.drop_duplicates(subset='timestamp')
            full_df = full_df.sort_values(by='timestamp').reset_index(drop=True)
            logger.info(f"Successfully fetched data for {symbol}. Shape: {full_df.shape}")
            return full_df
        except Exception as e:
            logger.error(f"Error concatenating data for {symbol}: {e}")
            return pd.DataFrame()


class HyperliquidRRSDataProcessor:
    """Handles data processing for RRS calculations"""
    
    @staticmethod
    def calculate_returns_and_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate log returns and rolling volatility"""
        if 'close' not in df.columns:
            logger.error("'close' column not found for returns calculation")
            return df
        
        df = df.copy()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return'] = df['log_return'].replace([np.inf, -np.inf], np.nan)
        
        if len(df.dropna(subset=['log_return'])) >= window:
            df['volatility'] = df['log_return'].rolling(window=window, min_periods=window).std()
        else:
            logger.warning(f"Not enough data points for volatility window {window}")
            df['volatility'] = np.nan
        
        logger.debug("Calculated log returns and volatility")
        return df
    
    @staticmethod
    def calculate_volume_metrics(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate rolling average volume and volume ratio"""
        if 'volume' not in df.columns:
            logger.error("'volume' column not found for volume metrics")
            return df
        
        df = df.copy()
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        if len(df.dropna(subset=['volume'])) >= window:
            df['average_volume'] = df['volume'].rolling(window=window, min_periods=window).mean()
        else:
            logger.warning(f"Not enough data points for volume window {window}")
            df['average_volume'] = np.nan
        
        df['volume_ratio'] = df['volume'] / df['average_volume']
        df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan)
        
        logger.debug("Calculated volume metrics")
        return df


class HyperliquidRRSCalculator:
    """Handles RRS calculations"""
    
    def calculate_rrs(self, symbol_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Rotation Strength metrics"""
        required_symbol_cols = ['timestamp', 'log_return', 'volatility', 'volume_ratio']
        required_benchmark_cols = ['timestamp', 'log_return']
        
        # Input validation
        if not all(col in symbol_df.columns for col in required_symbol_cols):
            logger.error(f"Symbol DataFrame missing required columns: {required_symbol_cols}")
            return symbol_df
        
        if not all(col in benchmark_df.columns for col in required_benchmark_cols):
            logger.error(f"Benchmark DataFrame missing required columns: {required_benchmark_cols}")
            return symbol_df
        
        if symbol_df.empty or benchmark_df.empty:
            logger.warning("Empty DataFrame provided for RRS calculation")
            return symbol_df
        
        logger.debug(f"Calculating RRS. Symbol shape: {symbol_df.shape}, Benchmark shape: {benchmark_df.shape}")
        
        try:
            # Ensure timestamps are datetime objects
            symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp'])
            benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
            
            df = symbol_df.set_index('timestamp')
            bench_df = benchmark_df.set_index('timestamp')
            
            # Join on timestamp (inner join for overlapping periods)
            df_merged = df.join(bench_df[['log_return']], rsuffix='_benchmark', how='inner')
            logger.debug(f"Shape after inner join: {df_merged.shape}")
            
            if df_merged.empty:
                logger.warning("No overlapping data after joining")
                return pd.DataFrame(columns=symbol_df.columns.tolist() + ['differential_return', 'normalized_return', 'rrs', 'smoothed_rrs'])
            
            # Handle NaNs
            initial_len = len(df_merged)
            df_merged = df_merged.ffill().bfill()
            
            essential_cols = ['log_return', 'log_return_benchmark', 'volatility', 'volume_ratio']
            df_merged.dropna(subset=essential_cols, inplace=True)
            
            if df_merged.empty:
                logger.warning("DataFrame empty after dropping NaNs")
                return pd.DataFrame(columns=symbol_df.columns.tolist() + ['differential_return', 'normalized_return', 'rrs', 'smoothed_rrs'])
            
            # RRS calculations
            df_merged['differential_return'] = df_merged['log_return'] - df_merged['log_return_benchmark']
            df_merged['normalized_return'] = (df_merged['differential_return'] / 
                                            df_merged['volatility'].replace(0, np.nan))
            df_merged['rrs'] = df_merged['normalized_return'] * df_merged['volume_ratio']
            df_merged['smoothed_rrs'] = df_merged['rrs'].ewm(span=14, adjust=False).mean()
            
            # Replace infinities
            df_merged = df_merged.replace([np.inf, -np.inf], np.nan)
            df_merged.reset_index(inplace=True)
            
            logger.debug(f"Finished RRS calculation. Final shape: {df_merged.shape}")
            return df_merged
            
        except Exception as e:
            logger.error(f"Error calculating RRS: {e}", exc_info=True)
            return pd.DataFrame(columns=symbol_df.columns.tolist() + ['differential_return', 'normalized_return', 'rrs', 'smoothed_rrs'])


class HyperliquidRRSTopAnalyzer:
    """Handles top RRS analysis across timeframes"""
    
    def process_results_files(self, results_dir: str) -> pd.DataFrame:
        """Process individual RRS result CSV files to find top performers"""
        results_path = Path(results_dir)
        if not results_path.is_dir():
            logger.error(f"Results directory not found: {results_dir}")
            return pd.DataFrame()
        
        all_top_performers: List[Dict] = []
        files_processed_count = 0
        excluded_files = ['historical_runs.csv', 'Top_RRS.csv']
        
        logger.info(f"Processing RRS result files in: {results_dir}")
        
        for file_path in results_path.glob('*.csv'):
            if file_path.name in excluded_files:
                logger.debug(f"Skipping excluded file: {file_path.name}")
                continue
            
            # Skip Birdeye results if they exist
            if 'So111111' in file_path.name:
                logger.debug(f"Skipping Birdeye result file: {file_path.name}")
                continue
            
            logger.debug(f"Processing file: {file_path.name}")
            try:
                df = pd.read_csv(file_path)
                files_processed_count += 1
                
                if 'symbol' not in df.columns or 'rrs' not in df.columns:
                    logger.warning(f"Skipping file {file_path.name}: Missing required columns")
                    continue
                
                df['rrs'] = pd.to_numeric(df['rrs'], errors='coerce')
                df.dropna(subset=['rrs'], inplace=True)
                
                if df.empty:
                    logger.warning(f"No valid RRS data in {file_path.name}")
                    continue
                
                top_3 = df.nlargest(3, 'rrs')
                
                try:
                    timeframe = file_path.stem.split('_')[0]
                except IndexError:
                    logger.warning(f"Could not extract timeframe from {file_path.name}")
                    timeframe = 'Unknown'
                
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
                logger.error(f"Error processing file {file_path.name}: {e}")
        
        if not all_top_performers:
            logger.warning(f"No top performers found after processing {files_processed_count} files")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(all_top_performers)
        results_df = results_df.sort_values('rrs', ascending=False).reset_index(drop=True)
        
        output_file = results_path / 'Top_RRS.csv'
        try:
            results_df.to_csv(output_file, index=False)
            logger.info(f"Top RRS results saved to: {output_file}")
            logger.info(f"Top RRS Scores ({len(results_df)} entries):")
            logger.info('\n' + results_df.to_string())
        except Exception as e:
            logger.error(f"Failed to save Top_RRS.csv: {e}")
        
        return results_df


class HyperliquidRRSAnalysisService:
    """Main service for Hyperliquid RRS analysis"""
    
    def __init__(self):
        self.data_fetcher = HyperliquidRRSDataFetcher()
        self.data_processor = HyperliquidRRSDataProcessor()
        self.calculator = HyperliquidRRSCalculator()
        self.top_analyzer = HyperliquidRRSTopAnalyzer()
        
        # Ensure directories exist
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self):
        """Create necessary directories"""
        try:
            Path(settings.HYPERLIQUID_RRS_DATA_DIR).mkdir(parents=True, exist_ok=True)
            Path(settings.HYPERLIQUID_RRS_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            Path(settings.HYPERLIQUID_RRS_BOT_DATA_DIR).mkdir(parents=True, exist_ok=True)
            logger.info("Ensured all RRS directories exist")
        except OSError as e:
            logger.error(f"Error creating directories: {e}")
    
    def _save_dataframe(self, df: pd.DataFrame, directory: str, filename: str):
        """Save DataFrame to CSV with error handling"""
        if df.empty:
            logger.debug(f"Skipping save for empty DataFrame: {filename}")
            return
        
        file_path = Path(directory) / filename
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
    
    async def fetch_and_process_symbol_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch and process data for a single symbol"""
        logger.info(f"Processing symbol: {symbol}")
        df = await self.data_fetcher.fetch_data(symbol, timeframe, start_time, end_time)
        
        if df.empty:
            logger.warning(f"No data fetched for {symbol}")
            return None
        
        actual_start = df['timestamp'].min()
        actual_end = df['timestamp'].max()
        logger.debug(f"Symbol {symbol} data range: {actual_start} to {actual_end}")
        
        df = self.data_processor.calculate_returns_and_volatility(df)
        df = self.data_processor.calculate_volume_metrics(df)
        
        essential_cols = ['log_return', 'volatility', 'volume_ratio']
        initial_len = len(df)
        df.dropna(subset=essential_cols, inplace=True)
        
        if len(df) < initial_len:
            logger.debug(f"Dropped {initial_len - len(df)} rows with NaNs for {symbol}")
        
        if df.empty:
            logger.warning(f"No valid processed data for {symbol}")
            return None
        
        logger.debug(f"Finished processing {symbol}. Shape: {df.shape}")
        return df
    
    def _update_historical_summary(self, run_summary: pd.DataFrame):
        """Update historical runs database"""
        historical_file = Path(settings.HYPERLIQUID_RRS_RESULTS_DIR) / "historical_runs.csv"
        try:
            if historical_file.exists():
                historical_df = pd.read_csv(historical_file)
                historical_df = pd.concat([historical_df, run_summary], ignore_index=True)
            else:
                historical_df = run_summary
            
            historical_df.to_csv(historical_file, index=False)
            logger.info(f"Historical database updated. Total runs: {len(historical_df)}")
        except Exception as e:
            logger.error(f"Error updating historical file: {e}")
    
    async def process_timeframe(self, timeframe: str, lookback_days: int, benchmark_symbol: str):
        """Process RRS calculation for a specific timeframe"""
        logger.info(f"===== Starting processing for timeframe: {timeframe} (Lookback: {lookback_days} days) =====")
        
        # Validate benchmark symbol
        if benchmark_symbol not in settings.HYPERLIQUID_RRS_SYMBOLS:
            logger.error(f"Benchmark symbol '{benchmark_symbol}' not found in symbols list")
            return
        
        # Get symbols excluding benchmark
        symbols_to_process = [s for s in settings.HYPERLIQUID_RRS_SYMBOLS if s != benchmark_symbol]
        if not symbols_to_process:
            logger.warning("No symbols to process (excluding benchmark)")
            return
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)
        logger.info(f"Time range (UTC): {start_time} to {end_time}")
        
        # Process benchmark data
        logger.info(f"Processing benchmark: {benchmark_symbol}")
        benchmark_df_processed = await self.fetch_and_process_symbol_data(
            benchmark_symbol, timeframe, start_time, end_time
        )
        
        if benchmark_df_processed is None or benchmark_df_processed.empty:
            logger.error(f"Failed to process benchmark data for {benchmark_symbol}")
            return
        
        benchmark_df_for_rrs = benchmark_df_processed[['timestamp', 'log_return']].copy()
        logger.debug(f"Processed benchmark data shape: {benchmark_df_for_rrs.shape}")
        
        # Process symbols
        rrs_results_list: List[Dict] = []
        for symbol_name in symbols_to_process:
            symbol_df_processed = await self.fetch_and_process_symbol_data(
                symbol_name, timeframe, start_time, end_time
            )
            
            if symbol_df_processed is None or symbol_df_processed.empty:
                continue
            
            # Calculate RRS
            rrs_df = self.calculator.calculate_rrs(symbol_df_processed, benchmark_df_for_rrs)
            if rrs_df.empty:
                logger.warning(f"No RRS data calculated for {symbol_name}")
                continue
            
            # Save processed data
            if settings.HYPERLIQUID_RRS_SAVE_PROCESSED_DATA:
                self._save_dataframe(rrs_df, settings.HYPERLIQUID_RRS_DATA_DIR, 
                                   f"{symbol_name}_{timeframe}_{lookback_days}d_processed_rrs.csv")
            
            # Get latest smoothed RRS value
            try:
                rrs_df = rrs_df.sort_values(by='timestamp')
                latest_rrs = rrs_df.iloc[-1]['smoothed_rrs']
                if pd.isna(latest_rrs):
                    logger.warning(f"Latest smoothed_rrs for {symbol_name} is NaN")
                    continue
                
                rrs_results_list.append({'symbol': symbol_name, 'rrs': latest_rrs})
                logger.debug(f"Latest RRS for {symbol_name}: {latest_rrs:.4f}")
            except (IndexError, KeyError) as e:
                logger.error(f"Error getting latest RRS for {symbol_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error for {symbol_name}: {e}")
        
        # Finalize timeframe results
        if not rrs_results_list:
            logger.warning(f"No symbols had valid RRS scores for timeframe {timeframe}")
            return
        
        final_rrs_df = pd.DataFrame(rrs_results_list)
        final_rrs_df.sort_values('rrs', ascending=False, inplace=True)
        logger.info(f"--- RRS Ranking for Timeframe: {timeframe} ---")
        logger.info('\n' + final_rrs_df.to_string())
        
        # Save current run results
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timeframe}_{lookback_days}d_{benchmark_symbol}_{timestamp_str}.csv"
        self._save_dataframe(final_rrs_df, settings.HYPERLIQUID_RRS_RESULTS_DIR, filename)
        
        # Update historical summary
        if settings.HYPERLIQUID_RRS_GENERATE_HISTORICAL_SUMMARY:
            try:
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
                
                logger.info(f"--- Run Summary for {timeframe} ---")
                logger.info(f"Benchmark: {benchmark_symbol}")
                logger.info(f"Symbols Analyzed: {len(symbols_to_process)}, Ranked: {len(final_rrs_df)}")
                logger.info(f"🏆 Top: {top_performer['symbol']} (RRS: {top_performer['rrs']:.4f})")
                logger.info(f"📉 Bottom: {bottom_performer['symbol']} (RRS: {bottom_performer['rrs']:.4f})")
                
            except Exception as e:
                logger.error(f"Error generating run summary for {timeframe}: {e}")
        
        logger.info(f"===== Finished processing for timeframe: {timeframe} =====")
    
    async def run_full_analysis(self):
        """Run complete RRS analysis across all timeframes"""
        logger.info("🚀 Starting Hyperliquid RRS Analyzer...")
        
        benchmark_symbol = settings.HYPERLIQUID_RRS_BENCHMARK_SYMBOL
        lookback = settings.HYPERLIQUID_RRS_LOOKBACK_DAYS
        timeframes = settings.HYPERLIQUID_RRS_TIMEFRAMES
        
        logger.info(f"Configuration: Benchmark='{benchmark_symbol}', Lookback={lookback} days")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Symbols: {len(settings.HYPERLIQUID_RRS_SYMBOLS)}")
        
        for timeframe in timeframes:
            await self.process_timeframe(timeframe, lookback, benchmark_symbol)
        
        logger.info("🏁 Hyperliquid RRS Analyzer Finished")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "Hyperliquid RRS Analysis",
            "enabled": settings.ENABLE_HYPERLIQUID_RRS_ANALYSIS,
            "symbols_count": len(settings.HYPERLIQUID_RRS_SYMBOLS),
            "timeframes": settings.HYPERLIQUID_RRS_TIMEFRAMES,
            "benchmark": settings.HYPERLIQUID_RRS_BENCHMARK_SYMBOL,
            "lookback_days": settings.HYPERLIQUID_RRS_LOOKBACK_DAYS,
            "data_dir": settings.HYPERLIQUID_RRS_DATA_DIR,
            "results_dir": settings.HYPERLIQUID_RRS_RESULTS_DIR
        }
    
    def get_rankings(self, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """Get latest rankings for timeframe or consolidated top rankings"""
        try:
            if timeframe:
                # Get latest ranking for specific timeframe
                results_path = Path(settings.HYPERLIQUID_RRS_RESULTS_DIR)
                pattern = f"{timeframe}_*.csv"
                files = list(results_path.glob(pattern))
                
                if not files:
                    return {"error": f"No rankings found for timeframe {timeframe}"}
                
                latest_file = max(files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest_file)
                
                return {
                    "timeframe": timeframe,
                    "file": latest_file.name,
                    "rankings": df.to_dict('records')
                }
            else:
                # Get consolidated top rankings
                top_rrs_file = Path(settings.HYPERLIQUID_RRS_RESULTS_DIR) / "Top_RRS.csv"
                if not top_rrs_file.exists():
                    # Generate it
                    self.top_analyzer.process_results_files(settings.HYPERLIQUID_RRS_RESULTS_DIR)
                
                if top_rrs_file.exists():
                    df = pd.read_csv(top_rrs_file)
                    return {
                        "type": "consolidated",
                        "rankings": df.to_dict('records')
                    }
                else:
                    return {"error": "No consolidated rankings available"}
                    
        except Exception as e:
            logger.error(f"Error getting rankings: {e}")
            return {"error": str(e)}
    
    def interpret_score(self, rrs_score: float) -> Dict[str, Any]:
        """Interpret RRS score using configured thresholds"""
        if rrs_score >= settings.HYPERLIQUID_RRS_EXCEPTIONAL_THRESHOLD:
            interpretation = "Exceptional Strength"
            color = "green"
        elif rrs_score >= settings.HYPERLIQUID_RRS_STRONG_THRESHOLD:
            interpretation = "Strong Strength"
            color = "lightgreen"
        elif rrs_score >= settings.HYPERLIQUID_RRS_MODERATE_THRESHOLD:
            interpretation = "Moderate Strength"
            color = "yellow"
        elif rrs_score >= settings.HYPERLIQUID_RRS_WEAK_THRESHOLD:
            interpretation = "Weak Strength"
            color = "orange"
        else:
            interpretation = "Underperforming"
            color = "red"
        
        return {
            "rrs_score": rrs_score,
            "interpretation": interpretation,
            "color": color,
            "thresholds": {
                "exceptional": settings.HYPERLIQUID_RRS_EXCEPTIONAL_THRESHOLD,
                "strong": settings.HYPERLIQUID_RRS_STRONG_THRESHOLD,
                "moderate": settings.HYPERLIQUID_RRS_MODERATE_THRESHOLD,
                "weak": settings.HYPERLIQUID_RRS_WEAK_THRESHOLD
            }
        }
    
    def get_symbols(self) -> Dict[str, Any]:
        """Get configured symbols"""
        return {
            "symbols": settings.HYPERLIQUID_RRS_SYMBOLS,
            "benchmark": settings.HYPERLIQUID_RRS_BENCHMARK_SYMBOL,
            "count": len(settings.HYPERLIQUID_RRS_SYMBOLS)
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current RRS configuration"""
        return {
            "enabled": settings.ENABLE_HYPERLIQUID_RRS_ANALYSIS,
            "api_url": settings.HYPERLIQUID_RRS_API_URL,
            "symbols_count": len(settings.HYPERLIQUID_RRS_SYMBOLS),
            "timeframes": settings.HYPERLIQUID_RRS_TIMEFRAMES,
            "default_timeframe": settings.HYPERLIQUID_RRS_DEFAULT_TIMEFRAME,
            "lookback_days": settings.HYPERLIQUID_RRS_LOOKBACK_DAYS,
            "benchmark_symbol": settings.HYPERLIQUID_RRS_BENCHMARK_SYMBOL,
            "volatility_window": settings.HYPERLIQUID_RRS_VOLATILITY_WINDOW,
            "volume_window": settings.HYPERLIQUID_RRS_VOLUME_WINDOW,
            "smoothing_span": settings.HYPERLIQUID_RRS_SMOOTHING_SPAN,
            "data_dir": settings.HYPERLIQUID_RRS_DATA_DIR,
            "results_dir": settings.HYPERLIQUID_RRS_RESULTS_DIR,
            "request_timeout": settings.HYPERLIQUID_RRS_REQUEST_TIMEOUT,
            "min_data_points": settings.HYPERLIQUID_RRS_MIN_DATA_POINTS,
            "save_raw_data": settings.HYPERLIQUID_RRS_SAVE_RAW_DATA,
            "save_processed_data": settings.HYPERLIQUID_RRS_SAVE_PROCESSED_DATA,
            "generate_historical_summary": settings.HYPERLIQUID_RRS_GENERATE_HISTORICAL_SUMMARY,
            "interpretation_thresholds": {
                "exceptional": settings.HYPERLIQUID_RRS_EXCEPTIONAL_THRESHOLD,
                "strong": settings.HYPERLIQUID_RRS_STRONG_THRESHOLD,
                "moderate": settings.HYPERLIQUID_RRS_MODERATE_THRESHOLD,
                "weak": settings.HYPERLIQUID_RRS_WEAK_THRESHOLD
            }
        }


# Global service instance
hyperliquid_rrs_service = HyperliquidRRSAnalysisService() 