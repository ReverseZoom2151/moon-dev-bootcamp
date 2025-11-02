"""
Trading Hours Data Filter Service

Filters OHLCV data to specific trading hours (first/last hour of trading sessions).
Based on cut2_1st_last_hr.py with enterprise enhancements.
EXTENDED: Market vs Non-Market hours filtering based on cut2_mkt_non_mkthrs.py
EXTENDED: Seasonal filtering based on cut2_nosummers.py
"""

import asyncio
import logging
import pandas as pd
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class FilteredDataResult:
    """Result structure for filtered data operations"""
    input_file: str
    output_file: str
    output_file_secondary: Optional[str] = None  # For market/non-market split
    original_records: int = 0
    filtered_records: int = 0
    filtered_records_secondary: Optional[int] = None  # For market/non-market split
    filter_config: Dict[str, str] = None
    processing_time_seconds: float = 0
    success: bool = False
    error_message: Optional[str] = None

@dataclass
class TradingHoursConfig:
    """Configuration for trading hours filtering"""
    market_open_utc: str = "13:30"      # 9:30 AM EDT in UTC
    one_hour_after_open_utc: str = "14:30"  # 10:30 AM EDT in UTC
    one_hour_before_close_utc: str = "19:00"  # 3:00 PM EDT in UTC
    market_close_utc: str = "20:00"     # 4:00 PM EDT in UTC
    exclude_weekends: bool = True
    output_suffix: str = "-1stlasthr"

@dataclass
class MarketHoursConfig:
    """Configuration for market vs non-market hours filtering"""
    market_open_utc: str = "13:30"      # 9:30 AM EDT in UTC
    market_close_utc: str = "20:00"     # 4:00 PM EDT in UTC
    exclude_weekends: bool = True
    market_hours_suffix: str = "-mkt-open"
    non_market_hours_suffix: str = "-mkt-closed"

@dataclass
class SeasonalFilterConfig:
    """Configuration for seasonal filtering (excluding months)"""
    exclude_start_month: int = 5        # May
    exclude_end_month: int = 9          # September
    output_suffix: str = "-nosummers"

class TradingHoursDataFilterService:
    """
    Service for filtering OHLCV data to specific trading hours
    
    Features:
    - Filter data to first hour after open and last hour before close
    - Filter data to market hours vs non-market hours
    - Filter data to exclude seasonal months (e.g., summer months)
    - Support for multiple file formats and data sources
    - Async processing for large datasets
    - Comprehensive logging and error handling
    - API integration ready
    """
    
    def __init__(self, base_data_folder: str = "test_data"):
        self.base_data_folder = Path(base_data_folder)
        self.default_config = TradingHoursConfig()
        self.default_market_config = MarketHoursConfig()
        self.default_seasonal_config = SeasonalFilterConfig()
        self.results: List[FilteredDataResult] = []
        
        logger.info("🕒 Trading Hours Data Filter Service initialized")
    
    async def filter_file(
        self,
        input_filename: str,
        config: Optional[TradingHoursConfig] = None,
        custom_data_folder: Optional[str] = None
    ) -> FilteredDataResult:
        """
        Filter a single file to trading hours (first/last hour)
        
        Args:
            input_filename: Name of the input CSV file
            config: Trading hours configuration (uses default if None)
            custom_data_folder: Custom data folder path (uses default if None)
            
        Returns:
            FilteredDataResult with processing details
        """
        start_time = datetime.now()
        
        try:
            # Use provided config or default
            filter_config = config or self.default_config
            
            # Determine data folder
            data_folder = Path(custom_data_folder) if custom_data_folder else self.base_data_folder
            input_path = data_folder / input_filename
            
            logger.info(f"🔄 Starting filter process for {input_filename}")
            
            # Load data
            data = await self._load_data_async(input_path)
            if data is None:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=0,
                    filtered_records=0,
                    filter_config=self._config_to_dict(filter_config),
                    processing_time_seconds=0,
                    success=False,
                    error_message="Failed to load data"
                )
            
            original_records = len(data)
            
            # Filter data
            filtered_data = await self._filter_trading_hours_async(data, filter_config)
            if filtered_data is None or filtered_data.empty:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=original_records,
                    filtered_records=0,
                    filter_config=self._config_to_dict(filter_config),
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message="Filtering resulted in empty data"
                )
            
            filtered_records = len(filtered_data)
            
            # Save filtered data
            output_path = await self._save_filtered_data_async(filtered_data, input_path, filter_config.output_suffix)
            if not output_path:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=original_records,
                    filtered_records=filtered_records,
                    filter_config=self._config_to_dict(filter_config),
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message="Failed to save filtered data"
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = FilteredDataResult(
                input_file=str(input_path),
                output_file=str(output_path),
                original_records=original_records,
                filtered_records=filtered_records,
                filter_config=self._config_to_dict(filter_config),
                processing_time_seconds=processing_time,
                success=True
            )
            
            # Store result
            self.results.append(result)
            
            logger.info(f"✅ Filter completed: {original_records} → {filtered_records} records ({processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            error_msg = f"Error filtering {input_filename}: {str(e)}"
            logger.error(error_msg)
            
            return FilteredDataResult(
                input_file=str(input_path) if 'input_path' in locals() else input_filename,
                output_file="",
                original_records=0,
                filtered_records=0,
                filter_config=self._config_to_dict(config or self.default_config),
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_message=error_msg
            )
    
    async def filter_market_non_market_hours(
        self,
        input_filename: str,
        config: Optional[MarketHoursConfig] = None,
        custom_data_folder: Optional[str] = None
    ) -> FilteredDataResult:
        """
        Filter a single file into market hours and non-market hours
        
        Args:
            input_filename: Name of the input CSV file
            config: Market hours configuration (uses default if None)
            custom_data_folder: Custom data folder path (uses default if None)
            
        Returns:
            FilteredDataResult with processing details for both market and non-market hours
        """
        start_time = datetime.now()
        
        try:
            # Use provided config or default
            filter_config = config or self.default_market_config
            
            # Determine data folder
            data_folder = Path(custom_data_folder) if custom_data_folder else self.base_data_folder
            input_path = data_folder / input_filename
            
            logger.info(f"🔄 Starting market/non-market hours filter for {input_filename}")
            
            # Load data
            data = await self._load_data_async(input_path)
            if data is None:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=0,
                    filtered_records=0,
                    filter_config=self._market_config_to_dict(filter_config),
                    processing_time_seconds=0,
                    success=False,
                    error_message="Failed to load data"
                )
            
            original_records = len(data)
            
            # Filter data into market and non-market hours
            market_data, non_market_data = await self._filter_market_non_market_async(data, filter_config)
            if market_data is None or non_market_data is None:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=original_records,
                    filtered_records=0,
                    filter_config=self._market_config_to_dict(filter_config),
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message="Market/Non-market filtering failed"
                )
            
            market_records = len(market_data)
            non_market_records = len(non_market_data)
            
            # Save both datasets
            market_output_path = None
            non_market_output_path = None
            
            if not market_data.empty:
                market_output_path = await self._save_filtered_data_async(
                    market_data, input_path, filter_config.market_hours_suffix
                )
            
            if not non_market_data.empty:
                non_market_output_path = await self._save_filtered_data_async(
                    non_market_data, input_path, filter_config.non_market_hours_suffix
                )
            
            if not market_output_path and not non_market_output_path:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=original_records,
                    filtered_records=market_records,
                    filtered_records_secondary=non_market_records,
                    filter_config=self._market_config_to_dict(filter_config),
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message="Failed to save filtered data"
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = FilteredDataResult(
                input_file=str(input_path),
                output_file=str(market_output_path) if market_output_path else "",
                output_file_secondary=str(non_market_output_path) if non_market_output_path else "",
                original_records=original_records,
                filtered_records=market_records,
                filtered_records_secondary=non_market_records,
                filter_config=self._market_config_to_dict(filter_config),
                processing_time_seconds=processing_time,
                success=True
            )
            
            # Store result
            self.results.append(result)
            
            logger.info(f"✅ Market/Non-market filter completed: {original_records} → {market_records} market + {non_market_records} non-market ({processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            error_msg = f"Error filtering market/non-market hours for {input_filename}: {str(e)}"
            logger.error(error_msg)
            
            return FilteredDataResult(
                input_file=str(input_path) if 'input_path' in locals() else input_filename,
                output_file="",
                original_records=0,
                filtered_records=0,
                filter_config=self._market_config_to_dict(config or self.default_market_config),
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_message=error_msg
            )
    
    async def filter_seasonal_months(
        self,
        input_filename: str,
        config: Optional[SeasonalFilterConfig] = None,
        custom_data_folder: Optional[str] = None
    ) -> FilteredDataResult:
        """
        Filter a single file to exclude seasonal months (e.g., summer months)
        
        Args:
            input_filename: Name of the input CSV file
            config: Seasonal filter configuration (uses default if None)
            custom_data_folder: Custom data folder path (uses default if None)
            
        Returns:
            FilteredDataResult with processing details
        """
        start_time = datetime.now()
        
        try:
            # Use provided config or default
            filter_config = config or self.default_seasonal_config
            
            # Determine data folder
            data_folder = Path(custom_data_folder) if custom_data_folder else self.base_data_folder
            input_path = data_folder / input_filename
            
            logger.info(f"🔄 Starting seasonal filter for {input_filename} (excluding months {filter_config.exclude_start_month}-{filter_config.exclude_end_month})")
            
            # Load data
            data = await self._load_data_async(input_path)
            if data is None:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=0,
                    filtered_records=0,
                    filter_config=self._seasonal_config_to_dict(filter_config),
                    processing_time_seconds=0,
                    success=False,
                    error_message="Failed to load data"
                )
            
            original_records = len(data)
            
            # Filter data to exclude seasonal months
            filtered_data = await self._filter_seasonal_months_async(data, filter_config)
            if filtered_data is None or filtered_data.empty:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=original_records,
                    filtered_records=0,
                    filter_config=self._seasonal_config_to_dict(filter_config),
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message="Seasonal filtering resulted in empty data"
                )
            
            filtered_records = len(filtered_data)
            
            # Save filtered data
            output_path = await self._save_filtered_data_async(filtered_data, input_path, filter_config.output_suffix)
            if not output_path:
                return FilteredDataResult(
                    input_file=str(input_path),
                    output_file="",
                    original_records=original_records,
                    filtered_records=filtered_records,
                    filter_config=self._seasonal_config_to_dict(filter_config),
                    processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                    success=False,
                    error_message="Failed to save filtered data"
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = FilteredDataResult(
                input_file=str(input_path),
                output_file=str(output_path),
                original_records=original_records,
                filtered_records=filtered_records,
                filter_config=self._seasonal_config_to_dict(filter_config),
                processing_time_seconds=processing_time,
                success=True
            )
            
            # Store result
            self.results.append(result)
            
            logger.info(f"✅ Seasonal filter completed: {original_records} → {filtered_records} records ({processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            error_msg = f"Error seasonal filtering {input_filename}: {str(e)}"
            logger.error(error_msg)
            
            return FilteredDataResult(
                input_file=str(input_path) if 'input_path' in locals() else input_filename,
                output_file="",
                original_records=0,
                filtered_records=0,
                filter_config=self._seasonal_config_to_dict(config or self.default_seasonal_config),
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_message=error_msg
            )
    
    async def filter_multiple_files(
        self,
        filenames: List[str],
        config: Optional[TradingHoursConfig] = None,
        custom_data_folder: Optional[str] = None
    ) -> List[FilteredDataResult]:
        """
        Filter multiple files to trading hours
        
        Args:
            filenames: List of input CSV filenames
            config: Trading hours configuration
            custom_data_folder: Custom data folder path
            
        Returns:
            List of FilteredDataResult
        """
        logger.info(f"🔄 Starting batch filter process for {len(filenames)} files")
        
        # Process files concurrently
        tasks = [
            self.filter_file(filename, config, custom_data_folder)
            for filename in filenames
        ]
        
        results = await asyncio.gather(*tasks)
        
        successful_count = sum(1 for r in results if r.success)
        logger.info(f"✅ Batch filter completed: {successful_count}/{len(filenames)} files processed successfully")
        
        return results
    
    async def filter_multiple_market_non_market(
        self,
        filenames: List[str],
        config: Optional[MarketHoursConfig] = None,
        custom_data_folder: Optional[str] = None
    ) -> List[FilteredDataResult]:
        """
        Filter multiple files into market/non-market hours
        
        Args:
            filenames: List of input CSV filenames
            config: Market hours configuration
            custom_data_folder: Custom data folder path
            
        Returns:
            List of FilteredDataResult
        """
        logger.info(f"🔄 Starting batch market/non-market filter for {len(filenames)} files")
        
        # Process files concurrently
        tasks = [
            self.filter_market_non_market_hours(filename, config, custom_data_folder)
            for filename in filenames
        ]
        
        results = await asyncio.gather(*tasks)
        
        successful_count = sum(1 for r in results if r.success)
        logger.info(f"✅ Batch market/non-market filter completed: {successful_count}/{len(filenames)} files processed successfully")
        
        return results
    
    async def filter_multiple_seasonal(
        self,
        filenames: List[str],
        config: Optional[SeasonalFilterConfig] = None,
        custom_data_folder: Optional[str] = None
    ) -> List[FilteredDataResult]:
        """
        Filter multiple files to exclude seasonal months
        
        Args:
            filenames: List of input CSV filenames
            config: Seasonal filter configuration
            custom_data_folder: Custom data folder path
            
        Returns:
            List of FilteredDataResult
        """
        logger.info(f"🔄 Starting batch seasonal filter for {len(filenames)} files")
        
        # Process files concurrently
        tasks = [
            self.filter_seasonal_months(filename, config, custom_data_folder)
            for filename in filenames
        ]
        
        results = await asyncio.gather(*tasks)
        
        successful_count = sum(1 for r in results if r.success)
        logger.info(f"✅ Batch seasonal filter completed: {successful_count}/{len(filenames)} files processed successfully")
        
        return results
    
    async def discover_and_filter_all(
        self,
        config: Optional[TradingHoursConfig] = None,
        custom_data_folder: Optional[str] = None,
        file_pattern: str = "*.csv"
    ) -> List[FilteredDataResult]:
        """
        Discover all CSV files in data folder and filter them
        
        Args:
            config: Trading hours configuration
            custom_data_folder: Custom data folder path
            file_pattern: File pattern to match (default: *.csv)
            
        Returns:
            List of FilteredDataResult
        """
        data_folder = Path(custom_data_folder) if custom_data_folder else self.base_data_folder
        
        if not data_folder.exists():
            logger.error(f"❌ Data folder not found: {data_folder}")
            return []
        
        # Discover CSV files
        csv_files = list(data_folder.glob(file_pattern))
        filenames = [f.name for f in csv_files]
        
        if not filenames:
            logger.warning(f"⚠️ No files matching '{file_pattern}' found in {data_folder}")
            return []
        
        logger.info(f"📁 Discovered {len(filenames)} files: {filenames}")
        
        return await self.filter_multiple_files(filenames, config, str(data_folder))
    
    async def discover_and_filter_market_non_market_all(
        self,
        config: Optional[MarketHoursConfig] = None,
        custom_data_folder: Optional[str] = None,
        file_pattern: str = "*.csv"
    ) -> List[FilteredDataResult]:
        """
        Discover all CSV files and filter them into market/non-market hours
        
        Args:
            config: Market hours configuration
            custom_data_folder: Custom data folder path
            file_pattern: File pattern to match (default: *.csv)
            
        Returns:
            List of FilteredDataResult
        """
        data_folder = Path(custom_data_folder) if custom_data_folder else self.base_data_folder
        
        if not data_folder.exists():
            logger.error(f"❌ Data folder not found: {data_folder}")
            return []
        
        # Discover CSV files
        csv_files = list(data_folder.glob(file_pattern))
        filenames = [f.name for f in csv_files]
        
        if not filenames:
            logger.warning(f"⚠️ No files matching '{file_pattern}' found in {data_folder}")
            return []
        
        logger.info(f"📁 Discovered {len(filenames)} files for market/non-market filtering: {filenames}")
        
        return await self.filter_multiple_market_non_market(filenames, config, str(data_folder))
    
    async def discover_and_filter_seasonal_all(
        self,
        config: Optional[SeasonalFilterConfig] = None,
        custom_data_folder: Optional[str] = None,
        file_pattern: str = "*.csv"
    ) -> List[FilteredDataResult]:
        """
        Discover all CSV files and filter them to exclude seasonal months
        
        Args:
            config: Seasonal filter configuration
            custom_data_folder: Custom data folder path
            file_pattern: File pattern to match (default: *.csv)
            
        Returns:
            List of FilteredDataResult
        """
        data_folder = Path(custom_data_folder) if custom_data_folder else self.base_data_folder
        
        if not data_folder.exists():
            logger.error(f"❌ Data folder not found: {data_folder}")
            return []
        
        # Discover CSV files
        csv_files = list(data_folder.glob(file_pattern))
        filenames = [f.name for f in csv_files]
        
        if not filenames:
            logger.warning(f"⚠️ No files matching '{file_pattern}' found in {data_folder}")
            return []
        
        logger.info(f"📁 Discovered {len(filenames)} files for seasonal filtering: {filenames}")
        
        return await self.filter_multiple_seasonal(filenames, config, str(data_folder))
    
    async def _load_data_async(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load CSV data asynchronously"""
        try:
            logger.debug(f"Loading data from: {file_path}")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, self._load_data_sync, file_path)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return None
    
    def _load_data_sync(self, file_path: Path) -> pd.DataFrame:
        """Synchronous data loading"""
        data = pd.read_csv(file_path, parse_dates=['datetime'])
        
        # Handle timezone localization
        if data['datetime'].dt.tz is None:
            logger.debug("Localizing naive datetime to UTC")
            data['datetime'] = data['datetime'].dt.tz_localize('UTC')
        else:
            logger.debug("Converting timezone-aware datetime to UTC")
            data['datetime'] = data['datetime'].dt.tz_convert('UTC')
        
        data.set_index('datetime', inplace=True)
        return data
    
    async def _filter_trading_hours_async(
        self,
        df: pd.DataFrame,
        config: TradingHoursConfig
    ) -> Optional[pd.DataFrame]:
        """Filter data to trading hours asynchronously"""
        try:
            logger.debug(f"Filtering for {config.market_open_utc}-{config.one_hour_after_open_utc} and {config.one_hour_before_close_utc}-{config.market_close_utc} UTC")
            
            # Run in thread pool for large datasets
            loop = asyncio.get_event_loop()
            filtered_data = await loop.run_in_executor(None, self._filter_trading_hours_sync, df, config)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error during filtering: {e}")
            return None
    
    def _filter_trading_hours_sync(self, df: pd.DataFrame, config: TradingHoursConfig) -> pd.DataFrame:
        """Synchronous trading hours filtering"""
        # Validate index
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz.zone != 'UTC':
            raise ValueError("DataFrame index must be a UTC-localized DatetimeIndex")
        
        # Filter first hour after open
        first_hour = df.between_time(config.market_open_utc, config.one_hour_after_open_utc)
        if config.exclude_weekends:
            first_hour = first_hour[first_hour.index.dayofweek < 5]
        
        # Filter last hour before close
        last_hour = df.between_time(config.one_hour_before_close_utc, config.market_close_utc)
        if config.exclude_weekends:
            last_hour = last_hour[last_hour.index.dayofweek < 5]
        
        # Combine and sort
        filtered_data = pd.concat([first_hour, last_hour]).sort_index()
        
        return filtered_data
    
    async def _filter_market_non_market_async(
        self,
        df: pd.DataFrame,
        config: MarketHoursConfig
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Filter data into market and non-market hours asynchronously"""
        try:
            logger.debug(f"Filtering market hours ({config.market_open_utc}-{config.market_close_utc} UTC) and non-market hours")
            
            # Run in thread pool for large datasets
            loop = asyncio.get_event_loop()
            market_data, non_market_data = await loop.run_in_executor(
                None, self._filter_market_non_market_sync, df, config
            )
            
            return market_data, non_market_data
            
        except Exception as e:
            logger.error(f"Error during market/non-market filtering: {e}")
            return None, None
    
    def _filter_market_non_market_sync(
        self, 
        df: pd.DataFrame, 
        config: MarketHoursConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Synchronous market vs non-market hours filtering"""
        import datetime as dt
        
        # Validate index
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz.zone != 'UTC':
            raise ValueError("DataFrame index must be a UTC-localized DatetimeIndex")
        
        # Convert time strings to time objects for comparison
        open_time = dt.datetime.strptime(config.market_open_utc, '%H:%M').time()
        close_time = dt.datetime.strptime(config.market_close_utc, '%H:%M').time()
        
        # Filter for weekdays first if requested
        if config.exclude_weekends:
            weekdays_df = df[df.index.dayofweek < 5].copy()  # Monday=0, Sunday=6
        else:
            weekdays_df = df.copy()
        
        # Filter for market hours: [open_time, close_time) - left inclusive
        market_hours_df = weekdays_df.between_time(
            config.market_open_utc, 
            config.market_close_utc, 
            inclusive='left'
        )
        
        # Filter for non-market hours: time < open_time OR time >= close_time
        non_market_hours_mask = (
            (weekdays_df.index.time < open_time) | 
            (weekdays_df.index.time >= close_time)
        )
        non_market_hours_df = weekdays_df[non_market_hours_mask]
        
        return market_hours_df, non_market_hours_df
    
    async def _filter_seasonal_months_async(
        self,
        df: pd.DataFrame,
        config: SeasonalFilterConfig
    ) -> Optional[pd.DataFrame]:
        """Filter data to exclude seasonal months asynchronously"""
        try:
            logger.debug(f"Filtering to exclude months {config.exclude_start_month}-{config.exclude_end_month}")
            
            # Run in thread pool for large datasets
            loop = asyncio.get_event_loop()
            filtered_data = await loop.run_in_executor(None, self._filter_seasonal_months_sync, df, config)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error during seasonal filtering: {e}")
            return None
    
    def _filter_seasonal_months_sync(self, df: pd.DataFrame, config: SeasonalFilterConfig) -> pd.DataFrame:
        """Synchronous seasonal months filtering"""
        # Validate index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        # Create a boolean mask for rows *within* the exclude range
        exclude_mask = (df.index.month >= config.exclude_start_month) & (df.index.month <= config.exclude_end_month)
        
        # Keep rows where the mask is False (i.e., outside the exclude range)
        filtered_df = df[~exclude_mask]
        
        return filtered_df
    
    async def _save_filtered_data_async(
        self,
        df: pd.DataFrame,
        input_path: Path,
        suffix: str
    ) -> Optional[Path]:
        """Save filtered data asynchronously"""
        try:
            # Generate output path
            output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, df.to_csv, str(output_path))
            
            logger.debug(f"Filtered data saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving filtered data: {e}")
            return None
    
    def _config_to_dict(self, config: TradingHoursConfig) -> Dict[str, str]:
        """Convert config to dictionary for JSON serialization"""
        return {
            "market_open_utc": config.market_open_utc,
            "one_hour_after_open_utc": config.one_hour_after_open_utc,
            "one_hour_before_close_utc": config.one_hour_before_close_utc,
            "market_close_utc": config.market_close_utc,
            "exclude_weekends": str(config.exclude_weekends),
            "output_suffix": config.output_suffix
        }
    
    def _market_config_to_dict(self, config: MarketHoursConfig) -> Dict[str, str]:
        """Convert market config to dictionary for JSON serialization"""
        return {
            "market_open_utc": config.market_open_utc,
            "market_close_utc": config.market_close_utc,
            "exclude_weekends": str(config.exclude_weekends),
            "market_hours_suffix": config.market_hours_suffix,
            "non_market_hours_suffix": config.non_market_hours_suffix
        }
    
    def _seasonal_config_to_dict(self, config: SeasonalFilterConfig) -> Dict[str, str]:
        """Convert seasonal config to dictionary for JSON serialization"""
        return {
            "exclude_start_month": str(config.exclude_start_month),
            "exclude_end_month": str(config.exclude_end_month),
            "output_suffix": config.output_suffix
        }
    
    async def get_service_status(self) -> Dict[str, any]:
        """Get service status and statistics"""
        total_processed = len(self.results)
        successful_processed = sum(1 for r in self.results if r.success)
        
        return {
            "service_name": "Trading Hours Data Filter Service",
            "status": "active",
            "base_data_folder": str(self.base_data_folder),
            "default_config": self._config_to_dict(self.default_config),
            "default_market_config": self._market_config_to_dict(self.default_market_config),
            "default_seasonal_config": self._seasonal_config_to_dict(self.default_seasonal_config),
            "available_modes": [
                "trading_hours_filter",  # First/last hour filtering
                "market_non_market_filter",  # Market vs non-market hours
                "seasonal_filter"  # Exclude seasonal months
            ],
            "statistics": {
                "total_files_processed": total_processed,
                "successful_files": successful_processed,
                "failed_files": total_processed - successful_processed,
                "success_rate": (successful_processed / total_processed * 100) if total_processed > 0 else 0
            },
            "recent_results": [
                {
                    "input_file": r.input_file,
                    "output_file": r.output_file,
                    "output_file_secondary": r.output_file_secondary,
                    "original_records": r.original_records,
                    "filtered_records": r.filtered_records,
                    "filtered_records_secondary": r.filtered_records_secondary,
                    "success": r.success,
                    "processing_time_seconds": r.processing_time_seconds
                }
                for r in self.results[-10:]  # Last 10 results
            ]
        }
    
    def get_recent_results(self, limit: int = 10) -> List[Dict]:
        """Get recent processing results"""
        return [
            {
                "input_file": r.input_file,
                "output_file": r.output_file,
                "output_file_secondary": r.output_file_secondary,
                "original_records": r.original_records,
                "filtered_records": r.filtered_records,
                "filtered_records_secondary": r.filtered_records_secondary,
                "filter_config": r.filter_config,
                "processing_time_seconds": r.processing_time_seconds,
                "success": r.success,
                "error_message": r.error_message
            }
            for r in self.results[-limit:]
        ]

# Convenience functions for API integration
async def filter_single_file(
    filename: str,
    data_folder: str = "test_data",
    market_open_utc: str = "13:30",
    one_hour_after_open_utc: str = "14:30",
    one_hour_before_close_utc: str = "19:00",
    market_close_utc: str = "20:00",
    output_suffix: str = "-1stlasthr"
) -> FilteredDataResult:
    """Convenience function to filter a single file"""
    service = TradingHoursDataFilterService(data_folder)
    config = TradingHoursConfig(
        market_open_utc=market_open_utc,
        one_hour_after_open_utc=one_hour_after_open_utc,
        one_hour_before_close_utc=one_hour_before_close_utc,
        market_close_utc=market_close_utc,
        output_suffix=output_suffix
    )
    
    return await service.filter_file(filename, config)

async def filter_market_non_market_single_file(
    filename: str,
    data_folder: str = "test_data",
    market_open_utc: str = "13:30",
    market_close_utc: str = "20:00",
    market_hours_suffix: str = "-mkt-open",
    non_market_hours_suffix: str = "-mkt-closed"
) -> FilteredDataResult:
    """Convenience function to filter a single file into market/non-market hours"""
    service = TradingHoursDataFilterService(data_folder)
    config = MarketHoursConfig(
        market_open_utc=market_open_utc,
        market_close_utc=market_close_utc,
        market_hours_suffix=market_hours_suffix,
        non_market_hours_suffix=non_market_hours_suffix
    )
    
    return await service.filter_market_non_market_hours(filename, config)

async def filter_seasonal_single_file(
    filename: str,
    data_folder: str = "test_data",
    exclude_start_month: int = 5,
    exclude_end_month: int = 9,
    output_suffix: str = "-nosummers"
) -> FilteredDataResult:
    """Convenience function to filter a single file excluding seasonal months"""
    service = TradingHoursDataFilterService(data_folder)
    config = SeasonalFilterConfig(
        exclude_start_month=exclude_start_month,
        exclude_end_month=exclude_end_month,
        output_suffix=output_suffix
    )
    
    return await service.filter_seasonal_months(filename, config)

async def get_trading_hours_filter_status() -> Dict[str, any]:
    """Get trading hours filter service status"""
    service = TradingHoursDataFilterService()
    return await service.get_service_status() 