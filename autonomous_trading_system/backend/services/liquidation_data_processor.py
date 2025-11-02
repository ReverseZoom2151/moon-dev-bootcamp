"""
Liquidation Data Processing Service
Integrated from Day 21 cutitup.py script

This service processes liquidation data from CSV files, filtering by symbols,
determining liquidation types, and creating aggregated outputs.
"""

import pandas as pd
import numpy as np
import os
import asyncio
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class LiquidationDataProcessor:
    """
    Processes liquidation CSV data files to extract symbol-specific data
    and create aggregated totals with resampling.
    
    Integrates Day 21 cutitup.py functionality into the autonomous trading system.
    """
    
    def __init__(self):
        self.settings = settings
        self.input_file = self.settings.LIQUIDATION_DATA_INPUT_FILE
        self.output_dir = self.settings.LIQUIDATION_DATA_OUTPUT_DIR
        self.symbols = self.settings.LIQUIDATION_DATA_SYMBOLS
        self.large_size_threshold = self.settings.LIQUIDATION_DATA_LARGE_SIZE_THRESHOLD
        self.resample_interval = self.settings.LIQUIDATION_DATA_RESAMPLE_INTERVAL
        self.auto_process = self.settings.LIQUIDATION_DATA_AUTO_PROCESS
        
        # Required CSV columns
        self.required_columns = [
            "symbol", "side", "order_type", "time_in_force",
            "original_quantity", "price", "average_price", "order_status",
            "order_last_filled_quantity", "order_filled_accumulated_quantity",
            "order_trade_time", "usd_size"
        ]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Liquidation Data Processor initialized")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Symbols to process: {self.symbols}")
        logger.info(f"Large size threshold: ${self.large_size_threshold:,.2f}")
    
    async def process_liquidation_file(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a liquidation CSV file and create symbol-specific outputs.
        
        Args:
            file_path: Optional path to CSV file. Uses default if not provided.
            
        Returns:
            Dict with processing results and statistics
        """
        if file_path is None:
            file_path = self.input_file
            
        try:
            # Check if input file exists
            if not os.path.exists(file_path):
                logger.error(f"Input file not found: {file_path}")
                return {"success": False, "error": f"Input file not found: {file_path}"}
            
            logger.info(f"Processing liquidation data from: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Validate required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Process the data
            processed_data = await self._process_dataframe(df)
            
            # Generate summary statistics
            summary = self._generate_summary_stats(processed_data['filtered_df'])
            
            result = {
                "success": True,
                "processed_file": file_path,
                "timestamp": datetime.now().isoformat(),
                "total_rows_processed": len(df),
                "liquidations_found": len(processed_data['filtered_df']),
                "symbols_processed": processed_data['symbols_with_data'],
                "output_files": processed_data['output_files'],
                "totals_file": processed_data['totals_file'],
                "summary_stats": summary
            }
            
            logger.info(f"Processing completed successfully")
            logger.info(f"Found {result['liquidations_found']} liquidations")
            logger.info(f"Processed symbols: {result['symbols_processed']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing liquidation file: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Core dataframe processing logic adapted from cutitup.py
        """
        # Determine liquidation type (vectorized operations)
        valid_symbol = (df['symbol'].astype(str).str.len() == 3) | (df['symbol'] == "1000PEPE")
        large_size = df['usd_size'] > self.large_size_threshold
        is_sell = df['side'] == "SELL"
        is_buy = df['side'] == "BUY"
        
        conditions = [
            valid_symbol & large_size & is_sell,
            valid_symbol & large_size & is_buy
        ]
        choices = ["L LIQ", "S LIQ"]
        
        df['LIQ_SIDE'] = np.select(conditions, choices, default=None)
        
        # Filter out rows where LIQ_SIDE is None
        df = df[df['LIQ_SIDE'].notna()].copy()
        
        if len(df) == 0:
            logger.warning("No liquidations found matching criteria")
            return {
                'filtered_df': df,
                'symbols_with_data': [],
                'output_files': [],
                'totals_file': None
            }
        
        # Convert epoch to datetime
        df['datetime'] = pd.to_datetime(df['order_trade_time'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Adjust symbol handling
        df['symbol'] = df['symbol'].astype(str).apply(
            lambda x: x if x == "1000PEPE" else x[:3]
        )
        
        # Process individual symbols
        symbols_with_data = []
        output_files = []
        
        for symbol in self.symbols:
            filtered_df = df[df['symbol'].str.upper() == symbol.upper()]
            
            if not filtered_df.empty:
                output_path = os.path.join(self.output_dir, f'{symbol}_liq_data.csv')
                filtered_df.to_csv(output_path)
                symbols_with_data.append(symbol)
                output_files.append(output_path)
                logger.info(f"Saved {len(filtered_df)} liquidations for {symbol} to {output_path}")
            else:
                logger.info(f"No liquidation data found for symbol {symbol}")
        
        # Create aggregated totals
        totals_file = await self._create_aggregated_totals(df)
        
        return {
            'filtered_df': df,
            'symbols_with_data': symbols_with_data,
            'output_files': output_files,
            'totals_file': totals_file
        }
    
    async def _create_aggregated_totals(self, df: pd.DataFrame) -> str:
        """
        Create aggregated totals with resampling
        """
        # Prepare DataFrame for resampling
        df_all = df.reset_index()
        df_all['datetime'] = pd.to_datetime(df_all['datetime'])
        df_all.set_index('datetime', inplace=True)
        
        # Separate resample for L LIQ and S LIQ
        df_totals_L = df_all[df_all['LIQ_SIDE'] == 'L LIQ'].resample(
            self.resample_interval
        ).agg({'usd_size': 'sum', 'price': 'mean'})
        
        df_totals_S = df_all[df_all['LIQ_SIDE'] == 'S LIQ'].resample(
            self.resample_interval
        ).agg({'usd_size': 'sum', 'price': 'mean'})
        
        # Add LIQ_SIDE column back
        df_totals_L['LIQ_SIDE'] = 'L LIQ'
        df_totals_S['LIQ_SIDE'] = 'S LIQ'
        
        # Combine the dataframes
        df_totals = pd.concat([df_totals_L, df_totals_S])
        
        # Add symbol column set to 'All'
        df_totals['symbol'] = 'All'
        
        # Reset index to have 'datetime' as a column
        df_totals.reset_index(inplace=True)
        
        # Reorder columns
        df_totals = df_totals[['datetime', 'symbol', 'LIQ_SIDE', 'price', 'usd_size']]
        
        # Save aggregated totals
        output_totals_path = os.path.join(self.output_dir, 'total_liq_data.csv')
        df_totals.to_csv(output_totals_path, index=False)
        
        logger.info(f"Saved aggregated totals to {output_totals_path}")
        logger.info(f"Aggregated totals shape: {df_totals.shape}")
        
        return output_totals_path
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the processed data
        """
        if len(df) == 0:
            return {}
        
        stats = {
            "total_liquidations": len(df),
            "long_liquidations": len(df[df['LIQ_SIDE'] == 'L LIQ']),
            "short_liquidations": len(df[df['LIQ_SIDE'] == 'S LIQ']),
            "total_usd_volume": float(df['usd_size'].sum()),
            "average_liquidation_size": float(df['usd_size'].mean()),
            "largest_liquidation": float(df['usd_size'].max()),
            "smallest_liquidation": float(df['usd_size'].min()),
            "unique_symbols": df['symbol'].nunique(),
            "time_range": {
                "start": df.index.min().isoformat() if len(df) > 0 else None,
                "end": df.index.max().isoformat() if len(df) > 0 else None
            },
            "symbol_breakdown": df.groupby('symbol')['usd_size'].agg([
                'count', 'sum', 'mean'
            ]).to_dict('index')
        }
        
        return stats
    
    async def monitor_for_new_files(self):
        """
        Monitor for new liquidation data files and process them automatically
        """
        if not self.auto_process:
            logger.info("Auto-processing disabled")
            return
        
        logger.info("Starting file monitoring for new liquidation data")
        
        processed_files = set()
        
        while True:
            try:
                # Look for CSV files in the current directory
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'liq_data' in f.lower()]
                
                for csv_file in csv_files:
                    if csv_file not in processed_files:
                        logger.info(f"Found new liquidation data file: {csv_file}")
                        result = await self.process_liquidation_file(csv_file)
                        
                        if result['success']:
                            processed_files.add(csv_file)
                            logger.info(f"Successfully processed: {csv_file}")
                        else:
                            logger.error(f"Failed to process {csv_file}: {result.get('error')}")
                
                # Wait before checking again
                await asyncio.sleep(self.settings.LIQUIDATION_DATA_PROCESSOR_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in file monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status and statistics
        """
        status = {
            "enabled": self.settings.ENABLE_LIQUIDATION_DATA_PROCESSOR,
            "auto_process": self.auto_process,
            "input_file": self.input_file,
            "output_directory": self.output_dir,
            "symbols_tracked": self.symbols,
            "large_size_threshold": self.large_size_threshold,
            "resample_interval": self.resample_interval,
            "output_files_exist": {}
        }
        
        # Check which output files exist
        for symbol in self.symbols:
            output_path = os.path.join(self.output_dir, f'{symbol}_liq_data.csv')
            status["output_files_exist"][symbol] = os.path.exists(output_path)
        
        # Check if totals file exists
        totals_path = os.path.join(self.output_dir, 'total_liq_data.csv')
        status["totals_file_exists"] = os.path.exists(totals_path)
        
        return status
    
    async def cleanup_old_files(self, days_old: int = 7):
        """
        Clean up old processed files
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
            
            for file_path in Path(self.output_dir).glob("*.csv"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")


# Global processor instance
liquidation_processor = LiquidationDataProcessor()

async def start_liquidation_data_processor():
    """
    Start the liquidation data processing service
    """
    if not settings.ENABLE_LIQUIDATION_DATA_PROCESSOR:
        logger.info("Liquidation data processor disabled")
        return
    
    logger.info("Starting liquidation data processor service")
    
    # Start file monitoring if auto-processing is enabled
    if settings.LIQUIDATION_DATA_AUTO_PROCESS:
        asyncio.create_task(liquidation_processor.monitor_for_new_files())
    
    logger.info("Liquidation data processor service started")

async def stop_liquidation_data_processor():
    """
    Stop the liquidation data processing service
    """
    logger.info("Liquidation data processor service stopped")

# Utility functions for external access
async def process_liquidation_file(file_path: str = None) -> Dict[str, Any]:
    """
    Process a liquidation file manually
    """
    return await liquidation_processor.process_liquidation_file(file_path)

async def get_processor_status() -> Dict[str, Any]:
    """
    Get processor status
    """
    return await liquidation_processor.get_processing_status() 