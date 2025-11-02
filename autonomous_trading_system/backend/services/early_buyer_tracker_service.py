"""
Early Buyer Tracker Service
Finds early buyers of tokens by analyzing transaction data from Birdeye API
"""

import asyncio
import logging
import json
import time
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import aiohttp
import requests
from core.config import get_settings

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EarlyBuyerConfig:
    """Configuration for early buyer tracking."""
    MIN_TRADE_SIZE_USD: float = 3000.0
    MAX_TRADE_SIZE_USD: float = 100000000.0
    SORT_TYPE: str = "asc"  # "asc" for earliest first, "desc" for latest first
    API_BASE_URL: str = "https://public-api.birdeye.so"
    EXPLORER_BASE_URL: str = "https://gmgn.ai/sol/address/"
    FETCH_LIMIT: int = 50
    MAX_OFFSET: int = 100000
    RETRY_DELAY_SECONDS: int = 5
    MAX_CONSECUTIVE_ERRORS: int = 3
    MAX_EMPTY_BATCHES: int = 3
    REQUEST_DELAY_SECONDS: float = 0.1

class EarlyBuyerTrackerService:
    """
    Service for tracking early buyers of tokens using Birdeye API
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.config = EarlyBuyerConfig()
        self.session = None
        self._api_key = None
        self.output_dir = Path("output_data/early_buyers")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @property
    def api_key(self) -> Optional[str]:
        """Get Birdeye API key from settings or environment"""
        if not self._api_key:
            # Try to get from settings first, then environment
            self._api_key = getattr(self.settings, 'BIRDEYE_API_KEY', None) or os.getenv("BIRDEYE_API_KEY")
        return self._api_key
    
    def parse_date_range(self, start_str: str, end_str: str) -> Tuple[datetime, datetime]:
        """Parse date strings into timezone-aware UTC datetime objects."""
        try:
            start_dt = datetime.strptime(start_str, "%m-%d-%Y").replace(tzinfo=timezone.utc)
            end_dt = (datetime.strptime(end_str, "%m-%d-%Y") + timedelta(days=1) - timedelta(seconds=1)).replace(tzinfo=timezone.utc)
            return start_dt, end_dt
        except ValueError as e:
            logger.error(f"Error parsing dates: {e}. Please use MM-DD-YYYY format.")
            raise
    
    async def make_api_request(self, url: str, headers: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Make async GET request to Birdeye API with error handling."""
        try:
            if self.session:
                async with self.session.get(url, headers=headers, timeout=30) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                # Fallback to sync request
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
        except asyncio.TimeoutError:
            logger.warning("Request timed out.")
        except aiohttp.ClientResponseError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except Exception as e:
            logger.error(f"Request exception occurred: {e}")
        return None
    
    def process_trade(self, trade: Dict[str, Any], start_dt: datetime, end_dt: datetime, 
                     min_trade_size: float, max_trade_size: float) -> Optional[Dict[str, Any]]:
        """Process a single trade, calculating USD value and checking filters."""
        try:
            trade_time_ts = trade.get('blockUnixTime')
            if trade_time_ts is None:
                return None
            
            trade_time = datetime.fromtimestamp(trade_time_ts, tz=timezone.utc)
            
            # Check date range
            if not (start_dt <= trade_time <= end_dt):
                return None
            
            quote = trade.get('quote', {})
            base = trade.get('base', {})
            owner = trade.get('owner')
            tx_hash = trade.get('txHash')
            
            if not all([quote, base, owner, tx_hash]):
                return None
            
            # Calculate trade size
            ui_amount_str = quote.get('uiAmountString')
            nearest_price_str = quote.get('nearestPrice')
            
            if ui_amount_str is None or nearest_price_str is None:
                return None
            
            try:
                ui_amount = float(ui_amount_str)
                nearest_price = float(nearest_price_str)
                trade_size_usd = abs(ui_amount * nearest_price)
            except (ValueError, TypeError):
                return None
            
            # Check trade size filter
            if not (min_trade_size <= trade_size_usd <= max_trade_size):
                return None
            
            # Format output
            owner_link = f"{self.config.EXPLORER_BASE_URL}{owner}"
            return {
                'Timestamp': trade_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'Owner': owner,
                'Owner Link': owner_link,
                'From Symbol': quote.get('symbol', 'Unknown'),
                'From Amount': ui_amount,
                'To Symbol': base.get('symbol', 'Unknown'),
                'To Amount': base.get('uiAmountString', 'Unknown'),
                'USD Value': trade_size_usd,
                'Tx Hash': tx_hash
            }
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
            return None
    
    async def fetch_and_process_trades(self, token_address: str, start_dt: datetime, end_dt: datetime,
                                     min_trade_size: float = None, max_trade_size: float = None,
                                     sort_type: str = None) -> pd.DataFrame:
        """Fetch trades, process them based on filters, and return a DataFrame."""
        if not self.api_key:
            raise ValueError("Birdeye API key not configured")
        
        min_trade_size = min_trade_size or self.config.MIN_TRADE_SIZE_USD
        max_trade_size = max_trade_size or self.config.MAX_TRADE_SIZE_USD
        sort_type = sort_type or self.config.SORT_TYPE
        
        all_processed_trades: List[Dict[str, Any]] = []
        offset = 0
        total_trades_fetched = 0
        consecutive_errors = 0
        consecutive_empty_batches = 0
        
        headers = {"accept": "application/json", "X-API-KEY": self.api_key}
        
        logger.info(f"Starting trade fetch for {token_address}...")
        
        while offset <= self.config.MAX_OFFSET:
            url = (
                f"{self.config.API_BASE_URL}/defi/txs/token?address={token_address}"
                f"&offset={offset}&limit={self.config.FETCH_LIMIT}&tx_type=swap&sort_type={sort_type}"
            )
            
            logger.info(f"Fetching trades from offset {offset}...")
            
            data = await self.make_api_request(url, headers)
            
            if not data:
                consecutive_errors += 1
                logger.warning(f"API request failed. Error count: {consecutive_errors}/{self.config.MAX_CONSECUTIVE_ERRORS}")
                if consecutive_errors >= self.config.MAX_CONSECUTIVE_ERRORS:
                    logger.error("Reached maximum consecutive errors. Stopping.")
                    break
                await asyncio.sleep(self.config.RETRY_DELAY_SECONDS)
                continue
            else:
                consecutive_errors = 0
            
            trades = data.get('data', {}).get('items', [])
            total_trades_fetched += len(trades)
            
            if not trades:
                consecutive_empty_batches += 1
                logger.info(f"No trades in batch. Empty batch count: {consecutive_empty_batches}/{self.config.MAX_EMPTY_BATCHES}")
                if consecutive_empty_batches >= self.config.MAX_EMPTY_BATCHES:
                    logger.info("Reached maximum consecutive empty batches. Stopping.")
                    break
                offset += self.config.FETCH_LIMIT
                await asyncio.sleep(self.config.REQUEST_DELAY_SECONDS)
                continue
            else:
                consecutive_empty_batches = 0
            
            logger.info(f"Processing {len(trades)} trades from batch...")
            batch_processed_count = 0
            stop_processing = False
            
            for trade in trades:
                processed = self.process_trade(trade, start_dt, end_dt, min_trade_size, max_trade_size)
                if processed:
                    all_processed_trades.append(processed)
                    batch_processed_count += 1
                elif processed is None and sort_type == 'asc':
                    try:
                        first_trade_time = datetime.fromtimestamp(trades[0].get('blockUnixTime'), tz=timezone.utc)
                        if first_trade_time > end_dt:
                            logger.info("First trade in batch is after end date. Stopping early.")
                            stop_processing = True
                            break
                    except Exception:
                        pass
            
            logger.info(f"Added {batch_processed_count} trades from this batch.")
            
            if stop_processing:
                break
            
            offset += self.config.FETCH_LIMIT
            if offset > self.config.MAX_OFFSET:
                logger.warning("Reached maximum offset limit. Stopping.")
                break
            
            await asyncio.sleep(self.config.REQUEST_DELAY_SECONDS)
        
        logger.info(f"Finished fetching. Total trades: {total_trades_fetched}. Matching filters: {len(all_processed_trades)}")
        return pd.DataFrame(all_processed_trades)
    
    async def save_trades(self, df: pd.DataFrame, token_address: str, custom_filename: str = None) -> str:
        """Save the DataFrame to a CSV file."""
        if df.empty:
            logger.info("No trades found matching criteria. No CSV file created.")
            return None
        
        try:
            filename = custom_filename or f"{token_address}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = self.output_dir / filename
            
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully saved {len(df)} trades to {output_path}")
            return str(output_path)
        except IOError as e:
            logger.error(f"Error saving trades: {e}")
            raise
    
    async def track_early_buyers(self, token_address: str, start_date: str = "01-01-2020", 
                               end_date: str = "12-31-2030", min_trade_size: float = None,
                               max_trade_size: float = None, sort_type: str = None,
                               save_to_file: bool = True) -> Dict[str, Any]:
        """
        Main method to track early buyers for a token
        
        Args:
            token_address: Token contract address
            start_date: Start date in MM-DD-YYYY format
            end_date: End date in MM-DD-YYYY format
            min_trade_size: Minimum trade size in USD
            max_trade_size: Maximum trade size in USD
            sort_type: Sort order ("asc" or "desc")
            save_to_file: Whether to save results to CSV
        
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting early buyer tracking for {token_address}")
            logger.info(f"Date range: {start_date} to {end_date}")
            
            start_dt, end_dt = self.parse_date_range(start_date, end_date)
            
            async with self:
                trades_df = await self.fetch_and_process_trades(
                    token_address, start_dt, end_dt, min_trade_size, max_trade_size, sort_type
                )
                
                output_file = None
                if save_to_file and not trades_df.empty:
                    output_file = await self.save_trades(trades_df, token_address)
                
                # Calculate summary statistics
                summary = {
                    "total_trades": len(trades_df),
                    "unique_buyers": trades_df['Owner'].nunique() if not trades_df.empty else 0,
                    "total_volume_usd": trades_df['USD Value'].sum() if not trades_df.empty else 0,
                    "avg_trade_size_usd": trades_df['USD Value'].mean() if not trades_df.empty else 0,
                    "earliest_trade": trades_df['Timestamp'].min() if not trades_df.empty else None,
                    "latest_trade": trades_df['Timestamp'].max() if not trades_df.empty else None
                }
                
                # Get top buyers by volume
                top_buyers = []
                if not trades_df.empty:
                    buyer_stats = trades_df.groupby('Owner').agg({
                        'USD Value': ['sum', 'count', 'mean'],
                        'Timestamp': 'min'
                    }).round(2)
                    
                    buyer_stats.columns = ['Total_Volume', 'Trade_Count', 'Avg_Trade_Size', 'First_Trade']
                    buyer_stats = buyer_stats.sort_values('Total_Volume', ascending=False)
                    
                    top_buyers = buyer_stats.head(20).reset_index().to_dict('records')
                    for buyer in top_buyers:
                        buyer['Explorer_Link'] = f"{self.config.EXPLORER_BASE_URL}{buyer['Owner']}"
                
                duration = time.time() - start_time
                
                return {
                    "status": "success",
                    "token_address": token_address,
                    "summary": summary,
                    "top_buyers": top_buyers,
                    "trades_data": trades_df.to_dict('records') if len(trades_df) <= 1000 else [],  # Limit for API response
                    "output_file": output_file,
                    "execution_time_seconds": round(duration, 2),
                    "parameters": {
                        "start_date": start_date,
                        "end_date": end_date,
                        "min_trade_size_usd": min_trade_size or self.config.MIN_TRADE_SIZE_USD,
                        "max_trade_size_usd": max_trade_size or self.config.MAX_TRADE_SIZE_USD,
                        "sort_type": sort_type or self.config.SORT_TYPE
                    }
                }
                
        except Exception as e:
            logger.error(f"Error tracking early buyers: {e}")
            return {
                "status": "error",
                "error": str(e),
                "token_address": token_address,
                "execution_time_seconds": round(time.time() - start_time, 2)
            }
    
    async def get_saved_analyses(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of saved early buyer analyses"""
        try:
            files = []
            for file_path in self.output_dir.glob("*.csv"):
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "token_address": file_path.stem.split('_')[0] if '_' in file_path.stem else file_path.stem,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "size_bytes": stat.st_size,
                    "file_path": str(file_path)
                })
            
            # Sort by creation time, newest first
            files.sort(key=lambda x: x['created_at'], reverse=True)
            return files[:limit]
            
        except Exception as e:
            logger.error(f"Error getting saved analyses: {e}")
            return []
    
    async def load_analysis(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a saved analysis from CSV file"""
        try:
            file_path = self.output_dir / filename
            if file_path.exists():
                return pd.read_csv(file_path)
            else:
                logger.error(f"File not found: {filename}")
                return None
        except Exception as e:
            logger.error(f"Error loading analysis {filename}: {e}")
            return None
    
    def get_config(self) -> Dict[str, Any]:
        """Get current service configuration"""
        return {
            "api_key_configured": self.api_key is not None,
            "output_directory": str(self.output_dir),
            "config": {
                "min_trade_size_usd": self.config.MIN_TRADE_SIZE_USD,
                "max_trade_size_usd": self.config.MAX_TRADE_SIZE_USD,
                "sort_type": self.config.SORT_TYPE,
                "api_base_url": self.config.API_BASE_URL,
                "explorer_base_url": self.config.EXPLORER_BASE_URL,
                "fetch_limit": self.config.FETCH_LIMIT,
                "max_offset": self.config.MAX_OFFSET,
                "retry_delay_seconds": self.config.RETRY_DELAY_SECONDS,
                "max_consecutive_errors": self.config.MAX_CONSECUTIVE_ERRORS,
                "max_empty_batches": self.config.MAX_EMPTY_BATCHES,
                "request_delay_seconds": self.config.REQUEST_DELAY_SECONDS
            }
        }

# Global service instance
early_buyer_tracker_service = EarlyBuyerTrackerService()