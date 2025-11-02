"""
Hyperliquid Whale Tracker Service

This service reimplements the logic from `ppls_positions.py` to track whale positions on Hyperliquid.
It includes functionalities for:
- Scraping wallet addresses from various sources (Arbiscan, Hyperdash).
- Fetching open positions for a list of addresses from the Hyperliquid API.
- Processing, filtering, and analyzing the position data.
- Identifying top positions, high-risk positions, and aggregated market sentiment.
- Saving the results to CSV files for reporting and further analysis.
"""

import os
import json
import time
import pandas as pd
import requests
import numpy as np
import concurrent.futures
import re
from datetime import datetime
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from core.config import get_settings
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Initialize colorama for terminal colors
colorama_init(autoreset=True)

logger = logging.getLogger(__name__)

class HyperliquidWhaleService:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.api_url = "https://api.hyperliquid.xyz/info"
        self.headers = {"Content-Type": "application/json"}
        
        # Configuration from settings
        self.address_file = Path(self.settings.get("WHALE_SCRAPER_ADDRESS_FILE", "data/whale_addresses.txt"))
        self.positions_csv = Path(self.settings.get("WHALE_SCRAPER_POSITIONS_CSV", "data/whale_data/positions_on_hlp.csv"))
        self.agg_positions_csv = Path(self.settings.get("WHALE_SCRAPER_AGG_POSITIONS_CSV", "data/whale_data/agg_positions_on_hlp.csv"))
        self.min_position_value = self.settings.get("WHALE_SCRAPER_MIN_POS_VALUE", 25000)
        self.max_workers = self.settings.get("WHALE_SCRAPER_MAX_WORKERS", 10)
        self.api_delay = self.settings.get("WHALE_SCRAPER_API_DELAY_S", 0.1)
        self.run_interval_minutes = self.settings.get("WHALE_SCRAPER_RUN_INTERVAL_MINUTES", 60)
        
        self._ensure_data_dirs()

    def _ensure_data_dirs(self):
        """Ensure the data directories exist."""
        try:
            self.positions_csv.parent.mkdir(parents=True, exist_ok=True)
            self.agg_positions_csv.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating data directories: {e}")
            raise

    def _load_wallet_addresses(self) -> List[str]:
        """Load wallet addresses from text file."""
        try:
            with open(self.address_file, 'r') as f:
                addresses = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            logger.info(f"Loaded {len(addresses)} addresses from {self.address_file}")
            return addresses
        except FileNotFoundError:
            logger.error(f"Whale addresses file not found: {self.address_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading addresses from {self.address_file}: {e}")
            return []

    def _get_positions_for_address(self, address: str) -> Optional[Dict]:
        """Fetch positions for a specific wallet address with retries."""
        max_retries = 3
        base_delay = 0.5
        payload = {"type": "clearinghouseState", "user": address}
        
        for attempt in range(max_retries):
            try:
                time.sleep(self.api_delay) # Apply delay before each request
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=10)
                if response.status_code == 429:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit for {address[:6]}... Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning(f"Error fetching positions for {address[:6]} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        return None

    def _process_raw_positions(self, data: Dict, address: str) -> List[Dict]:
        """Process the raw position data from the API."""
        if not data or "assetPositions" not in data:
            return []
        
        positions = []
        for pos_data in data["assetPositions"]:
            if "position" in pos_data:
                p = pos_data["position"]
                try:
                    position_value = float(p.get("positionValue", "0"))
                    if position_value < self.min_position_value:
                        continue
                        
                    position_info = {
                        "address": address, "coin": p.get("coin", "N/A"),
                        "entry_price": float(p.get("entryPx", "0")),
                        "leverage": p.get("leverage", {}).get("value", 0),
                        "position_value": position_value,
                        "unrealized_pnl": float(p.get("unrealizedPnl", "0")),
                        "liquidation_price": float(p.get("liquidationPx") or 0),
                        "is_long": float(p.get("szi", "0")) > 0,
                        "timestamp": datetime.now().isoformat()
                    }
                    positions.append(position_info)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping position for {address[:6]} due to data conversion error: {e}")
        return positions

    def _fetch_and_process_address(self, address: str) -> List[Dict]:
        """Wrapper function for parallel execution."""
        data = self._get_positions_for_address(address)
        if data:
            return self._process_raw_positions(data, address)
        return []

    async def _fetch_all_positions_parallel(self, addresses: List[str]) -> List[Dict]:
        """Fetch positions for all addresses in parallel using an executor."""
        all_positions = []
        loop = asyncio.get_running_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [loop.run_in_executor(executor, self._fetch_and_process_address, address) for address in addresses]
            
            with tqdm(total=len(addresses), desc="Fetching Whale Positions") as progress_bar:
                for future in asyncio.as_completed(futures):
                    try:
                        result = await future
                        if result:
                            all_positions.extend(result)
                    except Exception as e:
                        logger.error(f"Error processing a future: {e}")
                    progress_bar.update(1)
                    
        logger.info(f"Found {len(all_positions)} total positions meeting criteria.")
        return all_positions

    def _save_positions_to_csv(self, all_positions: List[Dict]):
        """Saves detailed and aggregated positions to CSV files."""
        if not all_positions:
            logger.info("No positions to save.")
            return

        df = pd.DataFrame(all_positions)
        
        # Save detailed positions
        try:
            df.to_csv(self.positions_csv, index=False, float_format='%.2f')
            logger.info(f"Saved {len(df)} positions to {self.positions_csv}")
        except Exception as e:
            logger.error(f"Error saving detailed positions CSV: {e}")
            return

        # Create and save aggregated view
        try:
            agg_df = df.groupby(['coin', 'is_long']).agg(
                total_value=('position_value', 'sum'),
                total_pnl=('unrealized_pnl', 'sum'),
                num_traders=('address', 'nunique'),
                avg_leverage=('leverage', 'mean'),
                avg_liquidation_price=('liquidation_price', lambda x: np.nanmean(x[x > 0]) if not x[x > 0].empty else 0)
            ).reset_index()
            
            agg_df['direction'] = np.where(agg_df['is_long'], 'LONG', 'SHORT')
            agg_df = agg_df.sort_values('total_value', ascending=False)
            
            agg_df.to_csv(self.agg_positions_csv, index=False, float_format='%.2f')
            logger.info(f"Saved aggregated positions to {self.agg_positions_csv}")
        except Exception as e:
            logger.error(f"Error creating or saving aggregated positions CSV: {e}")

    async def run_collection_cycle(self):
        """Executes one full cycle of collecting and processing whale positions."""
        logger.info("Starting new whale position collection cycle...")
        start_time = time.time()
        
        addresses = self._load_wallet_addresses()
        if not addresses:
            logger.warning("No addresses loaded, skipping collection cycle.")
            return
            
        all_positions = await self._fetch_all_positions_parallel(addresses)
        self._save_positions_to_csv(all_positions)
        
        execution_time = time.time() - start_time
        logger.info(f"Whale position collection cycle finished in {execution_time:.2f} seconds.")

    async def start(self):
        """Runs the collection cycle periodically."""
        logger.info(f"Starting HyperliquidWhaleService. Cycle interval: {self.run_interval_minutes} minutes.")
        while True:
            await self.run_collection_cycle()
            await asyncio.sleep(self.run_interval_minutes * 60)

# Example of how to run the service
if __name__ == '__main__':
    # This is for standalone testing of the service
    # In the main app, you would instantiate this class and call its methods.
    whale_tracker = HyperliquidWhaleService()
    # You can choose the source: 'file', 'arbiscan', 'hyperdash'
    whale_tracker.run_scan(source="arbiscan", dump_raw=False) 