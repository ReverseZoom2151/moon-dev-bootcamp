# autonomous_trading_system/backend/services/moondev_api_service.py

import os
import pandas as pd
import requests
import time
from pathlib import Path
import logging
from typing import Optional, List, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)

class MoonDevAPIService:
    def __init__(self, settings):
        self.settings = settings
        self.api_key = self.settings.get("MOONDEV_API_KEY")
        self.base_url = self.settings.get("MOONDEV_API_BASE_URL", "http://api.moondev.com:8000")
        
        # Define a directory for caching data
        self.data_dir = Path(self.settings.get("MOONDEV_DATA_DIR", "data/moondev_api"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers: Dict[str, str] = {'X-API-Key': self.api_key} if self.api_key else {}
        if not self.api_key:
            logger.warning("MoonDev API key not provided. Some endpoints may not be accessible.")
            
        self.session = requests.Session()
        self.max_retries = 3
        self.chunk_size = 8192 * 16
        logger.info(f"MoonDevAPIService initialized. Data directory: {self.data_dir}, Base URL: {self.base_url}")

    def _fetch_csv(self, filename: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        retry_delay = 2
        temp_file = self.data_dir / f"temp_{filename}"
        
        for attempt in range(self.max_retries):
            try:
                url = f'{self.base_url}/files/{filename}'
                if limit:
                    url += f'?limit={limit}'
                
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}: Fetching {url}")
                response = self.session.get(url, headers=self.headers, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        f.write(chunk)
                
                if temp_file.stat().st_size == 0:
                    logger.error(f"Downloaded file {temp_file} is empty.")
                    temp_file.unlink()
                    return None
                    
                df = pd.read_csv(temp_file)
                
                final_file = self.data_dir / filename
                if final_file.exists():
                    final_file.unlink()

                temp_file.rename(final_file)
                logger.info(f"Successfully fetched and saved {filename} to {final_file}")
                return df
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error fetching {filename} (attempt {attempt + 1}): {e.response.status_code} - {e.response.text}")
                if temp_file.exists(): temp_file.unlink(missing_ok=True)
                if e.response.status_code in [403, 404]:
                    logger.error(f"Unrecoverable error {e.response.status_code}. No more retries.")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching {filename} (attempt {attempt + 1}): {e}")
                if temp_file.exists(): temp_file.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Unexpected error fetching {filename} (attempt {attempt + 1}): {e}")
                if temp_file.exists(): temp_file.unlink(missing_ok=True)

            if attempt < self.max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Failed to fetch {filename} after {self.max_retries} attempts.")
        return None

    def get_liquidation_data(self, limit: Optional[int] = 10000) -> Optional[pd.DataFrame]:
        return self._fetch_csv("liq_data.csv", limit=limit)

    def get_funding_data(self) -> Optional[pd.DataFrame]:
        return self._fetch_csv("funding.csv")

    def get_token_addresses(self) -> Optional[pd.DataFrame]:
        return self._fetch_csv("new_token_addresses.csv")

    def get_oi_total(self) -> Optional[pd.DataFrame]:
        return self._fetch_csv("oi_total.csv")

    def get_oi_data(self) -> Optional[pd.DataFrame]:
        return self._fetch_csv("oi.csv")

    def get_agg_positions_hlp(self) -> Optional[pd.DataFrame]:
        return self._fetch_csv("agg_positions_on_hlp.csv")

    def get_positions_hlp(self) -> Optional[pd.DataFrame]:
        """ Fetches detailed positions on HLP. This is likely the main whale data source. """
        return self._fetch_csv("positions_on_hlp.csv")
        
    def _fetch_json_endpoint(self, endpoint: str, filename: str) -> Optional[pd.DataFrame]:
        logger.info(f"Fetching {filename} from {endpoint}...")
        if not self.api_key:
            logger.warning(f"API key is required for {endpoint}")
            return None
        
        try:
            response = self.session.get(f"{self.base_url}/{endpoint}", headers=self.headers, timeout=30)
            response.raise_for_status()
            
            save_path = self.data_dir / filename
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            df = pd.read_csv(save_path)
            logger.info(f"Successfully loaded {len(df)} rows from {filename}.")
            return df
        except Exception as e:
            logger.error(f"Error fetching/processing {filename}: {e}", exc_info=True)
            return None

    def get_copybot_follow_list(self) -> Optional[pd.DataFrame]:
        return self._fetch_json_endpoint("copybot/data/follow_list", "follow_list.csv")

    def get_copybot_recent_transactions(self) -> Optional[pd.DataFrame]:
        return self._fetch_json_endpoint("copybot/data/recent_txs", "recent_txs.csv")

    def get_whale_addresses(self) -> Optional[List[str]]:
        filename = "whale_addresses.txt"
        url = f'{self.base_url}/files/{filename}'
        save_path = self.data_dir / filename
        logger.info(f"Fetching {filename}...")
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            addresses = response.text.splitlines()
             
            if isinstance(addresses, list) and addresses:
                 logger.info(f"Successfully loaded {len(addresses)} whale addresses.")
                 return addresses
            else:
                logger.error(f"Failed to parse whale addresses from {save_path}.")
                return None
        except Exception as e:
            logger.error(f"Error fetching {filename}: {e}", exc_info=True)
            return None

@lru_cache()
def get_moondev_api_service(settings) -> "MoonDevAPIService":
    return MoonDevAPIService(settings)
