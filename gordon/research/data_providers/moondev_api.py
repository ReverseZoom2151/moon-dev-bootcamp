"""
Moon Dev API Data Provider
==========================
Day 45: Integration with Moon Dev API for liquidation data, funding rates, OI data, and whale addresses.

Provides access to:
- Historical liquidation data
- Current funding rates
- Open interest data
- Whale addresses and positions
- Copy trading signals
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path
import traceback
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MoonDevAPI:
    """
    Moon Dev API handler for trading data.
    
    Provides access to liquidation data, funding rates, open interest,
    whale addresses, and copy trading signals.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        data_dir: Optional[Path | str] = None
    ):
        """
        Initialize Moon Dev API handler.

        Args:
            api_key: Your MoonDev API key. Overrides MOONDEV_API_KEY env var.
            base_url: The base URL for the API. Overrides MOONDEV_API_BASE_URL env var.
            data_dir: Directory to store downloaded data. Defaults to ./data/moondev_api
        """
        # Determine data directory path (Arg > Env Var > Default)
        _data_dir_str = data_dir or os.getenv('MOONDEV_DATA_DIR')
        if _data_dir_str:
            self.base_dir = Path(_data_dir_str).resolve()
        else:
            # Use Gordon's data directory structure
            script_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.base_dir = script_dir / "data" / "moondev_api"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Moon Dev API data directory: {self.base_dir}")

        self.api_key = api_key or os.getenv('MOONDEV_API_KEY')
        self.base_url = base_url or os.getenv('MOONDEV_API_BASE_URL', "http://api.moondev.com:8000")
        logger.info(f"Moon Dev API Base URL: {self.base_url}")

        self.headers: Dict[str, str] = {'X-API-Key': self.api_key} if self.api_key else {}
        if not self.api_key:
            logger.warning("Moon Dev API key not provided. Some endpoints may fail.")

        self.session = requests.Session()
        self.max_retries = 3
        self.chunk_size = 8192 * 16

    def _fetch_csv(self, filename: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Fetch CSV data from the API with retry logic."""
        retry_delay = 2  # seconds
        temp_file: Path = self.base_dir / f"temp_{filename}"

        for attempt in range(self.max_retries):
            try:
                url = f'{self.base_url}/files/{filename}'
                if limit:
                    url += f'?limit={limit}'

                logger.debug(f"Fetching {url} (attempt {attempt + 1}/{self.max_retries})")
                response = self.session.get(url, headers=self.headers, stream=True, timeout=60)
                response.raise_for_status()

                # Clean up temp file if exists
                if temp_file.exists():
                    temp_file.unlink()

                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)

                if temp_file.stat().st_size == 0:
                    logger.error(f"Downloaded file {temp_file} is empty.")
                    temp_file.unlink()
                    return None

                df = pd.read_csv(temp_file)

                final_file = self.base_dir / filename
                if final_file.exists():
                    try:
                        final_file.unlink()
                    except OSError as e:
                        logger.error(f"Could not remove existing file {final_file}: {e}")
                        if temp_file.exists():
                            temp_file.unlink()
                        return None

                temp_file.rename(final_file)
                logger.info(f"Successfully fetched and saved {filename}")
                return df

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error fetching {filename}: {e.response.status_code}")
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
                if e.response.status_code in [403, 404]:
                    return None  # No retries for auth/not found errors
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return None

            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                logger.error(f"Network error fetching {filename}: {str(e)}")
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return None

            except Exception as e:
                logger.error(f"Unexpected error fetching {filename}: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
                if attempt < self.max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return None

        return None

    def get_liquidation_data(self, limit: Optional[int] = 10000) -> Optional[pd.DataFrame]:
        """Get liquidation data from API, limited to last N rows by default."""
        return self._fetch_csv("liq_data.csv", limit=limit)

    def get_funding_data(self) -> Optional[pd.DataFrame]:
        """Get funding data from API."""
        return self._fetch_csv("funding.csv")

    def get_token_addresses(self) -> Optional[pd.DataFrame]:
        """Get token addresses from API (CSV format expected)."""
        return self._fetch_csv("new_token_addresses.csv")

    def get_oi_total(self) -> Optional[pd.DataFrame]:
        """Get total open interest data from API."""
        return self._fetch_csv("oi_total.csv")

    def get_oi_data(self) -> Optional[pd.DataFrame]:
        """Get detailed open interest data from API."""
        return self._fetch_csv("oi.csv")

    def get_copybot_follow_list(self) -> Optional[pd.DataFrame]:
        """Get current copy trading follow list."""
        endpoint = "copybot/data/follow_list"
        filename = "follow_list.csv"
        logger.info(f"Fetching copybot follow list...")
        if not self.api_key:
            logger.warning("API key required for copybot endpoints")
            return None

        try:
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()

            save_path = self.base_dir / filename
            with open(save_path, 'wb') as f:
                f.write(response.content)

            df = pd.read_csv(save_path)
            logger.info(f"Loaded {len(df)} rows from follow list")
            return df

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} fetching follow list")
            return None
        except Exception as e:
            logger.error(f"Error fetching follow list: {str(e)}")
            return None

    def get_copybot_recent_transactions(self) -> Optional[pd.DataFrame]:
        """Get recent copy trading transactions."""
        endpoint = "copybot/data/recent_txs"
        filename = "recent_txs.csv"
        logger.info(f"Fetching copybot recent transactions...")
        if not self.api_key:
            logger.warning("API key required for copybot endpoints")
            return None

        try:
            response = self.session.get(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()

            save_path = self.base_dir / filename
            with open(save_path, 'wb') as f:
                f.write(response.content)

            df = pd.read_csv(save_path)
            logger.info(f"Loaded {len(df)} rows from recent transactions")
            return df

        except Exception as e:
            logger.error(f"Error fetching recent transactions: {str(e)}")
            return None

    def get_agg_positions_hlp(self) -> Optional[pd.DataFrame]:
        """Get aggregated positions on HLP data."""
        return self._fetch_csv("agg_positions_on_hlp.csv")

    def get_positions_hlp(self) -> Optional[pd.DataFrame]:
        """Get detailed positions on HLP data."""
        return self._fetch_csv("positions_on_hlp.csv")

    def get_whale_addresses(self) -> Optional[List[str]]:
        """Get list of whale addresses (TXT format expected)."""
        filename = "whale_addresses.txt"
        url = f'{self.base_url}/files/{filename}'
        save_path = self.base_dir / filename
        logger.info(f"Fetching whale addresses...")

        try:
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)

            with open(save_path, 'r') as f:
                addresses = f.read().splitlines()

            if isinstance(addresses, list) and addresses:
                logger.info(f"Loaded {len(addresses)} whale addresses")
                return addresses
            else:
                logger.error(f"Failed to parse whale addresses from {save_path}")
                return None

        except Exception as e:
            logger.error(f"Error fetching whale addresses: {str(e)}")
            return None

