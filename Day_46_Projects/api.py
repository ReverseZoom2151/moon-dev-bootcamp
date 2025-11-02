"""
🌙 Moon Dev's API Handler
Built with love by Moon Dev 🚀

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.

Quick Start Guide:
-----------------
1. Install required packages:
   ```
   pip install requests pandas python-dotenv
   ```

2. Create a .env file in your project root:
   ```
   MOONDEV_API_KEY=your_api_key_here
   ```

3. Basic Usage:
   ```python
   from agents.api import MoonDevAPI
   
   # Initialize with env variable (recommended)
   api = MoonDevAPI()
   
   # Or initialize with direct key
   api = MoonDevAPI(api_key="your_key_here")
   
   # Get data
   liquidations = api.get_liquidation_data(limit=1000)  # Last 1000 rows
   funding = api.get_funding_data()
   oi = api.get_oi_data()
   ```

Available Methods:
----------------
- get_liquidation_data(limit=None): Get historical liquidation data. Use limit parameter for most recent data
- get_funding_data(): Get current funding rate data for various tokens
- get_token_addresses(): Get new Solana token launches and their addresses
- get_oi_data(): Get detailed open interest data for ETH or BTC individually
- get_oi_total(): Get total open interest data for ETH & BTC combined
- get_copybot_follow_list(): Get Moon Dev's personal copy trading follow list (for reference only - DYOR!)
- get_copybot_recent_transactions(): Get recent transactions from the followed wallets above
- get_agg_positions_hlp(): Get aggregated positions on HLP data
- get_positions_hlp(): Get detailed positions on HLP data
- get_whale_addresses(): Get list of whale addresses



Data Details:
------------
- Liquidation Data: Historical liquidation events with timestamps and amounts
- Funding Rates: Current funding rates across different tokens
- Token Addresses: New token launches on Solana with contract addresses
- Open Interest: Both detailed (per-token) and combined OI metrics
- CopyBot Data: Moon Dev's personal trading signals (use as reference only, always DYOR!)

Rate Limits:
-----------
- 100 requests per minute per API key
- Larger datasets (like liquidations) recommended to use limit parameter

⚠️ Important Notes:
-----------------
1. This is not financial advice
2. There are no guarantees of any kind
3. Use at your own risk
4. Always do your own research (DYOR)
5. The copybot follow list is Moon Dev's personal list and should not be used alone

Need an API key? for a limited time, bootcamp members get free api keys for claude, openai, helius, birdeye & quant elite gets access to the moon dev api. join here: https://algotradecamp.com
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path
import traceback
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv # REINSTATED
import logging # Added logging
from typing import Optional, List, Dict # Added for type hints

# Load environment variables REINSTATED
load_dotenv()

# Configure logging
# Configure logging for the API module specifically
api_logger = logging.getLogger(__name__)
# If no handlers are configured for this logger, add a basic one.
# This prevents 'No handlers could be found for logger "__main__"'
# if the main script doesn't configure logging.
# if not api_logger.handlers:
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     api_logger.addHandler(handler)
#     api_logger.setLevel(logging.INFO)

# Get the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent

class MoonDevAPI:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, 
                 data_dir: Optional[Path | str] = None):
        """Initialize the API handler.

        Args:
            api_key: Your MoonDev API key. Overrides MOONDEV_API_KEY env var.
            base_url: The base URL for the API. Overrides MOONDEV_API_BASE_URL env var.
            data_dir: The directory to store downloaded data. Overrides MOONDEV_DATA_DIR env var.
                      Defaults to a 'data/moondev_api' subdir relative to this script.
        """
        # Determine data directory path (Arg > Env Var > Default)
        _data_dir_str = data_dir or os.getenv('MOONDEV_DATA_DIR')
        if _data_dir_str:
            self.base_dir = Path(_data_dir_str).resolve()
        else:
            self.base_dir = SCRIPT_DIR / "data" / "moondev_api"
            
        self.base_dir.mkdir(parents=True, exist_ok=True)
        api_logger.info(f"Data directory set to: {self.base_dir}")

        self.api_key = api_key or os.getenv('MOONDEV_API_KEY')
        self.base_url = base_url or os.getenv('MOONDEV_API_BASE_URL', "http://api.moondev.com:8000")
        api_logger.info(f"API Base URL set to: {self.base_url}")
        
        self.headers: Dict[str, str] = {'X-API-Key': self.api_key} if self.api_key else {}
        if not self.api_key:
            api_logger.warning("API key not provided (checked constructor arg and MOONDEV_API_KEY env var). Some endpoints may fail.")
            
        self.session = requests.Session()
        self.max_retries = 3
        self.chunk_size = 8192 * 16

    def _fetch_csv(self, filename: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Fetch CSV data from the API with retry logic"""
        retry_delay = 2  # seconds
        temp_file: Path = self.base_dir / f"temp_{filename}" # Define here for broader scope
        
        for attempt in range(self.max_retries):
            try:
                url = f'{self.base_url}/files/{filename}'
                if limit:
                    url += f'?limit={limit}'
                
                api_logger.info(f"Attempt {attempt + 1}/{self.max_retries}: Fetching {url}")
                response = self.session.get(url, headers=self.headers, stream=True, timeout=60) # Added timeout
                response.raise_for_status()
                
                # Ensure temp file exists only during download
                if temp_file.exists(): temp_file.unlink()
                
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                
                if temp_file.stat().st_size == 0:
                    api_logger.error(f"Downloaded file {temp_file} is empty.")
                    temp_file.unlink()
                    return None
                    
                df = pd.read_csv(temp_file)
                
                final_file = self.base_dir / filename
                # Ensure final file does not exist before renaming (overwrite)
                if final_file.exists():
                    try:
                        final_file.unlink()
                    except OSError as e_unlink:
                         api_logger.error(f"Could not remove existing file {final_file} before overwrite: {e_unlink}")
                         # Clean up temp file if we can't remove destination
                         if temp_file.exists(): temp_file.unlink()
                         return None # Can't proceed if we can't overwrite

                temp_file.rename(final_file) # Move only if successful
                api_logger.info(f"Successfully fetched and saved {filename} to {final_file}")
                return df
                
            except requests.exceptions.HTTPError as e:
                api_logger.error(f"HTTP error fetching {filename} (attempt {attempt + 1}/{self.max_retries}): {e.response.status_code} - {e.response.text}")
                # Clean up temp file on HTTP error
                if temp_file.exists(): 
                    try: temp_file.unlink() 
                    except OSError: pass
                if e.response.status_code == 403:
                    api_logger.error("Forbidden (403): Check API key and permissions. No retries.")
                    return None
                if e.response.status_code == 404:
                    api_logger.error(f"File not found (404): {url}. No retries.")
                    return None
                if attempt < self.max_retries - 1:
                    api_logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    api_logger.error(f"Failed to fetch {filename} after {self.max_retries} attempts due to HTTP error.")
                    return None
            except (requests.exceptions.ChunkedEncodingError, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                api_logger.error(f"Network/Timeout error fetching {filename} (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                # Clean up temp file on network error
                if temp_file.exists(): 
                    try: temp_file.unlink() 
                    except OSError: pass
                if attempt < self.max_retries - 1:
                    api_logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    api_logger.error(f"Failed to fetch {filename} after {self.max_retries} attempts.")
                    return None
                    
            except pd.errors.EmptyDataError:
                api_logger.error(f"Error processing {filename}: The downloaded file {temp_file} is empty or not valid CSV.")
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError as e_unlink:
                        api_logger.warning(f"Could not delete invalid temp file {temp_file}: {e_unlink}")
                return None
            except Exception as e:
                api_logger.error(f"Unexpected error fetching {filename} (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                api_logger.debug(f"Traceback: {traceback.format_exc()}") # Debug level for full traceback
                # Clean up temp file on unexpected error
                if temp_file.exists(): 
                    try: temp_file.unlink() 
                    except OSError: pass
                if attempt < self.max_retries - 1:
                    api_logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    api_logger.error(f"Failed to fetch {filename} after {self.max_retries} attempts due to unexpected error.")
                    return None
        return None # Should not be reached if loop finishes, but explicit return

    def get_liquidation_data(self, limit: Optional[int] = 10000) -> Optional[pd.DataFrame]:
        """Get liquidation data from API, limited to last N rows by default"""
        return self._fetch_csv("liq_data.csv", limit=limit)

    def get_funding_data(self) -> Optional[pd.DataFrame]:
        """Get funding data from API"""
        return self._fetch_csv("funding.csv")

    def get_token_addresses(self) -> Optional[pd.DataFrame]:
        """Get token addresses from API (CSV format expected)"""
        return self._fetch_csv("new_token_addresses.csv")

    def get_oi_total(self) -> Optional[pd.DataFrame]:
        """Get total open interest data from API"""
        return self._fetch_csv("oi_total.csv")

    def get_oi_data(self) -> Optional[pd.DataFrame]:
        """Get detailed open interest data from API"""
        return self._fetch_csv("oi.csv")

    def get_copybot_follow_list(self) -> Optional[pd.DataFrame]:
        """Get current copy trading follow list"""
        endpoint = "copybot/data/follow_list"
        filename = "follow_list.csv"
        api_logger.info(f"📋 Moon Dev CopyBot: Fetching {filename}...")
        if not self.api_key:
            api_logger.warning(f"❗ API key is required for {endpoint}")
            return None
            
        try:
            response = self.session.get(f"{self.base_url}/{endpoint}", headers=self.headers, timeout=30)
            response.raise_for_status()
            
            save_path = self.base_dir / filename
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            df = pd.read_csv(save_path)
            api_logger.info(f"✨ Successfully loaded {len(df)} rows from {filename}. Saved to {save_path}")
            return df
                
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            api_logger.error(f"HTTP error {status_code} fetching {endpoint}: {e.response.text}")
            if status_code == 403:
                api_logger.error(f"Forbidden (403): Check API key permissions. Key used: {'****' + self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else 'N/A'}")
            return None
        except requests.exceptions.RequestException as e:
            api_logger.error(f"Network error fetching {endpoint}: {str(e)}")
            return None
        except Exception as e:
            api_logger.error(f"💥 Error fetching/processing {filename}: {str(e)}")
            api_logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def get_copybot_recent_transactions(self) -> Optional[pd.DataFrame]:
        """Get recent copy trading transactions"""
        endpoint = "copybot/data/recent_txs"
        filename = "recent_txs.csv"
        api_logger.info(f"🔄 Moon Dev CopyBot: Fetching {filename}...")
        if not self.api_key:
            api_logger.warning(f"❗ API key is required for {endpoint}")
            return None

        try:
            response = self.session.get(f"{self.base_url}/{endpoint}", headers=self.headers, timeout=30)
            response.raise_for_status()
            
            save_path = self.base_dir / filename
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            df = pd.read_csv(save_path)
            api_logger.info(f"✨ Successfully loaded {len(df)} rows from {filename}. Saved to: {save_path}")
            return df

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            api_logger.error(f"HTTP error {status_code} fetching {endpoint}: {e.response.text}")
            if status_code == 403:
                 api_logger.error(f"Forbidden (403): Check API key permissions. Key used: {'****' + self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else 'N/A'}")
            return None
        except requests.exceptions.RequestException as e:
            api_logger.error(f"Network error fetching {endpoint}: {str(e)}")
            return None
        except Exception as e:
            api_logger.error(f"💥 Error fetching/processing {filename}: {str(e)}")
            api_logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def get_agg_positions_hlp(self) -> Optional[pd.DataFrame]:
        """Get aggregated positions on HLP data"""
        return self._fetch_csv("agg_positions_on_hlp.csv")

    def get_positions_hlp(self) -> Optional[pd.DataFrame]:
        """Get detailed positions on HLP data"""
        return self._fetch_csv("positions_on_hlp.csv")

    def get_whale_addresses(self) -> Optional[List[str]]:
        """Get list of whale addresses (TXT format expected)"""
        filename = "whale_addresses.txt"
        url = f'{self.base_url}/files/{filename}'
        save_path = self.base_dir / filename
        api_logger.info(f"🐋 Moon Dev API: Fetching {filename}...")
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Save content directly
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # Read back the text file
            with open(save_path, 'r') as f:
                addresses = f.read().splitlines()
             
            # Basic validation: Check if it's a list and not empty
            if isinstance(addresses, list) and addresses:
                 api_logger.info(f"✨ Successfully loaded {len(addresses)} whale addresses. Saved to: {save_path}")
                 return addresses
            else:
                api_logger.error(f"Failed to parse whale addresses from {save_path}. Content might be invalid.")
                return None

        except requests.exceptions.HTTPError as e:
            api_logger.error(f"HTTP error {e.response.status_code} fetching {filename}: {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            api_logger.error(f"Network error fetching {filename}: {str(e)}")
            return None
        except Exception as e:
            api_logger.error(f"💥 Error fetching/processing {filename}: {str(e)}")
            api_logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

if __name__ == "__main__":
    # Setup basic logging for the test suite
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    api_logger.info("🌙 Moon Dev API Test Suite 🚀")
    api_logger.info("=" * 50)
    
    api = MoonDevAPI()
    
    # api_logger.info("\n📊 Testing Data Storage...")
    # api_logger.info(f"📂 Data will be saved to: {api.base_dir}")
    
    # # Test Historical Liquidation Data
    # api_logger.info("\n💥 Testing Liquidation Data...")
    # liq_data = api.get_liquidation_data(limit=10000)
    # if liq_data is not None:
    #     api_logger.info(f"✨ Latest Liquidation Data Preview:\n{liq_data.head()}")
    
    # # Test Funding Rate Data
    # api_logger.info("\n💰 Testing Funding Data...")
    # funding_data = api.get_funding_data()
    # if funding_data is not None:
    #     api_logger.info(f"✨ Latest Funding Data Preview:\n{funding_data.head()}")
    
    # # Test Token Addresses
    # api_logger.info("\n🔑 Testing Token Addresses...")
    # token_data = api.get_token_addresses()
    # if token_data is not None:
    #     api_logger.info(f"✨ Token Addresses Preview:\n{token_data.head()}")
    
    # # Test Total OI Data for ETH & BTC combined
    # api_logger.info("\n📈 Testing Total OI Data...")
    # oi_total = api.get_oi_total()
    # if oi_total is not None:
    #     api_logger.info(f"✨ Total OI Data Preview:\n{oi_total.head()}")
    
    # # Test Detailed OI Data for ETH or BTC
    # api_logger.info("\n📊 Testing Detailed OI Data...")
    # oi_data = api.get_oi_data()
    # if oi_data is not None:
    #     api_logger.info(f"✨ Detailed OI Data Preview:\n{oi_data.head()}")
    
    # # Test CopyBot Follow List
    # api_logger.info("\n👥 Testing CopyBot Follow List...")
    # follow_list = api.get_copybot_follow_list()
    # if follow_list is not None:
    #     api_logger.info(f"✨ Follow List Preview:\n{follow_list.head()}")
    
    # # Test CopyBot Recent Transactions
    # api_logger.info("\n💸 Testing CopyBot Recent Transactions...")
    # recent_txs = api.get_copybot_recent_transactions()
    # if recent_txs is not None:
    #     api_logger.info(f"✨ Recent Transactions Preview:\n{recent_txs.head()}")
    
    api_logger.info("\n📊 Testing Aggregated HLP Positions...")
    agg_positions = api.get_agg_positions_hlp()
    if agg_positions is not None:
        api_logger.info(f"✨ Aggregated HLP Positions Preview:\n{agg_positions.head()}")

    api_logger.info("\n📊 Testing Detailed HLP Positions...")
    positions = api.get_positions_hlp()
    if positions is not None:
        api_logger.info(f"✨ Detailed HLP Positions Preview:\n{positions.head()}")
    else:
        api_logger.warning("Failed to retrieve Detailed HLP Positions.")

    api_logger.info("\n🐋 Testing Whale Addresses...")
    whale_addresses = api.get_whale_addresses()
    if whale_addresses is not None:
        api_logger.info(f"✨ First few whale addresses:\n{whale_addresses[:5]}")
    else:
        api_logger.warning("Failed to retrieve Whale Addresses.")
    
    api_logger.info("\n✨ Moon Dev API Test Complete! ✨")
    api_logger.info("\n💡 Check logs for details. Ensure MOONDEV_API_KEY is set for restricted endpoints.")
