'''
We're building a bot that looks at all of the open positions here on Hyperliquid. 

The tricky part is going to be getting the addresses of the whales. 
Because I know how to get positions for anybody. That's easy money. 
It's just, who are those whales and how do we identify them? 

todo 
- Make a list of people with big deposits and then follow those or see what they're up to and see their position. 

list of adderess of potentional whales
https://dune.com/x3research/hyperliquid
    - i got stopped at the start of the 4th page here
https://dune.com/kouei/hyperliquid-usdc-deposit
    - i put the first 500 on the list
    

all hyperliquid protocols
https://data.asxn.xyz/dashboard/hyperliquid-ecosystem
https://hyperdash.info/ -- this is a good one
'''

import time
import pandas as pd
import requests
from datetime import datetime
import numpy as np
import concurrent.futures
from tqdm import tqdm
import colorama
import argparse
from pathlib import Path
import logging # Add logging import

# --- Setup Logging ---
# Basic configuration sets up a default handler. 
# For more advanced scenarios, you might configure handlers, formatters, etc.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas display options
pd.set_option('display.float_format', '{:.2f}'.format)

# --- Configuration ---
CONFIG = {
    "API_URL": "https://api.hyperliquid.xyz/info",
    "HEADERS": {"Content-Type": "application/json"},
    "MIN_POSITION_VALUE": 25000,
    "MAX_WORKERS": 10,
    "API_REQUEST_DELAY": 0.1  # Default value, can be overridden by argparse
}

DATA_PATH = Path("bots/hyperliquid/data/ppls_positions")
WHALE_ADDRESSES_FILE = "whale_addresses.txt"
POSITIONS_CSV_FILE = "positions_on_hlp.csv"
AGG_POSITIONS_CSV_FILE = "agg_positions_on_hlp.csv"

# --- DataFrame Column Constants ---
COL_ADDRESS = "address"
COL_COIN = "coin"
COL_ENTRY_PRICE = "entry_price"
COL_LEVERAGE = "leverage"
COL_POSITION_VALUE = "position_value"
COL_UNREALIZED_PNL = "unrealized_pnl"
COL_LIQUIDATION_PRICE = "liquidation_price"
COL_IS_LONG = "is_long"
COL_TIMESTAMP = "timestamp"
COL_DIRECTION = "direction"
COL_NUM_TRADERS = "num_traders"
COL_TOTAL_VALUE = "total_value"
COL_TOTAL_PNL = "total_pnl"
COL_AVG_LEVERAGE = "avg_leverage"


def load_wallet_addresses():
    """Load wallet addresses from text file"""
    addresses_file = DATA_PATH / WHALE_ADDRESSES_FILE
    try:
        with open(addresses_file, 'r') as f:
            addresses = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"🌙 Loaded {len(addresses)} addresses from {addresses_file}")
        return addresses
    except FileNotFoundError:
        logger.warning(f"⚠️ Wallet addresses file not found: {addresses_file}")
        return []
    except Exception as e:
        logger.error(f"⚠️ Error loading addresses from {addresses_file}: {str(e)}")
        return []

def ensure_data_dir() -> bool: # Added type hint
    """Ensure the data directory exists"""
    try:
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Data directory ensured: {DATA_PATH}")
        return True
    except Exception as e:
        logger.error(f"⚠️ Error creating directory {DATA_PATH}: {str(e)}")
        return False

def get_positions_for_address(address: str) -> tuple[dict | None, str]: # Added type hints
    """Fetch positions for a specific wallet address"""
    max_retries = 3
    base_delay = 0.5
    
    for retry in range(max_retries):
        try:
            payload = {
                "type": "clearinghouseState",
                "user": address
            }
            
            response = requests.post(CONFIG["API_URL"], headers=CONFIG["HEADERS"], json=payload)
            
            if response.status_code == 429:
                delay = base_delay * (2 ** retry)
                logger.warning(f"⚠️ Rate limit hit for {address[:6]}...{address[-4:]}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
                continue
                
            response.raise_for_status()
            return response.json(), address
            
        except requests.exceptions.RequestException as e: # More specific exception
            if retry == max_retries - 1:
                logger.error(f"❌ Max retries reached. Error fetching positions for {address[:6]}...{address[-4:]}: {str(e)}")
            else:
                logger.warning(f"❌ Error fetching positions for {address[:6]}...{address[-4:]} (retry {retry+1}/{max_retries}): {str(e)}. Retrying in {base_delay * (2**retry):.2f}s...")
                delay = base_delay * (2 ** retry)
                time.sleep(delay) # Added delay on other request exceptions too
            
    return None, address

def process_positions(data, address):
    """Process the position data"""
    if not data or "assetPositions" not in data:
        return []
    
    positions = []
    for pos_data in data["assetPositions"]: # Renamed pos to pos_data for clarity
        if "position" in pos_data:
            p = pos_data["position"]
            coin_name = p.get("coin", "UNKNOWN_COIN") # Get coin name for logging
            
            try:
                size = float(p.get("szi", "0"))
                position_value = float(p.get("positionValue", "0"))
                
                if position_value < CONFIG["MIN_POSITION_VALUE"]:
                    continue
                    
                position_info = {
                    COL_ADDRESS: address,
                    COL_COIN: coin_name,
                    COL_ENTRY_PRICE: float(p.get("entryPx", "0")),
                    COL_LEVERAGE: p.get("leverage", {}).get("value", 0),
                    COL_POSITION_VALUE: position_value,
                    COL_UNREALIZED_PNL: float(p.get("unrealizedPnl", "0")),
                    COL_LIQUIDATION_PRICE: float(p.get("liquidationPx", "0") or 0), # Ensure "0" becomes 0.0
                    COL_IS_LONG: size > 0,
                    COL_TIMESTAMP: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                positions.append(position_info)
                
            except (ValueError, TypeError) as e: # Catch specific errors for conversion
                logger.warning(f"⚠️ Skipping position for {address[:6]}...{address[-4:]} (Coin: {coin_name}) due to data conversion error: {e}. Data: {p}")
                continue
    
    return positions

def process_address_data(address: str) -> list[dict]: # Added type hint
    """Process a single address - for parallel execution"""
    time.sleep(CONFIG["API_REQUEST_DELAY"])
    data, fetched_address = get_positions_for_address(address) # Renamed address to fetched_address for clarity
    if data:
        return process_positions(data, fetched_address)
    return []

def save_positions_to_csv(all_positions):
    """Save positions to CSV files"""
    if not all_positions:
        logger.info("No positions found to save!")
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(all_positions)
    
    # Format numeric columns
    numeric_cols = [COL_ENTRY_PRICE, COL_POSITION_VALUE, COL_UNREALIZED_PNL, COL_LIQUIDATION_PRICE]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # Save all positions
    positions_file = DATA_PATH / POSITIONS_CSV_FILE
    try:
        df.to_csv(positions_file, index=False, float_format='%.2f')
        logger.info(f"✨ Saved {len(all_positions)} positions to {positions_file}")
    except Exception as e:
        logger.error(f"⚠️ Error saving positions to {positions_file}: {str(e)}")
        return df, None # Return df even if agg fails or main CSV fails
    
    # Create and save aggregated view
    try:
        agg_df = df.groupby([COL_COIN, COL_IS_LONG]).agg(
            total_value=(COL_POSITION_VALUE, 'sum'),
            total_pnl=(COL_UNREALIZED_PNL, 'sum'),
            num_traders=(COL_ADDRESS, 'count'),
            avg_leverage=(COL_LEVERAGE, 'mean'),
            avg_liquidation_price=(COL_LIQUIDATION_PRICE, lambda x: np.nanmean(x) if not x.empty and not x.isnull().all() else np.nan)
        ).reset_index()
        
        agg_df[COL_DIRECTION] = agg_df[COL_IS_LONG].apply(lambda x: 'LONG' if x else 'SHORT')
        
        # Rename columns for agg_df (already done via named aggregation)
        # No, named aggregation renames them, but if you want to stick to constants, you would map them.
        # For now, the named aggregation output is clear. Let's refine column names based on constants if needed later.
        # Let's ensure the output names match the constants if we want to use them consistently.
        agg_df = agg_df.rename(columns={
            'total_value': COL_TOTAL_VALUE,
            'total_pnl': COL_TOTAL_PNL,
            'num_traders': COL_NUM_TRADERS,
            'avg_leverage': COL_AVG_LEVERAGE,
            'avg_liquidation_price': COL_LIQUIDATION_PRICE # Reusing, though it's an average here.
        })

        agg_df = agg_df.sort_values(COL_TOTAL_VALUE, ascending=False)
        
        agg_file = DATA_PATH / AGG_POSITIONS_CSV_FILE
        agg_df.to_csv(agg_file, index=False, float_format='%.2f')
        logger.info(f"✨ Saved aggregated positions to {agg_file}")
        return df, agg_df
    except Exception as e:
        logger.error(f"⚠️ Error creating or saving aggregated positions view: {str(e)}")
        return df, None


def fetch_all_positions_parallel(addresses: list[str]) -> list[dict]: # Added type hint
    """Fetch positions for all addresses in parallel"""
    total_addresses = len(addresses)
    logger.info(f"🚀 Processing {total_addresses} addresses with {CONFIG['MAX_WORKERS']} workers using {CONFIG['API_REQUEST_DELAY']}s delay per request.")
    
    all_positions_list = [] # Renamed for clarity
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        future_to_address = {executor.submit(process_address_data, address): address for address in addresses}
        
        with tqdm(total=total_addresses, desc="Fetching positions") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_address):
                try:
                    positions_result = future.result() # Renamed for clarity
                    if positions_result:
                        all_positions_list.extend(positions_result)
                except Exception as e:
                    address_associated = future_to_address[future]
                    logger.error(f"❌ Error processing future for address {address_associated[:6]}...{address_associated[-4:]}: {str(e)}")
                progress_bar.update(1)
                
    logger.info(f"✅ Found {len(all_positions_list)} total positions from {progress_bar.n} out of {total_addresses} addresses processed.")
    return all_positions_list

def main():
    """Main function to run the position tracker"""
    parser = argparse.ArgumentParser(description="HyperLiquid Position Tracker")
    parser.add_argument('--delay', type=float, 
                        help=f'Delay between API requests in seconds (default: {CONFIG["API_REQUEST_DELAY"]})')
    parser.add_argument('--min_pos_value', type=int, 
                        help=f'Minimum position value to track (default: {CONFIG["MIN_POSITION_VALUE"]})')
    parser.add_argument('--max_workers', type=int, 
                        help=f'Maximum number of parallel workers (default: {CONFIG["MAX_WORKERS"]})')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Set log level from command line
    # Ensures that the numeric value of the level is used for setLevel
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True) # force=True to override basicConfig if called before
    logger.setLevel(numeric_level) # Set logger's level as well

    # Update CONFIG with command-line arguments if they are provided (i.e., not None)
    if args.delay is not None:
        CONFIG["API_REQUEST_DELAY"] = args.delay
    if args.min_pos_value is not None:
        CONFIG["MIN_POSITION_VALUE"] = args.min_pos_value
    if args.max_workers is not None:
        CONFIG["MAX_WORKERS"] = args.max_workers
    
    logger.info(f"Script configuration: {CONFIG}")

    if not ensure_data_dir():
        logger.error("Exiting due to issues with data directory.")
        return

    addresses = load_wallet_addresses()
    if not addresses:
        logger.warning("⚠️ No addresses loaded! Exiting...")
        return
    
    all_positions = fetch_all_positions_parallel(addresses)
    if not all_positions:
        logger.info("No positions were fetched. Exiting.")
        return

    positions_df, agg_df = save_positions_to_csv(all_positions)
    
    if positions_df is not None:
        logger.info(f"\n--- Sample of All Positions ({len(positions_df)} entries) ---")
        print(positions_df.head().to_string()) # Use print for direct table output, logger for messages
    
    if agg_df is not None:
        logger.info(f"\n--- Aggregated Positions Summary ({len(agg_df)} entries) ---")
        print(agg_df.head().to_string()) # Use print for direct table output

    logger.info("Script finished successfully.")
    return positions_df, agg_df

if __name__ == "__main__":
    main()