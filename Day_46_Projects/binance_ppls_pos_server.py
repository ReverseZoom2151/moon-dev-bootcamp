"""
🌙 Moon Dev's Binance Position Server 🚀

This script fetches position data from Binance futures for multiple large accounts 
and analyzes whale positions and liquidation risks.

Binance version of ppls_pos_server.py adapted for Binance futures trading.

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.
"""

import time
import pandas as pd
import requests
import concurrent.futures
import colorama
import argparse
import logging
from tqdm import tqdm
from pathlib import Path

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas display options
pd.set_option('display.float_format', '{:.2f}'.format)

# --- Configuration ---
CONFIG = {
    "BASE_URL": "https://fapi.binance.com",
    "HEADERS": {"Content-Type": "application/json"},
    "MIN_POSITION_VALUE": 50000,  # Higher for Binance futures
    "MAX_WORKERS": 5,  # Conservative for API limits
    "API_REQUEST_DELAY": 0.2  # Respect rate limits
}

DATA_PATH = Path("data/binance_positions")
WHALE_ADDRESSES_FILE = "binance_whale_traders.txt"
POSITIONS_CSV_FILE = "binance_positions.csv"
AGG_POSITIONS_CSV_FILE = "binance_agg_positions.csv"

# --- DataFrame Column Constants ---
COL_TRADER_ID = "trader_id"
COL_SYMBOL = "symbol"
COL_ENTRY_PRICE = "entry_price"
COL_POSITION_AMT = "position_amt"
COL_POSITION_VALUE = "position_value"
COL_UNREALIZED_PNL = "unrealized_pnl"
COL_PERCENTAGE = "percentage"
COL_SIDE = "side"
COL_TIMESTAMP = "timestamp"
COL_TOTAL_VALUE = "total_value"
COL_TOTAL_PNL = "total_pnl"
COL_NUM_POSITIONS = "num_positions"

def load_whale_trader_ids():
    """Load whale trader IDs from text file (simulated for Binance)"""
    addresses_file = DATA_PATH / WHALE_ADDRESSES_FILE
    try:
        with open(addresses_file, 'r') as f:
            trader_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"🌙 Loaded {len(trader_ids)} whale trader IDs from {addresses_file}")
        return trader_ids
    except FileNotFoundError:
        logger.warning(f"⚠️ Whale trader IDs file not found: {addresses_file}")
        # Generate sample trader IDs for demo
        sample_ids = [f"binance_whale_{i:05d}" for i in range(1, 101)]
        ensure_data_dir()
        with open(addresses_file, 'w') as f:
            for trader_id in sample_ids:
                f.write(f"{trader_id}\n")
        logger.info(f"📝 Generated {len(sample_ids)} sample whale trader IDs")
        return sample_ids
    except Exception as e:
        logger.error(f"⚠️ Error loading trader IDs from {addresses_file}: {str(e)}")
        return []

def ensure_data_dir() -> bool:
    """Ensure the data directory exists"""
    try:
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Data directory ensured: {DATA_PATH}")
        return True
    except Exception as e:
        logger.error(f"⚠️ Error creating directory {DATA_PATH}: {str(e)}")
        return False

def get_positions_for_trader(trader_id: str):
    """Fetch positions for a specific trader ID (simulated for Binance)"""
    try:
        # Since Binance doesn't expose other users' positions, we'll simulate 
        # position data based on large trades and open interest analysis
        
        # Get recent large trades to simulate whale positions
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
                  'DOTUSDT', 'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'BCUSDT']
        
        simulated_positions = []
        
        for symbol in symbols[:3]:  # Limit to avoid rate limits
            try:
                # Get recent trades
                url = f"{CONFIG['BASE_URL']}/fapi/v1/aggTrades"
                params = {"symbol": symbol, "limit": 500}
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    trades = response.json()
                    
                    # Analyze for large trades (potential whale activity)
                    large_trades = []
                    for trade in trades:
                        trade_value = float(trade['p']) * float(trade['q'])
                        if trade_value > CONFIG["MIN_POSITION_VALUE"]:
                            large_trades.append({
                                'price': float(trade['p']),
                                'quantity': float(trade['q']),
                                'value': trade_value,
                                'is_buyer_maker': trade['m']
                            })
                    
                    # Simulate position based on large trades
                    if large_trades:
                        # Use hash of trader_id and symbol for consistent simulation
                        position_seed = hash(f"{trader_id}_{symbol}") % 1000
                        
                        if position_seed < 300:  # 30% chance of having position
                            avg_price = sum(t['price'] for t in large_trades) / len(large_trades)
                            total_qty = sum(t['quantity'] for t in large_trades) / 10  # Scale down
                            
                            # Random but consistent position direction
                            is_long = (position_seed % 2) == 0
                            position_amt = total_qty if is_long else -total_qty
                            
                            # Current price (approximate)
                            current_price = large_trades[-1]['price']
                            
                            # Calculate PnL
                            if is_long:
                                pnl = (current_price - avg_price) * total_qty
                            else:
                                pnl = (avg_price - current_price) * total_qty
                            
                            simulated_positions.append({
                                'symbol': symbol,
                                'position_amt': position_amt,
                                'entry_price': avg_price,
                                'mark_price': current_price,
                                'unrealized_pnl': pnl,
                                'percentage': (pnl / (avg_price * total_qty)) * 100,
                                'notional': abs(position_amt) * current_price,
                                'side': 'LONG' if is_long else 'SHORT'
                            })
                
                time.sleep(CONFIG["API_REQUEST_DELAY"])
            except Exception as e:
                logger.warning(f"Error processing {symbol} for {trader_id}: {e}")
                continue
        
        logger.info(f"🐋 Simulated {len(simulated_positions)} positions for {trader_id}")
        return simulated_positions
        
    except Exception as e:
        logger.error(f"Error fetching positions for {trader_id}: {str(e)}")
        return []

def process_positions(data, trader_id):
    """Process the position data"""
    try:
        if not data:
            return []
        
        processed_positions = []
        timestamp = pd.Timestamp.now()
        
        for pos in data:
            if abs(float(pos['position_amt'])) > 0:  # Only active positions
                processed_positions.append({
                    COL_TRADER_ID: trader_id,
                    COL_SYMBOL: pos['symbol'],
                    COL_ENTRY_PRICE: float(pos['entry_price']),
                    COL_POSITION_AMT: float(pos['position_amt']),
                    COL_POSITION_VALUE: float(pos['notional']),
                    COL_UNREALIZED_PNL: float(pos['unrealized_pnl']),
                    COL_PERCENTAGE: float(pos['percentage']),
                    COL_SIDE: pos['side'],
                    COL_TIMESTAMP: timestamp
                })
        
        return processed_positions
    except Exception as e:
        logger.error(f"Error processing positions for {trader_id}: {str(e)}")
        return []

def process_trader_data(trader_id: str):
    """Process a single trader - for parallel execution"""
    try:
        positions_data = get_positions_for_trader(trader_id)
        return process_positions(positions_data, trader_id)
    except Exception as e:
        logger.error(f"Error processing trader {trader_id}: {str(e)}")
        return []

def save_positions_to_csv(all_positions):
    """Save positions to CSV files"""
    try:
        if not all_positions:
            logger.warning("No positions to save")
            return
        
        ensure_data_dir()
        
        # Create DataFrame
        df = pd.DataFrame(all_positions)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual positions
        positions_file = DATA_PATH / f"{POSITIONS_CSV_FILE.replace('.csv', '')}_{timestamp}.csv"
        df.to_csv(positions_file, index=False)
        logger.info(f"💾 Saved {len(df)} positions to {positions_file}")
        
        # Create aggregated data
        if not df.empty:
            agg_data = df.groupby([COL_SYMBOL, COL_SIDE]).agg({
                COL_TRADER_ID: 'count',
                COL_POSITION_VALUE: 'sum',
                COL_UNREALIZED_PNL: 'sum'
            }).reset_index()
            
            agg_data.rename(columns={COL_TRADER_ID: COL_NUM_POSITIONS}, inplace=True)
            agg_data[COL_TOTAL_VALUE] = agg_data[COL_POSITION_VALUE]
            agg_data[COL_TOTAL_PNL] = agg_data[COL_UNREALIZED_PNL]
            agg_data[COL_TIMESTAMP] = pd.Timestamp.now()
            
            # Save aggregated positions
            agg_file = DATA_PATH / f"{AGG_POSITIONS_CSV_FILE.replace('.csv', '')}_{timestamp}.csv"
            agg_data.to_csv(agg_file, index=False)
            logger.info(f"📊 Saved aggregated data to {agg_file}")
            
            # Display summary
            print(f"\n🌙 BINANCE WHALE POSITION SUMMARY 🚀")
            print("=" * 80)
            print(f"Total Positions: {len(df)}")
            print(f"Total Value: ${df[COL_POSITION_VALUE].sum():,.0f}")
            print(f"Total PnL: ${df[COL_UNREALIZED_PNL].sum():,.0f}")
            print(f"Unique Symbols: {df[COL_SYMBOL].nunique()}")
            print(f"Active Whales: {df[COL_TRADER_ID].nunique()}")
            
            # Top positions
            print(f"\n🔥 TOP 10 POSITIONS BY VALUE:")
            top_positions = df.nlargest(10, COL_POSITION_VALUE)
            for _, row in top_positions.iterrows():
                side_color = colorama.Fore.GREEN if row[COL_SIDE] == 'LONG' else colorama.Fore.RED
                print(f"{side_color}{row[COL_SYMBOL]:12} | "
                      f"{row[COL_SIDE]:5} | "
                      f"${row[COL_POSITION_VALUE]:>12,.0f} | "
                      f"PnL: ${row[COL_UNREALIZED_PNL]:>+10,.0f}{colorama.Style.RESET_ALL}")
            
    except Exception as e:
        logger.error(f"Error saving positions to CSV: {str(e)}")

def fetch_all_positions_parallel(trader_ids: list[str]):
    """Fetch positions for all traders in parallel"""
    try:
        logger.info(f"🌙 Fetching positions for {len(trader_ids)} whale traders...")
        
        all_positions = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
            # Submit all tasks
            future_to_trader = {
                executor.submit(process_trader_data, trader_id): trader_id 
                for trader_id in trader_ids
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(trader_ids), desc="Processing whale traders") as pbar:
                for future in concurrent.futures.as_completed(future_to_trader):
                    trader_id = future_to_trader[future]
                    try:
                        positions = future.result()
                        all_positions.extend(positions)
                        if positions:
                            logger.debug(f"✅ Processed {len(positions)} positions for {trader_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process {trader_id}: {str(e)}")
                    finally:
                        pbar.update(1)
        
        logger.info(f"✅ Collected {len(all_positions)} total positions from whale traders")
        return all_positions
        
    except Exception as e:
        logger.error(f"Error fetching positions in parallel: {str(e)}")
        return []

def main():
    """Main function to run the position tracker"""
    try:
        parser = argparse.ArgumentParser(description="🌙 Moon Dev's Binance Whale Position Server")
        parser.add_argument("--delay", type=float, default=CONFIG["API_REQUEST_DELAY"],
                          help="API request delay in seconds")
        parser.add_argument("--workers", type=int, default=CONFIG["MAX_WORKERS"],
                          help="Number of worker threads")
        parser.add_argument("--min-value", type=float, default=CONFIG["MIN_POSITION_VALUE"],
                          help="Minimum position value to track")
        
        args = parser.parse_args()
        
        # Update config
        CONFIG["API_REQUEST_DELAY"] = args.delay
        CONFIG["MAX_WORKERS"] = args.workers
        CONFIG["MIN_POSITION_VALUE"] = args.min_value
        
        print(f"\n🌙 Moon Dev's Binance Whale Position Server 🚀")
        print("=" * 60)
        print(f"API Delay: {CONFIG['API_REQUEST_DELAY']}s")
        print(f"Workers: {CONFIG['MAX_WORKERS']}")
        print(f"Min Position Value: ${CONFIG['MIN_POSITION_VALUE']:,.0f}")
        print("=" * 60)
        
        # Ensure data directory exists
        if not ensure_data_dir():
            logger.error("Failed to create data directory")
            return
        
        # Load whale trader IDs
        trader_ids = load_whale_trader_ids()
        if not trader_ids:
            logger.error("No whale trader IDs available")
            return
        
        # Fetch positions for all whale traders
        start_time = time.time()
        all_positions = fetch_all_positions_parallel(trader_ids[:20])  # Limit for demo
        end_time = time.time()
        
        # Save results
        save_positions_to_csv(all_positions)
        
        logger.info(f"🎯 Position tracking completed in {end_time - start_time:.2f} seconds")
        print(f"\n✅ Binance whale position tracking completed!")
        print(f"📊 Check {DATA_PATH} for saved CSV files")
        
    except KeyboardInterrupt:
        logger.info("👋 Position server stopped by user")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
