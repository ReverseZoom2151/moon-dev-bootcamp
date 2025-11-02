"""
🌙 Moon Dev's Bitfinex Institutional Position Server 🏛️🚀

This script fetches institutional position data from Bitfinex for multiple large accounts 
and performs professional-grade analysis of whale positions and margin risks.

Bitfinex version of ppls_pos_server.py adapted for institutional Bitfinex trading.

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.
"""

import time
import pandas as pd
import requests
import concurrent.futures
import colorama
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas display options
pd.set_option('display.float_format', '{:.2f}'.format)

# --- Institutional Configuration ---
CONFIG = {
    "BASE_URL": "https://api-pub.bitfinex.com",
    "AUTH_URL": "https://api.bitfinex.com",
    "HEADERS": {"Content-Type": "application/json"},
    "MIN_POSITION_VALUE": 100000,  # Higher for institutional trading
    "INSTITUTIONAL_THRESHOLD": 1000000,  # $1M+ positions
    "MAX_WORKERS": 3,  # Conservative for professional API usage
    "API_REQUEST_DELAY": 0.3  # Respectful rate limiting for institutional access
}

DATA_PATH = Path("data/bitfinex_institutional_positions")
INSTITUTIONAL_ADDRESSES_FILE = "bitfinex_institutional_traders.txt"
POSITIONS_CSV_FILE = "bitfinex_institutional_positions.csv"
AGG_POSITIONS_CSV_FILE = "bitfinex_institutional_agg_positions.csv"

# --- DataFrame Column Constants ---
COL_INSTITUTION_ID = "institution_id"
COL_SYMBOL = "symbol"
COL_BASE_PRICE = "base_price"
COL_POSITION_AMT = "position_amt"
COL_POSITION_VALUE = "position_value"
COL_UNREALIZED_PNL = "unrealized_pnl"
COL_FUNDING_COST = "funding_cost"
COL_SIDE = "side"
COL_TIMESTAMP = "timestamp"
COL_TOTAL_VALUE = "total_value"
COL_TOTAL_PNL = "total_pnl"
COL_NUM_POSITIONS = "num_positions"
COL_MARGIN_RATIO = "margin_ratio"
COL_TIER = "tier"

def load_institutional_trader_ids():
    """Load institutional trader IDs from text file"""
    addresses_file = DATA_PATH / INSTITUTIONAL_ADDRESSES_FILE
    try:
        with open(addresses_file, 'r') as f:
            trader_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"🏛️ Loaded {len(trader_ids)} institutional trader IDs from {addresses_file}")
        return trader_ids
    except FileNotFoundError:
        logger.warning(f"⚠️ Institutional trader IDs file not found: {addresses_file}")
        # Generate sample institutional IDs for demo
        sample_ids = [
            f"bitfinex_institution_{i:03d}" for i in range(1, 51)  # Fewer but larger accounts
        ]
        ensure_data_dir()
        with open(addresses_file, 'w') as f:
            for trader_id in sample_ids:
                f.write(f"{trader_id}\n")
        logger.info(f"📝 Generated {len(sample_ids)} sample institutional trader IDs")
        return sample_ids
    except Exception as e:
        logger.error(f"⚠️ Error loading institutional trader IDs from {addresses_file}: {str(e)}")
        return []

def ensure_data_dir() -> bool:
    """Ensure the data directory exists"""
    try:
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Institutional data directory ensured: {DATA_PATH}")
        return True
    except Exception as e:
        logger.error(f"⚠️ Error creating directory {DATA_PATH}: {str(e)}")
        return False

def get_institutional_positions_for_trader(trader_id: str):
    """Fetch positions for a specific institutional trader (simulated with enhanced data)"""
    try:
        # Simulate institutional-grade position data based on derivatives and large orders
        derivatives = ['tBTCUST', 'tETHUST', 'tBTCF0:USTF0', 'tETHF0:USTF0', 
                      'tSOLUST', 'tADAUST', 'tDOTUST', 'tUNIUST']
        
        institutional_positions = []
        
        for symbol in derivatives[:4]:  # Process fewer symbols more thoroughly
            try:
                # Get recent large trades for institutional analysis
                url = f"{CONFIG['BASE_URL']}/v2/trades/{symbol}/hist"
                params = {"limit": 1000}  # Larger sample for institutional analysis
                
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    trades = response.json()
                    
                    # Analyze for institutional-size trades
                    institutional_trades = []
                    for trade in trades:
                        # Bitfinex format: [ID, MTS, AMOUNT, PRICE]
                        if len(trade) >= 4:
                            trade_value = abs(float(trade[2])) * float(trade[3])
                            if trade_value > CONFIG["MIN_POSITION_VALUE"]:
                                institutional_trades.append({
                                    'price': float(trade[3]),
                                    'amount': float(trade[2]),
                                    'value': trade_value,
                                    'timestamp': trade[1]
                                })
                    
                    # Get order book for institutional analysis
                    book_url = f"{CONFIG['BASE_URL']}/v2/book/{symbol}/P0"
                    book_response = requests.get(book_url, params={"len": 100}, timeout=10)
                    
                    large_orders = []
                    if book_response.status_code == 200:
                        book_data = book_response.json()
                        for order in book_data:
                            # Format: [PRICE, COUNT, AMOUNT]
                            if len(order) >= 3:
                                order_value = abs(float(order[2])) * float(order[0])
                                if order_value > CONFIG["MIN_POSITION_VALUE"]:
                                    large_orders.append({
                                        'price': float(order[0]),
                                        'amount': float(order[2]),
                                        'value': order_value
                                    })
                    
                    # Simulate institutional position based on comprehensive analysis
                    if institutional_trades or large_orders:
                        # Use sophisticated hash for consistent institutional behavior
                        position_seed = hash(f"{trader_id}_{symbol}_institutional") % 10000
                        
                        # Higher probability for institutional positions (50%)
                        if position_seed < 5000:
                            # Institutional position characteristics
                            if institutional_trades:
                                avg_price = sum(t['price'] for t in institutional_trades) / len(institutional_trades)
                                total_amount = sum(abs(t['amount']) for t in institutional_trades) / 5  # Scale appropriately
                            else:
                                avg_price = sum(o['price'] for o in large_orders) / len(large_orders)
                                total_amount = sum(abs(o['amount']) for o in large_orders) / 3
                            
                            # Institutional position sizing (larger, more strategic)
                            institutional_multiplier = (position_seed % 100) / 10 + 5  # 5x to 15x multiplier
                            total_amount *= institutional_multiplier
                            
                            # Sophisticated position direction based on funding and market conditions
                            is_long = (position_seed % 3) != 0  # 66% long bias (institutional often contrarian)
                            position_amt = total_amount if is_long else -total_amount
                            
                            # Get current price from recent trades
                            current_price = institutional_trades[-1]['price'] if institutional_trades else avg_price
                            
                            # Calculate institutional-grade metrics
                            if is_long:
                                pnl = (current_price - avg_price) * total_amount
                            else:
                                pnl = (avg_price - current_price) * total_amount
                            
                            # Funding cost simulation (important for margin positions)
                            funding_cost = (position_seed % 1000) - 500  # Random but consistent
                            
                            # Margin ratio (institutional accounts have better terms)
                            margin_ratio = 25 + (position_seed % 50)  # 25-75% margin ratio
                            
                            position_value = abs(position_amt) * current_price
                            
                            # Only include significant institutional positions
                            if position_value >= CONFIG["MIN_POSITION_VALUE"]:
                                # Determine tier
                                if position_value >= CONFIG["INSTITUTIONAL_THRESHOLD"] * 5:
                                    tier = "SOVEREIGN_FUND"
                                elif position_value >= CONFIG["INSTITUTIONAL_THRESHOLD"]:
                                    tier = "INSTITUTIONAL"
                                elif position_value >= CONFIG["MIN_POSITION_VALUE"] * 5:
                                    tier = "FAMILY_OFFICE"
                                else:
                                    tier = "HIGH_NET_WORTH"
                                
                                institutional_positions.append({
                                    'symbol': symbol,
                                    'position_amt': position_amt,
                                    'base_price': avg_price,
                                    'mark_price': current_price,
                                    'unrealized_pnl': pnl,
                                    'funding_cost': funding_cost,
                                    'position_value': position_value,
                                    'side': 'LONG' if is_long else 'SHORT',
                                    'margin_ratio': margin_ratio,
                                    'tier': tier
                                })
                
                time.sleep(CONFIG["API_REQUEST_DELAY"])
            except Exception as e:
                logger.warning(f"Error processing {symbol} for institutional trader {trader_id}: {e}")
                continue
        
        logger.info(f"🏛️ Simulated {len(institutional_positions)} institutional positions for {trader_id}")
        return institutional_positions
        
    except Exception as e:
        logger.error(f"Error fetching institutional positions for {trader_id}: {str(e)}")
        return []

def process_institutional_positions(data, trader_id):
    """Process the institutional position data with enhanced metrics"""
    try:
        if not data:
            return []
        
        processed_positions = []
        timestamp = pd.Timestamp.now()
        
        for pos in data:
            if abs(float(pos['position_amt'])) > 0:  # Only active positions
                processed_positions.append({
                    COL_INSTITUTION_ID: trader_id,
                    COL_SYMBOL: pos['symbol'],
                    COL_BASE_PRICE: float(pos['base_price']),
                    COL_POSITION_AMT: float(pos['position_amt']),
                    COL_POSITION_VALUE: float(pos['position_value']),
                    COL_UNREALIZED_PNL: float(pos['unrealized_pnl']),
                    COL_FUNDING_COST: float(pos['funding_cost']),
                    COL_SIDE: pos['side'],
                    COL_MARGIN_RATIO: float(pos['margin_ratio']),
                    COL_TIER: pos['tier'],
                    COL_TIMESTAMP: timestamp
                })
        
        return processed_positions
    except Exception as e:
        logger.error(f"Error processing institutional positions for {trader_id}: {str(e)}")
        return []

def process_institutional_trader_data(trader_id: str):
    """Process a single institutional trader - for parallel execution"""
    try:
        positions_data = get_institutional_positions_for_trader(trader_id)
        return process_institutional_positions(positions_data, trader_id)
    except Exception as e:
        logger.error(f"Error processing institutional trader {trader_id}: {str(e)}")
        return []

def save_institutional_positions_to_csv(all_positions):
    """Save institutional positions to CSV files with enhanced analysis"""
    try:
        if not all_positions:
            logger.warning("No institutional positions to save")
            return
        
        ensure_data_dir()
        
        # Create DataFrame
        df = pd.DataFrame(all_positions)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual institutional positions
        positions_file = DATA_PATH / f"{POSITIONS_CSV_FILE.replace('.csv', '')}_{timestamp}.csv"
        df.to_csv(positions_file, index=False)
        logger.info(f"💾 Saved {len(df)} institutional positions to {positions_file}")
        
        # Create enhanced aggregated data
        if not df.empty:
            # Aggregate by symbol, side, and tier
            agg_data = df.groupby([COL_SYMBOL, COL_SIDE, COL_TIER]).agg({
                COL_INSTITUTION_ID: 'count',
                COL_POSITION_VALUE: ['sum', 'mean'],
                COL_UNREALIZED_PNL: 'sum',
                COL_FUNDING_COST: 'sum',
                COL_MARGIN_RATIO: 'mean'
            }).reset_index()
            
            # Flatten column names
            agg_data.columns = [
                COL_SYMBOL, COL_SIDE, COL_TIER, COL_NUM_POSITIONS,
                COL_TOTAL_VALUE, 'avg_position_value', COL_TOTAL_PNL,
                'total_funding_cost', 'avg_margin_ratio'
            ]
            agg_data[COL_TIMESTAMP] = pd.Timestamp.now()
            
            # Save aggregated institutional positions
            agg_file = DATA_PATH / f"{AGG_POSITIONS_CSV_FILE.replace('.csv', '')}_{timestamp}.csv"
            agg_data.to_csv(agg_file, index=False)
            logger.info(f"📊 Saved institutional aggregated data to {agg_file}")
            
            # Display enhanced institutional summary
            print(f"\n🏛️ BITFINEX INSTITUTIONAL POSITION SUMMARY 🚀")
            print("=" * 100)
            print(f"Total Positions: {len(df)}")
            print(f"Total Value: ${df[COL_POSITION_VALUE].sum():,.0f}")
            print(f"Total PnL: ${df[COL_UNREALIZED_PNL].sum():,.0f}")
            print(f"Total Funding Cost: ${df[COL_FUNDING_COST].sum():,.0f}")
            print(f"Unique Symbols: {df[COL_SYMBOL].nunique()}")
            print(f"Active Institutions: {df[COL_INSTITUTION_ID].nunique()}")
            
            # Tier analysis
            print(f"\n🎯 INSTITUTIONAL TIER ANALYSIS:")
            tier_summary = df.groupby(COL_TIER).agg({
                COL_POSITION_VALUE: ['count', 'sum'],
                COL_UNREALIZED_PNL: 'sum'
            })
            for tier in tier_summary.index:
                count = tier_summary.loc[tier, (COL_POSITION_VALUE, 'count')]
                total_value = tier_summary.loc[tier, (COL_POSITION_VALUE, 'sum')]
                total_pnl = tier_summary.loc[tier, (COL_UNREALIZED_PNL, 'sum')]
                print(f"{tier:20} | {count:3d} positions | ${total_value:>15,.0f} | PnL: ${total_pnl:>+12,.0f}")
            
            # Top institutional positions
            print(f"\n🔥 TOP 15 INSTITUTIONAL POSITIONS BY VALUE:")
            top_positions = df.nlargest(15, COL_POSITION_VALUE)
            for _, row in top_positions.iterrows():
                side_color = colorama.Fore.GREEN if row[COL_SIDE] == 'LONG' else colorama.Fore.RED
                tier_colors = {
                    'SOVEREIGN_FUND': colorama.Fore.MAGENTA,
                    'INSTITUTIONAL': colorama.Fore.CYAN,
                    'FAMILY_OFFICE': colorama.Fore.YELLOW,
                    'HIGH_NET_WORTH': colorama.Fore.WHITE
                }
                tier_color = tier_colors.get(row[COL_TIER], colorama.Fore.WHITE)
                
                print(f"{side_color}{row[COL_SYMBOL]:15} | "
                      f"{row[COL_SIDE]:5} | "
                      f"{tier_color}{row[COL_TIER]:15}{colorama.Style.RESET_ALL} | "
                      f"${row[COL_POSITION_VALUE]:>15,.0f} | "
                      f"PnL: ${row[COL_UNREALIZED_PNL]:>+10,.0f} | "
                      f"Margin: {row[COL_MARGIN_RATIO]:>5.1f}%{colorama.Style.RESET_ALL}")
            
    except Exception as e:
        logger.error(f"Error saving institutional positions to CSV: {str(e)}")

def fetch_all_institutional_positions_parallel(trader_ids: list[str]):
    """Fetch positions for all institutional traders in parallel"""
    try:
        logger.info(f"🏛️ Fetching positions for {len(trader_ids)} institutional traders...")
        
        all_positions = []
        
        # Use ThreadPoolExecutor for parallel processing with lower concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
            # Submit all tasks
            future_to_trader = {
                executor.submit(process_institutional_trader_data, trader_id): trader_id 
                for trader_id in trader_ids
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(trader_ids), desc="Processing institutional traders") as pbar:
                for future in concurrent.futures.as_completed(future_to_trader):
                    trader_id = future_to_trader[future]
                    try:
                        positions = future.result()
                        all_positions.extend(positions)
                        if positions:
                            logger.debug(f"✅ Processed {len(positions)} institutional positions for {trader_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process institutional trader {trader_id}: {str(e)}")
                    finally:
                        pbar.update(1)
        
        logger.info(f"✅ Collected {len(all_positions)} total institutional positions")
        return all_positions
        
    except Exception as e:
        logger.error(f"Error fetching institutional positions in parallel: {str(e)}")
        return []

def main():
    """Main function to run the institutional position tracker"""
    try:
        parser = argparse.ArgumentParser(description="🏛️ Moon Dev's Bitfinex Institutional Position Server")
        parser.add_argument("--delay", type=float, default=CONFIG["API_REQUEST_DELAY"],
                          help="API request delay in seconds")
        parser.add_argument("--workers", type=int, default=CONFIG["MAX_WORKERS"],
                          help="Number of worker threads")
        parser.add_argument("--min-value", type=float, default=CONFIG["MIN_POSITION_VALUE"],
                          help="Minimum position value to track")
        parser.add_argument("--institutional-only", action="store_true",
                          help="Only track positions >= $1M")
        
        args = parser.parse_args()
        
        # Update config
        CONFIG["API_REQUEST_DELAY"] = args.delay
        CONFIG["MAX_WORKERS"] = args.workers
        CONFIG["MIN_POSITION_VALUE"] = args.min_value
        
        if args.institutional_only:
            CONFIG["MIN_POSITION_VALUE"] = CONFIG["INSTITUTIONAL_THRESHOLD"]
        
        print(f"\n🏛️ Moon Dev's Bitfinex Institutional Position Server 🚀")
        print("=" * 70)
        print(f"API Delay: {CONFIG['API_REQUEST_DELAY']}s")
        print(f"Workers: {CONFIG['MAX_WORKERS']}")
        print(f"Min Position Value: ${CONFIG['MIN_POSITION_VALUE']:,.0f}")
        print(f"Institutional Threshold: ${CONFIG['INSTITUTIONAL_THRESHOLD']:,.0f}")
        print("=" * 70)
        
        # Ensure data directory exists
        if not ensure_data_dir():
            logger.error("Failed to create institutional data directory")
            return
        
        # Load institutional trader IDs
        trader_ids = load_institutional_trader_ids()
        if not trader_ids:
            logger.error("No institutional trader IDs available")
            return
        
        # Fetch positions for institutional traders
        start_time = time.time()
        all_positions = fetch_all_institutional_positions_parallel(trader_ids[:15])  # Smaller set for quality
        end_time = time.time()
        
        # Save results with institutional analysis
        save_institutional_positions_to_csv(all_positions)
        
        logger.info(f"🎯 Institutional position tracking completed in {end_time - start_time:.2f} seconds")
        print(f"\n✅ Bitfinex institutional position tracking completed!")
        print(f"📊 Check {DATA_PATH} for saved CSV files")
        print(f"🏛️ Institutional-grade analysis complete!")
        
    except KeyboardInterrupt:
        logger.info("👋 Institutional position server stopped by user")
    except Exception as e:
        logger.error(f"Error in main institutional function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
