'''
Supply and Demand Zone Trading Bot for Hyper Liquid

This bot identifies and trades based on supply and demand zones.
It places limit orders at calculated supply/demand zones and manages positions automatically.
'''

import os, sys, time, logging, schedule, pandas as pd, nice_funcs as n, eth_account
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects import dontshare as d

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
SYMBOL = 'WIF'
TIMEFRAME = '15m'
SMA_WINDOW = 20
LOOKBACK_DAYS = 1
POSITION_SIZE = 1
TARGET_PROFIT = 5  # Percentage
MAX_LOSS = -10  # Percentage
LEVERAGE = 3
MAX_POSITIONS = 1

# Initialize account
try:
    SECRET = d.private_key
    ACCOUNT = eth_account.Account.from_key(SECRET)
    # Set initial leverage
    n.adjust_leverage_size_signal(SYMBOL, LEVERAGE, ACCOUNT)
    logger.info(f"Account initialized with leverage {LEVERAGE} for {SYMBOL}")
except Exception as e:
    logger.error(f"Failed to initialize account: {e}")
    sys.exit(1)

def bot():
    """
    Main trading function that executes the supply/demand zone strategy.
    
    1. Checks existing positions
    2. Manages PNL for open positions
    3. Calculates supply/demand zones
    4. Places limit orders based on price proximity to zones
    """
    try:
        pos_size = POSITION_SIZE
        
        # Check current positions
        positions, im_in_pos, mypos_size, pos_sym, entry_px, pnl_perc, long = n.get_position(SYMBOL, ACCOUNT)
        logger.info(f"Current positions: {positions}")
        
        # Manage open positions
        if im_in_pos:
            logger.info("In position - checking profit/loss targets")
            n.pnl_close(SYMBOL, TARGET_PROFIT, MAX_LOSS, ACCOUNT)
        else:
            logger.info("Not in position - looking for entry opportunities")
        
        # Handle partial positions
        if 0 < mypos_size < POSITION_SIZE:
            logger.info(f"Partial position detected: current size {mypos_size}")
            pos_size = POSITION_SIZE - mypos_size
            logger.info(f"Updated position size needed: {pos_size}")
            im_in_pos = False
        else:
            pos_size = POSITION_SIZE
        
        # Get latest SMA for reference
        latest_sma = n.get_latest_sma(SYMBOL, TIMEFRAME, SMA_WINDOW, 2)
        if latest_sma is not None:
            logger.info(f"Latest SMA for {SYMBOL} over {SMA_WINDOW} intervals: {latest_sma}")
        else:
            logger.warning("Could not retrieve SMA")
        
        # Get current price
        current_price = n.ask_bid(SYMBOL)[0]
        
        # If not in position, look for entry opportunities
        if not im_in_pos:
            try:
                # Calculate supply and demand zones
                sd_df = n.supply_demand_zones_hl(SYMBOL, TIMEFRAME, LOOKBACK_DAYS)
                logger.info(f"Supply/Demand zones calculated:\n{sd_df}")
                
                # Ensure numeric values for calculations
                sd_df[f'{TIMEFRAME}_dz'] = pd.to_numeric(sd_df[f'{TIMEFRAME}_dz'], errors='coerce')
                sd_df[f'{TIMEFRAME}_sz'] = pd.to_numeric(sd_df[f'{TIMEFRAME}_sz'], errors='coerce')
                
                # Calculate average demand (buy) and supply (sell) zones
                buy_price = sd_df[f'{TIMEFRAME}_dz'].mean()
                sell_price = sd_df[f'{TIMEFRAME}_sz'].mean()
                
                # Ensure prices are properly formatted
                buy_price = float(buy_price)
                sell_price = float(sell_price)
                
                logger.info(f"Current price: {current_price}, Buy zone: {buy_price}, Sell zone: {sell_price}")
                
                # Calculate price differences to determine closest zone
                diff_to_buy_price = abs(current_price - buy_price)
                diff_to_sell_price = abs(current_price - sell_price)
                
                # Place orders based on proximity to zones
                if diff_to_buy_price < diff_to_sell_price:
                    # Place buy order at demand zone
                    n.cancel_all_orders(ACCOUNT)
                    logger.info("Canceled all existing orders")
                    
                    n.limit_order(SYMBOL, True, pos_size, buy_price, False, ACCOUNT)
                    logger.info(f"Placed BUY limit order for {pos_size} {SYMBOL} at price {buy_price}")
                else:
                    # Place sell order at supply zone
                    n.cancel_all_orders(ACCOUNT)
                    logger.info("Canceled all existing orders")
                    
                    n.limit_order(SYMBOL, False, pos_size, sell_price, False, ACCOUNT)
                    logger.info(f"Placed SELL limit order for {pos_size} {SYMBOL} at price {sell_price}")
            except Exception as e:
                logger.error(f"Error calculating zones or placing orders: {e}")
        else:
            logger.info(f"Already in {pos_sym} position with size {mypos_size} - not placing new orders")
    except Exception as e:
        logger.error(f"Error in bot execution: {e}")

# Run the bot once at startup
try:
    logger.info("Starting initial bot execution")
    bot()
except Exception as e:
    logger.error(f"Error during initial bot execution: {e}")

# Schedule the bot to run every 30 seconds
schedule.every(30).seconds.do(bot)
logger.info(f"Bot scheduled to run every 30 seconds for {SYMBOL}")

# Main loop
logger.info("Starting main loop")
while True:
    try:
        schedule.run_pending()
        time.sleep(10)
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        logger.error(f"Exception details: {str(e)}")
        logger.error("Possible connectivity issue - sleeping 30 seconds before retry")
        time.sleep(30)
