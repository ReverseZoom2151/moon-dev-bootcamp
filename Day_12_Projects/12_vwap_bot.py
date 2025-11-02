'''
VWAP BOT

RBI system 
Research - 
Backtest - find 5 winning backtests
Implement - 
'''

import time, eth_account, sys, os, random, schedule, logging
from typing import Tuple, Dict, List

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import project modules
import nice_funcs as n
from Day_4_Projects import dontshare as d

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration parameters
SYMBOL = 'LINK'
TIMEFRAME = '1m'
SMA_WINDOW = 20
LOOKBACK_DAYS = 1 
SIZE = 1 
TARGET_PROFIT = 5
MAX_LOSS = -10
LEVERAGE = 3
MAX_POSITIONS = 1
LONG_PROBABILITY_ABOVE_VWAP = 0.7
LONG_PROBABILITY_BELOW_VWAP = 0.3
EXECUTION_INTERVAL = 3  # seconds
ERROR_SLEEP_TIME = 30  # seconds

def setup_account() -> eth_account.Account:
    """Initialize trading account"""
    try:
        secret = d.private_key
        account = eth_account.Account.from_key(secret)
        return account
    except Exception as e:
        logger.error(f"Error setting up account: {e}")
        raise

def check_positions(account: eth_account.Account) -> Tuple[Dict, bool, float, str, float, float, bool, int]:
    """Get current positions and status"""
    try:
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, is_long, num_of_pos = (
            n.get_position_andmaxpos(SYMBOL, account, MAX_POSITIONS)
        )
        logger.info(f'Current positions for {SYMBOL}: {positions}')
        return positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, is_long, num_of_pos
    except Exception as e:
        logger.error(f"Error checking positions: {e}")
        raise

def manage_existing_position(account: eth_account.Account, in_position: bool) -> None:
    """Handle existing position management"""
    try:
        if in_position:
            n.cancel_all_orders(account)
            logger.info('In position - checking PnL close conditions')
            n.pnl_close(SYMBOL, TARGET_PROFIT, MAX_LOSS, account)
        else:
            logger.info('Not in position - no PnL close needed')
    except Exception as e:
        logger.error(f"Error managing position: {e}")
        raise

def get_market_data(symbol: str) -> Tuple[float, float, List, float]:
    """Get current market data including VWAP"""
    try:
        ask, bid, l2_data = n.ask_bid(symbol)
        
        # Get 11th bid and ask levels
        bid11 = float(l2_data[0][10]['px'])
        ask11 = float(l2_data[1][10]['px'])
        
        # Get VWAP
        latest_vwap = n.calculate_vwap_with_symbol(symbol)[1]
        logger.info(f'Latest VWAP: {latest_vwap}')
        
        return ask, bid, ask11, bid11, latest_vwap
    except IndexError:
        logger.error("Could not access required order book level - insufficient market depth")
        raise
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise

def determine_trade_direction(bid: float, latest_vwap: float) -> bool:
    """Determine whether to go long based on VWAP and probability"""
    random_chance = random.random()
    
    if bid > latest_vwap:
        going_long = random_chance <= LONG_PROBABILITY_ABOVE_VWAP
        logger.info(f'Price above VWAP ({bid} > {latest_vwap}), going long: {going_long}')
    else:
        going_long = random_chance <= LONG_PROBABILITY_BELOW_VWAP
        logger.info(f'Price below VWAP ({bid} < {latest_vwap}), going long: {going_long}')
        
    return going_long

def execute_order(account: eth_account.Account, in_position: bool, going_long: bool, 
                 pos_size: float, bid11: float, ask11: float) -> None:
    """Execute trading orders based on position and direction"""
    try:
        # Check if position size is valid (greater than zero)
        if pos_size <= 0:
            logger.warning(f"Invalid position size: {pos_size}, skipping order placement")
            return
            
        if not in_position:
            n.cancel_all_orders(account)
            logger.info('Canceled all existing orders')
            
            if going_long:
                logger.info(f'Placing buy order at {bid11}')
                try:
                    n.limit_order(SYMBOL, True, pos_size, bid11, False, account)
                    logger.info(f'Placed buy order for {pos_size} at {bid11}')
                except Exception as order_error:
                    logger.error(f"Failed to place buy order: {str(order_error)}")
            else:
                logger.info(f'Placing sell order at {ask11}')
                try:
                    n.limit_order(SYMBOL, False, pos_size, ask11, False, account)
                    logger.info(f'Placed sell order for {pos_size} at {ask11}')
                except Exception as order_error:
                    logger.error(f"Failed to place sell order: {str(order_error)}")
        else:
            logger.info(f'Already in position')
    except Exception as e:
        logger.error(f"Error executing order: {str(e)}")
        # Don't raise here to allow the bot to continue

def bot():
    """Main trading bot function"""
    try:
        # Setup account
        account = setup_account()
        
        # Check current positions
        _, im_in_pos, _, _, _, _, _, _ = check_positions(account)
        
        # Adjust leverage and position size
        try:
            lev, pos_size = n.adjust_leverage_size_signal(SYMBOL, LEVERAGE, account)
            if pos_size <= 0:
                logger.warning(f"Position size {pos_size} is invalid, using minimum size")
                pos_size = 0.1  # Set a minimum size if API returns zero
        except Exception as lev_error:
            logger.error(f"Error adjusting leverage: {str(lev_error)}")
            logger.info("Using default position size")
            pos_size = 0.1  # Default fallback position size
            lev = LEVERAGE
        
        # Manage existing position if any
        manage_existing_position(account, im_in_pos)
        
        # Get market data
        ask, bid, ask11, bid11, latest_vwap = get_market_data(SYMBOL)
        
        # Determine trade direction
        going_long = determine_trade_direction(bid, latest_vwap)
        
        # Execute order if appropriate
        execute_order(account, im_in_pos, going_long, pos_size, bid11, ask11)
        
    except Exception as e:
        logger.error(f"Bot execution error: {str(e)}")
        # Don't raise here to allow the bot to continue on next iteration

def main():
    """Run the bot on schedule"""
    logger.info("Starting VWAP trading bot")
    
    # Run once immediately
    bot()
    
    # Schedule regular execution
    schedule.every(EXECUTION_INTERVAL).seconds.do(bot)
    
    while True:
        try:
            schedule.run_pending()
            # Sleep for a shorter time than the execution interval to ensure we don't miss scheduled runs
            time.sleep(min(1, EXECUTION_INTERVAL/3))
        except Exception as e:
            logger.error(f"Schedule execution error: {str(e)}")
            logger.info(f"Sleeping for {ERROR_SLEEP_TIME} seconds before retry")
            time.sleep(ERROR_SLEEP_TIME)

if __name__ == "__main__":
    main()




