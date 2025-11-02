import os, sys, time, logging, schedule, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_nice_funcs import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Configuration
SYMBOL = 'WIFUSDT'
TIMEFRAME = '15m'
SMA_WINDOW = 20
LOOKBACK_DAYS = 1
POSITION_SIZE = 1
TARGET_PROFIT = 5
MAX_LOSS = -10
LEVERAGE = 3
MAX_POSITIONS = 1

# Initialize exchange
try:
    exchange = create_exchange()
    logger.info(f"Exchange initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize exchange: {e}")
    sys.exit(1)

def bot():
    try:
        pos_size = POSITION_SIZE
        positions, im_in_pos, mypos_size, _, entry_px, pnl_perc, long = get_position(SYMBOL)
        logger.info(f"Current positions: {positions}")
        if im_in_pos:
            logger.info("In position - checking profit/loss targets")
            pnl_close(SYMBOL, TARGET_PROFIT, MAX_LOSS)
        else:
            logger.info("Not in position - looking for entry opportunities")
        if 0 < mypos_size < POSITION_SIZE:
            logger.info(f"Partial position detected: current size {mypos_size}")
            pos_size = POSITION_SIZE - mypos_size
            logger.info(f"Updated position size needed: {pos_size}")
            im_in_pos = False
        else:
            pos_size = POSITION_SIZE
        latest_sma = get_latest_sma(SYMBOL, TIMEFRAME, SMA_WINDOW, 2)
        if latest_sma is not None:
            logger.info(f"Latest SMA for {SYMBOL} over {SMA_WINDOW} intervals: {latest_sma}")
        else:
            logger.warning("Could not retrieve SMA")
        current_price = ask_bid(SYMBOL)[0]
        if not im_in_pos:
            try:
                sd_df = supply_demand_zones(SYMBOL, TIMEFRAME, LOOKBACK_DAYS)
                logger.info(f"Supply/Demand zones calculated:\n{sd_df}")
                sd_df[f'{TIMEFRAME}_dz'] = pd.to_numeric(sd_df[f'{TIMEFRAME}_dz'], errors='coerce')
                sd_df[f'{TIMEFRAME}_sz'] = pd.to_numeric(sd_df[f'{TIMEFRAME}_sz'], errors='coerce')
                buy_price = sd_df[f'{TIMEFRAME}_dz'].mean()
                sell_price = sd_df[f'{TIMEFRAME}_sz'].mean()
                buy_price = float(buy_price)
                sell_price = float(sell_price)
                logger.info(f"Current price: {current_price}, Buy zone: {buy_price}, Sell zone: {sell_price}")
                diff_to_buy_price = abs(current_price - buy_price)
                diff_to_sell_price = abs(current_price - sell_price)
                cancel_all_orders(SYMBOL)
                logger.info("Canceled all existing orders")
                if diff_to_buy_price < diff_to_sell_price:
                    limit_order(SYMBOL, True, pos_size, buy_price, False)
                    logger.info(f"Placed BUY limit order for {pos_size} {SYMBOL} at price {buy_price}")
                else:
                    limit_order(SYMBOL, False, pos_size, sell_price, False)
                    logger.info(f"Placed SELL limit order for {pos_size} {SYMBOL} at price {sell_price}")
            except Exception as e:
                logger.error(f"Error calculating zones or placing orders: {e}")
        else:
            logger.info(f"Already in {SYMBOL} position with size {mypos_size} - not placing new orders")
    except Exception as e:
        logger.error(f"Error in bot execution: {e}")

try:
    logger.info("Starting initial bot execution")
    bot()
except Exception as e:
    logger.error(f"Error during initial bot execution: {e}")
schedule.every(30).seconds.do(bot)
logger.info(f"Bot scheduled to run every 30 seconds for {SYMBOL}")

logger.info("Starting main loop")
while True:
    try:
        schedule.run_pending()
        time.sleep(10)
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        logger.error("Possible connectivity issue - sleeping 30 seconds before retry")
        time.sleep(30) 