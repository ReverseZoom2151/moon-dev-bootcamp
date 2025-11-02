'''
VWAP BOT (Binance Version)

RBI system 
Research - 
Backtest - find 5 winning backtests
Implement - 
'''

import time, sys, os, random, schedule, logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_nice_funcs import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Configuration parameters
SYMBOL = 'LINKUSDT'
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

exchange = create_exchange()

def bot():
    try:
        acct_value = acct_bal()
        positions, im_in_pos, _, _, _, _, long, num_of_pos = get_position_andmaxpos(SYMBOL, MAX_POSITIONS)
        logger.info(f'Current positions for {SYMBOL}: {positions}')
        lev, pos_size = adjust_leverage_size_signal(SYMBOL, LEVERAGE, acct_value)
        if pos_size <= 0:
            logger.warning(f'Invalid position size {pos_size}, using minimum size')
            pos_size = 0.1
        if im_in_pos:
            cancel_all_orders(SYMBOL)
            logger.info('In position - checking PnL close conditions')
            pnl_close(SYMBOL, TARGET_PROFIT, MAX_LOSS)
        else:
            logger.info('Not in position - no PnL close needed')
        ask, bid, _ = ask_bid(SYMBOL)
        if ask is None or bid is None:
            return
        _, latest_vwap = calculate_vwap_with_symbol(SYMBOL)
        logger.info(f'Latest VWAP: {latest_vwap}')
        random_chance = random.random()
        if bid > latest_vwap:
            going_long = random_chance <= LONG_PROBABILITY_ABOVE_VWAP
            logger.info(f'Price above VWAP ({bid} > {latest_vwap}), going long: {going_long}')
        else:
            going_long = random_chance <= LONG_PROBABILITY_BELOW_VWAP
            logger.info(f'Price below VWAP ({bid} < {latest_vwap}), going long: {going_long}')
        if not im_in_pos:
            cancel_all_orders(SYMBOL)
            logger.info('Canceled all existing orders')
            if going_long:
                limit_px = bid * 0.999  # Slightly below bid
                limit_order(SYMBOL, True, pos_size, limit_px, False)
                logger.info(f'Placed buy order for {pos_size} at {limit_px}')
            else:
                limit_px = ask * 1.001  # Slightly above ask
                limit_order(SYMBOL, False, pos_size, limit_px, False)
                logger.info(f'Placed sell order for {pos_size} at {limit_px}')
        else:
            logger.info(f'Already in position')
    except Exception as e:
        logger.error(f'Bot execution error: {str(e)}')

bot()  # Run once immediately
schedule.every(EXECUTION_INTERVAL).seconds.do(bot)

while True:
    schedule.run_pending()
    time.sleep(1) 