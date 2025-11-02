import os, sys, time, schedule, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from binance_nice_funcs import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Configuration
SYMBOL = 'BTCUSDT'
POS_SIZE = 30
TARGET_GAIN = 9
MAX_LOSS = -8
PARAMS = {'timeInForce': 'GTX'}

# Initialize exchange
try:
    exchange = create_exchange()
    logger.info(f"Exchange initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize exchange: {e}")
    sys.exit(1)

def get_market_data():
    try:
        ask, bid, _ = ask_bid(SYMBOL)
        logger.info(f'For {SYMBOL}... ask: {ask} | bid: {bid}')
        bars = exchange.fetch_ohlcv(SYMBOL, '15m', limit=289)
        df_sma = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_sma['timestamp'] = pd.to_datetime(df_sma['timestamp'], unit='ms')
        curr_support = df_sma['close'].min()
        curr_resis = df_sma['close'].max()
        logger.info(f'Support: {curr_support} | Resistance: {curr_resis}')
        return {'ask': ask, 'bid': bid, 'df_sma': df_sma, 'support': curr_support, 'resistance': curr_resis}
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return None

def retest():
    try:
        logger.info('Calculating retest levels...')
        market_data = get_market_data()
        if not market_data:
            return False, False, False, False
        bid = market_data['bid']
        df_sma = market_data['df_sma']
        resistance = float(df_sma['close'].max())
        support = float(df_sma['close'].min())
        logger.info(f"Comparing bid {bid} to resistance {resistance} and support {support}")
        buy_break_out = False
        sell_break_down = False
        breakoutprice = False
        breakdownprice = False
        if bid > resistance:
            logger.info(f'BREAKOUT DETECTED... Buy at previous resistance {market_data["resistance"]}')
            buy_break_out = True
            breakoutprice = int(resistance) * 1.001
        elif bid < support:
            logger.info(f'BREAKDOWN DETECTED... Sell at previous support {market_data["support"]}')
            sell_break_down = True
            breakdownprice = int(support) * 0.999
        return buy_break_out, sell_break_down, breakoutprice, breakdownprice
    except Exception as e:
        logger.error(f"Error in retest calculation: {str(e)}")
        return False, False, False, False

def bot():
    try:
        pnl_close(SYMBOL, TARGET_GAIN, MAX_LOSS)
        ask, bid, _ = ask_bid(SYMBOL)
        re_test = retest()
        break_out = re_test[0]
        break_down = re_test[1]
        breakoutprice = re_test[2]
        breakdownprice = re_test[3]
        logger.info(f'Breakout: {break_out} @ {breakoutprice} | Breakdown: {break_down} @ {breakdownprice}')
        positions, in_pos, curr_size, _, _, _, _ = get_position(SYMBOL)
        curr_size = int(curr_size) if curr_size else 0
        curr_p = bid
        logger.info(f'Symbol: {SYMBOL} | Breakout: {break_out} | Breakdown: {break_down} | In position: {in_pos} | Size: {curr_size} | Price: {curr_p}')
        if not in_pos and curr_size < POS_SIZE:
            cancel_all_orders(SYMBOL)
            ask, bid, _ = ask_bid(SYMBOL)
            if break_out:
                logger.info(f'Placing BUY order: {SYMBOL} size {POS_SIZE} @ {breakoutprice}')
                limit_order(SYMBOL, True, POS_SIZE, breakoutprice, False)
                logger.info('Order submitted - pausing for 2 minutes')
                time.sleep(120)
            elif break_down:
                logger.info(f'Placing SELL order: {SYMBOL} size {POS_SIZE} @ {breakdownprice}')
                limit_order(SYMBOL, False, POS_SIZE, breakdownprice, False)
                logger.info('Order submitted - pausing for 2 minutes')
                time.sleep(120)
            else:
                logger.info('No breakout/breakdown detected - waiting 1 minute')
                time.sleep(60)
        else:
            logger.info('Already in position - not placing new orders')
    except Exception as e:
        logger.error(f"Error in bot execution: {e}")

schedule.every(28).seconds.do(bot)
logger.info(f"Bot scheduled to run every 28 seconds for {SYMBOL}")

logger.info("Starting main loop")
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        logger.error("Possible internet connection issue - waiting 30 seconds before retry")
        time.sleep(30) 