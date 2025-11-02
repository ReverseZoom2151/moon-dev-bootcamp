############# Coding trading bot #3 - engulfing candle 2024 (Bitfinex Version)

'''
ENGULFING candle strategy:
An engulfing pattern occurs when the current candle completely engulfs the previous candle.
- Bullish engulfing: Current candle's body completely engulfs previous bearish candle's body
- Bearish engulfing: Current candle's body completely engulfs previous bullish candle's body

Trading logic:
1. Check for engulfing patterns in the most recent candles
2. Take a position in the direction of the engulfing pattern if confirmed by SMA
3. Manage risk with target profit and stop loss
'''

import os, sys, time, logging, schedule
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bitfinex_nice_funcs import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Configuration
SYMBOL = 'BTC:USDT'
TIMEFRAME = '15m'
SMA_WINDOW = 20
LOOKBACK_DAYS = 1
POSITION_SIZE = 1
TARGET_PROFIT = 9
MAX_LOSS = -8
LEVERAGE = 3
MAX_POSITIONS = 1

# Global state for recent orders
recent_order = False
recent_order_time = None

exchange = create_exchange()

def detect_engulfing_pattern(df):
    if len(df) < 3:
        return None
    prev_candle = df.iloc[-3]
    curr_candle = df.iloc[-2]
    prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
    curr_body_size = abs(curr_candle['close'] - curr_candle['open'])
    prev_bullish = prev_candle['close'] > prev_candle['open']
    curr_bullish = curr_candle['close'] > curr_candle['open']
    if (not prev_bullish and curr_bullish and curr_body_size > prev_body_size and curr_candle['open'] <= prev_candle['close'] and curr_candle['close'] >= prev_candle['open']):
        return 'bullish'
    elif (prev_bullish and not curr_bullish and curr_body_size > prev_body_size and curr_candle['open'] >= prev_candle['close'] and curr_candle['close'] <= prev_candle['open']):
        return 'bearish'
    return None

def bot():
    global recent_order, recent_order_time
    if recent_order and recent_order_time:
        elapsed = (datetime.now() - recent_order_time).total_seconds()
        if elapsed < 120:
            logger.info(f'Recent order placed {elapsed:.0f} seconds ago, waiting...')
            return
        else:
            recent_order = False
    try:
        pnl_close(SYMBOL, TARGET_PROFIT, MAX_LOSS)
        positions, in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(SYMBOL)
        if in_pos:
            logger.info(f'Already in position for {SYMBOL}, skipping new entry')
            return
        ask, bid, ob = ask_bid(SYMBOL)
        if ask is None or bid is None:
            return
        df = df_sma(SYMBOL, TIMEFRAME, limit=100, sma=SMA_WINDOW)
        sma_value = df[f'sma_{SMA_WINDOW}'].iloc[-1]
        pattern = detect_engulfing_pattern(df)
        logger.info(f'Engulfing pattern: {pattern}, SMA: {sma_value}, Bid: {bid}, Ask: {ask}')
        if pattern == 'bullish' and bid > sma_value:
            order_px = bid * 0.999
            limit_order(SYMBOL, True, POSITION_SIZE, order_px, False)
            recent_order = True
            recent_order_time = datetime.now()
            logger.info(f'Placed BUY order at {order_px}')
        elif pattern == 'bearish' and ask < sma_value:
            order_px = ask * 1.001
            limit_order(SYMBOL, False, POSITION_SIZE, order_px, False)
            recent_order = True
            recent_order_time = datetime.now()
            logger.info(f'Placed SELL order at {order_px}')
    except Exception as e:
        logger.error(f'Error in bot: {str(e)}')

schedule.every(30).seconds.do(bot)

while True:
    schedule.run_pending()
    time.sleep(1) 