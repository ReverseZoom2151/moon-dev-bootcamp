import ccxt
import pandas as pd
import pandas_ta as ta
import sys, os, time, schedule, logging
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import bitfinex_key, bitfinex_secret

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(f'bitfinex_bollinger_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')])
logger = logging.getLogger('bitfinex_bollinger_bot')

# Configuration Constants
SYMBOL = 'WIF:USDT'
TIMEFRAME = '15m'
SMA_WINDOW = 20
LOOKBACK_BARS = 100  # Adjusted for CCXT
SIZE = 1
TARGET_PROFIT = 5
MAX_LOSS = -10
LEVERAGE = 3
MAX_POSITIONS = 1
ORDER_PARAMS = {'timeInForce': 'POC'}

class BollingerBot:
    def __init__(self):
        try:
            self.exchange = ccxt.bitfinex({'enableRateLimit': True, 'apiKey': bitfinex_key, 'secret': bitfinex_secret})
            logger.info('Connected to Bitfinex')
        except Exception as e:
            logger.error(f'Failed to connect: {e}')
            raise

    def check_position_status(self):
        try:
            positions = self.exchange.fetch_positions([SYMBOL])
            im_in_pos = len(positions) > 0 and positions[0]['amount'] != 0
            pos_size = positions[0]['amount'] if im_in_pos else 0
            long = positions[0]['side'] == 'long' if im_in_pos else None
            entry_px = positions[0]['entry_price'] if im_in_pos else 0
            pnl_perc = positions[0]['unrealized_pnl'] / (entry_px * pos_size) * 100 if im_in_pos and entry_px != 0 else 0
            num_of_pos = len([p for p in self.exchange.fetch_positions() if p['amount'] != 0])
            logger.info(f'Current positions for {SYMBOL}: In pos: {im_in_pos}, Size: {pos_size}, Long: {long}')
            return positions, im_in_pos, pos_size, SYMBOL, entry_px, pnl_perc, long, num_of_pos
        except Exception as e:
            logger.error(f"Error checking position status: {e}")
            return [], False, 0, None, 0, 0, None, 0

    def prepare_trading_parameters(self):
        try:
            markets = self.exchange.load_markets()
            market = markets[SYMBOL]
            min_size = market['limits']['amount']['min']
            adjusted_size = max(SIZE, min_size)
            logger.info(f'Adjusted position size: {adjusted_size}')
            return LEVERAGE, adjusted_size
        except Exception as e:
            logger.error(f"Error preparing parameters: {e}")
            return LEVERAGE, SIZE

    def check_bollinger_bands(self):
        try:
            bars = self.exchange.fetch_ohlcv('BTC:USTF0', '1m', limit=500)  # Using BTC as proxy
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            bb = ta.bbands(df['close'], length=SMA_WINDOW, std=2)
            df = pd.concat([df, bb], axis=1)
            bandwidth = (df[f'BBU_{SMA_WINDOW}_2.0'] - df[f'BBL_{SMA_WINDOW}_2.0']) / df[f'BBM_{SMA_WINDOW}_2.0']
            bands_tight = bandwidth.iloc[-1] < 0.05  # Threshold for tight bands
            logger.info(f'Bollinger bands compression detected: {bands_tight}')
            return df, bands_tight
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return pd.DataFrame(), False

    def place_entry_orders(self, pos_size):
        try:
            ob = self.exchange.fetch_order_book(SYMBOL)
            bid = ob['bids'][0][0] if ob['bids'] else 0
            ask = ob['asks'][0][0] if ob['asks'] else 0
            bid10 = ob['bids'][10][0] if len(ob['bids']) > 10 else bid
            ask10 = ob['asks'][10][0] if len(ob['asks']) > 10 else ask
            self.exchange.cancel_all_orders(SYMBOL)
            logger.info('Cancelled existing orders')
            self.exchange.create_limit_buy_order(SYMBOL, pos_size, bid10, ORDER_PARAMS)
            logger.info(f'Placed buy order for {pos_size} at {bid10}')
            self.exchange.create_limit_sell_order(SYMBOL, pos_size, ask10, ORDER_PARAMS)
            logger.info(f'Placed sell order for {pos_size} at {ask10}')
        except Exception as e:
            logger.error(f"Error placing entry orders: {e}")

    def pnl_close(self):
        try:
            _, im_in_pos, pos_size, _, entry_px, _, long, _ = self.check_position_status()
            if not im_in_pos:
                return
            ticker = self.exchange.fetch_ticker(SYMBOL)
            current = ticker['last']
            diff = (current - entry_px) if long else (entry_px - current)
            pnl_perc = (diff / entry_px * LEVERAGE) * 100 if entry_px != 0 else 0
            if pnl_perc >= TARGET_PROFIT or pnl_perc <= MAX_LOSS:
                self.kill_switch()
        except Exception as e:
            logger.error(f"Error in PNL close: {e}")

    def kill_switch(self):
        try:
            self.exchange.cancel_all_orders(SYMBOL)
            _, im_in_pos, pos_size, _, _, _, long, _ = self.check_position_status()
            if not im_in_pos:
                return
            ask, bid = self.exchange.fetch_order_book(SYMBOL)['asks'][0][0], self.exchange.fetch_order_book(SYMBOL)['bids'][0][0]
            side = 'sell' if long else 'buy'
            px = ask if side == 'sell' else bid
            self.exchange.create_limit_order(SYMBOL, side, pos_size, px, ORDER_PARAMS)
            logger.info(f'Closed position: {side} {pos_size} at {px}')
        except Exception as e:
            logger.error(f"Error in kill switch: {e}")

    def bot_iteration(self):
        logger.info(f"--- Bot iteration starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        _, im_in_pos, _, _, _, _, _, _ = self.check_position_status()
        _, pos_size = self.prepare_trading_parameters()
        if im_in_pos:
            logger.info('Currently in position, monitoring PnL targets')
            self.exchange.cancel_all_orders(SYMBOL)
            self.pnl_close()
        else:
            logger.info('Not in position, checking for entry conditions')
        _, bands_tight = self.check_bollinger_bands()
        if not im_in_pos and bands_tight:
            logger.info('Entry condition met')
            self.place_entry_orders(pos_size)
        elif not bands_tight:
            logger.info('Bollinger bands not tight, cancelling orders and closing positions')
            self.exchange.cancel_all_orders(SYMBOL)
            self.kill_switch()
        else:
            logger.info(f'No action taken. In position: {im_in_pos}, Bands tight: {bands_tight}')
        logger.info(f"--- Bot iteration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

def run_bot():
    logger.info("Starting Bollinger Band Trading Bot")
    bot = BollingerBot()
    bot.bot_iteration()
    schedule.every(30).seconds.do(bot.bot_iteration)
    logger.info("Bot scheduled to run every 30 seconds")
    while True:
        schedule.run_pending()
        time.sleep(10)

if __name__ == "__main__":
    run_bot() 