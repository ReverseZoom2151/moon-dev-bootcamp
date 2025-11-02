import ccxt, pandas as pd, sys, os, time, schedule, configparser, logging
import pandas_ta as ta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import bitfinex_key, bitfinex_secret

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("bitfinex_trading_bot.log"), logging.StreamHandler()])
logger = logging.getLogger("BitfinexTradingBot")

last_api_call_time = 0
min_call_interval = 1.0

def rate_limited_api_call(func, *args, **kwargs):
    global last_api_call_time
    current_time = time.time()
    time_since_last_call = current_time - last_api_call_time
    if time_since_last_call < min_call_interval:
        sleep_time = min_call_interval - time_since_last_call
        time.sleep(sleep_time)
    result = func(*args, **kwargs)
    last_api_call_time = time.time()
    return result

def load_config(config_file='bitfinex_config.ini'):
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        return {'symbol': config['Trading']['symbol'], 'pos_size': int(config['Trading']['position_size']), 'target': float(config['Trading']['target']), 'max_loss': float(config['Trading']['max_loss']), 'vol_decimal': float(config['Trading']['vol_decimal'])}
    except:
        logger.warning(f"Could not load config from {config_file}, using defaults")
        return {'symbol': 'BTC:USTF0', 'pos_size': 30, 'target': 8, 'max_loss': -9, 'vol_decimal': 0.4}

class TradingBot:
    def __init__(self, exchange, config=None):
        self.exchange = exchange
        self.retry_count = {}
        if config is None:
            config = load_config()
        self.symbol = config['symbol']
        self.pos_size = config['pos_size']
        self.target = config['target']
        self.max_loss = config['max_loss']
        self.vol_decimal = config['vol_decimal']
        self.params = {'timeInForce': 'POC'}
        self._daily_sma_cache = None
        self._daily_sma_cache_time = 0
        self._f15_sma_cache = None
        self._f15_sma_cache_time = 0

    def ask_bid(self):
        logger.debug(f"Fetching order book for {self.symbol}")
        ob = rate_limited_api_call(self.exchange.fetch_order_book, self.symbol)
        bid = ob['bids'][0][0] if ob['bids'] else None
        ask = ob['asks'][0][0] if ob['asks'] else None
        return ask, bid

    def daily_sma(self, cache_time=300):
        current_time = time.time()
        if self._daily_sma_cache is not None and current_time - self._daily_sma_cache_time < cache_time:
            logger.debug("Using cached daily SMA data")
            return self._daily_sma_cache
        logger.info('Starting daily indicators calculation...')
        timeframe = '1d'
        num_bars = 100
        bars = rate_limited_api_call(self.exchange.fetch_ohlcv, self.symbol, timeframe=timeframe, limit=num_bars)
        df_d = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_d['timestamp'] = pd.to_datetime(df_d['timestamp'], unit='ms')
        df_d['sma20_d'] = ta.sma(df_d['close'], length=20)
        bid = self.ask_bid()[1]
        if bid is not None:
            df_d.loc[df_d['sma20_d'] > bid, 'sig'] = 'SELL'
            df_d.loc[df_d['sma20_d'] < bid, 'sig'] = 'BUY'
        self._daily_sma_cache = df_d
        self._daily_sma_cache_time = current_time
        return df_d

    def f15_sma(self, cache_time=150):
        current_time = time.time()
        if self._f15_sma_cache is not None and current_time - self._f15_sma_cache_time < cache_time:
            logger.debug("Using cached 15m SMA data")
            return self._f15_sma_cache
        logger.info('Starting 15 min SMA calculation...')
        timeframe = '15m'
        num_bars = 100
        bars = rate_limited_api_call(self.exchange.fetch_ohlcv, self.symbol, timeframe=timeframe, limit=num_bars)
        df_f = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], unit='ms')
        df_f['sma20_15'] = ta.sma(df_f['close'], length=20)
        df_f['bp_1'] = df_f['sma20_15'] * 1.001
        df_f['bp_2'] = df_f['sma20_15'] * 0.997
        df_f['sp_1'] = df_f['sma20_15'] * 0.999
        df_f['sp_2'] = df_f['sma20_15'] * 1.003
        self._f15_sma_cache = df_f
        self._f15_sma_cache_time = current_time
        return df_f

    def calculate_position_size(self):
        df = self.f15_sma()
        volatility = df['close'].pct_change().std() * 100
        adjusted_size = self.pos_size * (1 / (1 + volatility))
        return max(int(adjusted_size), 1)

    def open_positions(self):
        logger.debug("Checking open positions")
        positions = rate_limited_api_call(self.exchange.fetch_positions, [self.symbol])
        if not positions or positions[0]['amount'] == 0:
            return positions, False, 0, None
        pos = positions[0]
        openpos_bool = True
        long = pos['side'] == 'long'
        openpos_size = pos['amount']
        return positions, openpos_bool, openpos_size, long

    def kill_switch(self):
        logger.info('Starting the kill switch')
        openposi = self.open_positions()[1]
        long = self.open_positions()[3]
        kill_size = self.open_positions()[2]
        logger.info(f'Open position: {openposi}, Long: {long}, Size: {kill_size}')
        attempts = 0
        max_attempts = 5
        while openposi and attempts < max_attempts:
            attempts += 1
            rate_limited_api_call(self.exchange.cancel_all_orders, self.symbol)
            openposi = self.open_positions()[1]
            long = self.open_positions()[3]
            kill_size = int(self.open_positions()[2])
            ask, bid = self.ask_bid()
            if ask is None or bid is None:
                time.sleep(5)
                continue
            if not long:
                rate_limited_api_call(self.exchange.create_limit_buy_order, self.symbol, kill_size, bid, self.params)
                logger.info(f'Created BUY to CLOSE order of {kill_size} {self.symbol} at ${bid}')
            else:
                rate_limited_api_call(self.exchange.create_limit_sell_order, self.symbol, kill_size, ask, self.params)
                logger.info(f'Created SELL to CLOSE order of {kill_size} {self.symbol} at ${ask}')
            time.sleep(30)
            openposi = self.open_positions()[1]

    def sleep_on_close(self):
        logger.debug('Checking if we need to sleep after recent trade')
        closed_orders = rate_limited_api_call(self.exchange.fetch_closed_orders, self.symbol)
        for ord in closed_orders[-1::-1]:
            sincelasttrade = 59
            status = ord['status']
            if status == 'closed' and ord['filled'] > 0:
                txttime = ord['timestamp'] // 1000
                ex_timestamp = int(time.time())
                time_spread = (ex_timestamp - txttime) / 60
                if time_spread < sincelasttrade:
                    sleepy = round(sincelasttrade - time_spread) * 60
                    logger.info(f'Time since last trade ({time_spread} mins) is less than {sincelasttrade} mins, sleeping for {sleepy} secs')
                    time.sleep(sleepy)
                else:
                    logger.info(f'It has been {time_spread} mins since last fill, no sleep needed')
                break
        logger.debug('Completed sleep on close check')

    def ob(self):
        logger.info('Fetching and analyzing order book data')
        df = pd.DataFrame()
        for x in range(11):
            ob = rate_limited_api_call(self.exchange.fetch_order_book, self.symbol)
            bid_vol = sum(vol for _, vol in ob['bids'])
            ask_vol = sum(vol for _, vol in ob['asks'])
            temp_df = pd.DataFrame({'bid_vol': [bid_vol], 'ask_vol': [ask_vol]})
            df = pd.concat([df, temp_df], ignore_index=True)
            logger.debug(f"Order book sample {x}: bid_vol={bid_vol}, ask_vol={ask_vol}")
            time.sleep(5)
        total_bidvol = df['bid_vol'].sum()
        total_askvol = df['ask_vol'].sum()
        logger.info(f'Last minute volume summary - Bid Vol: {total_bidvol} | Ask Vol: {total_askvol}')
        if total_bidvol > total_askvol:
            control_dec = (total_askvol / total_bidvol)
            logger.info(f'Bulls are in control: {control_dec}')
            bullish = True
        else:
            control_dec = (total_bidvol / total_askvol)
            logger.info(f'Bears are in control: {control_dec}')
            bullish = False
        open_posi = self.open_positions()
        openpos_tf = open_posi[1]
        long = open_posi[3]
        logger.debug(f'Position check - In position: {openpos_tf}, Long: {long}')
        vol_under_dec = False
        if openpos_tf:
            if long:
                logger.info('Currently in a long position')
                if control_dec < self.vol_decimal:
                    vol_under_dec = True
            else:
                logger.info('Currently in a short position')
                if control_dec < self.vol_decimal:
                    vol_under_dec = True
        else:
            logger.info('Not currently in a position')
        return vol_under_dec

    def check_circuit_breakers(self):
        df = self.f15_sma()
        recent_volatility = df['close'].pct_change().rolling(5).std().iloc[-1] * 100
        if recent_volatility > 5:
            logger.warning(f"Circuit breaker triggered - high volatility: {recent_volatility}%")
            return False
        ask, bid = self.ask_bid()
        if ask is None or bid is None:
            return False
        spread_pct = (ask - bid) / bid * 100 if bid != 0 else 0
        if spread_pct > 0.5:
            logger.warning(f"Circuit breaker triggered - high spread: {spread_pct}%")
            return False
        return True

    def pnl_close(self):
        logger.info('Checking if it is time to exit positions')
        try:
            positions = rate_limited_api_call(self.exchange.fetch_positions, [self.symbol])
            if not positions:
                return False, False, 0, None
            pos = positions[0]
            side = pos['side']
            size = pos['amount']
            entry_price = pos['entry_price']
            leverage = pos['leverage']
            current_price = self.ask_bid()[1]
            if current_price is None:
                return False, False, 0, None
            logger.info(f'Position details - Side: {side}, Entry: {entry_price}, Leverage: {leverage}')
            if side == 'long':
                diff = current_price - entry_price
                long = True
            else:
                diff = entry_price - current_price
                long = False
            perc = round(((diff / entry_price) * leverage), 10) * 100 if entry_price != 0 else 0
            logger.info(f'Current PNL: {perc}%')
            pnlclose = False
            in_pos = size > 0
            if perc > 0 and in_pos:
                logger.info('Currently in a profitable position')
                if perc > self.target:
                    logger.info(f'Profit target of {self.target}% reached ({perc}%), checking volume before exit')
                    pnlclose = True
                    vol_under_dec = self.ob()
                    if vol_under_dec:
                        logger.info(f'Volume is under threshold {self.vol_decimal}, waiting 30s')
                        time.sleep(30)
                    else:
                        logger.info(f'Initiating exit - profit target reached and volume conditions met')
                        self.kill_switch()
            elif perc < 0 and in_pos:
                if perc <= self.max_loss:
                    logger.warning(f'Stop loss triggered at {perc}% (limit: {self.max_loss}%), initiating exit')
                    self.kill_switch()
            return pnlclose, in_pos, size, long
        except Exception as e:
            logger.error(f"Error in PNL calculation: {e}")
            return False, False, 0, None

    def bot(self):
        try:
            if not self.check_circuit_breakers():
                logger.warning("Circuit breakers active - skipping trading cycle")
                time.sleep(60)
                return
            pnl_close_result = self.pnl_close()
            self.sleep_on_close()
            df_d = self.daily_sma()
            df_f = self.f15_sma()
            ask, bid = self.ask_bid()
            sig = df_d.iloc[-1]['sig'] if not df_d.empty else None
            logger.debug(f"Trading signal: {sig}")
            calculated_pos_size = self.calculate_position_size()
            open_size = calculated_pos_size // 2
            logger.debug(f"Using calculated position size: {calculated_pos_size}, split size: {open_size}")
            in_pos = pnl_close_result[1]
            curr_size = int(self.open_positions()[2])
            curr_p = bid if bid is not None else 0
            last_sma15 = df_f.iloc[-1]['sma20_15'] if not df_f.empty else 0
            if not in_pos and curr_size < self.pos_size:
                logger.info("Looking for entry opportunities")
                rate_limited_api_call(self.exchange.cancel_all_orders, self.symbol)
                if sig == 'BUY' and curr_p > last_sma15:
                    logger.info('Setting up BUY orders')
                    bp_1 = df_f.iloc[-1]['bp_1']
                    bp_2 = df_f.iloc[-1]['bp_2']
                    logger.info(f'Buy prices - BP1: {bp_1}, BP2: {bp_2}')
                    rate_limited_api_call(self.exchange.create_limit_buy_order, self.symbol, open_size, bp_1, self.params)
                    rate_limited_api_call(self.exchange.create_limit_buy_order, self.symbol, open_size, bp_2, self.params)
                    logger.info('Buy orders placed, waiting for fills (120s)')
                    time.sleep(120)
                elif sig == 'SELL' and curr_p < last_sma15:
                    logger.info('Setting up SELL orders')
                    sp_1 = df_f.iloc[-1]['sp_1']
                    sp_2 = df_f.iloc[-1]['sp_2']
                    logger.info(f'Sell prices - SP1: {sp_1}, SP2: {sp_2}')
                    rate_limited_api_call(self.exchange.create_limit_sell_order, self.symbol, open_size, sp_1, self.params)
                    rate_limited_api_call(self.exchange.create_limit_sell_order, self.symbol, open_size, sp_2, self.params)
                    logger.info('Sell orders placed, waiting for fills (120s)')
                    time.sleep(120)
                else:
                    logger.info('Entry conditions not met, waiting for better setup (10m)')
                    time.sleep(600)
            else:
                logger.info('Already in position or at size limit, no new orders')
        except Exception as e:
            logger.error(f"Error in main bot logic: {e}")
            if '"code":10500' in str(e):
                self.retry_count[self.symbol] = self.retry_count.get(self.symbol, 0) + 1
                wait_time = min(60 * self.retry_count[self.symbol], 300)
                logger.warning(f"Rate limit hit, backing off for {wait_time}s (attempt {self.retry_count[self.symbol]})")
                time.sleep(wait_time)
            else:
                self.retry_count[self.symbol] = 0
                time.sleep(30)
            if max(self.retry_count.values(), default=0) > 5:
                logger.warning("Too many rate limits - pausing for 15 minutes")
                time.sleep(900)
                self.retry_count = {}
            schedule.clear()
            schedule.every(60).seconds.do(self.bot)

def create_default_config():
    if not os.path.exists('bitfinex_config.ini'):
        config = configparser.ConfigParser()
        config['Trading'] = {'symbol': 'BTC:USTF0', 'position_size': '30', 'target': '8', 'max_loss': '-9', 'vol_decimal': '0.4'}
        with open('bitfinex_config.ini', 'w') as configfile:
            config.write(configfile)
        logger.info("Created default bitfinex_config.ini")

def main():
    create_default_config()
    try:
        exchange = ccxt.bitfinex({'enableRateLimit': True, 'apiKey': bitfinex_key, 'secret': bitfinex_secret})
        bot = TradingBot(exchange)
        schedule.every(28).seconds.do(bot.bot)
        logger.info(f"Trading bot started with symbol: {bot.symbol}")
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error initializing bot: {e}")

if __name__ == "__main__":
    main() 