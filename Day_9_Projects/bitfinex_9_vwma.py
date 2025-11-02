import ccxt, pandas as pd, time, sys, os, logging, configparser
import pandas_ta as ta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import bitfinex_key, bitfinex_secret

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("bitfinex_vwma_trading.log"), logging.StreamHandler()])
logger = logging.getLogger("BitfinexVWMATrader")

last_api_call_time = 0
min_call_interval = 0.2

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

def load_config(config_file='bitfinex_vwma_config.ini'):
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        return {'symbol': config['Trading']['symbol'], 'size': int(config['Trading']['size']), 'target': float(config['Trading']['target']), 'max_loss': float(config['Trading']['max_loss']), 'timeframe': config['Indicators']['timeframe'], 'limit': int(config['Indicators']['limit'])}
    except:
        logger.warning(f"Could not load config from {config_file}, using defaults")
        return {'symbol': 'BTC:USTF0', 'size': 1, 'target': 9, 'max_loss': -8, 'timeframe': '1d', 'limit': 100}

class VWMATrader:
    def __init__(self, exchange, config=None):
        self.exchange = exchange
        if config is None:
            config = load_config()
        self.symbol = config['symbol']
        self.size = config['size']
        self.target = config['target']
        self.max_loss = config['max_loss']
        self.timeframe = config['timeframe']
        self.limit = config['limit']
        self.params = {'timeInForce': 'POC'}
        self._vwma_data_cache = None
        self._vwma_cache_time = 0
        self._vwap_data_cache = None
        self._vwap_cache_time = 0
        self._sma_data_cache = None
        self._sma_cache_time = 0
        self._rsi_data_cache = None
        self._rsi_cache_time = 0
        self.position_indices = {'BTC:USTF0': 0}  # Adjust as needed
        logger.info(f"VWMATrader initialized with symbol: {self.symbol}")

    def get_symbol_position_index(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        return self.position_indices.get(symbol, None)

    def open_positions(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        logger.debug(f"Checking open positions for {symbol}")
        index_pos = self.get_symbol_position_index(symbol)
        try:
            positions = rate_limited_api_call(self.exchange.fetch_positions, [symbol])
            if not positions or positions[0]['amount'] == 0:
                return positions, False, 0, None, index_pos
            pos = positions[0]
            side = 'Buy' if pos['side'] == 'long' else 'Sell'
            size = pos['amount']
            openpos_bool = True
            long = pos['side'] == 'long'
            logger.info(f"Position check - In position: {openpos_bool}, Size: {size}, Long: {long}, Index: {index_pos}")
            return positions, openpos_bool, size, long, index_pos
        except Exception as e:
            logger.error(f"Error checking open positions: {e}")
            return [], False, 0, None, index_pos

    def ask_bid(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        logger.debug(f"Fetching order book for {symbol}")
        try:
            ob = rate_limited_api_call(self.exchange.fetch_order_book, symbol)
            bid = ob['bids'][0][0] if ob['bids'] else None
            ask = ob['asks'][0][0] if ob['asks'] else None
            logger.info(f"Current prices for {symbol} - Ask: {ask}, Bid: {bid}")
            return ask, bid
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None, None

    def kill_switch(self, symbol=None):
        if symbol is None:
            symbol = self.symbol
        logger.info(f"Starting the kill switch for {symbol}")
        try:
            position_info = self.open_positions(symbol)
            openposi = position_info[1]
            long = position_info[3]
            kill_size = position_info[2]
            logger.info(f"Kill switch - In position: {openposi}, Long: {long}, Size: {kill_size}")
            attempts = 0
            max_attempts = 5
            while openposi and attempts < max_attempts:
                attempts += 1
                rate_limited_api_call(self.exchange.cancel_all_orders, symbol)
                position_info = self.open_positions(symbol)
                openposi = position_info[1]
                long = position_info[3]
                kill_size = position_info[2]
                kill_size = int(kill_size) if kill_size else 0
                ask, bid = self.ask_bid(symbol)
                if ask is None or bid is None:
                    time.sleep(5)
                    continue
                if long is False:
                    rate_limited_api_call(self.exchange.create_limit_buy_order, symbol, kill_size, bid, self.params)
                    logger.info(f"Created BUY to CLOSE order of {kill_size} {symbol} at ${bid}")
                elif long is True:
                    rate_limited_api_call(self.exchange.create_limit_sell_order, symbol, kill_size, ask, self.params)
                    logger.info(f"Created SELL to CLOSE order of {kill_size} {symbol} at ${ask}")
                time.sleep(30)
                position_info = self.open_positions(symbol)
                openposi = position_info[1]
            logger.info(f"Kill switch completed for {symbol}")
        except Exception as e:
            logger.error(f"Error in kill switch: {e}")

    def pnl_close(self, symbol=None, target=None, max_loss=None):
        if symbol is None:
            symbol = self.symbol
        if target is None:
            target = self.target
        if max_loss is None:
            max_loss = self.max_loss
        logger.info(f"Checking if it's time to exit for {symbol}...")
        try:
            positions = rate_limited_api_call(self.exchange.fetch_positions, [symbol])
            if not positions:
                return False, False, 0, None
            pos = positions[0]
            side = pos['side']
            size = pos['amount']
            entry_price = pos['entry_price']
            leverage = pos['leverage']
            current_price = self.ask_bid(symbol)[1]
            if current_price is None:
                return False, False, 0, None
            logger.info(f"Position details - Side: {side}, Entry: {entry_price}, Leverage: {leverage}")
            if side == 'long':
                diff = current_price - entry_price
                long = True
            else:
                diff = entry_price - current_price
                long = False
            perc = round(((diff / entry_price) * leverage), 10) * 100 if entry_price != 0 else 0
            logger.info(f"For {symbol} current PnL: {perc}%")
            pnlclose = False
            in_pos = size > 0
            if perc > 0 and in_pos:
                logger.info(f"For {symbol} we are in a profitable position")
                if perc > target:
                    logger.info(f"Profit target of {target}% reached ({perc}%), initiating exit")
                    pnlclose = True
                    self.kill_switch(symbol)
            elif perc < 0 and in_pos:
                if perc <= max_loss:
                    logger.warning(f"Stop loss triggered at {perc}% (limit: {max_loss}%), initiating exit")
                    self.kill_switch(symbol)
            return pnlclose, in_pos, size, long
        except Exception as e:
            logger.error(f"Error in PnL calculation: {e}")
            return False, False, 0, None

    def size_kill(self):
        max_risk = 1000
        logger.debug("Checking position size against risk limits")
        try:
            positions = rate_limited_api_call(self.exchange.fetch_positions)
            total_exposure = sum(float(p['notional']) for p in positions if p['notional'] > 0)
            logger.info(f"Total exposure: {total_exposure}")
            if total_exposure > max_risk:
                logger.critical(f"EMERGENCY KILL SWITCH ACTIVATED - Total exposure {total_exposure} exceeds max risk {max_risk}")
                for p in positions:
                    if p['notional'] > 0:
                        self.kill_switch(p['symbol'])
                time.sleep(300)
                return True
            else:
                logger.debug(f"Size check passed - Total exposure {total_exposure} within risk limit {max_risk}")
                return False
        except Exception as e:
            logger.error(f"Error in size kill check: {e}")
            return False

    def df_sma(self, symbol=None, timeframe=None, limit=None, sma=20, cache_time=300):
        if symbol is None:
            symbol = self.symbol
        if timeframe is None:
            timeframe = self.timeframe
        if limit is None:
            limit = self.limit
        current_time = time.time()
        if self._sma_data_cache is not None and current_time - self._sma_cache_time < cache_time:
            logger.debug("Using cached SMA data")
            return self._sma_data_cache
        logger.info(f"Starting SMA calculation for {symbol}, {timeframe} timeframe")
        try:
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
            df_sma = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_sma['timestamp'] = pd.to_datetime(df_sma['timestamp'], unit='ms')
            df_sma[f'sma{sma}_{timeframe}'] = df_sma.close.rolling(sma).mean()
            bid = self.ask_bid(symbol)[1]
            if bid is not None:
                df_sma.loc[df_sma[f'sma{sma}_{timeframe}'] > bid, 'sig'] = 'SELL'
                df_sma.loc[df_sma[f'sma{sma}_{timeframe}'] < bid, 'sig'] = 'BUY'
            df_sma['support'] = df_sma[:-1]['close'].min()
            df_sma['resis'] = df_sma[:-1]['close'].max()
            self._sma_data_cache = df_sma
            self._sma_cache_time = current_time
            logger.debug(f"SMA calculation completed for {symbol}")
            return df_sma
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.DataFrame()

    def df_rsi(self, symbol=None, timeframe=None, limit=None, cache_time=300):
        if symbol is None:
            symbol = self.symbol
        if timeframe is None:
            timeframe = self.timeframe
        if limit is None:
            limit = self.limit
        current_time = time.time()
        if self._rsi_data_cache is not None and current_time - self._rsi_cache_time < cache_time:
            logger.debug("Using cached RSI data")
            return self._rsi_data_cache
        logger.info(f"Starting RSI calculation for {symbol}, {timeframe} timeframe")
        try:
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
            df_rsi = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_rsi['timestamp'] = pd.to_datetime(df_rsi['timestamp'], unit='ms')
            df_rsi['rsi'] = ta.rsi(df_rsi['close'])
            self._rsi_data_cache = df_rsi
            self._rsi_cache_time = current_time
            logger.debug(f"RSI calculation completed for {symbol}")
            return df_rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.DataFrame()

    def get_df_vwap(self, symbol=None, timeframe=None, limit=None, cache_time=300):
        if symbol is None:
            symbol = self.symbol
        if timeframe is None:
            timeframe = self.timeframe
        if limit is None:
            limit = self.limit
        current_time = time.time()
        if self._vwap_data_cache is not None and current_time - self._vwap_cache_time < cache_time:
            logger.debug("Using cached VWAP data")
            return self._vwap_data_cache
        logger.info(f"Fetching data for VWAP calculation for {symbol}")
        try:
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
            df_vwap = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_vwap['timestamp'] = pd.to_datetime(df_vwap['timestamp'], unit='ms')
            self._vwap_data_cache = df_vwap
            self._vwap_cache_time = current_time
            logger.debug(f"VWAP data prepared for {symbol}")
            return df_vwap
        except Exception as e:
            logger.error(f"Error preparing VWAP data: {e}")
            return pd.DataFrame()

    def vwap_indi(self, cache_time=300):
        logger.info("Starting VWAP indicator calculation")
        try:
            df_vwap = self.get_df_vwap()
            if df_vwap.empty:
                return pd.DataFrame()
            df_vwap['volXclose'] = df_vwap['close'] * df_vwap['volume']
            df_vwap['cum_vol'] = df_vwap['volume'].cumsum()
            df_vwap['cum_volXclose'] = (df_vwap['volume'] * (df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3).cumsum()
            df_vwap['VWAP'] = df_vwap['cum_volXclose'] / df_vwap['cum_vol']
            df_vwap = df_vwap.fillna(0)
            logger.debug("VWAP calculation completed")
            return df_vwap
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return pd.DataFrame()

    def get_df_vwma(self, symbol=None, timeframe='1d', num_bars=100, cache_time=300):
        if symbol is None:
            symbol = self.symbol
        current_time = time.time()
        if self._vwma_data_cache is not None and current_time - self._vwma_cache_time < cache_time:
            logger.debug("Using cached VWMA data")
            return self._vwma_data_cache
        logger.info(f"Fetching data for VWMA calculation for {symbol}, {timeframe} timeframe")
        try:
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=num_bars)
            df_vwma = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_vwma['timestamp'] = pd.to_datetime(df_vwma['timestamp'], unit='ms')
            self._vwma_data_cache = df_vwma
            self._vwma_cache_time = current_time
            logger.debug(f"VWMA data prepared for {symbol}")
            return df_vwma
        except Exception as e:
            logger.error(f"Error preparing VWMA data: {e}")
            return pd.DataFrame()

    def vwma_indi(self, cache_time=300):
        logger.info("Starting VWMA indicator calculation")
        try:
            df_vwma = self.get_df_vwma()
            if df_vwma.empty:
                return pd.DataFrame()
            df_vwma['SMA(41)'] = df_vwma.close.rolling(41).mean()
            df_vwma['SMA(20)'] = df_vwma.close.rolling(20).mean()
            df_vwma['SMA(75)'] = df_vwma.close.rolling(75).mean()
            df_sma = df_vwma.fillna(0)
            vwmas = [20, 41, 75]
            for n in vwmas:
                df_vwma[f'sum_vol{n}'] = df_vwma['volume'].rolling(min_periods=1, window=n).sum()
                df_vwma['volXclose'] = df_vwma['volume'] * df_vwma['close']
                df_vwma[f'vXc{n}'] = df_vwma['volXclose'].rolling(min_periods=1, window=n).sum()
                df_vwma[f'VWMA({n})'] = df_vwma[f'vXc{n}'] / df_vwma[f'sum_vol{n}']
                df_vwma.loc[df_vwma[f'VWMA({n})'] > df_vwma['SMA(41)'], f'41sig{n}'] = 'BUY'
                df_vwma.loc[df_vwma[f'VWMA({n})'] > df_vwma['SMA(20)'], f'20sig{n}'] = 'BUY'
                df_vwma.loc[df_vwma[f'VWMA({n})'] > df_vwma['SMA(75)'], f'75sig{n}'] = 'BUY'
                df_vwma.loc[df_vwma[f'VWMA({n})'] < df_vwma['SMA(41)'], f'41sig{n}'] = 'SELL'
                df_vwma.loc[df_vwma[f'VWMA({n})'] < df_vwma['SMA(20)'], f'20sig{n}'] = 'SELL'
                df_vwma.loc[df_vwma[f'VWMA({n})'] < df_vwma['SMA(75)'], f'75sig{n}'] = 'SELL'
            logger.debug("VWMA calculation completed")
            return df_vwma
        except Exception as e:
            logger.error(f"Error calculating VWMA: {e}")
            return pd.DataFrame()

    def analyze_market(self):
        logger.info("Starting market analysis")
        try:
            df_vwma = self.vwma_indi()
            if df_vwma.empty:
                return None
            last_row = df_vwma.iloc[-1]
            vwma20 = last_row['VWMA(20)']
            vwma41 = last_row['VWMA(41)']
            vwma75 = last_row['VWMA(75)']
            _, current_bid = self.ask_bid(self.symbol)
            if current_bid is None:
                return None
            logger.info(f"Market analysis - Current price: {current_bid}, VWMA(20): {vwma20}, VWMA(41): {vwma41}")
            if vwma20 > vwma41:
                signal = 'BUY'
                strength = 'Strong' if vwma20 > vwma75 else 'Moderate'
            else:
                signal = 'SELL'
                strength = 'Strong' if vwma20 < vwma75 else 'Moderate'
            logger.info(f"Trading signal: {signal} ({strength})")
            return {'signal': signal, 'strength': strength, 'vwma20': vwma20, 'vwma41': vwma41, 'vwma75': vwma75, 'current_price': current_bid, 'df_vwma': df_vwma}
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return None

    def check_and_manage_positions(self):
        logger.info("Checking and managing positions")
        try:
            position_info = self.open_positions(self.symbol)
            is_open = position_info[1]
            is_long = position_info[3]
            position_size = position_info[2]
            if is_open:
                logger.info(f"Position status: {'Long' if is_long else 'Short'}, Size: {position_size}")
                self.pnl_close(self.symbol, self.target, self.max_loss)
            else:
                logger.info("No open positions")
            return is_open, is_long, position_size
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            return False, None, 0

    def run(self):
        logger.info(f"Starting VWMA trader for {self.symbol}")
        try:
            if self.size_kill():
                logger.warning("Size kill activated, exiting run")
                return
            analysis = self.analyze_market()
            if analysis is None:
                return
            is_open, is_long, position_size = self.check_and_manage_positions()
            logger.info(f"Market summary - Signal: {analysis['signal']}, Current: {analysis['current_price']}, VWMA(20): {analysis['vwma20']}, VWMA(41): {analysis['vwma41']}")
            logger.info("VWMA trading cycle completed")
        except Exception as e:
            logger.error(f"Error in main trading cycle: {e}")

def create_default_config():
    if not os.path.exists('bitfinex_vwma_config.ini'):
        config = configparser.ConfigParser()
        config['Trading'] = {'symbol': 'BTC:USTF0', 'size': '1', 'target': '9', 'max_loss': '-8'}
        config['Indicators'] = {'timeframe': '1d', 'limit': '100'}
        with open('bitfinex_vwma_config.ini', 'w') as configfile:
            config.write(configfile)
        logger.info("Created default bitfinex_vwma_config.ini")

def main():
    create_default_config()
    try:
        exchange = ccxt.bitfinex({'enableRateLimit': True, 'apiKey': bitfinex_key, 'secret': bitfinex_secret})
        trader = VWMATrader(exchange)
        trader.run()
        return trader
    except Exception as e:
        logger.error(f"Error initializing trader: {e}")
        return None

if __name__ == "__main__":
    try:
        trader = main()
        if trader:
            df_vwma = trader.vwma_indi()
            print(df_vwma.tail())
            positions, is_open, position_size, is_long, index_pos = trader.open_positions(trader.symbol)
            current_ask, current_bid = trader.ask_bid(trader.symbol)
            last_row = df_vwma.iloc[-1]
            vwma20 = last_row['VWMA(20)']
            vwma41 = last_row['VWMA(41)']
            print(f"Current price: {current_bid}, VWMA(20): {vwma20}, VWMA(41): {vwma41}")
            if is_open:
                pnl_close_result, in_position, size, is_long = trader.pnl_close(trader.symbol, trader.target, trader.max_loss)
                print(f"Position status: {'Long' if is_long else 'Short'}, Size: {size}")
    except Exception as e:
        print(f"Error running script: {e}") 