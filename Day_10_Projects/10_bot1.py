############# Coding trading bot #1 - sma bot w/ob data 2024

import ccxt, pandas as pd, sys, os, time, schedule, configparser, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import key as xP_KEY, secret as xP_SECRET

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

# Rate limiting for API calls
last_api_call_time = 0
min_call_interval = 1.0  # seconds - more conservative

def rate_limited_api_call(func, *args, **kwargs):
    """Implement rate limiting for API calls"""
    global last_api_call_time
    current_time = time.time()
    time_since_last_call = current_time - last_api_call_time
    
    if time_since_last_call < min_call_interval:
        sleep_time = min_call_interval - time_since_last_call
        time.sleep(sleep_time)
    
    result = func(*args, **kwargs)
    last_api_call_time = time.time()
    
    return result

def load_config(config_file='config.ini'):
    """Load configuration from file or use defaults"""
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        
        return {
            'symbol': config['Trading']['symbol'],
            'pos_size': int(config['Trading']['position_size']),
            'target': float(config['Trading']['target']),
            'max_loss': float(config['Trading']['max_loss']),
            'vol_decimal': float(config['Trading']['vol_decimal'])
        }
    except:
        logger.warning(f"Could not load config from {config_file}, using defaults")
        return {
            'symbol': 'BTC/USD:BTC',
            'pos_size': 30,
            'target': 8,
            'max_loss': -9,
            'vol_decimal': 0.4
        }

class TradingBot:
    def __init__(self, exchange, config=None):
        """Initialize trading bot with exchange connection and configuration"""
        self.exchange = exchange
        self.retry_count = {}
        
        # Load configuration
        if config is None:
            config = load_config()
            
        self.symbol = config['symbol']
        self.pos_size = config['pos_size']
        self.target = config['target']
        self.max_loss = config['max_loss']
        self.vol_decimal = config['vol_decimal']
        self.params = {'timeInForce': 'PostOnly'}
        
        # Cache for API data
        self._daily_sma_cache = None
        self._daily_sma_cache_time = 0
        self._f15_sma_cache = None
        self._f15_sma_cache_time = 0

    def ask_bid(self):
        """Get the current ask and bid prices"""
        logger.debug(f"Fetching order book for {self.symbol}")
        ob = rate_limited_api_call(self.exchange.fetch_order_book, self.symbol)

        bid = ob['bids'][0][0]
        ask = ob['asks'][0][0]

        return ask, bid
        
    def daily_sma(self, cache_time=300):
        """Calculate daily SMA with caching to reduce API calls"""
        current_time = time.time()
        
        # Use cached data if available and recent
        if self._daily_sma_cache is not None and current_time - self._daily_sma_cache_time < cache_time:
            logger.debug("Using cached daily SMA data")
            return self._daily_sma_cache
        
        logger.info('Starting daily indicators calculation...')

        timeframe = '1d'
        num_bars = 100

        bars = rate_limited_api_call(self.exchange.fetch_ohlcv, self.symbol, timeframe=timeframe, limit=num_bars)
        df_d = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_d['timestamp'] = pd.to_datetime(df_d['timestamp'], unit='ms')

        # DAILY SMA - 20 day
        df_d['sma20_d'] = df_d.close.rolling(20).mean()

        # if bid < the 20 day sma then = BEARISH, if bid > 20 day sma = BULLISH
        bid = self.ask_bid()[1]
        
        # if sma > bid = SELL, if sma < bid = BUY
        df_d.loc[df_d['sma20_d']>bid, 'sig'] = 'SELL'
        df_d.loc[df_d['sma20_d']<bid, 'sig'] = 'BUY'

        # Cache the result
        self._daily_sma_cache = df_d
        self._daily_sma_cache_time = current_time

        return df_d

    def f15_sma(self, cache_time=150):
        """Calculate 15-minute SMA with caching to reduce API calls"""
        current_time = time.time()
        
        # Use cached data if available and recent
        if self._f15_sma_cache is not None and current_time - self._f15_sma_cache_time < cache_time:
            logger.debug("Using cached 15m SMA data")
            return self._f15_sma_cache
            
        logger.info('Starting 15 min SMA calculation...')

        timeframe = '15m'
        num_bars = 100

        try:
            # Add debugging information
            logger.debug(f"Attempting to fetch OHLCV data for symbol: {self.symbol}, timeframe: {timeframe}")
            
            # Add additional parameters that might be required
            params = {}
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, self.symbol, timeframe=timeframe, limit=num_bars, params=params)
            
            if not bars or len(bars) == 0:
                logger.error(f"No OHLCV data returned for {self.symbol}")
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                             'sma20_15', 'bp_1', 'bp_2', 'sp_1', 'sp_2'])
            
            df_f = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_f['timestamp'] = pd.to_datetime(df_f['timestamp'], unit='ms')

            # 15m SMA - 20 period
            df_f['sma20_15'] = df_f.close.rolling(20).mean()

            # BUY PRICE 1+2 AND SELL PRICE1+2
            df_f['bp_1'] = df_f['sma20_15'] * 1.001 
            df_f['bp_2'] = df_f['sma20_15'] * .997
            df_f['sp_1'] = df_f['sma20_15'] * .999
            df_f['sp_2'] = df_f['sma20_15'] * 1.003
            
            # Cache the result
            self._f15_sma_cache = df_f
            self._f15_sma_cache_time = current_time

            return df_f
            
        except Exception as e:
            logger.error(f"Error in f15_sma: {e}")
            logger.info("Trying to debug symbol format issues...")
            
            try:
                # Get available markets to check valid symbols
                markets = rate_limited_api_call(self.exchange.fetch_markets)
                available_symbols = [market['symbol'] for market in markets]
                logger.info(f"Some available symbols: {available_symbols[:5]}...")
                
                # Check if our symbol is in the list
                if self.symbol in available_symbols:
                    logger.info(f"Symbol '{self.symbol}' is valid")
                else:
                    logger.warning(f"Symbol '{self.symbol}' not found in available markets")
                    
                    # Try to find similar symbols
                    similar_symbols = [s for s in available_symbols if 'BTC' in s and 'USD' in s]
                    logger.info(f"Similar symbols that might work: {similar_symbols[:5]}")
            except Exception as debug_error:
                logger.error(f"Error while debugging symbol: {debug_error}")
            
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                        'sma20_15', 'bp_1', 'bp_2', 'sp_1', 'sp_2'])

    def calculate_position_size(self):
        """Calculate position size based on volatility"""
        # Get historical volatility
        df = self.f15_sma()
        volatility = df['close'].pct_change().std() * 100
        
        # Scale position size inversely with volatility
        adjusted_size = self.pos_size * (1 / (1 + volatility))
        return max(int(adjusted_size), 1)  # Ensure minimum size of 1

    def open_positions(self):
        """Check for open positions"""
        logger.debug("Checking open positions")
        params = {'type':'swap', 'code':'USD'}
        phe_bal = rate_limited_api_call(self.exchange.fetch_balance, params=params)
        open_positions = phe_bal['info']['data']['positions']
        
        openpos_side = open_positions[0]['side']
        openpos_size = open_positions[0]['size']

        if openpos_side == ('Buy'):
            openpos_bool = True 
            long = True 
        elif openpos_side == ('Sell'):
            openpos_bool = True
            long = False
        else:
            openpos_bool = False
            long = None 

        return open_positions, openpos_bool, openpos_size, long
    
    def kill_switch(self):
        """Gracefully close positions using limit orders"""
        logger.info('Starting the kill switch')
        openposi = self.open_positions()[1]  # true or false
        long = self.open_positions()[3]  # t or false
        kill_size = self.open_positions()[2]

        logger.info(f'Open position: {openposi}, Long: {long}, Size: {kill_size}')

        while openposi == True:
            logger.info('Starting kill switch loop until limit fills...')
            
            rate_limited_api_call(self.exchange.cancel_all_orders, self.symbol)
            openposi = self.open_positions()[1]
            long = self.open_positions()[3]
            kill_size = self.open_positions()[2]
            kill_size = int(kill_size)
            
            ask, bid = self.ask_bid()

            if long == False:
                rate_limited_api_call(self.exchange.create_limit_buy_order, self.symbol, kill_size, bid, self.params)
                logger.info(f'Created BUY to CLOSE order of {kill_size} {self.symbol} at ${bid}')
                logger.info('Waiting 30 seconds to see if it fills...')
                time.sleep(30)
            elif long == True:
                rate_limited_api_call(self.exchange.create_limit_sell_order, self.symbol, kill_size, ask, self.params)
                logger.info(f'Created SELL to CLOSE order of {kill_size} {self.symbol} at ${ask}')
                logger.info('Waiting 30 seconds to see if it fills...')
                time.sleep(30)
            else:
                logger.warning('Unexpected condition in kill switch function')

            openposi = self.open_positions()[1]

    def sleep_on_close(self):
        """Pause trading after a recent fill"""
        logger.debug('Checking if we need to sleep after recent trade')

        closed_orders = rate_limited_api_call(self.exchange.fetch_closed_orders, self.symbol)

        for ord in closed_orders[-1::-1]:
            sincelasttrade = 59  # how long we pause

            status = ord['info']['ordStatus']
            txttime = ord['info']['transactTimeNs']
            txttime = int(txttime)
            txttime = round((txttime/1000000000))  # bc in nanoseconds
            logger.debug(f'Order status: {status} with epoch {txttime}')

            if status == 'Filled':
                logger.info('Found order with last fill')
                logger.debug(f'Transaction time: {txttime}, Order status: {status}')
                
                orderbook = rate_limited_api_call(self.exchange.fetch_order_book, self.symbol)
                ex_timestamp = orderbook['timestamp']  # in ms 
                ex_timestamp = int(ex_timestamp/1000)
                
                time_spread = (ex_timestamp - txttime)/60

                if time_spread < sincelasttrade:
                    sleepy = round(sincelasttrade-time_spread)*60
                    sleepy_min = sleepy/60

                    logger.info(f'Time since last trade ({time_spread} mins) is less than {sincelasttrade} mins, sleeping for 60 secs')
                    time.sleep(60)
                else:
                    logger.info(f'It has been {time_spread} mins since last fill, no sleep needed')
                break 
            else:
                continue 

        logger.debug('Completed sleep on close check')

    def ob(self):
        """Analyze order book volume to determine market bias"""
        logger.info('Fetching and analyzing order book data')
        
        df = pd.DataFrame()
        
        for x in range(11):
            ob = rate_limited_api_call(self.exchange.fetch_order_book, self.symbol)
            bids, asks = ob['bids'], ob['asks']
            
            bid_vol = sum(vol for _, vol in bids)
            ask_vol = sum(vol for _, vol in asks)
            
            temp_df = pd.DataFrame({'bid_vol': [bid_vol], 'ask_vol': [ask_vol]})
            df = pd.concat([df, temp_df], ignore_index=True)
            
            logger.debug(f"Order book sample {x}: bid_vol={bid_vol}, ask_vol={ask_vol}")
            time.sleep(5)
        
        total_bidvol = df['bid_vol'].sum()
        total_askvol = df['ask_vol'].sum()
        logger.info(f'Last minute volume summary - Bid Vol: {total_bidvol} | Ask Vol: {total_askvol}')

        if total_bidvol > total_askvol:
            control_dec = (total_askvol/total_bidvol)
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
        if openpos_tf == True:
            if long == True:
                logger.info('Currently in a long position')
                if control_dec < self.vol_decimal:
                    vol_under_dec = True
                else:
                    logger.info(f'Volume ratio {control_dec} is not under threshold {self.vol_decimal}')
                    vol_under_dec = False
            else:
                logger.info('Currently in a short position')
                if control_dec < self.vol_decimal:
                    vol_under_dec = True
                else:
                    logger.info(f'Volume ratio {control_dec} is not under threshold {self.vol_decimal}')
                    vol_under_dec = False
        else:
            logger.info('Not currently in a position')

        return vol_under_dec

    def check_circuit_breakers(self):
        """Check for extreme market conditions to pause trading if necessary"""
        # Check for extreme volatility
        df = self.f15_sma()
        recent_volatility = df['close'].pct_change().rolling(5).std().iloc[-1] * 100
        
        if recent_volatility > 5:  # 5% volatility threshold
            logger.warning(f"Circuit breaker triggered - high volatility: {recent_volatility}%")
            return False  # Don't trade
            
        # Check for abnormal spread
        ask, bid = self.ask_bid()
        spread_pct = (ask - bid) / bid * 100
        
        if spread_pct > 0.5:  # 0.5% spread threshold
            logger.warning(f"Circuit breaker triggered - high spread: {spread_pct}%")
            return False
            
        return True  # Safe to trade

    def pnl_close(self):
        """Check if we need to close positions based on PnL targets"""
        logger.info('Checking if it is time to exit positions')

        try:
            params = {'type':"swap", 'code':'USD'}
            pos_dict = rate_limited_api_call(self.exchange.fetch_positions, params=params)[0]
            
            side = pos_dict['side']
            size = pos_dict['contracts']
            entry_price = float(pos_dict['entryPrice'])
            leverage = float(pos_dict['leverage'])

            current_price = self.ask_bid()[1]

            logger.info(f'Position details - Side: {side}, Entry: {entry_price}, Leverage: {leverage}')
            
            if side == 'long':
                diff = current_price - entry_price
                long = True
            else: 
                diff = entry_price - current_price
                long = False

            try: 
                perc = round(((diff/entry_price) * leverage), 10)
            except:
                perc = 0

            perc = 100*perc
            logger.info(f'Current PNL: {perc}%')

            pnlclose = False 
            in_pos = False

            if perc > 0:
                in_pos = True
                logger.info('Currently in a profitable position')
                if perc > self.target:
                    logger.info(f'Profit target of {self.target}% reached ({perc}%), checking volume before exit')
                    pnlclose = True
                    vol_under_dec = self.ob()
                    if vol_under_dec == True:
                        logger.info(f'Volume is under threshold {self.vol_decimal}, waiting 30s')
                        time.sleep(30)
                    else:
                        logger.info(f'Initiating exit - profit target reached and volume conditions met')
                        self.kill_switch()
                else:
                    logger.info(f'Profit target not yet reached: {perc}% vs {self.target}%')

            elif perc < 0:
                in_pos = True

                if perc <= self.max_loss:
                    logger.warning(f'Stop loss triggered at {perc}% (limit: {self.max_loss}%), initiating exit')
                    self.kill_switch()
                else:
                    logger.info(f'Position underwater at {perc}%, but above stop loss of {self.max_loss}%')

            else:
                logger.info('Not in a position')

            if in_pos == True:
                # Check stop based on SMA
                df_f = self.f15_sma()
                last_sma15 = df_f.iloc[-1]['sma20_15']
                last_sma15 = int(last_sma15)
                
                curr_bid = self.ask_bid()[1]
                curr_bid = int(curr_bid)
                
                sl_val = last_sma15 * 1.008
                logger.debug(f'SMA stop check - Current: {curr_bid}, Stop value: {sl_val}')
                
                # Additional stop based on ATR could be implemented here
            
            logger.debug('PNL check completed')
            return pnlclose, in_pos, size, long
            
        except Exception as e:
            logger.error(f"Error in PNL calculation: {e}")
            return False, False, 0, None

    def bot(self):
        """Main trading logic"""
        try:
            # Check if it's safe to trade based on market conditions
            if not self.check_circuit_breakers():
                logger.warning("Circuit breakers active - skipping trading cycle")
                time.sleep(60)
                return
                
            # Check if we need to exit positions
            pnl_close_result = self.pnl_close()
            
            # Check if we need to wait after recent trades
            self.sleep_on_close()

            # Get market data
            df_d = self.daily_sma()
            df_f = self.f15_sma()
            ask, bid = self.ask_bid()

            # Trading signal - long or short
            sig = df_d.iloc[-1]['sig']
            logger.debug(f"Trading signal: {sig}")

            # Calculate dynamic position size based on volatility
            calculated_pos_size = self.calculate_position_size()
            open_size = calculated_pos_size // 2
            logger.debug(f"Using calculated position size: {calculated_pos_size}, split size: {open_size}")

            # Check if we can enter a position
            in_pos = pnl_close_result[1]
            curr_size = int(self.open_positions()[2])
            
            curr_p = bid 
            last_sma15 = df_f.iloc[-1]['sma20_15']

            # Only enter if we're not in a position and size is below limit
            if (in_pos == False) and (curr_size < self.pos_size):
                logger.info("Looking for entry opportunities")
                rate_limited_api_call(self.exchange.cancel_all_orders, self.symbol)

                if (sig == 'BUY') and (curr_p > last_sma15):
                    # Buy setup
                    logger.info('Setting up BUY orders')
                    bp_1 = df_f.iloc[-1]['bp_1']
                    bp_2 = df_f.iloc[-1]['bp_2']
                    logger.info(f'Buy prices - BP1: {bp_1}, BP2: {bp_2}')
                    
                    rate_limited_api_call(self.exchange.cancel_all_orders, self.symbol)
                    rate_limited_api_call(self.exchange.create_limit_buy_order, self.symbol, open_size, bp_1, self.params)
                    rate_limited_api_call(self.exchange.create_limit_buy_order, self.symbol, open_size, bp_2, self.params)

                    logger.info('Buy orders placed, waiting for fills (120s)')
                    time.sleep(120)
                    
                elif (sig == 'SELL') and (curr_p < last_sma15):
                    # Sell setup
                    logger.info('Setting up SELL orders')
                    sp_1 = df_f.iloc[-1]['sp_1']
                    sp_2 = df_f.iloc[-1]['sp_2']
                    logger.info(f'Sell prices - SP1: {sp_1}, SP2: {sp_2}')
                    
                    rate_limited_api_call(self.exchange.cancel_all_orders, self.symbol)
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
                wait_time = min(60 * self.retry_count[self.symbol], 300)  # Up to 5 minutes
                logger.warning(f"Rate limit hit, backing off for {wait_time}s (attempt {self.retry_count[self.symbol]})")
                time.sleep(wait_time)
            else:
                self.retry_count[self.symbol] = 0  # Reset on other errors
                time.sleep(30)

            # Check if we need to pause due to too many rate limits
            if max(self.retry_count.values(), default=0) > 5:
                logger.warning("Too many rate limits - pausing for 15 minutes")
                time.sleep(900)  # 15 minute cooldown
                self.retry_count = {}  # Reset counts

            # If rate limited frequently:
            schedule.clear()
            schedule.every(60).seconds.do(self.bot)  # Run less frequently

def create_default_config():
    """Create a default configuration file if none exists"""
    config = configparser.ConfigParser()
    config['Trading'] = {
        'symbol': 'BTC/USD:BTC',
        'position_size': '30',
        'target': '8',
        'max_loss': '-9',
        'vol_decimal': '0.4'
    }
    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    logger.info("Created default config.ini file")

def main():
    """Main function to run the trading bot"""
    # Check for config file, create if doesn't exist
    if not os.path.exists('config.ini'):
        create_default_config()
    
    # Initialize exchange connection
    phemex = ccxt.phemex({
        'enableRateLimit': True, 
        'apiKey': xP_KEY,
        'secret': xP_SECRET
    })
    
    # Get available markets to check symbols before starting
    try:
        logger.info("Fetching available markets to verify symbol format...")
        markets = rate_limited_api_call(phemex.fetch_markets)
        available_symbols = [market['symbol'] for market in markets]
        logger.info(f"Some available symbols: {available_symbols[:5]}...")
        
        # Load config
        config = load_config()
        symbol = config['symbol']
        
        if symbol not in available_symbols:
            logger.warning(f"Symbol '{symbol}' not found in available markets")
            # Try to find similar symbols
            similar_symbols = [s for s in available_symbols if 'BTC' in s and 'USD' in s]
            logger.info(f"Similar BTC/USD symbols that might work: {similar_symbols[:5]}")
            
            if similar_symbols:
                suggested_symbol = similar_symbols[0]
                logger.info(f"Automatically using suggested symbol: {suggested_symbol}")
                
                # Update config with suggested symbol
                config['symbol'] = suggested_symbol
                
                # Update config file
                config_parser = configparser.ConfigParser()
                config_parser.read('config.ini')
                config_parser['Trading']['symbol'] = suggested_symbol
                with open('config.ini', 'w') as configfile:
                    config_parser.write(configfile)
    except Exception as e:
        logger.error(f"Error verifying symbol: {e}")
    
    # Create trading bot instance
    bot = TradingBot(phemex, config)
    
    # Schedule the bot to run every 28 seconds
    schedule.every(28).seconds.do(bot.bot)
    
    logger.info(f"Trading bot started with symbol: {bot.symbol}")
    
    # Main loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)  # Add small sleep to prevent high CPU usage
        except ccxt.NetworkError as e:
            logger.error(f"Network error occurred: {e}")
            time.sleep(60)  # Longer sleep time for network issues
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error occurred: {e}")
            time.sleep(45)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()