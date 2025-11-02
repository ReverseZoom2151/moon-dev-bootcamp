############# Coding VWMA indicator 2024

import ccxt, pandas as pd, time, sys, os, logging, configparser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import key as xP_KEY, secret as xP_SECRET
from ta.momentum import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vwma_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VWMATrader")

# Rate limiting for API calls
last_api_call_time = 0
min_call_interval = 0.2  # seconds

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

def load_config(config_file='vwma_config.ini'):
    """Load configuration from file or use defaults"""
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        
        return {
            'symbol': config['Trading']['symbol'],
            'size': int(config['Trading']['size']),
            'target': float(config['Trading']['target']),
            'max_loss': float(config['Trading']['max_loss']),
            'timeframe': config['Indicators']['timeframe'],
            'limit': int(config['Indicators']['limit'])
        }
    except:
        logger.warning(f"Could not load config from {config_file}, using defaults")
        return {
            'symbol': 'BTC/USD:BTC',
            'size': 1,
            'target': 9,
            'max_loss': -8,
            'timeframe': '1d',
            'limit': 100
        }

class VWMATrader:
    def __init__(self, exchange, config=None):
        """Initialize VWMA trader with exchange connection and configuration"""
        self.exchange = exchange
        
        # Load configuration
        if config is None:
            config = load_config()
            
        self.symbol = config['symbol']
        self.size = config['size']
        self.target = config['target']
        self.max_loss = config['max_loss']
        self.timeframe = config['timeframe']
        self.limit = config['limit']
        self.params = {'timeInForce': 'PostOnly'}
        
        # Cache for API data
        self._vwma_data_cache = None
        self._vwma_cache_time = 0
        self._vwap_data_cache = None
        self._vwap_cache_time = 0
        self._sma_data_cache = None
        self._sma_cache_time = 0
        self._rsi_data_cache = None
        self._rsi_cache_time = 0
        
        logger.info(f"VWMATrader initialized with symbol: {self.symbol}")

    def get_symbol_position_index(self, symbol=None):
        """Determine the position index for the given symbol"""
        if symbol is None:
            symbol = self.symbol
            
        if symbol == 'BTC/USD:BTC':
            index_pos = 4
        elif symbol == 'ETH/USD:ETH':
            index_pos = 2
        elif symbol == 'ETHUSD':
            index_pos = 3
        elif symbol == 'DOGEUSD':
            index_pos = 1
        elif symbol == 'u100000SHIBUSD':
            index_pos = 0
        else:
            index_pos = None
            
        return index_pos

    def open_positions(self, symbol=None):
        """Check for open positions"""
        if symbol is None:
            symbol = self.symbol
            
        logger.debug(f"Checking open positions for {symbol}")
        
        # Get the position index for the symbol
        index_pos = self.get_symbol_position_index(symbol)
        
        try:
            params = {'type':'swap', 'code':'USD'}
            phe_bal = rate_limited_api_call(self.exchange.fetch_balance, params=params)
            open_positions = phe_bal['info']['data']['positions']
            
            # Extract position details
            openpos_side = open_positions[index_pos]['side']
            openpos_size = open_positions[index_pos]['size']
            
            # Determine position status
            if openpos_side == 'Buy':
                openpos_bool = True
                long = True
            elif openpos_side == 'Sell':
                openpos_bool = True
                long = False
            else:
                openpos_bool = False
                long = None
                
            logger.info(f"Position check - In position: {openpos_bool}, Size: {openpos_size}, Long: {long}, Index: {index_pos}")
            return open_positions, openpos_bool, openpos_size, long, index_pos
            
        except Exception as e:
            logger.error(f"Error checking open positions: {e}")
            return [], False, 0, None, index_pos

    def ask_bid(self, symbol=None):
        """Get the current ask and bid prices"""
        if symbol is None:
            symbol = self.symbol
            
        logger.debug(f"Fetching order book for {symbol}")
        
        try:
            ob = rate_limited_api_call(self.exchange.fetch_order_book, symbol)
            bid = ob['bids'][0][0]
            ask = ob['asks'][0][0]
            
            logger.info(f"Current prices for {symbol} - Ask: {ask}, Bid: {bid}")
            return ask, bid
            
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None, None

    def kill_switch(self, symbol=None):
        """Gracefully close positions using limit orders"""
        if symbol is None:
            symbol = self.symbol
            
        logger.info(f"Starting the kill switch for {symbol}")
        
        try:
            openposi, long, kill_size = None, None, None
            
            # Get position info
            position_info = self.open_positions(symbol)
            openposi = position_info[1]  # openpos_bool
            long = position_info[3]  # long boolean
            kill_size = position_info[2]  # size
            
            logger.info(f"Kill switch - In position: {openposi}, Long: {long}, Size: {kill_size}")
            
            # Loop until position is closed
            while openposi is True:
                logger.info("Starting kill switch loop until limit fills...")
                
                # Cancel existing orders
                rate_limited_api_call(self.exchange.cancel_all_orders, symbol)
                
                # Get updated position info
                position_info = self.open_positions(symbol)
                openposi = position_info[1]
                long = position_info[3]
                kill_size = position_info[2]
                kill_size = int(kill_size)
                
                # Get current prices
                ask, bid = self.ask_bid(symbol)
                
                # Create closing orders based on position side
                if long is False:
                    rate_limited_api_call(self.exchange.create_limit_buy_order, symbol, kill_size, bid, self.params)
                    logger.info(f"Created BUY to CLOSE order of {kill_size} {symbol} at ${bid}")
                    logger.info("Waiting 30 seconds to see if it fills...")
                    time.sleep(30)
                elif long is True:
                    rate_limited_api_call(self.exchange.create_limit_sell_order, symbol, kill_size, ask, self.params)
                    logger.info(f"Created SELL to CLOSE order of {kill_size} {symbol} at ${ask}")
                    logger.info("Waiting 30 seconds to see if it fills...")
                    time.sleep(30)
                else:
                    logger.warning("Unexpected condition in kill switch function")
                    break
                
                # Check if position is closed
                position_info = self.open_positions(symbol)
                openposi = position_info[1]
                
            logger.info(f"Kill switch completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error in kill switch: {e}")

    def pnl_close(self, symbol=None, target=None, max_loss=None):
        """Check if we need to close positions based on PnL targets"""
        if symbol is None:
            symbol = self.symbol
            
        if target is None:
            target = self.target
            
        if max_loss is None:
            max_loss = self.max_loss
            
        logger.info(f"Checking if it's time to exit for {symbol}...")
        
        try:
            # Get position details
            params = {'type': 'swap', 'code': 'USD'}
            pos_dict = rate_limited_api_call(self.exchange.fetch_positions, params=params)
            
            # Get the index for the symbol
            index_pos = self.open_positions(symbol)[4]
            pos_dict = pos_dict[index_pos]
            
            # Extract position data
            side = pos_dict['side']
            size = pos_dict['contracts']
            entry_price = float(pos_dict['entryPrice'])
            leverage = float(pos_dict['leverage'])
            
            # Get current market price
            current_price = self.ask_bid(symbol)[1]
            
            logger.info(f"Position details - Side: {side}, Entry: {entry_price}, Leverage: {leverage}")
            
            # Calculate PnL
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
                
            perc = 100 * perc
            logger.info(f"For {symbol} current PnL: {perc}%")
            
            pnlclose = False
            in_pos = False
            
            # Check profitability
            if perc > 0:
                in_pos = True
                logger.info(f"For {symbol} we are in a profitable position")
                
                if perc > target:
                    logger.info(f"Profit target of {target}% reached ({perc}%), initiating exit")
                    pnlclose = True
                    self.kill_switch(symbol)
                else:
                    logger.info(f"Profit target not yet reached: {perc}% vs {target}%")
                    
            elif perc < 0:
                in_pos = True
                
                if perc <= max_loss:
                    logger.warning(f"Stop loss triggered at {perc}% (limit: {max_loss}%), initiating exit")
                    self.kill_switch(symbol)
                else:
                    logger.info(f"Position underwater at {perc}%, but above stop loss of {max_loss}%")
                    
            else:
                logger.info("Not in a position")
                
            logger.debug(f"PnL check completed for {symbol}")
            return pnlclose, in_pos, size, long
            
        except Exception as e:
            logger.error(f"Error in PnL calculation: {e}")
            return False, False, 0, None

    def size_kill(self):
        """Emergency kill switch if position size exceeds risk limits"""
        max_risk = 1000
        logger.debug("Checking position size against risk limits")
        
        try:
            params = {'type': 'swap', 'code': 'USD'}
            all_phe_balance = rate_limited_api_call(self.exchange.fetch_balance, params=params)
            open_positions = all_phe_balance['info']['data']['positions']
            
            try:
                pos_cost = float(open_positions[0]['posCost'])
                openpos_side = open_positions[0]['side']
                openpos_size = open_positions[0]['size']
            except:
                pos_cost = 0
                openpos_side = 0
                openpos_size = 0
                
            logger.info(f"Position cost: {pos_cost}, Side: {openpos_side}")
            
            # Check if position exceeds risk limit
            if pos_cost > max_risk:
                logger.critical(f"EMERGENCY KILL SWITCH ACTIVATED - Position size {pos_cost} exceeds max risk {max_risk}")
                self.kill_switch(self.symbol)
                time.sleep(300)  # Wait 5 minutes after emergency exit
                return True
            else:
                logger.debug(f"Size check passed - Position cost {pos_cost} within risk limit {max_risk}")
                return False
                
        except Exception as e:
            logger.error(f"Error in size kill check: {e}")
            return False

    def df_sma(self, symbol=None, timeframe=None, limit=None, sma=20, cache_time=300):
        """Calculate SMA indicators with caching to reduce API calls"""
        if symbol is None:
            symbol = self.symbol
            
        if timeframe is None:
            timeframe = self.timeframe
            
        if limit is None:
            limit = self.limit
            
        current_time = time.time()
        
        # Use cached data if available and recent
        if self._sma_data_cache is not None and current_time - self._sma_cache_time < cache_time:
            logger.debug("Using cached SMA data")
            return self._sma_data_cache
            
        logger.info(f"Starting SMA calculation for {symbol}, {timeframe} timeframe")
        
        try:
            # Fetch market data
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
            df_sma = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_sma['timestamp'] = pd.to_datetime(df_sma['timestamp'], unit='ms')
            
            # Calculate SMA
            df_sma[f'sma{sma}_{timeframe}'] = df_sma.close.rolling(sma).mean()
            
            # Get current market price
            bid = self.ask_bid(symbol)[1]
            
            # Generate signals
            df_sma.loc[df_sma[f'sma{sma}_{timeframe}'] > bid, 'sig'] = 'SELL'
            df_sma.loc[df_sma[f'sma{sma}_{timeframe}'] < bid, 'sig'] = 'BUY'
            
            # Support and resistance
            df_sma['support'] = df_sma[:-1]['close'].min()
            df_sma['resis'] = df_sma[:-1]['close'].max()
            
            # Cache the result
            self._sma_data_cache = df_sma
            self._sma_cache_time = current_time
            
            logger.debug(f"SMA calculation completed for {symbol}")
            return df_sma
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.DataFrame()

    def df_rsi(self, symbol=None, timeframe=None, limit=None, cache_time=300):
        """Calculate RSI indicator with caching to reduce API calls"""
        if symbol is None:
            symbol = self.symbol
            
        if timeframe is None:
            timeframe = self.timeframe
            
        if limit is None:
            limit = self.limit
            
        current_time = time.time()
        
        # Use cached data if available and recent
        if self._rsi_data_cache is not None and current_time - self._rsi_cache_time < cache_time:
            logger.debug("Using cached RSI data")
            return self._rsi_data_cache
            
        logger.info(f"Starting RSI calculation for {symbol}, {timeframe} timeframe")
        
        try:
            # Fetch market data
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
            df_rsi = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_rsi['timestamp'] = pd.to_datetime(df_rsi['timestamp'], unit='ms')
            
            # Get current market price
            bid = self.ask_bid(symbol)[1]
            
            # Calculate RSI
            rsi = RSIIndicator(df_rsi['close'])
            df_rsi['rsi'] = rsi.rsi()
            
            # Cache the result
            self._rsi_data_cache = df_rsi
            self._rsi_cache_time = current_time
            
            logger.debug(f"RSI calculation completed for {symbol}")
            return df_rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.DataFrame()

    def get_df_vwap(self, symbol=None, timeframe=None, limit=None, cache_time=300):
        """Get dataframe for VWAP calculation with caching"""
        if symbol is None:
            symbol = self.symbol
            
        if timeframe is None:
            timeframe = self.timeframe
            
        if limit is None:
            limit = self.limit
            
        current_time = time.time()
        
        # Use cached data if available and recent
        if self._vwap_data_cache is not None and current_time - self._vwap_cache_time < cache_time:
            logger.debug("Using cached VWAP data")
            return self._vwap_data_cache
            
        logger.info(f"Fetching data for VWAP calculation for {symbol}")
        
        try:
            # Fetch market data
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
            df_vwap = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_vwap['timestamp'] = pd.to_datetime(df_vwap['timestamp'], unit='ms')
            
            # Calculate price range
            lo = df_vwap['low'].min()
            hi = df_vwap['high'].max()
            l2h = hi - lo
            avg = (hi + lo) / 2
            
            # Cache the result
            self._vwap_data_cache = df_vwap
            self._vwap_cache_time = current_time
            
            logger.debug(f"VWAP data prepared for {symbol}")
            return df_vwap
            
        except Exception as e:
            logger.error(f"Error preparing VWAP data: {e}")
            return pd.DataFrame()

    def vwap_indi(self, cache_time=300):
        """Calculate VWAP indicator"""
        logger.info("Starting VWAP indicator calculation")
        
        try:
            # Get base dataframe
            df_vwap = self.get_df_vwap()
            
            if df_vwap.empty:
                logger.error("Failed to get data for VWAP calculation")
                return pd.DataFrame()
            
            # Calculate volume x close price
            df_vwap['volXclose'] = df_vwap['close'] * df_vwap['volume']
            
            # Calculate cumulative sum of volume
            df_vwap['cum_vol'] = df_vwap['volume'].cumsum()
            
            # Calculate cumulative sum of volume x typical price
            df_vwap['cum_volXclose'] = (df_vwap['volume'] * (df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3).cumsum()
            
            # Calculate VWAP
            df_vwap['VWAP'] = df_vwap['cum_volXclose'] / df_vwap['cum_vol']
            df_vwap = df_vwap.fillna(0)
            
            logger.debug("VWAP calculation completed")
            return df_vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return pd.DataFrame()

    def get_df_vwma(self, symbol=None, timeframe='1d', num_bars=100, cache_time=300):
        """Get dataframe for VWMA calculation with caching"""
        if symbol is None:
            symbol = self.symbol
            
        current_time = time.time()
        
        # Use cached data if available and recent
        if self._vwma_data_cache is not None and current_time - self._vwma_cache_time < cache_time:
            logger.debug("Using cached VWMA data")
            return self._vwma_data_cache
            
        logger.info(f"Fetching data for VWMA calculation for {symbol}, {timeframe} timeframe")
        
        try:
            # Fetch market data
            bars = rate_limited_api_call(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=num_bars)
            df_vwma = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_vwma['timestamp'] = pd.to_datetime(df_vwma['timestamp'], unit='ms')
            
            # Cache the result
            self._vwma_data_cache = df_vwma
            self._vwma_cache_time = current_time
            
            logger.debug(f"VWMA data prepared for {symbol}")
            return df_vwma
            
        except Exception as e:
            logger.error(f"Error preparing VWMA data: {e}")
            return pd.DataFrame()

    def vwma_indi(self, cache_time=300):
        """Calculate VWMA indicator"""
        logger.info("Starting VWMA indicator calculation")
        
        try:
            # Get base dataframe
            df_vwma = self.get_df_vwma()
            
            if df_vwma.empty:
                logger.error("Failed to get data for VWMA calculation")
                return pd.DataFrame()
            
            # Calculate SMAs
            df_vwma['SMA(41)'] = df_vwma.close.rolling(41).mean()
            df_vwma['SMA(20)'] = df_vwma.close.rolling(20).mean()
            df_vwma['SMA(75)'] = df_vwma.close.rolling(75).mean()
            
            # Replace NaN values
            df_sma = df_vwma.fillna(0)
            
            # Calculate VWMAs for different periods
            vwmas = [20, 41, 75]
            for n in vwmas:
                # Sum of volume over the period
                df_vwma[f'sum_vol{n}'] = df_vwma['volume'].rolling(min_periods=1, window=n).sum()
                
                # Volume x close price
                df_vwma['volXclose'] = df_vwma['volume'] * df_vwma['close']
                
                # Sum of volume x close over the period
                df_vwma[f'vXc{n}'] = df_vwma['volXclose'].rolling(min_periods=1, window=n).sum()
                
                # Calculate VWMA
                df_vwma[f'VWMA({n})'] = df_vwma[f'vXc{n}'] / df_vwma[f'sum_vol{n}']
                
                # Generate signals comparing VWMA to SMAs
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
        """Analyze the market and generate trading signals"""
        logger.info("Starting market analysis")
        
        try:
            # Calculate VWMA indicator
            df_vwma = self.vwma_indi()
            
            if df_vwma.empty:
                logger.error("Failed to generate VWMA indicators")
                return None
                
            # Get the latest data row
            last_row = df_vwma.iloc[-1]
            
            # Extract key indicator values
            vwma20 = last_row['VWMA(20)']
            vwma41 = last_row['VWMA(41)']
            vwma75 = last_row['VWMA(75)']
            
            # Get current market price
            _, current_bid = self.ask_bid(self.symbol)
            
            logger.info(f"Market analysis - Current price: {current_bid}, VWMA(20): {vwma20}, VWMA(41): {vwma41}")
            
            # Generate signal based on VWMA crossovers
            if vwma20 > vwma41:
                signal = 'BUY'
                strength = 'Strong' if vwma20 > vwma75 else 'Moderate'
            else:
                signal = 'SELL'
                strength = 'Strong' if vwma20 < vwma75 else 'Moderate'
                
            logger.info(f"Trading signal: {signal} ({strength})")
            
            return {
                'signal': signal,
                'strength': strength,
                'vwma20': vwma20,
                'vwma41': vwma41,
                'vwma75': vwma75,
                'current_price': current_bid,
                'df_vwma': df_vwma
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return None

    def check_and_manage_positions(self):
        """Check and manage existing positions"""
        logger.info("Checking and managing positions")
        
        try:
            # Check if we have any open positions
            position_info = self.open_positions(self.symbol)
            is_open = position_info[1]
            position_size = position_info[2]
            is_long = position_info[3]
            
            if is_open:
                logger.info(f"Position status: {'Long' if is_long else 'Short'}, Size: {position_size}")
                
                # Check if we need to close based on PnL
                pnl_result = self.pnl_close(self.symbol, self.target, self.max_loss)
                
                # Additional position management logic could be added here
                # For example, trailing stops, partial profit taking, etc.
                
            else:
                logger.info("No open positions")
                
            return is_open, is_long, position_size
            
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            return False, None, 0

    def run(self):
        """Main execution function"""
        logger.info(f"Starting VWMA trader for {self.symbol}")
        
        try:
            # Check for emergency risk management
            if self.size_kill():
                logger.warning("Size kill activated, exiting run")
                return
                
            # Get market analysis
            analysis = self.analyze_market()
            
            if analysis is None:
                logger.error("Failed to analyze market, skipping trading cycle")
                return
                
            # Check and manage existing positions
            is_open, is_long, position_size = self.check_and_manage_positions()
            
            # Display current market status
            logger.info(f"Market summary - Signal: {analysis['signal']}, "
                       f"Current: {analysis['current_price']}, "
                       f"VWMA(20): {analysis['vwma20']}, "
                       f"VWMA(41): {analysis['vwma41']}")
                       
            # Add entry logic here if needed
            # For example, if not is_open and analysis['signal'] == 'BUY'...
            
            logger.info("VWMA trading cycle completed")
            
        except Exception as e:
            logger.error(f"Error in main trading cycle: {e}")

def create_default_config():
    """Create default configuration file if it doesn't exist"""
    if not os.path.exists('vwma_config.ini'):
        config = configparser.ConfigParser()
        
        config['Trading'] = {
            'symbol': 'BTC/USD:BTC',
            'size': '1',
            'target': '9',
            'max_loss': '-8'
        }
        
        config['Indicators'] = {
            'timeframe': '1d',
            'limit': '100'
        }
        
        with open('vwma_config.ini', 'w') as configfile:
            config.write(configfile)
            
        logger.info("Created default vwma_config.ini")

def main():
    """Main entry point for the VWMA trader"""
    # Create default config if needed
    create_default_config()
    
    # Initialize exchange connection
    try:
        phemex = ccxt.phemex({
            'enableRateLimit': True,
            'apiKey': xP_KEY,
            'secret': xP_SECRET
        })
        
        # Create trader instance
        trader = VWMATrader(phemex)
        
        # Run the trader once
        trader.run()
        
        # Return the trader object for further use
        return trader
        
    except Exception as e:
        logger.error(f"Error initializing trader: {e}")
        return None

if __name__ == "__main__":
    try:
        # Get market data and calculate indicators
        trader = main()
        
        if trader:
            df_vwma = trader.vwma_indi()
            print(df_vwma.tail())
            
            # Add error handling for potentially problematic API calls
            try:
                # Check if we have any open positions
                positions, is_open, position_size, is_long, index_pos = trader.open_positions(trader.symbol)
                
                # Check current market prices
                current_ask, current_bid = trader.ask_bid(trader.symbol)
                
                # Trading logic with position data
                last_row = df_vwma.iloc[-1]
                vwma20 = last_row['VWMA(20)']
                vwma41 = last_row['VWMA(41)']
                
                print(f"Current price: {current_bid}, VWMA(20): {vwma20}, VWMA(41): {vwma41}")
                
                if is_open:
                    pnl_close_result, in_position, size, is_long = trader.pnl_close(trader.symbol, trader.target, trader.max_loss)
                    print(f"Position status: {'Long' if is_long else 'Short'}, Size: {size}")
                    
            except ccxt.ExchangeError as e:
                print(f"API Error: {e}")
                print("Unable to fetch account data. Running in monitoring mode only.")
                
                # Still show basic indicator information even without position data
                last_row = df_vwma.iloc[-1]
                vwma20 = last_row['VWMA(20)']
                vwma41 = last_row['VWMA(41)']
                print(f"VWMA(20): {vwma20}, VWMA(41): {vwma41}")
                
                if vwma20 > vwma41:
                    print("VWMA indicates bullish conditions")
                else:
                    print("VWMA indicates bearish conditions")
                    
    except Exception as e:
        print(f"Error running script: {e}")