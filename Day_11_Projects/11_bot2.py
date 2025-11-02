############# Coding trading bot #2 - breakout bot 2024

'''
Trading bot that identifies breakout opportunities using support and resistance levels
on 15-minute timeframes over a 3-day period.

Strategy:
- Calculate support and resistance levels on 15m candles (289 periods = 3 days)
- On retest of these levels, place orders
- Target symbol: 'BTC/USD:BTC'
'''
import ccxt, os, sys, time, schedule, nice_funcs as n, logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import key as xP_hmv_KEY, secret as xP_hmv_SECRET

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
SYMBOL = 'BTC/USD:BTC'
POS_SIZE = 30  # Position size
TARGET_GAIN = 9  # Target percentage gain
MAX_LOSS = -8  # Maximum allowed loss percentage
INDEX_POS = 3  # Position index (exchange specific)
PAUSE_TIME = 10  # Time between trades in minutes
VOL_REPEAT = 11  # Volume calculation parameter
VOL_TIME = 5  # Volume time calculation parameter
VOL_DECIMAL = 0.4  # Volume decimal parameter

# Initialize exchange
try:
    phemex = ccxt.phemex({
        'enableRateLimit': True, 
        'apiKey': xP_hmv_KEY, 
        'secret': xP_hmv_SECRET
    })
    logger.info(f"Exchange initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize exchange: {e}")
    sys.exit(1)

# Trading parameters
PARAMS = {'timeInForce': 'PostOnly'}

def get_market_data():
    """Get current market data including ask/bid and support/resistance levels"""
    try:
        # Get current ask and bid
        askbid = n.ask_bid(SYMBOL)
        ask = askbid[0]
        bid = askbid[1]
        logger.info(f'For {SYMBOL}... ask: {ask} | bid: {bid}')

        # Get SMA data with support/resistance
        df_sma = n.df_sma(SYMBOL, '15m', 289, 20)  # 289 15m periods = 3 days
        
        # Calculate support & resistance based on close
        curr_support = df_sma['close'].min()
        curr_resis = df_sma['close'].max()
        logger.info(f'Support: {curr_support} | Resistance: {curr_resis}')
        
        return {
            'ask': ask,
            'bid': bid,
            'df_sma': df_sma,
            'support': curr_support,
            'resistance': curr_resis
        }
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return None

def retest():
    """
    Identify breakout/breakdown opportunities
    
    Returns:
        tuple: (buy_break_out, sell_break_down, breakoutprice, breakdownprice)
    """
    try:
        logger.info('Calculating retest levels...')
        
        # Get market data
        market_data = get_market_data()
        if not market_data:
            logger.error("Failed to get market data for retest calculation")
            return False, False, False, False
            
        bid = market_data['bid']
        df_sma = market_data['df_sma']
        
        buy_break_out = False
        sell_break_down = False
        breakoutprice = False
        breakdownprice = False

        # Get resistance and support values and ensure they're floats
        resistance = float(df_sma['resis'].iloc[-1])
        support = float(df_sma['support'].iloc[-1])
        
        logger.info(f"Comparing bid {bid} to resistance {resistance} and support {support}")

        # Check for breakout
        if bid > resistance:
            logger.info(f'BREAKOUT DETECTED... Buy at previous resistance {market_data["resistance"]}')
            buy_break_out = True
            breakoutprice = int(resistance) * 1.001
            
        # Check for breakdown
        elif bid < support:
            logger.info(f'BREAKDOWN DETECTED... Sell at previous support {market_data["support"]}')
            sell_break_down = True
            breakdownprice = int(support) * 0.999

        return buy_break_out, sell_break_down, breakoutprice, breakdownprice
    except Exception as e:
        logger.error(f"Error in retest calculation: {str(e)}")
        return False, False, False, False

def bot():
    """Main trading bot function executed on schedule"""
    try:
        # Get PNL information
        pnl_close = n.pnl_close(SYMBOL)
        
        # Check if we need to pause after closing a position
        sleep_on_close = n.sleep_on_close(SYMBOL, PAUSE_TIME)
        
        # Get current market prices
        askbid = n.ask_bid(SYMBOL)
        ask = askbid[0]
        bid = askbid[1]
        
        # Calculate retest levels
        re_test = retest()
        break_out = re_test[0]
        break_down = re_test[1]
        breakoutprice = re_test[2]
        breakdownprice = re_test[3]
        logger.info(f'Breakout: {break_out} @ {breakoutprice} | Breakdown: {break_down} @ {breakdownprice}')
        
        # Get current position information
        open_pos = n.open_positions(SYMBOL)
        in_pos = open_pos[1]
        curr_size = open_pos[2]
        curr_size = int(curr_size) if curr_size else 0
        curr_p = bid
        
        logger.info(f'Symbol: {SYMBOL} | Breakout: {break_out} | Breakdown: {break_down} | In position: {in_pos} | Size: {curr_size} | Price: {curr_p}')
        
        # Execute trading strategy
        if (not in_pos) and (curr_size < POS_SIZE):
            # Cancel existing orders before placing new ones
            phemex.cancel_all_orders(SYMBOL)
            
            # Get fresh market prices
            askbid = n.ask_bid(SYMBOL)
            ask = askbid[0]
            bid = askbid[1]
            
            # Place buy order on breakout
            if break_out:
                logger.info(f'Placing BUY order: {SYMBOL} size {POS_SIZE} @ {breakoutprice}')
                phemex.create_limit_buy_order(SYMBOL, POS_SIZE, breakoutprice, PARAMS)
                logger.info('Order submitted - pausing for 2 minutes')
                time.sleep(120)
                
            # Place sell order on breakdown
            elif break_down:
                logger.info(f'Placing SELL order: {SYMBOL} size {POS_SIZE} @ {breakdownprice}')
                phemex.create_limit_sell_order(SYMBOL, POS_SIZE, breakdownprice, PARAMS)
                logger.info('Order submitted - pausing for 2 minutes')
                time.sleep(120)
                
            else:
                logger.info('No breakout/breakdown detected - waiting 1 minute')
                time.sleep(60)
        else:
            logger.info('Already in position - not placing new orders')
    except Exception as e:
        logger.error(f"Error in bot execution: {e}")

# Initialize the first data pull
try:
    market_data = get_market_data()
    logger.info("Initial market data loaded successfully")
except Exception as e:
    logger.error(f"Failed to load initial market data: {e}")

# Schedule the bot to run every 28 seconds
schedule.every(28).seconds.do(bot)
logger.info(f"Bot scheduled to run every 28 seconds for {SYMBOL}")

# Main loop
logger.info("Starting main loop")
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        logger.error("Possible internet connection issue - waiting 30 seconds before retry")
        time.sleep(30)