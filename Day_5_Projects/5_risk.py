import ccxt, time, sys, os, logging, json, argparse

# Add parent directory to sys.path to allow imports across directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import key as xP_KEY, secret as xP_SECRET

# Symbol index positions mapping for better maintainability
SYMBOL_INDEX_MAP = {
    'BTC/USD:BTC': 4,
    'APE/USD:BTC': 2,
    'ETH/USD:BTC': 3,
    'DOGE/USD:BTC': 1,
    'u100000SHIB/USD:BTC': 0
}

# Configuration constants
DEFAULT_SYMBOL = 'BTC/USD:BTC'
DEFAULT_SIZE = 1
DEFAULT_BID = 29000
DEFAULT_TARGET = 9
DEFAULT_MAX_LOSS = -8
MAX_RISK = 1000
RETRY_DELAY = 30  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phemex_risk.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
demo_mode = False
phemex = None

def init_exchange(use_demo_mode=False):
    """Initialize exchange connection with option for demo mode"""
    global demo_mode, phemex
    demo_mode = use_demo_mode
    
    if demo_mode:
        logger.info("⚠️ Running in DEMO MODE with simulated data")
        return None
    else:
        # Initialize Phemex client with required parameters
        return ccxt.phemex({
            'enableRateLimit': True,
            'apiKey': xP_KEY,
            'secret': xP_SECRET
        })

def test_api_connection():
    """Test the API connection and return True if successful, False otherwise"""
    global demo_mode, phemex
    
    if demo_mode:
        logger.info("✅ Demo mode active - No API connection needed")
        return True
        
    try:
        # Try a public endpoint that doesn't require authentication
        phemex.fetch_ticker(DEFAULT_SYMBOL)
        logger.info("Public API connection successful")
        
        # Now try a private endpoint
        try:
            bal = phemex.fetch_balance()
            logger.info("✅ Private API authentication successful")
            return True
        except Exception as e:
            error_message = str(e)
            if "401 Request IP mismatch" in error_message:
                logger.error("\n⚠️ ERROR: IP MISMATCH - Your current IP address is not authorized")
                logger.error("You need to update your Phemex API key settings to allow your current IP address.")
                logger.error("1. Login to Phemex")
                logger.error("2. Go to 'Account & Security' -> 'API Management'")
                logger.error("3. Edit your API key to add your current IP address\n")
                
                # Offer to switch to demo mode
                if input("\nWould you like to continue in demo mode? (y/n): ").lower() == 'y':
                    demo_mode = True
                    logger.info("✅ Switching to demo mode")
                    return True
            else:
                logger.error(f"❌ Private API authentication failed: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Public API connection failed: {e}")
        return False

def open_positions(symbol=DEFAULT_SYMBOL):
    """Get information about open positions for a given symbol"""
    
    if demo_mode:
        # Simulate position data
        fake_position = {
            'side': 'Buy',
            'size': '0.1',
            'entryPrice': '42000',
            'leverage': '5',
            'unrealizedPnl': '50'
        }
        
        index_pos = SYMBOL_INDEX_MAP.get(symbol, 0)
        positions = [fake_position]
        
        long = True
        openpos_bool = True
        openpos_size = fake_position['size']
        
        logger.info(f"[DEMO] Position status: | Open: {openpos_bool} | Size: {openpos_size} | Long: {long}")
        return positions, openpos_bool, openpos_size, long, index_pos
    
    try:
        # what is the position index for that symbol?
        index_pos = SYMBOL_INDEX_MAP.get(symbol)
        if index_pos is None:
            logger.warning(f"Unknown symbol '{symbol}'. Using default index.")
            index_pos = 0

        params = {'type': 'swap', 'code': 'USD'}
        phe_bal = phemex.fetch_balance(params=params)
        open_positions = phe_bal['info']['data']['positions']

        # Prevent index out of range errors
        if index_pos < 0 or index_pos >= len(open_positions):
            logger.warning(f"Index {index_pos} out of range. No positions found.")
            return open_positions, False, 0, None, index_pos

        # dictionaries 
        openpos_side = open_positions[index_pos]['side']
        openpos_size = open_positions[index_pos]['size']

        # if statements 
        if openpos_side == 'Buy':
            openpos_bool = True 
            long = True 
        elif openpos_side == 'Sell':
            openpos_bool = True
            long = False
        else:
            openpos_bool = False
            long = None 

        logger.info(f'Position status: | Open: {openpos_bool} | Size: {openpos_size} | Long: {long} | Index: {index_pos}')

        # returning
        return open_positions, openpos_bool, openpos_size, long, index_pos
    except Exception as e:
        logger.error(f"Error in open_positions: {e}")
        return [], False, 0, None, None
   
def ask_bid(symbol=DEFAULT_SYMBOL):
    """Get current ask and bid prices for a symbol"""
    
    if demo_mode:
        # Simulate bid/ask with slight randomization
        import random
        base_price = 42000 + random.randint(-500, 500)
        bid = base_price
        ask = base_price + 10
        logger.info(f"[DEMO] Current prices for {symbol} - Ask: {ask}, Bid: {bid}")
        return ask, bid
    
    try:
        ob = phemex.fetch_order_book(symbol)
        if not ob['bids'] or not ob['asks']:
            logger.warning(f"Empty order book for {symbol}")
            return 0, 0
            
        bid = ob['bids'][0][0]
        ask = ob['asks'][0][0]
        
        logger.info(f"Current prices for {symbol} - Ask: {ask}, Bid: {bid}")
        return ask, bid
    except Exception as e:
        logger.error(f"Error fetching order book: {e}")
        return 0, 0

def kill_switch(symbol=DEFAULT_SYMBOL):
    """Close all open positions for a symbol"""
    
    if demo_mode:
        logger.info(f"[DEMO] Closing position for {symbol}")
        time.sleep(2)  # Simulate processing time
        logger.info("[DEMO] Position closed successfully")
        return True
    
    logger.info(f'Starting the kill switch for {symbol}')
    position_info = open_positions(symbol)
    openposi = position_info[1]  # True or False
    long = position_info[3]      # True or False
    kill_size = position_info[2] # Size that's open  

    logger.info(f'Position status: Open: {openposi}, Long: {long}, Size: {kill_size}')
    
    attempt = 0
    max_attempts = 5

    while openposi and attempt < max_attempts:
        attempt += 1
        logger.info(f'Kill switch attempt {attempt}/{max_attempts}')
        
        try:
            # Cancel existing orders
            phemex.cancel_all_orders(symbol)
            
            # Get updated position status
            position_info = open_positions(symbol)
            openposi = position_info[1]
            long = position_info[3]
            kill_size = position_info[2]
            
            if not kill_size or float(kill_size) == 0:
                logger.info("Position already closed")
                return True
                
            kill_size = int(kill_size)
            
            # Get current prices
            ask, bid = ask_bid(symbol)
            
            if ask == 0 or bid == 0:
                logger.warning("Invalid prices. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # Place appropriate order to close position
            params = {'timeInForce': 'PostOnly'}
            if long == False:
                phemex.create_limit_buy_order(symbol, kill_size, bid, params)
                logger.info(f'Created BUY to CLOSE order: {kill_size} {symbol} at ${bid}')
            elif long == True:
                phemex.create_limit_sell_order(symbol, kill_size, ask, params)
                logger.info(f'Created SELL to CLOSE order: {kill_size} {symbol} at ${ask}')
            else:
                logger.error('Unknown position direction')
                return False
            
            logger.info('Waiting 30 seconds for order to fill...')
            time.sleep(30)
            
            # Check if position is closed
            openposi = open_positions(symbol)[1]
            
        except Exception as e:
            logger.error(f"Error during kill switch: {e}")
            time.sleep(5)
    
    if openposi and attempt >= max_attempts:
        logger.warning(f"Failed to close position after {max_attempts} attempts")
        return False
    
    logger.info("Position closed successfully")
    return True

def pnl_close(symbol=DEFAULT_SYMBOL, target=DEFAULT_TARGET, max_loss=DEFAULT_MAX_LOSS):
    """Check PNL and close position if target or stop loss is hit"""
    
    if demo_mode:
        # Simulate PNL check with random value
        import random
        
        # Randomly choose between profit, loss, or breakeven
        scenario = random.choice(['profit', 'loss', 'breakeven'])
        if scenario == 'profit':
            perc = target + random.uniform(-3, 3)
            logger.info(f"[DEMO] Position for {symbol} is in profit: {perc:.2f}%")
            
            if perc > target:
                logger.info(f"[DEMO] Profit target of {target}% reached at {perc:.2f}%")
                pnlclose = True
                kill_switch(symbol)
            else:
                logger.info(f"[DEMO] Current profit {perc:.2f}% has not reached target {target}%")
                pnlclose = False
                
            return pnlclose, True, 0.1, True
            
        elif scenario == 'loss':
            perc = max_loss + random.uniform(-3, 3)
            logger.info(f"[DEMO] Position for {symbol} is in loss: {perc:.2f}%")
            
            if perc <= max_loss:
                logger.info(f"[DEMO] Stop loss of {max_loss}% triggered at {perc:.2f}%")
                kill_switch(symbol)
            else:
                logger.info(f"[DEMO] Current loss {perc:.2f}% has not reached max loss {max_loss}%")
                
            return False, True, 0.1, True
            
        else:  # breakeven
            logger.info(f"[DEMO] Position for {symbol} is near breakeven")
            return False, True, 0.1, True
    
    try:
        logger.info(f'Checking PNL exit conditions for {symbol}')

        params = {'type': 'swap', 'code': 'USD'}
        pos_dict = phemex.fetch_positions(params=params)

        index_pos = open_positions(symbol)[4]
        if index_pos is None or index_pos >= len(pos_dict):
            logger.warning(f"No position data available for {symbol}")
            return False, False, 0, None
            
        position = pos_dict[index_pos]
        side = position['side']
        size = position['contracts']
        
        try:
            entry_price = float(position['entryPrice'])
            leverage = float(position['leverage'])
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing position data: {e}")
            return False, False, 0, None
            
        current_price = ask_bid(symbol)[1]
        
        logger.info(f'Position: {side} | Entry: {entry_price} | Leverage: {leverage}')
        
        # Determine if position is long or short
        if side == 'long':
            diff = current_price - entry_price
            long = True
        else: 
            diff = entry_price - current_price
            long = False

        # Calculate PNL percentage
        try: 
            perc = round(((diff/entry_price) * leverage), 10) * 100
        except ZeroDivisionError:
            logger.warning("Division by zero when calculating PNL")
            perc = 0
            
        logger.info(f'PNL for {symbol}: {perc}%')

        pnlclose = False 
        in_pos = True
        
        # Check if we should close based on profit target or stop loss
        if perc > 0:
            logger.info(f'Position for {symbol} is in profit')
            if perc > target:
                logger.info(f'Profit target of {target}% reached at {perc}%, closing position')
                pnlclose = True
                kill_switch(symbol)
            else:
                logger.info(f'Current profit {perc}% has not reached target {target}%')
        elif perc < 0: 
            logger.info(f'Position for {symbol} is in loss: {perc}%')
            if perc <= max_loss: 
                logger.info(f'Stop loss of {max_loss}% triggered at {perc}%, closing position')
                kill_switch(symbol)
            else:
                logger.info(f'Current loss {perc}% has not reached max loss {max_loss}%')
        else:
            logger.info('No open position or zero PNL')
            in_pos = False

        logger.info(f'PNL check completed for {symbol}')
        return pnlclose, in_pos, size, long
    except Exception as e:
        logger.error(f"Error in PNL close: {e}")
        return False, False, 0, None

def size_kill():
    """Check if position size exceeds maximum risk and close if needed"""
    
    if demo_mode:
        logger.info("[DEMO] Checking position sizes against max risk")
        logger.info("[DEMO] No positions exceed risk limit")
        return
    
    try:
        max_risk = MAX_RISK
        logger.info(f"Checking position sizes against max risk of {max_risk}")

        params = {'type': 'swap', 'code': 'USD'}
        all_phe_balance = phemex.fetch_balance(params=params)
        open_positions = all_phe_balance['info']['data']['positions']
        
        if not open_positions:
            logger.info("No open positions found")
            return
        
        # Check all positions, not just the first one
        for index, position in enumerate(open_positions):
            try:
                pos_cost = float(position.get('posCost', 0))
                openpos_side = position.get('side', 'None')
                openpos_size = position.get('size', 0)
                symbol_info = position.get('symbol', 'Unknown')
                
                logger.info(f'Position {index}: {symbol_info} | Cost: {pos_cost} | Side: {openpos_side} | Size: {openpos_size}')
                
                if pos_cost > max_risk:
                    logger.warning(f'EMERGENCY: Position size {pos_cost} exceeds max risk {max_risk}')
                    kill_switch(symbol_info)
                    logger.info('Emergency exit completed')
            except Exception as e:
                logger.error(f"Error processing position {index}: {e}")
    except Exception as e:
        logger.error(f"Error in size kill: {e}")

def start_monitoring(check_interval=60, symbol=DEFAULT_SYMBOL, target=DEFAULT_TARGET, max_loss=DEFAULT_MAX_LOSS):
    """Start continuous monitoring of positions"""
    
    logger.info(f"Starting monitoring for {symbol} with check interval {check_interval}s")
    logger.info(f"Target: {target}% | Stop Loss: {max_loss}%")
    
    try:
        while True:
            # Check position size against maximum risk
            size_kill()
            
            # Check for profit/loss targets
            pnl_close(symbol, target, max_loss)
            
            logger.info(f"Sleeping for {check_interval} seconds...")
            time.sleep(check_interval)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")

def save_settings(settings):
    """Save user settings to a JSON file"""
    try:
        with open('phemex_risk_settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
        logger.info("Settings saved successfully")
    except Exception as e:
        logger.error(f"Error saving settings: {e}")

def load_settings():
    """Load user settings from a JSON file"""
    try:
        if os.path.exists('phemex_risk_settings.json'):
            with open('phemex_risk_settings.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return {}

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Phemex Risk Management Tool')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with simulated data')
    args = parser.parse_args()
    
    print("\n=== Phemex Risk Management Tool ===\n")
    
    # Initialize exchange - either real or demo mode
    phemex = init_exchange(args.demo)
    
    # Test API connection before proceeding
    if not test_api_connection():
        logger.error("\nExiting program due to API connection issues.")
        sys.exit(1)
    
    # If we reach here, API connection was successful or we're in demo mode
    try:
        # Load saved settings
        settings = load_settings()
        
        if demo_mode:
            logger.info("\n⚠️ DEMO MODE ACTIVE - Using simulated data\n")
            
        logger.info("\nFetching balance information...")
        if not demo_mode:
            bal = phemex.fetch_balance()
        logger.info("Balance information retrieved successfully")
        
        # Get position information
        logger.info("\nChecking for open positions...")
        position_data = open_positions()
        
        if position_data[1]:  # If open position exists
            logger.info(f"\nFound open position for {DEFAULT_SYMBOL}")
            logger.info(f"Side: {'Long' if position_data[3] else 'Short'}")
            logger.info(f"Size: {position_data[2]}")
            
            # Ask if user wants to close the position or start monitoring
            user_input = input("\nDo you want to (c)lose this position, (m)onitor it, or (q)uit? [c/m/q]: ")
            if user_input.lower() == 'c':
                logger.info("\nInitiating kill switch to close position...")
                kill_switch()
            elif user_input.lower() == 'm':
                interval = settings.get('check_interval', 60)
                target_val = settings.get('target', DEFAULT_TARGET)
                max_loss_val = settings.get('max_loss', DEFAULT_MAX_LOSS)
                
                check_interval = int(input(f"Enter check interval in seconds [{interval}]: ") or interval)
                target = float(input(f"Enter take profit percentage [{target_val}]: ") or target_val)
                max_loss = float(input(f"Enter stop loss percentage (negative value) [{max_loss_val}]: ") or max_loss_val)
                
                # Save settings for next time
                save_settings({
                    'check_interval': check_interval,
                    'target': target,
                    'max_loss': max_loss
                })
                
                logger.info("\nStarting position monitoring...")
                start_monitoring(check_interval, DEFAULT_SYMBOL, target, max_loss)
            else:
                logger.info("\nKeeping position open. Exiting program.")
        else:
            logger.info("\nNo open positions found for the specified symbol.")
            
    except Exception as e:
        logger.error(f"\nAn error occurred: {e}")
        logger.error("Please check your API key settings and try again.")
