import argparse, random, sys, os, logging, ccxt, time, schedule, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from Day_4_Projects.key_file import key as xP_KEY, secret as xP_SECRET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration (moved from global variables)
CONFIG = {
    'default_symbol': 'BTC/USDT',  # Changed to a symbol that exists on Phemex
    'default_size': 1,
    'timeframe': '15m',
    'limit': 100,
    'sma_period': 20,
    'target_profit_percentage': 9,
    'max_loss_percentage': -8,
    'max_risk_amount': 1000,
    'order_params': {'timeInForce': 'PostOnly'},
    'sleep_time': 30,  # seconds
    'emergency_timeout': 300  # seconds (reduced from 30000)
}

# Symbol mapping to avoid hardcoded indexes
SYMBOL_INDEX_MAP = {
    'BTC/USDT': 0,
    'ETH/USDT': 1,
    'SOL/USDT': 2,
    'XRP/USDT': 3,
    'DOGE/USDT': 4
}

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
        try:
            exchange = ccxt.phemex({
                'enableRateLimit': True,
                'apiKey': xP_KEY,
                'secret': xP_SECRET,
            })
            logger.info("Exchange initialized successfully")
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise

def test_api_connection():
    """Test the API connection and return True if successful, False otherwise"""
    global demo_mode, phemex
    
    if demo_mode:
        logger.info("✅ Demo mode active - No API connection needed")
        return True
        
    try:
        # Try a public endpoint that doesn't require authentication
        phemex.fetch_ticker(CONFIG['default_symbol'])
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

def get_symbol_index(symbol):
    """Get the index position for a symbol, with error handling."""
    if symbol in SYMBOL_INDEX_MAP:
        return SYMBOL_INDEX_MAP[symbol]
    logger.warning(f"Unknown symbol: {symbol}, position index not found")
    return None

def get_market_data(symbol=CONFIG['default_symbol']):
    """Get current market bid and ask prices."""
    if demo_mode:
        # Simulate market data
        base_price = 83000 + random.randint(-500, 500)
        bid = base_price
        ask = base_price + 10
        logger.info(f"[DEMO] Market data for {symbol}: ask={ask}, bid={bid}")
        return ask, bid
        
    try:
        ob = phemex.fetch_order_book(symbol)
        bid = ob['bids'][0][0] if ob['bids'] else None
        ask = ob['asks'][0][0] if ob['asks'] else None
        logger.debug(f"Market data for {symbol}: ask={ask}, bid={bid}")
        return ask, bid
    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching order book for {symbol}: {e}")
        return None, None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error fetching order book for {symbol}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error fetching order book for {symbol}: {e}")
        return None, None

def get_open_positions(symbol=CONFIG['default_symbol']):
    """Get information about open positions."""
    if demo_mode:
        # Simulate position data
        if random.random() > 0.7:  # 30% chance of having a position
            pos_type = random.choice(['Buy', 'Sell'])
            positions = [{
                'side': pos_type,
                'size': '0.1',
                'entryPrice': '83000',
                'leverage': '5',
                'unrealizedPnl': '50'
            }]
            
            is_open = True
            is_long = (pos_type == 'Buy')
            size = '0.1'
            index_pos = SYMBOL_INDEX_MAP.get(symbol, 0)
            
            logger.info(f"[DEMO] Position status for {symbol}: open={is_open}, size={size}, long={is_long}")
            return positions, is_open, size, is_long, index_pos
        else:
            logger.info(f"[DEMO] No position for {symbol}")
            return [], False, 0, None, SYMBOL_INDEX_MAP.get(symbol, 0)
    
    index_pos = get_symbol_index(symbol)
    
    if index_pos is None:
        logger.error(f"Cannot check positions for unknown symbol: {symbol}")
        return None, False, 0, None, None
    
    try:
        params = {'type': 'swap', 'code': 'USD'}
        phe_bal = phemex.fetch_balance(params=params)
        positions = phe_bal['info']['data']['positions']
        
        # Check if index is valid
        if index_pos >= len(positions):
            logger.error(f"Index {index_pos} out of range for positions: {len(positions)}")
            return positions, False, 0, None, index_pos
        
        position = positions[index_pos]
        side = position['side']
        size = position['size']
        
        if side == 'Buy':
            is_open = True
            is_long = True
        elif side == 'Sell':
            is_open = True
            is_long = False
        else:
            is_open = False
            is_long = None
        
        logger.info(f"Position status for {symbol}: open={is_open}, size={size}, long={is_long}")
        return positions, is_open, size, is_long, index_pos
        
    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching positions: {e}")
        return None, False, 0, None, index_pos
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error fetching positions: {e}")
        return None, False, 0, None, index_pos
    except Exception as e:
        logger.error(f"Unexpected error fetching positions: {e}")
        return None, False, 0, None, index_pos

def close_position(symbol=CONFIG['default_symbol']):
    """Close an open position (kill switch)."""
    if demo_mode:
        logger.info(f"[DEMO] Closing position for {symbol}")
        time.sleep(2)  # Simulate processing time
        logger.info("[DEMO] Position closed successfully")
        return True
        
    logger.info(f"Attempting to close position for {symbol}")
    
    try:
        positions, is_open, size, is_long, _ = get_open_positions(symbol)
        
        if not is_open:
            logger.info(f"No open position found for {symbol}")
            return True
            
        logger.info(f"Position details: is_open={is_open}, is_long={is_long}, size={size}")
        
        retry_count = 0
        max_retries = 5
        
        while is_open and retry_count < max_retries:
            # Cancel existing orders first
            try:
                phemex.cancel_all_orders(symbol)
                logger.info(f"Cancelled all existing orders for {symbol}")
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")
            
            # Get fresh position data and market prices
            _, is_open, size, is_long, _ = get_open_positions(symbol)
            if not is_open:
                logger.info("Position closed while cancelling orders")
                return True
                
            ask, bid = get_market_data(symbol)
            if ask is None or bid is None:
                logger.error("Failed to get market data, retrying...")
                time.sleep(5)
                retry_count += 1
                continue
            
            # Convert size to integer (can be a string from the API)
            try:
                size = int(size)
            except ValueError:
                logger.error(f"Invalid position size: {size}")
                size = 0
            
            if size <= 0:
                logger.info("No position size to close")
                return True
                
            # Place appropriate order to close the position
            try:
                if is_long:
                    logger.info(f"Placing SELL order to close LONG position: {size} at {ask}")
                    phemex.create_limit_sell_order(symbol, size, ask, CONFIG['order_params'])
                else:
                    logger.info(f"Placing BUY order to close SHORT position: {size} at {bid}")
                    phemex.create_limit_buy_order(symbol, size, bid, CONFIG['order_params'])
                
                logger.info(f"Order placed, waiting {CONFIG['sleep_time']} seconds to check fill...")
                time.sleep(CONFIG['sleep_time'])
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                time.sleep(5)
            
            # Check if position is closed
            _, is_open, _, _, _ = get_open_positions(symbol)
            retry_count += 1
            
        if is_open:
            logger.warning(f"Failed to close position after {max_retries} attempts")
            return False
        else:
            logger.info("Position successfully closed")
            return True
            
    except Exception as e:
        logger.error(f"Unexpected error in close_position: {e}")
        return False

def check_pnl_and_manage_position(symbol=CONFIG['default_symbol']):
    """Check profit/loss and manage position accordingly."""
    if demo_mode:
        # Simulate PNL check with random value
        if random.random() > 0.7:  # 30% chance of having a position
            # Randomly choose between profit, loss, or breakeven
            scenario = random.choice(['profit', 'loss', 'breakeven'])
            if scenario == 'profit':
                perc = CONFIG['target_profit_percentage'] + random.uniform(-3, 3)
                logger.info(f"[DEMO] Position for {symbol} is in profit: {perc:.2f}%")
                
                if perc > CONFIG['target_profit_percentage']:
                    logger.info(f"[DEMO] Profit target of {CONFIG['target_profit_percentage']}% reached at {perc:.2f}%")
                    close_position(symbol)
                else:
                    logger.info(f"[DEMO] Current profit {perc:.2f}% has not reached target {CONFIG['target_profit_percentage']}%")
                
                return True, True, 0.1, True
                
            elif scenario == 'loss':
                perc = CONFIG['max_loss_percentage'] + random.uniform(-3, 3)
                logger.info(f"[DEMO] Position for {symbol} is in loss: {perc:.2f}%")
                
                if perc <= CONFIG['max_loss_percentage']:
                    logger.info(f"[DEMO] Stop loss of {CONFIG['max_loss_percentage']}% triggered at {perc:.2f}%")
                    close_position(symbol)
                else:
                    logger.info(f"[DEMO] Current loss {perc:.2f}% has not reached max loss {CONFIG['max_loss_percentage']}%")
                    
                return False, True, 0.1, True
                
            else:  # breakeven
                logger.info(f"[DEMO] Position for {symbol} is near breakeven")
                return False, True, 0.1, True
        else:
            logger.info(f"[DEMO] No position for {symbol}")
            return False, False, 0, None
    
    logger.info(f"Checking P&L for {symbol}")
    
    try:
        # Get position details
        index_pos = get_symbol_index(symbol)
        if index_pos is None:
            logger.error(f"Unknown symbol index for {symbol}")
            return False, False, 0, None
            
        # Get position data
        params = {'type': 'swap', 'code': 'USD'}
        positions = phemex.fetch_positions(params=params)
        
        # Check if index is valid
        if index_pos >= len(positions):
            logger.error(f"Position index {index_pos} out of range")
            return False, False, 0, None
            
        position = positions[index_pos]
        side = position['side'].lower()  # Normalize to lowercase
        size = position['contracts']
        
        # Check if we have a position
        if not size or float(size) == 0:
            logger.info(f"No active position for {symbol}")
            return False, False, 0, None
            
        # Get entry price and current price
        entry_price = float(position['entryPrice'])
        leverage = float(position['leverage'])
        _, current_price = get_market_data(symbol)
        
        if current_price is None:
            logger.error("Failed to get current price")
            return False, False, size, (side == 'long')
            
        logger.info(f"Position details: side={side}, entry_price={entry_price}, leverage={leverage}, current={current_price}")
        
        # Calculate P&L
        is_long = (side == 'long')
        if is_long:
            diff = current_price - entry_price
        else:
            diff = entry_price - current_price
            
        try:
            percentage = round(((diff / entry_price) * leverage * 100), 2)
        except ZeroDivisionError:
            logger.error("Entry price is zero, cannot calculate percentage")
            percentage = 0
            
        logger.info(f"P&L percentage for {symbol}: {percentage}%")
        
        # Check if we need to close position
        if percentage > 0:
            if percentage >= CONFIG['target_profit_percentage']:
                logger.info(f"Target profit reached ({percentage}%), closing position")
                close_position(symbol)
                return True, True, size, is_long
            else:
                logger.info(f"In profit ({percentage}%), but below target ({CONFIG['target_profit_percentage']}%)")
        elif percentage < 0:
            if percentage <= CONFIG['max_loss_percentage']:
                logger.info(f"Max loss reached ({percentage}%), closing position")
                close_position(symbol)
                return True, True, size, is_long
            else:
                logger.info(f"In loss ({percentage}%), but above max loss ({CONFIG['max_loss_percentage']}%)")
        
        return False, True, size, is_long
        
    except ccxt.NetworkError as e:
        logger.error(f"Network error checking P&L: {e}")
        return False, False, 0, None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error checking P&L: {e}")
        return False, False, 0, None
    except Exception as e:
        logger.error(f"Unexpected error checking P&L: {e}")
        return False, False, 0, None

def check_risk_exposure():
    """Check if current position size exceeds maximum risk."""
    if demo_mode:
        logger.info("[DEMO] Checking risk exposure")
        logger.info("[DEMO] Risk check passed: $500 < $1000")
        return True
        
    logger.info("Checking risk exposure")
    
    try:
        params = {'type': 'swap', 'code': 'USD'}
        balance = phemex.fetch_balance(params=params)
        positions = balance['info']['data']['positions']
        
        total_exposure = 0
        
        for position in positions:
            try:
                pos_cost = float(position.get('posCost', 0))
                total_exposure += pos_cost
            except (ValueError, TypeError):
                logger.warning(f"Invalid position cost: {position.get('posCost')}")
                
        logger.info(f"Total position exposure: {total_exposure}")
        
        if total_exposure > CONFIG['max_risk_amount']:
            logger.critical(f"EMERGENCY: Position size ({total_exposure}) exceeds max risk ({CONFIG['max_risk_amount']})")
            # Close all positions
            for symbol in SYMBOL_INDEX_MAP.keys():
                close_position(symbol)
            return False
        else:
            logger.info(f"Risk check passed: {total_exposure} < {CONFIG['max_risk_amount']}")
            return True
            
    except Exception as e:
        logger.error(f"Error checking risk exposure: {e}")
        return False

def calculate_sma(symbol=CONFIG['default_symbol'], timeframe=CONFIG['timeframe'], 
                 limit=CONFIG['limit'], sma_period=CONFIG['sma_period']):
    """Calculate SMA and generate trading signals."""
    if demo_mode:
        # Simulate SMA calculation
        base_price = 83000 + random.randint(-500, 500)
        current_price = base_price
        
        # Create a fake dataframe with SMA
        dates = [datetime.now().timestamp() * 1000 - (i * 60000) for i in range(limit)]
        closes = [base_price + random.randint(-200, 200) for _ in range(limit)]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': closes,
            'high': [x + random.randint(10, 50) for x in closes],
            'low': [x - random.randint(10, 50) for x in closes],
            'close': closes,
            'volume': [random.randint(1, 100) for _ in range(limit)]
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate SMA
        sma_col = f'sma{sma_period}_{timeframe}'
        df[sma_col] = df['close'].rolling(sma_period).mean()
        
        # Generate signal
        last_sma = df[sma_col].iloc[-1] if not df[sma_col].isna().all() else None
        
        if last_sma:
            if last_sma > current_price:
                df['signal'] = 'SELL'
            else:
                df['signal'] = 'BUY'
                
            logger.info(f"[DEMO] SMA calculation complete. Last SMA: {last_sma}, Current price: {current_price}")
            logger.info(f"[DEMO] Signal: {df['signal'].iloc[-1] if 'signal' in df else 'None'}")
        
        return df
        
    logger.info(f"Calculating SMA{sma_period} for {symbol} on {timeframe} timeframe")
    
    try:
        # Get historical data
        bars = phemex.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not bars or len(bars) < sma_period:
            logger.error(f"Not enough data points for SMA calculation, got {len(bars)}")
            return None
            
        # Create dataframe
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate SMA
        sma_col = f'sma{sma_period}_{timeframe}'
        df[sma_col] = df['close'].rolling(sma_period).mean()
        
        # Get current market price
        _, current_price = get_market_data(symbol)
        
        if current_price is None:
            logger.error("Failed to get current price for signal generation")
            return df
            
        # Generate signals
        if not df[sma_col].isna().all():
            last_sma = df[sma_col].iloc[-1]
            
            # Generate signals based on price vs SMA
            df.loc[df[sma_col] > current_price, 'signal'] = 'SELL'
            df.loc[df[sma_col] < current_price, 'signal'] = 'BUY'
            
            # Calculate support and resistance
            df['support'] = df['low'].rolling(sma_period).min()
            df['resistance'] = df['high'].rolling(sma_period).max()
            
            logger.info(f"SMA calculation complete. Last SMA: {last_sma}, Current price: {current_price}")
            logger.info(f"Signal: {df['signal'].iloc[-1] if 'signal' in df and len(df) > 0 else 'None'}")
            
        return df
        
    except ccxt.NetworkError as e:
        logger.error(f"Network error calculating SMA: {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error calculating SMA: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calculating SMA: {e}")
        return None

def execute_trade(symbol=CONFIG['default_symbol'], size=CONFIG['default_size']):
    """Execute a trade based on SMA signal."""
    if demo_mode:
        logger.info(f"[DEMO] Checking for trade opportunities on {symbol}")
        signal = random.choice(['BUY', 'SELL', None])
        
        if signal:
            logger.info(f"[DEMO] {signal} signal for {symbol} at 83000")
            logger.info(f"[DEMO] {signal} order placed successfully")
        else:
            logger.info(f"[DEMO] No clear signal for {symbol}")
        return
        
    logger.info(f"Checking for trade opportunities on {symbol}")
    
    try:
        # Check if we already have a position
        _, has_position, _, is_long, _ = get_open_positions(symbol)
        
        if has_position:
            logger.info(f"Already have a position for {symbol}, checking P&L")
            check_pnl_and_manage_position(symbol)
            return
            
        # Calculate SMA and get signal
        df = calculate_sma(symbol)
        
        if df is None or 'signal' not in df or len(df) == 0:
            logger.warning("No valid signal available")
            return
            
        signal = df['signal'].iloc[-1]
        _, current_price = get_market_data(symbol)
        
        if current_price is None:
            logger.error("Failed to get current price for trade execution")
            return
            
        # Execute trade based on signal
        if signal == 'BUY':
            logger.info(f"BUY signal for {symbol} at {current_price}")
            try:
                order = phemex.create_limit_buy_order(symbol, size, current_price, CONFIG['order_params'])
                logger.info(f"BUY order placed: {order}")
            except Exception as e:
                logger.error(f"Error placing BUY order: {e}")
        elif signal == 'SELL':
            logger.info(f"SELL signal for {symbol} at {current_price}")
            try:
                order = phemex.create_limit_sell_order(symbol, size, current_price, CONFIG['order_params'])
                logger.info(f"SELL order placed: {order}")
            except Exception as e:
                logger.error(f"Error placing SELL order: {e}")
        else:
            logger.info(f"No clear signal for {symbol}")
            
    except Exception as e:
        logger.error(f"Unexpected error in execute_trade: {e}")

def run_trading_cycle(symbol=CONFIG['default_symbol']):
    """Run a complete trading cycle."""
    logger.info(f"Starting trading cycle for {symbol}")
    
    try:
        # Check risk exposure first
        if not check_risk_exposure():
            logger.warning("Risk check failed, skipping trading cycle")
            return
            
        # Check if we have an open position
        _, has_position, _, _, _ = get_open_positions(symbol)
        
        if has_position:
            # Manage existing position
            logger.info(f"Managing existing position for {symbol}")
            check_pnl_and_manage_position(symbol)
        else:
            # Look for new trade opportunities
            logger.info(f"Looking for new trade opportunities for {symbol}")
            execute_trade(symbol)
            
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")

def setup_schedule():
    """Set up the schedule for running trading functions."""
    # Run risk check every 5 minutes instead of every hour
    schedule.every(5).minutes.do(check_risk_exposure)
    
    # Run trading cycle every minute instead of every 15 minutes
    for symbol in SYMBOL_INDEX_MAP.keys():
        schedule.every(1).minutes.do(run_trading_cycle, symbol=symbol)
        
    logger.info("Trading schedule initialized with faster testing intervals")

def main():
    """Main function to start the trading bot."""
    global phemex, demo_mode
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SMA Trading Bot')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with simulated data')
    args = parser.parse_args()
    
    print("\n=== SMA Trading Bot ===\n")
    
    # Initialize exchange - either real or demo mode
    phemex = init_exchange(args.demo)
    
    # Test API connection before proceeding
    if not test_api_connection():
        logger.error("\nExiting program due to API connection issues.")
        if input("\nWould you like to restart in demo mode? (y/n): ").lower() == 'y':
            demo_mode = True
            logger.info("✅ Switching to demo mode")
        else:
            sys.exit(1)
    
    logger.info("Starting SMA trading bot")
    
    try:
        # Initial risk check
        check_risk_exposure()
        
        # Setup schedule
        setup_schedule()
        
        # Run initial analysis
        for symbol in SYMBOL_INDEX_MAP.keys():
            df = calculate_sma(symbol)
            if df is not None:
                logger.info(f"Initial SMA calculated for {symbol}")
        
        # Main loop
        logger.info("Entering main loop")
        heartbeat_counter = 0
        while True:
            schedule.run_pending()
            
            heartbeat_counter += 1
            if heartbeat_counter >= 60:  # Show heartbeat every minute
                logger.info("Bot heartbeat - waiting for scheduled tasks")
                heartbeat_counter = 0
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Critical error in main loop: {e}")
        
if __name__ == "__main__":
    main()
