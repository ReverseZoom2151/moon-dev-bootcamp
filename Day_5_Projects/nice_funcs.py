import requests, json, time, logging, sys, os
from hyperliquid.info import Info
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange
from Day_4_Projects import dontshare as d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperliquid_risk.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load private key (consider using environment variables instead)
secret_key = d.private_key

def get_position(symbol, account):
    """Retrieve current position information for a symbol
    
    Args:
        symbol (str): Trading symbol to query
        account (LocalAccount): Ethereum account for authentication
        
    Returns:
        tuple: (positions, in_pos, size, pos_sym, entry_px, pnl_perc, long)
            positions: List of position data
            in_pos: Boolean indicating if in position
            size: Size of position
            pos_sym: Symbol of position
            entry_px: Entry price
            pnl_perc: Current PNL percentage
            long: Boolean indicating if long position
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        logger.info(f'Current account value: {user_state["marginSummary"]["accountValue"]}')
        
        positions = []
        logger.info(f'Looking for positions in symbol: {symbol}')
        
        for position in user_state['positions']:
            if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
                positions.append(position["position"])
                in_pos = True
                size = float(position["position"]['szi'])
                pos_sym = position["position"]['coin']
                entry_px = float(position["position"]["entryPx"])
                pnl_perc = float(position["position"]["returnOnEquity"]) * 100
                logger.info(f'Found position - PNL: {pnl_perc}%')
                break
        else:
            in_pos = False 
            size = 0
            pos_sym = None
            entry_px = 0
            pnl_perc = 0
            logger.info(f'No position found for {symbol}')
            
        if size > 0:
            long = True
        elif size < 0:
            long = False 
        else:
            long = None

        return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long
        
    except Exception as e:
        logger.error(f"Error getting position information: {e}")
        return [], False, 0, None, 0, 0, None

def ask_bid(symbol):
    """Get the current ask and bid prices for a symbol
    
    Args:
        symbol (str): Trading symbol to query
        
    Returns:
        tuple: (ask, bid, l2_data) - Current ask price, bid price, and l2 book data
    """
    try:
        url = 'https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}
        data = {
            'type': 'l2Book',
            'coin': symbol
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        l2_data = response.json()
        l2_data = l2_data['levels']
        bid = float(l2_data[0][0]['px'])
        ask = float(l2_data[1][0]['px'])
        
        logger.info(f'Price data for {symbol} - Ask: {ask}, Bid: {bid}')
        return ask, bid, l2_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error in ask_bid: {e}")
        return 0, 0, []
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Error parsing response data in ask_bid: {e}")
        return 0, 0, []

def get_sz_px_decimals(symbol):
    """Get the size and price decimal precision for a symbol
    
    Args:
        symbol (str): Trading symbol to query
        
    Returns:
        tuple: (sz_decimals, px_decimals) - Decimal precision for size and price
    """
    try:
        url='https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}
        data = {'type': 'meta'}

        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
        response.raise_for_status()
        
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
        else:
            logger.warning(f'Symbol {symbol} not found in universe')
            sz_decimals = 0
            
        # Get price precision from current market prices
        ask, _, _ = ask_bid(symbol)
        if ask == 0:
            logger.warning("Could not determine price decimals - using default of 0")
            px_decimals = 0
        else:
            ask_str = str(ask)
            if '.' in ask_str:
                px_decimals = len(ask_str.split('.')[1])
            else:
                px_decimals = 0
                
        logger.info(f'{symbol} precision - Size: {sz_decimals} decimals, Price: {px_decimals} decimals')
        return sz_decimals, px_decimals
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error in get_sz_px_decimals: {e}")
        return 0, 0
    except (KeyError, IndexError, ValueError, StopIteration) as e:
        logger.error(f"Error parsing response data in get_sz_px_decimals: {e}")
        return 0, 0

def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    """Place a limit order
    
    Args:
        coin (str): Trading symbol
        is_buy (bool): True for buy order, False for sell order
        sz (float): Size of order
        limit_px (float): Limit price
        reduce_only (bool): Whether order is reduce-only
        account (LocalAccount): Ethereum account for authentication
        
    Returns:
        dict: Order result data
    """
    try:
        exchange = Exchange(account, constants.MAINNET_API_URL)
        
        # Get size precision
        rounding = get_sz_px_decimals(coin)[0]
        sz = round(sz, rounding)
        
        logger.info(f"Order parameters:")
        logger.info(f"- Coin: {coin} ({type(coin)})")
        logger.info(f"- Direction: {'Buy' if is_buy else 'Sell'} ({type(is_buy)})")
        logger.info(f"- Size: {sz} ({type(sz)})")
        logger.info(f"- Price: {limit_px} ({type(limit_px)})")
        logger.info(f"- Reduce only: {reduce_only} ({type(reduce_only)})")
        
        logger.info(f'Placing limit order for {coin} {sz} @ {limit_px}')
        order_result = exchange.order(
            coin, 
            is_buy, 
            sz, 
            limit_px, 
            {"limit": {"tif": "Gtc"}}, 
            reduce_only=reduce_only
        )
        
        direction = "BUY" if is_buy else "SELL"
        logger.info(f"Limit {direction} order placed, status: {order_result['response']['data']['statuses'][0]}")
        return order_result
        
    except Exception as e:
        logger.error(f"Error placing limit order: {e}")
        # Return a structured response even in error case
        return {"success": False, "error": str(e)}

def cancel_all_orders(account):
    """Cancel all open orders
    
    Args:
        account (LocalAccount): Ethereum account for authentication
    """
    try:
        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        
        open_orders = info.open_orders(account.address)
        logger.info(f'Found {len(open_orders)} open orders to cancel')
        
        for open_order in open_orders:
            try:
                exchange.cancel(open_order['coin'], open_order['oid'])
                logger.info(f"Cancelled order for {open_order['coin']}, ID: {open_order['oid']}")
            except Exception as e:
                logger.error(f"Error cancelling order {open_order['oid']}: {e}")
                
    except Exception as e:
        logger.error(f"Error in cancel_all_orders: {e}")

def kill_switch(symbol, account):
    """Close all positions for a symbol
    
    Args:
        symbol (str): Trading symbol to close position
        account (LocalAccount): Ethereum account for authentication
    """
    logger.info(f"Starting kill switch for {symbol}")
    
    try:
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
        
        if not im_in_pos:
            logger.info(f"No open position found for {symbol}, nothing to close")
            return
            
        logger.info(f"Closing position - Size: {pos_size}, Side: {'Long' if long else 'Short'}")
        
        while im_in_pos:
            # Cancel existing orders before placing new ones
            cancel_all_orders(account)
            
            # Get current market prices
            ask, bid, l2_data = ask_bid(pos_sym)
            if ask == 0 or bid == 0:
                logger.warning("Failed to get valid market prices, retrying in 5 seconds")
                time.sleep(5)
                continue
                
            # Use absolute value of position size
            pos_size = abs(pos_size)
            
            # Place order to close position
            if long == True:
                limit_order(pos_sym, False, pos_size, ask, True, account)
                logger.info('Kill switch: Sell order to close long position submitted')
            elif long == False:
                limit_order(pos_sym, True, pos_size, bid, True, account)
                logger.info('Kill switch: Buy order to close short position submitted')
            else:
                logger.error("Could not determine position direction (long/short)")
                break
                
            # Wait for order to fill
            logger.info("Waiting 5 seconds for order to fill...")
            time.sleep(5)
            
            # Check if position is closed
            positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
            
        logger.info('Position successfully closed')
        
    except Exception as e:
        logger.error(f"Error in kill_switch: {e}")

def pnl_close(symbol, target, max_loss, account):
    """Check if a position should be closed based on profit/loss targets
    
    Args:
        symbol (str): Trading symbol to check
        target (float): Profit target percentage
        max_loss (float): Maximum loss percentage (negative value)
        account (LocalAccount): Ethereum account for authentication
    """
    logger.info(f'Checking PNL for {symbol} (Target: {target}%, Max Loss: {max_loss}%)')
    
    try:
        # Get current position details
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
        
        if not im_in_pos:
            logger.info(f"No position found for {symbol}")
            return
            
        logger.info(f"Current PNL: {pnl_perc}%")

        # Check profit target
        if pnl_perc > target:
            logger.info(f'Profit target reached: {pnl_perc}% > {target}%, closing position')
            kill_switch(pos_sym, account)
        # Check stop loss
        elif pnl_perc <= max_loss:
            logger.info(f'Stop loss triggered: {pnl_perc}% <= {max_loss}%, closing position')
            kill_switch(pos_sym, account)
        else: 
            logger.info(f'Position within PNL parameters (Current: {pnl_perc}%, Target: {target}%, Stop: {max_loss}%)')
            
        logger.info('PNL check completed')
        
    except Exception as e:
        logger.error(f"Error in pnl_close: {e}")

def acct_bal(account):
    """Get the current account balance
    
    Args:
        account (LocalAccount): Ethereum account for authentication
        
    Returns:
        str: Account value
    """
    try:
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(account.address)
        value = user_state["marginSummary"]["accountValue"]
        logger.info(f'Current account value: {value}')
        return value
        
    except Exception as e:
        logger.error(f"Error fetching account balance: {e}")
        return "0"  # Return zero as string to maintain expected return type