import sys, os, ccxt, logging, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import binance_key, binance_secret

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('binance_nice_funcs.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Initialize Binance
binance = ccxt.binance({'enableRateLimit': True, 'apiKey': binance_key, 'secret': binance_secret, 'options': {'defaultType': 'future'}})

def get_position(symbol):
    try:
        positions = binance.fetch_positions([symbol])
        if not positions or positions[0]['contracts'] == 0:
            logger.info(f'No position for {symbol}')
            return [], False, 0, symbol, 0, 0, None
        pos = positions[0]
        in_pos = True
        size = pos['contracts']
        pos_sym = symbol
        entry_px = pos['entryPrice']
        unrealized_pnl = pos['unrealizedPnl']
        entry_value = entry_px * size
        pnl_perc = (unrealized_pnl / entry_value) * 100 if entry_value != 0 else 0
        long = pos['side'] == 'long'
        logger.info(f'Position PNL: {pnl_perc}%')
        return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long
    except Exception as e:
        logger.error(f'Error getting position: {e}')
        return [], False, 0, symbol, 0, 0, None

def ask_bid(symbol):
    try:
        ob = binance.fetch_order_book(symbol)
        bid = ob['bids'][0][0]
        ask = ob['asks'][0][0]
        logger.info(f'Prices for {symbol} - Ask: {ask}, Bid: {bid}')
        return ask, bid, ob
    except Exception as e:
        logger.error(f'Error getting prices: {e}')
        return 0, 0, []

def get_sz_px_decimals(symbol):
    try:
        market = binance.market(symbol)
        sz_decimals = market['precision']['amount']
        px_decimals = market['precision']['price']
        logger.info(f'{symbol} precision - Size: {sz_decimals}, Price: {px_decimals}')
        return sz_decimals, px_decimals
    except Exception as e:
        logger.error(f'Error getting precision: {e}')
        return 0, 0

def limit_order(coin, is_buy, sz, limit_px, reduce_only):
    try:
        side = 'buy' if is_buy else 'sell'
        params = {'reduceOnly': reduce_only}
        order = binance.create_limit_order(coin, side, sz, limit_px, params)
        logger.info(f'Limit {side.upper()} order placed for {coin}')
        return order
    except Exception as e:
        logger.error(f'Error placing order: {e}')
        return {}

def cancel_all_orders(symbol):
    try:
        binance.cancel_all_orders(symbol)
        logger.info(f'All orders cancelled for {symbol}')
    except Exception as e:
        logger.error(f'Error cancelling orders: {e}')

def kill_switch(symbol):
    try:
        while True:
            cancel_all_orders(symbol)
            ask, bid, _ = ask_bid(symbol)
            if ask == 0 or bid == 0:
                time.sleep(5)
                continue
            pos = get_position(symbol)
            if not pos[1]:
                break
            sz = abs(pos[2])
            side = not pos[6]  # Buy if short, sell if long
            px = bid if side else ask
            limit_order(symbol, side, sz, px, True)
            time.sleep(5)
            pos = get_position(symbol)
    except Exception as e:
        logger.error(f'Kill switch error: {e}')

def pnl_close(symbol, target, max_loss):
    pos = get_position(symbol)
    if not pos[1]: return
    pnl_perc = pos[5]
    if pnl_perc >= target or pnl_perc <= max_loss:
        kill_switch(symbol)

def acct_bal():
    try:
        bal = binance.fetch_balance()
        value = float(bal['info']['availableBalance'])
        logger.info(f'Account value: {value}')
        return value
    except Exception as e:
        logger.error(f'Error getting balance: {e}')
        return '0' 