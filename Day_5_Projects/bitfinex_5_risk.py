import ccxt, time, sys, os, logging, argparse, json

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import bitfinex_key, bitfinex_secret

# Configuration constants
DEFAULT_SYMBOL = 'BTCF0:USTF0'
DEFAULT_SIZE = 0.0001
DEFAULT_BID = 29000
DEFAULT_TARGET = 9
DEFAULT_MAX_LOSS = -8
MAX_RISK = 1000
RETRY_DELAY = 30

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('bitfinex_risk.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

demo_mode = False
bitfinex = None

def init_exchange(use_demo_mode=False):
    global demo_mode, bitfinex
    demo_mode = use_demo_mode
    if demo_mode:
        logger.info('⚠️ Running in DEMO MODE')
        return None
    else:
        return ccxt.bitfinex({'enableRateLimit': True, 'apiKey': bitfinex_key, 'secret': bitfinex_secret})

def test_api_connection():
    if demo_mode:
        return True
    try:
        bitfinex.fetch_ticker(DEFAULT_SYMBOL)
        bal = bitfinex.fetch_balance({'type': 'derivatives'})
        return True
    except Exception as e:
        logger.error(f'API connection failed: {e}')
        return False

def open_positions(symbol=DEFAULT_SYMBOL):
    if demo_mode:
        return [{'side': 'long', 'amount': 0.0001, 'entry_price': 42000, 'leverage': 5}], True, 0.0001, True
    try:
        positions = bitfinex.fetch_positions([symbol])
        if not positions:
            return [], False, 0, None
        pos = positions[0]
        side = pos['side']
        size = pos['amount']
        long = side == 'long'
        return positions, True, size, long
    except Exception as e:
        logger.error(f'Error fetching positions: {e}')
        return [], False, 0, None

def ask_bid(symbol=DEFAULT_SYMBOL):
    if demo_mode:
        return 42010, 42000
    try:
        ob = bitfinex.fetch_order_book(symbol)
        return ob['asks'][0][0], ob['bids'][0][0]
    except Exception as e:
        logger.error(f'Error fetching order book: {e}')
        return 0, 0

def kill_switch(symbol=DEFAULT_SYMBOL):
    if demo_mode:
        logger.info('[DEMO] Position closed')
        return True
    try:
        bitfinex.cancel_all_orders(symbol)
        pos = open_positions(symbol)
        if pos[1]:
            side = 'sell' if pos[3] else 'buy'
            bitfinex.create_market_order(symbol, side, pos[2])
            logger.info('Position closed')
            return True
        return False
    except Exception as e:
        logger.error(f'Kill switch error: {e}')
        return False

def pnl_close(symbol=DEFAULT_SYMBOL, target=DEFAULT_TARGET, max_loss=DEFAULT_MAX_LOSS):
    if demo_mode:
        return True, True, 0.0001, True
    try:
        pos = open_positions(symbol)
        if not pos[1]: return False, False, 0, None
        entry = pos[0][0]['entry_price']
        leverage = pos[0][0]['leverage']
        current = ask_bid(symbol)[1] if pos[3] else ask_bid(symbol)[0]
        diff = (current - entry) if pos[3] else (entry - current)
        perc = (diff / entry * leverage) * 100
        if perc > target or perc <= max_loss:
            kill_switch(symbol)
            return True, True, pos[2], pos[3]
        return False, True, pos[2], pos[3]
    except Exception as e:
        logger.error(f'PNL error: {e}')
        return False, False, 0, None

def size_kill():
    if demo_mode: return
    try:
        positions = bitfinex.fetch_positions()
        for pos in positions:
            if pos['notional'] > MAX_RISK:
                kill_switch(pos['symbol'])
    except Exception as e:
        logger.error(f'Size kill error: {e}')

def start_monitoring(check_interval=60, symbol=DEFAULT_SYMBOL, target=DEFAULT_TARGET, max_loss=DEFAULT_MAX_LOSS):
    while True:
        size_kill()
        pnl_close(symbol, target, max_loss)
        time.sleep(check_interval)

def save_settings(settings):
    try:
        with open('bitfinex_risk_settings.json', 'w') as f:
            json.dump(settings, f, indent=4)
        logger.info('Settings saved')
    except Exception as e:
        logger.error(f'Error saving: {e}')

def load_settings():
    try:
        if os.path.exists('bitfinex_risk_settings.json'):
            with open('bitfinex_risk_settings.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f'Error loading: {e}')
        return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bitfinex Risk Management Tool')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode')
    args = parser.parse_args()
    print('\n=== Bitfinex Risk Management Tool ===\n')
    bitfinex = init_exchange(args.demo)
    if not test_api_connection():
        sys.exit(1)
    try:
        settings = load_settings()
        if demo_mode:
            logger.info('\n⚠️ DEMO MODE ACTIVE\n')
        logger.info('\nFetching balance...')
        if not demo_mode:
            bal = bitfinex.fetch_balance({'type': 'derivatives'})
        logger.info('Balance retrieved')
        logger.info('\nChecking open positions...')
        position_data = open_positions()
        if position_data[1]:
            logger.info(f'\nFound open position for {DEFAULT_SYMBOL}')
            logger.info(f'Side: {"Long" if position_data[3] else "Short"}')
            logger.info(f'Size: {position_data[2]}')
            user_input = input('\nDo you want to (c)lose, (m)onitor, or (q)uit? [c/m/q]: ').lower()
            if user_input == 'c':
                kill_switch()
            elif user_input == 'm':
                interval = settings.get('check_interval', 60)
                target = settings.get('target', DEFAULT_TARGET)
                max_loss = settings.get('max_loss', DEFAULT_MAX_LOSS)
                check_interval = int(input(f'Interval seconds [{interval}]: ') or interval)
                target = float(input(f'Take profit % [{target}]: ') or target)
                max_loss = float(input(f'Stop loss % [{max_loss}]: ') or max_loss)
                save_settings({'check_interval': check_interval, 'target': target, 'max_loss': max_loss})
                logger.info('\nStarting monitoring...')
                start_monitoring(check_interval, DEFAULT_SYMBOL, target, max_loss)
            else:
                logger.info('\nExiting.')
        else:
            logger.info('\nNo open positions.')
    except Exception as e:
        logger.error(f'Error: {e}') 