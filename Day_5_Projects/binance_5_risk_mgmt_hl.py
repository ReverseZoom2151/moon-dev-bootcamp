import sys, os, ccxt, logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import binance_key, binance_secret
from binance_nice_funcs import get_position, pnl_close, acct_bal, kill_switch

# Configuration constants
SYMBOL = 'WIFUSDT'  # Adjusted for Binance USDT-margined
MAX_LOSS = -5
TARGET = 4
ACCT_MIN = 7
TIMEFRAME = '4h'
SIZE = 10

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Binance
binance = ccxt.binance({'enableRateLimit': True, 'apiKey': binance_key, 'secret': binance_secret, 'options': {'defaultType': 'future'}})

def bot():
    logging.info('Starting...')
    pos = get_position(SYMBOL)
    if pos[1]:
        pnl_close(SYMBOL, TARGET, MAX_LOSS)
    acct_val = float(acct_bal())
    if acct_val < ACCT_MIN:
        kill_switch(SYMBOL)

if __name__ == '__main__':
    bot() 