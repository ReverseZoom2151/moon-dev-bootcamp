import ccxt, time, os, logging, argparse
import numpy as np # Added for RSI calculation
import pandas as pd
import pandas_ta as ta

# Setup logging
logging.basicConfig(filename='binance_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argparse for configurable parameters
parser = argparse.ArgumentParser(description='Binance Trading Bot')
parser.add_argument('--symbol', default='WIF/USDT:USDT', help='Trading symbol')
parser.add_argument('--size', type=float, default=1, help='Order size')
parser.add_argument('--dynamic', action='store_true', help='Use dynamic pricing (1% below/above market)')
parser.add_argument('--dry-run', action='store_true', help='Simulate orders without executing')
parser.add_argument('--testnet', action='store_true', help='Use testnet')
parser.add_argument('--strategy', default='none', choices=['none', 'rsi'], help='Trading strategy')
parser.add_argument('--max-orders', type=int, default=10, help='Max orders before stopping')
args = parser.parse_args()

# Use environment variables for API keys
api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')
if not api_key or not api_secret:
    logging.error('Missing BINANCE_API_KEY or BINANCE_API_SECRET environment variables')
    raise ValueError('API keys not set in environment')

# Initialize Binance exchange for futures
logging.info('Initializing connection to Binance futures...')
binance = ccxt.binance({
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'defaultType': 'future'}
})

# Load markets and validate symbol
binance.load_markets()
if args.symbol not in binance.markets:
    logging.error(f'Invalid symbol: {args.symbol}')
    raise ValueError(f'Symbol {args.symbol} not available on Binance')

# Add to argparse
if args.testnet:
    binance.set_sandbox_mode(True)
    logging.info('Using Binance testnet')

# Add order counter and PnL tracker
order_count = 0
pnl = 0.0

# Function to get ask and bid
def ask_bid(symbol):
    try:
        orderbook = binance.fetch_order_book(symbol)
        bid = orderbook['bids'][0][0] if orderbook['bids'] else 0.0
        ask = orderbook['asks'][0][0] if orderbook['asks'] else 0.0
        logging.info(f'Fetched prices: Bid={bid}, Ask={ask}')
        return ask, bid
    except Exception as e:
        logging.error(f'Error fetching order book: {e}')
        return 0.0, 0.0

# Function to place limit order
def limit_order(symbol, is_buy, limit_px, sz, reduce_only=False):
    if args.dry_run:
        logging.info(f'DRY RUN: Would place {"BUY" if is_buy else "SELL"} order: {sz} @ {limit_px} (reduce_only={reduce_only})')
        return {'id': 'dry-run'}
    try:
        side = 'buy' if is_buy else 'sell'
        params = {'reduceOnly': reduce_only} if reduce_only else {}
        order = binance.create_limit_order(symbol, side, sz, limit_px, params)
        logging.info(f'{side.upper()} order placed: {sz} @ {limit_px}')
        return order
    except Exception as e:
        logging.error(f'Error placing order: {e}')
        return None

# Function to get OHLCV for strategies
def get_ohlcv(symbol, timeframe='1h', limit=100):
    return binance.fetch_ohlcv(symbol, timeframe, limit=limit)

# Main execution
def main():
    global order_count, pnl  # If needed
    try:
        coin = args.symbol.split('/')[0]  # e.g., 'WIF'
        is_buy = True
        sz = args.size
        
        # Get current prices
        ask, bid = ask_bid(args.symbol)
        
        if ask == 0.0 or bid == 0.0:
            logging.error('Failed to get valid prices. Exiting.')
            raise ValueError('Invalid prices')
        else:
            logging.info(f'Got prices for {coin}: Ask = {ask}, Bid = {bid}')
            
            buy_price = bid * 0.99 if args.dynamic else bid
            sell_price = ask * 1.01 if args.dynamic else ask
            
            # Add max orders check and strategy
            if order_count >= args.max_orders:
                logging.info(f'Max orders ({args.max_orders}) reached. Exiting.')
                return

            if args.strategy == 'rsi':
                ohlcv = get_ohlcv(args.symbol)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['rsi'] = ta.rsi(df['close'], length=14)
                rsi = df['rsi'].iloc[-1]
                if rsi < 30:
                    logging.info(f'RSI {rsi} < 30: Buying')
                elif rsi > 70:
                    logging.info(f'RSI {rsi} > 70: Selling')
                else:
                    logging.info(f'RSI {rsi}: No action')
                    return

            # Place buy order
            logging.info(f'Placing BUY order for {coin} at {buy_price}')
            limit_order(args.symbol, is_buy, buy_price, sz)
            
            time.sleep(5)
            
            # Place sell order
            is_buy = False 
            reduce_only = True
            logging.info(f'Placing SELL order for {coin} at {sell_price}')
            limit_order(args.symbol, is_buy, sell_price, sz, reduce_only)

            # After placing orders, simulate PnL (e.g., assume fill at limit_px)
            # For buy: pnl -= sz * buy_price
            # For sell: pnl += sz * sell_price
            logging.info(f'Simulated PnL: {pnl}')

            # Increment order_count after each order
            order_count += 1
    except Exception as e:
        logging.error(f'Error in main execution: {e}') 

if __name__ == '__main__':
    main()
    # Basic unit test example
    import unittest
    class TestBot(unittest.TestCase):
        def test_ask_bid(self):
            ask, bid = ask_bid('BTCUSDT')
            self.assertGreater(ask, 0)
    unittest.main(exit=False)  # Run tests without exiting 