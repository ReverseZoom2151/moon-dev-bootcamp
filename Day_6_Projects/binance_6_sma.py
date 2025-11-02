import argparse, random, sys, os, logging, ccxt, time, schedule, pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import binance_key, binance_secret

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('binance_trading_bot.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger()

# Configuration
CONFIG = {'default_symbol': 'BTCUSDT', 'default_size': 1, 'timeframe': '15m', 'limit': 100, 'sma_period': 20, 'target_profit_percentage': 9, 'max_loss_percentage': -8, 'max_risk_amount': 1000, 'order_params': {'timeInForce': 'GTX'}, 'sleep_time': 30, 'emergency_timeout': 300}

demo_mode = False
binance = None

def init_exchange(use_demo_mode=False):
    global demo_mode, binance
    demo_mode = use_demo_mode
    if demo_mode:
        logger.info('⚠️ DEMO MODE')
        return None
    else:
        exchange = ccxt.binance({'enableRateLimit': True, 'apiKey': binance_key, 'secret': binance_secret, 'options': {'defaultType': 'future'}})
        logger.info('Exchange initialized')
        return exchange

def test_api_connection():
    if demo_mode:
        return True
    try:
        binance.fetch_ticker(CONFIG['default_symbol'])
        bal = binance.fetch_balance()
        return True
    except Exception as e:
        logger.error(f'API failed: {e}')
        return False

def get_open_positions(symbol=CONFIG['default_symbol']):
    if demo_mode:
        return random.choice([([{'side': 'long', 'contracts': 0.1}], True, 0.1, True), ([], False, 0, None)])
    try:
        positions = binance.fetch_positions([symbol])
        if not positions or positions[0]['contracts'] == 0:
            return [], False, 0, None
        pos = positions[0]
        return positions, True, pos['contracts'], pos['side'] == 'long'
    except Exception as e:
        logger.error(f'Positions error: {e}')
        return [], False, 0, None

def close_position(symbol=CONFIG['default_symbol']):
    if demo_mode:
        logger.info('[DEMO] Position closed')
        return True
    try:
        binance.cancel_all_orders(symbol)
        pos = get_open_positions(symbol)
        if pos[1]:
            side = 'sell' if pos[3] else 'buy'
            binance.create_market_order(symbol, side, pos[2], {'reduceOnly': True})
            return True
        return False
    except Exception as e:
        logger.error(f'Close error: {e}')
        return False

def check_pnl_and_manage_position(symbol=CONFIG['default_symbol']):
    if demo_mode:
        return random.choice([True, False])
    try:
        pos = get_open_positions(symbol)
        if not pos[1]: return False
        ticker = binance.fetch_ticker(symbol)
        current = ticker['last']
        entry = pos[0][0]['entryPrice']
        leverage = pos[0][0]['leverage']
        diff = (current - entry) if pos[3] else (entry - current)
        perc = (diff / entry * leverage) * 100
        if perc >= CONFIG['target_profit_percentage'] or perc <= CONFIG['max_loss_percentage']:
            close_position(symbol)
            return True
        return False
    except Exception as e:
        logger.error(f'PNL error: {e}')
        return False

def check_risk_exposure():
    if demo_mode:
        return True
    try:
        positions = binance.fetch_positions()
        total_notional = sum(p['notional'] for p in positions if p['notional'] > 0)
        if total_notional > CONFIG['max_risk_amount']:
            for p in positions:
                if p['notional'] > 0:
                    close_position(p['symbol'])
            return False
        return True
    except Exception as e:
        logger.error(f'Risk error: {e}')
        return False

def calculate_sma(symbol=CONFIG['default_symbol'], timeframe=CONFIG['timeframe'], limit=CONFIG['limit'], sma_period=CONFIG['sma_period']):
    if demo_mode:
        return pd.DataFrame()  # Simulate
    try:
        bars = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['sma'] = df['close'].rolling(sma_period).mean()
        current = binance.fetch_ticker(symbol)['last']
        last_sma = df['sma'].iloc[-1]
        signal = 'BUY' if current > last_sma else 'SELL' if current < last_sma else None
        logger.info(f'Signal for {symbol}: {signal}')
        return df, signal
    except Exception as e:
        logger.error(f'SMA error: {e}')
        return None, None

def execute_trade(symbol, signal, size=CONFIG['default_size']):
    if demo_mode or not signal:
        return
    try:
        current = binance.fetch_ticker(symbol)['last']
        if signal == 'BUY':
            binance.create_limit_buy_order(symbol, size, current)
        elif signal == 'SELL':
            binance.create_limit_sell_order(symbol, size, current)
    except Exception as e:
        logger.error(f'Trade error: {e}')

def run_trading_cycle(symbol=CONFIG['default_symbol']):
    if check_risk_exposure():
        df, signal = calculate_sma(symbol)
        execute_trade(symbol, signal)

def setup_schedule():
    schedule.every(15).minutes.do(run_trading_cycle)

def main():
    # Similar to original main, with args, init, loop
    pass

if __name__ == '__main__':
    main() 