import sys, os, time, schedule, logging, pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.binance_nice_funcs import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

order_usd_size = 10
lookback_hours = 1
leverage = 3
exchange = create_exchange()

def get_symbols():
    markets = exchange.load_markets()
    symbols = [s for s in markets if s.endswith('USDT') and markets[s]['type'] == 'swap' and markets[s]['active']]
    logger.info(f'Fetched {len(symbols)} active USDT perpetual symbols')
    return symbols

def fetch_daily_data(symbol, days=20):
    return get_ohlcv2(symbol, '1d', days)

def calculate_daily_resistance(symbols):
    resistance_levels = {}
    for symbol in symbols:
        daily_data = fetch_daily_data(symbol)
        if daily_data:
            df = process_data_to_df(daily_data)
            resistance_levels[symbol] = df['High'].max()
            logger.info(f"Resistance level for {symbol}: {resistance_levels[symbol]}")
    return resistance_levels

def check_breakout(symbol, hourly_snapshots, daily_resistance):
    if hourly_snapshots.empty:
        return None
    current_close = float(hourly_snapshots['close'].iloc[-1])
    daily_resistance_level = daily_resistance.get(symbol)
    if not daily_resistance_level:
        logger.warning(f'Daily resistance not found for {symbol}')
        return None
    if current_close > daily_resistance_level:
        entry_price = current_close
        stop_loss = entry_price * (1 - 18 / 100)
        take_profit = entry_price * (1 + 3 / 100)
        if stop_loss < entry_price < take_profit:
            return {
                'Symbol': symbol,
                'Entry Price': entry_price,
                'Stop Loss': stop_loss,
                'Take Profit': take_profit
            }
    return None

def main():
    symbols = get_symbols()
    resistance_levels = calculate_daily_resistance(symbols)
    breakout_data = []
    for symbol in symbols:
        snapshot_data = get_ohlcv2(symbol, '1h', lookback_hours)
        hourly_snapshots = process_data_to_df(snapshot_data)
        breakout_info = check_breakout(symbol, hourly_snapshots, resistance_levels)
        if breakout_info:
            breakout_data.append(breakout_info)
    breakout_df = pd.DataFrame(breakout_data)
    logger.info("Breakout DataFrame:")
    logger.info(breakout_df)
    breakout_df.to_csv('binance_breakouts.csv', index=False)
    for _, symbol_info in breakout_df.iterrows():
        symbol = symbol_info['Symbol']
        acct_value = acct_bal()
        lev, size = adjust_leverage_size_signal(symbol, leverage, acct_value)
        positions, im_in_pos, _, _, _, _, _ = get_position(symbol)
        if not im_in_pos:
            logger.info(f'Not in position for {symbol}, placing order')
            entry_px = symbol_info['Entry Price']
            limit_order(symbol, True, size, entry_px, False)  # Assuming long entry
            logger.info(f'Order opened for {symbol}')
        else:
            logger.info(f'Already in position for {symbol}')

print('running algorithm...')
main()
schedule.every(1).minutes.do(main)

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        logger.error(f'Encountered an error: {e}')
        time.sleep(10) 