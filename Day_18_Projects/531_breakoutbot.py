from datetime import datetime, timedelta
import nice_funcs as n
from eth_account.signers.local import LocalAccount
import eth_account
import json
import time, random
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import ccxt
import pandas as pd
import schedule 
import requests
import sys, os
# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import project modules
from Day_4_Projects import dontshare as d
from Day_4_Projects import key_file as k
from eth_account.signers.local import LocalAccount

order_usd_size = 10

lookback_hours = 1
leverage = 3

def get_symbols():
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print('Error:', response.status_code)
        return []
    
    data = response.json()
    symbols = [symbol['name'] for symbol in data['universe']]
    print('Fetched Symbols:', symbols)
    return symbols

def fetch_candle_snapshot(symbol, interval, start_time, end_time):
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol, 
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000)
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        snapshot_data = response.json()
        if 'candles' in snapshot_data:
            return snapshot_data['candles']
        else:
            return snapshot_data
    else:
        print(f"Error fetching data for {symbol}: {response.status_code}")
        return None
    
def fetch_daily_data(symbol, days=20):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    return fetch_candle_snapshot(symbol, '1d', start_time, end_time)

def calculate_daily_resistance(symbols):
    resistance_levels = {}
    for symbol in symbols:
        daily_data = fetch_daily_data(symbol)
        if daily_data:
            high_prices = [float(day['h']) for day in daily_data]
            resistance_levels[symbol] = max(high_prices)
            print(f"Resistance level for {symbol}: {resistance_levels[symbol]}")
    return resistance_levels

def check_breakout(symbol, hourly_snapshots, daily_resistance):
    current_close = float(hourly_snapshots['close'].iloc[-1])
    print(f'current close {current_close}')
    daily_resistance_level = daily_resistance.get(symbol, None)
    print(f'this is the daily resistance {daily_resistance_level}')
    if not daily_resistance_level:
        print(f'Daily resistance level not found for {symbol}')
        return None
    print(f'current close: {current_close} resistance level: {daily_resistance_level}')
    if current_close > daily_resistance_level:
        entry_price = current_close
        stop_loss = entry_price * (1 - 18 / 100)
        take_profit = entry_price * (1 + 3 / 100)
        print(f"Breakout detected for {symbol}: Entry Price: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
        if stop_loss < entry_price < take_profit:
            return {
                'Symbol': symbol,
                'Entry Price': entry_price,
                'Stop Loss': stop_loss,
                'Take Profit': take_profit
            }
    return None

def main():
    account1 = LocalAccount = eth_account.Account.from_key(k.secret)
    symbols = get_symbols()
    resistance_levels = calculate_daily_resistance(symbols)
    breakout_data = []
    print(resistance_levels)
    for symbol in symbols:
        snapshot_data = n.get_ohlcv2(symbol, '1h', 2)
        hourly_snapshots = n.process_data_to_df(snapshot_data)
        if not hourly_snapshots.empty:
            breakout_info = check_breakout(symbol, hourly_snapshots, resistance_levels)
            if breakout_info:
                breakout_data.append(breakout_info)
    breakout_df = pd.DataFrame(breakout_data)
    print("Breakout DataFrame:")
    print(breakout_df)

    breakout_df.to_csv('', index=False)

    for index, symbol_info in breakout_df.iterrows():
        print(f'This si the symbol info for index: {index}')
        print(symbol_info)
        symbol = symbol_info['Symbol']
        print(f'passing {symbol} to adjust leverage and open border...')
        lev, size = n.adjust_leverage_usd_size(symbol, order_usd_size, leverage, account1)
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = n.get_position(symbol, account1)
        if not im_in_pos:
            print(f'not in position for {symbol}')
            n.open_order_deluxe(symbol_info, size, account1)
            print(f'Order opened for {symbol}')
        else:
            print(f'already in position for {symbol}')

print('running algorithm...')
main()
schedule.every(1).minuters.do(main)

while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(f'Encountered an error: {e}')
        time.sleep(10)
