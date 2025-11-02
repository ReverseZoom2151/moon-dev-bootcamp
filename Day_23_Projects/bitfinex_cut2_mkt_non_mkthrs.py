import pandas as pd
import os
import warnings
import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.bitfinex_nice_funcs import create_exchange

warnings.filterwarnings('ignore')

MARKET_OPEN_UTC_STR = '13:30'
MARKET_CLOSE_UTC_STR = '20:00'
MARKET_HOURS_SUFFIX = '-mkt-open'
NON_MARKET_HOURS_SUFFIX = '-mkt-closed'

def fetch_data(exchange, symbol, timeframe='1m', limit=10000):
    print(f'Fetching data for {symbol}...')
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms').dt.tz_localize('UTC')
        df.set_index('datetime', inplace=True)
        print('Data fetched successfully.')
        return df
    except Exception as e:
        print(f'Error fetching data: {e}')
        return None

def filter_market_non_market_hours(df, open_time_str, close_time_str):
    print(f'Filtering for market hours ({open_time_str}-{close_time_str} UTC) and non-market hours, weekdays only...')
    try:
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz.zone != 'UTC':
            raise ValueError('DataFrame index must be a UTC-localized DatetimeIndex.')
        open_time = datetime.datetime.strptime(open_time_str, '%H:%M').time()
        close_time = datetime.datetime.strptime(close_time_str, '%H:%M').time()
        weekdays_df = df[df.index.dayofweek < 5].copy()
        market_hours_df = weekdays_df.between_time(open_time_str, close_time_str, inclusive='left')
        non_market_hours_mask = (weekdays_df.index.time < open_time) | (weekdays_df.index.time >= close_time)
        non_market_hours_df = weekdays_df[non_market_hours_mask]
        print(f'Filtering complete. Market hours shape: {market_hours_df.shape}, Non-market hours shape: {non_market_hours_df.shape}')
        return market_hours_df, non_market_hours_df
    except Exception as e:
        print(f'Error during filtering: {e}')
        return None, None

def save_data(df, symbol, suffix):
    try:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{symbol.replace(':', '')}{suffix}.csv')
        df.to_csv(output_path)
        print(f'Data saved to: {output_path}')
        return output_path
    except Exception as e:
        print(f'Error saving data: {e}')
        return None

def main():
    exchange = create_exchange()
    symbol = 'BTC:USDT'
    data = fetch_data(exchange, symbol)
    if data is None:
        return
    market_hours_data, non_market_hours_data = filter_market_non_market_hours(data, MARKET_OPEN_UTC_STR, MARKET_CLOSE_UTC_STR)
    if market_hours_data is None or non_market_hours_data is None:
        print('Filtering failed. No files saved.')
        return
    if not market_hours_data.empty:
        save_data(market_hours_data, symbol, MARKET_HOURS_SUFFIX)
    else:
        print('Market hours data is empty. Skipping save.')
    if not non_market_hours_data.empty:
        save_data(non_market_hours_data, symbol, NON_MARKET_HOURS_SUFFIX)
    else:
        print('Non-market hours data is empty. Skipping save.')

if __name__ == '__main__':
    main() 