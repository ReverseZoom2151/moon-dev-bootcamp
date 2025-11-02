import pandas as pd
import os
import warnings
import sys
import ccxt
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.bitfinex_nice_funcs import create_exchange

warnings.filterwarnings('ignore')

NYSE_OPEN_UTC = '13:30'
ONE_HOUR_AFTER_OPEN_UTC = '14:30'
ONE_HOUR_BEFORE_CLOSE_UTC = '19:00'
NYSE_CLOSE_UTC = '20:00'
OUTPUT_SUFFIX = '-1stlasthr'

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

def filter_trading_hours(df, start_time_first_hour, end_time_first_hour, start_time_last_hour, end_time_last_hour):
    print(f'Filtering for {start_time_first_hour}-{end_time_first_hour} and {start_time_last_hour}-{end_time_last_hour} UTC, weekdays only...')
    try:
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz.zone != 'UTC':
            raise ValueError('DataFrame index must be a UTC-localized DatetimeIndex.')
        open_hours = df.between_time(start_time_first_hour, end_time_first_hour)
        open_hours = open_hours[open_hours.index.dayofweek < 5]
        close_hours = df.between_time(start_time_last_hour, end_time_last_hour)
        close_hours = close_hours[close_hours.index.dayofweek < 5]
        filtered_data = pd.concat([open_hours, close_hours]).sort_index()
        print(f'Filtering complete. Result shape: {filtered_data.shape}')
        return filtered_data
    except Exception as e:
        print(f'Error during filtering: {e}')
        return None

def save_filtered_data(df, symbol, suffix):
    try:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{symbol.replace(':', '')}{suffix}.csv')
        df.to_csv(output_path)
        print(f'Filtered data saved to: {output_path}')
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
    filtered_data = filter_trading_hours(data, NYSE_OPEN_UTC, ONE_HOUR_AFTER_OPEN_UTC, ONE_HOUR_BEFORE_CLOSE_UTC, NYSE_CLOSE_UTC)
    if filtered_data is None or filtered_data.empty:
        print('Filtering resulted in empty data or an error occurred. No file saved.')
        return
    save_filtered_data(filtered_data, symbol, OUTPUT_SUFFIX)

if __name__ == '__main__':
    main() 