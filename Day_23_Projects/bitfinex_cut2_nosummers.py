import pandas as pd
import os
import warnings
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.bitfinex_nice_funcs import create_exchange

warnings.filterwarnings('ignore')

EXCLUDE_START_MONTH = 5
EXCLUDE_END_MONTH = 9
OUTPUT_SUFFIX = '-nosummers'

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

def filter_exclude_months(df, start_month, end_month):
    print(f'Filtering data to exclude months {start_month} through {end_month}...')
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError('DataFrame index must be a DatetimeIndex.')
        exclude_mask = (df.index.month >= start_month) & (df.index.month <= end_month)
        filtered_df = df[~exclude_mask]
        print(f'Filtering complete. Result shape: {filtered_df.shape}')
        return filtered_df
    except Exception as e:
        print(f'Error during filtering: {e}')
        return None

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
    filtered_data = filter_exclude_months(data, EXCLUDE_START_MONTH, EXCLUDE_END_MONTH)
    if filtered_data is None or filtered_data.empty:
        print('Filtering resulted in empty data or an error occurred. No file saved.')
        return
    save_data(filtered_data, symbol, OUTPUT_SUFFIX)

if __name__ == '__main__':
    main() 