import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_12_Projects.binance_nice_funcs import create_exchange

output_dir = './output_data'
os.makedirs(output_dir, exist_ok=True)
exchange = create_exchange()

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'WIFUSDT', '1000PEPEUSDT']

def fetch_liquidations(symbol, since):
    liqs = exchange.fetch_liquidations(symbol, since=since)
    data = []
    for liq in liqs:
        info = liq['info']
        data.append({
            'symbol': symbol,
            'side': info.get('side'),
            'order_type': info.get('type'),
            'time_in_force': info.get('timeInForce'),
            'original_quantity': float(info.get('origQty', 0)),
            'price': float(info.get('price', 0)),
            'average_price': float(info.get('avgPrice', 0)),
            'order_status': info.get('status'),
            'order_last_filled_quantity': float(info.get('lastFilled', 0)),
            'order_filled_accumulated_quantity': float(info.get('cumQty', 0)),
            'order_trade_time': int(info.get('updateTime', 0)),
            'usd_size': float(info.get('cumQty', 0)) * float(info.get('avgPrice', 0))
        })
    return pd.DataFrame(data)

all_liqs = pd.DataFrame()
since = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
for symbol in symbols:
    df = fetch_liquidations(symbol, since)
    all_liqs = pd.concat([all_liqs, df])

all_liqs['LIQ_SIDE'] = np.select([
    (all_liqs['symbol'].str.len() <= 7) & (all_liqs['usd_size'] > 3000) & (all_liqs['side'] == 'SELL'),
    (all_liqs['symbol'].str.len() <= 7) & (all_liqs['usd_size'] > 3000) & (all_liqs['side'] == 'BUY')
], ['L LIQ', 'S LIQ'], default=None)

all_liqs = all_liqs[all_liqs['LIQ_SIDE'].notna()].copy()
all_liqs['datetime'] = pd.to_datetime(all_liqs['order_trade_time'], unit='ms')
all_liqs.set_index('datetime', inplace=True)
all_liqs['symbol'] = all_liqs['symbol'].str.replace('USDT', '')

for symbol in [s.replace('USDT', '') for s in symbols]:
    filtered_df = all_liqs[all_liqs['symbol'] == symbol]
    if not filtered_df.empty:
        output_path = os.path.join(output_dir, f'{symbol}_liq_data.csv')
        filtered_df.to_csv(output_path)
        print(f"Saved data for {symbol} to {output_path}")
    else:
        print(f"No data found for symbol {symbol}")

df_totals_L = all_liqs[all_liqs['LIQ_SIDE'] == 'L LIQ'].resample('5min').agg({'usd_size': 'sum', 'price': 'mean'})
df_totals_S = all_liqs[all_liqs['LIQ_SIDE'] == 'S LIQ'].resample('5min').agg({'usd_size': 'sum', 'price': 'mean'})
df_totals_L['LIQ_SIDE'] = 'L LIQ'
df_totals_S['LIQ_SIDE'] = 'S LIQ'
df_totals = pd.concat([df_totals_L, df_totals_S])
df_totals['symbol'] = 'All'
df_totals.reset_index(inplace=True)
df_totals = df_totals[['datetime', 'symbol', 'LIQ_SIDE', 'price', 'usd_size']]
output_totals_path = os.path.join(output_dir, 'total_liq_data.csv')
df_totals.to_csv(output_totals_path, index=False)
print(f"Saved aggregated totals to {output_totals_path}")
print("Aggregated Totals Head:")
print(df_totals.head()) 