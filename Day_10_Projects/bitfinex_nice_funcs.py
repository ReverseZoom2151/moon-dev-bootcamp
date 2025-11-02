import ccxt
import pandas as pd
import pandas_ta as ta
import sys, os, time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import bitfinex_key, bitfinex_secret

# Constants
DEFAULT_SYMBOL = 'WIF:USDT'
ORDER_PARAMS = {'timeInForce': 'POC'}

def create_exchange():
    return ccxt.bitfinex({'enableRateLimit': True, 'apiKey': bitfinex_key, 'secret': bitfinex_secret})

def ask_bid(symbol):
    exchange = create_exchange()
    ob = exchange.fetch_order_book(symbol)
    bid = ob['bids'][0][0] if ob['bids'] else None
    ask = ob['asks'][0][0] if ob['asks'] else None
    return ask, bid, ob

def get_sz_px_decimals(coin):
    exchange = create_exchange()
    market = exchange.load_markets()[coin]
    sz_decimals = market['precision']['amount']
    px_decimals = market['precision']['price']
    print(f'{coin} size precision: {sz_decimals} decimals, price precision: {px_decimals} decimals')
    return sz_decimals, px_decimals

def limit_order(coin, is_buy, sz, limit_px, reduce_only):
    exchange = create_exchange()
    side = 'buy' if is_buy else 'sell'
    params = ORDER_PARAMS.copy()
    params['reduceOnly'] = reduce_only
    print(f'Placing limit order for {coin}: {"BUY" if is_buy else "SELL"} {sz} @ {limit_px}')
    if is_buy:
        order = exchange.create_limit_buy_order(coin, sz, limit_px, params)
    else:
        order = exchange.create_limit_sell_order(coin, sz, limit_px, params)
    print(f'Limit {"BUY" if is_buy else "SELL"} order placed: {order}')
    return order

def acct_bal():
    exchange = create_exchange()
    balance = exchange.fetch_balance()
    acct_value = balance['info']['wallet_balance'] if 'wallet_balance' in balance['info'] else 0
    print(f'Current account value: {acct_value}')
    return float(acct_value)

def adjust_leverage_size_signal(symbol, leverage, acct_value):
    exchange = create_exchange()
    # Note: Bitfinex leverage is per position, may need adjustment
    price = ask_bid(symbol)[0]
    size = (acct_value * 0.95 / price) * leverage
    rounding = get_sz_px_decimals(symbol)[0]
    size = round(size, rounding)
    return leverage, size

def get_position(symbol):
    exchange = create_exchange()
    positions = exchange.fetch_positions([symbol])
    if not positions or positions[0]['amount'] == 0:
        return [], False, 0, None, 0, 0, None
    pos = positions[0]
    in_pos = True
    size = pos['amount']
    pos_sym = symbol
    entry_px = pos['entry_price']
    pnl_perc = pos['unrealized_pnl'] / (entry_px * abs(size)) * 100 if entry_px != 0 else 0
    print(f'Current {symbol} PnL: {pnl_perc:.2f}%')
    long = pos['side'] == 'long'
    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long

def cancel_all_orders(symbol):
    exchange = create_exchange()
    exchange.cancel_all_orders(symbol)
    print(f'All orders cancelled for {symbol}')

def kill_switch(symbol):
    exchange = create_exchange()
    positions, im_in_pos, pos_size, _, _, _, long = get_position(symbol)
    if not im_in_pos:
        print(f"No position to close for {symbol}")
        return
    print(f"Executing kill switch for {symbol} position")
    attempts = 0
    max_attempts = 5
    while im_in_pos and attempts < max_attempts:
        cancel_all_orders(symbol)
        ask, bid, _ = ask_bid(symbol)
        side = 'sell' if long else 'buy'
        px = ask if side == 'sell' else bid
        limit_order(symbol, side == 'buy', abs(pos_size), px, True)
        time.sleep(5)
        positions, im_in_pos, pos_size, _, _, _, long = get_position(symbol)
        attempts += 1
    print(f"Position successfully closed for {symbol}")

def pnl_close(symbol, target, max_loss):
    positions, im_in_pos, pos_size, pos_sym, entry_px, _, long = get_position(symbol)
    if not im_in_pos:
        print(f"No position to monitor for {symbol}")
        return
    exchange = create_exchange()
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    diff = (current_price - entry_px) if long else (entry_px - current_price)
    pnl_perc = (diff / entry_px) * 100 if entry_px != 0 else 0  # Assuming leverage=1, adjust if needed
    print(f'Current {symbol} PnL: {pnl_perc:.2f}%')
    if pnl_perc >= target:
        print(f"Target reached! Closing position.")
        kill_switch(pos_sym)
        return True
    elif pnl_perc <= max_loss:
        print(f"Stop loss hit! Closing position.")
        kill_switch(pos_sym)
        return True
    return False

def get_position_andmaxpos(symbol, max_positions):
    exchange = create_exchange()
    all_positions = exchange.fetch_positions()
    open_positions = [p for p in all_positions if p['amount'] != 0]
    num_of_pos = len(open_positions)
    if num_of_pos > max_positions:
        print(f"Position limit exceeded: {num_of_pos}/{max_positions}. Closing all positions.")
        for p in open_positions:
            kill_switch(p['symbol'])
        return [], False, 0, None, 0, 0, None, 0
    else:
        print(f"Position count: {num_of_pos}/{max_positions}")
    return get_position(symbol) + (num_of_pos,)

def close_all_positions():
    exchange = create_exchange()
    all_positions = exchange.fetch_positions()
    open_positions = [p['symbol'] for p in all_positions if p['amount'] != 0]
    if not open_positions:
        print("No open positions to close")
        return
    print(f"Closing {len(open_positions)} open positions")
    for symbol in open_positions:
        kill_switch(symbol)
    print('All positions have been closed')

def calculate_bollinger_bands(df, length=20, std_dev=2):
    bb = ta.bbands(df['close'], length=length, std=std_dev)
    if bb.empty:
        print("Warning: Unable to calculate Bollinger Bands")
        df['BBL'] = df['BBM'] = df['BBU'] = df['BandWidth'] = None
        return df, False, False
    bb = bb.iloc[:, [0, 1, 2]]
    bb.columns = ['BBL', 'BBM', 'BBU']
    df = pd.concat([df, bb], axis=1)
    df['BandWidth'] = df['BBU'] - df['BBL']
    if len(df) < 5:
        return df, False, False
    tight_threshold = df['BandWidth'].quantile(0.2)
    wide_threshold = df['BandWidth'].quantile(0.8)
    current_band_width = df['BandWidth'].iloc[-1]
    tight = current_band_width <= tight_threshold
    wide = current_band_width >= wide_threshold
    return df, tight, wide

def process_data_to_df(data):
    if not data:
        return pd.DataFrame()
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    processed_data = []
    for item in data:
        timestamp = datetime.fromtimestamp(item['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        processed_data.append([timestamp, item['o'], item['h'], item['l'], item['c'], item['v']])
    df = pd.DataFrame(processed_data, columns=columns)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if len(df) > 2:
        df['support'] = df[:-2]['close'].min()
        df['resis'] = df[:-2]['close'].max()
    else:
        df['support'] = df['close'].min()
        df['resis'] = df['close'].max()
    return df

def get_ohlcv2(symbol, interval, lookback_days):
    exchange = create_exchange()
    since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
    return exchange.fetch_ohlcv(symbol, interval, since=since, limit=500)

def fetch_candle_snapshot(symbol, interval, start_time, end_time):
    exchange = create_exchange()
    since = int(start_time.timestamp() * 1000)
    return exchange.fetch_ohlcv(symbol, interval, since=since, limit=1000, params={'to': int(end_time.timestamp() * 1000)})

def calculate_sma(prices, window):
    if len(prices) < window:
        return None
    return prices.rolling(window=window).mean().iloc[-1]

def get_latest_sma(symbol, interval, window, lookback_days=1):
    start_time = datetime.now() - timedelta(days=lookback_days)
    end_time = datetime.now()
    data = fetch_candle_snapshot(symbol, interval, start_time, end_time)
    if not data or len(data) < window:
        return None
    prices = pd.Series([candle[4] for candle in data])
    return calculate_sma(prices, window)

def supply_demand_zones_hl(symbol, timeframe, limit):
    print(f'Calculating supply and demand zones for {symbol} on {timeframe} timeframe...')
    sd_df = pd.DataFrame()
    data = get_ohlcv2(symbol, timeframe, limit)
    if not data:
        return sd_df
    df = process_data_to_df(data)
    if df.empty:
        return sd_df
    supp = df.iloc[-1]['support']
    resis = df.iloc[-1]['resis']
    df['supp_lo'] = df[:-2]['low'].min()
    supp_lo = df.iloc[-1]['supp_lo']
    df['res_hi'] = df[:-2]['high'].max()
    res_hi = df.iloc[-1]['res_hi']
    sd_df[f'{timeframe}_dz'] = [supp_lo, supp]
    sd_df[f'{timeframe}_sz'] = [res_hi, resis]
    print(f'Supply and demand zones for {symbol} on {timeframe}:')
    print(sd_df)
    return sd_df

def calculate_vwap_with_symbol(symbol):
    data = get_ohlcv2(symbol, '15m', 300)
    if not data:
        return pd.DataFrame(), None
    df = process_data_to_df(data)
    if df.empty:
        return df, None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    numeric_columns = ['high', 'low', 'close', 'volume']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(subset=numeric_columns, inplace=True)
    df.sort_index(inplace=True)
    if len(df) < 2:
        return df, None
    df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    latest_vwap = df['VWAP'].iloc[-1]
    return df, latest_vwap 