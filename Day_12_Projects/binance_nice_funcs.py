import ccxt
import pandas as pd
import pandas_ta as ta
import sys, os, time, logging
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import binance_key, binance_secret

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYMBOL = 'WIFUSDT'
ORDER_PARAMS = {'timeInForce': 'GTX'}

cache = {}
cache_ttl = {}

def cached_get(key, ttl_seconds, fetch_func):
    now = time.time()
    if key in cache and now - cache_ttl[key] < ttl_seconds:
        return cache[key]
    result = fetch_func()
    if result is not None:
        cache[key] = result
        cache_ttl[key] = now
    return result

def create_exchange():
    return ccxt.binance({'enableRateLimit': True, 'apiKey': binance_key, 'secret': binance_secret, 'options': {'defaultType': 'future'}})

def ask_bid(symbol, max_retries=3, retry_delay=2):
    retry_count = 0
    while retry_count < max_retries:
        try:
            exchange = create_exchange()
            ob = exchange.fetch_order_book(symbol)
            bid = ob['bids'][0][0] if ob['bids'] else None
            ask = ob['asks'][0][0] if ob['asks'] else None
            return ask, bid, ob
        except Exception as e:
            logger.error(f"Error in ask_bid for {symbol}: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
    cache_key = f"ask_bid_{symbol}"
    if cache_key in cache:
        return cache[cache_key]
    logger.error(f"All ask_bid attempts failed for {symbol}")
    return None, None, None

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
    acct_value = balance['total']['USDT'] if 'USDT' in balance['total'] else 0
    print(f'Current account value: {acct_value}')
    return float(acct_value)

def adjust_leverage_size_signal(symbol, leverage, acct_value):
    exchange = create_exchange()
    price = ask_bid(symbol)[0]
    if price is None:
        return leverage, 0
    size = (acct_value * 0.95 / price) * leverage
    rounding = get_sz_px_decimals(symbol)[0]
    size = round(size, rounding)
    return leverage, size

def get_position(symbol):
    exchange = create_exchange()
    positions = exchange.fetch_positions([symbol])
    if not positions or positions[0]['contracts'] == 0:
        return [], False, 0, None, 0, 0, None
    pos = positions[0]
    in_pos = True
    size = pos['contracts']
    pos_sym = symbol
    entry_px = pos['entryPrice']
    pnl_perc = pos['unrealizedPnl'] / (entry_px * size) * 100 if entry_px != 0 else 0
    print(f'Current {symbol} PnL: {pnl_perc:.2f}%')
    long = pos['side'] == 'long'
    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long

def get_position_andmaxpos(symbol, max_positions):
    exchange = create_exchange()
    all_positions = exchange.fetch_positions()
    open_positions = [p for p in all_positions if p['contracts'] != 0]
    num_of_pos = len(open_positions)
    if num_of_pos > max_positions:
        print(f"Position limit exceeded: {num_of_pos}/{max_positions}. Closing all positions.")
        for p in open_positions:
            kill_switch(p['symbol'])
        return [], False, 0, None, 0, 0, None, 0
    else:
        print(f"Position count: {num_of_pos}/{max_positions}")
    return get_position(symbol) + (num_of_pos,)

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
        if ask is None or bid is None:
            time.sleep(3)
            continue
        side = 'sell' if long else 'buy'
        px = ask if side == 'sell' else bid
        limit_order(symbol, side == 'buy', pos_size, px, True)
        time.sleep(5)
        positions, im_in_pos, pos_size, _, _, _, long = get_position(symbol)
        attempts += 1
    if im_in_pos:
        print(f"Failed to close position after {max_attempts} attempts")
    else:
        print(f"Position successfully closed for {symbol}")

def pnl_close(symbol, target=9, max_loss=-8):
    positions, im_in_pos, pos_size, pos_sym, entry_px, _, long = get_position(symbol)
    if not im_in_pos:
        print(f"No position to check PNL for {symbol}")
        return False
    exchange = create_exchange()
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    if current_price is None:
        return False
    diff = (current_price - entry_px) if long else (entry_px - current_price)
    pnl_perc = (diff / entry_px) * 100 if entry_px != 0 else 0
    print(f'Current {symbol} PnL: {pnl_perc:.2f}%')
    if pnl_perc >= target:
        print(f'PNL gain is {pnl_perc:.2f}% and target is {target}% - closing position (WIN)')
        kill_switch(pos_sym)
        return True
    elif pnl_perc <= max_loss:
        print(f'PNL loss is {pnl_perc:.2f}% and max loss is {max_loss}% - closing position (LOSS)')
        kill_switch(pos_sym)
        return True
    else:
        print(f'PNL is {pnl_perc:.2f}%, target is {target}%, max loss is {max_loss}% - holding position')
        return False

def close_all_positions():
    exchange = create_exchange()
    all_positions = exchange.fetch_positions()
    open_positions = [p['symbol'] for p in all_positions if p['contracts'] != 0]
    if not open_positions:
        print("No open positions to close")
        return
    print(f"Closing {len(open_positions)} open positions")
    for symbol in open_positions:
        kill_switch(symbol)
    print('All positions have been closed')

def calculate_bollinger_bands(df, length=20, std_dev=2):
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    bollinger_bands = ta.bbands(df['close'], length=length, std=std_dev)
    if bollinger_bands.empty:
        df['BBL'] = df['BBM'] = df['BBU'] = df['BandWidth'] = None
        return df, False, False
    bollinger_bands = bollinger_bands.iloc[:, [0, 1, 2]]
    bollinger_bands.columns = ['BBL', 'BBM', 'BBU']
    df = pd.concat([df, bollinger_bands], axis=1)
    df['BandWidth'] = df['BBU'] - df['BBL']
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
        timestamp = datetime.fromtimestamp(item[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        processed_data.append([timestamp, item[1], item[2], item[3], item[4], item[5]])
    df = pd.DataFrame(processed_data, columns=columns)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if len(df) > 2:
        df['support'] = df[:-2]['close'].min()
        df['resis'] = df[:-2]['close'].max()
    else:
        df['support'] = df['close'].min()
        df['resis'] = df['close'].max()
    df['support'] = df['support'].fillna(df['close'].min())
    df['resis'] = df['resis'].fillna(df['close'].max())
    return df

def get_ohlcv2(symbol, timeframe, lookback_days, max_retries=3, retry_delay=2):
    retry_count = 0
    logger.info(f"Fetching OHLCV for {symbol} on {timeframe} with {lookback_days} days lookback")
    while retry_count < max_retries:
        try:
            exchange = create_exchange()
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
            if ohlcv:
                cache_key = f"ohlcv_{symbol}_{timeframe}_{lookback_days}"
                cache[cache_key] = ohlcv
                cache_ttl[cache_key] = time.time()
                return ohlcv
            else:
                logger.error(f"No OHLCV data returned for {symbol}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Error in get_ohlcv2 for {symbol}: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(retry_delay)
    cache_key = f"ohlcv_{symbol}_{timeframe}_{lookback_days}"
    if cache_key in cache:
        ttl = 3600
        if time.time() - cache_ttl[cache_key] < ttl:
            logger.warning(f"Using cached OHLCV data for {symbol}")
            return cache[cache_key]
    logger.error(f"All get_ohlcv2 attempts failed for {symbol}")
    # Mock data fallback
    mock_data = []
    for i in range(10):
        timestamp = int((datetime.now() - timedelta(minutes=i*15)).timestamp() * 1000)
        mock_data.append([timestamp, 20000, 20100, 19900, 20050, 100])
    logger.warning(f"Using mock data for {symbol}")
    return mock_data

def fetch_candle_snapshot(symbol, timeframe, start_time, end_time):
    exchange = create_exchange()
    since = int(start_time.timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=None)
    return ohlcv

def calculate_sma(prices, window):
    sma = prices.rolling(window=window).mean()
    return sma.iloc[-1]

def get_latest_sma(symbol, interval, window, lookback_days=1):
    start_time = datetime.now() - timedelta(days=lookback_days)
    end_time = datetime.now()
    snapshots = fetch_candle_snapshot(symbol, interval, start_time, end_time)
    if snapshots:
        prices = pd.Series([float(snapshot[4]) for snapshot in snapshots])
        return calculate_sma(prices, window)
    else:
        return None

def supply_demand_zones(symbol, timeframe, lookback_days):
    logger.info(f'Calculating supply and demand zones for {symbol}')
    sd_df = pd.DataFrame()
    snapshot_data = get_ohlcv2(symbol, timeframe, lookback_days)
    if not snapshot_data:
        return pd.DataFrame()
    df = process_data_to_df(snapshot_data)
    if df.empty:
        return pd.DataFrame()
    supp = df.iloc[-1]['support']
    resis = df.iloc[-1]['resis']
    df['supp_lo'] = df[:-2]['low'].min()
    supp_lo = df.iloc[-1]['supp_lo']
    df['res_hi'] = df[:-2]['high'].max()
    res_hi = df.iloc[-1]['res_hi']
    sd_df[f'{timeframe}_dz'] = [supp_lo, supp]
    sd_df[f'{timeframe}_sz'] = [res_hi, resis]
    logger.info(f'Supply and demand zones for {symbol}:\n{sd_df}')
    return sd_df 

def df_sma(symbol, timeframe='15m', limit=100, sma=20):
    logger.info(f'Getting SMA data for {symbol} on {timeframe} timeframe')
    timeframe_days = {'1m': 1/1440, '5m': 5/1440, '15m': 15/1440, '30m': 30/1440, '1h': 1/24, '4h': 4/24, '1d': 1}.get(timeframe, 15/1440)
    lookback_days = max(1, (limit * timeframe_days) * 1.1)
    logger.info(f'Fetching {limit} candles with {lookback_days:.2f} days lookback')
    snapshot_data = get_ohlcv2(symbol, timeframe, lookback_days)
    df = process_data_to_df(snapshot_data)
    if df.empty:
        logger.warning(f"Creating mock DataFrame for {symbol}")
        timestamps = [(datetime.now() - timedelta(minutes=i*int(timeframe.replace('m', '')))).strftime('%Y-%m-%d %H:%M:%S') for i in range(limit)]
        prices = []
        base_price = 20000
        for i in range(limit):
            mod = (i % 10) - 5
            price = base_price + mod * 100
            prices.append(price)
        data = []
        for i, ts in enumerate(timestamps):
            price = prices[i]
            data.append([ts, price - 10, price + 20, price - 20, price, 100])
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['support'] = df['close'].min() - 50
        df['resis'] = df['close'].max() + 50
    if len(df) > limit:
        df = df.iloc[-limit:]
    df[f'sma_{sma}'] = df['close'].rolling(window=sma).mean()
    if 'support' not in df.columns:
        df['support'] = df['close'].min()
    if 'resis' not in df.columns:
        df['resis'] = df['close'].max()
    df['support'] = df['support'].fillna(df['close'].min())
    df['resis'] = df['resis'].fillna(df['close'].max())
    logger.info(f'Successfully built DataFrame with {len(df)} rows for {symbol}')
    cache_key = f"df_sma_{symbol}_{timeframe}_{limit}_{sma}"
    cache[cache_key] = df
    cache_ttl[cache_key] = time.time()
    return df

def calculate_vwap_with_symbol(symbol):
    snapshot_data = get_ohlcv2(symbol, '15m', 300)
    if snapshot_data is None:
        return pd.DataFrame(), None
    df = process_data_to_df(snapshot_data)
    if df.empty:
        return df, None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    numeric_columns = ['high', 'low', 'close', 'volume']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df.dropna(subset=numeric_columns, inplace=True)
    df.sort_index(inplace=True)
    df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    latest_vwap = df['VWAP'].iloc[-1]
    logger.info(f'VWAP for {symbol}: {latest_vwap}')
    return df, latest_vwap

def open_positions(symbol):
    positions, in_pos, size, pos_sym, entry_px, pnl_perc, long = get_position(symbol)
    index_pos = 3
    logger.info(f"open_positions result for {symbol}: in_pos={in_pos}, size={size}")
    return positions, in_pos, size, long, index_pos

def sleep_on_close(symbol, pause_time):
    logger.info(f"Sleep on close called for {symbol} with pause time {pause_time} minutes")
    return False

def handle_api_response(response, operation):
    if response.status_code == 429:
        logger.warning(f"Rate limit hit during {operation}, backing off")
        time.sleep(2)
        return None
    elif response.status_code != 200:
        logger.error(f"API error during {operation}: {response.status_code}")
        return None
    return response.json()

def validate_symbol(symbol):
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("Symbol must be a non-empty string") 

def open_order_deluxe(symbol_info, size):
    exchange = create_exchange()
    symbol = symbol_info['Symbol']
    entry_price = symbol_info['Entry Price']
    stop_loss = symbol_info['Stop Loss']
    take_profit = symbol_info['Take Profit']
    _, px_decimals = get_sz_px_decimals(symbol)
    entry_price = round(entry_price, px_decimals)
    stop_loss = round(stop_loss, px_decimals)
    take_profit = round(take_profit, px_decimals)
    is_buy = True
    cancel_all_orders(symbol)
    order_result = limit_order(symbol, is_buy, size, entry_price, False)
    logger.info(f'Limit order result for {symbol}: {order_result}')
    stop_params = {'stopPrice': stop_loss, 'type': 'stop_market', 'reduceOnly': True}
    stop_side = 'sell' if is_buy else 'buy'
    stop_result = exchange.create_order(symbol, 'stop_market', stop_side, size, params=stop_params)
    logger.info(f'Stop loss order result for {symbol}: {stop_result}')
    tp_params = {'stopPrice': take_profit, 'type': 'take_profit_market', 'reduceOnly': True}
    tp_side = 'sell' if is_buy else 'buy'
    tp_result = exchange.create_order(symbol, 'take_profit_market', tp_side, size, params=tp_params)
    logger.info(f'Take profit order result for {symbol}: {tp_result}') 

class PositionManager:
    def __init__(self):
        self.exchange = create_exchange()
    def get_position(self, symbol):
        return get_position(symbol)
    def get_position_and_check_max(self, symbol, max_positions):
        return get_position_andmaxpos(symbol, max_positions)
    def adjust_leverage(self, symbol, leverage):
        logger.info(f'Setting leverage to {leverage}x for {symbol}')
        return self.exchange.set_leverage(leverage, symbol)
    def adjust_leverage_and_calculate_size(self, symbol, leverage):
        acct_value = acct_bal()
        return adjust_leverage_size_signal(symbol, leverage, acct_value)
    def adjust_leverage_usd_size(self, symbol, usd_size, leverage):
        acct_value = acct_bal()
        lev, _ = adjust_leverage_size_signal(symbol, leverage, acct_value)
        price = ask_bid(symbol)[0]
        size = (usd_size / price) * leverage
        rounding = get_sz_px_decimals(symbol)[0]
        size = round(size, rounding)
        return lev, size
    def kill_switch(self, symbol):
        kill_switch(symbol)
        return True
    def pnl_close(self, symbol, target, max_loss):
        pnl_close(symbol, target, max_loss)
    def cancel_all_orders(self):
        cancel_all_orders(None)  # Cancel all
        return True
    def cancel_symbol_orders(self, symbol):
        cancel_all_orders(symbol)
        return True
    def close_all_positions(self):
        close_all_positions()
        return True

class MarketAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
    def get_market_data(self, symbol, interval, lookback_days):
        return process_data_to_df(get_ohlcv2(symbol, interval, lookback_days))
    def should_quote_orders(self):
        logger.info("Checking if we should quote orders")
        df = self.get_market_data('BTCUSDT', '1m', 1/6)  # ~4 hours
        if df.empty:
            return False
        price_range = self.calculate_range(df, 240)
        logger.info(f"BTC price range in last 4 hours: ${price_range}")
        if price_range > 500:
            logger.warning("Market too volatile for quoting orders")
            return False
        logger.info("Market conditions acceptable for quoting orders")
        return True
    def calculate_range(self, df, window):
        window = min(window, len(df))
        recent_df = df[-window:].copy()
        recent_df['high'] = pd.to_numeric(recent_df['high'], errors='coerce')
        recent_df['low'] = pd.to_numeric(recent_df['low'], errors='coerce')
        recent_df.dropna(subset=['high', 'low'], inplace=True)
        if recent_df.empty:
            return 0.0
        return recent_df['high'].max() - recent_df['low'].min()
    def volume_spike(self, df):
        df['MA_Volume'] = df['volume'].rolling(window=20).mean()
        df['MA_Close'] = df['close'].rolling(window=20).mean()
        if df.empty or df['MA_Volume'].isna().all() or df['MA_Close'].isna().all():
            return False
        latest_data = df.iloc[-1]
        return (latest_data['volume'] > self.config.get('volume_multiplier', 3) * latest_data['MA_Volume'] and latest_data['MA_Close'] > latest_data['close'])
    def linear_regression_bollinger(self, df, bb_length=20, bb_std_dev=2, lrc_length=20, proximity_threshold=0.02):
        df, tight, wide = calculate_bollinger_bands(df, bb_length, bb_std_dev)
        df, lrc_quote_buy, lrc_quote_sell, lrc_quote_both = self.calculate_linear_regression_channel(df, lrc_length, proximity_threshold)
        quote_buy_orders = lrc_quote_buy and not wide
        quote_sell_orders = lrc_quote_sell and not wide
        quote_both_orders = lrc_quote_both or tight
        return df, quote_buy_orders, quote_sell_orders, quote_both_orders
    def calculate_linear_regression_channel(self, df, length=20, proximity_threshold=0.02):
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        linreg_values = ta.linreg(df['close'], length=length)
        df['LRCM_'] = linreg_values
        channel_width = df['close'].rolling(window=length).std()
        df['LRCT_'] = df['LRCM_'] + channel_width * 2
        df['LRCB_'] = df['LRCM_'] - channel_width * 2
        slope = (df['LRCM_'].iloc[-1] - df['LRCM_'].iloc[-length]) / length if len(df) >= length else 0
        quote_buy_orders = slope > 0 and df['close'].iloc[-1] < df['LRCT_'].iloc[-1]
        quote_sell_orders = slope < 0 and df['close'].iloc[-1] > df['LRCB_'].iloc[-1]
        proximity_to_middle = abs(df['close'].iloc[-1] - df['LRCM_'].iloc[-1])
        is_near_middle = proximity_to_middle <= (channel_width.iloc[-1] * 2 * proximity_threshold)
        quote_both_orders = is_near_middle
        return df, quote_buy_orders, quote_sell_orders, quote_both_orders
    def supply_demand_zones(self, symbol, timeframe, limit):
        return supply_demand_zones(symbol, timeframe, limit)
    def calculate_bollinger_bands(self, df, length=20, std_dev=2):
        return calculate_bollinger_bands(df, length, std_dev)

def get_open_interest():
    exchange = create_exchange()
    ticker = exchange.fetch_ticker('BTCUSDT')
    return ticker.get('info', {}).get('openInterest')

def get_liquidations():
    exchange = create_exchange()
    liqs = exchange.fetch_liquidations('BTCUSDT')
    total = sum(liq['info']['amount'] for liq in liqs)
    long = sum(liq['info']['amount'] for liq in liqs if liq['info']['side'] == 'long')
    short = sum(liq['info']['amount'] for liq in liqs if liq['info']['side'] == 'short')
    return {'Total Liquidations': total, 'Long Liquidations': long, 'Short Liquidations': short}

def get_funding_rate():
    exchange = create_exchange()
    funding = exchange.fetch_funding_rate('BTCUSDT')
    rate = funding.get('info', {}).get('fundingRate')
    yearly = rate * 24 * 365 if rate else 0
    return {'Funding Rate': rate, 'Yearly Funding Rate': yearly} 