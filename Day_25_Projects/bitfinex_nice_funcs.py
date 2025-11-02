import os
import json
import time
import math
import pprint
import re as reggie
import asyncio
import sys
import ccxt
import pandas as pd
import pandas_ta as ta
from termcolor import cprint
from openai import OpenAI
from datetime import datetime, timedelta
from websockets import connect

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (MINIMUM_TRADES_IN_LAST_HOUR, SELL_AT_MULTIPLE, USD_SIZE, STOP_LOSS_PERCENTAGE, DO_NOT_TRADE_LIST, PRIORITY_FEE, WALLET_ADDRESS)
from Day_4_Projects import dontshare as d
OPENAI_API_KEY = d.openai_key
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def create_exchange():
    exchange = ccxt.bitfinex2({
        'apiKey': d.bitfinex_api_key,
        'secret': d.bitfinex_secret,
        'enableRateLimit': True,
    })
    exchange.load_markets()
    return exchange

def print_pretty_json(data):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

def find_urls(string):
    return reggie.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(string))

def round_down(value, decimals):
    factor = 10 ** decimals
    try:
        return math.floor(float(value) * factor) / factor
    except ValueError:
        return 0

def get_time_range(days_back=10):
    now = datetime.now()
    start_date = now - timedelta(days=days_back)
    time_to = int(now.timestamp())
    time_from = int(start_date.timestamp())
    return time_from, time_to

def token_price(symbol):
    exchange = create_exchange()
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f'Error fetching price for {symbol}: {e}')
        return None

def get_data(symbol, timeframe='1m', limit=1000):
    exchange = create_exchange()
    print(f'Fetching {timeframe} OHLCV data for {symbol} ({limit} bars)...')
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        required_rows_for_ta = 40
        if len(df) < required_rows_for_ta and not df.empty:
            rows_to_add = required_rows_for_ta - len(df)
            first_row_replicated = pd.concat([df.iloc[0:1]] * rows_to_add, ignore_index=False)
            df = pd.concat([first_row_replicated, df])
            df.sort_index(inplace=True)
        if not df.empty:
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MA40'] = ta.sma(df['Close'], length=40)
            df['Price_above_MA20'] = df['Close'] > df['MA20']
            df['Price_above_MA40'] = df['Close'] > df['MA40']
            df['MA20_above_MA40'] = df['MA20'] > df['MA40']
            df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f'Error fetching OHLCV for {symbol}: {e}')
        return pd.DataFrame()

def get_position(symbol):
    exchange = create_exchange()
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos['symbol'] == symbol:
                return float(pos['contracts']) * (1 if pos['side'] == 'long' else -1)
        return 0.0
    except Exception as e:
        print(f'Error getting position for {symbol}: {e}')
        return 0.0

def market_buy(symbol, amount_usd, slippage):
    exchange = create_exchange()
    try:
        price = token_price(symbol)
        if price is None:
            return None
        amount = amount_usd / price
        order = exchange.create_market_buy_order(symbol, amount)
        return order['id']
    except Exception as e:
        print(f'Error market buy {symbol}: {e}')
        return None

def market_sell(symbol, amount_usd, slippage):
    exchange = create_exchange()
    try:
        price = token_price(symbol)
        if price is None:
            return None
        amount = amount_usd / price
        order = exchange.create_market_sell_order(symbol, amount)
        return order['id']
    except Exception as e:
        print(f'Error market sell {symbol}: {e}')
        return None

def pnl_close(symbol):
    print(f'Checking PNL close for {symbol}...')
    pos = get_position(symbol)
    if pos == 0:
        return
    price = token_price(symbol)
    if price is None:
        return
    usd_value = abs(pos) * price
    initial_investment_usd = USD_SIZE
    target_tp_usd = initial_investment_usd * SELL_AT_MULTIPLE
    target_sl_usd = initial_investment_usd * (1 + STOP_LOSS_PERCENTAGE)
    print(f'  Current Value: ${usd_value:.2f} | Target TP: >= ${target_tp_usd:.2f} | Target SL: <= ${target_sl_usd:.2f}')
    close_reason = None
    if usd_value >= target_tp_usd:
        close_reason = 'Take Profit'
    elif usd_value <= target_sl_usd:
        close_reason = 'Stop Loss'
    if close_reason:
        cprint(f'  {close_reason} triggered for {symbol}! Current value ${usd_value:.2f}. Attempting to close...', 'white', 'on_magenta')
        chunk_kill(symbol, 500)

def chunk_kill(symbol, max_usd_sell_size):
    print(f'Starting chunk kill for {symbol}, max chunk ${max_usd_sell_size:.2f}...')
    max_retries = 3
    retry_delay_s = 5
    while True:
        pos = get_position(symbol)
        if abs(pos) <= 0.000001:
            print(f'  Position for {symbol} is effectively zero. Chunk kill complete.')
            break
        price = token_price(symbol)
        if price is None or price <= 0:
            time.sleep(retry_delay_s)
            continue
        usd_value = abs(pos) * price
        print(f'  Current balance: {pos:.6f} (${usd_value:.2f})')
        chunk_usd = min(usd_value, max_usd_sell_size)
        for attempt in range(max_retries):
            tx_id = market_sell(symbol, chunk_usd, 1000) if pos > 0 else market_buy(symbol, chunk_usd, 1000)
            if tx_id:
                cprint(f'  Sell chunk submitted (Attempt {attempt+1}). Tx: {tx_id}', 'white', 'on_blue')
                break
            else:
                cprint(f'  Sell chunk failed (Attempt {attempt+1}). Retrying in {retry_delay_s}s...', 'yellow')
                time.sleep(retry_delay_s)
        time.sleep(10)
    print(f'Chunk kill process finished for {symbol}.')

def kill_switch(symbol):
    print(f'!!! KILL SWITCH ACTIVATED for {symbol} !!!')
    pos = get_position(symbol)
    if abs(pos) <= 0.000001:
        print(f'  No position found for {symbol}.')
        return
    price = token_price(symbol)
    if price is None:
        return
    usd_value = abs(pos) * price
    print(f'  Attempting to close full position: {pos:.6f} (${usd_value:.2f})')
    max_retries = 5
    retry_delay_s = 3
    for attempt in range(max_retries):
        tx_id = market_sell(symbol, usd_value, 1000) if pos > 0 else market_buy(symbol, usd_value, 1000)
        if tx_id:
            cprint(f'  Kill switch close submitted (Attempt {attempt+1}). Tx: {tx_id}', 'white', 'on_red')
            return
        else:
            cprint(f'  Kill switch close failed (Attempt {attempt+1}). Retrying in {retry_delay_s}s...', 'yellow')
            time.sleep(retry_delay_s)
    cprint(f'!!! KILL SWITCH FAILED for {symbol} after {max_retries} attempts. !!!', 'red', attrs=['bold'])

def close_all_positions():
    exchange = create_exchange()
    positions = exchange.fetch_positions()
    closed_count = 0
    for pos in positions:
        symbol = pos['symbol']
        if symbol in DO_NOT_TRADE_LIST:
            continue
        amount = abs(float(pos['contracts']))
        usd_value = amount * token_price(symbol)
        if usd_value > 0.10:
            print(f'Closing position for {symbol} (Value: ${usd_value:.2f})...')
            kill_switch(symbol)
            closed_count += 1
            time.sleep(5)
    print(f'--- Close all positions process finished. Attempted to close {closed_count} positions. ---')

def delete_dont_overtrade_file(filename='dont_overtrade.txt'):
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"'{filename}' has been deleted.")
        except OSError as e:
            print(f'Error deleting file "{filename}": {e}')

def supply_demand_zones(symbol, timeframe='1h', limit=50):
    print(f'Calculating Supply/Demand zones for {symbol} ({timeframe}, {limit} periods)...')
    df = get_data(symbol, timeframe=timeframe, limit=limit)
    if df is None or df.empty or len(df) < 3:
        print('  Not enough data to calculate S/D zones.')
        return None
    df_calc = df.iloc[-(limit+2):-2]
    if df_calc.empty:
        print('  Not enough historical data for S/D calculation.')
        return None
    supp_close = df_calc['Close'].min()
    resis_close = df_calc['Close'].max()
    supp_low = df_calc['Low'].min()
    resis_high = df_calc['High'].max()
    sd_zones = pd.DataFrame({'demand_zone': [supp_low, supp_close], 'supply_zone': [resis_high, resis_close]})
    return sd_zones

def should_not_trade(symbol, no_trade_hours_list=[], filename='dont_overtrade.txt'):
    now_hour = datetime.now().hour
    if now_hour in no_trade_hours_list:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as file:
                    if symbol in file.read():
                        print(f'Trading disallowed for {symbol}: Within no-trade hour ({now_hour}) AND in exclusion file.')
                        return True
            except Exception as e:
                print(f'Error reading exclusion file "{filename}": {e}')
    return False

def vibe_check(token_name):
    if not OPENAI_CLIENT:
        print('OpenAI client not initialized. Skipping vibe check.')
        return None
    prompt = f'Based on what you know about what is culturally relevant, or funny, or you think has a chance to be a viral meme, on a scale of 1-10, what score do you give this token name: {token_name}? Please reply with just a numeric score, where 10 is the best meme for the current period, and 1 is the least impactful.'
    print(f'Performing vibe check for: {token_name}')
    try:
        response = OPENAI_CLIENT.chat.completions.create(model='gpt-4o', messages=[{'role': 'system', 'content': 'You are a helpful assistant providing numeric scores.'}, {'role': 'user', 'content': prompt}], timeout=15)
        if response.choices:
            score_text = response.choices[0].message.content.strip()
            match = reggie.search(r'\d+', score_text)
            if match:
                score = int(match.group(0))
                print(f'  Vibe score: {score}')
                return score
            else:
                print(f'  Could not parse numeric score from AI response: "{score_text}"')
                return None
        else:
            print('  No response choices received from OpenAI.')
            return None
    except Exception as e:
        print(f'Error during OpenAI vibe check for "{token_name}": {e}')
        return None

def serialize_df_for_prompt(df, max_rows=20):
    if df is None or df.empty:
        return 'No market data available.'
    cols_to_include = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'RSI', 'MA40', 'Price_above_MA20', 'Price_above_MA40', 'MA20_above_MA40']
    available_cols = [col for col in cols_to_include if col in df.columns]
    df_portion = df.tail(max_rows)[available_cols]
    df_str = df_portion.reset_index().to_string(index=False)
    return df_str

def gpt4(prompt, df):
    if not OPENAI_CLIENT:
        print('OpenAI client not initialized. Skipping GPT-4 decision.')
        return None
    df_str = serialize_df_for_prompt(df)
    detailed_prompt = f'{prompt}\n\nMarket Data Summary:\n{df_str}\nBased on this recent market data, should I buy (True) or sell (False)? Respond ONLY with the word True or False.'
    print('Prompting GPT-4 for trade decision...')
    try:
        response = OPENAI_CLIENT.chat.completions.create(model='gpt-4o', messages=[{'role': 'system', 'content': 'You are a Financial Trading Expert analyzing OHLCV data and indicators. Respond ONLY with True (for Buy) or False (for Sell).'}, {'role': 'user', 'content': detailed_prompt}], timeout=30)
        if response.choices:
            decision_text = response.choices[0].message.content.strip().lower()
            print(f'  GPT-4 Response: "{decision_text}"')
            if 'true' in decision_text:
                return True
            elif 'false' in decision_text:
                return False
            else:
                print('  Warning: Could not parse True/False from GPT-4 response.')
                return None
        else:
            print('  No response choices received from OpenAI.')
            return None
    except Exception as e:
        print(f'Error during OpenAI GPT-4 decision: {e}')
        return None

async def async_get_btc_funding_rate():
    symbol = 'btcusdt'
    websocket_url = f'wss://fstream.bitfinex.com/ws/{symbol}@markPrice'  # Adjust if needed for Bitfinex
    async with connect(websocket_url) as websocket:
        try:
            message = await websocket.recv()
            data = json.loads(message)
            funding_rate = float(data['r'])  # Adjust key as per Bitfinex response
            yearly_funding_rate = (funding_rate * 3 * 365) * 100
            return yearly_funding_rate
        except Exception as e:
            print(f'An exception occurred while fetching funding rate: {e}')
            return None

def get_btc_funding_rate():
    return asyncio.get_event_loop().run_until_complete(async_get_btc_funding_rate()) 