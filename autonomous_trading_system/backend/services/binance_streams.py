import asyncio
import json
import os
import pytz
from datetime import datetime
from websockets import connect
from termcolor import cprint

# File names and symbols
LIQS_FILE = 'binance.csv'
BIG_LIQS_FILE = 'binance_bigliqs.csv'
FUNDING_SYMBOLS = ['btcusdt', 'ethusdt', 'solusdt', 'wifusdt']
HUGE_TRADES_FILE = 'binance_trades_big.csv'
RECENT_TRADES_FILE = 'binance_trades.csv'

# Ensure CSV headers

def ensure_liqs_file():
    if not os.path.isfile(LIQS_FILE):
        with open(LIQS_FILE, 'w') as f:
            f.write(",".join([
                'symbol', 'side', 'order_type', 'time_in_force', 'original_quantity',
                'price', 'average_price', 'order_status', 'order_last_filled_quantity',
                'order_filled_accumulated_quantity', 'order_trade_time', 'usd_size'
            ]) + "\n")

def ensure_big_liqs_file():
    if not os.path.isfile(BIG_LIQS_FILE):
        with open(BIG_LIQS_FILE, 'w') as f:
            f.write(",".join([
                'symbol', 'side', 'order_type', 'time_in_force', 'original_quantity',
                'price', 'average_price', 'order_status', 'order_last_filled_quantity',
                'order_filled_accumulated_quantity', 'order_trade_time', 'usd_size'
            ]) + "\n")

def ensure_huge_trades_file():
    if not os.path.isfile(HUGE_TRADES_FILE):
        with open(HUGE_TRADES_FILE, 'w') as f:
            f.write('Event Time,Symbol,Aggregate Trade ID,Price,Quantity,First Trade ID,Trade Time,Is Buyer Maker\n')

def ensure_recent_trades_file():
    if not os.path.isfile(RECENT_TRADES_FILE):
        with open(RECENT_TRADES_FILE, 'w') as f:
            f.write('Event Time,Symbol,Aggregate Trade ID,Price,Quantity,Trade Time,Is Buyer Maker,Trade Type\n')


async def liqs_stream():
    """
    General liquidation stream (Day 2 liqs.py)
    """
    uri = 'wss://fstream.binance.com/ws/!forceOrder@arr'
    ensure_liqs_file()
    while True:
        try:
            async with connect(uri) as websocket:
                while True:
                    msg = await websocket.recv()
                    order = json.loads(msg)['o']
                    symbol = order['s'].replace('USDT', '')
                    side = order['S']
                    timestamp = int(order['T'])
                    filled_qty = float(order['z'])
                    price = float(order['p'])
                    usd_size = filled_qty * price
                    est = pytz.timezone('Europe/Bucharest')
                    time_str = datetime.fromtimestamp(timestamp/1000, est).strftime('%H:%M:%S')
                    if usd_size > 3000:
                        liq_type = 'L LIQ' if side == 'SELL' else 'S LIQ'
                        output = f"{liq_type} {symbol[:4]} {time_str} {usd_size:.0f}"
                        color = 'green' if side == 'SELL' else 'red'
                        attrs = ['bold'] if usd_size > 10000 else []
                        if usd_size > 250000:
                            stars = '***'
                            attrs.append('blink')
                            output = f"{stars} {output}"
                            for _ in range(4):
                                cprint(output, 'white', f'on_{color}', attrs=attrs)
                        elif usd_size > 100000:
                            stars = '*'
                            attrs.append('blink')
                            output = f"{stars} {output}"
                            for _ in range(2):
                                cprint(output, 'white', f'on_{color}', attrs=attrs)
                        else:
                            cprint(output, 'white', f'on_{color}', attrs=attrs)
                            print('')
                    # Log to CSV
                    row = [str(order.get(k)) for k in ['s','S','o','f','q','p','ap','X','l','z','T']]
                    row.append(str(usd_size))
                    with open(LIQS_FILE, 'a') as f:
                        f.write(','.join(row) + '\n')
        except Exception as e:
            print(f"Error in liqs_stream: {e}")
            await asyncio.sleep(5)


async def big_liqs_stream():
    """
    Big liquidation stream (Day 2 big_liqs.py)
    """
    uri = 'wss://fstream.binance.com/ws/!forceOrder@arr'
    ensure_big_liqs_file()
    while True:
        try:
            async with connect(uri) as websocket:
                while True:
                    msg = await websocket.recv()
                    order = json.loads(msg)['o']
                    symbol = order['s'].replace('USDT', '')
                    side = order['S']
                    timestamp = int(order['T'])
                    filled_qty = float(order['z'])
                    price = float(order['p'])
                    usd_size = filled_qty * price
                    est = pytz.timezone('Europe/Bucharest')
                    time_str = datetime.fromtimestamp(timestamp/1000, est).strftime('%H:%M:%S')
                    if usd_size > 100000:
                        liq_type = 'L LIQ' if side == 'SELL' else 'S LIQ'
                        output = f"{liq_type} {symbol[:4]} {time_str} {usd_size:.2f}"
                        color = 'blue' if side == 'SELL' else 'magenta'
                        attrs = ['bold'] if usd_size > 10000 else []
                        cprint(output, 'white', f'on_{color}', attrs=attrs)
                        print('')
                    # Log to CSV
                    row = [str(order.get(k)) for k in ['s','S','o','f','q','p','ap','X','l','z','T']]
                    row.append(str(usd_size))
                    with open(BIG_LIQS_FILE, 'a') as f:
                        f.write(','.join(row) + '\n')
        except Exception as e:
            print(f"Error in big_liqs_stream: {e}")
            await asyncio.sleep(5)


async def funding_stream():
    """
    Funding rate stream (Day 2 funding.py)
    """
    print_lock = asyncio.Lock()
    counter = {'count': 0}
    while True:
        try:
            tasks = []
            for sym in FUNDING_SYMBOLS:
                tasks.append(_funding_symbol(sym, counter, print_lock))
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Error in funding_stream: {e}")
            await asyncio.sleep(5)

async def _funding_symbol(symbol, counter, print_lock):
    uri = f"wss://fstream.binance.com/ws/{symbol}@markPrice"
    async with connect(uri) as websocket:
        while True:
            try:
                async with print_lock:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    rate = float(data['r'])
                    yearly = rate * 3 * 365 * 100
                    time_str = datetime.fromtimestamp(data['E']/1000).strftime('%H:%M:%S')
                    disp = symbol.upper().replace('USDT', '')
                    if yearly > 50:
                        text, bg = 'white', 'on_red'
                    elif yearly > 30:
                        text, bg = 'white', 'on_yellow'
                    elif yearly > 5:
                        text, bg = 'black', 'on_cyan'
                    elif yearly < -10:
                        text, bg = 'white', 'on_green'
                    else:
                        text, bg = 'white', 'on_light_green'
                    cprint(f"{disp} funding: {yearly:.2f}%", text, bg)
                    counter['count'] += 1
                    if counter['count'] >= len(FUNDING_SYMBOLS):
                        cprint(f"{time_str} yearly fund", 'white', 'on_black')
                        counter['count'] = 0
            except Exception:
                await asyncio.sleep(5)


async def huge_trades_stream():
    """
    Aggregated huge trades stream (Day 2 huge_trades.py)
    """
    symbols = ['btcusdt','ethusdt','solusdt','bnbusdt','dogeusdt','wifusdt']
    ensure_huge_trades_file()
    class Aggregator:
        def __init__(self):
            self.buckets = {}
        async def add(self, sym, sec, size, is_buyer):
            key = (sym, sec, is_buyer)
            self.buckets[key] = self.buckets.get(key, 0) + size
        async def flush(self):
            now = datetime.utcnow().strftime('%H:%M:%S')
            to_del = []
            for (sym, sec, is_buyer), size in list(self.buckets.items()):
                if sec < now and size > 500000:
                    attrs = ['bold']
                    bg = 'on_blue' if not is_buyer else 'on_magenta'
                    tp = 'BUY' if not is_buyer else 'SELL'
                    if size > 3000000:
                        size /= 1000000
                        cprint(f"\033[5m{tp} {sym} {sec} ${size:.2f}m\033[0m", 'white', bg, attrs=attrs)
                    else:
                        size /= 1000000
                        cprint(f"{tp} {sym} {sec} ${size:.2f}m", 'white', bg, attrs=attrs)
                    to_del.append((sym, sec, is_buyer))
            for k in to_del:
                del self.buckets[k]
    agg = Aggregator()
    async def worker(sym):
        uri = f"wss://fstream.binance.com/ws/{sym}@aggTrade"
        async with connect(uri) as ws:
            while True:
                try:
                    msg = await ws.recv()
                    d = json.loads(msg)
                    size = float(d['p']) * float(d['q'])
                    sec = datetime.fromtimestamp(d['T']/1000, pytz.timezone('US/Eastern')).strftime('%H:%M:%S')
                    await agg.add(sym.upper().replace('USDT', ''), sec, size, d['m'])
                except:
                    await asyncio.sleep(5)
    async def flusher():
        while True:
            await asyncio.sleep(1)
            await agg.flush()
    tasks = [worker(s) for s in symbols] + [flusher()]
    await asyncio.gather(*tasks)


async def recent_trades_stream():
    """
    Recent large trades stream (Day 2 recent_trades.py)
    """
    symbols = ['btcusdt','ethusdt','solusdt','dogeusdt','bnbusdt','wifusdt']
    ensure_recent_trades_file()
    async def worker(sym):
        uri = f"wss://fstream.binance.com/ws/{sym}@aggTrade"
        async with connect(uri) as ws:
            while True:
                try:
                    msg = await ws.recv()
                    d = json.loads(msg)
                    size = float(d['p']) * float(d['q'])
                    if size > 14999:
                        event_time = int(d['E'])
                        agg_id = int(d['a'])
                        price = float(d['p'])
                        q = float(d['q'])
                        tt = int(d['T'])
                        m = d['m']
                        t_str = datetime.fromtimestamp(tt/1000, pytz.timezone('Europe/Bucharest')).strftime('%H:%M:%S')
                        tp = 'SELL' if m else 'BUY'
                        col = 'red' if tp == 'SELL' else 'green'
                        attrs = ['bold'] if size >= 50000 else []
                        cprint(f"{tp} {sym.upper().replace('USDT', '')} {t_str} ${size:,.0f}", color=col, attrs=attrs)
                        with open(RECENT_TRADES_FILE, 'a') as f:
                            f.write(f"{event_time}, {sym.upper()}, {agg_id}, {price}, {q}, {t_str}, {m}, {tp}\n")
                except:
                    await asyncio.sleep(5)
    await asyncio.gather(*(worker(s) for s in symbols)) 