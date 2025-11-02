import sys, os, ccxt, time, pandas as pd
from ta.momentum import RSIIndicator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import bitfinex_key, bitfinex_secret

# Initialize Bitfinex
bitfinex = ccxt.bitfinex({'enableRateLimit': True, 'apiKey': bitfinex_key, 'secret': bitfinex_secret})

# Default values
symbol = 'BTC:UST'
size = 0.0001
default_params = {'timeInForce': 'POC'}
timeframe = '15m'
limit = 100
sma = 20
target = 9
max_loss = -8

def open_positions(symbol=symbol):
    try:
        positions = bitfinex.fetch_positions([symbol])
        if not positions or positions[0]['amount'] == 0:
            print(f'No position for {symbol}')
            return [], False, 0, None
        pos = positions[0]
        openpos_bool = True
        openpos_size = pos['amount']
        long = pos['side'] == 'long'
        print(f'Position status: Open: {openpos_bool} | Size: {openpos_size} | Long: {long}')
        return positions, openpos_bool, openpos_size, long
    except Exception as e:
        print(f'Error fetching positions: {e}')
        return [], False, 0, None

def ask_bid(symbol=symbol):
    try:
        ob = bitfinex.fetch_order_book(symbol)
        bid = ob['bids'][0][0] if ob['bids'] else 0
        ask = ob['asks'][0][0] if ob['asks'] else 0
        print(f'Current prices for {symbol} - Ask: {ask}, Bid: {bid}')
        return ask, bid
    except Exception as e:
        print(f'Error fetching order book: {e}')
        return 0, 0

def kill_switch(symbol=symbol):
    print(f'Starting kill switch for {symbol}')
    try:
        bitfinex.cancel_all_orders(symbol)
        pos = open_positions(symbol)
        if pos[1]:
            side = 'sell' if pos[3] else 'buy'
            kill_size = pos[2]
            ask, bid = ask_bid(symbol)
            if side == 'sell':
                bitfinex.create_limit_sell_order(symbol, kill_size, ask, default_params)
            else:
                bitfinex.create_limit_buy_order(symbol, kill_size, bid, default_params)
            time.sleep(30)
            if open_positions(symbol)[1]:
                print('Position still open after attempt')
    except Exception as e:
        print(f'Error in kill switch: {e}')

def pnl_close(symbol=symbol, target=target, max_loss=max_loss):
    try:
        pos = open_positions(symbol)
        if not pos[1]:
            return False, False, 0, None
        entry_price = pos[0]['entry_price']
        leverage = pos[0]['leverage']
        current_price = ask_bid(symbol)[1] if pos[3] else ask_bid(symbol)[0]
        diff = (current_price - entry_price) if pos[3] else (entry_price - current_price)
        perc = round(((diff / entry_price) * leverage), 10) * 100
        if perc >= target or perc <= max_loss:
            kill_switch(symbol)
            return True, True, pos[2], pos[3]
        return False, True, pos[2], pos[3]
    except Exception as e:
        print(f'Error in PNL close: {e}')
        return False, False, 0, None

def size_kill(max_risk=1000):
    print(f'Checking position sizes against max risk of {max_risk}')
    try:
        positions = bitfinex.fetch_positions()
        for position in positions:
            if position['notional'] > max_risk:
                kill_switch(position['symbol'])
    except Exception as e:
        print(f'Error in size kill: {e}')

def df_rsi(symbol=symbol, timeframe=timeframe, limit=limit, rsi_period=14):
    print(f'Calculating RSI({rsi_period}) for {symbol} on {timeframe} timeframe')
    try:
        bars = bitfinex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not bars:
            print('No price data received')
            return pd.DataFrame()
        df_rsi = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        rsi = RSIIndicator(df_rsi['close'], window=rsi_period)
        df_rsi['rsi'] = rsi.rsi()
        df_rsi.loc[df_rsi['rsi'] > 70, 'signal'] = 'SELL'
        df_rsi.loc[df_rsi['rsi'] < 30, 'signal'] = 'BUY'
        print(f'RSI calculation completed with {len(df_rsi)} candles')
        return df_rsi
    except Exception as e:
        print(f'Error in RSI calculation: {e}')
        return pd.DataFrame()

if __name__ == '__main__':
    print('Starting RSI trading bot...')
    positions, is_position_open, position_size, is_long = open_positions(symbol)
    print(f'Current position status: {"Open" if is_position_open else "Closed"}')
    rsi_data = df_rsi(symbol)
    if not rsi_data.empty:
        latest_rsi = rsi_data['rsi'].iloc[-1]
        signal = rsi_data['signal'].iloc[-1] if pd.notna(rsi_data['signal'].iloc[-1]) else 'NEUTRAL'
        print(f'Latest RSI for {symbol}: {latest_rsi:.2f}')
        print(f'Signal: {signal}')
    size_kill()
    print('RSI trading bot execution completed') 