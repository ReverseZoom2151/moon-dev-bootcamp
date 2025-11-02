#from dontshare import key  
#key = 'klhjklhjklhjklhjk' -- this is whats in the dontshareconfig.py (private key, keep safe af)

import eth_account
import json
import time 
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import ccxt
import pandas as pd
import datetime
import requests 
from datetime import datetime, timedelta
import pandas_ta as ta
import ccxt 
print('sup dawg')

# hide all warnings
import warnings
warnings.filterwarnings('ignore')

symbol = 'SOL' 
timeframe = '15m'

max_loss = -1
target = 5
pos_size = 200
leverage = 10
vol_multiplier = 3
rounding = 4

cb_symbol = symbol + '/USDT' #BTC/USD

def ask_bid(symbol):

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {
        'type': 'l2Book',
        'coin': symbol
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    print(f'symbol: {symbol}....')
    #print(l2_data)
    l2_data = l2_data['levels']
    #print(l2_data)

    # get bid and ask 
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data


def spot_price_and_hoe_ass_spot_symbol(symbol):
    url = "https://api.hyperliquid.xyz/info"
    headers = {
        "Content-Type": "application/json"
    }
    body = {
        "type": "spotMetaAndAssetCtxs"
    }

    # Make the request to the API
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract tokens and universe data
        tokens = data[0]['tokens']
        universe = data[0]['universe']
        asset_ctxs = data[1]
        
        # Find the token index and szDecimals for the given symbol
        token_index = None
        sz_decimals = None
        for token in tokens:
            if token['name'] == symbol:
                token_index = token['index']
                sz_decimals = token['szDecimals']
                break

        if token_index is None:
            return f"Token symbol {symbol} not found."
        
        # Find the corresponding universe entry and hoe_ass_symbol
        for pair in universe:
            if token_index in pair['tokens']:
                hoe_ass_symbol = pair['name']
                universe_index = pair['index']
                mid_px = asset_ctxs[universe_index]['midPx']
                
                # Compute the number of decimal points in the mid price
                mid_px_str = str(mid_px)
                if '.' in mid_px_str:
                    px_decimals = len(mid_px_str.split('.')[1])
                    px_decimals = px_decimals-1
                else:
                    px_decimals = 0
                
                return mid_px, hoe_ass_symbol, sz_decimals, px_decimals

        return f"Token pair for symbol {symbol} not found in universe."
    else:
        return f"Error: {response.status_code} - {response.text}"


# symbol = "KOBE"  # Example token symbol, change as needed
# result = spot_price_and_hoe_ass_spot_symbol(symbol)

# mid_px, hoe_ass_symbol, sz_decimals, px_decimals = result
# print(f"The mid price for {symbol} is: {mid_px}")
# print(f"The hoe_ass_symbol for {symbol} is: {hoe_ass_symbol}")
# print(f"Size decimals for {symbol}: {sz_decimals}")
# print(f"Price decimals for {symbol}: {px_decimals}")



def spot_limit_order(coin, is_buy, sz, limit_px, account, sz_decimals, px_decimals):
    #account: LocalAccount = eth_account.Account.from_key(key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    # info = Info(constants.MAINNET_API_URL, skip_ws=True)
    # user_state = info.user_state(account.address)
    # print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')

    sz = round(sz,sz_decimals)
    limit_px = float(limit_px)
    limit_px = round(limit_px,px_decimals)
    print(f"coin: {coin}, type: {type(coin)}")
    print(f"is_buy: {is_buy}, type: {type(is_buy)}")
    print(f"sz: {sz}, type: {type(sz)}")
    print(f"limit_px: {limit_px}, type: {type(limit_px)}")
    
    #limit_px = str(limit_px)
    # sz = str(sz)
    #print(f"limit_px: {limit_px}, type: {type(limit_px)}")
    # print(f"sz: {sz}, type: {type(sz)}")
    print(f'placing limit order for {coin} {sz} @ {limit_px}')
    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=False)

    if is_buy == True:
        print(f"limit BUY order placed thanks moon, resting: {order_result['response']['data']['statuses'][0]}")
    else:
        print(f"limit SELL order placed thanks moon, resting: {order_result['response']['data']['statuses'][0]}")

    return order_result

# time.sleep(8978)

import os 

from termcolor import colored

def all_spot_symbols():
    url = "https://api.hyperliquid.xyz/info"
    headers = {
        "Content-Type": "application/json"
    }
    body = {
        "type": "spotMetaAndAssetCtxs"
    }

    # Make the request to the API
    response = requests.post(url, headers=headers, json=body)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract tokens data
        tokens = data[0]['tokens']
        
        # Extract symbol names
        symbols = [token['name'] for token in tokens]
        
        # Get current date and time
        current_time = datetime.now().strftime("%m/%d/%Y %H:%M")
        
        # Create a new row with current time and symbols
        row_data = [current_time] + symbols

        # Convert row data to DataFrame
        new_df = pd.DataFrame([row_data])

        # Check if CSV file exists
        file_path = '/Users/md/Dropbox/dev/github/hyper-liquid-trading-bots/hlp_spot_tokens.csv'
        new_symbols = symbols
        if os.path.exists(file_path):
            try:
                # Load existing data
                existing_df = pd.read_csv(file_path, header=None)
                # Check if the new symbols are different from the last recorded symbols
                if not existing_df.empty:
                    last_row_symbols = existing_df.iloc[-1, 1:].dropna().tolist()
                    new_symbols = [sym for sym in symbols if sym not in last_row_symbols]
                    if not new_symbols:
                        print("Symbols are the same as the last recorded. No new row added.")
                        return []
                # Concatenate the new row to the existing data
                df = pd.concat([existing_df, new_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = new_df
        else:
            df = new_df
        
        # Save updated tokens to CSV
        df.to_csv(file_path, index=False, header=False)

        # Print all symbols
        print("Available symbols on the hyper liquid spot:")
        for symbol in symbols:
            print(symbol)
        
        # Return new symbols as a list
        return new_symbols

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []
    
    
def get_sz_px_decimals(symbol):

    '''
    this is succesfully returns Size decimals and Price decimals

    this outputs the size decimals for a given symbol
    which is - the SIZE you can buy or sell at
    ex. if sz decimal == 1 then you can buy/sell 1.4
    if sz decimal == 2 then you can buy/sell 1.45
    if sz decimal == 3 then you can buy/sell 1.456

    if size isnt right, we get this error. to avoid it use the sz decimal func
    {'error': 'Invalid order size'}
    '''
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        # Success
        data = response.json()
        #print(data)
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
            
        else:
            print('Symbol not found')
    else:
        # Error
        print('Error:', response.status_code)

    ask = ask_bid(symbol)[0]
    # print(f'this is the ask {ask}')

    # Compute the number of decimal points in the ask price
    ask_str = str(ask)
    print(f'this is the ask str {ask_str}')
    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0

    print(f'{symbol} this is the price: {ask}  sz decimal(s) {sz_decimals}, px decimal(s) {px_decimals}')

    return sz_decimals, px_decimals

def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    #account: LocalAccount = eth_account.Account.from_key(key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    # info = Info(constants.MAINNET_API_URL, skip_ws=True)
    # user_state = info.user_state(account.address)
    # print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    
    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz,rounding)
    print(f"coin: {coin}, type: {type(coin)}")
    print(f"is_buy: {is_buy}, type: {type(is_buy)}")
    print(f"sz: {sz}, type: {type(sz)}")
    print(f"limit_px: {limit_px}, type: {type(limit_px)}")
    print(f"reduce_only: {reduce_only}, type: {type(reduce_only)}")


    #limit_px = str(limit_px)
    # sz = str(sz)
    #print(f"limit_px: {limit_px}, type: {type(limit_px)}")
    # print(f"sz: {sz}, type: {type(sz)}")
    print(f'placing limit order for {coin} {sz} @ {limit_px}')
    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=reduce_only)

    if is_buy == True:
        print(f"limit BUY order placed thanks moon, resting: {order_result['response']['data']['statuses'][0]}")
    else:
        print(f"limit SELL order placed thanks moon, resting: {order_result['response']['data']['statuses'][0]}")

    return order_result




def adjust_leverage_size_signal(symbol, leverage, account):

        '''
        this calculates size based off what we want.
        95% of balance
        '''

        print('leverage:', leverage)

        #account: LocalAccount = eth_account.Account.from_key(key)
        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get the user state and print out leverage information for ETH
        user_state = info.user_state(account.address)
        acct_value = user_state["marginSummary"]["accountValue"]
        acct_value = float(acct_value)
        #print(acct_value)
        acct_val95 = acct_value * .95

        print(exchange.update_leverage(leverage, symbol))

        price = ask_bid(symbol)[0]

        # size == balance / price * leverage
        # INJ 6.95 ... at 10x lev... 10 INJ == $cost 6.95
        size = (acct_val95 / price) * leverage
        size = float(size)
        rounding = get_sz_px_decimals(symbol)[0]
        size = round(size, rounding)
        #print(f'this is the size we can use 95% fo acct val {size}')
    
        user_state = info.user_state(account.address)
            
        return leverage, size

def adjust_leverage_usd_size(symbol, usd_size, leverage, account):

        '''
        this calculates size based off a specific USD dollar amount
        '''

        print('leverage:', leverage)

        #account: LocalAccount = eth_account.Account.from_key(key)
        exchange = Exchange(account, constants.MAINNET_API_URL)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        # Get the user state and print out leverage information for ETH
        user_state = info.user_state(account.address)
        acct_value = user_state["marginSummary"]["accountValue"]
        acct_value = float(acct_value)

        print(exchange.update_leverage(leverage, symbol))

        price = ask_bid(symbol)[0]

        # size == balance / price * leverage
        # INJ 6.95 ... at 10x lev... 10 INJ == $cost 6.95
        size = (usd_size / price) * leverage
        size = float(size)
        rounding = get_sz_px_decimals(symbol)[0]
        size = round(size, rounding)
        print(f'this is the size of crypto we will be using {size}')
    
        user_state = info.user_state(account.address)
            
        return leverage, size

def adjust_leverage(symbol, leverage, private_key):
    account = LocalAccount = eth_account.Account.from_key(private_key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    print('leverage:', leverage)

    exchange.update_leverage(leverage, symbol)

def get_ohclv(cb_symbol, timeframe, limit):

    coinbase = ccxt.kraken()

    ohlcv = coinbase.fetch_ohlcv(cb_symbol, timeframe, limit)
    #print(ohlcv)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    df = df.tail(limit)

    df['support'] = df[:-2]['close'].min()
    df['resis'] = df[:-2]['close'].max()

    # Save the dataframe to a CSV file
    df.to_csv('ohlcv_data.csv', index=False)

    return df 

def get_ohlcv2(symbol, interval, lookback_days):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
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
        return snapshot_data
    else:
        print(f"Error fetching data for {symbol}: {response.status_code}")
        return None

def process_data_to_df(snapshot_data):
    if snapshot_data:
        # Assuming the response contains a list of candles
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        for snapshot in snapshot_data:
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            open_price = snapshot['o']
            high_price = snapshot['h']
            low_price = snapshot['l']
            close_price = snapshot['c']
            volume = snapshot['v']
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=columns)

        # Calculate support and resistance, excluding the last two rows for the calculation
        if len(df) > 2:  # Check if DataFrame has more than 2 rows to avoid errors
            df['support'] = df[:-2]['close'].min()
            df['resis'] = df[:-2]['close'].max()
        else:  # If DataFrame has 2 or fewer rows, use the available 'close' prices for calculation
            df['support'] = df['close'].min()
            df['resis'] = df['close'].max()

        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data
    

# TODO-  NOT DELETING ABOVE TIL BELOW IS CONFIRMED WORKING

def process_data_to_df(snapshot_data, time_period=20):
    if snapshot_data:
        # Assuming the response contains a list of candles
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data = []
        for snapshot in snapshot_data:
            timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
            open_price = snapshot['o']
            high_price = snapshot['h']
            low_price = snapshot['l']
            close_price = snapshot['c']
            volume = snapshot['v']
            data.append([timestamp, open_price, high_price, low_price, close_price, volume])

        df = pd.DataFrame(data, columns=columns)

        # Calculate rolling support and resistance
        df['support'] = df['close'].rolling(window=time_period, min_periods=1).min().shift(1)
        df['resis'] = df['close'].rolling(window=time_period, min_periods=1).max().shift(1)

        return df
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data



def calculate_vwap_with_symbol(symbol):
    # Fetch and process data
    snapshot_data = get_ohlcv2(symbol, '15m', 300)
    df = process_data_to_df(snapshot_data)

    # Convert the 'timestamp' column to datetime and set as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Ensure all columns used for VWAP calculation are of numeric type
    numeric_columns = ['high', 'low', 'close', 'volume']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # 'coerce' will set errors to NaN

    # Drop rows with NaNs created during type conversion (if any)
    df.dropna(subset=numeric_columns, inplace=True)

    # Ensure the DataFrame is ordered by datetime
    df.sort_index(inplace=True)

    # Calculate VWAP and add it as a new column
    df['VWAP'] = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])

    # Retrieve the latest VWAP value from the DataFrame
    latest_vwap = df['VWAP'].iloc[-1]

    return df, latest_vwap

def supply_demand_zones_hl(symbol, timeframe, limit):

    print('starting moons supply and demand zone calculations..')

    sd_df = pd.DataFrame()

    snapshot_data = get_ohlcv2(symbol, timeframe, limit)
    df = process_data_to_df(snapshot_data)

    
    supp = df.iloc[-1]['support']
    resis = df.iloc[-1]['resis']
    #print(f'this is moons support for 1h {supp_1h} this is resis: {resis_1h}')

    df['supp_lo'] = df[:-2]['low'].min()
    supp_lo = df.iloc[-1]['supp_lo']

    df['res_hi'] = df[:-2]['high'].max()
    res_hi = df.iloc[-1]['res_hi']

    print(df)


    sd_df[f'{timeframe}_dz'] = [supp_lo, supp]
    sd_df[f'{timeframe}_sz'] = [res_hi, resis]

    print('here are moons supply and demand zones')
    print(sd_df)

    return sd_df 


def get_spot_position(symbol, account):
    '''
    gets the current position info, like size etc.
    '''

    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.spot_user_state(account.address)
    #print(user_state)
    positions = []
    print(f'This is the symbol: {symbol}')
    # print(user_state["balances"])
    # time.sleep(787)
    
    for balance in user_state["balances"]:
        if balance["coin"] == symbol:
            size = float(balance["total"])
            pos_sym = balance["coin"]
            entry_px = float(balance["entryNtl"]) / size if size != 0 else 0
            pnl_perc = 0  # Assuming we don't have returnOnEquity in the provided data
            positions.append(balance)
            in_pos = size > 1
            long = size > 1
            # print(f'This is the pnl perc: {pnl_perc}')
            break
    else:
        in_pos = False
        size = 0
        pos_sym = None
        entry_px = 0
        pnl_perc = 0
        long = None

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long


def get_position(symbol, account):

    '''
    gets the current position info, like size etc. 
    '''

    # account = LocalAccount = eth_account.Account.from_key(key)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    positions = []
    print(f'this is the symbol {symbol}')
    print(user_state["assetPositions"])
    for position in user_state["assetPositions"]:
        if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True 
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"])*100
            print(f'this is the pnl perc {pnl_perc}')
            break 
    else:
        in_pos = False 
        size = 0 
        pos_sym = None 
        entry_px = 0 
        pnl_perc = 0

    if size > 0:
        long = True 
    elif size < 0:
        long = False 
    else:
        long = None 

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long

def get_position_andmaxpos(symbol, account, max_positions):

    '''
    gets the current position info, like size etc. 
    '''

    # account = LocalAccount = eth_account.Account.from_key(key)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    positions = []
    open_positions = []
    # print(f'this is the symbol {symbol}')
    # print(user_state["assetPositions"])

    #CHECKING MAXX POSITIONS FIRST
    # Iterate over each position in the assetPositions list
    for position in user_state["assetPositions"]:
        # Check if the position size ('szi') is not zero, indicating an open position
        if float(position["position"]["szi"]) != 0:
            # If it's an open position, add the coin symbol to the open_positions list
            open_positions.append(position["position"]["coin"])

    # print(open_positions)
    num_of_pos = len(open_positions)
    #print(f'we are in {len(positions)} positions and max pos is {max_pos}... closing positions')

    if len(open_positions) > max_positions:

        print(f'we are in {len(open_positions)} positions and max pos is {max_positions}... closing positions')
        
        # for position in positions we need to call the kill switch 
        for position in open_positions:
            kill_switch(position, account)

    else:
        print(f'we are in {len(open_positions)} positions and max pos is {max_positions}... not closing positions')


    for position in user_state["assetPositions"]:
        if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True 
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"])*100
            print(f'this is the pnl perc {pnl_perc}')
            break 
    else:
        in_pos = False 
        size = 0 
        pos_sym = None 
        entry_px = 0 
        pnl_perc = 0

    if size > 0:
        long = True 
    elif size < 0:
        long = False 
    else:
        long = None 

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long, num_of_pos

def cancel_all_orders(account):
    # this cancels all open orders
    #account = LocalAccount = eth_account.Account.from_key(key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    open_orders = info.open_orders(account.address)
    #print(open_orders)

    print('above are the open orders... need to cancel any...')
    for open_order in open_orders:
        #print(f'cancelling order {open_order}')
        exchange.cancel(open_order['coin'], open_order['oid'])

def cancel_symbol_orders(account, symbol):
    """
    Cancels all open orders for the specified symbol.
    
    Parameters:
    - account: The trading account
    - symbol: The symbol (coin) for which to cancel open orders
    """
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    open_orders = info.open_orders(account.address)
    print(open_orders)
    

    print('Above are the open orders... need to cancel any...')
    for open_order in open_orders:
        if open_order['coin'] == symbol:
            print(f'Cancelling order {open_order}')
            exchange.cancel(open_order['coin'], open_order['oid'])

def volume_spike(df):
    # A volume spike would be significantly larger than the current moving average of volume
    df['MA_Volume'] = df['volume'].rolling(window=20).mean()

    # A downward trend can be seen when the current close price is below the moving average of close price
    df['MA_Close'] = df['close'].rolling(window=20).mean()
    # print(df['MA_Volume'])
    # print(df['MA_Close'])

    latest_data = df.iloc[-1]
    volume_spike_and_price_downtrend = latest_data['volume'] > vol_multiplier * latest_data['MA_Volume'] and latest_data['MA_Close'] > latest_data['close']

    return volume_spike_and_price_downtrend

def spot_kill_switch(sym, hoe_ass_symbol, account, market_order=False):
    positions, in_pos, size, pos_sym, entry_px, pnl_perc, long = get_spot_position(sym, account)
    
    print(f'inpos {in_pos} size {size} symbol {pos_sym} long {long}' )

    cancel_symbol_orders(account, hoe_ass_symbol)

    while in_pos == True:
        print('while in pos line...')
        cancel_symbol_orders(account, hoe_ass_symbol)
        print('just canceled orders line...')
        mid_px, hoe_ass_symbol, sz_decimals, px_decimals = spot_price_and_hoe_ass_spot_symbol(sym)
        print('bunch of stuff line...')
        
        # Debug prints
        print(f'Debug - Original size: {size}')
        
        # Just floor to integer
        adjusted_size = int(size)
        print(f'Debug - Using integer size: {adjusted_size}')
        
        if long == True:
            if market_order == True:
                print('MARKET ORDER CLOSING')
                mid_px = float(mid_px)
                market_price = mid_px * .96
                market_price = str(market_price)
                spot_limit_order(hoe_ass_symbol, False, adjusted_size, market_price, account, 0, px_decimals)
            else:
                print(f'closing at mid price limit order...')
                spot_limit_order(hoe_ass_symbol, False, adjusted_size, mid_px, account, 0, px_decimals)
        
        # Check position again
        positions, in_pos, size, pos_sym, entry_px, pnl_perc, long = get_spot_position(sym, account)
def kill_switch(symbol, account):

    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    while im_in_pos == True:

        cancel_all_orders(account)

        # get bid_ask
        askbid = ask_bid(pos_sym)
        ask = askbid[0]
        bid = askbid[1]

        pos_size = abs(pos_size)

        if long == True:
            limit_order(pos_sym, False, pos_size, ask, True, account)
            print('kill switch - SELL TO CLOSE SUBMITTED ')
            time.sleep(5)
        elif long == False:
            limit_order(pos_sym, True, pos_size, bid, True, account)
            print('kill switch - BUY TO CLOSE SUBMITTED ')
            time.sleep(5)
        
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    print('position successfully closed in kill switch')

def kill_switch_mkt(symbol, account):
    """
    Market close positions using 5th bid/ask from L2 data for more aggressive fills
    🌙 Moon Dev's turbocharged kill switch 🚀
    """
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    while im_in_pos == True:
        cancel_all_orders(account)

        # Get L2 data and extract 5th bid/ask
        askbid = ask_bid(pos_sym)
        l2_data = askbid[2]
        
        # Extract 5th bid and ask if available
        try:
            fifth_bid = float(l2_data[0][4]['px'])  # 5th bid
            fifth_ask = float(l2_data[1][4]['px'])  # 5th ask
            print(f"🎯 Using 5th bid: {fifth_bid} and 5th ask: {fifth_ask}")
        except (IndexError, KeyError):
            print("⚠️ Couldn't get 5th bid/ask, using best bid/ask instead")
            fifth_bid = float(l2_data[0][0]['px'])  # Best bid
            fifth_ask = float(l2_data[1][0]['px'])  # Best ask

        pos_size = abs(pos_size)

        if long == True:
            limit_order(pos_sym, False, pos_size, fifth_bid, True, account)
            print('🌙 Kill switch MKT - SELL TO CLOSE SUBMITTED at 5th bid')
            time.sleep(5)
        elif long == False:
            limit_order(pos_sym, True, pos_size, fifth_ask, True, account)
            print('🌙 Kill switch MKT - BUY TO CLOSE SUBMITTED at 5th ask')
            time.sleep(5)
        
        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    print('✨ Position successfully closed in market kill switch - Thanks Moon Dev! ✨')


def pnl_close(symbol, target, max_loss, account):

    print('entering pnl close')
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    if pnl_perc > target:
        print(f'pnl gain is {pnl_perc} and target is {target}... closing position WIN')
        kill_switch(pos_sym, account)
    elif pnl_perc <= max_loss:
        print(f'pnl loss is {pnl_perc} and max loss is {max_loss}... closing position LOSS')
        kill_switch(pos_sym, account)
    else:
        print(f'pnl loss is {pnl_perc} and max loss is {max_loss} and target {target}... not closing position')
    print('finished with pnl close')



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
        return snapshot_data
    else:
        print(f"Error fetching data for {symbol}: {response.status_code}")
        return None


def calculate_sma(prices, window):
    sma = prices.rolling(window=window).mean()
    return sma.iloc[-1]  # Return the most recent SMA value

def get_latest_sma(symbol, interval, window, lookback_days=1):
    start_time = datetime.now() - timedelta(days=lookback_days)
    end_time = datetime.now()

    snapshots = fetch_candle_snapshot(symbol, interval, start_time, end_time)
    if snapshots:
        prices = pd.Series([float(snapshot['c']) for snapshot in snapshots])
        latest_sma = calculate_sma(prices, window)
        return latest_sma
    else:
        return None

def close_all_positions(account):

    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    positions = []
    open_positions = []
    print(f'this is the symbol {symbol}')
    print(user_state["assetPositions"])

    # cancel all orders
    cancel_all_orders(account)
    print('all orders have been cancelled')

    #CHECKING MAXX POSITIONS FIRST
    # Iterate over each position in the assetPositions list
    for position in user_state["assetPositions"]:
        # Check if the position size ('szi') is not zero, indicating an open position
        if float(position["position"]["szi"]) != 0:
            # If it's an open position, add the coin symbol to the open_positions list
            open_positions.append(position["position"]["coin"])

    
    # for position in positions we need to call the kill switch 
    for position in open_positions:
        kill_switch(position, account)

    print('all positions have been closed')

    # code me 


def close_all_positions_mkt(account):

    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    positions = []
    open_positions = []
    print(f'this is the symbol {symbol}')
    print(user_state["assetPositions"])

    # cancel all orders
    cancel_all_orders(account)
    print('all orders have been cancelled')

    #CHECKING MAXX POSITIONS FIRST
    # Iterate over each position in the assetPositions list
    for position in user_state["assetPositions"]:
        # Check if the position size ('szi') is not zero, indicating an open position
        if float(position["position"]["szi"]) != 0:
            # If it's an open position, add the coin symbol to the open_positions list
            open_positions.append(position["position"]["coin"])

    
    # for position in positions we need to call the kill switch 
    for position in open_positions:
        kill_switch_mkt(position, account)

    print('all positions have been closed')

    # code me 


def calculate_atr(df, window):
    """
    Calculate the Average True Range (ATR) for a given DataFrame and window period.

    Parameters:
    - df: DataFrame containing 'High', 'Low', and 'Close' price columns.
    - window: The period over which the ATR is calculated. Default is 14.

    Returns:
    - DataFrame with an additional 'ATR' column representing the calculated ATR values.
    """

    
     # Ensure 'high', 'low', and 'close' are numeric
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Calculate the true ranges
    df['High-Low'] = df['high'] - df['low']
    df['High-PrevClose'] = abs(df['high'] - df['close'].shift())
    df['Low-PrevClose'] = abs(df['low'] - df['close'].shift())  # Corrected 'how' to 'low'

    # The true range is the max of the above three calculations
    df['TrueRange'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Calculate the ATR by taking the moving average of the true ranges
    df['ATR'] = df['TrueRange'].rolling(window=window, min_periods=1).mean()

        # Get the last ATR value
    last_atr = df['ATR'].iloc[-1]

    # Drop the intermediate columns to clean up the DataFrame
    #df.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose', 'TrueRange'], axis=1, inplace=True)

    return df, last_atr

def calculate_range(df, window):
    """
    Calculate the range between the highest high and the lowest low for the last 'window' rows of a given DataFrame.

    Parameters:
    - df: DataFrame containing 'High' and 'Low' price columns.
    - window: The number of rows from the end to include in the range calculation.

    Returns:
    - The range as a float value.
    """
      # Ensure the window is not greater than the DataFrame's length
    window = min(window, len(df))

    # Work on a copy of the DataFrame to avoid SettingWithCopyWarning
    recent_df = df[-window:].copy()

    # Convert 'high' and 'low' to numeric, coercing errors to NaN
    recent_df['high'] = pd.to_numeric(recent_df['high'], errors='coerce')
    recent_df['low'] = pd.to_numeric(recent_df['low'], errors='coerce')

    # Drop any rows with NaN values that resulted from coercion
    recent_df.dropna(subset=['high', 'low'], inplace=True)

    # Calculate the range
    highest_high = recent_df['high'].max()
    lowest_low = recent_df['low'].min()
    price_range = highest_high - lowest_low

    return price_range


def should_we_quote_orders():

    '''
    # BUILD FUNCTION THAT CHECKS TO SEE IF WE SHOULD QUOTE ORDERS OR NOT
    # return True if we should quote, false if not

    Market makers play a crucial role in providing liquidity to financial markets by continuously quoting buy and sell prices for securities. They stand ready to buy or sell at these publicly quoted prices. However, market makers may decide to cancel orders based on various strategic and operational considerations. Here are some factors and strategies, ranked by their significance and commonality, that market makers might use to decide whether to cancel orders:

    Market Conditions and Volatility: High-quality decisions often involve adjusting to current market conditions. In periods of high volatility, market makers might cancel orders to avoid being picked off by faster participants who act on news or market movements more quickly. They might also adjust their quotes to manage risk more effectively during these times.

    Order Flow Imbalance: Market makers closely monitor the balance between buy and sell orders. An imbalance might prompt a market maker to cancel orders to avoid excessive exposure on one side of the market. They aim to maintain a balanced book to mitigate risk.

    Price Movements in Related Securities: Changes in the prices of related securities, such as those within the same sector or index, or correlated assets like futures or options, can lead market makers to adjust their orders. If a related security moves significantly, it might signal an impending move in the securities they are making a market in, prompting them to cancel orders to reevaluate their pricing.

    Liquidity Levels: Market makers might cancel orders in response to changes in liquidity. During periods of low liquidity, they might widen their spreads and reduce the size of their quotes or cancel some orders to manage risk.

    Risk Management: Managing inventory risk is crucial for market makers. They might cancel orders if holding a position becomes too risky or if it exceeds their risk tolerance levels. This can be due to a variety of factors, including unexpected market moves, approaching news events, or simply the accumulation of positions that skew their book too far in one direction.

    Regulatory and Operational Constraints: Sometimes, orders are canceled due to regulatory requirements, capital constraints, or operational issues. For example, a market maker might reduce or cancel orders to comply with position limits or to manage capital requirements.

    Technological Factors: Latency issues or the anticipation of faster market participants acting on new information might lead to order cancellations. Market makers aim to avoid being "picked off" by competitors with faster access to information or execution capabilities.

    Strategic Order Placement: Market makers might cancel orders as part of a strategy to bait or flush out other market participants. For instance, placing and then canceling large orders might be used to give the illusion of liquidity or interest at certain price levels, influencing other traders' behavior.

    Economic News and Events: Anticipation of or reaction to economic news, earnings reports, geopolitical events, or central bank announcements can lead to order cancellations. Market makers might preemptively cancel orders before such events to avoid being caught on the wrong side of a sudden market move.

    End-of-Day Position Squaring: Towards the close of the trading session, market makers might cancel outstanding orders to square their positions and limit overnight exposure, especially if they aim to operate on a near-flat book.
    
    VOLUME 
    '''

    # How do we determine when to quote or not?
    print('checking if we should quote orders')

    # CHECK SMA BTC TO MAKE SURE NOT TO VOL

    # CHECK THE PRICE CHANGE ON BTC NOT TO VOL

    # CHECK THE ORDERBOOK SKEEW

    # CHECK OUR PORTOFLIO FOR BALANCE

    # GET THE ATR OF BTC
    snapshot_data = get_ohlcv2('BTC', '1m', 500)
    df = process_data_to_df(snapshot_data)
    
    price_range = calculate_range(df, 240)# 240 mins == 4 hours.. 

    print(f'this is the Range: {price_range} in the last 4 hours')

    # IF MAX RANGE > 500 
    if price_range > 500:

        quote_orders = False

    else: 
        quote_orders = True

    return quote_orders

def calculate_bollinger_bands(df, length=20, std_dev=2):
    """
    Calculate Bollinger Bands for a given DataFrame and classify when the bands are tight vs wide.

    Parameters:
    - df: DataFrame with a 'close' column.
    - length: The period over which the SMA is calculated. Default is 20.
    - std_dev: The number of standard deviations to plot above and below the SMA. Default is 2.

    Returns:
    - df: DataFrame with Bollinger Bands and classifications for 'tight' and 'wide' bands.
    - tight: Boolean indicating if the bands are currently tight.
    - wide: Boolean indicating if the bands are currently wide.
    """

    # Ensure 'close' is numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Calculate Bollinger Bands using pandas_ta
    bollinger_bands = ta.bbands(df['close'], length=length, std=std_dev)

    # Select only the Bollinger Bands columns (ignoring additional columns like bandwidth and percent bandwidth)
    bollinger_bands = bollinger_bands.iloc[:, [0, 1, 2]]  # Assuming the first 3 columns are BBL, BBM, and BBU
    bollinger_bands.columns = ['BBL', 'BBM', 'BBU']

    # Merge the Bollinger Bands into the original DataFrame
    df = pd.concat([df, bollinger_bands], axis=1)

    # Calculate Band Width
    df['BandWidth'] = df['BBU'] - df['BBL']

    # Determine thresholds for 'tight' and 'wide' bands
    tight_threshold = df['BandWidth'].quantile(0.2)
    wide_threshold = df['BandWidth'].quantile(0.8)

    # Classify the current state of the bands
    current_band_width = df['BandWidth'].iloc[-1]
    tight = current_band_width <= tight_threshold
    wide = current_band_width >= wide_threshold

    return df, tight, wide


# DO NOT USE TINY STOP LOSSES AND TAKE PROFITS

def calculate_linear_regression_channel(df, length=20, proximity_threshold=0.02):
    """
    Calculate Linear Regression Channel for a given DataFrame and determine when to quote orders.

    Parameters:
    - df: DataFrame with 'high', 'low', and 'close' columns.
    - length: The period over which the Linear Regression Channel is calculated. Default is 20.
    - proximity_threshold: Proximity to the Middle Line expressed as a percentage of the channel width. Default is 2%.

    Returns:
    - df: DataFrame with Linear Regression Channel added.
    - quote_buy_orders: Boolean indicating if conditions are favorable for quoting buy orders.
    - quote_sell_orders: Boolean indicating if conditions are favorable for quoting sell orders.
    - quote_both_orders: Boolean indicating if conditions are favorable for quoting both buy and sell orders.
    """
    # Ensure 'close' is numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # Calculate Linear Regression using pandas_ta
    linreg_values = ta.linreg(df['close'], length=length)
    df['LRCM_'] = linreg_values  # Middle Line

    # Calculate the channel width based on standard deviation
    channel_width = df['close'].rolling(window=length).std()

    # Add the channel lines to the DataFrame
    df['LRCT_'] = df['LRCM_'] + channel_width * 2  # Top Channel, 2 standard deviations
    df['LRCB_'] = df['LRCM_'] - channel_width * 2  # Bottom Channel, 2 standard deviations

    # Determine the slope of the Linear Regression Channel
    slope = (df['LRCM_'].iloc[-1] - df['LRCM_'].iloc[-length]) / length

    # Determine conditions for quoting orders
    quote_buy_orders = slope > 0 and df['close'].iloc[-1] < df['LRCT_'].iloc[-1]
    quote_sell_orders = slope < 0 and df['close'].iloc[-1] > df['LRCB_'].iloc[-1]

    # Determine if the price is near the Middle Line
    proximity_to_middle = abs(df['close'].iloc[-1] - df['LRCM_'].iloc[-1])
    is_near_middle = proximity_to_middle <= (channel_width.iloc[-1] * 2 * proximity_threshold)

    # Quote both orders if the price is near the Middle Line
    quote_both_orders = is_near_middle

    return df, quote_buy_orders, quote_sell_orders, quote_both_orders


def linear_regression_bollinger(df, bb_length=20, bb_std_dev=2, lrc_length=20, proximity_threshold=0.02):
    """
    Determine when to quote buy, sell, or both orders based on Bollinger Bands and Linear Regression Channel.

    Parameters:
    - df: DataFrame with 'high', 'low', and 'close' columns.
    - bb_length: Period for Bollinger Bands SMA calculation. Default is 20.
    - bb_std_dev: Standard deviations for Bollinger Bands. Default is 2.
    - lrc_length: Period for Linear Regression Channel calculation. Default is 20.
    - proximity_threshold: Proximity to LRC Middle Line as a percentage of the channel width. Default is 2%.

    Returns:
    - df: DataFrame with Bollinger Bands and Linear Regression Channel added.
    - quote_buy_orders: Boolean indicating if conditions are favorable for quoting buy orders.
    - quote_sell_orders: Boolean indicating if conditions are favorable for quoting sell orders.
    - quote_both_orders: Boolean indicating if conditions are favorable for quoting both buy and sell orders.
    """

    # Calculate Bollinger Bands
    df, tight, wide = calculate_bollinger_bands(df, length=bb_length, std_dev=bb_std_dev)

    # Calculate Linear Regression Channel and determine quoting conditions
    df, lrc_quote_buy, lrc_quote_sell, lrc_quote_both = calculate_linear_regression_channel(df, length=lrc_length, proximity_threshold=proximity_threshold)

    # Integrate decisions: Consider market conditions both from Bollinger Bands and Linear Regression Channel
    quote_buy_orders = lrc_quote_buy and not wide  # Favor buy orders in an uptrend and not in wide Bollinger Bands
    quote_sell_orders = lrc_quote_sell and not wide  # Favor sell orders in a downtrend and not in wide Bollinger Bands
    quote_both_orders = lrc_quote_both or tight  # Quote both in a sideways market (near LRC middle line or tight Bollinger Bands)

    return df, quote_buy_orders, quote_sell_orders, quote_both_orders

# snapshot_data = get_ohlcv2('WIF', '15m', 50000)
# df = process_data_to_df(snapshot_data)
# # save to csv
# df.to_csv('ohlcv_data.csv', index=False)

# df = calculate_linear_regression_channel(df)
# print(df)


# DEPRECATEDDDDDDD
# def ob_data(symbol):
#     ''' This function gets the OB data from Binance, Bybit, and Coinbase, limits Coinbase 
#     to the newest 100 bids and asks, calculates average prices for specified positions, 
#     outputs the combined order books, the prices with the biggest sizes for each exchange, 
#     and the overall biggest bid and ask sizes across all exchanges 
    
#     add later - OKX because its the 2nd biggest exchange
#     '''

#     # Initialize clients for Binance, Bybit, and Coinbase
#     binance = ccxt.binance({'enableRateLimit': True})
#     bybit = ccxt.bybit({'enableRateLimit': True})
#     coinbasepro = ccxt.coinbasepro({'enableRateLimit': True})
#     # # do same for okx
#     # okx = ccxt.okex({'enableRateLimit': True})

#     # Symbols for each exchange
#     binance_sym = symbol + '/USDT'
#     bybit_sym = symbol + 'USDT'  # Adjust if Bybit uses a different format
#     coinbase_sym = symbol + '-USD'  # Coinbase generally uses 'BTC-USD'

#     # Fetch order books
#     ob_binance = binance.fetch_order_book(binance_sym)
#     ob_bybit = bybit.fetch_order_book(bybit_sym)
#     ob_coinbase = coinbasepro.fetch_order_book(coinbase_sym)


#     # Function to create combined DataFrame from order book
#     def create_combined_df(ob, exchange):
#         bids_df = pd.DataFrame(ob['bids'], columns=['Bid', 'Bid Size'])
#         asks_df = pd.DataFrame(ob['asks'], columns=['Ask', 'Ask Size'])
#         # Limit Coinbase to the newest 100 bids and asks
#         if exchange == 'coinbasepro' and len(bids_df) > 100:
#             bids_df = bids_df.head(100)
#             asks_df = asks_df.head(100)
#         return pd.concat([asks_df.reset_index(drop=True), bids_df.reset_index(drop=True)], axis=1)

#     # Function to find max size prices
#     def find_max_sizes(df):
#         max_bid_size_row = df.loc[df['Bid Size'].idxmax()]
#         max_ask_size_row = df.loc[df['Ask Size'].idxmax()]
#         return max_bid_size_row['Bid'], max_bid_size_row['Bid Size'], max_ask_size_row['Ask'], max_ask_size_row['Ask Size']

#     # Create combined DataFrames
#     combined_df_binance = create_combined_df(ob_binance, 'binance')
#     combined_df_bybit = create_combined_df(ob_bybit, 'bybit')
#     combined_df_coinbase = create_combined_df(ob_coinbase, 'coinbasepro')

#     # Track max sizes across exchanges
#     max_bid_size = max_ask_size = 0
#     max_bid_price = max_ask_price = None

#     # Find and print max size prices for each exchange and track overall max
#     for name, df in zip(['Binance', 'Bybit', 'Coinbase'], [combined_df_binance, combined_df_bybit, combined_df_coinbase]):
#         bid_price, bid_size, ask_price, ask_size = find_max_sizes(df)
#         print(f"\n{name} Max Bid Size Price: {bid_price} with Size: {bid_size}")
#         print(f"{name} Max Ask Size Price: {ask_price} with Size: {ask_size}")

#         if bid_size > max_bid_size:
#             max_bid_size, max_bid_price = bid_size, bid_price
#         if ask_size > max_ask_size:
#             max_ask_size, max_ask_price = ask_size, ask_price

#     # Print overall max sizes across all exchanges
#     print("\nOverall Max Bid Size across all exchanges:")
#     print(f"Price: {max_bid_price}, Size: {max_bid_size}")
#     print("\nOverall Max Ask Size across all exchanges:")
#     print(f"Price: {max_ask_price}, Size: {max_ask_size}")

#     return combined_df_binance, combined_df_bybit, combined_df_coinbase, max_bid_price, max_ask_price

# # Call the function and capture the returned values
# combined_df_binance, combined_df_bybit, combined_df_coinbase, max_bid_price, max_ask_price = ob_data('SOL')

import ccxt
import pandas as pd

def ob_data(symbol):
    ''' This function gets the OB data from Binance, Bybit, and Coinbase, limits Coinbase 
    to the newest 100 bids and asks, calculates average prices for specified positions, 
    outputs the combined order books, the prices with the biggest sizes for each exchange, 
    and the overall biggest bid and ask sizes across all exchanges. 
    Additionally, it finds the bid and ask right before the biggest ones.
    '''

    binance = ccxt.binance({'enableRateLimit': True})
    bybit = ccxt.bybit({'enableRateLimit': True})
    coinbasepro = ccxt.coinbasepro({'enableRateLimit': True})

    binance_sym = symbol + '/USDT'
    bybit_sym = symbol + 'USDT'
    coinbase_sym = symbol + '-USD'

    ob_binance = binance.fetch_order_book(binance_sym)
    ob_bybit = bybit.fetch_order_book(bybit_sym)
    ob_coinbase = coinbasepro.fetch_order_book(coinbase_sym)

    def create_combined_df(ob, exchange):
        bids_df = pd.DataFrame(ob['bids'], columns=['Bid', 'Bid Size'])
        asks_df = pd.DataFrame(ob['asks'], columns=['Ask', 'Ask Size'])
        if exchange == 'coinbasepro' and len(bids_df) > 100:
            bids_df = bids_df.head(100)
            asks_df = asks_df.head(100)
        return pd.concat([asks_df.reset_index(drop=True), bids_df.reset_index(drop=True)], axis=1)

    def find_max_sizes(df):
        max_bid_size_row = df.loc[df['Bid Size'].idxmax()]
        max_ask_size_row = df.loc[df['Ask Size'].idxmax()]
        return max_bid_size_row['Bid'], max_bid_size_row['Bid Size'], max_ask_size_row['Ask'], max_ask_size_row['Ask Size']

    def find_before_biggest(df, max_price, col_name, is_bid=True):
        #print(f'finding before biggest for {col_name}')
        #print(df)
        if is_bid:
            sorted_df = df.sort_values(by=col_name, ascending=True)
            before_biggest_df = sorted_df[sorted_df[col_name] > max_price]
        else:
            sorted_df = df.sort_values(by=col_name, ascending=False)
            before_biggest_df = sorted_df[sorted_df[col_name] < max_price]

        #print(f'sorted df: {sorted_df}')

        if before_biggest_df.empty:
            return None, None
        else:
            before_biggest_row = before_biggest_df.iloc[0]
            return before_biggest_row[col_name], before_biggest_row[col_name + ' Size']

        
    combined_df_binance = create_combined_df(ob_binance, 'binance')
    combined_df_bybit = create_combined_df(ob_bybit, 'bybit')
    combined_df_coinbase = create_combined_df(ob_coinbase, 'coinbasepro')

    #

    max_bid_size = max_ask_size = 0
    max_bid_price = max_ask_price = None

    for df in [combined_df_binance, combined_df_bybit, combined_df_coinbase]:
        bid_price, bid_size, ask_price, ask_size = find_max_sizes(df)
        if bid_size > max_bid_size:
            max_bid_size, max_bid_price = bid_size, bid_price
        if ask_size > max_ask_size:
            max_ask_size, max_ask_price = ask_size, ask_price

    bid_before_biggest = ask_before_biggest = (None, None)

    if max_bid_price:
        bid_before, bid_size_before = find_before_biggest(df, max_bid_price, 'Bid', is_bid=True)
        if bid_before is not None:
            bid_before_biggest = bid_before

    if max_ask_price:
        ask_before, ask_size_before = find_before_biggest(df, max_ask_price, 'Ask', is_bid=False)
        if ask_before is not None:
            ask_before_biggest = ask_before

    
    print("\nOverall Max Bid Size across all exchanges:")
    print(f"Price: {max_bid_price}, Size: {max_bid_size}")
    print("\nOverall Max Ask Size across all exchanges:")
    print(f"Price: {max_ask_price}, Size: {max_ask_size}")

    print("\nBid right before the biggest across all exchanges:")
    print(f"Price: {bid_before_biggest}")
    print("\nAsk right before the biggest across all exchanges:")
    print(f"Price: {ask_before_biggest}")

    return combined_df_binance, combined_df_bybit, combined_df_coinbase, max_bid_price, max_ask_price, float(bid_before_biggest), float(ask_before_biggest)

# # Call the function and capture the returned values
# combined_df_binance, combined_df_bybit, combined_df_coinbase, max_bid_price, max_ask_price, bid_before_biggest, ask_before_biggest = ob_data('BTC')



# SCANNER FOR TOP VOLUME TODAY

def open_order_deluxe_spot(symbol_info, account):
    """
    Places a limit order and sets stop loss and take profit orders.

    Parameters:
    - symbol_info: A row from a DataFrame containing symbol, entry price, stop loss, and take profit
    - tp: The take profit price
    - sl: The stop loss price
    """
    # config = utils.get_config()
    # account = eth_account.Account.from_key(config["secret_key"])

    size = symbol_info["Size"]

    print(f'opening order for {symbol_info["Symbol"]} size {size}')
    
    exchange = Exchange(account, constants.MAINNET_API_URL)
    
    symbol = symbol_info["Symbol"]
    entry_price = symbol_info["Entry Price"]

    sl = symbol_info["Stop Loss"]
    tp = symbol_info["Take Profit"]

    print(f'symbol: {symbol}, entry price: {entry_price}, stop loss: {sl}, take profit: {tp}')

    # Determine the order side (buy or sell)
    is_buy = True

    cancel_symbol_orders(account, symbol)

    print(f'enrty price: {entry_price} type{type(entry_price)}, stop loss: {sl} type{type(tp)}, take profit: {tp} type{type(tp)}')

    
    order_result = exchange.order(
        symbol,
        is_buy,
        size,  # Assuming a fixed quantity; adjust as needed
        entry_price,
        {"limit": {"tif": "Gtc"}}
    )

    # order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=False)

    
    print(f"Limit order result for {symbol}: {order_result}")

    # Place the stop loss order
    stop_order_type = {"trigger": {"triggerPx": sl, "isMarket": True, "tpsl": "sl"}}
    stop_result = exchange.order(
        symbol,
        not is_buy,
        size,  # Assuming a fixed quantity; adjust as needed
        sl,
        stop_order_type,
        reduce_only=False
    )
    print(f"Stop loss order result for {symbol}: {stop_result}")

    # Place the take profit order
    tp_order_type = {"trigger": {"triggerPx": tp, "isMarket": True, "tpsl": "tp"}}
    tp_result = exchange.order(
        symbol,
        not is_buy,
        size,  # Assuming a fixed quantity; adjust as needed
        tp,
        tp_order_type,
        reduce_only=False
    )
    print(f"Take profit order result for {symbol}: {tp_result}")

def open_order_deluxe(symbol_info, size, account):
    """
    Places a limit order and sets stop loss and take profit orders.

    Parameters:
    - symbol_info: A row from a DataFrame containing symbol, entry price, stop loss, and take profit
    - tp: The take profit price
    - sl: The stop loss price
    """
    # config = utils.get_config()
    # account = eth_account.Account.from_key(config["secret_key"])

    print(f'opening order for {symbol_info["Symbol"]} size {size} type{type(size)}')
    
    exchange = Exchange(account, constants.MAINNET_API_URL)
    
    symbol = symbol_info["Symbol"]
    entry_price = symbol_info["Entry Price"]

    sl = symbol_info["Stop Loss"]
    tp = symbol_info["Take Profit"]

    _, rounding = get_sz_px_decimals(symbol)
    # round tp and sl 

    if symbol == 'BTC':
        tp = int(tp)
        sl = int(sl)
    else:
        tp = round(tp, rounding)
        sl = round(sl, rounding)

    print(f'symbol: {symbol}, entry price: {entry_price}, stop loss: {sl}, take profit: {tp}')

    # Determine the order side (buy or sell)
    is_buy = True

    cancel_symbol_orders(account, symbol)

    print(f'enrty price: {entry_price} type{type(entry_price)}, stop loss: {sl} type{type(tp)}, take profit: {tp} type{type(tp)}')

    order_result = exchange.order(
        symbol,
        is_buy,
        size,  # Assuming a fixed quantity; adjust as needed
        entry_price,
        {"limit": {"tif": "Gtc"}}
    )
    print(f"Limit order result for {symbol}: {order_result}")

    # Place the stop loss order
    stop_order_type = {"trigger": {"triggerPx": sl, "isMarket": True, "tpsl": "sl"}}
    stop_result = exchange.order(
        symbol,
        not is_buy,
        size,  # Assuming a fixed quantity; adjust as needed
        sl,
        stop_order_type,
        reduce_only=True
    )
    print(f"Stop loss order result for {symbol}: {stop_result}")

    # Place the take profit order
    tp_order_type = {"trigger": {"triggerPx": tp, "isMarket": True, "tpsl": "tp"}}
    tp_result = exchange.order(
        symbol,
        not is_buy,
        size,  # Assuming a fixed quantity; adjust as needed
        tp,
        tp_order_type,
        reduce_only=True
    )
    print(f"Take profit order result for {symbol}: {tp_result}")


def get_open_interest():
    
    # read in this csv and get the most recent which is the 2nd column
    open_interest = pd.read_csv('/Users/md/Dropbox/dev/github/liquidation-trading-bot/data/oi_total.csv')
    oi = int(open_interest.iloc[-1, 1]) 

    return oi

def get_liquidations():
    # Read in the CSV file
    liquidations_df = pd.read_csv('/Users/md/Dropbox/dev/github/liquidation-trading-bot/data/recent_liqs.csv')

    # Convert DataFrame to JSON
    liquidations_json = {}

    for index, row in liquidations_df.iterrows():
        interval = row['Interval']
        liquidations_json[interval] = {
            'Total Liquidations': row['Total Liquidations'],
            'Long Liquidations': row['Long Liquidations'],
            'Short Liquidations': row['Short Liquidations']
        }

    return json.dumps(liquidations_json, indent=4)


def get_funding_rate():
    # Read in the CSV file
    funding_df = pd.read_csv('/Users/md/Dropbox/dev/github/liquidation-trading-bot/data/funding.csv')

    # Convert DataFrame to JSON
    funding_json = {}

    for index, row in funding_df.iterrows():
        symbol = row['symbol']
        funding_json[symbol] = {
            'Funding Rate': row['funding_rate'],
            'Yearly Funding Rate': row['yearly_funding_rate']
        }

    return json.dumps(funding_json, indent=4)


def symbol_info(symbol, price, stop_loss, take_profit, px_decimals, sz_decimals, ideal_size):

    price = float(price)
    size = round(ideal_size, sz_decimals)

    # calculate the stop loss and take profit based on the price. tp is 50 like 50% and sl is -25 like 25% below the price
    stop_loss = price - (price * abs(stop_loss) / 100)

    # 50 is 50% above the price
    take_profit = price + (price * take_profit / 100)

    # -25 is 25% below the price
    stop_loss = round(stop_loss, px_decimals)
    take_profit = round(take_profit, px_decimals)
    price = round(price, px_decimals)

    return {
        "Symbol": symbol,
        "Size": size,
        "Entry Price": price,
        "Stop Loss": stop_loss,
        "Take Profit": str(take_profit)
    }

