# Patch for numpy.NaN compatibility
import numpy
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan

import requests, pandas as pd, pandas_ta as ta, logging, datetime, time, json, eth_account, sys, os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import project modules
from Day_4_Projects import dontshare as d
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info 
from hyperliquid.exchange import Exchange 
from hyperliquid.utils import constants 
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("hyperliquid_trading")

# Constants
API_URL = constants.MAINNET_API_URL
API_HEADERS = {'Content-Type': 'application/json'}
DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_LEVERAGE = 3
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # seconds

# Configuration
secret_key = d.private_key

symbol='WIF'

def make_api_request(url: str, data: Dict, method: str = "post", 
                   timeout: int = DEFAULT_TIMEOUT, 
                   max_retries: int = MAX_RETRY_ATTEMPTS) -> Dict:
    """
    Make API request with error handling and retries.
    
    Args:
        url: API endpoint
        data: Request data
        method: HTTP method (default "post")
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        API response as dict
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if method.lower() == "post":
                response = requests.post(
                    url, 
                    headers=API_HEADERS, 
                    json=data,
                    timeout=timeout
                )
            else:
                response = requests.get(
                    url, 
                    headers=API_HEADERS, 
                    params=data,
                    timeout=timeout
                )
            
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"API request failed after {max_retries} attempts: {str(e)}")
            
            logger.warning(f"API request failed, retrying ({retry_count}/{max_retries}): {str(e)}")
            time.sleep(RETRY_DELAY)

def ask_bid(symbol: str) -> Tuple[float, float, List]:
    '''this gets the ask and bid for any symbol passed in'''

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {
        'type': 'l2Book', 
        'coin': symbol
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        l2_data = response.json()
        l2_data = l2_data['levels']

        # get ask bid 
        bid = float(l2_data[0][0]['px'])
        ask = float(l2_data[1][0]['px'])

        return ask, bid, l2_data
    except (requests.RequestException, KeyError, IndexError) as e:
        raise Exception(f"Error fetching ask/bid for {symbol}: {str(e)}")

def get_sz_px_decimals(coin: str) -> Tuple[int, int]:

    ''' this returns size devimals and price decimals '''

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == coin), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']

        else:
            raise ValueError(f'Symbol {coin} not found')

    except (requests.RequestException, KeyError, ValueError) as e:
        raise Exception(f"Error getting decimals for {coin}: {str(e)}")

    ask = ask_bid(coin)[0]

    ask_str = str(ask)
    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0 

    print(f'{coin} this is the price {sz_decimals} decimals')

    return sz_decimals, px_decimals


# MAKE A BUY AND A SELL ORDER
def limit_order(coin: str, is_buy: bool, sz: float, limit_px: float, reduce_only: bool, account: LocalAccount) -> Dict:
    exchange = Exchange(account, constants.MAINNET_API_URL)
    rounding, _ = get_sz_px_decimals(coin)
    sz = round(sz, rounding)
    print(f'coin: {coin}, type: {type(coin)}')
    print(f'is_buy: {is_buy}, type: {type(is_buy)}')
    print(f'sz: {sz}, type: {type(limit_px)}')
    print(f'reduce_only: {reduce_only}, type: {type(reduce_only)}')

    print(f'placing limit order for {coin} {sz} @ {limit_px}')
    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": 'Gtc'}}, reduce_only=reduce_only)

    # Check if order result is an error
    if isinstance(order_result, dict) and 'response' in order_result:
        # Check if response is a dictionary or a string
        if isinstance(order_result['response'], dict) and 'data' in order_result['response']:
            if is_buy == True:
                print(f"limit BUY order placed thanks moon dev, resting: {order_result['response']['data']['statuses'][0]}")
            else:
                print(f"limit SELL order placed thanks moon dev, resting: {order_result['response']['data']['statuses'][0]}")
        else:
            # Handle error response (when response is not a dict or doesn't have 'data')
            print(f"Error placing order: {order_result}")
    else:
        print(f"Unexpected order result format: {order_result}")

    return order_result

def acct_bal(account: LocalAccount) -> float:

    # account = LocalAccount = eth_account.Account.from_key(key)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')

    acct_value = user_state["marginSummary"]["accountValue"]

    return acct_value


def adjust_leverage_size_signal(symbol: str, leverage: float, account: LocalAccount) -> Tuple[float, float]:

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

        acct_val95 = acct_value * .95

        print(exchange.update_leverage(leverage, symbol))

        price = ask_bid(symbol)[0]

        # size == balance / price * leverage
        # INJ 6.95 ... at 10x lev... 10 INJ == $cost 6.95
        size = (acct_val95 / price) * leverage
        size = float(size)
        rounding, _ = get_sz_px_decimals(symbol)
        size = round(size, rounding)
        #print(f'this is the size we can use 95% fo acct val {size}')
    
        user_state = info.user_state(account.address)
            
        return leverage, size


def get_position_andmaxpos(symbol: str, account: LocalAccount, max_positions: int) -> Tuple[List, bool, float, Optional[str], float, float, Optional[bool], int]:

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


def get_position(symbol: str, account: LocalAccount) -> Tuple[List, bool, float, Optional[str], float, float, Optional[bool]]:

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


def cancel_all_orders(account: LocalAccount) -> None:

    # this cancels all open orders
    #account = LocalAccount = eth_account.Account.from_key(key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    open_orders = info.open_orders(account.address)

    print('above are the open orders... need to cancel any...')
    for open_order in open_orders:
        #print(f'cancelling order {open_order}')
        exchange.cancel(open_order['coin'], open_order['oid'])


def kill_switch(symbol: str, account: LocalAccount) -> None:

    position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    while im_in_pos == True:

        cancel_all_orders(account)

        ask, bid, l2 = ask_bid(symbol)

        pos_size = abs(pos_size)

        if long == True:
            limit_order(pos_sym, False, pos_size, ask, True, account)
            print('kill switch - SELL TO CLOSE SUBMITTED ')
            time.sleep(5)
        elif long == False:
            limit_order(pos_sym, True, pos_size, bid, True, account)
            print('kill switch - BUY TO CLOSE SUBMITTED ')
            time.sleep(5)

        position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    print('position succesfully closed in the kill switch')



    # snapshot_data = n.get_ohlcv2('BTC', '1m', 500)
    # df = n.process_data_to_df(snapshot_data)
    # bbdf = n.calculate_bollinger_bands(df)

    # bollinger_bands_tight = n.calculate_bollinger_bands(df)


def pnl_close(symbol: str, target: float, max_loss: float, account: LocalAccount) -> None:

    '''
    monitors positions for their pnl and will close the position when you hit the tp/sl

    '''

    print('starting pnl close')

    position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    if pnl_perc > target:
        print(f'pnl gain is {pnl_perc} and target is {target}... closing position WIN')
        kill_switch(pos_sym, account)
    elif pnl_perc <= max_loss:
        print(f'pnl loss is {pnl_perc} and max loss is {max_loss}... closing position LOSS')
        kill_switch(pos_sym, account)
    else:
        print(f'pnl loss is {pnl_perc} and max loss is {max_loss} and target {target}... not closing position')

    print('finished with pnl close')



    # snapshot_data = n.get_ohlcv2('BTC', '1m', 500)
    # df = n.process_data_to_df(snapshot_data)
    # bbdf = n.calculate_bollinger_bands(df)

    # bollinger_bands_tight = n.calculate_bollinger_bands(df)


def close_all_positions(account: LocalAccount) -> None:

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


def calculate_bollinger_bands(df: pd.DataFrame, length: int = 20, std_dev: int = 2) -> Tuple[pd.DataFrame, bool, bool]:
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

    try:
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
    except Exception as e:
        raise Exception(f"Error calculating Bollinger Bands: {str(e)}")

def process_data_to_df(snapshot_data: List) -> pd.DataFrame:
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

def get_ohlcv2(symbol: str, interval: str, lookback_days: int) -> List:
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
    
def fetch_candle_snapshot(symbol: str, interval: str, start_time: datetime, end_time: datetime) -> List:

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

def calculate_sma(prices: pd.Series, window: int) -> float:
    sma = prices.rolling(window=window).mean()
    return sma.iloc[-1]  # Return the most recent SMA value

def get_latest_sma(symbol: str, interval: str, window: int, lookback_days: int = 1) -> Optional[float]:
    start_time = datetime.now() - timedelta(days=lookback_days)
    end_time = datetime.now()

    snapshots = fetch_candle_snapshot(symbol, interval, start_time, end_time)

    if snapshots:
        prices = pd.Series([float(snapshot['c']) for snapshot in snapshots])
        latest_sma = calculate_sma(prices, window)
        return latest_sma
    else:
        return None

def supply_demand_zones_hl(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:

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

    #print(df)

    sd_df[f'{timeframe}_dz'] = [supp_lo, supp]
    sd_df[f'{timeframe}_sz'] = [res_hi, resis]

    print('here are moons supply and demand zones')
    print(sd_df)

    return sd_df 

def calculate_vwap_with_symbol(symbol: str) -> Tuple[pd.DataFrame, float]:
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


#######


def get_position(symbol: str, account: LocalAccount) -> Tuple[List, bool, float, Optional[str], float, float, Optional[bool]]:
    ''' gets the position info we need'''
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f'this is the currenct account val {user_state["marginSummary"]["accountValue"]}')
    positions = []
    print(f'this is the symbol {symbol}')
    for position in user_state["assetPositions"]:
        if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True 
            size = float(position["position"]['szi'])
            pos_sym = position["position"]['coin']
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
    elif size <0:
        long = False 
    else:
        long = None 

    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long

def ask_bid(symbol: str):

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {
        'type': 'l2Book',
        'coin': symbol
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']
    #print(l2_data)

    # get bid and ask 
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data

def get_sz_px_decimals(symbol: str):

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
    #print(f'this is the ask {ask}')

    # Compute the number of decimal points in the ask price
    ask_str = str(ask)
    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0

    print(f'{symbol} this is the price {sz_decimals} decimal(s)')

    return sz_decimals, px_decimals

def limit_order(coin: str, is_buy: bool, sz: float, limit_px: float, reduce_only: bool, account: LocalAccount):
    #account: LocalAccount = eth_account.Account.from_key(key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    # info = Info(constants.MAINNET_API_URL, skip_ws=True)
    # user_state = info.user_state(account.address)
    # print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    
    rounding, _ = get_sz_px_decimals(coin)
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

def cancel_all_orders(account: LocalAccount) -> None:
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


def kill_switch(symbol: str, account: LocalAccount) -> None:
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
    while im_in_pos:
        cancel_all_orders(account)

        # get bid ask
        ask, bid, l2_data = ask_bid(pos_sym)

        pos_size = abs(pos_size)

        if long == True:
            limit_order(pos_sym, False, pos_size, ask, True, account)
            print('kill switch sell to close submitted')
            time.sleep(5)
        elif long == False:
            limit_order(pos_sym, True, pos_size, bid, True, account)
            print('kill switch buy to close submitted')
            time.sleep(5)

        positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    print('position successfully closed in kill switch')


def pnl_close(symbol: str, target: float, max_loss: float, account: LocalAccount) -> None:

    ''' this checks if we hit our target or max loss'''

    print('entering pnl close')
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)

    if pnl_perc > target:
        print(f'pnl gain is {pnl_perc} and target is {target} closing position as a WIN')
        kill_switch(pos_sym, account)
    elif pnl_perc <= max_loss:
        print(f'pnl loss is {pnl_perc} and max loss is {max_loss} closing position as a LOSS')
        kill_switch(pos_sym, account)
    else:
        print(f'pnl loss is {pnl_perc} and max loss is {max_loss} target {target} not closing')

    print('finsihed pnl close')

def acct_bal(account: LocalAccount) -> float:

    account: LocalAccount = eth_account.Account.from_key(secret_key)
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')

    value = user_state["marginSummary"]["accountValue"]

    return value
