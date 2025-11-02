# this is a second nice_funcs file to be used with a diff solana wallet

# Moved imports to the top
import requests
import json
import base64
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
import time
from termcolor import cprint
from datetime import datetime, timedelta
import pandas as pd
import pprint
import re as reggie
from dotenv import load_dotenv

# from CONFIG import * # Removed wildcard import

# Placeholder constants previously from CONFIG.py
# These should be reviewed for a better configuration management strategy (e.g., pass as arguments or via a config object)
# USDC_SIZE = 1  # Example value, confirm from actual CONFIG.py if different for this context - REMOVED, use config['USDC_SIZE']
# MIN_TRADES_LAST_HOUR = 60 # From Day_50_Projects/CONFIG.py - Will be removed, accessed via config - ALREADY REMOVED
# DO_NOT_TRADE_LIST = ['So11111111111111111111111111111111111111111','KhALz1buce8zVHjoGzGjGqiJXu9bYyJ1H1shT3vNq1i','hntyVP6YFm1Hg25TN9WGLqM12b8TQmcknKrdu1oxWux','4ySwgHAevvDi65ZQ1rEhc5L5ZK386Fm95VgTiLqVpump', 'EXukF7ynKHPC1BTGxJ4ajDusoEhNvpF8xbzZxU8mpump','3qvnewRfvjum6SMunykwXYswmeAF2QKr6khrLqmipump','kvoPvdfHUVnxD4t2jdqkm4a34HMRMseKx19JFZPpump','FtMXRF47Jm9yB6abJeK7pgeeDhwxa2GiPfAdv9Fipump','5qvqjA8vxwS7rhMEmox1Vi7v3C8qWr74wgcK4TQapump','FivkLHJiYhoZT1YrSRWsD6XPBvSNVwmwff1y82vZpump','3jN3zPSE2nNHuQ7bT2JH2JYhkzbL7vnvJnoMNQ8LyTTM','LoxQiS7XLhbtZsdYFCSKjjGPfu4En6pdoerRuTzpump','GMrCLQP6XeiJAmJfFj3ApJPwKQt4xusd69rve3Dxpump','CwkcC3MF58gxcxfGt46onbqNb3jq4EhVnJVMAa1ipump','Fs6Ucti3HjXjqLaQLiXfgVg7wxJw5msrrYDAQPDFpump', '5146zXQWqRvGPytJtRFXYjcbYBwg6XfkWQUobZcxJZHX','7ZWweKCCyVzWTToS1EuB3V3krTxz142KmcbbnUEypump','D1ySHVWnaWQsf8WiskayoF7oHuvXLp4CXvYw3PaS8N7B','BXoz2cxZi65oQBhcSvxgv7dwFMhg184v1N3idhJhpump','5NegGDJBWqfJbKFYwuuUnWcnKQtez7jkRkzAx6Rjpump','DsfwbGtT2pSFaFTZUe6hwwir2wQvFvXsYahC4uv6T85y','rjyT9DdKu47fsX9JfozX4z6Qk25y12kFaVuV7cFpump','GV2HFvE9quMWMp5BDuRz3XudHyRPEAMMtdAqoF2u7yE8','Eq1qrNGCiCtcZhFUGDYAmRJQ2w9kdLPak1Gx4mvkTQCE', 'Q1BaFmfN8TXdMVS98RYMhFZWRzVTCp8tUDhqM9CgcAL','HiZZAjSHf8W53QPtWYzj1y9wqhdirg124fiEHFGiUpQh', 'AuabGXArmR3QwuKxT3jvSViVPscQASkFAvnGDQCE8tfm','rxkExwV2Gay2Bf1so4chsZj7f4MiLKTx45bd9hQy6dK','BmDXugmfBhqKE7S2KVdDnVSNGER5LXhZfPkRmsDfVuov','423scBCY2bzX6YyqwkjCfWN114JY3xvyNNZ1WsWytZbF','7S6i87ZY29bWNbkviR2hyEgRUdojjMzs1fqMSXoe3HHy', '8nBNfJsvtVmZXhbyLCBg3ndVW2Zwef7oHuCPjQVbRqfc','FqW3CJYF3TfR49WXRusxqCbJMNSjnay1A51sqP34ZxcB','EwsHNUuAtPc6SHkhMu8sQoyL6R4jnWYUU1ugstHXo5qQ','EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', '9Y9yqdNUL76v1ybpkQnVUj35traGEHXTBJB2b1iszFVv', 'Fd1hzhprThxCwz2tv5rTKyFeVCyEKRHaGqhT7hDh4fsW', '83227N9Fq4h1HMNnuKut61beYcB7fsCnRbzuFDCt2rRQ', 'J1oqg1WphZaiRDTfq7gAXho6K1xLoRMxVvVG5BBva3fh', 'GEvQuL9DT2UDtuTCCyjxm6KXEc7B5oguTHecPhKad8Dr',"3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh", "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263","7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr","7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",'zBTCug3er3tLyffELcvDNrKkCymbPWysGcWihESYfLg','EyzgnBfHGe9hh169B8993muBVcoeURCnSgPbddBeSybo'] # From Day_50_Projects/CONFIG.py - REMOVED, use config['DO_NOT_TRADE_LIST']
# SELL_AMOUNT_PERC = 0.7 # From Day_50_Projects/CONFIG.py - REMOVED, use config['SELL_AMOUNT_PERC']
# ORDERS_PER_OPEN = 1 # From Day_50_Projects/CONFIG.py - REMOVED, use config['ORDERS_PER_OPEN']
# MIN_TRADES_30M_ONCE_IN_POSITION = 5 # From Day_50_Projects/CONFIG.py - Will be removed, accessed via config - ALREADY REMOVED
# SLIPPAGE = 499  # From Day_50_Projects/CONFIG.py - REMOVED, use config['SLIPPAGE']
# PRIORITY_FEE = 20000 # From Day_50_Projects/CONFIG.py - REMOVED, use config['PRIORITY_FEE']
# CLOSED_POSITIONS_TXT = 'data/closed_positions.txt' # From Day_50_Projects/CONFIG.py - REMOVED, use config['CLOSED_POSITIONS_TXT']
# ALL_TRENDING_EVER_CSV = 'csvs/all_trending_ever.csv' # From Day_50_Projects/CONFIG.py - REMOVED, use config['ALL_TRENDING_EVER_CSV']
# LEADERBOARD_CSV = 'csvs/leaderboard.csv' # From Day_50_Projects/CONFIG.py - REMOVED, use config['LEADERBOARD_CSV']
# READY_TO_BUY_CSV = 'csvs/trending_tokens.csv' # From Day_50_Projects/CONFIG.py - REMOVED, use config['READY_TO_BUY_CSV']
# MY_ADDRESS = '4wgfCBf2WwLSRKLef9iW7JXZ2AfkxUxGM4XcKpHm3Sin' # From Day_50_Projects/CONFIG.py - REMOVED, use config['MY_ADDRESS']
# MY_ADDRESS2 = 'G1vNV2SkzGPTBTDd7c3iypL4TPNGbe37rEbJb7cxVTQS' # From Day_50_Projects/CONFIG.py - REMOVED, use config['MY_ADDRESS2']
# TP = 49 # From Day_50_Projects/CONFIG.py - REMOVED, use config['TP']
# SL = 0.6 # From Day_50_Projects/CONFIG.py - REMOVED, use config['SL']
# SMA_DAYS_BACK = 2 # Will be removed, accessed via config - ALREADY REMOVED
# SMA_TIMEFRAME = '1H' # Will be removed, accessed via config - ALREADY REMOVED


# Load environment variables
load_dotenv()

# API Keys from environment variables
# API_KEY = os.getenv('BIRDEYE_KEY') # Removed, will use config['BIRDEYE_KEY']
# SOL_KEY = os.getenv('SOL_KEY2') # Will be accessed via config
# RPC_URL = os.getenv('RPC_URL') # Will be accessed via config

sample_address = "2yXTyarttn2pTZ6cwt4DqmrRuBw1G7pmFv9oT6MStdKP"

BASE_URL = "https://public-api.birdeye.so/defi"

# Custom function to print JSON in a human-readable format
def print_pretty_json(data):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

# Function to print JSON in a human-readable format - assuming you already have it as print_pretty_json
# Helper function to find URLs in text
def find_urls(string):
    # Regex to extract URLs
    return reggie.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)

# UPDATED TO RMEOVE THE OTHER ONE so now we can just use this filter instead of filtering twice
def token_overview(config, address):
    """
    Fetch token overview for a given address and return structured information, including specific links,
    and assess if any price change suggests a rug pull.
    """
    
    #print(f'Getting the token overview for {address}')
    BASE_URL = "https://public-api.birdeye.so/defi" # This could also be part of config if it varies
    overview_url = f"{BASE_URL}/token_overview?address={address}"
    # headers = {"X-API-KEY": API_KEY} # OLD
    headers = {"X-API-KEY": config['BIRDEYE_KEY']}

    response = requests.get(overview_url, headers=headers)
    result = {}

    if response.status_code == 200:
        overview_data = response.json().get('data', {})
        
        # Retrieve buy1h, sell1h, and calculate trade1h
        buy1h = overview_data.get('buy1h', 0)
        sell1h = overview_data.get('sell1h', 0)
        trade1h = buy1h + sell1h

        # Add the calculated values to the result
        result['buy1h'] = buy1h
        result['sell1h'] = sell1h
        result['trade1h'] = trade1h

        # Calculate buy and sell percentages
        total_trades = trade1h  # Assuming total_trades is the sum of buy and sell
        buy_percentage = (buy1h / total_trades * 100) if total_trades else 0
        sell_percentage = (sell1h / total_trades * 100) if total_trades else 0
        result['buy_percentage'] = buy_percentage
        result['sell_percentage'] = sell_percentage

        # Check if trade1h is bigger than MIN_TRADES_LAST_HOUR
        # result['minimum_trades_met'] = True if trade1h >= MIN_TRADES_LAST_HOUR else False # OLD
        result['minimum_trades_met'] = True if trade1h >= config['MIN_TRADES_LAST_HOUR'] else False

        # Extract price changes over different timeframes
        price_changes = {k: v for k, v in overview_data.items() if 'priceChange' in k}
        result['priceChangesXhrs'] = price_changes

        # Check for rug pull indicator
        rug_pull = any(value < -80 for key, value in price_changes.items() if value is not None)
        result['rug_pull'] = rug_pull
        if rug_pull:
            print("Warning: Price change percentage below -80%, potential rug pull")

        # Extract other metrics
        unique_wallet2hr = overview_data.get('uniqueWallet24h', 0)
        v24USD = overview_data.get('v24hUSD', 0)
        watch = overview_data.get('watch', 0)
        view24h = overview_data.get('view24h', 0)
        liquidity = overview_data.get('liquidity', 0)

        # Add the retrieved data to result
        result.update({
            'uniqueWallet2hr': unique_wallet2hr,
            'v24USD': v24USD,
            'watch': watch,
            'view24h': view24h,
            'liquidity': liquidity,
        })

        # Extract and process description links if extensions are not None
        extensions = overview_data.get('extensions', {})
        description = extensions.get('description', '') if extensions else ''
        urls = find_urls(description)
        links = []
        for url in urls:
            if 't.me' in url:
                links.append({'telegram': url})
            elif 'twitter.com' in url:
                links.append({'twitter': url})
            elif 'youtube' not in url:  # Assume other URLs are for website
                links.append({'website': url})

        # Add extracted links to result
        result['description'] = links


        # Return result dictionary with all the data
        return result
    else:
        print(f"Failed to retrieve token overview for address {address}: HTTP status code {response.status_code}")
        return None


def token_security_info(config, address):

    '''

    bigmatter
​freeze authority is like renouncing ownership on eth

    Token Security Info:
{   'creationSlot': 242801308,
    'creationTime': 1705679481,
    'creationTx': 'ZJGoayaNDf2dLzknCjjaE9QjqxocA94pcegiF1oLsGZ841EMWBEc7TnDKLvCnE8cCVfkvoTNYCdMyhrWFFwPX6R',
    'creatorAddress': 'AGWdoU4j4MGJTkSor7ZSkNiF8oPe15754hsuLmwcEyzC',
    'creatorBalance': 0,
    'creatorPercentage': 0,
    'freezeAuthority': None,
    'freezeable': None,
    'isToken2022': False,
    'isTrueToken': None,
    'lockInfo': None,
    'metaplexUpdateAuthority': 'AGWdoU4j4MGJTkSor7ZSkNiF8oPe15754hsuLmwcEyzC',
    'metaplexUpdateAuthorityBalance': 0,
    'metaplexUpdateAuthorityPercent': 0,
    'mintSlot': 242801308,
    'mintTime': 1705679481,
    'mintTx': 'ZJGoayaNDf2dLzknCjjaE9QjqxocA94pcegiF1oLsGZ841EMWBEc7TnDKLvCnE8cCVfkvoTNYCdMyhrWFFwPX6R',
    'mutablefrom sMetadata': True,
    'nonTransferable': None,
    'ownerAddress': None,
    'ownerBalance': None,
    'ownerPercentage': None,
    'preMarketHolder': [],
    'top10HolderBalance': 357579981.3372284,
    'top10HolderPercent': 0.6439307358062863,
    'top10UserBalance': 138709981.9366756,
    'top10UserPercent': 0.24978920911102176,
    'totalSupply': 555308143.3354646,
    'transferFeeData': None,
    'transferFeeEnable': None}
    '''

    # API endpoint for getting token security information
    url = f"{BASE_URL}/token_security?address={address}"
    # headers = {"X-API-KEY": API_KEY} # OLD - Assuming this function might also need config if used independently
    # For now, if token_security_info is only called by scripts that setup their own API_KEY, it might be fine.
    # However, for consistency, it should also take config if it makes API calls.
    # Let's assume it should take config for BIRDEYE_KEY for now.
    headers = {"X-API-KEY": config['BIRDEYE_KEY']}

    # Sending a GET request to the API
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse the JSON response
        security_data = response.json()['data']
        print_pretty_json(security_data)
    else:
        print("Failed to retrieve token security info:", response.status_code)

def token_creation_info(config, address):

    '''
    output sampel = 

    Token Creation Info:
{   'decimals': 9,
    'owner': 'AGWdoU4j4MGJTkSor7ZSkNiF8oPe15754hsuLmwcEyzC',
    'slot': 242801308,
    'tokenAddress': '9dQi5nMznCAcgDPUMDPkRqG8bshMFnzCmcyzD8afjGJm',
    'txHash': 'ZJGoayaNDf2dLzknCjjaE9QjqxocA94pcegiF1oLsGZ841EMWBEc7TnDKLvCnE8cCVfkvoTNYCdMyhrWFFwPX6R'}
    '''
    # API endpoint for getting token creation information
    url = f"{BASE_URL}/token_creation_info?address={address}"
    # headers = {"X-API-KEY": API_KEY} # OLD - Similar to token_security_info, should take config
    headers = {"X-API-KEY": config['BIRDEYE_KEY']}

    # Sending a GET request to the API
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse the JSON response
        creation_data = response.json()['data']
        print_pretty_json(creation_data)
    else:
        print("Failed to retrieve token creation info:", response.status_code)

def market_buy(config, token, amount, slippage_bps=None):
    # import requests # Moved to top
    # import sys # Moved to top
    # import json # Moved to top
    # import base64 # Moved to top
    # from solders.keypair import Keypair # Moved to top
    # from solders.transaction import VersionedTransaction # Moved to top
    # from solana.rpc.api import Client # Moved to top
    # from solana.rpc.types import TxOpts # Moved to top
    # import time # Moved to top

    KEY = Keypair.from_base58_string(config['SOL_KEY2']) # Assuming SOL_KEY2 from original env load is the one for this wallet
    current_slippage_bps = slippage_bps if slippage_bps is not None else config['SLIPPAGE']
    priority_fee_lamports = config['PRIORITY_FEE']
    rpc_url = config['RPC_URL']

    QUOTE_TOKEN = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # usdc

    # Use RPC endpoint
    http_client = Client(rpc_url)


    #quote_url = f'https://quote-api.jup.ag/v6/quote?inputMint={QUOTE_TOKEN}&outputMint={token}&amount={amount}&slippageBps={SLIPPAGE}'
    # testing &restrictIntermediateTokens=true which only allows routes that are with liquid tokens
    quote_url = f'https://quote-api.jup.ag/v6/quote?inputMint={QUOTE_TOKEN}&outputMint={token}&amount={amount}&slippageBps={current_slippage_bps}&restrictIntermediateTokens=true'
    swap_url = 'https://quote-api.jup.ag/v6/swap'
    
    while True:
        try:
            quote = requests.get(quote_url).json()
            #print(quote)

            txRes = requests.post(swap_url,
                                  headers={"Content-Type": "application/json"},
                                  data=json.dumps({
                                      "quoteResponse": quote,
                                      "userPublicKey": str(KEY.pubkey()),
                                      "prioritizationFeeLamports": priority_fee_lamports 
                                  })).json()
            # print(txRes)
            swapTx = base64.b64decode(txRes['swapTransaction'])
            #print(swapTx)
            tx1 = VersionedTransaction.from_bytes(swapTx)
            #print(tx1)
            tx = VersionedTransaction(tx1.message, [KEY])
            txId = http_client.send_raw_transaction(bytes(tx), TxOpts(skip_preflight=True)).value
            print(f"https://solscan.io/tx/{str(txId)}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying


def market_sell(config, input_token_mint, amount_atomic, slippage_bps=None):
    # import requests # Moved to top
    # import json # Moved to top
    # import base64 # Moved to top
    # from solders.keypair import Keypair # Moved to top
    # from solders.transaction import VersionedTransaction # Moved to top
    # from solana.rpc.api import Client # Moved to top
    # from solana.rpc.types import TxOpts # Moved to top

    # print(f'selling {QUOTE_TOKEN[:4]} at {amount}') # QUOTE_TOKEN here was the input token
    print(f'selling {input_token_mint[:4]} for {amount_atomic} atomic units')
    KEY = Keypair.from_base58_string(config['SOL_KEY2']) # Assuming SOL_KEY2
    output_token_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # usdc (output)
    rpc_url = config['RPC_URL']
    current_slippage_bps = slippage_bps if slippage_bps is not None else config['SLIPPAGE']
    priority_fee_lamports = config['PRIORITY_FEE']

    http_client = Client(rpc_url)
    #quote_url = f'https://quote-api.jup.ag/v6/quote?inputMint={QUOTE_TOKEN}&outputMint={token}&amount={amount}&restrictIntermediateTokens=true'
    quote_url = f'https://quote-api.jup.ag/v6/quote?inputMint={input_token_mint}&outputMint={output_token_mint}&amount={amount_atomic}'
    #print('we here.. quate don')
    #print(quote_url)
    # Fixed minimum slippage
    min_slippage_jupiter = 5 # This seems to be a Jupiter API specific minimum, not from global config

    quote = requests.get(quote_url).json()
    #print(quote)
    
    # Post request to swap with dynamic slippage
    txRes = requests.post('https://quote-api.jup.ag/v6/swap',
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({
                              "quoteResponse": quote,
                              "userPublicKey": str(KEY.pubkey()),
                              "prioritizationFeeLamports": priority_fee_lamports,
                              "dynamicSlippage": {"minBps": min_slippage_jupiter, "maxBps": current_slippage_bps},
                          })).json() 

    swapTx = base64.b64decode(txRes['swapTransaction'])
    #print(swapTx)
    tx1 = VersionedTransaction.from_bytes(swapTx)
    #print(tx1)
    tx = VersionedTransaction(tx1.message, [KEY])
    #print(tx)
    txId = http_client.send_raw_transaction(bytes(tx), TxOpts(skip_preflight=True)).value
    #print(txId)



def get_token_overview(config, address):

    '''
    update this function so that i can cache it and not call it as much

    '''
    url = f"https://public-api.birdeye.so/defi/token_overview?address={address}"
    # headers = {"X-API-KEY": API_KEY} # OLD
    headers = {"X-API-KEY": config['BIRDEYE_KEY']}
    response = requests.get(url, headers=headers)
    if response.ok:
        json_response = response.json()
        return json_response['data']
    else:
        # Return empty dict if there's an error
        print(f"Error fetching data for address {address}: {response.status_code}")
        return {}
    
def get_names_nosave(config, df_input):
    """
    Processes a DataFrame of tokens, adds names, and identifies low-volume tokens.
    Modifies df_input by adding a 'name' column and dropping 'Mint Address' and 'Amount'.
    Returns the modified DataFrame and a list of low-volume token addresses.
    """
    df = df_input.copy() # Work on a copy to avoid modifying the original DataFrame passed in, unless intended by caller
    names = []
    low_volume_tokens = []

    for index, row in df.iterrows():
        token_mint_address = row['Mint Address']
        token_data = get_token_overview(config, token_mint_address)

        token_name = token_data.get('name', 'N/A')
        names.append(token_name)
        
        trade30m = token_data.get('trade30m', 0)
        cprint(f'{token_name} has {trade30m} trades in the last 30 mins', 'yellow') # Changed color for visibility
        
        is_protected = token_mint_address in config['DO_NOT_TRADE_LIST']
        
        if trade30m < config['MIN_TRADES_30M_ONCE_IN_POSITION']:
            if not is_protected:
                cprint(f'🔍 Low volume token: {token_mint_address} detected - WILL be added to kill list', 'yellow')
                low_volume_tokens.append(token_mint_address)
            else:
                cprint(f'🚫 Low volume token: {token_mint_address} is PROTECTED in DO_NOT_TRADE_LIST - WILL NOT be killed', 'green')
    
    if low_volume_tokens:
        cprint(f"⚠️ Potential low-volume tokens identified for action: {low_volume_tokens}", 'yellow')
    
    if 'name' in df.columns:
        df['name'] = names
    else:
        df.insert(0, 'name', names)

    # It's unusual for a 'get_names' function to drop columns unrelated to names from the input df.
    # Consider if this column dropping should be done by the caller or if this function is more of a 'process_tokens_df'
    if 'Mint Address' in df.columns: # Check if columns exist before dropping
        df.drop('Mint Address', axis=1, inplace=True)
    if 'Amount' in df.columns:
        df.drop('Amount', axis=1, inplace=True)

    cprint(f'Identified low_volume_tokens: {low_volume_tokens}', 'magenta')
    return df, low_volume_tokens

from typing import Dict, List
# Assuming DO_NOT_TRADE_LIST, cprint, get_names_nosave, kill_switch, and other utilities are globally available.
# 
import requests
import pandas as pd
import time


def kill_switch(config, token_mint_address):

    # FIRST safety check: Never process tokens in DO_NOT_TRADE_LIST
    if token_mint_address in config['DO_NOT_TRADE_LIST']:
        print(f'🛑 PROTECTED TOKEN: {token_mint_address} is in DO_NOT_TRADE_LIST - NOT killing!')
        return

    print(f'kill switch for {token_mint_address[:4]}')

    # if time is on the 5 minute do the balance check, if not grab from data/current_position.csv
    # balance = fetch_wallet_holdings_nosaving_names(MY_ADDRESS, token_mint_address) # MY_ADDRESS needs to come from config
    balance_df = fetch_wallet_holdings_nosaving_names(config, config['MY_ADDRESS2'], token_mint_address) # Assuming MY_ADDRESS2 for this wallet context

    try:
        balance_amount_tokens = balance_df['Amount'].iloc[0]
        #print(f'balance: {balance_amount_tokens}')
    except (IndexError, KeyError):
        #print(f'no balance for {token_mint_address[:4]}')
        balance_amount_tokens = 0
        return
    
    # save to data/current_position.csv w/ pandas # This line was a comment, no action needed
    # sell_size = balance_amount_tokens 
    # decimals = 0 # Not needed here
    decimals = get_decimals(config, token_mint_address)
    if decimals is None:
        cprint(f"Could not get decimals for {token_mint_address[:4]}. Cannot proceed with kill_switch.", "red")
        return

    sell_size_atomic = int(balance_amount_tokens * (10 **decimals))
    
    #print(f'bal: {balance_amount_tokens} price: {price} usdVal: {usd_value} TP: {tp} sell size: {sell_size_atomic} decimals: {decimals}')

    while sell_size_atomic > 0:

        # log this mint address to a file and save as a new line, keep the old lines there, so it will continue to grow this file is called data/closed_positions.txt
        # only add it to the file if it's not already there
        # with open(CLOSED_POSITIONS_TXT, 'r') as f: # CLOSED_POSITIONS_TXT needs to come from config
        with open(config['CLOSED_POSITIONS_TXT'], 'r') as f_read:
            lines = [line.strip() for line in f_read.readlines()]  # Strip the newline character from each line
            if token_mint_address not in lines:  # Now the comparison should work as expected
                with open(config['CLOSED_POSITIONS_TXT'], 'a') as f_append:
                    f_append.write(token_mint_address + '\n')

        #print(f'closing {token_mint_address[:4]}')
        try:
            # market_sell(token_mint_address, sell_size_atomic) # OLD CALL
            market_sell(config, token_mint_address, sell_size_atomic)
            #cprint(f'just made an order {token_mint_address[:4]} selling {sell_size_atomic} ...', 'white', 'on_green')
            time.sleep(1)
            # market_sell(token_mint_address, sell_size_atomic) # OLD CALL
            market_sell(config, token_mint_address, sell_size_atomic)
            #cprint(f'just made an order {token_mint_address[:4]} selling {sell_size_atomic} ...', 'white', 'on_green')
            time.sleep(1)
            # market_sell(token_mint_address, sell_size_atomic) # OLD CALL
            market_sell(config, token_mint_address, sell_size_atomic)
            cprint(f'just made 3 orders {token_mint_address[:4]} selling {sell_size_atomic} ...', 'white', 'on_blue')
            time.sleep(15)
            
        except Exception as e: # Catch specific exceptions if possible
            cprint(f'order error.. trying again. Error: {e}', 'white', 'on_red')
            # time.sleep(7) # Consider if retry logic here is needed or if market_sell handles it

        # if time is on the 5 minute do the balance check, if not grab from data/current_position.csv
        # balance_df = fetch_wallet_holdings_nosaving_names(MY_ADDRESS, token_mint_address)
        balance_df = fetch_wallet_holdings_nosaving_names(config, config['MY_ADDRESS2'], token_mint_address)

        try:
            balance_amount_tokens = balance_df['Amount'].iloc[0]
            #print(f'balance: {balance_amount_tokens}')
        except (IndexError, KeyError):
            #print(f'no balance for {token_mint_address[:4]}')
            balance_amount_tokens = 0
            # return # Removed return, loop should break if balance_amount_tokens is 0

        # sell_size = balance_amount_tokens 
        # decimals = get_decimals(config, token_mint_address) # Already fetched
        sell_size_atomic = int(balance_amount_tokens * (10 **decimals))
        #print(f'sell size: {sell_size_atomic}')
        if sell_size_atomic == 0: # Break if no more tokens to sell
            break


    else: # This else clause for a while loop executes if the loop terminated normally (not by break)
        #print(f'for {token_mint_address[:4]} value is {usd_value} and tp is {tp} so not closing...')
        if sell_size_atomic == 0: # Check condition again for clarity
            print(f'looks like {token_mint_address[:4]} has been closed or balance is zero')
        #time.sleep(10)




def fetch_wallet_holdings_nosaving_names(config, wallet_address, token_mint_address):
    url = f"https://public-api.birdeye.so/v1/wallet/token_list?wallet={wallet_address}"
    headers = {"x-chain": "solana", "X-API-KEY": config['BIRDEYE_KEY']}
    response = requests.get(url, headers=headers)
    df = pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

    if response.status_code == 200:
        json_response = response.json()
        if 'data' in json_response and 'items' in json_response['data']:
            df_response = pd.DataFrame(json_response['data']['items'])
            if not df_response.empty:
                try:
                    df = df_response[['address', 'uiAmount', 'valueUsd']]
                    df = df.rename(columns={'address': 'Mint Address', 'uiAmount': 'Amount', 'valueUsd': 'USD Value'})
                    df = df.dropna()
                    df = df[df['USD Value'] > 0.05]
                except KeyError as e:
                    cprint(f"Error processing data in fetch_wallet_holdings_nosaving_names: {e}. Available columns: {df_response.columns}", 'white', 'on_red')
                    return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value']) 
        else:
            cprint("No data available in the response for fetch_wallet_holdings_nosaving_names.", 'white', 'on_red')
            return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])
    else:
        cprint(f"Failed to retrieve token list for {wallet_address[-3:]} in fetch_wallet_holdings_nosaving_names: {response.status_code}", 'white', 'on_magenta')
        time.sleep(10) # Consider if this sleep is essential or should be handled by caller/retry logic
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

    exclude_addresses = ['EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', 'So11111111111111111111111111111111111111112']
    config_do_not_trade_list = config.get('DO_NOT_TRADE_LIST', []) # Safely get, default to empty list
    updated_dont_trade_list = [mint for mint in config_do_not_trade_list if mint not in exclude_addresses]

    for mint_to_exclude in updated_dont_trade_list:
        df = df[df['Mint Address'] != mint_to_exclude]

    df = df[df['Mint Address'] == token_mint_address]
    return df



def get_decimals(config, token_mint_address):
    # import requests # Moved to top
    # import base64 # Moved to top # base64 not used here, but kept if it was a copy-paste error
    # import json # Moved to top
    # Solana Mainnet RPC endpoint
    # url = RPC_URL # Use the globally defined RPC_URL
    url = config['RPC_URL']
    
    headers = {"Content-Type": "application/json"}

    # Request payload to fetch account information
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [
            token_mint_address,
            {
                "encoding": "jsonParsed"
            }
        ]
    })

    # Make the request to Solana RPC
    response = requests.post(url, headers=headers, data=payload)
    response_json = response.json()

    # Parse the response to extract the number of decimals
    try:
        decimals = response_json['result']['value']['data']['parsed']['info']['decimals']
        #print(f"Decimals for {token_mint_address[:4]} token: {decimals}")
    except (KeyError, TypeError) as e:
        cprint(f"Error parsing decimals for {token_mint_address[:4]}: {e}. Response: {response_json}", 'red')
        return None

    return decimals



# kill_switch('HuAncxDEsakCDgZS2Yfo9xJbHmtHXMnxxkT9jqdXnHhm')
# time.sleep(8978)


def fetch_wallet_new25(config, wallet_address):
    """
    New BirdEye API endpoint test for Solana wallets
    """
    # API_KEY = os.getenv('BIRDEYE_KEY')
    url = f"https://public-api.birdeye.so/v1/wallet/token_list?wallet={wallet_address}"
    
    headers = {
        "accept": "application/json",
        "x-chain": "solana",
        # "X-API-KEY": API_KEY # OLD
        "X-API-KEY": config['BIRDEYE_KEY']
    }
    
    try:
        response = requests.get(url, headers=headers)
        print("\n🔍 DEBUG: API Response")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n🔄 Sample of Available Fields:")
            if 'data' in data and 'items' in data['data'] and data['data']['items']:
                sample_item = data['data']['items'][0]
                print("\n📊 Sample Token Data Fields:")
                for key, value in sample_item.items():
                    print(f"{key}: {value}")
                
                print("\n💰 Converting to DataFrame for analysis...")
                df = pd.DataFrame(data['data']['items'])
                print("\nColumns in response:")
                print(df.columns.tolist())
                
                print("\nFirst few rows of data:")
                print(df.head())
                
                return df
            else:
                print("❌ No token data found in response")
                return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

# Test the new endpoint
# print("\n🧪 Testing new BirdEye API endpoint...")
# test_wallet = MY_ADDRESS
# result = fetch_wallet_new25(test_wallet)
# print("\n💫 Test complete!")
# time.sleep(1000)  # Wait to examine the output

def fetch_wallet_holdings_og(config, address):
    url = f"https://public-api.birdeye.so/v1/wallet/token_list?wallet={address}"
    headers = {"x-chain": "solana", "X-API-KEY": config['BIRDEYE_KEY']}
    response = requests.get(url, headers=headers)
    df_main = pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

    if response.status_code == 200:
        json_response = response.json()
        if 'data' in json_response and 'items' in json_response['data']:
            df_response = pd.DataFrame(json_response['data']['items'])
            if not df_response.empty:
                try:
                    df_main = df_response[['address', 'uiAmount', 'valueUsd']]
                    df_main = df_main.rename(columns={'address': 'Mint Address', 'uiAmount': 'Amount', 'valueUsd': 'USD Value'})
                    df_main = df_main.dropna()
                    df_main = df_main[df_main['USD Value'] > 0.05]
                except KeyError as e:
                    cprint(f"Error processing data in fetch_wallet_holdings_og: {e}. Available columns: {df_response.columns}", 'white', 'on_red')
                    # Return an empty DataFrame with expected columns on error
                    return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value']) 
        else:
            cprint("No data items available in the response for fetch_wallet_holdings_og.", 'white', 'on_red')
            return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])
    else:
        cprint(f"Failed to retrieve token list for {address[-3:]} in fetch_wallet_holdings_og: {response.status_code}", 'white', 'on_magenta')
        time.sleep(10) # Consider if this sleep is essential
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

    # Filter out tokens from DO_NOT_TRADE_LIST (excluding SOL and USDC)
    exclude_addresses = ['EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', 'So11111111111111111111111111111111111111112']
    config_do_not_trade_list = config.get('DO_NOT_TRADE_LIST', [])
    # Filter df_main based on the updated_dont_trade_list
    # Create the list of mints to exclude by filtering config_do_not_trade_list
    mints_to_filter_out = [mint for mint in config_do_not_trade_list if mint not in exclude_addresses]
    df_main = df_main[~df_main['Mint Address'].isin(mints_to_filter_out)]

    if not df_main.empty:
        TOKEN_PER_ADDY_CSV = config.get('TOKEN_PER_ADDY_CSV', 'filtered_wallet_holdings.csv') 
        try:
            df_main.to_csv(TOKEN_PER_ADDY_CSV, index=False)
        except Exception as e:
            cprint(f"Error saving to {TOKEN_PER_ADDY_CSV}: {e}", "red")
        
        # Process for names and low volume, get_names_nosave modifies a copy
        df_with_names, low_volume_tokens = get_names_nosave(config, df_main.copy()) 

        if low_volume_tokens:
            cprint(f"Final low_volume_tokens in fetch_wallet_holdings_og: {[token[:4] for token in low_volume_tokens]}", 'yellow')
        
        config_dnt_list_for_kill = config.get('DO_NOT_TRADE_LIST', []) # Re-fetch for clarity or use previous
        for token_to_check in low_volume_tokens:
            if token_to_check not in config_dnt_list_for_kill:
                cprint(f'🔄 Closing low volume token in fetch_wallet_holdings_og: {token_to_check} - NOT protected', 'magenta')
                kill_switch(config, token_to_check) 
            else:
                cprint(f'🚫 PROTECTED low volume token in fetch_wallet_holdings_og: {token_to_check} - in DO_NOT_TRADE_LIST', 'green')

        print('')
        # print(df_with_names.head(50)) # Print the DataFrame that has names and dropped columns
        # Decide what to print for debugging, e.g. the original df_main or the processed one
        if not df_with_names.empty:
            print("Processed holdings (with names, some cols dropped by get_names_nosave):")
            print(df_with_names.head(50))
        elif not df_main.empty:
            print("Original filtered holdings (before name processing):")
            print(df_main.head(50))
        
        print(' ')
        # time.sleep(7) # Consider if this sleep is essential
    else:
        cprint("No wallet holdings to display after filtering in fetch_wallet_holdings_og.", 'white', 'on_red')
        # time.sleep(30)

    # The function get_names_nosave returns a modified DataFrame (df_with_names) 
    # where 'Mint Address' and 'Amount' are dropped.
    # If the caller of fetch_wallet_holdings_og expects these columns, returning df_main is correct.
    # If the processed version is desired, return df_with_names.
    # Based on typical usage, caller likely expects the original structure after filtering DNT list.
    return df_main 

# def fetch_wallet_token_single(config, wallet_address, token_mint_address):
# ... (ensure this is correct)

# def get_position(config, token_mint_address):
# ... (ensure this is correct)

# def token_price(config, address):
# ... (ensure this is correct)

# ... other functions like pnl_close, open_position etc. should already be using config for their calls ...

# def get_data(config, address, days_back_4_data, timeframe):
# ... (ensure this is correct, uses config['BIRDEYE_KEY'])

# def supply_demand_zones(config, CONTRACT_ADDRESS, days_back_4_data, timeframe):
# ... (ensure this is correct, calls get_data(config, ...))

# def check_trend(config, symbol): # (nice_funcs2.py version)
# ... (ensure this is correct, calls token_price(config,...) and get_data(config,...), uses config for SMA params)

def get_ohlcv_hl(symbol, interval, lookback_days):
    """
    Get OHLCV data from Hyperliquid
    """
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

def process_data_to_df_hl(snapshot_data, time_period=20):
    """
    Process Hyperliquid OHLCV data into a DataFrame
    """
    if snapshot_data:
        try:
            # Assuming the response contains a list of candles
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            data = []
            for snapshot in snapshot_data:
                timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                # Explicitly convert to float
                open_price = float(snapshot['o'])
                high_price = float(snapshot['h'])
                low_price = float(snapshot['l'])
                close_price = float(snapshot['c'])
                volume = float(snapshot['v'])
                data.append([timestamp, open_price, high_price, low_price, close_price, volume])

            df = pd.DataFrame(data, columns=columns)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)  # Ensure data is sorted by time

            # Calculate rolling support and resistance
            df['support'] = df['close'].rolling(window=time_period, min_periods=1).min().shift(1)
            df['resis'] = df['close'].rolling(window=time_period, min_periods=1).max().shift(1)

            return df
        except Exception as e:
            cprint(f'❌ Error processing Hyperliquid data: {str(e)}', 'red')
            return pd.DataFrame()  # Return empty DataFrame on error
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data

def ask_bid_hl(symbol):
    """
    Get ask/bid data from Hyperliquid
    """
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {
        'type': 'l2Book',
        'coin': symbol
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']

    # get bid and ask 
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data

# Update check_market_conditions to use the new HL functions
def check_market_conditions():
    """
    Check if price is above BOTH 20 AND 40 SMA on the daily timeframe for ETH, SOL, and BTC.
    A symbol must be above both SMAs to be considered bullish.
    Returns True if at least 2 out of 3 symbols are above both SMAs.
    """
    major_symbols = ['ETH', 'SOL', 'BTC']
    fully_above_count = 0
    SMA_20 = 20
    SMA_40 = 40
    TIMEFRAME = '1d'
    LOOKBACK_DAYS = 60  # Get enough days for SMA calculation
    
    cprint('\n🌙 MoonDev Major Market Analysis - 20 & 40 SMA Daily', 'cyan', attrs=['bold'])
    cprint('=' * 50, 'cyan')
    cprint(f'📊 Checking both {SMA_20} & {SMA_40} SMA on Daily timeframe', 'yellow')
    cprint('Must be above BOTH SMAs to be considered bullish', 'yellow')
    
    for symbol in major_symbols:
        try:
            # Get daily OHLCV data for the symbol using Hyperliquid
            snapshot_data = get_ohlcv_hl(symbol, TIMEFRAME, LOOKBACK_DAYS)
            if not snapshot_data:
                cprint(f'❌ Could not get daily data for {symbol}', 'red')
                continue
                
            df = process_data_to_df_hl(snapshot_data)
            if df.empty:
                cprint(f'❌ Empty dataframe for {symbol}', 'red')
                continue
            
            # Calculate both SMAs
            df[f'SMA_{SMA_20}'] = df['close'].rolling(window=SMA_20).mean()
            df[f'SMA_{SMA_40}'] = df['close'].rolling(window=SMA_40).mean()
            
            # Get current price and SMAs
            current_price = float(df['close'].iloc[-1])
            current_sma_20 = float(df[f'SMA_{SMA_20}'].iloc[-1])
            current_sma_40 = float(df[f'SMA_{SMA_40}'].iloc[-1])
            
            # Check if price is above both SMAs
            above_sma_20 = current_price > current_sma_20
            above_sma_40 = current_price > current_sma_40
            is_fully_above = above_sma_20 and above_sma_40
            
            if is_fully_above:
                fully_above_count += 1
                cprint(f'✅ {symbol} above BOTH SMAs:', 'green')
                cprint(f'   Price: ${current_price:.2f}', 'green')
                cprint(f'   20 SMA: ${current_sma_20:.2f}', 'green')
                cprint(f'   40 SMA: ${current_sma_40:.2f}', 'green')
            else:
                status_20 = "✅" if above_sma_20 else "❌"
                status_40 = "✅" if above_sma_40 else "❌"
                cprint(f'❌ {symbol} not above both SMAs:', 'red')
                cprint(f'   Price: ${current_price:.2f}', 'yellow')
                cprint(f'   20 SMA: ${current_sma_20:.2f} {status_20}', 'yellow')
                cprint(f'   40 SMA: ${current_sma_40:.2f} {status_40}', 'yellow')
                
        except Exception as e:
            cprint(f'❌ Error processing {symbol}: {str(e)}', 'red')
            continue
    
    market_conditions_good = fully_above_count >= 2
    cprint(f'\n{"✅ Market conditions favorable" if market_conditions_good else "❌ Market conditions unfavorable"}', 
           'green' if market_conditions_good else 'red', attrs=['bold'])
    cprint(f'📊 {fully_above_count}/3 major symbols above BOTH 20 & 40 SMA Daily', 'yellow')
    
    return market_conditions_good




    


    

def get_hyperliquid_daily_data(symbol, lookback_days):
    """
    Get daily OHLCV data from Hyperliquid and return as pandas DataFrame
    """
    try:
        # Get raw data from Hyperliquid
        snapshot_data = get_ohlcv_hl(symbol, '1d', lookback_days)
        if not snapshot_data:
            cprint(f'❌ No data returned from Hyperliquid for {symbol}', 'red')
            return None

        # Convert to DataFrame
        data = []
        for candle in snapshot_data:
            data.append({
                'timestamp': datetime.fromtimestamp(candle['t'] / 1000),
                'Open': float(candle['o']),
                'High': float(candle['h']),
                'Low': float(candle['l']),
                'Close': float(candle['c']),
                'Volume': float(candle['v'])
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)  # Ensure data is sorted by time

        cprint(f'✅ Got {len(df)} days of data for {symbol}', 'green')
        return df

    except Exception as e:
        cprint(f'❌ Error getting daily data for {symbol}: {str(e)}', 'red')
        return None




    


    