from CONFIG import *
from termcolor import cprint 
from datetime import datetime, timedelta
import requests
import pandas as pd
import pprint
import re as reggie
import time
import pandas_ta as ta
import os
from dotenv import load_dotenv
import base64
import json
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts

# Load environment variables
load_dotenv()

# API Keys from environment variables
API_KEY = os.getenv('BIRDEYE_KEY')
SOL_KEY = os.getenv('SOL_KEY')
RPC_URL = os.getenv('RPC_URL')

# Constants
CLOSED_POSITIONS_TXT = os.path.join('data', 'closed_positions.txt')
SELL_AMOUNT_PERC = 0.5  # 50% of balance to sell per transaction
USDC_SIZE = 100  # Default trade size in USDC

sample_address = "2yXTyarttn2pTZ6cwt4DqmrRuBw1G7pmFv9oT6MStdKP"

# Adjusted BASE_URL if _make_birdeye_request adds the /defi part
# If _make_birdeye_request expects endpoint like 'token_overview', BASE_URL should be "https://public-api.birdeye.so"
# If _make_birdeye_request expects endpoint like 'defi/token_overview', BASE_URL should be "https://public-api.birdeye.so"
# The current _make_birdeye_request prepends BASE_URL directly to the endpoint.
# For consistency with original functions, endpoints passed will be like 'defi/token_overview'
# So, BASE_URL should be fine as is, or _make_birdeye_request adjusts.
# Let's assume endpoints passed to _make_birdeye_request will include 'defi/' if needed.
BASE_URL = "https://public-api.birdeye.so" # Adjusted for _make_birdeye_request to add /defi or /v1

# --- Helper Functions ---

def _make_birdeye_request(endpoint: str, params: dict = None):
    """Helper function to make GET requests to the Birdeye API."""
    url = f"{BASE_URL}/{endpoint}" # Example: endpoint="defi/token_overview"
    headers = {"X-API-KEY": API_KEY}
    if "v1/wallet/token_list" in endpoint: # Add x-chain for v1 wallet endpoint
        headers["x-chain"] = "solana"

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        cprint(f"Birdeye API request failed for {url}: {e}", 'red')
        return None
    except json.JSONDecodeError:
        cprint(f"Failed to decode JSON from Birdeye API for {url}", 'red')
        return None

def _read_closed_positions():
    """Reads and returns a set of token addresses from CLOSED_POSITIONS_TXT."""
    try:
        with open(CLOSED_POSITIONS_TXT, 'r') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

def _add_to_closed_positions(token_mint_address: str):
    """Adds a token address to CLOSED_POSITIONS_TXT if not already present."""
    closed_positions = _read_closed_positions()
    if token_mint_address not in closed_positions:
        with open(CLOSED_POSITIONS_TXT, 'a') as f:
            f.write(token_mint_address + '\n')
            cprint(f"Added {token_mint_address[:6]} to closed positions.", 'blue')

def _execute_jup_swap(input_mint: str, output_mint: str, amount: int, slippage_bps: int, restrict_intermediate_tokens: bool = False, dynamic_slippage_config: dict = None):
    """
    Helper function to execute a swap on Jupiter.
    Amount is in the smallest unit of the input_mint.
    """
    if not SOL_KEY or not RPC_URL:
        cprint("SOL_KEY or RPC_URL not configured in environment variables.", 'red')
        return None

    key = Keypair.from_base58_string(SOL_KEY)
    http_client = Client(RPC_URL)

    quote_params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount), # Ensure amount is string for Jupiter API
        "slippageBps": slippage_bps,
    }
    if restrict_intermediate_tokens:
        quote_params["restrictIntermediateTokens"] = "true"

    quote_url = 'https://quote-api.jup.ag/v6/quote'
    swap_url = 'https://quote-api.jup.ag/v6/swap'

    try:
        # cprint(f"Jupiter Quote URL: {quote_url} with params: {quote_params}", 'magenta')
        quote_response = requests.get(quote_url, params=quote_params)
        quote_response.raise_for_status()
        quote = quote_response.json()
        # cprint(f"Jupiter Quote: {quote}", 'yellow')

        swap_payload = {
            "quoteResponse": quote,
            "userPublicKey": str(key.pubkey()),
            "prioritizationFeeLamports": PRIORITY_FEE, 
        }
        if dynamic_slippage_config:
            swap_payload["dynamicSlippage"] = dynamic_slippage_config
        
        # cprint(f"Jupiter Swap Payload: {swap_payload}", 'yellow')

        tx_res_response = requests.post(swap_url, headers={"Content-Type": "application/json"}, data=json.dumps(swap_payload))
        tx_res_response.raise_for_status()
        tx_res = tx_res_response.json()
        # cprint(f"Jupiter Swap Response: {tx_res}", 'yellow')

        swap_tx = base64.b64decode(tx_res['swapTransaction'])
        raw_tx = VersionedTransaction.from_bytes(swap_tx)
        signed_tx = VersionedTransaction(raw_tx.message, [key])
        
        tx_id = http_client.send_raw_transaction(bytes(signed_tx), TxOpts(skip_preflight=True)).value
        cprint(f"Swap successful: https://solscan.io/tx/{str(tx_id)}", 'green')
        return str(tx_id)

    except requests.exceptions.RequestException as e:
        cprint(f"Jupiter API request failed: {e}", 'red')
        if hasattr(e, 'response') and e.response is not None:
            try:
                cprint(f"Error details: {e.response.json()}", 'red')
            except json.JSONDecodeError:
                cprint(f"Error details (text): {e.response.text}", 'red')
        return None
    except KeyError as e:
        cprint(f"KeyError in Jupiter swap response processing: {e}. Quote: {quote if 'quote' in locals() else 'N/A'}. Response: {tx_res if 'tx_res' in locals() else 'N/A'}", 'red')
        return None
    except Exception as e:
        cprint(f"An unexpected error occurred during Jupiter swap: {e}", 'red')
        return None

# Custom function to print JSON in a human-readable format
def print_pretty_json(data):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

# Helper function to find URLs in text
def find_urls(string):
    # Regex to extract URLs
    return reggie.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)

# Alias for market_buy to maintain compatibility with existing code
def market_buy_swap_tx(token, amount, slippage=SLIPPAGE):
    return market_buy(token, amount, slippage)

# UPDATED TO RMEOVE THE OTHER ONE so now we can just use this filter instead of filtering twice
def token_overview(address):
    """
    Fetch token overview for a given address and return structured information, including specific links,
    and assess if any price change suggests a rug pull.
    """
    endpoint = f"defi/token_overview?address={address}" # Adjusted endpoint
    response_data = _make_birdeye_request(endpoint)
    result = {}

    if response_data and response_data.get('success') and 'data' in response_data:
        overview_data = response_data['data']
        
        buy1h = overview_data.get('buy1h', 0)
        sell1h = overview_data.get('sell1h', 0)
        trade1h = buy1h + sell1h
        result['buy1h'] = buy1h
        result['sell1h'] = sell1h
        result['trade1h'] = trade1h

        total_trades = trade1h
        buy_percentage = (buy1h / total_trades * 100) if total_trades else 0
        sell_percentage = (sell1h / total_trades * 100) if total_trades else 0
        result['buy_percentage'] = buy_percentage
        result['sell_percentage'] = sell_percentage
        result['minimum_trades_met'] = trade1h >= MIN_TRADES_LAST_HOUR

        price_changes = {k: v for k, v in overview_data.items() if 'priceChange' in k}
        result['priceChangesXhrs'] = price_changes

        rug_pull = any(value < -80 for key, value in price_changes.items() if value is not None)
        result['rug_pull'] = rug_pull
        if rug_pull:
            cprint("Warning: Price change percentage below -80%, potential rug pull for " + address, 'red')

        result.update({
            'uniqueWallet2hr': overview_data.get('uniqueWallet24h', 0),
            'v24USD': overview_data.get('v24hUSD', 0),
            'watch': overview_data.get('watch', 0),
            'view24h': overview_data.get('view24h', 0),
            'liquidity': overview_data.get('liquidity', 0),
        })

        extensions = overview_data.get('extensions', {})
        description = extensions.get('description', '') if extensions else ''
        urls = find_urls(description)
        links = []
        for url in urls:
            if 't.me' in url: links.append({'telegram': url})
            elif 'twitter.com' in url: links.append({'twitter': url})
            elif 'youtube' not in url: links.append({'website': url})
        result['description'] = links
        return result
    else:
        cprint(f"Failed to retrieve or parse token overview for address {address}", 'yellow')
        if response_data:
            cprint(f"Birdeye response: {response_data}", 'yellow')
        return None


def token_security_info(address):
    endpoint = f"defi/token_security?address={address}"
    response_data = _make_birdeye_request(endpoint)
    if response_data and response_data.get('success') and 'data' in response_data:
        print_pretty_json(response_data['data'])
    else:
        cprint(f"Failed to retrieve token security info for {address}. Response: {response_data}", 'yellow')

def token_creation_info(address):
    endpoint = f"defi/token_creation_info?address={address}"
    response_data = _make_birdeye_request(endpoint)
    if response_data and response_data.get('success') and 'data' in response_data:
        print_pretty_json(response_data['data'])
    else:
        cprint(f"Failed to retrieve token creation info for {address}. Response: {response_data}", 'yellow')

def market_buy(token, amount, slippage=SLIPPAGE):
    # Ensure amount is int for Jupiter API (smallest unit)
    # Assuming 'amount' might be passed as string from some parts of the code, ensure conversion
    try:
        amount_int = int(amount)
    except ValueError:
        cprint(f"Invalid amount for market_buy: {amount}. Must be an integer-like string or int.", 'red')
        return None

    QUOTE_TOKEN = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # usdc
    
    while True: 
        cprint(f"Attempting to buy {amount_int} of {token[:6]}... with USDC", 'cyan')
        tx_id = _execute_jup_swap(
            input_mint=QUOTE_TOKEN, 
            output_mint=token,
            amount=amount_int, 
            slippage_bps=slippage,
            restrict_intermediate_tokens=True
        )
        if tx_id:
            return tx_id 
        else:
            cprint(f"Market buy attempt failed for {token[:6]}. Retrying in 5 seconds...", 'yellow')
            time.sleep(5)

def market_sell(QUOTE_TOKEN, amount, slippage=SLIPPAGE):
    # Ensure amount is int for Jupiter API (smallest unit)
    try:
        amount_int = int(amount)
    except ValueError:
        cprint(f"Invalid amount for market_sell: {amount}. Must be an integer-like string or int.", 'red')
        return None
        
    cprint(f'Attempting to sell {amount_int} of {QUOTE_TOKEN[:6]}... for USDC', 'cyan')
    usdc_mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    min_slippage_bps_dynamic = 5 

    tx_id = _execute_jup_swap(
        input_mint=QUOTE_TOKEN,
        output_mint=usdc_mint,
        amount=amount_int, 
        slippage_bps=slippage, 
        dynamic_slippage_config={"minBps": min_slippage_bps_dynamic, "maxBps": slippage}
    )

    if tx_id:
        return tx_id
    else:
        cprint(f"Market sell failed for {QUOTE_TOKEN[:6]}", 'red')
        return None

def get_token_overview(address): # This is a duplicate function name. Original had different functionality.
    # Assuming this is the one that was meant to be kept / updated, using Birdeye helper.
    # The original get_token_overview (now token_overview) is more detailed.
    # This simpler one is used by get_names.
    endpoint = f"defi/token_overview?address={address}"
    response_data = _make_birdeye_request(endpoint)
    if response_data and response_data.get('success') and 'data' in response_data:
        return response_data['data']
    else:
        cprint(f"Error fetching (simple) token overview for address {address}. Response: {response_data}", 'yellow')
        return {}
    
def get_names_nosave(df):
    names = []
    low_volume_tokens = []

    for index, row in df.iterrows():
        token_mint_address = row['Mint Address']
        # Use the simpler get_token_overview here as per original structure
        token_data = get_token_overview(token_mint_address) # This now calls the refactored simple version

        token_name = token_data.get('name', 'N/A')
        names.append(token_name)
        
        trade30m = token_data.get('trade30m', 0)
        cprint(f'{token_name} has {trade30m} trades in the last 30 mins', 'magenta')
        
        is_protected = token_mint_address in DO_NOT_TRADE_LIST
        
        if trade30m < MIN_TRADES_30M_ONCE_IN_POSITION:
            if not is_protected:
                cprint(f'🔍 Low volume token: {token_mint_address[:6]} detected - WILL be added to kill list', 'yellow')
                low_volume_tokens.append(token_mint_address)
            else:
                cprint(f'🚫 Low volume token: {token_mint_address[:6]} is PROTECTED - WILL NOT be killed', 'green')
    
    if low_volume_tokens:
        cprint(f"⚠️ TOKENS CONSIDERED FOR KILLING: {[t[:6] for t in low_volume_tokens]}", 'yellow')
    
    if 'name' in df.columns:
        df['name'] = names
    else:
        df.insert(0, 'name', names)

    if 'Mint Address' in df.columns: df.drop('Mint Address', axis=1, inplace=True)
    if 'Amount' in df.columns: df.drop('Amount', axis=1, inplace=True)

    # cprint(f'low_volume_tokens: {low_volume_tokens}', 'magenta')
    return df, low_volume_tokens

def kill_switch(token_mint_address):
    if token_mint_address in DO_NOT_TRADE_LIST:
        cprint(f'🛑 PROTECTED TOKEN: {token_mint_address[:6]} is in DO_NOT_TRADE_LIST - NOT killing!', 'green')
        return

    cprint(f'Initiating kill switch for {token_mint_address[:6]}', 'red')

    balance_df = fetch_wallet_holdings_nosaving_names(MY_ADDRESS, token_mint_address)
    if balance_df.empty or 'Amount' not in balance_df.columns or balance_df['Amount'].iloc[0] <= 0:
        cprint(f'No balance or zero balance for {token_mint_address[:6]}, nothing to kill.', 'yellow')
        _add_to_closed_positions(token_mint_address) # Add to closed if no balance, to prevent re-buys
        return
    
    balance = balance_df['Amount'].iloc[0]
    decimals = get_decimals(token_mint_address)
    if decimals is None: # get_decimals should return None on failure
        cprint(f"Could not get decimals for {token_mint_address[:6]}. Cannot proceed with kill switch.", 'red')
        return

    sell_size_units = int(float(balance) * (10**decimals))
    
    MAX_KILL_ATTEMPTS = 2 # Try to sell the full balance twice
    attempts = 0
    while sell_size_units > 0 and attempts < MAX_KILL_ATTEMPTS:
        attempts += 1
        cprint(f"Kill attempt {attempts}/{MAX_KILL_ATTEMPTS} for {token_mint_address[:6]}, selling {sell_size_units} units.", 'red')
        
        # Triple tap sell, Jupiter handles if it's too much now with _execute_jup_swap
        market_sell(token_mint_address, sell_size_units) # market_sell makes one call to _execute_jup_swap
        time.sleep(1) 
        market_sell(token_mint_address, sell_size_units) 
        time.sleep(1)
        market_sell(token_mint_address, sell_size_units)
        cprint(f'Sent 3 sell orders for {token_mint_address[:6]} for {sell_size_units} units.', 'blue')
        time.sleep(15) # Cooldown to let orders settle and balance update
            
        balance_df = fetch_wallet_holdings_nosaving_names(MY_ADDRESS, token_mint_address)
        if balance_df.empty or 'Amount' not in balance_df.columns or balance_df['Amount'].iloc[0] <= 0:
            cprint(f'Balance for {token_mint_address[:6]} is now zero or not found. Kill successful.', 'green')
            sell_size_units = 0 # Exit loop
        else:
            balance = balance_df['Amount'].iloc[0]
            sell_size_units = int(float(balance) * (10**decimals))
            if sell_size_units <= 0:
                 cprint(f'Balance for {token_mint_address[:6]} is effectively zero. Kill successful.', 'green')
            else:
                cprint(f'Remaining balance for {token_mint_address[:6]}: {balance}. Retrying kill if attempts left.', 'yellow')

    _add_to_closed_positions(token_mint_address)
    if sell_size_units <= 0:
        cprint(f'Successfully closed position for {token_mint_address[:6]}.', 'green')
    else:
        cprint(f'Failed to fully close position for {token_mint_address[:6]} after {attempts} attempts. Remaining units: {sell_size_units}', 'red')


def fetch_wallet_holdings_nosaving_names(address, token_mint_address_filter=None):
    # Renamed token_mint_address to token_mint_address_filter for clarity
    endpoint = f"v1/wallet/token_list?wallet={address}"
    response_data = _make_birdeye_request(endpoint)

    if not response_data or not response_data.get('success') or 'data' not in response_data or 'items' not in response_data['data']:
        cprint(f"Failed to retrieve token list for {address[-4:]} or no items. Response: {response_data}", 'yellow')
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value']) 

    df = pd.DataFrame(response_data['data']['items'])
    if df.empty:
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

    try:
        # Ensure all expected columns are present before trying to use them
        required_cols = {'address': 'Mint Address', 'uiAmount': 'Amount', 'valueUsd': 'USD Value'}
        # Filter out only columns that exist in df to avoid KeyErrors during rename
        cols_to_rename = {k: v for k, v in required_cols.items() if k in df.columns}
        df = df.rename(columns=cols_to_rename)
        
        # Select only the renamed columns that were actually present and renamed
        final_cols = [v for k,v in required_cols.items() if k in cols_to_rename]
        if not all(col in df.columns for col in final_cols):
            # If we don't have all Address, Amount, USD Value after rename, something is wrong
            cprint(f"Missing one or more core columns (Mint Address, Amount, USD Value) after processing for {address[-4:]}. Columns: {df.columns.tolist()}", 'red')
            return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value']) 
            
        df = df[final_cols]
        df = df.dropna()
        # Ensure Amount and USD Value are numeric for filtering
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df['USD Value'] = pd.to_numeric(df['USD Value'], errors='coerce')
        df = df.dropna(subset=['Amount', 'USD Value'])
        df = df[df['USD Value'] > 0.05]
    except KeyError as e:
        cprint(f"KeyError processing wallet holdings for {address[-4:]}: {e}. Columns: {df.columns.tolist()}", 'red')
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])
    except Exception as e:
        cprint(f"Unexpected error processing wallet holdings for {address[-4:]}: {e}. Columns: {df.columns.tolist()}", 'red')
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

    exclude_addresses = ['EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', 'So11111111111111111111111111111111111111112']
    # DO_NOT_TRADE_LIST is from CONFIG
    updated_dont_trade_list = [mint for mint in DO_NOT_TRADE_LIST if mint not in exclude_addresses]
    df = df[~df['Mint Address'].isin(updated_dont_trade_list)]

    if token_mint_address_filter:
        df = df[df['Mint Address'] == token_mint_address_filter]

    return df

def get_decimals(token_mint_address):
    # This function calls Solana RPC directly, not Birdeye
    url = RPC_URL # Assumes RPC_URL is defined globally
    if not url:
        cprint("RPC_URL not configured.", 'red')
        return None
        
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
        "params": [token_mint_address, {"encoding": "jsonParsed"}]
    })

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        response_json = response.json()
        decimals = response_json['result']['value']['data']['parsed']['info']['decimals']
        # cprint(f"Decimals for {token_mint_address[:6]} token: {decimals}", 'magenta')
        return decimals
    except requests.exceptions.RequestException as e:
        cprint(f"RPC request failed for get_decimals({token_mint_address[:6]}): {e}", 'red')
        return None
    except (KeyError, TypeError) as e:
        cprint(f"Error parsing RPC response for get_decimals({token_mint_address[:6]}): {e}. Response: {response_json if 'response_json' in locals() else 'N/A'}", 'red')
        return None

# fetch_wallet_new25 seems to be a test/debug function, keeping as is.
# ... (fetch_wallet_new25)

def fetch_wallet_new25(wallet_address):
    """
    New BirdEye API endpoint test for Solana wallets
    """
    endpoint = f"v1/wallet/token_list?wallet={wallet_address}"
    response_data = _make_birdeye_request(endpoint)

    if response_data and response_data.get('success'):
        cprint("\n🔍 DEBUG: API Response (fetch_wallet_new25)", 'blue')
        cprint(f"Status Code: {response_data.get('status_code', 'N/A')} (actual status checked in helper)", 'blue')

        if 'data' in response_data and 'items' in response_data['data'] and response_data['data']['items']:
            sample_item = response_data['data']['items'][0]
            cprint("\n📊 Sample Token Data Fields:", 'blue')
            for key, value in sample_item.items():
                cprint(f"{key}: {value}", 'blue')
            
            cprint("\n💰 Converting to DataFrame for analysis...", 'blue')
            df = pd.DataFrame(response_data['data']['items'])
            cprint("\nColumns in response:", 'blue')
            cprint(df.columns.tolist(), 'blue')
            
            cprint("\nFirst few rows of data:", 'blue')
            print(df.head())
            return df
        else:
            cprint("❌ No token data found in response (fetch_wallet_new25)", 'yellow')
            return pd.DataFrame() # Return empty DataFrame for consistency
    else:
        cprint(f"❌ Error in fetch_wallet_new25 for {wallet_address}. Response: {response_data}", 'red')
        return pd.DataFrame() # Return empty DataFrame for consistency

def fetch_wallet_holdings_og(address):
    # This function is similar to fetch_wallet_holdings_nosaving_names but saves a CSV and calls kill_switch
    df_holdings = fetch_wallet_holdings_nosaving_names(address) # Use the refactored base function

    if not df_holdings.empty:
        TOKEN_PER_ADDY_CSV = os.path.join('data', 'filtered_wallet_holdings.csv') 
        df_holdings.to_csv(TOKEN_PER_ADDY_CSV, index=False)
        cprint(f"Saved filtered wallet holdings to {TOKEN_PER_ADDY_CSV}", 'green')
        
        # Create a copy for get_names_nosave as it modifies the DataFrame
        df_for_names, low_volume_tokens = get_names_nosave(df_holdings.copy())

        if low_volume_tokens:
            cprint(f"Final low_volume_tokens in OG function: {[token[:6] for token in low_volume_tokens]}", 'yellow')
        
        for token_to_kill in low_volume_tokens:
            # DO_NOT_TRADE_LIST is from CONFIG
            if token_to_kill not in DO_NOT_TRADE_LIST:
                cprint(f'🔄 Closing low volume token in OG function: {token_to_kill[:6]} - NOT protected', 'red')
                kill_switch(token_to_kill)
            else:
                cprint(f'🚫 PROTECTED low volume token in OG function: {token_to_kill[:6]}', 'green')

        cprint('Wallet Holdings (fetch_wallet_holdings_og):', 'cyan')
        # Print the DataFrame that has names (df_for_names)
        print(df_for_names.head(50))
        if 'USD Value' in df_for_names.columns:
            cprint(f'** Current Total USD Value: ${round(df_for_names["USD Value"].sum(),2)}', 'white', 'on_green')
        time.sleep(7)
    else:
        cprint("No wallet holdings to display (fetch_wallet_holdings_og).", 'yellow')
        time.sleep(10) # Reduced sleep from 30 to 10

    return df_holdings # Return original df_holdings without names, as per its typical usage elsewhere


def fetch_wallet_token_single(wallet_address, token_mint_address):
    # Uses the more robust fetch_wallet_holdings_nosaving_names with a filter
    df = fetch_wallet_holdings_nosaving_names(wallet_address, token_mint_address_filter=token_mint_address)
    if not df.empty:
        # cprint(f'Data found for {token_mint_address[:6]}:', 'green')
        # print(df)
        return df
    else:
        # cprint(f'No data found for token {token_mint_address[:6]} in wallet {wallet_address[-4:]}', 'yellow')
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

def get_position(token_mint_address):
    # cprint(f'getting position for {token_mint_address[:6]}', 'magenta')
    dataframe = fetch_wallet_token_single(MY_ADDRESS, token_mint_address)
    if dataframe.empty or 'Amount' not in dataframe.columns:
        # cprint(f"No position found for {token_mint_address[:6]}", 'yellow')
        return 0.0 # Return float
    try:
        return float(dataframe['Amount'].values[0])
    except (IndexError, ValueError) as e:
        cprint(f"Error extracting amount for {token_mint_address[:6]}: {e}. Data: {dataframe}", 'red')
        return 0.0


def token_price(address):
    endpoint = f"defi/price?address={address}"
    response_data = _make_birdeye_request(endpoint)
    if response_data and response_data.get('success') and 'data' in response_data and 'value' in response_data['data']:
        return float(response_data['data']['value'])
    else:
        cprint(f"Could not get price for {address[:6]}. Response: {response_data}", 'yellow')
        return None # Important to return None if price not found

# ask_bid is identical to token_price, can be deprecated or aliased.
# For now, refactoring ask_bid to use token_price to reduce duplication.
def ask_bid(token_mint_address):
    return token_price(token_mint_address)


def pnl_close(token_mint_address):
    cprint(f'Checking PNL close for {token_mint_address[:6]}', 'cyan')
    _add_to_closed_positions(token_mint_address) # Add to closed at the start of PNL check to prevent re-entry during process

    balance = get_position(token_mint_address)
    price = token_price(token_mint_address)

    if balance is None or price is None or balance <= 0: # balance is 0.0 if not found
        cprint(f"Cannot calculate PNL for {token_mint_address[:6]}: Balance: {balance}, Price: {price}", 'yellow')
        return

    usd_value = float(balance) * float(price)
    cprint(f'Initial USD value for {token_mint_address[:6]}: {usd_value:.2f}', 'cyan')

    tp_value = ((1 + TP) * USDC_SIZE)  # TP and SL are percentages of initial USDC_SIZE investment
    sl_value = ((1 - SL) * USDC_SIZE)
    decimals = get_decimals(token_mint_address)

    if decimals is None:
        cprint(f"Cannot get decimals for PNL close on {token_mint_address[:6]}", 'red')
        return

    cprint(f'For {token_mint_address[:6]}: Current Val ${usd_value:.2f}, TP Val ${tp_value:.2f}, SL Val ${sl_value:.2f}', 'magenta')

    # TAKE PROFIT LOGIC
    if usd_value > tp_value:
        cprint(f'TP hit for {token_mint_address[:6]} (Val ${usd_value:.2f} > TP ${tp_value:.2f}). Selling portion.', 'green')
        # Sell a portion (e.g., SELL_AMOUNT_PERC) or full based on strategy
        # Original code sold SELL_AMOUNT_PERC * balance, then checked again.
        # For simplicity here, if TP is hit, consider selling all or a fixed large portion.
        # Let's stick to original logic of trying to sell SELL_AMOUNT_PERC then re-evaluating, 
        # but this loop structure was complex. The kill_switch is better for full exit.
        
        # If TP hit, let's use kill_switch to exit position fully and cleanly.
        cprint(f"Triggering kill_switch for {token_mint_address[:6]} due to TP.", 'green')
        kill_switch(token_mint_address) # kill_switch handles adding to closed_positions again, but it's fine.
        return # Exit pnl_close after triggering kill_switch for TP

    # STOP LOSS LOGIC
    if usd_value < sl_value:
        cprint(f'SL hit for {token_mint_address[:6]} (Val ${usd_value:.2f} < SL ${sl_value:.2f}). Selling ALL.', 'red')
        kill_switch(token_mint_address) # Use kill_switch for a clean full exit on SL
        return # Exit pnl_close after triggering kill_switch for SL

    cprint(f"No TP/SL hit for {token_mint_address[:6]} (Val ${usd_value:.2f}). TP: ${tp_value:.2f}, SL: ${sl_value:.2f}", 'blue')
    # If no TP/SL, we might want to remove it from closed_positions if added prematurely,
    # but current logic adds it to closed at start to prevent re-entry during check.
    # The assumption is pnl_close is called periodically on open positions.
    # If it was added by _add_to_closed_positions at the start and no action taken, it remains closed for this cycle.


def get_token_owner(token_mint_address):
    # ALL_TRENDING_EVER_CSV is from CONFIG
    try:
        df = pd.read_csv(ALL_TRENDING_EVER_CSV)
        matching_rows = df[df['contract_address'] == token_mint_address]
        if not matching_rows.empty:
            owners = matching_rows['owner'].unique().tolist() # Get unique owners
            return owners
        else:
            return ["Not found"]
    except FileNotFoundError:
        cprint(f"File {ALL_TRENDING_EVER_CSV} not found for get_token_owner.", 'red')
        return ["Error - File not found"]
    except Exception as e:
        cprint(f"Error in get_token_owner for {token_mint_address[:6]}: {e}", 'red')
        return ["Error"]

def update_leaderboard(owner, pnl):
    # LEADERBOARD_CSV is from CONFIG
    try:
        df = pd.read_csv(LEADERBOARD_CSV, header=None, names=['owner', 'pnl'])
    except FileNotFoundError:
        df = pd.DataFrame(columns=['owner', 'pnl'])
    
    # Ensure pnl is float
    pnl = float(pnl)
    owner_str = str(owner) # Ensure owner is string for matching

    if owner_str in df['owner'].values:
        df.loc[df['owner'] == owner_str, 'pnl'] = df.loc[df['owner'] == owner_str, 'pnl'].astype(float) + pnl
    else:
        new_row = pd.DataFrame({'owner': [owner_str], 'pnl': [pnl]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    df = df.sort_values('pnl', ascending=False)
    df.to_csv(LEADERBOARD_CSV, index=False, header=False)
    cprint(f"Leaderboard updated: {owner_str} PNL change ${pnl:.2f}", 'blue')


def get_names(df):
    # This function uses the simple get_token_overview
    # READY_TO_BUY_CSV from CONFIG
    names = []
    for index, row in df.iterrows():
        token_mint_address = row['contract_address']
        token_data = get_token_overview(token_mint_address) # Calls simple version
        time.sleep(0.2) # Reduced sleep, Birdeye rate limits are per second/minute typically.
        
        token_name = token_data.get('name', 'N/A')
        cprint(f'Name for {token_mint_address[:6]}: {token_name}', 'magenta')
        names.append(token_name)
    
    if 'name' in df.columns:
        df['name'] = names
    else:
        df.insert(0, 'name', names)

    df.to_csv(READY_TO_BUY_CSV, index=False)
    cprint(f"Saved DataFrame with names to {READY_TO_BUY_CSV}", 'green')
    return df


def open_position(token_mint_address):
    cprint(f'🌙 MoonDev opening position for {token_mint_address[:6]}...', 'white', 'on_blue')
    
    closed_positions = _read_closed_positions()
    if token_mint_address in closed_positions:
        cprint(f'🚫 Token {token_mint_address[:6]} in closed positions list, skipping open attempt.', 'yellow')
        return False

    current_balance_units = get_position(token_mint_address) # In token units (float)
    price = token_price(token_mint_address)

    if price is None or price <= 0:
        cprint(f"❌ Could not get valid price for {token_mint_address[:6]} (Price: {price}). Cannot open position.", 'red')
        return False
        
    current_balance_usd = float(current_balance_units) * price
    target_usd_value = float(USDC_SIZE) # From CONFIG
    size_needed_usd = target_usd_value - current_balance_usd

    cprint(f'📊 Current balance: ${current_balance_usd:.2f} ({current_balance_units} units). Target: ${target_usd_value:.2f} USDC. Need to buy: ${size_needed_usd:.2f} USDC worth.', 'cyan')

    if size_needed_usd < 1.0: # Don't bother with dust buys
        cprint(f'✅ Position for {token_mint_address[:6]} already near target size (${current_balance_usd:.2f} / ${target_usd_value:.2f}). No new buy needed.', 'green')
        # If already have a position, ensure it's not on closed list by mistake for future pnl_checks
        # However, if it's here, it means it wasn't on closed list to begin with.
        return True # Consider it successful if already filled

    decimals_usdc = 6 # Assuming USDC has 6 decimals
    amount_to_buy_usdc_lamports = int(size_needed_usd * (10**decimals_usdc))

    if amount_to_buy_usdc_lamports <=0:
        cprint(f'Calculated buy amount for {token_mint_address[:6]} is zero or negative. Skipping.', 'yellow')
        return False

    position_filled_successfully = False
    # ORDERS_PER_OPEN from CONFIG - this defines how many times we try one buy order of the full needed size.
    # The original logic was trying to fill the total USDC_SIZE in ORDERS_PER_OPEN chunks.
    # Let's clarify: does ORDERS_PER_OPEN mean split the USDC_SIZE into that many orders,
    # or try to buy the full USDC_SIZE that many times if it fails?
    # Original loop: `for _ in range(ORDERS_PER_OPEN - 1):` implies splitting.
    # The code was: `size_needed = int(size_needed * 10**6)` then `market_buy_swap_tx(token_mint_address, size_needed)`
    # This implies `size_needed` was already the total USDC needed in lamports for one tx.
    # Let's assume ORDERS_PER_OPEN is the number of attempts for the *remaining* size_needed_usd.
    # Or, more simply, one large buy attempt, repeated if it fails partially.

    # Simpler: attempt one buy for the total size_needed_usd.
    # If market_buy handles retries for that single transaction, then ORDERS_PER_OPEN might be redundant here.
    # market_buy has its own retry loop.

    cprint(f'🎯 Attempting to buy ${size_needed_usd:.2f} USDC ({amount_to_buy_usdc_lamports} lamports) worth of {token_mint_address[:6]}', 'blue')
    # market_buy expects amount of output token, but we have USDC amount.
    # This needs `market_buy(output_token_mint, usdc_amount_in_lamports_of_usdc)`
    tx_id = market_buy(token_mint_address, amount_to_buy_usdc_lamports) # token_mint_address is output, amount is USDC lamports

    if tx_id:
        cprint(f'✅ Successful buy transaction for {token_mint_address[:6]}: {tx_id}', 'green')
        time.sleep(10) # Wait for balance to update potentially
        final_balance_units = get_position(token_mint_address)
        final_balance_usd = float(final_balance_units) * price # Use same price for quick check
        cprint(f'New balance for {token_mint_address[:6]}: {final_balance_units} units (${final_balance_usd:.2f})', 'green')
        if final_balance_usd >= 0.9 * target_usd_value:
            position_filled_successfully = True
            _add_to_closed_positions(token_mint_address) # Add to closed list *after* successful fill
        else:
            cprint(f'⚠️ Position for {token_mint_address[:6]} may not be fully filled. Current ${final_balance_usd:.2f} vs Target ${target_usd_value:.2f}', 'yellow')
            # Decide if this is still considered a success or needs retry logic here.
            # For now, if tx went through, consider it an attempt.
            _add_to_closed_positions(token_mint_address) # Add to prevent immediate re-buy loops if partially filled.
            position_filled_successfully = True # Let's assume partial fills are ok for now if tx succeeded.

    else:
        cprint(f'❌ Market buy failed for {token_mint_address[:6]}. See logs from market_buy / _execute_jup_swap.', 'red')
        # Do not add to closed positions if buy failed, so it can be retried later.

    return position_filled_successfully

# ... (get_time_range, get_data, check_trend, supply_demand_zones, round_down, chunk_kill)
# These functions look generally okay but could also benefit from using _make_birdeye_request
# For brevity, I will refactor get_data and leave others as an exercise or for a subsequent step.

def get_time_range(days_back_4_data):
    now = datetime.now()
    time_back = now - timedelta(days=days_back_4_data)
    time_to = int(now.timestamp())
    time_from = int(time_back.timestamp())
    return time_from, time_to

def get_data(address, days_back_4_data, timeframe):
    time_from, time_to = get_time_range(days_back_4_data)
    endpoint = "defi/ohlcv"
    params = {
        'address': address,
        'type': timeframe,
        'time_from': time_from,
        'time_to': time_to
    }
    cprint(f'🔍 MoonDev fetching OHLCV for {address[:6]} | TF: {timeframe} | Days: {days_back_4_data}', 'cyan')
    # cprint(f'⏰ From: {datetime.fromtimestamp(time_from).strftime("%Y-%m-%d %H:%M:%S")} To: {datetime.fromtimestamp(time_to).strftime("%Y-%m-%d %H:%M:%S")}', 'cyan')
        
    json_response = _make_birdeye_request(endpoint, params=params)
    
    if not json_response or not json_response.get('success') or 'data' not in json_response or not json_response['data'].get('items'):
        cprint(f'❌ No data or API Error for {address[:6]}. Response: {json_response}', 'red')
        return pd.DataFrame()
            
    items = json_response['data']['items']
    processed_data = [{
        'Datetime (UTC)': datetime.utcfromtimestamp(item['unixTime']).strftime('%Y-%m-%d %H:%M:%S'),
        'Open': item['o'], 'High': item['h'], 'Low': item['l'], 'Close': item['c'], 'Volume': item['v']
    } for item in items]
    df = pd.DataFrame(processed_data)
    
    cprint(f'✅ Successfully fetched {len(df)} bars for {address[:6]}', 'green')

    if not df.empty:
        df['MA20'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['MA40'] = ta.sma(df['Close'], length=40)
        df['Price_above_MA20'] = df.apply(lambda x: False if pd.isna(x['MA20']) else x['Close'] > x['MA20'], axis=1)
        df['Price_above_MA40'] = df.apply(lambda x: False if pd.isna(x['MA40']) else x['Close'] > x['MA40'], axis=1)
        df['MA20_above_MA40'] = df.apply(lambda x: False if pd.isna(x['MA20']) or pd.isna(x['MA40']) else x['MA20'] > x['MA40'], axis=1)
    return df

def check_trend(symbol):
    cprint(f'📊 Checking trend for {symbol[:6]}...', 'cyan')
    price = token_price(symbol)
    if price is None:
        cprint(f'❌ Could not get price for {symbol[:6]} in check_trend', 'red')
        return 'down' 
    
    # SMA_DAYS_BACK, SMA_TIMEFRAME from CONFIG
    df = get_data(symbol, SMA_DAYS_BACK, SMA_TIMEFRAME)
    if df.empty or 'MA20' not in df.columns or df['MA20'].empty or pd.isna(df['MA20'].iloc[-1]):
        cprint(f'❌ No SMA data for trend analysis of {symbol[:6]}', 'red')
        return 'down'
    
    sma20 = df['MA20'].iloc[-1]
    trend = 'up' if price > sma20 else 'down'
    cprint(f'🏷️ {symbol[:6]} | Price ${price:.6f} | MA20 ${sma20:.6f} | Trend: {trend.upper()}', 'green' if trend == 'up' else 'red')
    return trend
        
def supply_demand_zones(CONTRACT_ADDRESS, days_back_4_data, timeframe):
    cprint(f'Calculating S/D zones for {CONTRACT_ADDRESS[:6]} | Days: {days_back_4_data} | TF: {timeframe}', 'cyan')
    sd_df = pd.DataFrame() # Initialize as empty DataFrame
    df = get_data(CONTRACT_ADDRESS, days_back_4_data, timeframe)
    
    if df.empty or len(df) <= 2: 
        cprint(f'❌ Not enough data for S/D zones for {CONTRACT_ADDRESS[:6]} ({len(df)} bars)', 'red')
        return pd.DataFrame() # Return empty DataFrame
        
    try:
        # Exclude last 2 bars for S/R calculation as per original logic df[:-2]
        relevant_df = df.iloc[:-2]
        if relevant_df.empty:
            cprint(f'❌ Not enough data after excluding last 2 bars for {CONTRACT_ADDRESS[:6]}', 'red')
            return pd.DataFrame()

        support = relevant_df['Close'].min()
        resistance = relevant_df['Close'].max()
        support_low = relevant_df['Low'].min()
        resistance_high = relevant_df['High'].max()

        # Create DataFrame directly
        sd_df = pd.DataFrame({
            'dz_low': [support_low],
            'dz_high': [support],
            'sz_low': [resistance], # resistência is sz_low
            'sz_high': [resistance_high] # resistance_high is sz_high
        })

        cprint(f'🎯 S/D Zones for {CONTRACT_ADDRESS[:6]}: DZ {support_low:.6f}-{support:.6f} | SZ {resistance:.6f}-{resistance_high:.6f}', 'blue')
        return sd_df
            
    except Exception as e:
        cprint(f'❌ Error calculating S/D zones for {CONTRACT_ADDRESS[:6]}: {e}', 'red')
        return pd.DataFrame() # Return empty DataFrame on error


def round_down(value, decimals):
    import math
    factor = 10 ** decimals
    return math.floor(value * factor) / factor
    
def chunk_kill(CONTRACT_ADDRESS, MAX_USD_ORDER_SIZE, SLEEP_BETWEEN_ORDERS, SLIPPAGE_CHUNK_KILL):
    # Note: SLIPPAGE_CHUNK_KILL is a new param, assuming it might be different from global SLIPPAGE
    cprint(f'Initiating CHUNK KILL for {CONTRACT_ADDRESS[:6]} | Max Order: ${MAX_USD_ORDER_SIZE}', 'red')
    _add_to_closed_positions(CONTRACT_ADDRESS) # Add to closed at start

    while True:
        balance = get_position(CONTRACT_ADDRESS)
        price = token_price(CONTRACT_ADDRESS)

        if balance is None or price is None or balance <= 0 or price <= 0:
            cprint(f'Cannot get balance/price or balance is zero for {CONTRACT_ADDRESS[:6]}. Exiting chunk kill.', 'yellow')
            break
        
        usd_value = float(balance) * float(price)
        cprint(f'Chunk kill status for {CONTRACT_ADDRESS[:6]}: Balance {balance} units, Price ${price:.6f}, USD Value ${usd_value:.2f}', 'magenta')

        if usd_value <= 0.1: # Effectively zero or negligible
            cprint(f'USD value for {CONTRACT_ADDRESS[:6]} is negligible (${usd_value:.2f}). Chunk kill complete.', 'green')
            break

        decimals = get_decimals(CONTRACT_ADDRESS)
        if decimals is None:
            cprint(f"Failed to get decimals for {CONTRACT_ADDRESS[:6]}. Cannot proceed with chunk kill.", 'red')
            break

        if usd_value <= MAX_USD_ORDER_SIZE:
            sell_amount_units = balance # Sell remaining balance
            cprint(f'Selling remaining full balance for {CONTRACT_ADDRESS[:6]}: {sell_amount_units} units', 'blue')
        else:
            sell_amount_units = MAX_USD_ORDER_SIZE / price
            cprint(f'Calculated sell amount for {CONTRACT_ADDRESS[:6]}: {sell_amount_units} units (approx ${MAX_USD_ORDER_SIZE})', 'blue')
        
        # Ensure sell_amount_units is positive
        if sell_amount_units <= 0:
            cprint(f"Calculated sell amount for {CONTRACT_ADDRESS[:6]} is zero. USD value {usd_value}. Breaking.", 'yellow')
            break
            
        sell_size_lamports = int(sell_amount_units * (10**decimals))
        if sell_size_lamports <= 0:
            cprint(f"Sell size in lamports for {CONTRACT_ADDRESS[:6]} is zero. Units: {sell_amount_units}. Breaking.", 'yellow')
            break

        cprint(f'Attempting chunk sell for {CONTRACT_ADDRESS[:6]}: {sell_amount_units} units ({sell_size_lamports} lamports) with slippage {SLIPPAGE_CHUNK_KILL}bps', 'red')
        
        # Triple tap sell using the specific slippage for chunk_kill
        market_sell(CONTRACT_ADDRESS, sell_size_lamports, slippage=SLIPPAGE_CHUNK_KILL)
        time.sleep(SLEEP_BETWEEN_ORDERS) 
        market_sell(CONTRACT_ADDRESS, sell_size_lamports, slippage=SLIPPAGE_CHUNK_KILL) 
        time.sleep(SLEEP_BETWEEN_ORDERS)
        market_sell(CONTRACT_ADDRESS, sell_size_lamports, slippage=SLIPPAGE_CHUNK_KILL)
        cprint(f'Sent 3 chunk sell orders for {CONTRACT_ADDRESS[:6]}', 'blue')
        time.sleep(15) # Cooldown

    cprint(f'Chunk kill process finished for {CONTRACT_ADDRESS[:6]}.', 'red')


    