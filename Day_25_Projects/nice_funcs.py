"""A collection of utility functions for Solana trading operations, interacting with APIs (Birdeye, Jupiter, OpenAI), and performing calculations."""

# --- Standard Library Imports ---
import os
import json
from datetime import datetime, timedelta
import time
import math
import pprint
import re as reggie
import base64
import asyncio
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Third-party Imports ---
import requests
import pandas as pd
import pandas_ta as ta
import pytz
from termcolor import cprint
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from websockets import connect # For Binance funding rate
from openai import OpenAI

# --- Local Imports ---
# Import specific configuration constants
try:
    from config import (
        MINIMUM_TRADES_IN_LAST_HOUR, 
        SELL_AT_MULTIPLE, 
        USD_SIZE,
        STOP_LOSS_PERCENTAGE,
        DO_NOT_TRADE_LIST,
        PRIORITY_FEE,
        WALLET_ADDRESS # Assuming WALLET_ADDRESS is set in config
        # Add any other config constants used globally in this module
    )
    # Import secrets securely
    from Day_4_Projects import dontshare as d 
    API_KEY = d.birdeye
    SOL_KEY_SECRET = d.sol_key # Your Solana wallet secret key
    RPC_ENDPOINT = d.ankr_key # Your Solana RPC endpoint
    OPENAI_API_KEY = d.openai_key
except ImportError as e:
    print(f"Error importing from config.py or dontshare.py: {e}")
    print("Please ensure both files exist and contain the required variables.")
    exit()
except AttributeError as e:
     print(f"Error: A required variable might be missing from dontshare.py: {e}")
     exit()
except NameError as e:
    print(f"Error: A required variable might be missing or misspelled in config.py: {e}")
    exit()

# --- Constants specific to this module ---
BIRDEYE_BASE_URL = "https://public-api.birdeye.so/defi"
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com/" # Fallback RPC
DEFAULT_WALLET_KEYPAIR = Keypair.from_base58_string(SOL_KEY_SECRET) if SOL_KEY_SECRET else None
RPC_CLIENT = Client(RPC_ENDPOINT if RPC_ENDPOINT else SOLANA_RPC_URL)
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# --- Helper Functions ---

def print_pretty_json(data):
    """Prints JSON data in a readable format."""
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

def find_urls(string):
    """Extracts URLs from a string using regex."""
    # Regex to extract URLs
    return reggie.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(string))

def round_down(value, decimals):
    """Rounds a float down to a specified number of decimal places."""
    factor = 10 ** decimals
    try:
        return math.floor(float(value) * factor) / factor
    except ValueError:
        return 0 # Or handle appropriately
        
def get_time_range(days_back=10):
    """Calculates the timestamp range for the last N days."""
    now = datetime.now()
    start_date = now - timedelta(days=days_back)
    time_to = int(now.timestamp())
    time_from = int(start_date.timestamp())
    return time_from, time_to
    
def calculate_buy_chunk_lamports(target_usd_size, current_pos_usd, max_chunk_usd):
    """Calculates the size of the next buy chunk in lamports (as string)."""
    size_needed_usd = target_usd_size - current_pos_usd
    if size_needed_usd <= 0:
        return "0" # No more needed
        
    chunk_usd = min(size_needed_usd, max_chunk_usd)
    # Assuming 6 decimals for USDC when calculating lamports from USD
    # Ensure MAX_USD_ORDER_SIZE is correctly configured in config.py
    chunk_lamports = int(chunk_usd * 10**6) 
    return str(chunk_lamports)

def attempt_market_buy(symbol_mint, chunk_size_lamports_str, slippage_basis_points, num_orders, order_delay_s):
    """Attempts to place market buy orders, returns True if potentially successful."""
    if chunk_size_lamports_str == "0":
        return True # Nothing to buy
        
    try:
        for i in range(num_orders):
            # Assuming market_buy takes mint address, size in base token lamports (USDC), slippage BPS
            # Ensure market_buy function exists and handles parameters correctly
            tx_id = market_buy(symbol_mint, chunk_size_lamports_str, slippage_basis_points) 
            if tx_id:
                 cprint(f'Chunk buy submitted for {symbol_mint[-6:]}, size: {chunk_size_lamports_str} lamports. Tx: {tx_id}', 'white', 'on_blue')
            else:
                 cprint(f'Chunk buy failed for {symbol_mint[-6:]}, size: {chunk_size_lamports_str} lamports.', 'yellow')
                 # Decide if we should continue trying other orders in the burst or fail early
                 # return False # Option: Fail the whole attempt if one order fails
                 
            if i < num_orders - 1: # Don't sleep after the last order in the burst
                 time.sleep(order_delay_s)
        return True # Submitted orders (or attempted all)
    except Exception as e:
        cprint(f"Error during market buy attempt: {e}", 'yellow', 'on_red')
        return False

# --- Birdeye API Functions ---

def token_overview(token_mint_address):
    """
    Fetches token overview from Birdeye and returns structured information.
    Checks for minimum hourly trades and potential rug pull indicators.
    """
    print(f"Getting Birdeye token overview for {token_mint_address[-6:]}...")
    overview_url = f"{BIRDEYE_BASE_URL}/token_overview?address={token_mint_address}"
    headers = {"X-API-KEY": API_KEY}
    result = {}

    try:
        response = requests.get(overview_url, headers=headers, timeout=10) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        overview_data = response.json().get('data', {})
        if not overview_data: # Handle empty data case
             print(f"Warning: Received empty data in overview for {token_mint_address[-6:]}")
             return None
             
        # --- Calculations ---
        buy1h = overview_data.get('buy1h', 0)
        sell1h = overview_data.get('sell1h', 0)
        trade1h = buy1h + sell1h
        total_trades = trade1h
        buy_percentage = (buy1h / total_trades * 100) if total_trades else 0
        sell_percentage = (sell1h / total_trades * 100) if total_trades else 0
        price_changes = {k: v for k, v in overview_data.items() if 'priceChange' in k}
        # Check for rug pull: significant drop (e.g., >80%) in any recent period
        rug_pull = any(value < -80 for key, value in price_changes.items() if isinstance(value, (int, float)))

        # --- Result Assembly ---
        result['buy1h'] = buy1h
        result['sell1h'] = sell1h
        result['trade1h'] = trade1h
        result['buy_percentage'] = buy_percentage
        result['sell_percentage'] = sell_percentage
        result['minimum_trades_met'] = bool(trade1h >= MINIMUM_TRADES_IN_LAST_HOUR)
        result['priceChangesXhrs'] = price_changes
        result['rug_pull'] = rug_pull
        if rug_pull:
            cprint("Warning: Significant price drop detected, potential rug pull", 'red')

        # Other metrics
        result.update({
            'uniqueWallet2hr': overview_data.get('uniqueWallet24h', 0),
            'v24USD': overview_data.get('v24hUSD', 0),
            'watch': overview_data.get('watch', 0),
            'view24h': overview_data.get('view24h', 0),
            'liquidity': overview_data.get('liquidity', 0),
        })

        # Extract social links from description
        extensions = overview_data.get('extensions', {})
        description = extensions.get('description', '') if extensions else ''
        urls = find_urls(description)
        links = []
        for url in urls:
            if 't.me' in url: links.append({'telegram': url})
            elif 'twitter.com' in url: links.append({'twitter': url})
            # Basic website detection (needs refinement for edge cases)
            elif 'discord' not in url and 'youtube' not in url: 
                links.append({'website': url})
        result['social_links'] = links # Renamed for clarity
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"API Request Error fetching overview for {token_mint_address[-6:]}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"JSON Decode Error fetching overview for {token_mint_address[-6:]}. Response: {response.text[:100]}")
        return None
    except Exception as e:
         print(f"Unexpected error in token_overview for {token_mint_address[-6:]}: {e}")
         return None

def token_price(token_mint_address):
    """Fetches the current price of a token from Birdeye."""
    url = f"{BIRDEYE_BASE_URL}/price?address={token_mint_address}"
    headers = {"X-API-KEY": API_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        price_data = response.json()
        
        if price_data.get('success') and 'value' in price_data.get('data', {}):
            return price_data['data']['value']
        else:
            # print(f"Warning: Could not get price for {token_mint_address[-6:]}. Response: {price_data}")
            return None # Indicate failure clearly
    except requests.exceptions.RequestException as e:
        print(f"API Request Error fetching price for {token_mint_address[-6:]}: {e}")
        return None
    except json.JSONDecodeError:
         print(f"JSON Decode Error fetching price for {token_mint_address[-6:]}. Response: {response.text[:100]}")
         return None
    except Exception as e:
         print(f"Unexpected error in token_price for {token_mint_address[-6:]}: {e}")
         return None

def get_data(token_mint_address, days_back=10, timeframe='1m'):
    """Fetches OHLCV data from Birdeye and adds TA indicators."""
    print(f"Fetching {timeframe} OHLCV data for {token_mint_address[-6:]} ({days_back} days back)...")
    time_from, time_to = get_time_range(days_back)
    url = f"{BIRDEYE_BASE_URL}/ohlcv?address={token_mint_address}&type={timeframe}&time_from={time_from}&time_to={time_to}"
    headers = {"X-API-KEY": API_KEY}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        json_response = response.json()
        items = json_response.get('data', {}).get('items', [])
        
        if not items:
            print(f"Warning: No OHLCV items found for {token_mint_address[-6:]} in the given range.")
            return pd.DataFrame() # Return empty DataFrame

        processed_data = [{
            'datetime': datetime.utcfromtimestamp(item['unixTime']), # Keep as datetime object
            'Open': item['o'],
            'High': item['h'],
            'Low': item['l'],
            'Close': item['c'],
            'Volume': item['v']
        } for item in items]
        
        df = pd.DataFrame(processed_data)
        df['datetime'] = df['datetime'].dt.tz_localize('UTC') # Localize
        df.set_index('datetime', inplace=True)
        
        # Pad if less than 40 rows for indicator calculation
        required_rows_for_ta = 40
        if len(df) < required_rows_for_ta and not df.empty:
            rows_to_add = required_rows_for_ta - len(df)
            first_row_replicated = pd.concat([df.iloc[0:1]] * rows_to_add, ignore_index=False)
            df = pd.concat([first_row_replicated, df])
            df.sort_index(inplace=True) # Ensure chronological order after padding
            
        # Calculate TA indicators
        if not df.empty:
             df['MA20'] = ta.sma(df['Close'], length=20)
             df['RSI'] = ta.rsi(df['Close'], length=14)
             df['MA40'] = ta.sma(df['Close'], length=40)
             df['Price_above_MA20'] = df['Close'] > df['MA20']
             df['Price_above_MA40'] = df['Close'] > df['MA40']
             df['MA20_above_MA40'] = df['MA20'] > df['MA40']
             df.dropna(inplace=True) # Drop rows with NaNs generated by TA indicators
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"API Request Error fetching OHLCV for {token_mint_address[-6:]}: {e}")
        return pd.DataFrame()
    except json.JSONDecodeError:
         print(f"JSON Decode Error fetching OHLCV for {token_mint_address[-6:]}. Response: {response.text[:100]}")
         return pd.DataFrame()
    except Exception as e:
         print(f"Unexpected error in get_data for {token_mint_address[-6:]}: {e}")
         return pd.DataFrame()

# --- Solana Wallet & Blockchain Functions ---

def get_decimals(token_mint_address):
    """Fetches the number of decimals for a given token mint from Solana RPC."""
    print(f"Fetching decimals for {token_mint_address[-6:]}...")
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
        "params": [token_mint_address, {"encoding": "jsonParsed"}]
    })
    try:
        response = requests.post(SOLANA_RPC_URL, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        response_json = response.json()
        decimals = response_json['result']['value']['data']['parsed']['info']['decimals']
        return decimals
    except requests.exceptions.RequestException as e:
        print(f"RPC Request Error fetching decimals for {token_mint_address[-6:]}: {e}")
        return None
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"Error parsing RPC response for decimals ({token_mint_address[-6:]}): {e}. Response: {response.text[:200]}")
        return None
    except Exception as e:
        print(f"Unexpected error in get_decimals for {token_mint_address[-6:]}: {e}")
        return None

def fetch_wallet_holdings_og(wallet_address):
    """Fetches token holdings for a given wallet address from Birdeye."""
    # Use the globally configured API Key
    url = f"https://public-api.birdeye.so/v1/wallet/token_list?wallet={wallet_address}"
    headers = {"x-chain": "solana", "X-API-KEY": API_KEY}
    holdings_df = pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value']) # Initialize empty

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        json_response = response.json()

        if 'data' in json_response and 'items' in json_response['data']:
            items = json_response['data']['items']
            if items: # Check if items list is not empty
                temp_df = pd.DataFrame(items)
                # Select and rename columns robustly
                if all(col in temp_df.columns for col in ['address', 'uiAmount', 'valueUsd']):
                    holdings_df = temp_df[['address', 'uiAmount', 'valueUsd']].copy()
                    holdings_df.rename(columns={'address': 'Mint Address', 'uiAmount': 'Amount', 'valueUsd': 'USD Value'}, inplace=True)
                    holdings_df = holdings_df.dropna(subset=['USD Value']) # Drop only if USD Value is NaN
                    holdings_df = holdings_df[holdings_df['USD Value'] > 0.05]
                else:
                    print("Warning: Expected columns missing in Birdeye wallet response.")
            else:
                 print(f"No token items found for wallet {wallet_address}.")
        else:
            print(f"No 'data' or 'items' key found in Birdeye response for wallet {wallet_address}.")

    except requests.exceptions.RequestException as e:
        cprint(f"API Request Error fetching holdings for {wallet_address}: {e}", 'white', 'on_red')
    except Exception as e:
        cprint(f"Unexpected error in fetch_wallet_holdings_og for {wallet_address}: {e}", 'white', 'on_red')

    if not holdings_df.empty:
        # print(holdings_df)
        cprint(f'** Wallet {wallet_address[-6:]} Total USD: ${holdings_df["USD Value"].sum():.2f}', 'white', 'on_green')
    else:
        cprint(f"No significant wallet holdings found or error occurred for {wallet_address[-6:]}.", 'yellow')
        
    return holdings_df

def fetch_wallet_token_single(wallet_address, token_mint_address):
    """Fetches holdings and filters for a single token mint address."""
    holdings_df = fetch_wallet_holdings_og(wallet_address)
    if holdings_df is not None and not holdings_df.empty:
        # Filter by token mint address (ensure case-insensitivity if needed, though mints usually aren't)
        single_token_df = holdings_df[holdings_df['Mint Address'].str.fullmatch(token_mint_address, case=False)].copy()
        return single_token_df
    else:
        return pd.DataFrame() # Return empty DataFrame if holdings fetch failed or was empty

def get_position(token_mint_address):
    """
    Fetches the balance (amount) of a specific token in the configured wallet.
    Relies on the global WALLET_ADDRESS constant from config.
    """
    if not WALLET_ADDRESS:
        print("Error: WALLET_ADDRESS not configured in config.py.")
        return 0.0
        
    # print(f"Getting position for {token_mint_address[-6:]} in wallet {WALLET_ADDRESS[-6:]}...")
    single_token_df = fetch_wallet_token_single(WALLET_ADDRESS, token_mint_address)

    if not single_token_df.empty:
        balance = single_token_df['Amount'].iloc[0]
        # print(f"  Balance: {balance}")
        try:
            return float(balance)
        except (ValueError, TypeError):
             print(f"Warning: Could not convert balance '{balance}' to float for {token_mint_address[-6:]}")
             return 0.0
    else:
        # print(f"  Token {token_mint_address[-6:]} not found in wallet {WALLET_ADDRESS[-6:]} or error occurred.")
        return 0.0 # Indicate zero balance

# --- Jupiter API Trading Functions ---

def _get_jupiter_quote(input_mint, output_mint, amount_lamports, slippage_bps):
    """Helper to get a quote from Jupiter API."""
    quote_url = f'{JUPITER_QUOTE_URL}?inputMint={input_mint}&outputMint={output_mint}&amount={amount_lamports}&slippageBps={slippage_bps}'
    try:
        response = requests.get(quote_url, timeout=10)
        response.raise_for_status()
        quote_data = response.json()
        # Basic validation of quote response
        if 'outAmount' not in quote_data:
             print(f"Warning: Jupiter quote missing 'outAmount'. Response: {quote_data}")
             return None
        return quote_data
    except requests.exceptions.RequestException as e:
        print(f"Jupiter Quote API Request Error: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Jupiter Quote JSON Decode Error. Response: {response.text[:100]}")
        return None
    except Exception as e:
        print(f"Unexpected error getting Jupiter quote: {e}")
        return None

def _perform_jupiter_swap(quote_response, keypair, priority_fee_lamports):
    """Helper to perform a swap using Jupiter API and a quote response."""
    if not quote_response or not keypair:
        return None
        
    payload = json.dumps({
        "quoteResponse": quote_response,
        "userPublicKey": str(keypair.pubkey()),
        "wrapAndUnwrapSol": True, # Usually desired
        "prioritizationFeeLamports": priority_fee_lamports
    })
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(JUPITER_SWAP_URL, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        swap_data = response.json()
        
        if 'swapTransaction' not in swap_data:
             print(f"Warning: Jupiter swap response missing 'swapTransaction'. Response: {swap_data}")
             return None
             
        swap_tx_b64 = swap_data['swapTransaction']
        raw_tx_bytes = base64.b64decode(swap_tx_b64)
        versioned_tx = VersionedTransaction.from_bytes(raw_tx_bytes)
        signed_tx = VersionedTransaction(versioned_tx.message, [keypair]) # Sign with the keypair
        
        # Send the transaction
        tx_id = RPC_CLIENT.send_raw_transaction(bytes(signed_tx), TxOpts(skip_preflight=True)).value
        print(f"  Transaction Sent: https://solscan.io/tx/{str(tx_id)}")
        # Optional: Add confirmation logic here if needed
        return str(tx_id)
        
    except requests.exceptions.RequestException as e:
        print(f"Jupiter Swap API Request Error: {e}")
        return None
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"Error processing Jupiter swap response: {e}. Response: {response.text[:200]}")
        return None
    except Exception as e:
        print(f"Unexpected error performing Jupiter swap: {e}")
        return None

def market_buy(output_token_mint, usdc_amount_lamports_str, slippage_bps):
    """Buys a token using USDC via Jupiter."""
    print(f"Attempting market buy: {usdc_amount_lamports_str} lamports USDC for {output_token_mint[-6:]}, slippage: {slippage_bps} BPS")
    if not DEFAULT_WALLET_KEYPAIR:
         print("Error: Wallet keypair not loaded.")
         return None
         
    quote = _get_jupiter_quote(USDC_MINT, output_token_mint, usdc_amount_lamports_str, slippage_bps)
    if quote:
        return _perform_jupiter_swap(quote, DEFAULT_WALLET_KEYPAIR, PRIORITY_FEE)
    return None

def market_sell(input_token_mint, input_token_amount_lamports_str, slippage_bps):
    """Sells a token for USDC via Jupiter."""
    print(f"Attempting market sell: {input_token_amount_lamports_str} lamports of {input_token_mint[-6:]} for USDC, slippage: {slippage_bps} BPS")
    if not DEFAULT_WALLET_KEYPAIR:
         print("Error: Wallet keypair not loaded.")
         return None
         
    quote = _get_jupiter_quote(input_token_mint, USDC_MINT, input_token_amount_lamports_str, slippage_bps)
    if quote:
        return _perform_jupiter_swap(quote, DEFAULT_WALLET_KEYPAIR, PRIORITY_FEE)
    return None

# --- Position Management Functions ---

def pnl_close(token_mint_address):
    """Closes a position based on PnL (Take Profit or Stop Loss)."""
    print(f"Checking PNL close for {token_mint_address[-6:]}...")
    balance_tokens = get_position(token_mint_address)
    if balance_tokens <= 0:
        # print(f"  No position to close for {token_mint_address[-6:]}.")
        return # No position exists

    price = token_price(token_mint_address)
    if price is None:
        print(f"  Cannot check PNL, failed to get price for {token_mint_address[-6:]}.")
        return

    usd_value = balance_tokens * price
    decimals = get_decimals(token_mint_address)
    if decimals is None:
        print(f"  Cannot close PNL, failed to get decimals for {token_mint_address[-6:]}.")
        return

    # Calculate target prices based on config
    initial_investment_usd = USD_SIZE # Assumes initial buy size is configured USD_SIZE
    target_tp_usd = initial_investment_usd * (1 + (SELL_AT_MULTIPLE - 1)) # Simplified TP based on multiple
    target_sl_usd = initial_investment_usd * (1 + STOP_LOSS_PERCENTAGE)

    print(f"  Current Value: ${usd_value:.2f} | Target TP: >= ${target_tp_usd:.2f} | Target SL: <= ${target_sl_usd:.2f}")

    # Determine if TP or SL is hit
    close_reason = None
    if usd_value >= target_tp_usd:
        close_reason = "Take Profit"
    elif usd_value <= target_sl_usd:
        close_reason = "Stop Loss"

    if close_reason:
        cprint(f"  {close_reason} triggered for {token_mint_address[-6:]}! Current value ${usd_value:.2f}. Attempting to close...", 'white', 'on_magenta')
        # Use kill_switch for simplicity, assuming it sells the full balance
        kill_switch(token_mint_address, 500) # Using a moderate 5% slippage for close
        # Add token to dont_overtrade list after closing
        try:
            with open('dont_overtrade.txt', 'a') as file:
                file.write(token_mint_address + '\n')
            print(f"  Added {token_mint_address[-6:]} to dont_overtrade.txt")
        except Exception as e:
            print(f"Error writing to dont_overtrade.txt: {e}")
    # else:
        # print(f"  No PNL close condition met for {token_mint_address[-6:]}.")

def chunk_kill(token_mint_address, max_usd_sell_size, slippage_bps):
    """Closes a position gradually in chunks up to max_usd_sell_size per attempt."""
    print(f"Starting chunk kill for {token_mint_address[-6:]}, max chunk ${max_usd_sell_size:.2f}...")
    max_retries = 3
    retry_delay_s = 5

    while True:
        balance_tokens = get_position(token_mint_address)
        if balance_tokens <= 0.000001: # Use a small threshold for floating point
            print(f"  Position for {token_mint_address[-6:]} is effectively zero. Chunk kill complete.")
            break

        price = token_price(token_mint_address)
        if price is None or price <= 0:
            print(f"  Cannot execute chunk kill, invalid price ({price}) for {token_mint_address[-6:]}. Retrying price...")
            time.sleep(retry_delay_s)
            continue

        usd_value = balance_tokens * price
        print(f"  Current balance: {balance_tokens:.6f} tokens (${usd_value:.2f})")

        decimals = get_decimals(token_mint_address)
        if decimals is None:
            print(f"  Cannot execute chunk kill, failed to get decimals for {token_mint_address[-6:]}. Aborting.")
            break

        # Calculate sell size in tokens
        if usd_value <= max_usd_sell_size:
            sell_size_tokens = balance_tokens # Sell the remainder
        else:
            sell_size_tokens = max_usd_sell_size / price

        # Ensure sell size respects decimals and minimums, round down
        sell_size_tokens_rounded = round_down(sell_size_tokens, decimals) 
        if sell_size_tokens_rounded <= 0:
             print("  Calculated sell size is too small after rounding. Likely residual dust. Ending chunk kill.")
             break
             
        sell_size_lamports_str = str(int(sell_size_tokens_rounded * (10**decimals)))
        print(f"  Attempting to sell chunk: {sell_size_tokens_rounded:.{decimals}f} tokens ({sell_size_lamports_str} lamports)")

        # Try selling the chunk with retries
        sell_success = False
        for attempt in range(max_retries):
            tx_id = market_sell(token_mint_address, sell_size_lamports_str, slippage_bps)
            if tx_id:
                cprint(f"  Sell chunk submitted (Attempt {attempt+1}). Tx: {tx_id}", 'white', 'on_blue')
                sell_success = True
                break # Exit retry loop on success
            else:
                cprint(f"  Sell chunk failed (Attempt {attempt+1}). Retrying in {retry_delay_s}s...", 'yellow')
                time.sleep(retry_delay_s)
        
        if not sell_success:
            cprint(f"  Failed to sell chunk for {token_mint_address[-6:]} after {max_retries} attempts. Aborting chunk kill.", 'red')
            break # Exit outer while loop if sell fails after retries
            
        # Wait for blockchain state to potentially update
        print(f"  Waiting briefly after sell attempt...")
        time.sleep(10) # Increased delay after successful sell attempt

    print(f"Chunk kill process finished for {token_mint_address[-6:]}.")

def kill_switch(token_mint_address, slippage_bps):
    """Closes the entire position for a token as quickly as possible."""
    # Note: This implementation is similar to chunk_kill but aims for full closure.
    # Consider if a simpler full balance sell attempt is sufficient.
    print(f"!!! KILL SWITCH ACTIVATED for {token_mint_address[-6:]} !!!")
    balance_tokens = get_position(token_mint_address)
    if balance_tokens <= 0.000001:
        print(f"  No position found for {token_mint_address[-6:]}.")
        return

    decimals = get_decimals(token_mint_address)
    if decimals is None:
        print(f"  Cannot execute kill switch, failed to get decimals for {token_mint_address[-6:]}.")
        return

    sell_size_lamports_str = str(int(balance_tokens * (10**decimals)))
    print(f"  Attempting to sell full balance: {balance_tokens:.{decimals}f} tokens ({sell_size_lamports_str} lamports)")

    # Try selling the full amount with retries
    max_retries = 5 # More retries for kill switch
    retry_delay_s = 3
    sell_success = False
    for attempt in range(max_retries):
        tx_id = market_sell(token_mint_address, sell_size_lamports_str, slippage_bps)
        if tx_id:
            cprint(f"  Kill switch sell submitted (Attempt {attempt+1}). Tx: {tx_id}", 'white', 'on_red')
            sell_success = True
            break
        else:
            cprint(f"  Kill switch sell failed (Attempt {attempt+1}). Retrying in {retry_delay_s}s...", 'yellow')
            time.sleep(retry_delay_s)
    
    if not sell_success:
        cprint(f"!!! KILL SWITCH FAILED for {token_mint_address[-6:]} after {max_retries} attempts. Manual intervention likely needed. !!!", 'red', attrs=['bold'])
    else:
        print(f"Kill switch process finished for {token_mint_address[-6:]}.")

def close_all_positions(wallet_address):
    """Fetches all wallet holdings and attempts to close non-excluded positions."""
    print(f"--- Attempting to close all positions for wallet {wallet_address[-6:]} --- ")
    open_positions_df = fetch_wallet_holdings_og(wallet_address)

    if open_positions_df is None or open_positions_df.empty:
        print("No positions found or error fetching holdings. Nothing to close.")
        return

    # Loop through positions and close if not in DO_NOT_TRADE_LIST
    closed_count = 0
    for index, row in open_positions_df.iterrows():
        token_mint_address = row['Mint Address']
        usd_value = row['USD Value']
        
        if token_mint_address in DO_NOT_TRADE_LIST:
            # print(f"Skipping excluded token: {token_mint_address}")
            continue 
        
        if usd_value > 0.10: # Only attempt to close if value is somewhat significant
             print(f"Closing position for {token_mint_address[-6:]} (Value: ${usd_value:.2f})...")
             kill_switch(token_mint_address, 1000) # Use higher slippage (e.g., 10%) for emergency close
             closed_count += 1
             time.sleep(5) # Small delay between closing different tokens
        # else: 
            # print(f"Skipping token {token_mint_address[-6:]} due to low value (${usd_value:.2f}).")
            
    print(f"--- Close all positions process finished. Attempted to close {closed_count} positions. ---")

# --- File System Functions ---
def delete_dont_overtrade_file(filename="dont_overtrade.txt"):
    """Deletes the specified file if it exists."""
    if os.path.exists(filename):
        try:
            os.remove(filename)
            print(f"'{filename}' has been deleted.")
        except OSError as e:
             print(f"Error deleting file '{filename}': {e}")
    # else:
        # print(f"File '{filename}' does not exist, nothing to delete.")

# --- Analysis Functions ---
def supply_demand_zones(token_address, timeframe='1h', limit=50):
    """Calculates simple support/resistance zones based on recent OHLC data."""
    print(f'Calculating Supply/Demand zones for {token_address[-6:]} ({timeframe}, {limit} periods)...')
    df = get_data(token_address, days_back=int(limit/24)+1, timeframe=timeframe) # Fetch enough days
    if df is None or df.empty or len(df) < 3:
        print("  Not enough data to calculate S/D zones.")
        return None

    # Use last `limit` periods, exclude last 2 for calculation
    df_calc = df.iloc[-(limit+2):-2] 
    if df_calc.empty:
         print("  Not enough historical data (excluding recent bars) for S/D calculation.")
         return None
         
    supp_close = df_calc['Close'].min()
    resis_close = df_calc['Close'].max()
    supp_low = df_calc['Low'].min()
    resis_high = df_calc['High'].max()

    sd_zones = pd.DataFrame({
        f'demand_zone': [supp_low, supp_close],
        f'supply_zone': [resis_high, resis_close]
    })
    # print('Supply/Demand Zones:')
    # print(sd_zones)
    return sd_zones

def should_not_trade(token_address, no_trade_hours_list=[], filename="dont_overtrade.txt"):
    """Checks if trading is disallowed based on time or exclusion file."""
    now_hour = datetime.now().hour # Assumes local time for hour check
    if now_hour in no_trade_hours_list:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as file:
                    if token_address in file.read():
                        print(f"Trading disallowed for {token_address[-6:]}: Within no-trade hour ({now_hour}) AND in exclusion file.")
                        return True # Excluded due to time AND file
            except Exception as e:
                 print(f"Error reading exclusion file '{filename}': {e}")
    return False # Allowed by default

# --- AI / Experimental Functions ---
def vibe_check(token_name):
    """Uses OpenAI GPT-4 to get a meme-worthiness score (1-10) for a token name."""
    if not OPENAI_CLIENT:
         print("OpenAI client not initialized. Skipping vibe check.")
         return None
         
    prompt = f"Based on what you know about what is culturally relevant, or funny, or you think has a chance to be a viral meme, on a scale of 1-10, what score do you give this token name: {token_name}? Please reply with just a numeric score, where 10 is the best meme for the current period, and 1 is the least impactful."
    print(f"Performing vibe check for: {token_name}")
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o", # Updated to latest model
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing numeric scores."}, 
                {"role": "user", "content": prompt}
            ],
            timeout=15 # Added timeout
        )
        
        if response.choices:
            score_text = response.choices[0].message.content.strip()
            # Try to extract just the number
            match = reggie.search(r'\d+', score_text)
            if match:
                score = int(match.group(0))
                print(f"  Vibe score: {score}")
                return score
            else:
                 print(f"  Could not parse numeric score from AI response: '{score_text}'")
                 return None # Could not parse score
        else:
            print("  No response choices received from OpenAI.")
            return None
    except Exception as e:
        print(f"Error during OpenAI vibe check for '{token_name}': {e}")
        return None

def serialize_df_for_prompt(df, max_rows=20):
    """
    Converts the most recent portion of the DataFrame into a string format.
    Selects key columns relevant for trading decisions.
    """
    if df is None or df.empty:
        return "No market data available."
        
    cols_to_include = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'RSI', 'MA40', 
                       'Price_above_MA20', 'Price_above_MA40', 'MA20_above_MA40']
    # Select only columns that actually exist in the df
    available_cols = [col for col in cols_to_include if col in df.columns]
    df_portion = df.tail(max_rows)[available_cols]
    # Reset index to include datetime in the string output easily
    df_str = df_portion.reset_index().to_string(index=False)
    return df_str

def gpt4(prompt, df):
    """
    Sends a prompt and market data summary to GPT-4 for a buy/sell decision.
    Returns True for Buy, False for Sell, None on error or unclear response.
    """
    if not OPENAI_CLIENT:
         print("OpenAI client not initialized. Skipping GPT-4 decision.")
         return None
         
    df_str = serialize_df_for_prompt(df)
    detailed_prompt = f"{prompt}\n\nMarket Data Summary:\n{df_str}\nBased on this recent market data, should I buy (True) or sell (False)? Respond ONLY with the word True or False."

    print("Prompting GPT-4 for trade decision...")
    # print(detailed_prompt) # Optionally print full prompt for debugging
    
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o", # Updated to latest model
            messages=[
                {"role": "system", "content": "You are a Financial Trading Expert analyzing OHLCV data and indicators. Respond ONLY with True (for Buy) or False (for Sell)."},
                {"role": "user", "content": detailed_prompt}
            ],
            timeout=30 # Increased timeout for potentially complex analysis
        )

        if response.choices:
            decision_text = response.choices[0].message.content.strip().lower()
            print(f"  GPT-4 Response: '{decision_text}'")
            if 'true' in decision_text:
                return True # Buy
            elif 'false' in decision_text:
                return False # Sell
            else:
                print("  Warning: Could not parse True/False from GPT-4 response.")
                return None # Unclear decision
        else:
            print("  No response choices received from OpenAI.")
            return None
    except Exception as e:
        print(f"Error during OpenAI GPT-4 decision: {e}")
        return None
        
# --- Entry/Exit Logic Functions (Potentially use in strategies) ---
# These look like specific strategy snippets, refactored slightly for clarity

def elegant_entry(symbol_mint, buy_under_price, target_usd_size, max_chunk_usd, orders_per_tx, slippage_bps, tx_delay_s):
    """Attempts to enter a position if price is below threshold, up to target size."""
    print(f'Checking elegant entry for {symbol_mint[-6:]} below {buy_under_price}...')
    price = token_price(symbol_mint)
    if price is None or price > buy_under_price:
        # print(f"  Price {price} is not below threshold {buy_under_price}. No entry.")
        return False # Condition not met
        
    pos_tokens = get_position(symbol_mint)
    price = token_price(symbol_mint) # Re-fetch price as it might have changed slightly
    if price is None: return False # Need price for USD calculation
    pos_usd = pos_tokens * price
    
    print(f"  Current position: ${pos_usd:.2f}. Target: ${target_usd_size:.2f}. Price: {price:.6f}")
    
    if pos_usd >= target_usd_size * 0.97:
        print("  Position already near target size. No entry needed.")
        return True # Already filled

    # Buy loop (similar to open_position_target_size but includes price check)
    while pos_usd < target_usd_size * 0.97:
        price = token_price(symbol_mint) # Re-check price inside loop
        if price is None or price > buy_under_price:
             print(f"  Price ({price}) moved above threshold ({buy_under_price}) during entry. Stopping buy.")
             break # Stop if price moves unfavorably
             
        chunk_lamports_str = calculate_buy_chunk_lamports(target_usd_size, pos_usd, max_chunk_usd)
        if chunk_lamports_str == "0": break # Target reached
        
        print(f"  Price ({price:.6f}) OK. Buying chunk {chunk_lamports_str} lamports...")
        buy_successful = attempt_market_buy(symbol_mint, chunk_lamports_str, slippage_bps, orders_per_tx, 1)
        
        if buy_successful:
            print(f"  Waiting {tx_delay_s}s after buy attempt...")
            time.sleep(tx_delay_s)
            # Update position status after waiting
            pos_tokens = get_position(symbol_mint)
            price = token_price(symbol_mint) 
            if price is None: 
                 print("Warning: Failed to get price after buy. Stopping entry loop.")
                 break
            pos_usd = pos_tokens * price 
        else:
            print("  Buy attempt failed. Stopping entry.") # Consider retry logic here if needed
            return False # Indicate entry failed
            
    print(f"Elegant entry process finished for {symbol_mint[-6:]}.")
    return True # Reached target or stopped due to price

# Note: elegant_time_entry seems very similar to elegant_entry with a different sleep. 
# Consider merging or clarifying the difference.

def elegant_time_entry(symbol_mint, buy_under_price, target_usd_size, max_chunk_usd, orders_per_tx, slippage_bps, tx_delay_s, order_burst_delay_s=7):
    """Similar to elegant_entry but with a potentially different delay between burst orders."""
    print(f'Checking elegant time entry for {symbol_mint[-6:]} below {buy_under_price}...')
    # Initial sleep before starting?
    # time.sleep(5) 
    price = token_price(symbol_mint)
    if price is None or price > buy_under_price:
        return False 

    pos_tokens = get_position(symbol_mint)
    price = token_price(symbol_mint) # Re-fetch price
    if price is None: return False
    pos_usd = pos_tokens * price
    
    print(f"  Current position: ${pos_usd:.2f}. Target: ${target_usd_size:.2f}. Price: {price:.6f}")

    if pos_usd >= target_usd_size * 0.97:
        print("  Position already near target size.")
        return True

    while pos_usd < target_usd_size * 0.97:
        price = token_price(symbol_mint)
        if price is None or price > buy_under_price:
            print(f"  Price ({price}) moved above threshold ({buy_under_price}). Stopping buy.")
            break

        chunk_lamports_str = calculate_buy_chunk_lamports(target_usd_size, pos_usd, max_chunk_usd)
        if chunk_lamports_str == "0": break

        print(f"  Price ({price:.6f}) OK. Buying chunk {chunk_lamports_str} lamports...")
        buy_successful = attempt_market_buy(symbol_mint, chunk_lamports_str, slippage_bps, orders_per_tx, order_burst_delay_s)

        if buy_successful:
            print(f"  Waiting {tx_delay_s}s after buy attempt...")
            time.sleep(tx_delay_s)
            # Update position status after waiting
            pos_tokens = get_position(symbol_mint)
            price = token_price(symbol_mint)
            if price is None: 
                 print("Warning: Failed to get price after buy. Stopping entry loop.")
                 break
            pos_usd = pos_tokens * price
        else:
            print("  Buy attempt failed. Stopping entry.") 
            return False
            
    print(f"Elegant time entry process finished for {symbol_mint[-6:]}.")
    return True

# --- Liquidation Data Functions (Requires specific CSV file) ---
def _read_liq_data(file_path):
    """Helper to read and preprocess the liquidation CSV data."""
    try:
        df = pd.read_csv(file_path)
        # Define expected columns for validation
        expected_cols = [
            "symbol", "side", "order_type", "time_in_force",
            "original_quantity", "price", "average_price", "order_status",
            "order_last_filled_quantity", "order_filled_accumulated_quantity",
            "order_trade_time", "usd_size"
        ]
        df.columns = expected_cols # Assume fixed order if no header
        df['order_trade_time'] = pd.to_datetime(df['order_trade_time'], unit='ms')
        # Localize to UTC if naive
        if df['order_trade_time'].dt.tz is None:
             df['order_trade_time'] = df['order_trade_time'].dt.tz_localize('UTC')
        else:
             df['order_trade_time'] = df['order_trade_time'].dt.tz_convert('UTC')
        return df
    except FileNotFoundError:
        print(f"Error: Liquidation data file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error reading or processing liquidation file '{file_path}': {e}")
        return None

def calculate_liquidations(symbol_contains, lookback_minutes, liq_data_csv_path):
    """Calculates summed S/L liquidations for a symbol within a lookback period."""
    print(f"Calculating liquidations containing '{symbol_contains}' in last {lookback_minutes} mins from {os.path.basename(liq_data_csv_path)}...")
    df = _read_liq_data(liq_data_csv_path)
    if df is None:
        return False, False, 0, 0, 0 # Return default values on error
        
    # Filter time
    current_time_utc = datetime.now(pytz.utc)
    start_time_utc = current_time_utc - timedelta(minutes=lookback_minutes)
    df_filtered = df[df['order_trade_time'] > start_time_utc].copy()

    # Filter symbol
    sym_df = df_filtered[df_filtered['symbol'].astype(str).str.contains(symbol_contains, case=False, na=False)]
    
    if sym_df.empty:
         # print(f"  No liquidations found for '{symbol_contains}' in the last {lookback_minutes} minutes.")
         return False, False, 0, 0, 0
         
    # Calculate sums
    s_liqs_usd = sym_df[sym_df['side'] == 'BUY']['usd_size'].sum()
    l_liqs_usd = sym_df[sym_df['side'] == 'SELL']['usd_size'].sum()
    total_liqs_usd = s_liqs_usd + l_liqs_usd    

    # Threshold checks are better handled by the calling strategy
    # Returning raw sums is more flexible
    # print(f"  S Liqs: ${s_liqs_usd:.2f}, L Liqs: ${l_liqs_usd:.2f}, Total: ${total_liqs_usd:.2f}")
    return s_liqs_usd, l_liqs_usd, total_liqs_usd # Return sums directly

# --- Binance Funding Rate Functions ---
async def async_get_btc_funding_rate():
    symbol = 'btcusdt'
    websocket_url = f'wss://fstream.binance.com/ws/{symbol}@markPrice'

    async with connect(websocket_url) as websocket:
        try:
            message = await websocket.recv()
            data = json.loads(message)
            funding_rate = float(data['r'])
            yearly_funding_rate = (funding_rate * 3 * 365) * 100
            return yearly_funding_rate
        except Exception as e:
            print(f"An exception occurred while fetching funding rate: {e}")
            return None

# Synchronous wrapper for the async function
def get_btc_funding_rate():
    return asyncio.get_event_loop().run_until_complete(async_get_btc_funding_rate())