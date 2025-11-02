"""
Trading utility functions for Solana blockchain interactions.

This module contains functions for token data retrieval, trade execution,
position management, and market analysis.
"""

# --- Standard Library Imports ---
import json
import math
import os
import re as reggie
import time
import base64
from datetime import datetime, timedelta
import asyncio
import sys

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Third-Party Imports ---
import pandas as pd
import pandas_ta as ta
import pytz
import requests
from termcolor import cprint
from websockets import connect
from openai import OpenAI
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts

# --- Local Imports ---
from config import *
from Day_4_Projects import dontshare as d

# --- Constants ---
MIN_TRADES_LAST_HOUR = MINIMUM_TRADES_IN_LAST_HOUR
API_KEY = d.birdeye
BASE_URL = "https://public-api.birdeye.so/defi"
PRIORITY_FEE = 5000  # Hardcoded prioritization fee (adjust as needed)
USDC_MINT_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOL_MINT_ADDRESS = "So11111111111111111111111111111111111111111"
VIBE_CHECKED_CSV = "vibe_checked_tokens.csv"  # File to save tokens with vibe scores

# --- OpenAI Client Setup ---
client = OpenAI(api_key=d.openai_key)

# --- Core Utility Functions ---
def round_down(value, decimals):
    """
    Round a number down to a specific number of decimal places.
    
    Args:
        value (float): Value to round
        decimals (int): Number of decimal places
        
    Returns:
        float: Rounded down value
    """
    factor = 10 ** decimals
    return math.floor(value * factor) / factor

# --- Module Sections ---
# 1. Data Retrieval Functions
# 2. Position Management Functions 
# 3. Trade Execution Functions
# 4. Analysis & Strategy Functions
# 5. Utility Helper Functions
# 6. AI Integration Functions
# 7. Market Data Functions

# Custom function to print JSON in a human-readable format
def print_pretty_json(data):
    """
    Print JSON data in a human-readable format.
    
    Args:
        data: Data to print (usually dict or list)
    """
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(data)

# Function to print JSON in a human-readable format - assuming you already have it as print_pretty_json
# Helper function to find URLs in text
def find_urls(string):
    """
    Extract URLs from a text string.
    
    Args:
        string (str): Text to search for URLs
        
    Returns:
        list: List of found URLs
    """
    if not string:
        return []
        
    # Regex to extract URLs
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return reggie.findall(pattern, string)

# UPDATED TO RMEOVE THE OTHER ONE so now we can just use this filter instead of filtering twice
def token_overview(address):
    """
    Fetch comprehensive token information including price changes,
    trading activity, and links.
    
    Args:
        address (str): Token mint address
        
    Returns:
        dict: Token overview data or None if fetch failed
    """
    print(f'Getting the token overview for {address[-4:]}')
    overview_url = f"{BASE_URL}/token_overview?address={address}"
    headers = {"X-API-KEY": API_KEY}

    try:
        response = requests.get(overview_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if not data.get('data'):
            print(f"No data returned for token {address[-4:]}")
            return None
            
        overview_data = data['data']
        result = {}
        
        # Trading metrics
        buy1h = overview_data.get('buy1h', 0)
        sell1h = overview_data.get('sell1h', 0)
        trade1h = buy1h + sell1h
        
        # Calculate percentages
        buy_percentage = (buy1h / trade1h * 100) if trade1h else 0
        sell_percentage = (sell1h / trade1h * 100) if trade1h else 0
        
        # Store metrics
        result.update({
            'buy1h': buy1h,
            'sell1h': sell1h,
            'trade1h': trade1h,
            'buy_percentage': buy_percentage,
            'sell_percentage': sell_percentage,
            'minimum_trades_met': trade1h >= MIN_TRADES_LAST_HOUR
        })
        
        # Price changes
        price_changes = {k: v for k, v in overview_data.items() if 'priceChange' in k}
        result['priceChangesXhrs'] = price_changes
        
        # Rug pull check
        rug_pull = any(value < -80 for key, value in price_changes.items() if value is not None)
        result['rug_pull'] = rug_pull
        if rug_pull:
            print("Warning: Price change percentage below -80%, potential rug pull")
        
        # Additional metrics
        result.update({
            'uniqueWallet24h': overview_data.get('uniqueWallet24h', 0),
            'v24USD': overview_data.get('v24hUSD', 0),
            'watch': overview_data.get('watch', 0),
            'view24h': overview_data.get('view24h', 0),
            'liquidity': overview_data.get('liquidity', 0)
        })
        
        # Extract links from description
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
        
        result['description'] = links
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve token overview for {address[-4:]}: {e}")
        return None


def token_security_info(address):

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
    'mutableMetadata': True,
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
    headers = {"X-API-KEY": API_KEY}

    # Sending a GET request to the API
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse the JSON response
        security_data = response.json()['data']
        print_pretty_json(security_data)
    else:
        print("Failed to retrieve token security info:", response.status_code)

def token_creation_info(address):

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
    headers = {"X-API-KEY": API_KEY}

    # Sending a GET request to the API
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse the JSON response
        creation_data = response.json()['data']
        print_pretty_json(creation_data)
    else:
        print("Failed to retrieve token creation info:", response.status_code)

def market_buy(token, amount, slippage):
    """
    Execute a market buy order for a token using Jupiter Aggregator.

    Args:
        token (str): Mint address of the token to buy
        amount (str): Amount to buy in lamports (USDC has 6 decimals)
        slippage (int): Slippage tolerance in basis points (e.g., 50 = 0.5%)

    Returns:
        str: Transaction URL if successful

    Raises:
        Exception: If the API request fails or transaction fails
    """
    KEY = Keypair.from_base58_string(d.sol_key)
    QUOTE_TOKEN = USDC_MINT_ADDRESS  # USDC mint address
    
    # Use Ankr's RPC endpoint
    http_client = Client(d.ankr_key)
    
    quote_url = f'https://quote-api.jup.ag/v6/quote?inputMint={QUOTE_TOKEN}&outputMint={token}&amount={amount}&slippageBps={slippage}'
    swap_url = 'https://quote-api.jup.ag/v6/swap'
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get quote
            response = requests.get(quote_url)
            response.raise_for_status()  # Raise exception for HTTP errors
            quote = response.json()
            
            # Execute swap
            tx_response = requests.post(
                swap_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "quoteResponse": quote,
                    "userPublicKey": str(KEY.pubkey()),
                    "prioritizationFeeLamports": PRIORITY_FEE
                })
            )
            tx_response.raise_for_status()
            tx_data = tx_response.json()
            
            # Process transaction
            swap_tx = base64.b64decode(tx_data['swapTransaction'])
            tx1 = VersionedTransaction.from_bytes(swap_tx)
            tx = VersionedTransaction(tx1.message, [KEY])
            
            # Send transaction
            tx_id = http_client.send_raw_transaction(
                bytes(tx), 
                TxOpts(skip_preflight=True)
            ).value
            
            tx_url = f"https://solscan.io/tx/{str(tx_id)}"
            print(tx_url)
            return tx_url
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            error_msg = f"Request failed (attempt {retry_count}/{max_retries}): {e}"
            if retry_count < max_retries:
                print(f"{error_msg} Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise Exception(f"Maximum retries exceeded: {error_msg}")
                
        except Exception as e:
            retry_count += 1
            error_msg = f"Transaction error (attempt {retry_count}/{max_retries}): {e}"
            if retry_count < max_retries:
                print(f"{error_msg} Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise Exception(f"Maximum retries exceeded: {error_msg}")


def market_sell(token_mint_address, amount, slippage):
    """
    Execute a market sell order for a token using Jupiter Aggregator.
    
    Args:
        token_mint_address (str): Mint address of the token to sell
        amount (str): Amount to sell in lamports (token-specific decimals)
        slippage (int): Slippage tolerance in basis points (e.g., 50 = 0.5%)
        
    Returns:
        str: Transaction URL if successful
        
    Raises:
        Exception: If the API request fails or transaction fails
    """
    KEY = Keypair.from_base58_string(d.sol_key)
    output_token = USDC_MINT_ADDRESS  # Selling to get USDC
    
    # Use Ankr's RPC endpoint
    http_client = Client(d.ankr_key)
    
    quote_url = f'https://quote-api.jup.ag/v6/quote?inputMint={token_mint_address}&outputMint={output_token}&amount={amount}&slippageBps={slippage}'
    swap_url = 'https://quote-api.jup.ag/v6/swap'
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get quote
            response = requests.get(quote_url)
            response.raise_for_status()
            quote = response.json()
            
            # Execute swap
            tx_response = requests.post(
                swap_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "quoteResponse": quote,
                    "userPublicKey": str(KEY.pubkey()),
                    "prioritizationFeeLamports": PRIORITY_FEE
                })
            )
            tx_response.raise_for_status()
            tx_data = tx_response.json()
            
            # Process transaction
            swap_tx = base64.b64decode(tx_data['swapTransaction'])
            tx1 = VersionedTransaction.from_bytes(swap_tx)
            tx = VersionedTransaction(tx1.message, [KEY])
            
            # Send transaction
            tx_id = http_client.send_raw_transaction(
                bytes(tx), 
                TxOpts(skip_preflight=True)
            ).value
            
            tx_url = f"https://solscan.io/tx/{str(tx_id)}"
            print(tx_url)
            return tx_url
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            error_msg = f"Request failed (attempt {retry_count}/{max_retries}): {e}"
            if retry_count < max_retries:
                cprint(f"{error_msg} Retrying in 5 seconds...", 'yellow')
                time.sleep(5)
            else:
                raise Exception(f"Maximum retries exceeded: {error_msg}")
                
        except Exception as e:
            retry_count += 1
            error_msg = f"Transaction error (attempt {retry_count}/{max_retries}): {e}"
            if retry_count < max_retries:
                cprint(f"{error_msg} Retrying in 5 seconds...", 'yellow')
                time.sleep(5)
            else:
                raise Exception(f"Maximum retries exceeded: {error_msg}")


def elegant_entry(symbol, buy_under_price):
    """
    Enter a position gradually in small chunks until the target USD size is reached.
    
    Args:
        symbol (str): Token mint address to buy
        buy_under_price (float): Maximum price to buy at
        
    Returns:
        float: Final position value in USD
    """
    print(f'Executing elegant entry for {symbol[-4:]}...')
    
    # Get current position and price
    pos = get_position(symbol)
    price = token_price(symbol)
    pos_usd = pos * price
    
    # Calculate needed amount
    size_needed = USD_SIZE - pos_usd
    
    # Check if position already filled
    if pos_usd >= (0.97 * USD_SIZE):
        cprint(f'Position already filled (${round(pos_usd, 2)}/${USD_SIZE})', 'green')
        return pos_usd
        
    # Initialize chunk size
    chunk_size = min(size_needed, MAX_USD_ORDER_SIZE)
    chunk_lamports = int(chunk_size * 10**6)  # Convert to lamports (USDC has 6 decimals)
    chunk_size_str = str(chunk_lamports)
    
    print(f'Initial state - Position: {round(pos, 2)}, '
          f'Price: {round(price, 8)}, Value: ${round(pos_usd, 2)}, '
          f'Target: ${USD_SIZE}, Buy under: {buy_under_price}')
    
    # Loop until position is filled or price exceeds buy_under
    while pos_usd < (0.97 * USD_SIZE) and price <= buy_under_price:
        print(f'Execution - Position: {round(pos, 2)}, Price: {round(price, 8)}, '
              f'Value: ${round(pos_usd, 2)}, Chunk: {chunk_size_str} lamports')
        
        try:
            # Execute multiple smaller orders
            for _ in range(ORDERS_PER_OPEN):
                market_buy(symbol, chunk_size_str, SLIPPAGE)
                cprint(f'Chunk buy submitted for {symbol[-4:]} '
                       f'size: {chunk_size_str} lamports', 'white', 'on_blue')
                time.sleep(0.1)
            
            # Wait for transactions to process
            time.sleep(TX_SLEEP)
            
        except Exception as e:
            cprint(f'Buy error: {e}. Retrying in 30 seconds...', 'light_blue', 'on_light_magenta')
            time.sleep(30)
            
            try:
                # Retry the buys
                for _ in range(ORDERS_PER_OPEN):
                    market_buy(symbol, chunk_size_str, SLIPPAGE)
                    cprint(f'Retry chunk buy submitted for {symbol[-4:]} '
                           f'size: {chunk_size_str} lamports', 'white', 'on_blue')
                    time.sleep(0.1)
                
                # Wait for transactions to process
                time.sleep(TX_SLEEP)
                
            except Exception as retry_error:
                cprint(f'Final error in the buy: {retry_error}. Exiting.', 'white', 'on_red')
                time.sleep(10)
                break
        
        # Update position status
        pos = get_position(symbol)
        price = token_price(symbol)
        pos_usd = pos * price
        
        # Recalculate needed amount
        size_needed = USD_SIZE - pos_usd
        chunk_size = min(size_needed, MAX_USD_ORDER_SIZE)
        chunk_lamports = int(chunk_size * 10**6)
        chunk_size_str = str(chunk_lamports)
    
    # Report final status
    final_status = "filled" if pos_usd >= (0.97 * USD_SIZE) else "partially filled"
    final_reason = f"position {final_status}" if pos_usd >= (0.97 * USD_SIZE) else f"price {price} > {buy_under_price}"
    cprint(f'Elegant entry complete - {final_reason}. Final position: ${round(pos_usd, 2)}', 'cyan')
    
    return pos_usd


def chunk_kill(token_mint_address, max_usd_sell_size, slippage):
    """
    Close a position by selling in chunks until fully closed.
    
    Args:
        token_mint_address (str): Token mint address to sell
        max_usd_sell_size (float): Maximum USD value to sell in each chunk
        slippage (int): Slippage tolerance in basis points
        
    Returns:
        bool: True if position was completely closed
    """
    # Get current position details
    balance = get_position(token_mint_address)
    if balance <= 0:
        print(f'No position to close for {token_mint_address[-4:]}')
        return True
        
    cprint(f'Closing position of {balance} tokens for {token_mint_address[-4:]}', 'white', 'on_magenta')
    
    # Get token price and calculate value
    price = token_price(token_mint_address)
    price = float(price)
    usd_value = balance * price
    
    # Get token decimals
    decimals = get_decimals(token_mint_address)
    
    # Track success
    completely_closed = False
    
    # Continue until position is closed
    while usd_value > 0:
        # Calculate sell size for this chunk
        if usd_value < max_usd_sell_size:
            sell_size = balance 
        else:
            sell_size = max_usd_sell_size / price
        
        # Round down to prevent "insufficient funds" errors
        sell_size = round_down(sell_size, 2)
        sell_size_lamports = int(sell_size * 10**decimals)
        
        cprint(f'Selling {sell_size} tokens (~${round(sell_size * price, 2)}) '
               f'of {token_mint_address[-4:]}', 'white', 'on_blue')
        
        # Execute multiple sells to ensure order goes through
        try:
            for _ in range(3):  # Try to sell 3 times to ensure it goes through
                market_sell(token_mint_address, sell_size_lamports, slippage)
                time.sleep(1)
            
            time.sleep(15)  # Wait for transactions to process
            
        except Exception as e:
            cprint(f'Sell error: {e}. Retrying in 5 seconds...', 'red')
            time.sleep(5)
        
        # Update position status
        balance = get_position(token_mint_address)
        price = token_price(token_mint_address)
        usd_value = balance * price
        
        # Report current status
        if usd_value > 0:
            cprint(f'Remaining: {round(balance, 4)} tokens, ${round(usd_value, 2)}', 'yellow')
        else:
            cprint(f'Position for {token_mint_address[-4:]} fully closed', 'white', 'on_green')
            completely_closed = True
    
    return completely_closed

# --- 6. AI Integration Functions ---

def vibe_check(name):
    """
    Use AI to evaluate the meme/viral potential of a token name.
    
    Args:
        name (str): Token name to evaluate
        
    Returns:
        int: Score from 1-10 or None if error
    """
    if not name or not isinstance(name, str):
        return None
        
    prompt = (
        f"Based on what you know about what is culturally relevant, or funny, "
        f"or you think has a chance to be a viral meme, on a scale of 1-10, "
        f"what score do you give this token name: {name}? "
        f"Please reply with just a numeric score, where 10 is the best meme "
        f"for the current period, and 1 is the least impactful."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract score from response
        if response.choices:
            score_text = response.choices[0].message.content.strip()
            
            # Try to convert to integer
            try:
                score = int(score_text.split()[0] if ' ' in score_text else score_text)
                print(f"Vibe score for '{name}': {score}/10")
                return score
            except ValueError:
                print(f"Invalid score format received for '{name}': '{score_text}'")
                return None
        else:
            print(f"No response received for '{name}'")
            return None
            
    except Exception as e:
        print(f"Error during vibe check for '{name}': {e}")
        return None


def serialize_df_for_prompt(df, max_rows=20):
    """
    Converts DataFrame into a string format for AI prompt.
    
    Args:
        df (pd.DataFrame): DataFrame to serialize
        max_rows (int): Maximum rows to include
        
    Returns:
        str: Formatted string representation
    """
    if df is None or df.empty:
        return "No data available."
        
    # Focus on the most recent rows and relevant columns
    key_columns = [
        'Datetime (UTC)', 'Open', 'High', 'Low', 'Close', 'Volume',
        'MA20', 'RSI', 'MA40', 'Price_above_MA20', 'Price_above_MA40', 'MA20_above_MA40'
    ]
    
    # Use only columns that exist in the DataFrame
    available_columns = [col for col in key_columns if col in df.columns]
    
    # Select data and format
    df_portion = df.tail(max_rows)[available_columns]
    return df_portion.to_string(index=False)


def gpt4(prompt, df):
    """
    Use GPT-4 to analyze market data and make a buy/sell decision.
    
    Args:
        prompt (str): Base prompt describing what to evaluate
        df (pd.DataFrame): Market data for analysis
        
    Returns:
        str: "True" for buy, "False" for sell, or None if error
    """
    # Serialize DataFrame for inclusion in prompt
    df_str = serialize_df_for_prompt(df)
    
    # Build detailed prompt
    detailed_prompt = (
        f"{prompt}\n\n"
        f"Market Data Summary:\n{df_str}\n\n"
        f"Based on this recent market data, should I buy (True) or sell (False)? "
        f"ONLY SEND BACK True or False"
    )
    
    print("Analyzing market data via AI...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a Financial Trading Expert that works with OHLCV data to make "
                               "a True or False decision on whether to buy or sell a token. "
                               "True is buy, False is sell."
                },
                {"role": "user", "content": detailed_prompt}
            ]
        )

        # Extract decision
        if response.choices:
            decision = response.choices[0].message.content.strip()
            print(f"AI recommendation: {decision}")
            return decision
        else:
            print("No response from AI")
            return None
            
    except Exception as e:
        print(f"Error getting AI trading decision: {e}")
        return None


def ai_vibe_check(df, min_vibe_score):
    """
    Filter a DataFrame based on token name 'vibe scores'.
    
    Args:
        df (pd.DataFrame): DataFrame containing token data with 'name' column
        min_vibe_score (int): Minimum score to keep (1-10)
        
    Returns:
        pd.DataFrame: Filtered DataFrame with added 'Vibe Score' column
    """
    if df is None or df.empty or 'name' not in df.columns:
        print("DataFrame is empty or missing 'name' column")
        return df
        
    print(f"Running AI vibe check on {len(df)} tokens...")
    
    # Apply vibe_check to each name
    df['Vibe Score'] = df['name'].apply(vibe_check)
    
    # Save results
    if 'VIBE_CHECKED_CSV' in globals():
        df.to_csv(VIBE_CHECKED_CSV, index=False)
        print(f"Saved vibe scores to {VIBE_CHECKED_CSV}")
    
    # Filter based on minimum score
    filtered_df = df[df['Vibe Score'] >= min_vibe_score]
    print(f"Filtered from {len(df)} to {len(filtered_df)} tokens with vibe score >= {min_vibe_score}")
    
    return filtered_df



def elegant_time_entry(symbol, buy_under, seconds_to_sleep):

    print('inside elegant entry...')

    time.sleep(5)
    pos = get_position(symbol)
    price = token_price(symbol)
    pos_usd = pos * price
    size_needed = USD_SIZE - pos_usd
    if size_needed > MAX_USD_ORDER_SIZE: chunk_size = MAX_USD_ORDER_SIZE 
    else: chunk_size = size_needed

    print(f'position: {round(pos,2)} price: {round(price,8)} pos_usd: ${round(pos_usd,2)}')
    
    chunk_size = int(chunk_size * 10**6)
    chunk_size = str(chunk_size)

    if pos_usd > (.97 * USD_SIZE):
        print('position filled')
        time.sleep(10)

    print(f'pos_usd: {pos_usd} USD_SIZE: {USD_SIZE} buy_under: {buy_under}')
    while pos_usd < (.97 * USD_SIZE) and (price <= buy_under):

        print(f'position: {round(pos,2)} price: {round(price,8)} pos_usd: ${round(pos_usd,2)}')

        try:

            for i in range(ORDERS_PER_OPEN):
                market_buy(symbol, chunk_size, SLIPPAGE)
                # cprint green background black text
                cprint(f'chunk buy submitted of {symbol[-4:]} sz: {chunk_size} you my dawg moon dev', 'white', 'on_blue')
                time.sleep(7)
        
            time.sleep(seconds_to_sleep)

            pos = get_position(symbol)
            price = token_price(symbol)
            pos_usd = pos * price
            size_needed = USD_SIZE - pos_usd
            if size_needed > MAX_USD_ORDER_SIZE: chunk_size = MAX_USD_ORDER_SIZE 
            else: chunk_size = size_needed
            chunk_size = int(chunk_size * 10**6)
            chunk_size = str(chunk_size)
            
        except:

            try:
                cprint(f'trying again to make the order in 30 seconds.....', 'light_blue', 'on_light_magenta')
                time.sleep(30)
                for i in range(ORDERS_PER_OPEN):
                    market_buy(symbol, chunk_size, SLIPPAGE)
                    # cprint green background black text
                    cprint(f'chunk buy submitted of {symbol[-4:]} sz: {chunk_size} you my dawg moon dev', 'white', 'on_blue')
                    time.sleep(7)
                
                time.sleep(TX_SLEEP)
                pos = get_position(symbol)
                price = token_price(symbol)
                pos_usd = pos * price
                size_needed = USD_SIZE - pos_usd
                if size_needed > MAX_USD_ORDER_SIZE: chunk_size = MAX_USD_ORDER_SIZE 
                else: chunk_size = size_needed
                chunk_size = int(chunk_size * 10**6)
                chunk_size = str(chunk_size)


            except:
                cprint(f'Final Error in the buy, restart needed', 'white', 'on_red')
                time.sleep(10)
                break

        pos = get_position(symbol)
        price = token_price(symbol)
        pos_usd = pos * price
        size_needed = USD_SIZE - pos_usd
        if size_needed > MAX_USD_ORDER_SIZE: chunk_size = MAX_USD_ORDER_SIZE 
        else: chunk_size = size_needed
        chunk_size = int(chunk_size * 10**6)
        chunk_size = str(chunk_size)

def calculate_btc_liquidations():
    # Define the time period we want to look at (the last 30 minutes)
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    thirty_minutes_ago = current_time - timedelta(minutes=30)
    
    # Read the CSV file
    df = pd.read_csv('/Users/md/Dropbox/dev/github/hyper-liquid-trading-bots/backtests/liquidations/data/liq_data.csv')
    
    # Convert the timestamp from unix milliseconds to datetime, and filter for the last 30 minutes
    df['order_trade_time'] = pd.to_datetime(df['order_trade_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    df = df[df['order_trade_time'] > thirty_minutes_ago]

    #print(df)
    
    # Filter for symbol names that contain 'BTC'
    btc_df = df[df['symbol'].str.contains('BTC')]
    
    # Split by BUY and SELL
    s_liqs = btc_df[btc_df['side'] == 'BUY']['usd_size'].sum()
    l_liqs = btc_df[btc_df['side'] == 'SELL']['usd_size'].sum()
    
    # Return the sum of USD size for both BUY and SELL liquidations
    return s_liqs, l_liqs

def calculate_liquidations(sym, l_liq_amount, s_liq_amount, time_of_liqs):
    # Define the time period we want to look at (the last x minutes)
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    x_minutes_ago = current_time - timedelta(minutes=time_of_liqs)
    
    ## Update on server to ensure it's the fresh liquidation data
    df = pd.read_csv('/Users/md/Dropbox/dev/github/hyper-liquid-trading-bots/backtests/liquidations/data/liq_data.csv')
    
    # Add these columns
    df.columns = [
        "symbol", "side", "order_type", "time_in_force",
        "original_quantity", "price", "average_price", "order_status",
        "order_last_filled_quantity", "order_filled_accumulated_quantity",
        "order_trade_time", "usd_size"
    ]
    
    # Convert the timestamp from unix milliseconds to datetime, and filter for the last x minutes
    df['order_trade_time'] = pd.to_datetime(df['order_trade_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    df = df[df['order_trade_time'] > x_minutes_ago]

    # Filter for symbol names that contain 'BTC'
    sym_df = df[df['symbol'].str.contains(sym)]
    
    # Split by BUY and SELL
    s_liqs = sym_df[sym_df['side'] == 'BUY']['usd_size'].sum()
    l_liqs = sym_df[sym_df['side'] == 'SELL']['usd_size'].sum()
    total_liqs = s_liqs + l_liqs    

    # Check if short or long liquidation thresholds are hit
    short_liq_thres_hit = s_liqs > s_liq_amount
    long_liq_thres_hit = l_liqs > l_liq_amount

    # Return the results
    return short_liq_thres_hit, long_liq_thres_hit, s_liqs, l_liqs, total_liqs




import asyncio
import json
from websockets import connect

# New async function to get BTC funding rate
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

# --- 5. Utility Helper Functions ---

def calculate_chunk_size(current_position, current_price, target_usd_size, max_chunk_size):
    """
    Calculate appropriate chunk size for an order.
    
    Args:
        current_position (float): Current token amount owned
        current_price (float): Current token price
        target_usd_size (float): Target position size in USD
        max_chunk_size (float): Maximum chunk size in USD
        
    Returns:
        tuple: (chunk_size_lamports_str, size_needed)
    """
    # Calculate current position value and remaining needed
    position_value = current_position * current_price
    size_needed = target_usd_size - position_value
    
    # Determine chunk size, respecting maximum
    chunk_size = min(size_needed, max_chunk_size) if size_needed > 0 else 0
    
    # Convert to lamports (assuming USDC with 6 decimals)
    chunk_lamports = int(chunk_size * 10**6)
    chunk_size_str = str(chunk_lamports)
    
    return chunk_size_str, size_needed

# --- 2. Position Management Functions ---

def pnl_close(token_mint_address):
    """
    Check if a position should be closed based on profit/loss thresholds.
    
    Args:
        token_mint_address (str): Token mint address to evaluate
        
    Returns:
        bool: True if position was closed, False otherwise
    """
    print(f'Checking PnL exit conditions for {token_mint_address[-4:]}...')
    
    # Get current position details
    balance = get_position(token_mint_address)
    if balance <= 0:
        print(f'No position to evaluate for {token_mint_address[-4:]}')
        return False
        
    # Get price and calculate value
    price = token_price(token_mint_address)
    if price is None:
        print(f'Could not get price for {token_mint_address[-4:]}')
        return False
        
    usd_value = balance * price
    
    # Calculate take profit and stop loss thresholds
    tp = SELL_AT_MULTIPLE * USD_SIZE
    sl = ((1 + STOP_LOSS_PERCENTAGE) * USD_SIZE)
    
    # Get token decimals for order sizing
    decimals = get_decimals(token_mint_address)
    
    # Calculate sell size
    sell_size = balance
    sell_size_lamports = int(sell_size * 10**decimals)
    
    print(f'Position: {round(balance, 4)}, Price: ${round(price, 8)}, Value: ${round(usd_value, 2)}')
    print(f'Take Profit: ${round(tp, 2)}, Stop Loss: ${round(sl, 2)}')
    
    # --- Take Profit Logic ---
    if usd_value > tp:
        cprint(f'Take Profit triggered for {token_mint_address[-4:]} (${round(usd_value, 2)} > ${round(tp, 2)})', 'white', 'on_green')
        
        # Execute sell orders in batches to ensure execution
        sell_attempts = 0
        max_attempts = 3
        
        while usd_value > tp and sell_attempts < max_attempts:
            try:
                market_sell(token_mint_address, sell_size_lamports, SLIPPAGE)
                cprint(f'Profit taking: Sell order {sell_attempts+1}/{max_attempts} sent', 'white', 'on_green')
                time.sleep(1)
                sell_attempts += 1
                
            except Exception as e:
                cprint(f'Sell error: {e}. Retrying...', 'white', 'on_red')
                time.sleep(2)
            
        # Update position details after sells
        balance = get_position(token_mint_address)
        price = token_price(token_mint_address)
        usd_value = balance * price
        
        print(f'After take profit: Position = {round(balance, 4)}, Value = ${round(usd_value, 2)}')
        return usd_value <= 1  # Consider closed if less than $1 remains
    
    # --- Stop Loss Logic ---
    elif usd_value < sl and usd_value > 0.05:  # Only trigger if position value is meaningful
        cprint(f'Stop Loss triggered for {token_mint_address[-4:]} (${round(usd_value, 2)} < ${round(sl, 2)})', 'white', 'on_blue')
        
        # Execute sell orders in batches to ensure execution
        sell_attempts = 0
        max_attempts = 3
        
        while usd_value > 0.05 and sell_attempts < max_attempts:
            try:
                market_sell(token_mint_address, sell_size_lamports, SLIPPAGE)
                cprint(f'Stop loss: Sell order {sell_attempts+1}/{max_attempts} sent', 'white', 'on_blue')
                time.sleep(1)
                sell_attempts += 1
                
            except Exception as e:
                cprint(f'Sell error: {e}. Retrying...', 'white', 'on_red')
                time.sleep(2)
        
        # Update position details after sells
        balance = get_position(token_mint_address)
        price = token_price(token_mint_address)
        usd_value = balance * price
        
        # Add to don't overtrade list if successfully closed
        if usd_value <= 0.05:
            print(f'Successfully closed {token_mint_address[-4:]} via stop loss')
            try:
                with open('dont_overtrade.txt', 'a') as file:
                    file.write(token_mint_address + '\n')
            except Exception as e:
                print(f"Error writing to dont_overtrade.txt: {e}")
            
            return True
    
    # No action taken
    return False


def chunk_kill_mm(token_mint_address, max_usd_sell_size, slippage, sell_over_p, seconds_to_sleep):
    """
    Close a position gradually for market making when price is above threshold.
    
    Args:
        token_mint_address (str): Token mint address to sell
        max_usd_sell_size (float): Maximum USD value to sell in each chunk
        slippage (int): Slippage tolerance in basis points
        sell_over_p (float): Price threshold above which to sell
        seconds_to_sleep (int): Seconds to wait between chunks
        
    Returns:
        bool: True if completely closed, False otherwise
    """
    # Get current position details
    balance = get_position(token_mint_address)
    if balance <= 0:
        print(f'No position to close for {token_mint_address[-4:]}')
        return True
        
    cprint(f'Market making: Checking {token_mint_address[-4:]} position: {balance}', 'white', 'on_black')
    
    # Get token price and calculate value
    price = token_price(token_mint_address)
    if price is None:
        print(f'Could not get price for {token_mint_address[-4:]}')
        return False
        
    price = float(price)
    usd_value = balance * price
    
    # Check if price is below threshold
    if price <= sell_over_p:
        print(f'Price ${price} is below sell threshold ${sell_over_p}. Not selling.')
        return False
    
    # Calculate sell size for this chunk
    if usd_value < max_usd_sell_size:
        sell_size = balance
    else:
        sell_size = max_usd_sell_size / price
    
    # Round to prevent "insufficient funds" errors
    sell_size = round_down(sell_size, 2)
    
    # Get token decimals
    decimals = get_decimals(token_mint_address)
    sell_size_lamports = int(sell_size * 10**decimals)
    
    cprint(f'Market making: Selling {sell_size} tokens of {token_mint_address[-4:]} (${round(sell_size * price, 2)}) at ${price}', 'white', 'on_black')
    
    # Loop until position is closed or price drops below threshold
    while (usd_value > 0) and (price > sell_over_p):
        try:
            # Execute multiple sell orders to ensure it goes through
            for _ in range(3):
                market_sell(token_mint_address, sell_size_lamports, slippage)
                cprint(f'Market making: Sell order sent for {token_mint_address[-4:]}', 'white', 'on_blue')
                time.sleep(1)
            
            time.sleep(seconds_to_sleep)  # Wait between chunks
            
        except Exception as e:
            cprint(f'Sell error: {e}. Retrying...', 'white', 'on_red')
            time.sleep(5)
        
        # Update position details
        balance = get_position(token_mint_address)
        price = token_price(token_mint_address)
        price = float(price)
        usd_value = balance * price
        
        # Recalculate sell size for next chunk
        if usd_value < max_usd_sell_size:
            sell_size = balance
        else:
            sell_size = max_usd_sell_size / price
            
        sell_size = round_down(sell_size, 2)
        sell_size_lamports = int(sell_size * 10**decimals)
    
    # Final status
    if usd_value <= 0:
        cprint(f'Position fully closed for {token_mint_address[-4:]}', 'white', 'on_green')
        return True
    else:
        print(f'Price dropped below threshold. Remaining: {round(balance, 4)} tokens (${round(usd_value, 2)})')
        return False


def close_all_positions():
    """
    Close all open positions except those in the do-not-trade list.
    
    Returns:
        int: Number of positions successfully closed
    """
    print('Closing all open positions...')
    
    # Get all positions
    open_positions = fetch_wallet_holdings_og(WALLET_ADDRESS)
    if open_positions.empty:
        print('No open positions found')
        return 0
    
    # Filter out positions in DO_NOT_TRADE_LIST
    positions_to_close = open_positions[~open_positions['Mint Address'].isin(DO_NOT_TRADE_LIST)]
    
    if positions_to_close.empty:
        print('No positions to close (all in DO_NOT_TRADE_LIST)')
        return 0
    
    print(f'Found {len(positions_to_close)} positions to close')
    closed_count = 0
    
    # Loop through and close each position
    for index, row in positions_to_close.iterrows():
        token_mint_address = row['Mint Address']
        token_value = row['USD Value']
        
        if token_value < 1:
            print(f'Skipping {token_mint_address[-4:]} - value too small (${round(token_value, 2)})')
            continue
            
        print(f'Closing position for {token_mint_address[-4:]} (${round(token_value, 2)})...')
        
        try:
            # Use chunk_kill to close the entire position
            if chunk_kill(token_mint_address, MAX_USD_ORDER_SIZE, SLIPPAGE):
                closed_count += 1
        except Exception as e:
            print(f'Error closing position for {token_mint_address[-4:]}: {e}')
    
    print(f'Successfully closed {closed_count}/{len(positions_to_close)} positions')
    return closed_count

# --- 1. Data Retrieval Functions ---

def token_price(address):
    """
    Get the current price of a token.
    
    Args:
        address (str): Token mint address
        
    Returns:
        float: Token price in USD or None if not found
    """
    url = f"https://public-api.birdeye.so/defi/price?address={address}"
    headers = {"X-API-KEY": API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        price_data = response.json()
        if price_data.get('success'):
            return price_data['data']['value']
        else:
            print(f"API returned unsuccessful response for {address[-4:]}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching token price for {address[-4:]}: {e}")
        return None


def get_decimals(token_mint_address):
    """
    Get the decimal places for a token.
    
    Args:
        token_mint_address (str): Token mint address
        
    Returns:
        int: Number of decimal places the token uses
    """
    url = "https://api.mainnet-beta.solana.com/"
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
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        
        response_json = response.json()
        if 'result' not in response_json or not response_json['result']['value']:
            print(f"No data returned for {token_mint_address[-4:]}")
            return 6  # Default to 6 decimals (USDC) if data not available
            
        # Parse the response to extract the number of decimals
        decimals = response_json['result']['value']['data']['parsed']['info']['decimals']
        return decimals
        
    except Exception as e:
        print(f"Error fetching decimals for {token_mint_address[-4:]}: {e}")
        return 6  # Default to 6 decimals (USDC) if error occurs


def fetch_wallet_holdings_og(address):
    """
    Fetch all token holdings for a wallet address.
    
    Args:
        address (str): Wallet address to query
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Mint Address', 'Amount', 'USD Value']
                     or empty DataFrame if no holdings found
    """
    url = f"https://public-api.birdeye.so/v1/wallet/token_list?wallet={address}"
    headers = {"x-chain": "solana", "X-API-KEY": API_KEY}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        json_response = response.json()
        
        # Initialize empty DataFrame
        df = pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])
        
        if 'data' in json_response and 'items' in json_response['data']:
            df = pd.DataFrame(json_response['data']['items'])
            
            # Extract and rename relevant columns
            if not df.empty:
                df = df[['address', 'uiAmount', 'valueUsd']]
                df = df.rename(columns={
                    'address': 'Mint Address', 
                    'uiAmount': 'Amount', 
                    'valueUsd': 'USD Value'
                })
                
                # Clean up data
                df = df.dropna()
                df = df[df['USD Value'] > 0.05]  # Filter out dust amounts
        
        # Print summary if we have results
        if not df.empty:
            print(df)
            cprint(f'** Total USD balance is ${df["USD Value"].sum()}', 'white', 'on_green')
        else:
            cprint("No significant wallet holdings found.", 'yellow')
            
        return df
        
    except requests.exceptions.RequestException as e:
        cprint(f"Failed to retrieve token list for {address}: {e}", 'white', 'on_red')
        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])


def fetch_wallet_token_single(address, token_mint_address):
    """
    Fetch single token holding info for a wallet.
    
    Args:
        address (str): Wallet address
        token_mint_address (str): Token mint address to query
        
    Returns:
        pd.DataFrame: DataFrame with single token data or empty DataFrame if not found
    """
    # Get all holdings
    df = fetch_wallet_holdings_og(address)
    
    # Filter for target token
    if not df.empty:
        return df[df['Mint Address'] == token_mint_address]
    
    return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])


def get_position(token_mint_address):
    """
    Get the token balance for a specific token in the wallet.
    
    Args:
        token_mint_address (str): Token mint address to query
        
    Returns:
        float: Token balance or 0 if not found
    """
    try:
        dataframe = fetch_wallet_token_single(WALLET_ADDRESS, token_mint_address)
        
        # Check if the DataFrame is empty
        if dataframe.empty:
            return 0
        
        # Extract the balance for the specified token
        balance = dataframe.iloc[0]['Amount']
        return balance
        
    except Exception as e:
        print(f"Error retrieving position for {token_mint_address[-4:]}: {e}")
        return 0