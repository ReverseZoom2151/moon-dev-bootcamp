"""
Bitfinex Trading Bot Infrastructure

This script provides a basic framework for executing different trading actions
based on user input or eventually automated signals.

Available modes:
0 - Close position (in chunks)
1 - Open a buying position (in chunks)
2 - ETH SMA based strategy
4 - Close positions based on PNL thresholds
5 - Simple market making (buy under/sell over)
"""

import time
import requests
import hmac
import hashlib
import base64
import json
from termcolor import cprint

# Import specific config variables (adapted for Bitfinex)
from bitfinex_config import (
    PRIMARY_SYMBOL, USD_SIZE, MAX_USD_ORDER_SIZE, SLIPPAGE_PERCENT,
    ORDERS_PER_OPEN, TX_SLEEP, TOKEN_BATCH, API_KEY, API_SECRET,
    DO_NOT_TRADE_LIST, LOWEST_BALANCE, TARGET_BALANCE,
    BUY_UNDER_PERCENT, SELL_OVER_PERCENT
)

# --- Mode Constants ---
CLOSE_MODE = 0
BUY_MODE = 1
ETH_TRADE_MODE = 2
PNL_CLOSE_MODE = 4
MARKET_MAKER_MODE = 5

# --- Bitfinex API Helper Class ---
class BitfinexAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.bitfinex.com"
    
    def _nonce(self):
        return str(int(time.time() * 1000000))
    
    def _sign_payload(self, payload):
        j = json.dumps(payload)
        data = base64.standard_b64encode(j.encode('utf8'))
        h = hmac.new(self.api_secret.encode('utf8'), data, hashlib.sha384)
        signature = h.hexdigest()
        return {
            "X-BFX-APIKEY": self.api_key,
            "X-BFX-SIGNATURE": signature,
            "X-BFX-PAYLOAD": data
        }
    
    def get_ticker(self, symbol):
        """Get current ticker price for symbol"""
        url = f"{self.base_url}/v1/pubticker/{symbol}"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            return float(data['last_price'])
        except Exception as e:
            cprint(f"Error fetching ticker for {symbol}: {e}", 'red')
            return None
    
    def get_balance(self, currency):
        """Get balance for specific currency"""
        url = f"{self.base_url}/v1/balances"
        nonce = self._nonce()
        payload = {
            "request": "/v1/balances",
            "nonce": nonce
        }
        headers = self._sign_payload(payload)
        
        try:
            response = requests.post(url, headers=headers, timeout=10)
            balances = response.json()
            
            for balance in balances:
                if balance['currency'].upper() == currency.upper() and balance['type'] == 'exchange':
                    return float(balance['available'])
            return 0.0
        except Exception as e:
            cprint(f"Error fetching balance for {currency}: {e}", 'red')
            return 0.0
    
    def get_position(self, symbol):
        """Get current position size for symbol"""
        balance = self.get_balance(symbol.replace('usd', '').replace('USD', ''))
        return balance if balance else 0.0
    
    def market_buy(self, symbol, amount_usd):
        """Execute market buy order"""
        url = f"{self.base_url}/v1/order/new"
        nonce = self._nonce()
        
        payload = {
            "request": "/v1/order/new",
            "nonce": nonce,
            "symbol": symbol,
            "amount": str(amount_usd),
            "price": "1",
            "exchange": "bitfinex",
            "side": "buy",
            "type": "market"
        }
        headers = self._sign_payload(payload)
        
        try:
            response = requests.post(url, headers=headers, timeout=10)
            return response.json()
        except Exception as e:
            cprint(f"Error executing market buy: {e}", 'red')
            raise e
    
    def market_sell(self, symbol, amount):
        """Execute market sell order"""
        url = f"{self.base_url}/v1/order/new"
        nonce = self._nonce()
        
        payload = {
            "request": "/v1/order/new",
            "nonce": nonce,
            "symbol": symbol,
            "amount": str(amount),
            "price": "1",
            "exchange": "bitfinex",
            "side": "sell",
            "type": "market"
        }
        headers = self._sign_payload(payload)
        
        try:
            response = requests.post(url, headers=headers, timeout=10)
            return response.json()
        except Exception as e:
            cprint(f"Error executing market sell: {e}", 'red')
            raise e
    
    def limit_buy(self, symbol, amount_usd, price):
        """Execute limit buy order"""
        url = f"{self.base_url}/v1/order/new"
        nonce = self._nonce()
        
        payload = {
            "request": "/v1/order/new",
            "nonce": nonce,
            "symbol": symbol,
            "amount": str(amount_usd / price),  # Convert USD to base currency amount
            "price": str(price),
            "exchange": "bitfinex",
            "side": "buy",
            "type": "exchange limit"
        }
        headers = self._sign_payload(payload)
        
        try:
            response = requests.post(url, headers=headers, timeout=10)
            return response.json()
        except Exception as e:
            cprint(f"Error executing limit buy: {e}", 'red')
            raise e
    
    def limit_sell(self, symbol, amount, price):
        """Execute limit sell order"""
        url = f"{self.base_url}/v1/order/new"
        nonce = self._nonce()
        
        payload = {
            "request": "/v1/order/new",
            "nonce": nonce,
            "symbol": symbol,
            "amount": str(amount),
            "price": str(price),
            "exchange": "bitfinex",
            "side": "sell",
            "type": "exchange limit"
        }
        headers = self._sign_payload(payload)
        
        try:
            response = requests.post(url, headers=headers, timeout=10)
            return response.json()
        except Exception as e:
            cprint(f"Error executing limit sell: {e}", 'red')
            raise e

# Initialize API client
api = BitfinexAPI(API_KEY, API_SECRET)

# --- Helper Functions ---

def _get_position_details(symbol):
    """Fetches position size, price, and USD value."""
    pos = api.get_position(symbol)
    price = api.get_ticker(symbol)
    pos_usd = pos * price if price else 0
    return pos, price, pos_usd

def _calculate_chunk_size(size_needed, max_chunk):
    """Calculates the chunk size for an order, respecting the max limit."""
    chunk = max_chunk if size_needed > max_chunk else size_needed
    return chunk

def _attempt_market_buy(symbol, chunk_size, orders_count, sleep_time):
    """Attempts to execute market buy orders with retries."""
    try:
        for _ in range(orders_count):
            api.market_buy(symbol, chunk_size)
            cprint(f'Chunk buy submitted for {symbol} size: ${chunk_size}', 'white', 'on_blue')
            time.sleep(1) # Small delay between individual orders
        time.sleep(sleep_time) # Longer delay after a burst
        return True
    except Exception as e:
        cprint(f"Error during market buy attempt: {e}", 'light_yellow', 'on_red')
        return False

def _retry_market_buy(symbol, chunk_size, orders_count, sleep_time, retry_delay=30):
    """Retries market buy after a delay."""
    cprint(f'Retrying market buy in {retry_delay} seconds...', 'light_blue', 'on_light_magenta')
    time.sleep(retry_delay)
    try:
        for _ in range(orders_count):
            api.market_buy(symbol, chunk_size)
            cprint(f'Retry chunk buy submitted for {symbol} size: ${chunk_size}', 'white', 'on_blue')
            time.sleep(1)
        time.sleep(sleep_time)
        return True
    except Exception as e:
        cprint(f"Error during retry market buy attempt: {e}", 'white', 'on_red')
        return False

# --- Mode Execution Functions ---

def run_close_mode(symbol):
    """Closes the position for the given symbol in chunks."""
    print(f'--- Mode {CLOSE_MODE}: Closing Position for {symbol} ---')
    pos = api.get_position(symbol)
    
    while pos > 0.1: # Using a small threshold instead of exactly 0
        print(f"Current position: {pos}. Closing in chunks...")
        try:
            chunk_size = min(pos, MAX_USD_ORDER_SIZE / api.get_ticker(symbol))
            api.market_sell(symbol, chunk_size)
            cprint(f'Chunk sell order sent for {symbol}', 'white', 'on_magenta')
        except Exception as e:
            cprint(f"Error during chunk_sell: {e}. Retrying in 5 sec...", 'red')
            time.sleep(5) # Wait before retrying get_position
        
        time.sleep(1) # Wait a bit before checking position again
        pos = api.get_position(symbol) # Refresh position status

    cprint(f'Position for {symbol} closed.', 'white', 'on_green')

def run_buy_mode(symbol):
    """Opens a buying position up to USD_SIZE for the given symbol."""
    print(f'--- Mode {BUY_MODE}: Opening Position for {symbol} ---')
    
    # Initial check
    pos, price, pos_usd = _get_position_details(symbol)
    print(f'Initial State - Position: {round(pos,2)}, Price: {round(price,8)}, Value: ${round(pos_usd,2)}, Target: ${USD_SIZE}')

    if pos_usd >= (0.97 * USD_SIZE):
        cprint('Position already filled or close to target size.', 'yellow')
        return

    while pos_usd < (0.97 * USD_SIZE):
        size_needed = USD_SIZE - pos_usd
        chunk_size = _calculate_chunk_size(size_needed, MAX_USD_ORDER_SIZE)
        
        print(f'Need ${round(size_needed, 2)}. Buying chunk: ${chunk_size}.')

        # Attempt 1
        if not _attempt_market_buy(symbol, chunk_size, ORDERS_PER_OPEN, TX_SLEEP):
            # Attempt 2 (Retry)
            if not _retry_market_buy(symbol, chunk_size, ORDERS_PER_OPEN, TX_SLEEP):
                cprint(f'FINAL ERROR in buy process for {symbol}. Exiting buy mode.', 'white', 'on_red')
                break # Exit the while loop on final failure

        # Refresh position status after attempts
        try:
            pos, price, pos_usd = _get_position_details(symbol)
            print(f'Updated State - Position: {round(pos,2)}, Price: {round(price,8)}, Value: ${round(pos_usd,2)}')
        except Exception as e:
             cprint(f"Error fetching position details after buy attempt: {e}. Exiting.", 'red')
             break

    if pos_usd >= (0.97 * USD_SIZE):
        cprint(f'Position filled for {symbol}, total value: ${round(pos_usd,2)}', 'white', 'on_green')
    else:
         cprint(f'Exited buy mode for {symbol}, final value: ${round(pos_usd,2)}', 'yellow')

def run_eth_trade_mode():
    """Executes trades based on ETH SMA strategy."""
    print(f'--- Mode {ETH_TRADE_MODE}: Starting ETH SMA Trade Logic ---')
    
    # For Bitfinex, we'll need to implement candle data fetching
    # This is a simplified version - you may want to use a proper technical analysis library
    try:
        eth_price = api.get_ticker('ethusd')
        if not eth_price:
            cprint("Failed to fetch ETH price", 'red')
            return
            
        print(f"ETH Price: {eth_price}")
        
        # Simplified SMA logic - in practice, you'd want to fetch historical data
        # and calculate proper SMAs using pandas or similar
        
        # For demonstration, using a simple price-based entry/exit
        # You should implement proper SMA calculation with historical candle data
        
        for symbol in TOKEN_BATCH:
            if symbol in DO_NOT_TRADE_LIST:
                print(f"Skipping {symbol} (in DO_NOT_TRADE_LIST)")
                continue
            
            try:
                pos, price, pos_usd = _get_position_details(symbol)
                print(f"Checking {symbol}: Position Value ${round(pos_usd, 2)}")
                
                # Simplified entry logic - replace with proper SMA comparison
                if pos_usd < (USD_SIZE * 0.1):
                    print(f'Entering position for {symbol}...')
                    buy_amount = min(USD_SIZE * 0.2, MAX_USD_ORDER_SIZE)  # 20% of target size
                    api.market_buy(symbol, buy_amount)
                    cprint(f'Entry order submitted for {symbol}', 'cyan')
                    time.sleep(7)
                else:
                    print(f"Already have sufficient position in {symbol} (${round(pos_usd, 2)})")
                    
            except Exception as e:
                cprint(f"Error processing entry for {symbol}: {e}", 'red')
                time.sleep(5)
                
    except Exception as e:
        cprint(f"Error in ETH trade mode: {e}", 'red')

def run_pnl_close_mode():
    """Closes positions if total balance is outside target range."""
    print(f'--- Mode {PNL_CLOSE_MODE}: PNL-based Position Management ---')
    
    try:
        # Calculate total portfolio value
        total_usd = api.get_balance('usd')
        
        for symbol in TOKEN_BATCH:
            if symbol in DO_NOT_TRADE_LIST:
                continue
                
            pos, price, pos_usd = _get_position_details(symbol)
            total_usd += pos_usd
        
        print(f"Total Portfolio Value: ${round(total_usd, 2)}")
        print(f"Target Range: ${LOWEST_BALANCE} - ${TARGET_BALANCE}")
        
        if total_usd < LOWEST_BALANCE:
            cprint(f"Portfolio below minimum threshold (${LOWEST_BALANCE})", 'yellow')
            # Could implement buy logic here
            
        elif total_usd > TARGET_BALANCE:
            cprint(f"Portfolio above target, closing some positions", 'yellow')
            excess = total_usd - TARGET_BALANCE
            
            for symbol in TOKEN_BATCH:
                if symbol in DO_NOT_TRADE_LIST:
                    continue
                    
                pos, price, pos_usd = _get_position_details(symbol)
                if pos_usd > 0 and excess > 0:
                    sell_amount = min(pos, excess / price)
                    api.market_sell(symbol, sell_amount)
                    cprint(f"Sold {sell_amount} {symbol} for profit taking", 'green')
                    excess -= sell_amount * price
                    time.sleep(2)
                    
                if excess <= 0:
                    break
        else:
            cprint("Portfolio within target range", 'green')
            
    except Exception as e:
        cprint(f"Error in PNL close mode: {e}", 'red')

def run_market_maker_mode(symbol):
    """Executes simple buy under/sell over logic."""
    print(f'--- Mode {MARKET_MAKER_MODE}: Market Making for {symbol} ---')
    
    try:
        pos, price, pos_usd = _get_position_details(symbol)
        print(f"Current State - Position: {pos}, Price: ${price}, Value: ${pos_usd}")
        
        # Calculate buy under and sell over prices
        buy_under_price = price * (1 - BUY_UNDER_PERCENT / 100)
        sell_over_price = price * (1 + SELL_OVER_PERCENT / 100)
        
        print(f"Buy Under: ${buy_under_price}, Sell Over: ${sell_over_price}")
        
        # Place buy order if we have room in position
        if pos_usd < USD_SIZE * 0.8:  # Don't over-allocate
            buy_amount = min(MAX_USD_ORDER_SIZE, USD_SIZE - pos_usd) / buy_under_price
            api.limit_buy(symbol, buy_amount * buy_under_price, buy_under_price)
            cprint(f"Limit buy order placed at ${buy_under_price}", 'blue')
        
        # Place sell order if we have position
        if pos > 0.1:
            sell_amount = min(pos * 0.5, MAX_USD_ORDER_SIZE / sell_over_price)  # Sell half position max
            api.limit_sell(symbol, sell_amount, sell_over_price)
            cprint(f"Limit sell order placed at ${sell_over_price}", 'magenta')
        
        time.sleep(30)  # Wait before next market making cycle
        
    except Exception as e:
        cprint(f"Error during Market Maker execution for {symbol}: {e}", 'red')
        time.sleep(30)

# --- Main Execution ---

def main():
    """Gets user input and runs the selected mode."""
    print('--- Bitfinex Trading Bot ---')
    print(f'{CLOSE_MODE}: Close Position')
    print(f'{BUY_MODE}: Open Buy Position')
    print(f'{ETH_TRADE_MODE}: ETH SMA Strategy')
    print(f'{PNL_CLOSE_MODE}: PNL Close')
    print(f'{MARKET_MAKER_MODE}: Market Maker (Buy Under/Sell Over)')
    print('--------------------------')

    try:
        mode = int(input('Select mode: '))
        print(f'Mode selected: {mode}')
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    # Use primary symbol for modes that operate on a single symbol
    symbol = PRIMARY_SYMBOL

    if mode == CLOSE_MODE:
        run_close_mode(symbol)
    elif mode == BUY_MODE:
        run_buy_mode(symbol)
    elif mode == ETH_TRADE_MODE:
        run_eth_trade_mode()
    elif mode == PNL_CLOSE_MODE:
        run_pnl_close_mode()
    elif mode == MARKET_MAKER_MODE:
        while True:
             try:
                 run_market_maker_mode(symbol)
             except KeyboardInterrupt:
                 print("\nMarket Maker mode stopped by user.")
                 break
             except Exception as e:
                  cprint(f"Unhandled error in Market Maker loop: {e}. Sleeping...", 'red')
                  time.sleep(60)
    else:
        print(f"Unknown mode: {mode}")

    print('--- Script Finished ---')

if __name__ == "__main__":
    main()
