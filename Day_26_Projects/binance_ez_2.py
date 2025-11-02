"""
Binance Trading Bot Infrastructure

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
from urllib.parse import urlencode
from termcolor import cprint

# Import specific config variables (adapted for Binance)
from binance_config import (
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

# --- Binance API Helper Class ---
class BinanceAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        self.headers = {"X-MBX-APIKEY": self.api_key}
    
    def _get_timestamp(self):
        return int(time.time() * 1000)
    
    def _sign(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method, endpoint, params=None, signed=False):
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = self._get_timestamp()
            params['signature'] = self._sign(params)
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, data=params, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers, params=params, timeout=10)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            cprint(f"API request error: {e}", 'red')
            raise e
    
    def get_ticker(self, symbol):
        """Get current ticker price for symbol"""
        try:
            data = self._request('GET', '/api/v3/ticker/price', {'symbol': symbol.upper()})
            return float(data['price'])
        except Exception as e:
            cprint(f"Error fetching ticker for {symbol}: {e}", 'red')
            return None
    
    def get_balance(self, asset):
        """Get balance for specific asset"""
        try:
            account_info = self._request('GET', '/api/v3/account', signed=True)
            
            for balance in account_info['balances']:
                if balance['asset'].upper() == asset.upper():
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            cprint(f"Error fetching balance for {asset}: {e}", 'red')
            return 0.0
    
    def get_position(self, symbol):
        """Get current position size for symbol"""
        # For spot trading, position = available balance of base asset
        base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        return self.get_balance(base_asset)
    
    def market_buy(self, symbol, amount_usdt):
        """Execute market buy order"""
        params = {
            'symbol': symbol.upper(),
            'side': 'BUY',
            'type': 'MARKET',
            'quoteOrderQty': str(amount_usdt)  # Buy with USDT amount
        }
        
        try:
            result = self._request('POST', '/api/v3/order', params, signed=True)
            return result
        except Exception as e:
            cprint(f"Error executing market buy: {e}", 'red')
            raise e
    
    def market_sell(self, symbol, quantity):
        """Execute market sell order"""
        params = {
            'symbol': symbol.upper(),
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': str(quantity)
        }
        
        try:
            result = self._request('POST', '/api/v3/order', params, signed=True)
            return result
        except Exception as e:
            cprint(f"Error executing market sell: {e}", 'red')
            raise e
    
    def limit_buy(self, symbol, quantity, price):
        """Execute limit buy order"""
        params = {
            'symbol': symbol.upper(),
            'side': 'BUY',
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': str(quantity),
            'price': str(price)
        }
        
        try:
            result = self._request('POST', '/api/v3/order', params, signed=True)
            return result
        except Exception as e:
            cprint(f"Error executing limit buy: {e}", 'red')
            raise e
    
    def limit_sell(self, symbol, quantity, price):
        """Execute limit sell order"""
        params = {
            'symbol': symbol.upper(),
            'side': 'SELL',
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': str(quantity),
            'price': str(price)
        }
        
        try:
            result = self._request('POST', '/api/v3/order', params, signed=True)
            return result
        except Exception as e:
            cprint(f"Error executing limit sell: {e}", 'red')
            raise e
    
    def get_klines(self, symbol, interval='1d', limit=200):
        """Get historical kline/candlestick data"""
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        try:
            data = self._request('GET', '/api/v3/klines', params)
            return data
        except Exception as e:
            cprint(f"Error fetching klines for {symbol}: {e}", 'red')
            return None
    
    def calculate_sma(self, prices, window):
        """Calculate Simple Moving Average"""
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window
    
    def get_sma_data(self, symbol, interval='1d', limit=200, sma_windows=[20, 41]):
        """Get price data with SMA calculations"""
        klines = self.get_klines(symbol, interval, limit)
        if not klines:
            return None
        
        # Extract closing prices
        closes = [float(kline[4]) for kline in klines]
        current_price = closes[-1]
        
        sma_data = {'price': current_price}
        
        for window in sma_windows:
            sma = self.calculate_sma(closes, window)
            sma_data[f'sma_{window}'] = sma
            if sma:
                sma_data[f'price_gt_sma_{window}'] = current_price > sma
        
        return sma_data

# Initialize API client
api = BinanceAPI(API_KEY, API_SECRET)

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
            # Calculate chunk size based on max USD order size
            price = api.get_ticker(symbol)
            chunk_size = min(pos, MAX_USD_ORDER_SIZE / price) if price else pos * 0.1
            
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
    
    try:
        # Fetch ETH data with SMA calculations
        eth_data = api.get_sma_data('ETHUSDT', '1d', 200, sma_windows=[20, 41])
        
        if not eth_data:
            cprint("Failed to fetch ETH data", 'red')
            return
            
        eth_price = eth_data['price']
        sma20 = eth_data.get('sma_20')
        sma41 = eth_data.get('sma_41')
        price_gt_sma_20 = eth_data.get('price_gt_sma_20', False)
        price_gt_sma_41 = eth_data.get('price_gt_sma_41', False)
        
        print(f"ETH Price: {eth_price}, SMA20: {sma20}, SMA41: {sma41}")
        print(f"Price > SMA20: {price_gt_sma_20}, Price > SMA41: {price_gt_sma_41}")
        
        # --- Entry Logic (Price > SMA20) ---
        if price_gt_sma_20:
            print(f'ENTRY Signal: ETH Price {eth_price} > SMA20 {sma20}')
            
            for symbol in TOKEN_BATCH:
                if symbol in DO_NOT_TRADE_LIST:
                    print(f"Skipping {symbol} (in DO_NOT_TRADE_LIST)")
                    continue
                
                try:
                    pos, price, pos_usd = _get_position_details(symbol)
                    print(f"Checking {symbol}: Position Value ${round(pos_usd, 2)}")
                    
                    # Enter only if position is small/zero
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
        
        # --- Exit Logic (Price < SMA41) ---
        elif not price_gt_sma_41:
            print(f'EXIT Signal: ETH Price {eth_price} < SMA41 {sma41}')
            
            for symbol in TOKEN_BATCH:
                if symbol in DO_NOT_TRADE_LIST:
                    continue
                    
                try:
                    pos, price, pos_usd = _get_position_details(symbol)
                    
                    if pos_usd > (USD_SIZE * 0.05):  # Exit if position > 5% of target
                        print(f'Exiting position for {symbol}...')
                        api.market_sell(symbol, pos)
                        cprint(f'Exit order submitted for {symbol}', 'red')
                        time.sleep(5)
                        
                except Exception as e:
                    cprint(f"Error processing exit for {symbol}: {e}", 'red')
                    time.sleep(5)
        else:
            print("No clear signal - holding current positions")
            
    except Exception as e:
        cprint(f"Error in ETH trade mode: {e}", 'red')

def run_pnl_close_mode():
    """Closes positions if total balance is outside target range."""
    print(f'--- Mode {PNL_CLOSE_MODE}: PNL-based Position Management ---')
    
    try:
        # Calculate total portfolio value
        total_usdt = api.get_balance('USDT')
        
        for symbol in TOKEN_BATCH:
            if symbol in DO_NOT_TRADE_LIST:
                continue
                
            pos, price, pos_usd = _get_position_details(symbol)
            total_usdt += pos_usd
        
        print(f"Total Portfolio Value: ${round(total_usdt, 2)}")
        print(f"Target Range: ${LOWEST_BALANCE} - ${TARGET_BALANCE}")
        
        if total_usdt < LOWEST_BALANCE:
            cprint(f"Portfolio below minimum threshold (${LOWEST_BALANCE})", 'yellow')
            # Could implement buy logic here
            
        elif total_usdt > TARGET_BALANCE:
            cprint(f"Portfolio above target, closing some positions", 'yellow')
            excess = total_usdt - TARGET_BALANCE
            
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
            buy_amount_usd = min(MAX_USD_ORDER_SIZE, USD_SIZE - pos_usd)
            buy_quantity = buy_amount_usd / buy_under_price
            api.limit_buy(symbol, buy_quantity, buy_under_price)
            cprint(f"Limit buy order placed: {buy_quantity} at ${buy_under_price}", 'blue')
        
        # Place sell order if we have position
        if pos > 0.1:
            sell_amount = min(pos * 0.5, MAX_USD_ORDER_SIZE / sell_over_price)  # Sell half position max
            api.limit_sell(symbol, sell_amount, sell_over_price)
            cprint(f"Limit sell order placed: {sell_amount} at ${sell_over_price}", 'magenta')
        
        time.sleep(30)  # Wait before next market making cycle
        
    except Exception as e:
        cprint(f"Error during Market Maker execution for {symbol}: {e}", 'red')
        time.sleep(30)

# --- Main Execution ---

def main():
    """Gets user input and runs the selected mode."""
    print('--- Binance Trading Bot ---')
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
