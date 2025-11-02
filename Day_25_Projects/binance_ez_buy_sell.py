import time
import sys, os
from termcolor import colored, cprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (PRIMARY_SYMBOL, USD_SIZE, MAX_USD_ORDER_SIZE, ORDERS_PER_OPEN, TX_SLEEP, SLIPPAGE)
except ImportError as e:
    print(f'Error importing from config.py: {e}')
    exit()
except NameError as e:
    print(f'Error: A required variable might be missing in config.py: {e}')
    exit()

try:
    from Day_12_Projects.binance_nice_funcs import get_position, token_price, market_buy, market_sell
except ImportError:
    print('Error: binance_nice_funcs.py not found.')
    exit()

def get_position_details(symbol):
    try:
        pos = get_position(symbol)
        price = token_price(symbol)
        pos_usd = pos * price
        return pos, price, pos_usd
    except Exception as e:
        cprint(f'Error getting position details for {symbol}: {e}', 'white', 'on_red')
        return 0, 0, 0

def calculate_buy_chunk(target_usd_size, current_pos_usd, max_chunk_usd):
    size_needed_usd = target_usd_size - current_pos_usd
    if size_needed_usd <= 0:
        return 0
    chunk_usd = min(size_needed_usd, max_chunk_usd)
    return chunk_usd

def attempt_market_buy(symbol, chunk_size_usd, slippage, num_orders, order_delay_s):
    if chunk_size_usd == 0:
        return True
    try:
        for i in range(num_orders):
            market_buy(symbol, chunk_size_usd, slippage)
            cprint(f'Chunk buy submitted for {symbol}, size: ${chunk_size_usd:.2f}', 'white', 'on_blue')
            if i < num_orders - 1:
                time.sleep(order_delay_s)
        return True
    except Exception as e:
        cprint(f'Error during market buy attempt: {e}', 'yellow', 'on_red')
        return False

def close_position_fully(symbol, max_chunk_usd, slippage):
    print(f'Attempting to fully close position for {symbol}...')
    while True:
        pos, price, pos_usd = get_position_details(symbol)
        print(f'  Current position: {pos:.6f} ({pos_usd:.2f} USD)')
        if abs(pos_usd) < 0.50:
            cprint('Position considered closed.', 'white', 'on_green')
            break
        chunk_usd = min(abs(pos_usd), max_chunk_usd)
        try:
            if pos > 0:
                market_sell(symbol, chunk_usd, slippage)
            elif pos < 0:
                market_buy(symbol, chunk_usd, slippage)  # Buy to close short
            cprint(f'Submitted close chunk for {symbol}.', 'white', 'on_magenta')
            time.sleep(TX_SLEEP)
        except Exception as e:
            cprint(f'Error during close: {e}. Retrying in {TX_SLEEP * 2}s...', 'white', 'on_red')
            time.sleep(TX_SLEEP * 2)
    print('Close position process finished.')

def open_position_target_size(symbol, target_usd, max_chunk_usd, orders_per_tx, slippage, tx_delay_s):
    print(f'Attempting to open long position for {symbol} to target size ${target_usd:.2f}...')
    while True:
        pos, price, pos_usd = get_position_details(symbol)
        print(f'  Current position: {pos:.6f} ({pos_usd:.2f} USD) | Target: ${target_usd:.2f}')
        if pos_usd >= target_usd * 0.97:
            cprint(f'Target position size reached for {symbol} (${pos_usd:.2f}).', 'white', 'on_green')
            break
        chunk_usd = calculate_buy_chunk(target_usd, pos_usd, max_chunk_usd)
        if chunk_usd == 0:
            cprint('Calculated chunk size is 0, likely reached target. Exiting buy loop.', 'yellow')
            break
        print(f'  Need ~${target_usd - pos_usd:.2f} more. Buying chunk of ${chunk_usd:.2f}...')
        buy_successful = attempt_market_buy(symbol, chunk_usd, slippage, orders_per_tx, 1)
        if buy_successful:
            print(f'  Waiting {tx_delay_s}s after buy attempt...')
            time.sleep(tx_delay_s)
        else:
            print(f'  Buy attempt failed. Retrying in {tx_delay_s * 2}s...')
            time.sleep(tx_delay_s * 2)
            buy_successful = attempt_market_buy(symbol, chunk_usd, slippage, orders_per_tx, 1)
            if buy_successful:
                print(f'  Retry successful. Waiting {tx_delay_s}s...')
                time.sleep(tx_delay_s)
            else:
                cprint('  Retry failed. Exiting buy loop.', 'white', 'on_red')
                break
    print('Open position process finished.')

def main():
    print('=' * 20)
    print(colored(' Binance Manual Buy/Close Tool', 'cyan', attrs=['bold']))
    print('=' * 20)
    print(f'Using symbol: {PRIMARY_SYMBOL}')
    print(f'Target Size (for buy): ${USD_SIZE:.2f}')
    print(f'Max Chunk Size: ${MAX_USD_ORDER_SIZE:.2f}')
    print('\nWARNING: This script executes trades based on your input.')
    print('Ensure configuration in config.py is correct BEFORE running.')
    while True:
        try:
            action = input('Enter 0 to CLOSE position fully, or 1 to OPEN position to target size: ')
            action = int(action)
            if action in [0, 1]:
                break
            else:
                print('Invalid input. Please enter 0 or 1.')
        except ValueError:
            print('Invalid input. Please enter a number (0 or 1).')
    symbol_to_trade = PRIMARY_SYMBOL
    if action == 0:
        close_position_fully(symbol_to_trade, MAX_USD_ORDER_SIZE, SLIPPAGE)
    elif action == 1:
        open_position_target_size(symbol_to_trade, USD_SIZE, MAX_USD_ORDER_SIZE, ORDERS_PER_OPEN, SLIPPAGE, TX_SLEEP)
    print('\nOperation complete.')
    print('The script will now exit. Re-run to perform another action.')

if __name__ == '__main__':
    main() 