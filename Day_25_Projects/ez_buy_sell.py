"""
Manual trigger script for buying to a target USDC size or closing a position fully.
Reads configuration from config.py.
"""

import time
from termcolor import colored, cprint
import sys, os

# Add parent directory to path so we can import from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import config and nice_funcs, handle if not found
try:
    # Use ALL_CAPS imports from config
    from config import (
        PRIMARY_SYMBOL_MINT, 
        USD_SIZE, 
        MAX_USD_ORDER_SIZE, 
        ORDERS_PER_OPEN, 
        TX_SLEEP, 
        SLIPPAGE
        # Add other necessary configs if used elsewhere, e.g., PRIORITY_FEE if needed by nice_funcs
    )
except ImportError as e:
    print(f"Error importing from config.py: {e}")
    print("Please ensure config.py exists and contains the required variables (e.g., PRIMARY_SYMBOL_MINT, USD_SIZE).")
    exit()
except NameError as e:
    print(f"Error: A required variable might be missing or misspelled in config.py: {e}")
    exit()

try:
    from Day_25_Projects import nice_funcs as n # Assumes nice_funcs.py exists and functions work as expected
except ImportError:
    print("Error: nice_funcs.py not found. Please ensure it's in the same directory or Python path.")
    exit()

# --- Helper Functions ---

def get_position_details(symbol_mint):
    """Gets current position size, token price, and position value in USD."""
    try:
        pos_tokens = n.get_position(symbol_mint)
        price = n.token_price(symbol_mint)
        pos_usd = pos_tokens * price
        return pos_tokens, price, pos_usd
    except Exception as e:
        cprint(f"Error getting position details for {symbol_mint}: {e}", 'white', 'on_red')
        return 0, 0, 0 # Return zeros on error

def calculate_buy_chunk_lamports(target_usd_size, current_pos_usd, max_chunk_usd):
    """Calculates the size of the next buy chunk in lamports (as string)."""
    size_needed_usd = target_usd_size - current_pos_usd
    if size_needed_usd <= 0:
        return "0" # No more needed
        
    chunk_usd = min(size_needed_usd, max_chunk_usd)
    chunk_lamports = int(chunk_usd * 10**6) # Assuming 6 decimals for USDC
    return str(chunk_lamports)

def attempt_market_buy(symbol_mint, chunk_size_lamports_str, slippage_basis_points, num_orders, order_delay_s):
    """Attempts to place market buy orders, returns True if potentially successful."""
    if chunk_size_lamports_str == "0":
        return True # Nothing to buy
        
    try:
        for i in range(num_orders):
            # Assuming market_buy takes mint address, size in base token lamports (USDC), slippage BPS
            n.market_buy(symbol_mint, chunk_size_lamports_str, slippage_basis_points) 
            cprint(f'Chunk buy submitted for {symbol_mint[-6:]}, size: {chunk_size_lamports_str} lamports', 'white', 'on_blue')
            if i < num_orders - 1: # Don't sleep after the last order in the burst
                 time.sleep(order_delay_s)
        return True # Submitted orders
    except Exception as e:
        cprint(f"Error during market buy attempt: {e}", 'yellow', 'on_red')
        return False

# --- Core Logic Functions ---

def close_position_fully(symbol_mint, max_chunk_usd, slippage_basis_points):
    """Closes the entire position for the given symbol in chunks."""
    print(f"Attempting to fully close position for {symbol_mint}...")
    while True:
        pos_tokens, price, pos_usd = get_position_details(symbol_mint)
        print(f"  Current position: {pos_tokens:.6f} tokens (${pos_usd:.2f})")

        if pos_usd < 0.50: # Consider position closed if less than 50 cents USD
            cprint('Position considered closed.', 'white', 'on_green')
            break

        print(f"  Closing in chunks of max ${max_chunk_usd:.2f}...")
        try:
            # Assuming chunk_kill sells max_chunk_usd worth or remaining if less
            n.chunk_kill(symbol_mint, max_chunk_usd, slippage_basis_points) 
            cprint(f"  Submitted sell chunk(s) for {symbol_mint[-6:]}.", 'white', 'on_magenta')
            # Add a small delay to allow blockchain to update and avoid spamming get_position
            time.sleep(TX_SLEEP) 
        except Exception as e:
            cprint(f"Error during chunk_kill: {e}. Retrying in {TX_SLEEP * 2}s...", 'white', 'on_red')
            time.sleep(TX_SLEEP * 2)
            
    print("Close position process finished.")

def open_position_target_size(symbol_mint, target_usd, max_chunk_usd, orders_per_tx, slippage_basis_points, tx_delay_s):
    """Opens a position by buying chunks until the target USDC size is reached."""
    print(f"Attempting to open position for {symbol_mint} to target size ${target_usd:.2f}...")
    
    while True:
        pos_tokens, price, pos_usd = get_position_details(symbol_mint)
        print(f"  Current position: {pos_tokens:.6f} tokens (${pos_usd:.2f}) | Target: ${target_usd:.2f}")
        
        # Check if target reached (within a small tolerance, e.g., 97%)
        if pos_usd >= target_usd * 0.97:
            cprint(f'Target position size reached for {symbol_mint[-6:]} (${pos_usd:.2f}).', 'white', 'on_green')
            break
        
        # Calculate next chunk size
        chunk_lamports_str = calculate_buy_chunk_lamports(target_usd, pos_usd, max_chunk_usd)
        
        if chunk_lamports_str == "0":
             cprint("Calculated chunk size is 0, likely reached target. Exiting buy loop.", 'yellow')
             break
             
        print(f"  Need ~${target_usd - pos_usd:.2f} more. Buying chunk(s) of {chunk_lamports_str} lamports...")

        # Attempt to buy
        buy_successful = attempt_market_buy(symbol_mint, chunk_lamports_str, slippage_basis_points, orders_per_tx, 1) # 1s delay between burst orders
        
        if buy_successful:
            print(f"  Waiting {tx_delay_s}s after buy attempt...")
            time.sleep(tx_delay_s) 
        else:
            # Basic retry mechanism
            print(f"  Buy attempt failed. Retrying in {tx_delay_s * 2}s...")
            time.sleep(tx_delay_s * 2)
            buy_successful = attempt_market_buy(symbol_mint, chunk_lamports_str, slippage_basis_points, orders_per_tx, 1)
            if buy_successful:
                 print(f"  Retry successful. Waiting {tx_delay_s}s...")
                 time.sleep(tx_delay_s)
            else:
                 cprint("  Retry failed. Exiting buy loop to prevent issues.", 'white', 'on_red')
                 break # Exit loop after failed retry
                 
    print("Open position process finished.")

# --- Main Execution Logic ---
def main():
    """Gets user input and triggers the appropriate buy/sell action."""
    print('=' * 20)
    print(colored(' Solana Manual Buy/Close Tool', 'cyan', attrs=['bold']))
    print('=' * 20)
    print(f"Using symbol: {PRIMARY_SYMBOL_MINT}")
    print(f"Target Size (for buy): ${USD_SIZE:.2f}")
    print(f"Max Chunk Size: ${MAX_USD_ORDER_SIZE:.2f}")
    print('\nWARNING: This script executes trades based on your input.')
    print('Ensure configuration in config.py is correct BEFORE running.')
    
    while True:
        try:
            action = input('Enter 0 to CLOSE position fully, or 1 to OPEN position to target size: ')
            action = int(action)
            if action in [0, 1]:
                break
            else:
                print("Invalid input. Please enter 0 or 1.")
        except ValueError:
            print("Invalid input. Please enter a number (0 or 1).")
            
    symbol_to_trade = PRIMARY_SYMBOL_MINT

    if action == 0:
        close_position_fully(symbol_to_trade, MAX_USD_ORDER_SIZE, SLIPPAGE)
    elif action == 1:
        open_position_target_size(symbol_to_trade, USD_SIZE, MAX_USD_ORDER_SIZE, ORDERS_PER_OPEN, SLIPPAGE, TX_SLEEP)
        
    print("\nOperation complete.")
    print("The script will now exit. Re-run to perform another action.")

# --- Main execution guard ---
if __name__ == '__main__':
    main()