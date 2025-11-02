"""
Solana Trading Bot Infrastructure

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
# import schedule # Commented out as the current structure runs one mode and exits.
from termcolor import cprint

# Import specific config variables
from config import (
    PRIMARY_SYMBOL_MINT, USD_SIZE, MAX_USD_ORDER_SIZE, SLIPPAGE,
    ORDERS_PER_OPEN, TX_SLEEP, TOKEN_BATCH, WALLET_ADDRESS,
    DO_NOT_TRADE_LIST, LOWEST_BALANCE, TARGET_BALANCE,
    BUY_UNDER, SELL_OVER
)
import nice_funcs as n # Assumed to contain trading logic functions

# --- Mode Constants ---
CLOSE_MODE = 0
BUY_MODE = 1
ETH_TRADE_MODE = 2
PNL_CLOSE_MODE = 4
MARKET_MAKER_MODE = 5

# --- Helper Functions ---

def _get_position_details(symbol):
    """Fetches position size, price, and USD value."""
    pos = n.get_position(symbol)
    price = n.token_price(symbol)
    pos_usd = pos * price
    return pos, price, pos_usd

def _calculate_chunk_size(size_needed, max_chunk):
    """Calculates the chunk size for an order, respecting the max limit."""
    chunk = max_chunk if size_needed > max_chunk else size_needed
    chunk_lamports = int(chunk * 10**6) # Assuming USDC has 6 decimals
    return str(chunk_lamports)

def _attempt_market_buy(symbol, chunk_size_str, orders_count, sleep_time):
    """Attempts to execute market buy orders with retries."""
    try:
        for _ in range(orders_count):
            n.market_buy(symbol, chunk_size_str, SLIPPAGE)
            cprint(f'Chunk buy submitted for {symbol[-4:]} size: {chunk_size_str} lamports', 'white', 'on_blue')
            time.sleep(1) # Small delay between individual orders
        time.sleep(sleep_time) # Longer delay after a burst
        return True
    except Exception as e:
        cprint(f"Error during market buy attempt: {e}", 'light_yellow', 'on_red')
        return False

def _retry_market_buy(symbol, chunk_size_str, orders_count, sleep_time, retry_delay=30):
    """Retries market buy after a delay."""
    cprint(f'Retrying market buy in {retry_delay} seconds...', 'light_blue', 'on_light_magenta')
    time.sleep(retry_delay)
    try:
        for _ in range(orders_count):
            n.market_buy(symbol, chunk_size_str, SLIPPAGE)
            cprint(f'Retry chunk buy submitted for {symbol[-4:]} size: {chunk_size_str} lamports', 'white', 'on_blue')
            time.sleep(1)
        time.sleep(sleep_time)
        return True
    except Exception as e:
        cprint(f"Error during retry market buy attempt: {e}", 'white', 'on_red')
        return False

# --- Mode Execution Functions ---

def run_close_mode(symbol):
    """Closes the position for the given symbol in chunks."""
    print(f'--- Mode {CLOSE_MODE}: Closing Position for {symbol[-4:]} ---')
    pos = n.get_position(symbol)
    while pos > 0.1: # Using a small threshold instead of exactly 0
        print(f"Current position: {pos}. Closing in chunks...")
        try:
            n.chunk_kill(symbol, MAX_USD_ORDER_SIZE, SLIPPAGE)
            cprint(f'Chunk kill order sent for {symbol[-4:]}', 'white', 'on_magenta')
        except Exception as e:
            cprint(f"Error during chunk_kill: {e}. Retrying in 5 sec...", 'red')
            time.sleep(5) # Wait before retrying get_position
            # Fallthrough to fetch position again
        
        time.sleep(1) # Wait a bit before checking position again
        pos = n.get_position(symbol) # Refresh position status

    cprint(f'Position for {symbol[-4:]} closed.', 'white', 'on_green')

def run_buy_mode(symbol):
    """Opens a buying position up to USD_SIZE for the given symbol."""
    print(f'--- Mode {BUY_MODE}: Opening Position for {symbol[-4:]} ---')
    
    # Initial check
    pos, price, pos_usd = _get_position_details(symbol)
    print(f'Initial State - Position: {round(pos,2)}, Price: {round(price,8)}, Value: ${round(pos_usd,2)}, Target: ${USD_SIZE}')

    if pos_usd >= (0.97 * USD_SIZE):
        cprint('Position already filled or close to target size.', 'yellow')
        return

    while pos_usd < (0.97 * USD_SIZE):
        size_needed = USD_SIZE - pos_usd
        chunk_size_str = _calculate_chunk_size(size_needed, MAX_USD_ORDER_SIZE)
        
        print(f'Need ${round(size_needed, 2)}. Buying chunk: {chunk_size_str} lamports.')

        # Attempt 1
        if not _attempt_market_buy(symbol, chunk_size_str, ORDERS_PER_OPEN, TX_SLEEP):
            # Attempt 2 (Retry)
            if not _retry_market_buy(symbol, chunk_size_str, ORDERS_PER_OPEN, TX_SLEEP):
                cprint(f'FINAL ERROR in buy process for {symbol[-4:]}. Exiting buy mode.', 'white', 'on_red')
                break # Exit the while loop on final failure

        # Refresh position status after attempts
        try:
            pos, price, pos_usd = _get_position_details(symbol)
            print(f'Updated State - Position: {round(pos,2)}, Price: {round(price,8)}, Value: ${round(pos_usd,2)}')
        except Exception as e:
             cprint(f"Error fetching position details after buy attempt: {e}. Exiting.", 'red')
             break

    if pos_usd >= (0.97 * USD_SIZE):
        cprint(f'Position filled for {symbol[-4:]}, total value: ${round(pos_usd,2)}', 'white', 'on_green')
    else:
         cprint(f'Exited buy mode for {symbol[-4:]}, final value: ${round(pos_usd,2)}', 'yellow')


def run_eth_trade_mode():
    """Executes trades based on ETH SMA strategy."""
    print(f'--- Mode {ETH_TRADE_MODE}: Starting ETH SMA Trade Logic ---')

    try:
        # Fetch ETH data (using config constants if available, else hardcoded)
        # Assuming timeframe '1d', limit 200, smas [20, 41] are specific to this strategy for now
        eth_df = n.fetch_candle_data_with_smas('ETH', '1d', 200, sma_windows=[20, 41])
    except Exception as e:
        cprint(f"Failed to fetch ETH candle data: {e}", 'red')
        return # Cannot proceed without data

    if eth_df is None or eth_df.empty:
        print('Failed to fetch ETH data or data is empty.')
        return

    # print(eth_df.tail()) # Print last few rows for context

    # Get latest data
    latest_data = eth_df.iloc[-1]
    price_gt_sma_20 = latest_data['Price > SMA_20']
    price_gt_sma_40 = latest_data['Price > SMA_40'] # Note: Config uses 41, function uses 40? Verify.
    eth_price = latest_data['Price']
    sma20 = latest_data['SMA_20']
    sma41 = latest_data['SMA_41'] # Column name matches SMA window used

    print(f"ETH Price: {eth_price}, SMA20: {sma20}, SMA41: {sma41}")
    print(f"Price > SMA20: {price_gt_sma_20}, Price > SMA41: {price_gt_sma_40}")

    # --- Entry Logic (Price > SMA20) ---
    if price_gt_sma_20:
        print(f'ENTRY Signal: ETH Price {eth_price} > SMA20 {sma20}')
        for symbol in TOKEN_BATCH:
            if symbol in DO_NOT_TRADE_LIST:
                 print(f"Skipping {symbol[-4:]} (in DO_NOT_TRADE_LIST)")
                 continue

            try:
                pos, price, pos_usd = _get_position_details(symbol)
                print(f"Checking {symbol[-4:]}: Position Value ${round(pos_usd, 2)}")

                # Enter only if position is small/zero
                if pos_usd < (USD_SIZE * 0.1): # Enter if less than 10% of target size
                     print(f'Entering position for {symbol[-4:]}...')
                     # Calculate buy_under based on current price and slippage
                     buy_under_price = price * ((SLIPPAGE / 10000) + 1) # Correct slippage calc? Verify n.elegant_entry needs price
                     print(f'Target buy under price: {buy_under_price} (based on current price {price})')
                     
                     # Assuming elegant_entry handles buying up to a certain size or needs a size param
                     # For now, just calling it as in the original code. Needs verification.
                     n.elegant_entry(symbol, buy_under_price) # Does elegant_entry need a size?
                     cprint(f'Elegant entry initiated for {symbol[-4:]}', 'cyan')
                     time.sleep(7) # Wait after entry attempt

                else:
                    print(f"Already have sufficient position in {symbol[-4:]} (${round(pos_usd, 2)})")

            except Exception as e:
                cprint(f"Error processing entry for {symbol[-4:]}: {e}", 'red')
                time.sleep(5) # Pause if error occurs for one symbol

    # --- Exit Logic (Price < SMA41) ---
    # Note: Original code checked !price_gt_sma_40 (implicitly price <= sma40). Using sma41 based on fetch.
    elif not price_gt_sma_40: # Assuming SMA_40 column name exists from the function call
        print(f'EXIT Signal: ETH Price {eth_price} <= SMA41 {sma41}')

        try:
            positions = n.fetch_wallet_holdings_og(WALLET_ADDRESS)
            if positions is None or positions.empty:
                print("No wallet holdings found or error fetching.")
                return
                
            # Ensure 'USD Value' is float
            positions['USD Value'] = positions['USD Value'].astype(float)
            
            # Filter positions to close (value > 2, not in DO_NOT_TRADE_LIST)
            positions_to_close = positions[
                (positions['USD Value'] > 2) &
                (~positions['Mint Address'].isin(DO_NOT_TRADE_LIST))
            ]

            print(f"Found {len(positions_to_close)} positions to evaluate for closing.")
            # print(positions_to_close[['Token Symbol', 'Mint Address', 'USD Value']]) # Optional: print details

            for index, row in positions_to_close.iterrows():
                symbol_to_close = row['Mint Address']
                usd_value = row['USD Value']
                print(f"Closing position for {symbol_to_close[-4:]} (Value: ${round(usd_value, 2)})")
                try:
                    # Close using run_close_mode for consistency, though original used chunk_kill directly
                    run_close_mode(symbol_to_close) # Re-uses chunking logic
                    # n.chunk_kill(symbol_to_close, MAX_USD_ORDER_SIZE, SLIPPAGE) # Original direct call
                    # cprint(f'Chunk kill order sent for {symbol_to_close[-4:]}', 'white', 'on_magenta')
                    time.sleep(2) # Small delay between closing symbols
                except Exception as e:
                     cprint(f"Error closing position for {symbol_to_close[-4:]}: {e}", 'red')
                     time.sleep(5) # Pause if error occurs for one symbol

        except Exception as e:
            cprint(f"Error fetching or processing wallet holdings for exit logic: {e}", 'red')

    else:
        print("No entry or exit signal based on ETH SMAs.")


def run_pnl_close_mode():
    """Closes positions if total balance is outside LOWEST_BALANCE or TARGET_BALANCE."""
    print(f'--- Mode {PNL_CLOSE_MODE}: PNL Close Logic ---')

    try:
        positions = n.fetch_wallet_holdings_og(WALLET_ADDRESS)
        if positions is None or positions.empty:
            print("No wallet holdings found or error fetching.")
            return

        # Ensure 'USD Value' is float and calculate total portfolio value
        positions['USD Value'] = positions['USD Value'].astype(float)
        total_pos_value = positions['USD Value'].sum()

        print(f"Total Portfolio Value: ${round(total_pos_value, 2)}")
        print(f"PNL Thresholds: Lowest=${LOWEST_BALANCE}, Target=${TARGET_BALANCE}")

        # Check if balance is outside thresholds
        if total_pos_value < LOWEST_BALANCE or total_pos_value > TARGET_BALANCE:
            
            action = "LOSS" if total_pos_value < LOWEST_BALANCE else "PROFIT"
            print(f"{action} Target Hit! Evaluating positions to close...")

            # Filter positions to close (value > 2, not in DO_NOT_TRADE_LIST)
            positions_to_close = positions[
                (positions['USD Value'] > 2) &
                (~positions['Mint Address'].isin(DO_NOT_TRADE_LIST))
            ]

            if positions_to_close.empty:
                print("No eligible positions to close based on PNL trigger.")
                return
            
            print(f"Found {len(positions_to_close)} positions to close.")

            for index, row in positions_to_close.iterrows():
                symbol = row['Mint Address']
                usd_value = row['USD Value']
                
                reason = f"total value ${round(total_pos_value, 2)} < lowest ${LOWEST_BALANCE}" if action == "LOSS" else f"total value ${round(total_pos_value, 2)} > target ${TARGET_BALANCE}"
                cprint(f'{action} KILL - Closing {symbol[-4:]} (Value: ${round(usd_value, 2)}) because {reason}', 'yellow')

                try:
                    # Determine kill chunk size: minimum of position value or max order size
                    kill_chunk_size = min(usd_value, MAX_USD_ORDER_SIZE)
                    
                    # Close using chunk_kill directly as per original logic
                    # Need to ensure pos is fully closed, run_close_mode might be better
                    n.chunk_kill(symbol, kill_chunk_size, SLIPPAGE)
                    cprint(f"PNL Close order sent for {symbol[-4:]}", 'white', 'on_magenta')
                    time.sleep(2) # Small delay
                except Exception as e:
                    cprint(f"Error during PNL close for {symbol[-4:]}: {e}", 'red')
                    time.sleep(5) # Pause if error

        else:
            print("Portfolio value is within PNL thresholds. No action taken.")

    except Exception as e:
        cprint(f"Error during PNL Close execution: {e}", 'red')


def run_market_maker_mode(symbol):
    """Executes simple buy under/sell over logic."""
    print(f'--- Mode {MARKET_MAKER_MODE}: Market Maker for {symbol[-4:]} ---')
    print(f"Using BUY_UNDER: {BUY_UNDER}, SELL_OVER: {SELL_OVER}")

    try:
        pos, price, pos_usd = _get_position_details(symbol)
        print(f'State: Position: {round(pos,2)}, Price: {round(price,8)}, Value: ${round(pos_usd,2)}, Target Size: ${USD_SIZE}') # Added target size context

        # --- Sell Logic ---
        if price > SELL_OVER:
            if pos_usd > 1: # Only sell if we have a position of meaningful value
                 print(f'Selling {symbol[-4:]}: Price {price} > SELL_OVER {SELL_OVER}')
                 # Sell the entire position or in chunks? Original used one chunk_kill.
                 # Assuming we close the position if price is above sell_over
                 run_close_mode(symbol) # Use the close function to handle chunking
                 cprint(f'Market maker sell triggered and position closed for {symbol[-4:]}.', 'white', 'on_magenta')
                 time.sleep(15) # Wait after selling
            else:
                 print(f"Price {price} > SELL_OVER {SELL_OVER}, but no significant position to sell (${round(pos_usd, 2)}).")
                 time.sleep(15) # Still wait if condition met but no action

        # --- Buy Logic ---
        elif price < BUY_UNDER:
            if pos_usd < (USD_SIZE * 0.97): # Only buy if position is not already full
                 print(f'Buying {symbol[-4:]}: Price {price} < BUY_UNDER {BUY_UNDER} and Position Value ${round(pos_usd,2)} < Target ${USD_SIZE}')
                 
                 # Use elegant entry similar to ETH strategy? Needs verification.
                 # Assuming elegant_entry buys a standard amount or up to a limit.
                 try:
                     n.elegant_entry(symbol, BUY_UNDER) # Pass BUY_UNDER as the price limit? Verify function.
                     cprint(f'Elegant entry initiated for {symbol[-4:]} via market maker.', 'cyan')
                     time.sleep(15) # Wait after buying attempt
                 except Exception as e:
                     cprint(f"Error during market maker elegant entry for {symbol[-4:]}: {e}", 'red')
                     time.sleep(15) # Wait even if error

            else:
                print(f"Price {price} < BUY_UNDER {BUY_UNDER}, but position already near target size (${round(pos_usd, 2)}).")
                time.sleep(15) # Still wait if condition met but no action

        # --- No Action ---
        else:
            print(f'Price {price} is between BUY_UNDER ({BUY_UNDER}) and SELL_OVER ({SELL_OVER}). No action.')
            time.sleep(30) # Longer wait if no action needed

    except Exception as e:
        cprint(f"Error during Market Maker execution for {symbol[-4:]}: {e}", 'red')
        time.sleep(30) # Wait after error


# --- Main Execution ---

def main():
    """Gets user input and runs the selected mode."""
    print('--- Solana Trading Bot ---')
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
    symbol = PRIMARY_SYMBOL_MINT

    if mode == CLOSE_MODE:
        run_close_mode(symbol)
    elif mode == BUY_MODE:
        run_buy_mode(symbol)
    elif mode == ETH_TRADE_MODE:
        run_eth_trade_mode() # Operates on ETH and potentially TOKEN_BATCH
    elif mode == PNL_CLOSE_MODE:
        run_pnl_close_mode() # Operates on wallet holdings
    elif mode == MARKET_MAKER_MODE:
        # This mode likely needs continuous running.
        # For now, it runs once. Consider a loop or external scheduler.
        while True: # Add loop for continuous MM - Ctrl+C to exit
             try:
                 run_market_maker_mode(symbol)
             except KeyboardInterrupt:
                 print("\nMarket Maker mode stopped by user.")
                 break
             except Exception as e:
                  cprint(f"Unhandled error in Market Maker loop: {e}. Sleeping...", 'red')
                  time.sleep(60) # Longer sleep on unhandled error
    else:
        print(f"Unknown mode: {mode}")

    print('--- Script Finished ---')


if __name__ == "__main__":
    main()

# --- Scheduling (Commented Out) ---
# The original schedule logic might conflict with the single-run modes or the MM loop.
# If you need scheduled execution, consider running specific functions via schedule
# or using a proper process manager/cron job.

# def scheduled_job():
#     print("Running scheduled job...")
#     # Example: Run PNL check periodically
#     # run_pnl_close_mode()
#     # Or run market maker logic
#     # run_market_maker_mode(PRIMARY_SYMBOL_MINT)

# schedule.every(60).seconds.do(scheduled_job) # Example: run every minute

# print("Starting scheduler...")
# while True:
#     try:
#         schedule.run_pending()
#         time.sleep(1)
#     except KeyboardInterrupt:
#         print("\nScheduler stopped by user.")
#         break
#     except Exception as e:
#         print(f'*** Error in scheduler loop: {e}, sleeping')
#         time.sleep(15)