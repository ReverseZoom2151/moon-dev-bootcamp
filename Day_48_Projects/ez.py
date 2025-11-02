'''
This bot is intended for people just getting started with algorithmic trading 
If you top blast meaning by at the point of the top of the market, you could be getting a 50% worse price. 
for example i could buy a million of a token and if i would have just waited 30 minutes i could have bought a million and a half 
But humans don't have that patience. So this bot will go ahead and buy the demand zones and other ways to buy that will essentially have them wait 
It will help humans average in, average in at good places. 

'''

import nice_funcs as n  # Import nice_funcs.py from the same directory
import time 
from termcolor import cprint # Removed colored, as only cprint is used
# import schedule # Removed schedule as it conflicts with the current bot structure

# --- Configuration ---
CONTRACT_ADDRESS = '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr'
TOTAL_USD_POSITION_SIZE = 100.0  
MAX_USD_ORDER_SIZE = 50.0     
# ORDERS_PER_OPEN = 2 # This constant is no longer used by _handle_market_buy_loop due to n.market_buy having its own retries.

SLIPPAGE_BPS = 50             

# Timeouts and Sleeps
SLEEP_BETWEEN_ORDERS_CLOSE = 600 
SLEEP_AFTER_BUY_ATTEMPT = 10    # Sleep after a market buy call (n.market_buy)
SDZ_SLEEP_AFTER_BUY_ATTEMPT = 15 # Sleep for S/D zone buy attempts
GENERAL_RETRY_SLEEP = 15
SDZ_POLL_INTERVAL = 60         

# S/D Zone Parameters
SD_DZ_DAYS_BACK = 1
SDDZ_TIMEFRAME = '15m'

# Thresholds
TARGET_FILL_RATIO = 0.97       # Target fill ratio (e.g., 97% of TOTAL_USD_POSITION_SIZE)
MIN_SIGNIFICANT_UNITS = 0.001  # Minimum token units to be considered a non-zero position
MIN_BUY_USD_THRESHOLD = 0.50   # Minimum USD value for a buy attempt (e.g., $0.50)
DUST_USD_THRESHOLD = 0.10      # USD value below which a position is considered dust/negligible for closure checks

# --- Helper Functions ---

def _get_current_position_usd(contract_address):
    """Fetches current position size in units and its USD value."""
    pos_units = n.get_position(contract_address)
    price = n.token_price(contract_address)
    if pos_units is not None and price is not None and price > 0:
        return pos_units, pos_units * price
    elif pos_units is not None and (price is None or price <=0):
         cprint(f"Have {pos_units} units of {contract_address[:6]} but cannot get valid price. Assuming 0 USD value for now.", "yellow")
         return pos_units, 0.0
    return 0.0, 0.0

def _calculate_usdc_lamports_for_buy(usd_amount):
    """Converts USD amount to USDC lamports (assuming 6 decimals for USDC)."""
    if usd_amount <= 0:
        return 0
    return int(usd_amount * (10**6))


def _handle_market_buy_loop(contract_address, target_pos_usd, max_order_usd, sleep_after_buy_attempt):
    """Handles the logic to reach a target USD position by market buying in chunks."""
    pos_units, pos_usd = _get_current_position_usd(contract_address)
    
    while pos_usd < (TARGET_FILL_RATIO * target_pos_usd):
        cprint(f"Current position: {pos_units:.4f} units (${pos_usd:.2f} USD). Target: ${target_pos_usd:.2f} USD (aiming for {TARGET_FILL_RATIO*100}% fill).", "cyan")
        size_needed_usd = target_pos_usd - pos_usd
        
        if size_needed_usd <= MIN_BUY_USD_THRESHOLD:
            cprint(f"Position very close to target (need ${size_needed_usd:.2f}, threshold ${MIN_BUY_USD_THRESHOLD:.2f}). Stopping buys.", "green")
            break

        buy_chunk_usd = min(size_needed_usd, max_order_usd)
        buy_chunk_lamports = _calculate_usdc_lamports_for_buy(buy_chunk_usd)

        if buy_chunk_lamports <= 0:
            cprint(f"Calculated buy chunk in lamports is {buy_chunk_lamports}. Skipping buy attempt.", "yellow")
            break 

        cprint(f"Need to buy ${size_needed_usd:.2f} USD more. Attempting to buy chunk of ${buy_chunk_usd:.2f} ({buy_chunk_lamports} lamports).", "blue")

        try:
            tx_id = n.market_buy(contract_address, buy_chunk_lamports, SLIPPAGE_BPS)
            if tx_id:
                cprint(f"Market buy order processed for {contract_address[:6]}, amount ~${buy_chunk_usd:.2f}. Tx: {tx_id}", "green")
            else:
                cprint(f"Market buy attempt for {contract_address[:6]} failed after retries in n.market_buy. Will re-evaluate.", "red")
            
            cprint(f"Sleeping for {sleep_after_buy_attempt}s after buy attempt...", "magenta")
            time.sleep(sleep_after_buy_attempt)

            # Re-check position after the buy attempt and sleep
            pos_units, pos_usd = _get_current_position_usd(contract_address)
            if pos_usd >= (TARGET_FILL_RATIO * target_pos_usd):
                cprint("Target position reached after buy attempt.", "green")
                break # Exit the while loop for buying

        except Exception as e:
            cprint(f"Error during market buy loop for {contract_address[:6]}: {e}. Retrying outer loop after {GENERAL_RETRY_SLEEP}s.", "red")
            time.sleep(GENERAL_RETRY_SLEEP)
        
        # Brief pause before the next main while loop iteration if not broken out
        # This is if the try block completed without breaking and pos_usd is still below target
        time.sleep(3)

    cprint(f"Market buy loop for {contract_address[:6]} finished. Final position: ${pos_usd:.2f} USD.", "green")


# --- Main Bot Logic ---
def run_bot_action(action_code):
    cprint(f"Bot starting action: {action_code}", "yellow")
    
    current_price = n.token_price(CONTRACT_ADDRESS)
    if current_price is not None:
        cprint(f"Initial price for {CONTRACT_ADDRESS[:6]}: ${current_price:.6f}", "cyan")
    else:
        cprint(f"Could not fetch initial price for {CONTRACT_ADDRESS[:6]}.", "yellow")

    if action_code == 0:  # Close full position
        cprint(f"Action 0: Closing full position for {CONTRACT_ADDRESS[:6]}.", "blue")
        pos_units, pos_usd = _get_current_position_usd(CONTRACT_ADDRESS)
        cprint(f"Current position: {pos_units:.4f} units (${pos_usd:.2f} USD).", "cyan")

        if pos_units > MIN_SIGNIFICANT_UNITS: 
            n.chunk_kill(CONTRACT_ADDRESS, MAX_USD_ORDER_SIZE, SLEEP_BETWEEN_ORDERS_CLOSE, SLIPPAGE_BPS)
            
            time.sleep(15) 
            pos_units_after, pos_usd_after = _get_current_position_usd(CONTRACT_ADDRESS)
            if pos_units_after < MIN_SIGNIFICANT_UNITS:
                cprint(f"Position for {CONTRACT_ADDRESS[:6]} successfully closed. Final units: {pos_units_after:.6f}", "green")
            else:
                cprint(f"Position for {CONTRACT_ADDRESS[:6]} may not be fully closed. Remaining units: {pos_units_after:.6f} (${pos_usd_after:.2f} USD)", "yellow")
        else:
            cprint(f"No significant position ({MIN_SIGNIFICANT_UNITS} units) to close for {CONTRACT_ADDRESS[:6]}.", "green")
        cprint("Action 0 finished.", "blue")

    elif action_code == 1:  # Market buy to target USD
        cprint(f"Action 1: Market buying {CONTRACT_ADDRESS[:6]} to target ${TOTAL_USD_POSITION_SIZE:.2f} USD.", "blue")
        _handle_market_buy_loop(CONTRACT_ADDRESS, TOTAL_USD_POSITION_SIZE, MAX_USD_ORDER_SIZE, SLEEP_AFTER_BUY_ATTEMPT)
        cprint("Action 1 finished.", "blue")

    elif action_code == 2:  # Demand zone buy
        cprint(f"Action 2: Demand zone buying for {CONTRACT_ADDRESS[:6]}.", "blue")
        pos_units, pos_usd = _get_current_position_usd(CONTRACT_ADDRESS)

        if pos_usd >= (TARGET_FILL_RATIO * TOTAL_USD_POSITION_SIZE):
            cprint(f"Position for {CONTRACT_ADDRESS[:6]} already {TARGET_FILL_RATIO*100}% filled (${pos_usd:.2f} / ${TOTAL_USD_POSITION_SIZE:.2f} USD). Skipping S/D buy.", "green")
            return

        while pos_usd < (TARGET_FILL_RATIO * TOTAL_USD_POSITION_SIZE):
            current_price = n.token_price(CONTRACT_ADDRESS)
            sd_df = n.supply_demand_zones(CONTRACT_ADDRESS, SD_DZ_DAYS_BACK, SDDZ_TIMEFRAME)

            if current_price is None:
                cprint(f"Cannot get current price for {CONTRACT_ADDRESS[:6]}. Waiting {SDZ_POLL_INTERVAL}s.", "yellow")
                time.sleep(SDZ_POLL_INTERVAL)
                continue
            
            if sd_df.empty or not all(col in sd_df.columns for col in ['dz_low', 'dz_high']):
                cprint(f"Could not get valid S/D zones for {CONTRACT_ADDRESS[:6]}. Waiting {SDZ_POLL_INTERVAL}s.", "yellow")
                time.sleep(SDZ_POLL_INTERVAL)
                continue

            dz_low = sd_df['dz_low'].iloc[0]
            dz_high = sd_df['dz_high'].iloc[0]
            
            cprint(f"S/D Check: Price ${current_price:.6f}. DZ: ${dz_low:.6f} - ${dz_high:.6f}", "magenta")
            
            if dz_low <= current_price <= dz_high:
                cprint(f"🎯 Price ${current_price:.6f} IS in demand zone (${dz_low:.6f} - ${dz_high:.6f}). Initiating buys.", "green")
                _handle_market_buy_loop(CONTRACT_ADDRESS, TOTAL_USD_POSITION_SIZE, MAX_USD_ORDER_SIZE, SDZ_SLEEP_AFTER_BUY_ATTEMPT)
                
                _, pos_usd = _get_current_position_usd(CONTRACT_ADDRESS) 
                if pos_usd >= (TARGET_FILL_RATIO * TOTAL_USD_POSITION_SIZE):
                    cprint(f"Target position reached during S/D buy for {CONTRACT_ADDRESS[:6]}.", "green")
                    break 
                else:
                    cprint(f"Target not fully reached after S/D buy attempt. Current pos: ${pos_usd:.2f}. Re-evaluating S/D zone.", "cyan")
            else:
                cprint(f"⏳ Price ${current_price:.6f} is NOT in demand zone. Waiting {SDZ_POLL_INTERVAL}s...", "yellow")
                time.sleep(SDZ_POLL_INTERVAL)
            
            _, pos_usd = _get_current_position_usd(CONTRACT_ADDRESS)
            if pos_usd >= (TARGET_FILL_RATIO * TOTAL_USD_POSITION_SIZE):
                 cprint(f"Target position for {CONTRACT_ADDRESS[:6]} seems filled. Exiting S/D buy loop.", "green")
                 break
        cprint(f"Action 2 (S/D Buy) for {CONTRACT_ADDRESS[:6]} finished. Final pos USD: ${pos_usd:.2f}", "blue")


    elif action_code == 3:  # Supply zone close
        cprint(f"Action 3: Supply zone closing for {CONTRACT_ADDRESS[:6]}.", "blue")
        
        while True: 
            pos_units, pos_usd = _get_current_position_usd(CONTRACT_ADDRESS)
            if pos_units < MIN_SIGNIFICANT_UNITS: 
                cprint(f"Position for {CONTRACT_ADDRESS[:6]} is already closed (Units: {pos_units}). Exiting S/Z close.", "green")
                break

            current_price = n.token_price(CONTRACT_ADDRESS)
            sd_df = n.supply_demand_zones(CONTRACT_ADDRESS, SD_DZ_DAYS_BACK, SDDZ_TIMEFRAME)

            if current_price is None:
                cprint(f"Cannot get current price for {CONTRACT_ADDRESS[:6]}. Waiting {SDZ_POLL_INTERVAL}s.", "yellow")
                time.sleep(SDZ_POLL_INTERVAL)
                continue
            
            if sd_df.empty or not all(col in sd_df.columns for col in ['sz_low', 'sz_high']):
                cprint(f"Could not get valid S/D zones for {CONTRACT_ADDRESS[:6]}. Waiting {SDZ_POLL_INTERVAL}s.", "yellow")
                time.sleep(SDZ_POLL_INTERVAL)
                continue

            sz_low = sd_df['sz_low'].iloc[0]
            sz_high = sd_df['sz_high'].iloc[0]
            
            cprint(f"S/D Check: Price ${current_price:.6f}. SZ: ${sz_low:.6f} - ${sz_high:.6f}", "magenta")

            if sz_low <= current_price <= sz_high:
                cprint(f"🎯 Price ${current_price:.6f} IS in supply zone (${sz_low:.6f} - ${sz_high:.6f}). Initiating close.", "red")
                n.chunk_kill(CONTRACT_ADDRESS, MAX_USD_ORDER_SIZE, SDZ_SLEEP_AFTER_BUY_ATTEMPT, SLIPPAGE_BPS) # Using SDZ_SLEEP_AFTER_BUY_ATTEMPT for consistency if chunk_kill uses it internally between its own orders
                
                time.sleep(15) 
                pos_units_after_kill, _ = _get_current_position_usd(CONTRACT_ADDRESS)
                if pos_units_after_kill < MIN_SIGNIFICANT_UNITS:
                    cprint(f"Position for {CONTRACT_ADDRESS[:6]} closed via S/Z logic.", "green")
                    break 
                else:
                    cprint(f"Position for {CONTRACT_ADDRESS[:6]} not fully closed. Remaining: {pos_units_after_kill}. Will re-evaluate S/Z.", "yellow")
            else:
                cprint(f"⏳ Price ${current_price:.6f} is NOT in supply zone. Waiting {SDZ_POLL_INTERVAL}s...", "yellow")
                time.sleep(SDZ_POLL_INTERVAL)
        cprint("Action 3 finished.", "blue")

    else:
        cprint(f"Invalid action code: {action_code}. Please choose 0, 1, 2, or 3.", "red")

if __name__ == "__main__":
    chosen_action = -1
    while chosen_action not in [0, 1, 2, 3]:
        try:
            action_input = input(
                "Choose action:\n"
                "  0: Close full position\n"
                "  1: Market buy to target USD size\n"
                "  2: Buy when price enters Demand Zone (DZ)\n"
                "  3: Close full position when price enters Supply Zone (SZ)\n"
                "Enter action (0-3): "
            )
            chosen_action = int(action_input)
            if chosen_action not in [0, 1, 2, 3]:
                cprint("Invalid input. Please enter a number between 0 and 3.", "red")
        except ValueError:
            cprint("Invalid input. Please enter a number.", "red")

    run_bot_action(chosen_action)

    cprint("Bot run complete.", "yellow")

# Removed conflicting schedule logic:
# schedule.every(30).seconds.do(bot)
# while True:
#     try:
#         schedule.run_pending()
#         time.sleep(3)
#     except Exception as e:
#         print(f'error: {e}')
#         time.sleep(3)