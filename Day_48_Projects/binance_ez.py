'''
Binance Trading Bot - Easy Entry and Averaging System
This bot is designed for algorithmic trading on Binance with smart entry strategies.
It helps traders average in at better prices using demand zones and technical analysis.

Disclaimer: This is not financial advice and there is no guarantee of any kind. Use at your own risk.
'''

import binance_nice_funcs as n  # Import binance_nice_funcs.py from the same directory
import time 
from termcolor import cprint

# --- Configuration ---
DEFAULT_SYMBOL = 'BTCUSDT'  # Default trading symbol
TOTAL_USD_POSITION_SIZE = 100.0  
MAX_USD_ORDER_SIZE = 50.0     

SLIPPAGE_BPS = 50  # 0.5% slippage in basis points

# Timeouts and Sleeps
SLEEP_BETWEEN_ORDERS_CLOSE = 600  # 10 minutes between closing orders
SLEEP_AFTER_BUY_ATTEMPT = 10    # Sleep after a market buy call
SDZ_SLEEP_AFTER_BUY_ATTEMPT = 15 # Sleep for Supply/Demand zone buy attempts
GENERAL_RETRY_SLEEP = 15
SDZ_POLL_INTERVAL = 60         # 1 minute polling for S/D zones

# S/D Zone Parameters
SD_DZ_DAYS_BACK = 1
SDDZ_TIMEFRAME = '15m'

# Thresholds
TARGET_FILL_RATIO = 0.97       # Target fill ratio (97% of total position size)
MIN_SIGNIFICANT_UNITS = 0.001  # Minimum token units to be considered a non-zero position
MIN_BUY_USD_THRESHOLD = 10.0   # Minimum USD value for a buy attempt (Binance minimum)
DUST_USD_THRESHOLD = 5.0       # USD value below which a position is considered dust

# --- Helper Functions ---

def _get_current_position_usd(symbol):
    """Fetches current position size in units and its USD value."""
    pos_units = n.get_position(symbol)
    price = n.token_price(symbol)
    if pos_units is not None and price is not None and price > 0:
        return pos_units, pos_units * price
    elif pos_units is not None and (price is None or price <= 0):
        cprint(f"Have {pos_units} units of {symbol} but cannot get valid price. Assuming 0 USD value for now.", "yellow")
        return pos_units, 0.0
    else:
        return 0.0, 0.0

def _calculate_usd_for_buy(usd_amount):
    """Validates and returns USD amount for buy orders."""
    if usd_amount < MIN_BUY_USD_THRESHOLD:
        cprint(f"Buy amount ${usd_amount:.2f} is below minimum threshold ${MIN_BUY_USD_THRESHOLD:.2f}", "yellow")
        return None
    return usd_amount

def _handle_market_buy_loop(symbol, target_pos_usd, max_order_usd, sleep_after_buy_attempt):
    """Handles the logic to reach a target USD position by market buying in chunks."""
    cprint(f"Starting market buy loop for {symbol}. Target: ${target_pos_usd:.2f}, Max order: ${max_order_usd:.2f}", "cyan")
    
    max_attempts = 10  # Prevent infinite loops
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        # Get current position
        pos_units, pos_value_usd = _get_current_position_usd(symbol)
        
        if pos_value_usd >= target_pos_usd * TARGET_FILL_RATIO:
            cprint(f"Target reached! Current position: ${pos_value_usd:.2f} (Target: ${target_pos_usd:.2f})", "green")
            break
        
        # Calculate how much more we need to buy
        remaining_usd = target_pos_usd - pos_value_usd
        buy_amount_usd = min(remaining_usd, max_order_usd)
        
        # Validate buy amount
        validated_amount = _calculate_usd_for_buy(buy_amount_usd)
        if validated_amount is None:
            cprint(f"Buy amount too small, stopping. Remaining: ${remaining_usd:.2f}", "yellow")
            break
        
        cprint(f"Attempt {attempt}: Buying ${validated_amount:.2f} worth of {symbol}", "cyan")
        
        # Execute buy order
        result = n.market_buy(symbol, quote_quantity=validated_amount)
        
        if result:
            cprint(f"Buy order successful: {result.get('orderId', 'N/A')}", "green")
        else:
            cprint(f"Buy order failed on attempt {attempt}", "red")
        
        # Sleep before next attempt
        if attempt < max_attempts:
            cprint(f"Sleeping {sleep_after_buy_attempt} seconds before next attempt...", "cyan")
            time.sleep(sleep_after_buy_attempt)
    
    # Final position check
    final_units, final_value = _get_current_position_usd(symbol)
    cprint(f"Market buy loop completed. Final position: {final_units:.6f} units (${final_value:.2f})", "green")
    
    return final_units, final_value

# --- Main Bot Logic ---

def run_bot_action(action_code, symbol=None):
    """Main function that executes different bot actions based on action code."""
    
    if symbol is None:
        symbol = DEFAULT_SYMBOL
    
    cprint(f"=== Starting Binance EZ Bot Action {action_code} for {symbol} ===", "magenta")
    
    # Validate symbol exists and is tradeable
    symbol_info = n.get_symbol_info(symbol)
    if not symbol_info:
        cprint(f"Error: Could not get information for symbol {symbol}", "red")
        return
    
    cprint(f"Symbol: {symbol}, Price: ${symbol_info['price']:.6f}, 24h Volume: ${symbol_info['quote_volume']:,.0f}", "cyan")
    
    if action_code == 0:
        # Close position
        cprint("=== CLOSE POSITION MODE ===", "red")
        pos_units, pos_value = _get_current_position_usd(symbol)
        
        if pos_units <= MIN_SIGNIFICANT_UNITS:
            cprint(f"No significant position to close for {symbol} (Position: {pos_units:.6f})", "yellow")
            return
        
        if pos_value < DUST_USD_THRESHOLD:
            cprint(f"Position value ${pos_value:.2f} is below dust threshold ${DUST_USD_THRESHOLD:.2f}", "yellow")
            cprint("Consider closing manually or adjusting dust threshold", "yellow")
            return
        
        cprint(f"Closing position: {pos_units:.6f} units (${pos_value:.2f})", "red")
        result = n.pnl_close(symbol)
        
        if result:
            cprint("Position closed successfully!", "green")
        else:
            cprint("Failed to close position", "red")
    
    elif action_code == 1:
        # Market buy to target position size
        cprint("=== MARKET BUY MODE ===", "green")
        
        # Check account balance
        balance = n.get_account_balance("USDT")
        if balance < TOTAL_USD_POSITION_SIZE:
            cprint(f"Insufficient balance. Available: ${balance:.2f}, Required: ${TOTAL_USD_POSITION_SIZE:.2f}", "red")
            return
        
        pos_units, pos_value = _get_current_position_usd(symbol)
        cprint(f"Current position: {pos_units:.6f} units (${pos_value:.2f})", "cyan")
        
        if pos_value >= TOTAL_USD_POSITION_SIZE * TARGET_FILL_RATIO:
            cprint(f"Already at target position size: ${pos_value:.2f} >= ${TOTAL_USD_POSITION_SIZE * TARGET_FILL_RATIO:.2f}", "yellow")
            return
        
        _handle_market_buy_loop(symbol, TOTAL_USD_POSITION_SIZE, MAX_USD_ORDER_SIZE, SLEEP_AFTER_BUY_ATTEMPT)
    
    elif action_code == 2:
        # SMA-based trend following buy
        cprint("=== SMA TREND FOLLOWING MODE ===", "blue")
        
        # Check trend
        trend_data = n.check_trend(symbol, periods=20)
        if not trend_data:
            cprint(f"Could not analyze trend for {symbol}", "red")
            return
        
        cprint(f"Price: ${trend_data['price']:.6f}, SMA: ${trend_data['sma']:.6f}", "cyan")
        cprint(f"Trend strength: {trend_data['trend_strength']:.2f}%", "cyan")
        
        if trend_data['is_uptrend']:
            cprint("✅ Uptrend detected! Executing buy...", "green")
            
            # Check current position
            pos_units, pos_value = _get_current_position_usd(symbol)
            
            if pos_value >= TOTAL_USD_POSITION_SIZE * TARGET_FILL_RATIO:
                cprint(f"Already at target position size", "yellow")
                return
            
            # Buy additional position
            buy_amount = min(MAX_USD_ORDER_SIZE, TOTAL_USD_POSITION_SIZE - pos_value)
            validated_amount = _calculate_usd_for_buy(buy_amount)
            
            if validated_amount:
                result = n.market_buy(symbol, quote_quantity=validated_amount)
                if result:
                    cprint(f"Trend-following buy executed: ${validated_amount:.2f}", "green")
                else:
                    cprint("Trend-following buy failed", "red")
        else:
            cprint("❌ Downtrend detected. No buy signal.", "red")
    
    elif action_code == 3:
        # Supply/Demand zone buying
        cprint("=== SUPPLY/DEMAND ZONE MODE ===", "magenta")
        
        # Analyze supply/demand zones
        sd_data = n.supply_demand_zones(symbol, SD_DZ_DAYS_BACK, SDDZ_TIMEFRAME)
        if not sd_data:
            cprint(f"Could not analyze supply/demand zones for {symbol}", "red")
            return
        
        cprint(f"Current Price: ${sd_data['current_price']:.6f}", "cyan")
        cprint(f"Support Level: ${sd_data['support']:.6f} (Distance: {sd_data['support_distance']:.2f}%)", "green")
        cprint(f"Resistance Level: ${sd_data['resistance']:.6f} (Distance: {sd_data['resistance_distance']:.2f}%)", "red")
        
        if sd_data['near_support']:
            cprint("🎯 Price is near support/demand zone! Executing buy...", "green")
            
            # Check current position
            pos_units, pos_value = _get_current_position_usd(symbol)
            
            if pos_value >= TOTAL_USD_POSITION_SIZE * TARGET_FILL_RATIO:
                cprint(f"Already at target position size", "yellow")
                return
            
            # Buy at demand zone
            buy_amount = min(MAX_USD_ORDER_SIZE, TOTAL_USD_POSITION_SIZE - pos_value)
            validated_amount = _calculate_usd_for_buy(buy_amount)
            
            if validated_amount:
                result = n.market_buy(symbol, quote_quantity=validated_amount)
                if result:
                    cprint(f"Demand zone buy executed: ${validated_amount:.2f}", "green")
                else:
                    cprint("Demand zone buy failed", "red")
                
                # Sleep after demand zone buy
                time.sleep(SDZ_SLEEP_AFTER_BUY_ATTEMPT)
        else:
            cprint("⏳ Not near demand zone. Current conditions:", "yellow")
            cprint(f"  Support distance: {sd_data['support_distance']:.2f}%", "yellow")
            cprint(f"  Resistance distance: {sd_data['resistance_distance']:.2f}%", "yellow")
            cprint(f"  Waiting for better entry point...", "yellow")
    
    else:
        cprint(f"Invalid action code: {action_code}", "red")
        return
    
    # Final status
    final_units, final_value = _get_current_position_usd(symbol)
    cprint(f"=== Action {action_code} completed ===", "magenta")
    cprint(f"Final position: {final_units:.6f} units (${final_value:.2f})", "cyan")

if __name__ == "__main__":
    chosen_action = -1
    chosen_symbol = None
    
    # Display menu
    cprint("🚀 Binance EZ Trading Bot 🚀", "magenta")
    cprint("=" * 40, "cyan")
    cprint("Available actions:", "white")
    cprint("0 - Close position", "red")
    cprint("1 - Market buy to target position", "green")
    cprint("2 - SMA trend following buy", "blue")
    cprint("3 - Supply/Demand zone buying", "magenta")
    cprint("=" * 40, "cyan")
    
    while chosen_action not in [0, 1, 2, 3]:
        try:
            chosen_action = int(input("Choose action (0-3): "))
            if chosen_action not in [0, 1, 2, 3]:
                cprint("Please enter a number between 0 and 3", "red")
        except ValueError:
            cprint("Please enter a valid number", "red")
    
    # Get symbol input
    symbol_input = input(f"Enter trading symbol (default: {DEFAULT_SYMBOL}): ").strip().upper()
    if symbol_input:
        chosen_symbol = symbol_input
        # Ensure it ends with USDT if not specified
        if not chosen_symbol.endswith('USDT') and not chosen_symbol.endswith('BTC') and not chosen_symbol.endswith('ETH'):
            chosen_symbol += 'USDT'
    else:
        chosen_symbol = DEFAULT_SYMBOL
    
    # Confirm action
    action_names = {
        0: "Close position",
        1: "Market buy to target position", 
        2: "SMA trend following buy",
        3: "Supply/Demand zone buying"
    }
    
    cprint(f"\n🎯 Executing: {action_names[chosen_action]} for {chosen_symbol}", "cyan")
    confirm = input("Continue? (y/N): ").lower()
    
    if confirm == 'y':
        try:
            run_bot_action(chosen_action, chosen_symbol)
        except KeyboardInterrupt:
            cprint("\n❌ Bot stopped by user", "yellow")
        except Exception as e:
            cprint(f"\n💥 Error: {e}", "red")
            import traceback
            traceback.print_exc()
    else:
        cprint("❌ Action cancelled", "yellow")
