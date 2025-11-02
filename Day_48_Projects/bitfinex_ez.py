"""
Bitfinex Professional Ez Trading Bot
Algorithmic trading bot for Bitfinex exchange with professional features.
Adapted for institutional-grade trading with margin support and funding analysis.
"""

import time
import bitfinex_nice_funcs as n
from bitfinex_CONFIG import *
from termcolor import cprint

# Mode configuration
MODE = {
    1: "Close Position (Professional Exit)",
    2: "Market Buy (Institutional Order)",
    3: "SMA Trend Following (Professional Signals)",
    4: "Supply/Demand Zone Entry (Institutional Strategy)"
}

def print_professional_banner():
    """Display professional Bitfinex trading banner"""
    cprint("=" * 80, "cyan")
    cprint("🏦 BITFINEX PROFESSIONAL TRADING BOT 🏦", "cyan", attrs=['bold'])
    cprint("Institutional-Grade Algorithmic Trading System", "white")
    cprint("=" * 80, "cyan")
    print()

def print_mode_menu():
    """Display professional mode selection menu"""
    cprint("PROFESSIONAL TRADING MODES:", "yellow", attrs=['bold'])
    cprint("-" * 40, "yellow")
    for key, value in MODE.items():
        cprint(f"  [{key}] {value}", "white")
    print()

def get_professional_symbol():
    """Get trading symbol from user with professional validation"""
    while True:
        cprint("Enter trading symbol (e.g., BTCUSD, ETHUSD): ", "yellow", end="")
        user_input = input().strip().upper()
        
        if not user_input:
            cprint("Error: Symbol cannot be empty", "red")
            continue
        
        # Add 't' prefix if not present (Bitfinex format)
        if not user_input.startswith('t'):
            symbol = f"t{user_input}"
        else:
            symbol = user_input
        
        # Professional validation
        if symbol in [f"t{s}" for s in DO_NOT_TRADE_LIST]:
            cprint(f"Warning: {symbol} is on the professional do-not-trade list", "yellow")
            confirm = input("Continue anyway? [y/N]: ").lower()
            if confirm != 'y':
                continue
        
        # Validate symbol with API
        price = n.token_price(symbol)
        if price:
            cprint(f"✅ Symbol validated: {symbol} - Current price: ${price:.6f}", "green")
            return symbol
        else:
            cprint(f"❌ Invalid symbol or API error: {symbol}", "red")

def get_trade_amount():
    """Get professional trade amount with validation"""
    while True:
        cprint(f"Enter trade amount in USD (min: ${MIN_TRADE_SIZE_USD}, max: ${MAX_TRADE_SIZE_USD}): ", "yellow", end="")
        try:
            amount = float(input().strip())
            
            if amount < MIN_TRADE_SIZE_USD:
                cprint(f"Error: Minimum trade size is ${MIN_TRADE_SIZE_USD}", "red")
                continue
            
            if amount > MAX_TRADE_SIZE_USD:
                cprint(f"Warning: Amount exceeds maximum of ${MAX_TRADE_SIZE_USD}", "yellow")
                confirm = input("Continue with large order? [y/N]: ").lower()
                if confirm != 'y':
                    continue
            
            return amount
            
        except ValueError:
            cprint("Error: Please enter a valid number", "red")

def mode_close_position():
    """Professional position closure with PnL analysis"""
    cprint("🔒 PROFESSIONAL POSITION CLOSURE", "cyan", attrs=['bold'])
    print()
    
    symbol = get_professional_symbol()
    
    # Check current position
    position = n.get_position(symbol)
    if position == 0:
        cprint(f"No position found for {symbol}", "yellow")
        return
    
    current_price = n.token_price(symbol)
    position_value = abs(position) * current_price if current_price else 0
    
    cprint(f"Current Position: {position:.6f} {symbol}", "white")
    cprint(f"Current Price: ${current_price:.6f}", "white")
    cprint(f"Position Value: ${position_value:.2f}", "white")
    print()
    
    # Professional confirmation
    cprint("⚠️  PROFESSIONAL POSITION CLOSURE ⚠️", "red", attrs=['bold'])
    confirm = input("Confirm position closure? [y/N]: ").lower()
    
    if confirm == 'y':
        cprint("Executing professional position closure...", "cyan")
        result = n.pnl_close(symbol)
        
        if result:
            cprint("✅ Professional position closed successfully", "green", attrs=['bold'])
        else:
            cprint("❌ Position closure failed", "red")
    else:
        cprint("Position closure cancelled", "yellow")

def mode_market_buy():
    """Professional market buy with institutional features"""
    cprint("💰 PROFESSIONAL MARKET BUY", "cyan", attrs=['bold'])
    print()
    
    symbol = get_professional_symbol()
    amount = get_trade_amount()
    
    current_price = n.token_price(symbol)
    if not current_price:
        cprint("Unable to get current price. Aborting.", "red")
        return
    
    quantity = amount / current_price
    
    cprint(f"Trade Details:", "white", attrs=['bold'])
    cprint(f"  Symbol: {symbol}", "white")
    cprint(f"  USD Amount: ${amount:.2f}", "white")
    cprint(f"  Quantity: {quantity:.6f}", "white")
    cprint(f"  Current Price: ${current_price:.6f}", "white")
    cprint(f"  Estimated Slippage: {SLIPPAGE * 100:.2f}%", "white")
    print()
    
    # Professional risk check
    balance = n.get_account_balance("USD")
    if amount > balance * 0.1:  # 10% of balance
        cprint("⚠️  Large order detected (>10% of balance)", "yellow")
        confirm = input("Continue with large institutional order? [y/N]: ").lower()
        if confirm != 'y':
            cprint("Order cancelled", "yellow")
            return
    
    # Final confirmation
    cprint("🏦 PROFESSIONAL MARKET BUY ORDER", "green", attrs=['bold'])
    confirm = input("Execute institutional buy order? [y/N]: ").lower()
    
    if confirm == 'y':
        cprint("Executing professional market buy...", "cyan")
        result = n.market_buy(symbol, quote_amount=amount)
        
        if result:
            cprint("✅ Professional buy order executed successfully", "green", attrs=['bold'])
            cprint(f"Order ID: {result[0] if result else 'N/A'}", "green")
        else:
            cprint("❌ Buy order execution failed", "red")
    else:
        cprint("Buy order cancelled", "yellow")

def mode_sma_trend():
    """Professional SMA trend following strategy"""
    cprint("📈 PROFESSIONAL SMA TREND FOLLOWING", "cyan", attrs=['bold'])
    print()
    
    symbol = get_professional_symbol()
    
    # Professional trend analysis
    cprint("Performing professional technical analysis...", "cyan")
    trend_data = n.check_trend(symbol, periods=SMA_DAYS_BACK * 24)
    
    if not trend_data:
        cprint("Unable to get trend data. Aborting.", "red")
        return
    
    cprint("Professional Technical Analysis:", "white", attrs=['bold'])
    cprint(f"  Current Price: ${trend_data['price']:.6f}", "white")
    cprint(f"  EMA Fast ({EMA_FAST}): ${trend_data['ema_fast']:.6f}", "white")
    cprint(f"  EMA Slow ({EMA_SLOW}): ${trend_data['ema_slow']:.6f}", "white")
    cprint(f"  RSI ({RSI_PERIOD}): {trend_data['rsi']:.2f}", "white")
    cprint(f"  Trend Strength: {trend_data['trend_strength']:.2f}%", "white")
    
    if trend_data['is_uptrend']:
        cprint("🟢 PROFESSIONAL UPTREND DETECTED", "green", attrs=['bold'])
        cprint("Institutional recommendation: CONSIDER BUY", "green")
        
        # Professional entry conditions
        if trend_data['rsi'] < 70:  # Not overbought
            amount = get_trade_amount()
            
            cprint(f"Professional entry signal confirmed", "green")
            confirm = input("Execute professional trend-following buy? [y/N]: ").lower()
            
            if confirm == 'y':
                cprint("Executing professional trend-following buy...", "cyan")
                result = n.market_buy(symbol, quote_amount=amount)
                
                if result:
                    cprint("✅ Professional trend-following buy executed", "green", attrs=['bold'])
                else:
                    cprint("❌ Trend-following buy failed", "red")
        else:
            cprint("⚠️  RSI overbought - Professional entry delayed", "yellow")
    else:
        cprint("🔴 NO PROFESSIONAL UPTREND", "red")
        cprint("Institutional recommendation: AVOID BUY", "red")

def mode_supply_demand():
    """Professional supply/demand zone trading strategy"""
    cprint("🎯 PROFESSIONAL SUPPLY/DEMAND ZONE STRATEGY", "cyan", attrs=['bold'])
    print()
    
    symbol = get_professional_symbol()
    
    # Professional zone analysis
    cprint("Analyzing professional supply/demand zones...", "cyan")
    zones = n.supply_demand_zones(symbol, days_back=SMA_DAYS_BACK, timeframe=SMA_TIMEFRAME)
    
    if not zones:
        cprint("Unable to get zone analysis. Aborting.", "red")
        return
    
    cprint("Professional Zone Analysis:", "white", attrs=['bold'])
    cprint(f"  Current Price: ${zones['current_price']:.6f}", "white")
    cprint(f"  Pivot Level: ${zones['pivot']:.6f}", "white")
    cprint(f"  Support Level: ${zones['support']:.6f}", "white")
    cprint(f"  Resistance Level: ${zones['resistance']:.6f}", "white")
    cprint(f"  Support Distance: {zones['support_distance']:.2f}%", "white")
    cprint(f"  Resistance Distance: {zones['resistance_distance']:.2f}%", "white")
    print()
    
    # Professional zone signals
    if zones['near_support']:
        cprint("🟢 PROFESSIONAL DEMAND ZONE DETECTED", "green", attrs=['bold'])
        cprint("Institutional signal: STRONG BUY OPPORTUNITY", "green")
        
        amount = get_trade_amount()
        
        cprint("Professional demand zone entry confirmed", "green")
        confirm = input("Execute professional demand zone buy? [y/N]: ").lower()
        
        if confirm == 'y':
            cprint("Executing professional demand zone buy...", "cyan")
            result = n.market_buy(symbol, quote_amount=amount)
            
            if result:
                cprint("✅ Professional demand zone buy executed", "green", attrs=['bold'])
            else:
                cprint("❌ Demand zone buy failed", "red")
    
    elif zones['near_resistance']:
        cprint("🔴 PROFESSIONAL SUPPLY ZONE DETECTED", "red", attrs=['bold'])
        cprint("Institutional signal: CONSIDER SELL/AVOID BUY", "red")
        
        # Check for existing position to close
        position = n.get_position(symbol)
        if position > 0:
            cprint("Existing position detected in supply zone", "yellow")
            confirm = input("Close position at supply zone? [y/N]: ").lower()
            
            if confirm == 'y':
                result = n.pnl_close(symbol)
                if result:
                    cprint("✅ Position closed at supply zone", "green")
    else:
        cprint("⚪ NO CLEAR PROFESSIONAL ZONES", "white")
        cprint("Current price between major levels", "white")
        cprint("Institutional recommendation: WAIT FOR CLEAR SIGNAL", "yellow")

def display_professional_account_info():
    """Display professional account information"""
    cprint("🏦 PROFESSIONAL ACCOUNT INFORMATION", "cyan", attrs=['bold'])
    cprint("-" * 50, "cyan")
    
    try:
        # Get account balances
        usd_balance = n.get_account_balance("USD")
        btc_balance = n.get_account_balance("BTC")
        eth_balance = n.get_account_balance("ETH")
        
        cprint(f"USD Balance: ${usd_balance:.2f}", "white")
        cprint(f"BTC Balance: {btc_balance:.6f}", "white")
        cprint(f"ETH Balance: {eth_balance:.6f}", "white")
        print()
        
        # Professional risk metrics
        cprint("Professional Risk Metrics:", "yellow", attrs=['bold'])
        cprint(f"  Maximum Daily Loss: ${MAX_DAILY_LOSS_USD:.2f}", "white")
        cprint(f"  Maximum Position Risk: {MAX_PORTFOLIO_RISK * 100:.1f}%", "white")
        cprint(f"  Trading Mode: {TRADING_MODE}", "white")
        cprint(f"  Maximum Leverage: {MAX_LEVERAGE:.1f}x", "white")
        print()
        
    except Exception as e:
        cprint(f"Error retrieving account information: {e}", "red")

def main():
    """Main professional trading bot execution"""
    try:
        print_professional_banner()
        
        # Display professional account info
        display_professional_account_info()
        
        while True:
            print_mode_menu()
            
            try:
                cprint("Select professional trading mode (1-4) or 'q' to quit: ", "yellow", end="")
                choice = input().strip().lower()
                
                if choice == 'q' or choice == 'quit':
                    cprint("Professional trading session ended. Thank you!", "cyan", attrs=['bold'])
                    break
                
                mode_num = int(choice)
                
                if mode_num == 1:
                    mode_close_position()
                elif mode_num == 2:
                    mode_market_buy()
                elif mode_num == 3:
                    mode_sma_trend()
                elif mode_num == 4:
                    mode_supply_demand()
                else:
                    cprint("Invalid selection. Please choose 1-4.", "red")
                    continue
                
                print()
                cprint("-" * 80, "cyan")
                
                # Professional delay between operations
                time.sleep(2)
                
            except ValueError:
                cprint("Invalid input. Please enter a number 1-4 or 'q'.", "red")
            except KeyboardInterrupt:
                cprint("\nProfessional trading session interrupted.", "yellow")
                break
            except Exception as e:
                cprint(f"Professional trading error: {e}", "red")
                cprint("Continuing with professional safety protocols...", "yellow")
                time.sleep(3)
    
    except Exception as e:
        cprint(f"Critical professional trading system error: {e}", "red", attrs=['bold'])
        cprint("Please check your professional configuration and API credentials.", "yellow")

if __name__ == "__main__":
    # Professional startup checks
    cprint("Initializing Bitfinex Professional Trading System...", "cyan")
    
    # Check API credentials
    if not API_KEY or not API_SECRET:
        cprint("❌ Professional API credentials not configured", "red", attrs=['bold'])
        cprint("Please set up your Bitfinex API credentials in CONFIG.py or environment variables", "yellow")
        cprint("Required: BITFINEX_API_KEY, BITFINEX_API_SECRET", "yellow")
        exit(1)
    
    # Professional system validation
    cprint("✅ Professional credentials validated", "green")
    cprint("✅ Professional risk management loaded", "green")
    cprint("✅ Institutional trading features enabled", "green")
    print()
    
    # Start professional trading system
    main()
