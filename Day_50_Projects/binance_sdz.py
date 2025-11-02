"""
Binance Supply & Demand Zone Trading Bot

Professional Supply & Demand Zone strategy adapted for Binance exchange.
Identifies key S&D zones and executes trades based on price action around these levels.
Includes trend analysis, position management, and risk controls.
"""

import time
import schedule
import pandas as pd
import sys
import binance_nice_funcs2 as n
from binance_CONFIG import get_binance_sdz_config, validate_config
from termcolor import cprint

# Load configuration
CONFIG = get_binance_sdz_config()

# Validate configuration on startup
validation_errors = validate_config()
if validation_errors:
    cprint("❌ Configuration Errors:", "red", attrs=['bold'])
    for error in validation_errors:
        cprint(f"  - {error}", "red")
    cprint("Please fix configuration errors before running the bot.", "yellow")
    sys.exit(1)

# Define Trend constants
TREND_UP = 'up'
TREND_DOWN = 'down'
TREND_SIDEWAYS = 'sideways'

def print_banner():
    """Print startup banner"""
    cprint("="*80, "cyan")
    cprint("🏦 BINANCE SUPPLY & DEMAND ZONE TRADING BOT", "cyan", attrs=['bold'])
    cprint("Professional SDZ Strategy with Trend Analysis", "white")
    cprint("="*80, "cyan")
    print()

def aggregate_to_timeframe(df, target_timeframe='2h'):
    """Aggregate data to specified timeframe"""
    if 'Datetime (UTC)' in df.columns:
        df = df.set_index('Datetime (UTC)')
    
    # Resample to target timeframe
    df_resampled = df.resample(target_timeframe, closed='left', label='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return df_resampled.reset_index()

def check_trend(symbol):
    """Check trend using SMA analysis with buffer zone"""
    try:
        # Get historical data
        timeframe = CONFIG['SMA_TIMEFRAME']
        days_back = CONFIG['SMA_DAYS_BACK']
        limit = days_back * 24 if timeframe == '1h' else days_back * 24 * 4  # Adjust for timeframe
        
        df = n.get_historical_data(CONFIG, symbol, timeframe, min(limit, 500))
        
        if df is None or len(df) < CONFIG['SMA_BARS']:
            cprint(f"❌ Insufficient data for trend analysis: {symbol}", "red")
            return TREND_SIDEWAYS
        
        # Calculate SMA
        import pandas_ta as ta
        df['sma'] = ta.sma(df['Close'], length=CONFIG['SMA_BARS'])
        
        current_price = df['Close'].iloc[-1]
        current_sma = df['sma'].iloc[-1]
        
        if pd.isna(current_sma):
            return TREND_SIDEWAYS
        
        buffer_pct = CONFIG['SMA_BUFFER_PCT']
        
        # Determine trend with buffer
        if current_price > current_sma * (1 + buffer_pct):
            return TREND_UP
        elif current_price < current_sma * (1 - buffer_pct):
            return TREND_DOWN
        else:
            return TREND_SIDEWAYS
    
    except Exception as e:
        cprint(f"Error checking trend for {symbol}: {e}", "red")
        return TREND_SIDEWAYS

def try_position_entry(symbol, amount_to_buy_usdt):
    """Execute position entry with verification"""
    try:
        cprint(f"🟢 Executing BUY order: ${amount_to_buy_usdt} {symbol}", "green")
        
        result = n.market_buy(CONFIG, symbol, amount_to_buy_usdt)
        
        if result:
            # Wait for order to settle
            time.sleep(3)
            
            # Verify position
            current_value = n.get_position_value(CONFIG, symbol)
            
            if current_value > amount_to_buy_usdt * 0.8:  # 80% threshold for success
                cprint(f"✅ Position entry successful: ${current_value:.2f} value", "green")
                return True
            else:
                cprint(f"⚠️  Position entry may have failed: ${current_value:.2f} value", "yellow")
                return False
        else:
            cprint(f"❌ Buy order failed for {symbol}", "red")
            return False
    
    except Exception as e:
        cprint(f"Error in position entry: {e}", "red")
        return False

def verify_position_exit(symbol, expected_max_value):
    """Verify position was reduced to expected level"""
    try:
        time.sleep(2)  # Wait for order settlement
        
        current_value = n.get_position_value(CONFIG, symbol)
        
        if current_value <= expected_max_value * 1.1:  # 10% tolerance
            cprint(f"✅ Position exit verified: ${current_value:.2f} remaining", "green")
            return True
        else:
            cprint(f"⚠️  Position exit incomplete: ${current_value:.2f} remaining (target: ${expected_max_value:.2f})", "yellow")
            return False
    
    except Exception as e:
        cprint(f"Error verifying position exit: {e}", "red")
        return False

def execute_multiple_sells(symbol, sell_value_usdt, num_orders=2):
    """Execute multiple sell orders in succession"""
    try:
        base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        current_balance = n.get_account_balance(CONFIG, base_asset)
        
        if current_balance <= 0:
            cprint(f"No {base_asset} balance to sell", "yellow")
            return False
        
        # Get current price to calculate quantity
        price_data = n.make_binance_request(CONFIG, "/api/v3/ticker/price", {"symbol": symbol})
        if not price_data:
            cprint(f"Failed to get price for {symbol}", "red")
            return False
        
        current_price = float(price_data['price'])
        sell_quantity_per_order = (sell_value_usdt / num_orders) / current_price
        
        success_count = 0
        
        for i in range(num_orders):
            if sell_quantity_per_order > 0 and sell_quantity_per_order <= current_balance:
                cprint(f"🔴 Executing SELL order {i+1}/{num_orders}: {sell_quantity_per_order:.6f} {base_asset}", "red")
                
                result = n.market_sell(CONFIG, symbol, sell_quantity_per_order)
                
                if result:
                    success_count += 1
                    current_balance -= sell_quantity_per_order
                    time.sleep(1)  # Brief pause between orders
                else:
                    cprint(f"❌ Sell order {i+1} failed", "red")
            else:
                cprint(f"⚠️  Invalid sell quantity for order {i+1}: {sell_quantity_per_order:.6f}", "yellow")
        
        return success_count > 0
    
    except Exception as e:
        cprint(f"Error in multiple sells: {e}", "red")
        return False

def verify_and_sell(symbol, sell_value_usdt, expected_max_value):
    """Execute sells until position is reduced to target"""
    max_attempts = 3
    attempt = 1
    
    while attempt <= max_attempts:
        cprint(f"🔄 Sell attempt {attempt}/{max_attempts}", "cyan")
        
        execute_multiple_sells(symbol, sell_value_usdt, CONFIG['ORDERS_PER_SELL'])
        
        if verify_position_exit(symbol, expected_max_value):
            return True
        
        if attempt < max_attempts:
            cprint(f"⏳ Waiting before retry...", "yellow")
            time.sleep(5)
        
        attempt += 1
    
    cprint(f"⚠️  Failed to reduce position to target after {max_attempts} attempts", "yellow")
    return False

def print_zone_info(symbol, zones_data, current_price, position_value, trend):
    """Print current zone analysis information"""
    cprint(f"\n📊 ZONE ANALYSIS: {symbol}", "cyan", attrs=['bold'])
    cprint(f"Current Price: ${current_price:.6f}", "white")
    cprint(f"Position Value: ${position_value:.2f}", "white")
    cprint(f"Trend: {trend.upper()}", "green" if trend == TREND_UP else "red" if trend == TREND_DOWN else "yellow")
    
    # Show nearest supply zones
    supply_zones = zones_data.get('supply_zones', [])
    if supply_zones:
        nearest_supply = supply_zones[0]
        distance_to_supply = (nearest_supply['low'] - current_price) / current_price * 100
        cprint(f"🔴 Nearest Supply Zone: ${nearest_supply['low']:.6f} - ${nearest_supply['high']:.6f} ({distance_to_supply:.2f}% away)", "red")
    
    # Show nearest demand zones
    demand_zones = zones_data.get('demand_zones', [])
    if demand_zones:
        nearest_demand = demand_zones[0]
        distance_to_demand = (current_price - nearest_demand['high']) / current_price * 100
        cprint(f"🟢 Nearest Demand Zone: ${nearest_demand['low']:.6f} - ${nearest_demand['high']:.6f} ({distance_to_demand:.2f}% away)", "green")

def handle_demand_zone_logic(symbol, trend, current_value, position_pct, current_price):
    """Handle trading logic when price is in demand zone"""
    try:
        target_position = CONFIG['POSITION_SIZE_USD']
        
        if current_value < target_position * CONFIG['MINIMUM_POSITION_PCT']:
            # We have minimal or no position, consider buying
            amount_to_buy = target_position * (1 - position_pct)  # Buy remaining amount
            
            # Adjust buy amount based on trend
            if trend == TREND_UP:
                amount_to_buy *= 1.2  # Buy more in uptrend
            elif trend == TREND_DOWN:
                amount_to_buy *= 0.6  # Buy less in downtrend
            
            amount_to_buy = min(amount_to_buy, CONFIG['POSITION_SIZE_USD'])
            
            cprint(f"🟢 IN DEMAND ZONE - Trend: {trend} - Executing BUY: ${amount_to_buy:.2f}", "green", attrs=['bold'])
            
            success = try_position_entry(symbol, amount_to_buy)
            if success:
                cprint(f"✅ Demand zone entry successful", "green")
            else:
                cprint(f"❌ Demand zone entry failed", "red")
        else:
            cprint(f"📊 In demand zone but already have sufficient position ({position_pct:.1%})", "cyan")
    
    except Exception as e:
        cprint(f"Error in demand zone logic: {e}", "red")

def handle_supply_zone_logic(symbol, trend, position_value, position_pct, sell_percentage):
    """Handle trading logic when price is in supply zone"""
    try:
        if position_value < CONFIG['POSITION_SIZE_USD'] * CONFIG['MINIMUM_POSITION_PCT']:
            cprint(f"📊 In supply zone but position too small to sell (${position_value:.2f})", "cyan")
            return
        
        # Calculate sell amount based on trend and percentage
        sell_value = position_value * sell_percentage
        
        # Adjust sell amount based on trend  
        if trend == TREND_UP:
            sell_value *= 0.7  # Sell less in uptrend (hold for more gains)
        elif trend == TREND_DOWN:
            sell_value *= 1.2  # Sell more in downtrend (reduce risk)
        
        expected_remaining_value = position_value - sell_value
        
        cprint(f"🔴 IN SUPPLY ZONE - Trend: {trend} - Executing SELL: ${sell_value:.2f} (keep ${expected_remaining_value:.2f})", 
               "red", attrs=['bold'])
        
        success = verify_and_sell(symbol, sell_value, expected_remaining_value)
        if success:
            cprint(f"✅ Supply zone exit successful", "green")
        else:
            cprint(f"⚠️  Supply zone exit partially completed", "yellow")
    
    except Exception as e:
        cprint(f"Error in supply zone logic: {e}", "red")

def check_zone_and_trade(symbol):
    """Main trading logic - check zones and execute trades"""
    try:
        cprint(f"\n🔍 Analyzing {symbol}...", "cyan")
        
        # Get current position value
        position_value = n.get_position_value(CONFIG, symbol)
        position_pct = position_value / CONFIG['POSITION_SIZE_USD']
        
        # Get supply & demand zones
        zones_data = n.calculate_supply_demand_zones(CONFIG, symbol, CONFIG['SMA_TIMEFRAME'], 100)
        current_price = zones_data['current_price']
        
        if current_price <= 0:
            cprint(f"❌ Invalid price data for {symbol}", "red")
            return
        
        # Check trend
        trend = check_trend(symbol)
        
        # Print zone analysis
        print_zone_info(symbol, zones_data, current_price, position_value, trend)
        
        # Check if price is in a supply zone
        in_supply_zone = False
        for zone in zones_data.get('supply_zones', []):
            if zone['low'] <= current_price <= zone['high']:
                in_supply_zone = True
                zone_proximity = CONFIG['SDZ_CONFIG']['ZONE_PROXIMITY_PCT']
                
                # Determine sell percentage based on trend
                if trend == TREND_UP:
                    sell_pct = CONFIG['SELL_PERCENTAGE_TRENDING_UP']
                else:
                    sell_pct = CONFIG['SELL_PERCENTAGE_TRENDING_DOWN']
                
                handle_supply_zone_logic(symbol, trend, position_value, position_pct, sell_pct)
                break
        
        # Check if price is in a demand zone (only if not in supply zone)
        if not in_supply_zone:
            for zone in zones_data.get('demand_zones', []):
                if zone['low'] <= current_price <= zone['high']:
                    handle_demand_zone_logic(symbol, trend, position_value, position_pct, current_price)
                    break
        
        if not in_supply_zone:
            # Check proximity to zones
            supply_zones = zones_data.get('supply_zones', [])
            demand_zones = zones_data.get('demand_zones', [])
            
            if supply_zones:
                nearest_supply = supply_zones[0]
                supply_distance = abs(current_price - nearest_supply['low']) / current_price
                if supply_distance <= CONFIG['SDZ_CONFIG']['ZONE_PROXIMITY_PCT']:
                    cprint(f"⚠️  Approaching supply zone: {supply_distance:.2%} away", "yellow")
            
            if demand_zones:
                nearest_demand = demand_zones[0]
                demand_distance = abs(current_price - nearest_demand['high']) / current_price
                if demand_distance <= CONFIG['SDZ_CONFIG']['ZONE_PROXIMITY_PCT']:
                    cprint(f"⚠️  Approaching demand zone: {demand_distance:.2%} away", "yellow")
    
    except Exception as e:
        cprint(f"Error in zone analysis for {symbol}: {e}", "red")

def run_bot():
    """Main bot execution function"""
    try:
        print_banner()
        
        # Check market conditions
        market_conditions = n.check_market_conditions(CONFIG)
        cprint(f"📊 Market Condition: {market_conditions['condition'].upper()} ({market_conditions['bullish_percentage']:.1%} bullish)", 
               "green" if market_conditions['condition'] == 'bullish' else "red" if market_conditions['condition'] == 'bearish' else "yellow")
        
        symbols = CONFIG['SYMBOLS']
        cprint(f"🎯 Analyzing {len(symbols)} symbols...", "white")
        
        for symbol in symbols:
            if symbol in CONFIG['DO_NOT_TRADE_LIST']:
                cprint(f"⚠️  Skipping {symbol} - on do not trade list", "yellow")
                continue
            
            try:
                check_zone_and_trade(symbol)
                time.sleep(2)  # Brief pause between symbols
            except Exception as e:
                cprint(f"Error processing {symbol}: {e}", "red")
                continue
        
        # Print portfolio summary
        total_portfolio_value = 0
        cprint(f"\n📊 PORTFOLIO SUMMARY", "cyan", attrs=['bold'])
        
        for symbol in symbols:
            if symbol not in CONFIG['DO_NOT_TRADE_LIST']:
                try:
                    value = n.get_position_value(CONFIG, symbol)
                    if value > 1:  # Only show positions > $1
                        total_portfolio_value += value
                        cprint(f"{symbol}: ${value:.2f}", "white")
                except:
                    pass
        
        # Add USDT balance
        usdt_balance = n.get_account_balance(CONFIG, 'USDT')
        total_portfolio_value += usdt_balance
        
        cprint(f"USDT Balance: ${usdt_balance:.2f}", "white")
        cprint(f"Total Portfolio: ${total_portfolio_value:.2f}", "green", attrs=['bold'])
        
    except Exception as e:
        cprint(f"Error in bot execution: {e}", "red")

def schedule_bot():
    """Schedule the bot to run at intervals"""
    # Run every 15 minutes during active trading hours
    schedule.every(15).minutes.do(run_bot)
    
    cprint("⏰ Bot scheduled to run every 15 minutes", "cyan")
    cprint("Press Ctrl+C to stop the bot", "yellow")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        cprint("\n🛑 Bot stopped by user", "yellow")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Binance Supply & Demand Zone Trading Bot')
    parser.add_argument('--mode', choices=['once', 'schedule'], default='once',
                       help='Run mode: once (single run) or schedule (continuous)')
    parser.add_argument('--symbol', type=str, help='Trade specific symbol only')
    
    args = parser.parse_args()
    
    try:
        if args.symbol:
            # Trade specific symbol
            CONFIG['SYMBOLS'] = [args.symbol.upper()]
            cprint(f"🎯 Trading single symbol: {args.symbol.upper()}", "cyan")
        
        if args.mode == 'once':
            run_bot()
        else:
            run_bot()  # Run once immediately
            schedule_bot()  # Then start scheduling
    
    except KeyboardInterrupt:
        cprint("\n👋 Bot shutdown gracefully", "green")
    except Exception as e:
        cprint(f"💥 Bot crashed: {e}", "red")
        raise

if __name__ == "__main__":
    main()
