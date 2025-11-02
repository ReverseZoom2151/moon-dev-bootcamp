"""
Bitfinex Professional Supply & Demand Zone Trading Bot

Institutional-grade Supply & Demand Zone strategy for Bitfinex exchange.
Features advanced zone detection, funding rate analysis, hidden orders, and professional risk management.
Includes margin trading capabilities and sophisticated market analysis.
"""

import bitfinex_nice_funcs2 as n
import time
import schedule
import pandas as pd
import sys
from bitfinex_CONFIG import get_bitfinex_sdz_config, validate_config
from termcolor import cprint

# Load configuration
CONFIG = get_bitfinex_sdz_config()

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

def print_professional_banner():
    """Print institutional startup banner"""
    cprint("="*100, "cyan")
    cprint("🏛️  BITFINEX INSTITUTIONAL SUPPLY & DEMAND ZONE TRADING SYSTEM", "cyan", attrs=['bold'])
    cprint("Professional SDZ Strategy with Advanced Market Analysis & Funding Rate Integration", "white")
    cprint("="*100, "cyan")
    print()

def check_trend(symbol):
    """Check trend using advanced SMA analysis with institutional parameters"""
    try:
        timeframe = CONFIG['SMA_TIMEFRAME']
        days_back = CONFIG['SMA_DAYS_BACK']
        limit = days_back * 24 if timeframe == '1h' else days_back * 24 * 4
        
        df = n.get_historical_data(CONFIG, symbol, timeframe, min(limit, 200))
        
        if df is None or len(df) < CONFIG['SMA_BARS']:
            cprint(f"❌ Insufficient data for trend analysis: {symbol}", "red")
            return TREND_SIDEWAYS
        
        # Calculate multiple SMAs for confirmation
        import pandas_ta as ta
        sma_period = CONFIG['SMA_BARS']
        df['sma_fast'] = ta.sma(df['Close'], length=sma_period)
        df['sma_slow'] = ta.sma(df['Close'], length=sma_period * 2)
        df['sma_trend'] = ta.sma(df['Close'], length=sma_period * 4)
        
        current_price = df['Close'].iloc[-1]
        sma_fast = df['sma_fast'].iloc[-1]
        sma_slow = df['sma_slow'].iloc[-1]
        sma_trend = df['sma_trend'].iloc[-1]
        
        if pd.isna(sma_fast) or pd.isna(sma_slow):
            return TREND_SIDEWAYS
        
        buffer_pct = CONFIG['SMA_BUFFER_PCT']
        
        # Multi-timeframe trend confirmation
        if (current_price > sma_fast * (1 + buffer_pct) and 
            sma_fast > sma_slow and 
            sma_slow > sma_trend):
            return TREND_UP
        elif (current_price < sma_fast * (1 - buffer_pct) and 
              sma_fast < sma_slow and 
              sma_slow < sma_trend):
            return TREND_DOWN
        else:
            return TREND_SIDEWAYS
    
    except Exception as e:
        cprint(f"Error checking trend for {symbol}: {e}", "red")
        return TREND_SIDEWAYS

def try_position_entry(symbol, amount_usd):
    """Execute professional position entry with institutional verification"""
    try:
        cprint(f"🟢 INSTITUTIONAL BUY ORDER: ${amount_usd} {symbol}", "green", attrs=['bold'])
        
        # Check funding rate before entry if enabled
        if CONFIG.get('ENABLE_FUNDING_ANALYSIS'):
            funding_info = n.get_funding_rate(CONFIG, symbol)
            if funding_info:
                funding_rate = funding_info['funding_rate']
                if funding_rate > CONFIG.get('FUNDING_RATE_THRESHOLD', 0.008):
                    cprint(f"⚠️  High funding rate: {funding_rate:.4%} daily - Proceeding with caution", "yellow")
                else:
                    cprint(f"💰 Favorable funding rate: {funding_rate:.4%} daily", "green")
        
        result = n.market_buy(CONFIG, symbol, amount_usd)
        
        if result:
            # Wait for settlement
            time.sleep(CONFIG.get('CHECK_DELAY', 3))
            
            # Verify position with higher threshold for institutional trading
            current_value = n.get_position_value(CONFIG, symbol)
            success_threshold = amount_usd * CONFIG.get('PARTIAL_FILL_THRESHOLD', 0.85)
            
            if current_value >= success_threshold:
                cprint(f"✅ INSTITUTIONAL POSITION ESTABLISHED: ${current_value:.2f} value", "green", attrs=['bold'])
                return True
            else:
                cprint(f"⚠️  Position establishment incomplete: ${current_value:.2f} vs expected ${amount_usd:.2f}", "yellow")
                return False
        else:
            cprint(f"❌ Institutional buy order failed for {symbol}", "red")
            return False
    
    except Exception as e:
        cprint(f"Error in institutional position entry: {e}", "red")
        return False

def verify_position_exit(symbol, expected_max_value):
    """Verify position reduction with institutional precision"""
    try:
        time.sleep(CONFIG.get('CHECK_DELAY', 3))
        
        current_value = n.get_position_value(CONFIG, symbol)
        tolerance = expected_max_value * 1.15  # 15% tolerance for institutional trading
        
        if current_value <= tolerance:
            cprint(f"✅ INSTITUTIONAL EXIT VERIFIED: ${current_value:.2f} remaining", "green", attrs=['bold'])
            return True
        else:
            cprint(f"⚠️  Institutional exit requires review: ${current_value:.2f} remaining (target: ${expected_max_value:.2f})", "yellow")
            return False
    
    except Exception as e:
        cprint(f"Error verifying institutional position exit: {e}", "red")
        return False

def execute_professional_sells(symbol, sell_value_usd):
    """Execute multiple sell orders with institutional precision"""
    try:
        base_currency = symbol.replace('t', '').replace('USD', '').replace('UST', '').replace('USDT', '')
        current_balance = n.get_account_balance(CONFIG, base_currency)
        
        if current_balance <= 0:
            cprint(f"No {base_currency} balance to sell", "yellow")
            return False
        
        # Get current price for quantity calculation
        ticker_symbol = symbol[1:] if symbol.startswith('t') else symbol
        ticker_data = n.make_bitfinex_request(CONFIG, f"/v1/pubticker/{ticker_symbol}")
        
        if not ticker_data:
            cprint(f"Failed to get price for {symbol}", "red")
            return False
        
        current_price = float(ticker_data.get('bid', 0))  # Use bid price for selling
        if current_price <= 0:
            return False
        
        total_sell_quantity = sell_value_usd / current_price
        num_orders = CONFIG.get('ORDERS_PER_SELL', 3)
        sell_quantity_per_order = total_sell_quantity / num_orders
        
        success_count = 0
        
        for i in range(num_orders):
            if sell_quantity_per_order > 0 and sell_quantity_per_order <= current_balance:
                cprint(f"🔴 INSTITUTIONAL SELL ORDER {i+1}/{num_orders}: {sell_quantity_per_order:.6f} {base_currency}", "red", attrs=['bold'])
                
                result = n.market_sell(CONFIG, symbol, sell_quantity_per_order)
                
                if result:
                    success_count += 1
                    current_balance -= sell_quantity_per_order
                    time.sleep(1)  # Professional execution pace
                else:
                    cprint(f"❌ Institutional sell order {i+1} failed", "red")
            else:
                cprint(f"⚠️  Invalid sell quantity for order {i+1}: {sell_quantity_per_order:.6f}", "yellow")
        
        return success_count > 0
    
    except Exception as e:
        cprint(f"Error in professional sell execution: {e}", "red")
        return False

def verify_and_sell(symbol, sell_value_usd, expected_max_value):
    """Execute institutional sells with verification"""
    max_attempts = CONFIG.get('MAX_RETRIES', 5)
    attempt = 1
    
    while attempt <= max_attempts:
        cprint(f"🔄 INSTITUTIONAL SELL ATTEMPT {attempt}/{max_attempts}", "cyan", attrs=['bold'])
        
        execute_professional_sells(symbol, sell_value_usd)
        
        if verify_position_exit(symbol, expected_max_value):
            return True
        
        if attempt < max_attempts:
            cprint(f"⏳ Institutional retry delay...", "yellow")
            time.sleep(CONFIG.get('RETRY_DELAY_SECONDS', 2))
        
        attempt += 1
    
    cprint(f"⚠️  Institutional position reduction incomplete after {max_attempts} attempts", "yellow")
    return False

def print_professional_zone_info(symbol, zones_data, current_price, position_value, trend):
    """Print comprehensive zone analysis with institutional detail"""
    cprint(f"\n📊 INSTITUTIONAL ZONE ANALYSIS: {symbol}", "cyan", attrs=['bold'])
    cprint(f"💲 Current Price: ${current_price:.6f}", "white")
    cprint(f"💼 Position Value: ${position_value:.2f}", "white")
    cprint(f"📈 Trend Analysis: {trend.upper()}", "green" if trend == TREND_UP else "red" if trend == TREND_DOWN else "yellow")
    
    # Show technical indicators
    if 'rsi' in zones_data:
        rsi = zones_data['rsi']
        rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "yellow"
        cprint(f"📊 RSI: {rsi:.1f}", rsi_color)
    
    if 'momentum' in zones_data:
        momentum = zones_data.get('momentum', 0)
        momentum_color = "green" if momentum > 0 else "red"
        cprint(f"🚀 Price Momentum: {momentum:.2%}", momentum_color)
    
    # Display funding rate if available
    if CONFIG.get('ENABLE_FUNDING_ANALYSIS'):
        funding_info = n.get_funding_rate(CONFIG, symbol)
        if funding_info:
            rate = funding_info['funding_rate']
            rate_color = "green" if rate < 0 else "red" if rate > 0.005 else "yellow"
            cprint(f"💰 Funding Rate: {rate:.4%} daily", rate_color)
    
    # Show supply zones
    supply_zones = zones_data.get('supply_zones', [])
    if supply_zones:
        nearest_supply = supply_zones[0]
        distance = nearest_supply['distance_pct'] * 100
        strength_stars = "⭐" * min(nearest_supply['strength'], 5)
        cprint(f"🔴 Nearest Supply Zone: ${nearest_supply['low']:.6f} - ${nearest_supply['high']:.6f} ({distance:.2f}% away) {strength_stars}", "red")
    
    # Show demand zones
    demand_zones = zones_data.get('demand_zones', [])
    if demand_zones:
        nearest_demand = demand_zones[0]
        distance = nearest_demand['distance_pct'] * 100
        strength_stars = "⭐" * min(nearest_demand['strength'], 5)
        cprint(f"🟢 Nearest Demand Zone: ${nearest_demand['low']:.6f} - ${nearest_demand['high']:.6f} ({distance:.2f}% away) {strength_stars}", "green")

def handle_demand_zone_logic(symbol, trend, current_value, position_pct, current_price, zone_info):
    """Handle institutional demand zone trading logic"""
    try:
        target_position = CONFIG['POSITION_SIZE_USD']
        min_position_pct = CONFIG['MINIMUM_POSITION_PCT']
        
        if current_value < target_position * min_position_pct:
            # Calculate buy amount with trend adjustment
            amount_to_buy = target_position * (1 - position_pct)
            
            # Institutional trend-based position sizing
            if trend == TREND_UP:
                amount_to_buy *= 1.3  # More aggressive in confirmed uptrend
            elif trend == TREND_DOWN:
                amount_to_buy *= 0.5  # Very conservative in downtrend
            else:
                amount_to_buy *= 0.8  # Moderate in sideways trend
            
            # Check zone strength for additional confidence
            if zone_info and zone_info.get('strength', 0) >= 4:
                amount_to_buy *= 1.2  # Increase size for strong zones
                cprint(f"💪 Strong demand zone detected (strength: {zone_info['strength']})", "green")
            
            amount_to_buy = min(amount_to_buy, target_position)
            
            cprint(f"🟢 INSTITUTIONAL DEMAND ZONE ENTRY - Trend: {trend} - Size: ${amount_to_buy:.2f}", "green", attrs=['bold'])
            
            success = try_position_entry(symbol, amount_to_buy)
            if success:
                cprint(f"✅ Institutional demand zone entry completed successfully", "green", attrs=['bold'])
            else:
                cprint(f"❌ Institutional demand zone entry failed", "red")
        else:
            cprint(f"📊 In demand zone but institutional position already sufficient ({position_pct:.1%})", "cyan")
    
    except Exception as e:
        cprint(f"Error in institutional demand zone logic: {e}", "red")

def handle_supply_zone_logic(symbol, trend, position_value, position_pct, sell_percentage, zone_info):
    """Handle institutional supply zone trading logic"""
    try:
        min_position_pct = CONFIG['MINIMUM_POSITION_PCT']
        target_position = CONFIG['POSITION_SIZE_USD']
        
        if position_value < target_position * min_position_pct:
            cprint(f"📊 In supply zone but institutional position too small to sell (${position_value:.2f})", "cyan")
            return
        
        # Calculate sell amount with institutional trend analysis
        sell_value = position_value * sell_percentage
        
        # Adjust based on trend with institutional logic
        if trend == TREND_UP:
            sell_value *= 0.6  # Keep more position in strong uptrend
            cprint(f"📈 Reducing sell amount due to strong uptrend", "yellow")
        elif trend == TREND_DOWN:
            sell_value *= 1.4  # Sell more aggressively in downtrend
            cprint(f"📉 Increasing sell amount due to downtrend", "yellow")
        
        # Check zone strength for institutional confidence
        if zone_info and zone_info.get('strength', 0) >= 4:
            sell_value *= 1.2  # Sell more at strong resistance
            cprint(f"💪 Strong supply zone - increasing sell amount (strength: {zone_info['strength']})", "red")
        
        # Check funding rate for additional alpha
        if CONFIG.get('ENABLE_FUNDING_ANALYSIS'):
            funding_info = n.get_funding_rate(CONFIG, symbol)
            if funding_info and funding_info['funding_rate'] > CONFIG.get('FUNDING_RATE_THRESHOLD', 0.008):
                sell_value *= 1.1  # Sell more if funding is expensive
                cprint(f"💸 High funding rate - increasing sell amount", "red")
        
        expected_remaining_value = position_value - sell_value
        
        cprint(f"🔴 INSTITUTIONAL SUPPLY ZONE EXIT - Trend: {trend} - Sell: ${sell_value:.2f} (Keep: ${expected_remaining_value:.2f})", 
               "red", attrs=['bold'])
        
        success = verify_and_sell(symbol, sell_value, expected_remaining_value)
        if success:
            cprint(f"✅ Institutional supply zone exit completed successfully", "green", attrs=['bold'])
        else:
            cprint(f"⚠️  Institutional supply zone exit partially completed", "yellow")
    
    except Exception as e:
        cprint(f"Error in institutional supply zone logic: {e}", "red")

def check_zone_and_trade(symbol):
    """Main institutional trading logic with comprehensive analysis"""
    try:
        cprint(f"\n🔍 INSTITUTIONAL ANALYSIS: {symbol}...", "cyan", attrs=['bold'])
        
        # Get current position
        position_value = n.get_position_value(CONFIG, symbol)
        position_pct = position_value / CONFIG['POSITION_SIZE_USD']
        
        # Get advanced supply & demand zones
        zones_data = n.calculate_supply_demand_zones(CONFIG, symbol, CONFIG['SMA_TIMEFRAME'], 150)
        current_price = zones_data['current_price']
        
        if current_price <= 0:
            cprint(f"❌ Invalid price data for {symbol}", "red")
            return
        
        # Professional trend analysis
        trend = check_trend(symbol)
        
        # Print comprehensive analysis
        print_professional_zone_info(symbol, zones_data, current_price, position_value, trend)
        
        # Check supply zone interaction
        in_supply_zone = False
        active_supply_zone = None
        
        for zone in zones_data.get('supply_zones', []):
            if zone['low'] <= current_price <= zone['high']:
                in_supply_zone = True
                active_supply_zone = zone
                
                # Determine sell percentage based on trend
                if trend == TREND_UP:
                    sell_pct = CONFIG['SELL_PERCENTAGE_TRENDING_UP']
                else:
                    sell_pct = CONFIG['SELL_PERCENTAGE_TRENDING_DOWN']
                
                handle_supply_zone_logic(symbol, trend, position_value, position_pct, sell_pct, zone)
                break
        
        # Check demand zone interaction (only if not in supply zone)
        if not in_supply_zone:
            active_demand_zone = None
            
            for zone in zones_data.get('demand_zones', []):
                if zone['low'] <= current_price <= zone['high']:
                    active_demand_zone = zone
                    handle_demand_zone_logic(symbol, trend, position_value, position_pct, current_price, zone)
                    break
            
            # Show proximity warnings for institutional traders
            if not active_demand_zone:
                supply_zones = zones_data.get('supply_zones', [])
                demand_zones = zones_data.get('demand_zones', [])
                
                proximity_threshold = CONFIG['SDZ_CONFIG']['ZONE_PROXIMITY_PCT']
                
                if supply_zones:
                    nearest_supply = supply_zones[0]
                    if nearest_supply['distance_pct'] <= proximity_threshold:
                        cprint(f"⚠️  APPROACHING SUPPLY ZONE: {nearest_supply['distance_pct']:.2%} away", "yellow", attrs=['bold'])
                
                if demand_zones:
                    nearest_demand = demand_zones[0]
                    if nearest_demand['distance_pct'] <= proximity_threshold:
                        cprint(f"⚠️  APPROACHING DEMAND ZONE: {nearest_demand['distance_pct']:.2%} away", "yellow", attrs=['bold'])
    
    except Exception as e:
        cprint(f"Error in institutional zone analysis for {symbol}: {e}", "red")

def run_institutional_bot():
    """Main institutional bot execution with comprehensive analysis"""
    try:
        print_professional_banner()
        
        # Professional market conditions analysis
        market_conditions = n.check_market_conditions(CONFIG)
        condition = market_conditions['condition']
        bullish_pct = market_conditions['bullish_percentage']
        
        condition_color = "green" if condition == 'bullish' else "red" if condition == 'bearish' else "yellow"
        cprint(f"📊 INSTITUTIONAL MARKET ANALYSIS: {condition.upper()} ({bullish_pct:.1%} bullish sentiment)", 
               condition_color, attrs=['bold'])
        
        # Show individual major coin trends
        for symbol, trend in market_conditions['individual_trends'].items():
            trend_color = "green" if trend == 'up' else "red" if trend == 'down' else "yellow"
            cprint(f"   {symbol}: {trend.upper()}", trend_color)
        
        symbols = CONFIG['SYMBOLS']
        cprint(f"🎯 Analyzing {len(symbols)} institutional trading pairs...", "white")
        
        # Track portfolio performance
        total_portfolio_value = 0
        active_positions = 0
        
        for symbol in symbols:
            if symbol in CONFIG['DO_NOT_TRADE_LIST']:
                cprint(f"⚠️  Skipping {symbol} - on institutional exclusion list", "yellow")
                continue
            
            try:
                position_value = n.get_position_value(CONFIG, symbol)
                if position_value > 1:
                    active_positions += 1
                    total_portfolio_value += position_value
                
                check_zone_and_trade(symbol)
                time.sleep(2)  # Professional execution pace
                
            except Exception as e:
                cprint(f"Error processing {symbol}: {e}", "red")
                continue
        
        # Professional portfolio summary
        cprint(f"\n📊 INSTITUTIONAL PORTFOLIO SUMMARY", "cyan", attrs=['bold'])
        cprint(f"Active Positions: {active_positions}", "white")
        
        for symbol in symbols:
            if symbol not in CONFIG['DO_NOT_TRADE_LIST']:
                try:
                    value = n.get_position_value(CONFIG, symbol)
                    if value > 5:  # Only show significant institutional positions
                        percentage = (value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                        cprint(f"{symbol}: ${value:.2f} ({percentage:.1f}%)", "white")
                except:
                    pass
        
        # Add USD balance
        usd_balance = n.get_account_balance(CONFIG, 'USD')
        total_portfolio_value += usd_balance
        
        cprint(f"USD Balance: ${usd_balance:.2f}", "white")
        cprint(f"Total Institutional Portfolio: ${total_portfolio_value:.2f}", "green", attrs=['bold'])
        
        # Risk monitoring
        if total_portfolio_value > 0:
            position_concentration = max([n.get_position_value(CONFIG, s) for s in symbols if s not in CONFIG['DO_NOT_TRADE_LIST']] + [0]) / total_portfolio_value
            if position_concentration > CONFIG.get('POSITION_CONCENTRATION_LIMIT', 0.3):
                cprint(f"⚠️  HIGH POSITION CONCENTRATION: {position_concentration:.1%}", "yellow", attrs=['bold'])
        
    except Exception as e:
        cprint(f"Error in institutional bot execution: {e}", "red")

def schedule_institutional_bot():
    """Schedule the institutional bot with professional intervals"""
    interval = CONFIG.get('TRADING_SCHEDULE', {}).get('CHECK_INTERVAL_MINUTES', 15)
    
    schedule.every(interval).minutes.do(run_institutional_bot)
    
    cprint(f"⏰ Institutional bot scheduled every {interval} minutes", "cyan", attrs=['bold'])
    cprint("Press Ctrl+C to stop the institutional trading system", "yellow")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        cprint("\n🛑 Institutional trading system stopped by operator", "yellow", attrs=['bold'])

def main():
    """Main institutional execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bitfinex Institutional Supply & Demand Zone Trading Bot')
    parser.add_argument('--mode', choices=['once', 'schedule'], default='once',
                       help='Execution mode: once (single analysis) or schedule (continuous operation)')
    parser.add_argument('--symbol', type=str, help='Analyze specific symbol only')
    
    args = parser.parse_args()
    
    try:
        if args.symbol:
            # Single symbol institutional analysis
            symbol = args.symbol.upper()
            if not symbol.startswith('t'):
                symbol = f"t{symbol}USD"  # Add Bitfinex format
            
            CONFIG['SYMBOLS'] = [symbol]
            cprint(f"🎯 INSTITUTIONAL SINGLE SYMBOL ANALYSIS: {symbol}", "cyan", attrs=['bold'])
        
        if args.mode == 'once':
            run_institutional_bot()
        else:
            run_institutional_bot()  # Run once immediately
            schedule_institutional_bot()  # Then start continuous operation
    
    except KeyboardInterrupt:
        cprint("\n👋 Institutional bot shutdown completed", "green", attrs=['bold'])
    except Exception as e:
        cprint(f"💥 Institutional bot critical error: {e}", "red", attrs=['bold'])
        raise

if __name__ == "__main__":
    main()
