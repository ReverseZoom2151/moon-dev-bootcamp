'''
Building out a supply and demand zone for the hand-picked winners out of my copy and trending Solana bots. 

the idea is that some 'winners' will float to the top of my copy bot, sniper and trending solana bots
and those can be placed into this bot to trade a Supply and Demand Zone strategy 
this will allow for better entries and accumulation of free tokens 




'''

import nice_funcs2 as n
from termcolor import cprint
import time
import schedule
from datetime import datetime

# motion - FFVcrc7wxCQ2HsxY6jhKNPwWUeKJsL472K28Vw16pump
# housecoin - DitHyRMQiSDhn5cnKMJV2CDDt6sVct96YrECiM49pump

# Moon Dev's public address
# MY_ADDRESS = "G1vNV2SkzGPTBTDd7c3iypL4TPNGbe37rEbJb7cxVTQS" # Will use CONFIG

symbols = [
    'DitHyRMQiSDhn5cnKMJV2CDDt6sVct96YrECiM49pump',  # housecoin
#     '8BtoThi2ZoXnF7QQK1Wjmh2JuBw9FjVvhnGMVZ2vpump',
#     'Df6yfrKC8kZE3KNkrHERKzAetSxbrWeniQfyJY4Jpump',
#     'Hax9LTgsQkze1YFychnBLtFH8gYbQKtKfWKKg2SP6gdD',
#     'C3DwDjT17gDvvCYC2nsdGHxDHVmQRdhKfpAdqQ29pump',
#     'FtUEW73K6vEYHfbkfpdBZfWpxgQar2HipGdbutEhpump',
#     'A8C3xuqscfmyLrte3VmTqrAq8kgMASius9AFNANwpump',
#     'HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC',
]
# timeframe = '15m' # Will use CONFIG
# days_back_4_data = .2 # Will use CONFIG
SLIPPAGE = 99 # .99% # Will use CONFIG

# Trend-based sell percentages
# SELL_PERCENTAGE_TRENDING_UP = 0.50   # 50% sell in uptrend # Will use CONFIG
# SELL_PERCENTAGE_TRENDING_DOWN = 0.95  # 95% sell in downtrend # Will use CONFIG

# Position thresholds
# POSITION_SIZE_USD = 150  # Target position size in USD # Will use CONFIG
# MINIMUM_POSITION_PCT = 0.05  # 5% of target position size as minimum to consider trading # Will use CONFIG
# BUFFER_PCT = 0.05  # 5% buffer to account for price fluctuations # Will use CONFIG

# Order execution settings
# ORDERS_PER_OPEN = 1  # Number of buy orders to place when opening position # Will use CONFIG
# ORDERS_PER_SELL = 3  # Number of sell orders to place in each sell attempt # Will use CONFIG

# SMA settings
# SMA_TIMEFRAME = '1H'  # Use 1H data directly from Birdeye # Will use CONFIG
# SMA_DAYS_BACK = 2     # Get 2 days of data to ensure enough bars # Will use CONFIG
# SMA_BARS = 10         # Use 10 bars for SMA # Will use CONFIG
# SMA_BUFFER_PCT = 0.15  # 10% buffer zone around SMA for trend determination # Will use CONFIG

# Define Trend constants
TREND_UP = 'up'
TREND_DOWN = 'down'

CONFIG = {
    "MY_ADDRESS": "G1vNV2SkzGPTBTDd7c3iypL4TPNGbe37rEbJb7cxVTQS",
    "SYMBOLS": [
        'DitHyRMQiSDhn5cnKMJV2CDDt6sVct96YrECiM49pump',  # housecoin
        # '8BtoThi2ZoXnF7QQK1Wjmh2JuBw9FjVvhnGMVZ2vpump',
        # 'Df6yfrKC8kZE3KNkrHERKzAetSxbrWeniQfyJY4Jpump',
        # 'Hax9LTgsQkze1YFychnBLtFH8gYbQKtKfWKKg2SP6gdD',
        # 'C3DwDjT17gDvvCYC2nsdGHxDHVmQRdhKfpAdqQ29pump',
        # 'FtUEW73K6vEYHfbkfpdBZfWpxgQar2HipGdbutEhpump',
        # 'A8C3xuqscfmyLrte3VmTqrAq8kgMASius9AFNANwpump',
        # 'HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC',
    ],
    "TIMEFRAME": '15m',
    "DAYS_BACK_4_DATA": 0.2,
    "SLIPPAGE": 99,  # .99%
    "SELL_PERCENTAGE_TRENDING_UP": 0.50,
    "SELL_PERCENTAGE_TRENDING_DOWN": 0.95,
    "POSITION_SIZE_USD": 150,
    "MINIMUM_POSITION_PCT": 0.05,
    "BUFFER_PCT": 0.05,
    "ORDERS_PER_OPEN": 1,
    "ORDERS_PER_SELL": 3,
    "SMA_TIMEFRAME": '1H',
    "SMA_DAYS_BACK": 2,
    "SMA_BARS": 10,
    "SMA_BUFFER_PCT": 0.15,
}

def aggregate_to_2h(df):
    """
    Aggregate 1H data into 2H bars
    """
    # Set the datetime column as index if it isn't already
    if 'Datetime (UTC)' in df.columns:
        df = df.set_index('Datetime (UTC)')
    
    # Resample to 2H
    df_2h = df.resample('2H', closed='left', label='left').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return df_2h

def check_trend(symbol):
    """
    Check if price is above/below SMA for trend determination
    Returns: 
        - 'up' if price > SMA * (1 - buffer)
        - 'down' if price < SMA * (1 - buffer)
    """
    try:
        price = n.token_price(symbol)
        if price is None:
            cprint(f'❌ Could not get price for {symbol[:4]}', 'red')
            return TREND_DOWN
        
        cprint(f'🏷️ Current price for {symbol[:4]}: ${price:.8f}', 'cyan')
        
        df = n.get_data(symbol, CONFIG["SMA_DAYS_BACK"], CONFIG["SMA_TIMEFRAME"])
        if df.empty:
            cprint(f'❌ No data available for trend analysis of {symbol[:4]}', 'red')
            return TREND_DOWN
            
        available_bars = len(df)
        if available_bars < 3:
            cprint(f'⚠️ Not enough bars for {symbol[:4]} - defaulting to down trend', 'yellow')
            return TREND_DOWN
            
        bars_to_use = min(available_bars, CONFIG["SMA_BARS"])
        sma = df['Close'].tail(bars_to_use).mean()
        sma_lower = sma * (1 - CONFIG["SMA_BUFFER_PCT"])
        
        trend = TREND_UP if price > sma_lower else TREND_DOWN
        cprint(f'🌙 MoonDev Trend determined: {trend.upper()}', 'green' if trend == TREND_UP else 'red')
        
        return trend
        
    except Exception as e:
        cprint(f'❌ Error in trend analysis for {symbol[:4]}: {str(e)}', 'red')
        return TREND_DOWN

def try_position_entry(symbol, amount_to_buy_usdc):
    """
    Simple position entry that checks if we have a position after ordering
    Returns True if we have any position
    """
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        # Check if we already have a position
        position = n.get_position(symbol)
        
        if position > 0:
            current_value = position * n.token_price(symbol)
            cprint(f'✅ Have position: ${current_value:.2f}', 'green')
            return True
            
        # No position, try to buy
        cprint(f'🌙 MoonDev Order Attempt {attempt + 1}/{max_attempts}', 'yellow')
        n.market_buy(symbol, amount_to_buy_usdc, CONFIG["SLIPPAGE"])
        cprint(f'✨ Buy order placed', 'yellow')
        
        # Quick check if we got the position
        time.sleep(2)  # Brief pause to let order process
        position = n.get_position(symbol)
        if position > 0:
            current_value = position * n.token_price(symbol)
            cprint(f'✅ Got position: ${current_value:.2f}', 'green')
            return True
            
        attempt += 1
        if attempt < max_attempts:
            cprint(f'🔄 No position detected, trying again...', 'yellow')
            time.sleep(1)
    
    cprint(f'❌ Failed to get position after {max_attempts} attempts', 'red')
    return False

def verify_position_exit(symbol, expected_max_value):
    """
    Verify position was reduced and retry if needed
    Returns True if position is reduced to target
    """
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        position = n.get_position(symbol)
        current_value = position * n.token_price(symbol)
        
        if current_value <= expected_max_value:
            cprint(f'✅ Position reduction verified: ${current_value:.2f}', 'green')
            return True
            
        attempt += 1
        cprint(f'⏳ Waiting for position reduction (attempt {attempt}/{max_attempts})...', 'yellow')
        time.sleep(5)
    
    cprint(f'❌ Failed to verify position reduction after {max_attempts} attempts', 'red')
    return False

def execute_triple_sell(symbol, sell_size, slippage):
    """Execute three sell orders in quick succession"""
    for _ in range(CONFIG["ORDERS_PER_SELL"]):
        n.market_sell(symbol, sell_size, slippage)
        time.sleep(1)  # Small delay between orders
    time.sleep(5)  # Longer delay after triple-sell to let orders process

def verify_and_sell(symbol, sell_size, expected_max_value):
    """Execute triple-sells until position is reduced"""
    attempts = 0
    original_sell_size = sell_size  # Store original sell size to maintain percentage
    
    while True:  # Keep trying until position is reduced
        attempts += 1
        cprint(f'🔄 Sell attempt {attempts}...', 'yellow')
        execute_triple_sell(symbol, sell_size, CONFIG["SLIPPAGE"])
        
        # Check if position is reduced
        position = n.get_position(symbol)
        current_value = position * n.token_price(symbol)
        
        if current_value <= expected_max_value:
            cprint('✅ Position successfully reduced!', 'green')
            return True
            
        # If not reduced enough, calculate remaining percentage to sell
        if position > 0:
            decimals = n.get_decimals(symbol)
            # Calculate how much of our intended sale is left to do
            remaining_to_sell = (original_sell_size - ((original_sell_size - (position * 10**decimals))))
            sell_size = int(min(remaining_to_sell, position * 10**decimals))  # Ensure we don't try to sell more than we have
            cprint(f'⚠️ Position not fully reduced. Trying again with {sell_size} tokens...', 'yellow')
            
            if attempts >= 3:  # Prevent infinite loops
                cprint('⚠️ Max sell attempts reached', 'yellow')
                return False
        else:
            cprint('✅ Position fully closed!', 'green')
            return True

def print_zone_info(symbol, zones, price, position, trend, current_value):
    cprint(f'\n🌙 MoonDev Zone Analysis for {symbol}', 'cyan', attrs=['bold'])
    cprint('=' * 50, 'cyan')
    
    cprint(f'📝 Contract: {symbol}', 'yellow')
    cprint(f'💰 Current Price: {price}', 'green')
    cprint(f'📊 Current Position: {position} tokens', 'magenta')
    cprint(f'💵 Position Value: ${current_value:.2f}', 'magenta')
    cprint(f'📈 Trend: {trend.upper()}', 'green' if trend == TREND_UP else 'red')
    
    position_pct = (current_value / CONFIG["POSITION_SIZE_USD"]) * 100
    cprint(f'📏 Position is at {position_pct:.1f}% of target size', 'yellow')
    
    cprint('\n🎯 Zone Ranges:', 'magenta')
    cprint(f'Demand Zone: {zones["dz"].min():.8f} - {zones["dz"].max():.8f}', 'blue')
    cprint(f'Supply Zone: {zones["sz"].min():.8f} - {zones["sz"].max():.8f}', 'red')
    cprint('=' * 50, 'cyan')

def handle_demand_zone_logic(symbol, trend, current_value, position_pct, price):
    """Handles trading logic when price is in the demand zone."""
    cprint(f'🎯 Price within demand zone range for {symbol[:4]}!', 'green', attrs=['bold'])
    
    if trend == TREND_UP and position_pct < (1 - CONFIG["BUFFER_PCT"]):
        if current_value < CONFIG["POSITION_SIZE_USD"]:
            amount_to_buy_usd = CONFIG["POSITION_SIZE_USD"] - current_value
            amount_to_buy_usdc = int(amount_to_buy_usd * 10**6)
            
            cprint(f'🛒 Need to buy ${amount_to_buy_usd:.2f} worth of tokens', 'green')
            try_position_entry(symbol, amount_to_buy_usdc)
        else:
            cprint(f'✅ Already at or above target position size', 'green')
    else:
        if trend != TREND_UP:
            cprint(f'⚠️ Not buying - Trend is not {TREND_UP} or below SMA buffer zone', 'yellow')
        else: # Implies position_pct >= (1 - CONFIG["BUFFER_PCT"])
            cprint(f'⚠️ Not buying - Position already near or at target size', 'yellow')

def handle_supply_zone_logic(symbol, trend, position, current_value, position_pct, sell_percentage):
    """Handles trading logic when price is in the supply zone."""
    cprint(f'🎯 Price within supply zone range for {symbol[:4]}!', 'red', attrs=['bold'])
    
    # Calculate expected value *after* the potential sell, considering a small buffer for verification
    # The core idea is that after selling `sell_percentage` of the `current_value`,
    # the remaining value should be `current_value * (1 - sell_percentage)`.
    # We allow a small buffer (e.g., 10% higher) for verification to account for price moves during sell.
    expected_value_after_sell = current_value * (1 - sell_percentage)
    verification_max_value = expected_value_after_sell * (1 + CONFIG["BUFFER_PCT"]) # e.g. 1.05 for 5% buffer during verification
    
    # We should sell if our current position percentage is significantly above
    # what it would be if we had already sold the `sell_percentage`.
    # Or, more simply, if the current position is substantial enough to warrant a sell.
    # The MINIMUM_POSITION_PCT check ensures we don't try to sell dust or very small amounts.
    # The comparison `position_pct > (CONFIG["MINIMUM_POSITION_PCT"] + BUFFER_PCT)` can be used.
    # Let's simplify the condition to primarily check if there is a position to sell
    # and if that position is meaningful.
    
    if position > 0 and current_value > (CONFIG["POSITION_SIZE_USD"] * CONFIG["MINIMUM_POSITION_PCT"]):
        cprint(f'💸 Selling {sell_percentage*100:.0f}% of position (Trend: {trend.upper()})', 'red')
        
        # Calculate sell_size in tokens
        # We want to sell `sell_percentage` of the current `position` (in tokens)
        sell_size_tokens = position * sell_percentage
        decimals = n.get_decimals(symbol)
        if decimals is None:
            cprint(f'❌ Could not get decimals for {symbol[:4]}. Cannot calculate sell size.', 'red')
            return
        sell_size_atomic = int(sell_size_tokens * (10 ** decimals))
        
        cprint(f'🧮 Sell size: {sell_size_tokens} tokens ({sell_size_atomic} atomic units)', 'yellow')
        cprint(f'📊 Expected position value after sell: ${expected_value_after_sell:.2f}', 'yellow')
        cprint(f'🔒 Verifying against max value: ${verification_max_value:.2f}', 'yellow')
        
        verify_and_sell(symbol, sell_size_atomic, expected_max_value=verification_max_value)
    else:
        if position == 0:
            cprint(f'❌ No position to sell for {symbol[:4]}', 'red')
        else:
            cprint(f'⚠️ Position value ${current_value:.2f} too small to trigger sell based on MINIMUM_POSITION_PCT or already sold down.', 'yellow')

def check_zone_and_trade(symbol):
    print(f'🌙 MoonDev checking {symbol[:4]}...')
    
    trend = check_trend(symbol)
    # Determine sell percentage based on trend
    if trend == TREND_UP:
        sell_percentage = CONFIG["SELL_PERCENTAGE_TRENDING_UP"]
    else: # TREND_DOWN or error (defaulting to down trend)
        sell_percentage = CONFIG["SELL_PERCENTAGE_TRENDING_DOWN"]
    
    position = n.get_position(symbol)
    price = n.token_price(symbol)

    if price is None: # Ensure price is fetched before using it
        cprint(f'❌ Could not get price for {symbol[:4]} in check_zone_and_trade. Skipping.', 'red')
        return
    current_value = position * price
    
    zones = n.supply_demand_zones(symbol, CONFIG["DAYS_BACK_4_DATA"], CONFIG["TIMEFRAME"])
    if zones is None:
        cprint(f'❌ Not enough data for S/D zones for {symbol[:4]}', 'red')
        return # Can't proceed without zones
    
    print_zone_info(symbol, zones, price, position, trend, current_value)
    
    demand_zone_min = zones['dz'].min()
    demand_zone_max = zones['dz'].max()
    supply_zone_min = zones['sz'].min()
    supply_zone_max = zones['sz'].max()
    
    # Calculate position percentage relative to target size
    position_pct = 0
    if CONFIG["POSITION_SIZE_USD"] > 0: # Avoid division by zero
        position_pct = current_value / CONFIG["POSITION_SIZE_USD"]
    
    if demand_zone_min <= price <= demand_zone_max:
        handle_demand_zone_logic(symbol, trend, current_value, position_pct, price)
    
    elif supply_zone_min <= price <= supply_zone_max:
        handle_supply_zone_logic(symbol, trend, position, current_value, position_pct, sell_percentage)
    
    else:
        cprint(f'⏳ Price not in any defined zone range for {symbol[:4]}', 'yellow')

def run_bot():
    try:
        cprint('\n' + '='*50, 'cyan')
        cprint(f'🌙 MoonDev Bot Run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 'cyan', attrs=['bold'])
        cprint('='*50, 'cyan')
        
        cprint('🚀 Supply/Demand Zone Bot Settings:', 'cyan', attrs=['bold'])
        cprint(f'💰 Target Position Size: ${CONFIG["POSITION_SIZE_USD"]}', 'cyan')
        cprint(f'📈 Sell Size Up Trend: {CONFIG["SELL_PERCENTAGE_TRENDING_UP"]*100}%', 'green')
        cprint(f'📉 Sell Size Down Trend: {CONFIG["SELL_PERCENTAGE_TRENDING_DOWN"]*100}%', 'red')
        cprint(f'📊 Using {CONFIG["SMA_BARS"]}-bar SMA for trend', 'cyan')
        cprint(f'🔒 Minimum Position: {CONFIG["MINIMUM_POSITION_PCT"]*100}%', 'yellow')
        cprint(f'📏 Position Buffer: {CONFIG["BUFFER_PCT"]*100}%', 'yellow')

        for symbol in CONFIG["SYMBOLS"]:
            try:
                check_zone_and_trade(symbol)
                time.sleep(5)  # Small delay between symbols
            except Exception as e:
                cprint(f'❌ Error processing symbol {symbol[:4]}: {str(e)}', 'red')
                continue  # Continue with next symbol even if one fails
                
    except Exception as e:
        cprint(f'❌ Major error in bot run: {str(e)}', 'red', attrs=['bold'])
        cprint('🔄 Bot will continue on next scheduled run', 'yellow')

if __name__ == "__main__":
    # Run immediately on start
    run_bot()
    
    # Schedule to run every 30 seconds
    schedule.every(30).seconds.do(run_bot)
    
    # Keep running forever
    while True:
        try:
            schedule.run_pending()
            time.sleep(3)
        except KeyboardInterrupt:
            break
        except Exception as e:
            cprint(f'❌ Error in main loop: {str(e)}', 'red')
            cprint('🔄 Restarting in 60 seconds...', 'yellow')
            time.sleep(60)  # Wait a minute before retrying on error