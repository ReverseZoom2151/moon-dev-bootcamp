import time
import asyncio, os, pytz, requests
from datetime import datetime
from termcolor import cprint

symbols = ['tBTCUSD', 'tETHUSD', 'tSOLUSD', 'tDOGEUSD', 'tBNBUSD']
base_url = 'https://api-pub.bitfinex.com/v2'
trades_filename = 'bitfinex_trades.csv'

if not os.path.exists(trades_filename):
    with open(trades_filename, 'w') as f:
        f.write('Event Time, Symbol, Trade ID, Price, Amount, USD Size, Side, Is Large\n')

class BitfinexTradeMonitor:
    def __init__(self):
        self.base_url = base_url
        self.last_trade_ids = {}
        self.processed_trades = set()
        
    async def get_recent_trades(self, symbol, limit=50):
        """Get recent trades from Bitfinex"""
        try:
            url = f"{self.base_url}/trades/{symbol}/hist?limit={limit}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                trades = []
                
                for trade in data:
                    # Bitfinex format: [ID, MTS, AMOUNT, PRICE]
                    trade_id, timestamp, amount, price = trade[0], trade[1], float(trade[2]), float(trade[3])
                    
                    # Create unique identifier for this trade
                    trade_key = f"{symbol}_{trade_id}"
                    
                    # Skip if already processed
                    if trade_key in self.processed_trades:
                        continue
                    
                    # Skip old trades (only process recent ones)
                    current_time = int(time.time() * 1000)
                    if timestamp < current_time - (300 * 1000):  # Only last 5 minutes
                        continue
                    
                    usd_size = abs(amount) * price
                    side = 'SELL' if amount < 0 else 'BUY'
                    
                    trades.append({
                        'id': trade_id,
                        'timestamp': timestamp,
                        'amount': abs(amount),
                        'price': price,
                        'side': side,
                        'usd_size': usd_size,
                        'symbol': symbol,
                        'key': trade_key
                    })
                    
                    # Mark as processed
                    self.processed_trades.add(trade_key)
                
                # Clean old processed trades to prevent memory buildup
                if len(self.processed_trades) > 10000:
                    # Remove oldest half
                    old_trades = list(self.processed_trades)[:5000]
                    for old_trade in old_trades:
                        self.processed_trades.discard(old_trade)
                
                return trades
                
        except Exception as e:
            print(f"Error getting trades for {symbol}: {e}")
            return []

async def bitfinex_trade_stream(symbol, monitor):
    """Monitor trades for a specific symbol"""
    print(f"🏛️  Starting Bitfinex trade monitor for {symbol}")
    
    while True:
        try:
            trades = await monitor.get_recent_trades(symbol)
            
            for trade in trades:
                usd_size = trade['usd_size']
                
                # Only show trades above threshold
                if usd_size > 14999:
                    est_time = pytz.timezone('Europe/Bucharest')
                    readable_trade_time = datetime.fromtimestamp(trade['timestamp'] / 1000, est_time).strftime('%H:%M:%S')
                    
                    display_symbol = symbol.replace('t', '').replace('USD', '')
                    trade_type = trade['side']
                    color = 'red' if trade_type == 'SELL' else 'green'
                    stars = ''
                    attrs = ['bold'] if usd_size >= 50000 else []
                    repeat_count = 1
                    
                    # Enhanced categorization
                    if usd_size >= 2000000:  # $2M+
                        stars = '🚨🚨🚨'
                        attrs.extend(['blink'])
                        color = 'magenta' if trade_type == 'SELL' else 'blue'
                        repeat_count = 2
                    elif usd_size >= 1000000:  # $1M+
                        stars = '🚨🚨'
                        attrs.append('blink')
                        color = 'magenta' if trade_type == 'SELL' else 'blue'
                        repeat_count = 1
                    elif usd_size >= 500000:  # $500k+
                        stars = '🚨*'
                        color = 'magenta' if trade_type == 'SELL' else 'blue'
                    elif usd_size >= 100000:  # $100k+
                        stars = '**'
                        attrs.append('bold')
                    elif usd_size >= 50000:  # $50k+
                        stars = '*'
                    
                    # Enhanced output format
                    if usd_size >= 1000000:
                        size_display = f"${usd_size/1000000:.2f}M"
                    elif usd_size >= 1000:
                        size_display = f"${usd_size/1000:.0f}K"
                    else:
                        size_display = f"${usd_size:,.0f}"
                    
                    output = f"{stars} {trade_type} {display_symbol} {readable_trade_time} {size_display}"
                    
                    # Print multiple times for very large trades
                    for _ in range(repeat_count):
                        cprint(output, 'white', f'on_{color}', attrs=attrs)
                        if repeat_count > 1:
                            await asyncio.sleep(0.2)  # Brief pause between repeats
                    
                    # Log to CSV
                    with open(trades_filename, 'a') as f:
                        is_large = usd_size >= 50000
                        f.write(f"{readable_trade_time}, {symbol}, {trade['id']}, {trade['price']}, {trade['amount']}, {usd_size}, {trade_type}, {is_large}\n")
            
        except Exception as e:
            print(f"Error in trade stream for {symbol}: {e}")
            
        await asyncio.sleep(5)  # Check every 5 seconds

async def display_trade_info():
    """Display startup information"""
    cprint("🏛️  Bitfinex Recent Trades Monitor", 'cyan', attrs=['bold'])
    cprint("=" * 50, 'cyan')
    print()
    cprint("📊 Monitoring significant trades across major pairs", 'white')
    cprint("Minimum size: $15,000", 'white')
    print()
    cprint("🎯 Trade Size Categories:", 'white')
    cprint("  * $15K-$50K: Standard large trades", 'cyan')
    cprint("  ** $50K-$100K: Very large trades", 'yellow')
    cprint("  *** $100K-$500K: Huge trades", 'orange')
    cprint("  🚨* $500K-$1M: Massive trades", 'red')
    cprint("  🚨🚨 $1M-$2M: Institutional trades", 'magenta')
    cprint("  🚨🚨🚨 $2M+: Whale trades", 'red', attrs=['bold'])
    print()
    cprint("💹 Colors:", 'white')
    cprint("  🟢 GREEN: BUY orders", 'green')
    cprint("  🔴 RED: SELL orders", 'red')
    cprint("  🟣 PURPLE/BLUE: Mega trades", 'magenta')
    print()

async def market_summary_task():
    """Provide periodic market summary"""
    monitor = BitfinexTradeMonitor()
    
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            
            current_time = datetime.now().strftime("%H:%M:%S")
            cprint(f"\n📊 {current_time} - Market Activity Summary", 'cyan', attrs=['bold'])
            
            total_volume = 0
            active_symbols = 0
            
            for symbol in symbols:
                trades = await monitor.get_recent_trades(symbol, 20)
                recent_volume = sum(trade['usd_size'] for trade in trades)
                
                if recent_volume > 0:
                    active_symbols += 1
                    total_volume += recent_volume
                    symbol_display = symbol.replace('t', '').replace('USD', '')
                    
                    if recent_volume > 5000000:  # $5M+
                        cprint(f"  🔥 {symbol_display}: ${recent_volume/1000000:.1f}M volume", 'red', attrs=['bold'])
                    elif recent_volume > 1000000:  # $1M+
                        cprint(f"  📈 {symbol_display}: ${recent_volume/1000000:.1f}M volume", 'yellow')
                    elif recent_volume > 100000:  # $100K+
                        cprint(f"  💰 {symbol_display}: ${recent_volume/1000:.0f}K volume", 'green')
            
            if total_volume > 0:
                cprint(f"🌊 Total 5min volume: ${total_volume/1000000:.1f}M across {active_symbols} pairs\n", 'white', 'on_blue')
            else:
                cprint("😴 Market quiet - low activity detected\n", 'yellow')
                
        except Exception as e:
            print(f"Error in market summary: {e}")

async def detect_unusual_activity():
    """Detect unusual trading activity patterns"""
    monitor = BitfinexTradeMonitor()
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            for symbol in symbols:
                trades = await monitor.get_recent_trades(symbol, 30)
                
                if len(trades) > 0:
                    # Check for rapid large trades
                    large_trades = [t for t in trades if t['usd_size'] > 100000]
                    
                    if len(large_trades) >= 3:
                        symbol_display = symbol.replace('t', '').replace('USD', '')
                        total_volume = sum(t['usd_size'] for t in large_trades)
                        
                        cprint(f"⚡ ACTIVITY SPIKE: {symbol_display} - {len(large_trades)} large trades, ${total_volume/1000000:.1f}M volume", 
                               'yellow', 'on_black', attrs=['bold'])
                    
                    # Check for unusual price gaps (potential liquidations)
                    if len(trades) >= 5:
                        prices = [t['price'] for t in trades[:5]]
                        price_range = (max(prices) - min(prices)) / min(prices)
                        
                        if price_range > 0.02:  # >2% price range in recent trades
                            symbol_display = symbol.replace('t', '').replace('USD', '')
                            cprint(f"💥 PRICE VOLATILITY: {symbol_display} - {price_range*100:.1f}% range in recent trades", 
                                   'red', attrs=['bold'])
                
        except Exception as e:
            print(f"Error in unusual activity detection: {e}")

async def main():
    # Display startup info
    await display_trade_info()
    
    # Create monitor instance
    monitor = BitfinexTradeMonitor()
    
    # Start monitoring tasks
    trade_tasks = [bitfinex_trade_stream(symbol, monitor) for symbol in symbols]
    summary_task = market_summary_task()
    activity_task = detect_unusual_activity()
    
    try:
        await asyncio.gather(*trade_tasks, summary_task, activity_task)
    except KeyboardInterrupt:
        cprint("\n🛑 Bitfinex recent trades monitor stopped", 'yellow')

if __name__ == "__main__":
    asyncio.run(main())
