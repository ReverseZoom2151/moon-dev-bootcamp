import time
import asyncio, os, pytz, requests
from datetime import datetime
from termcolor import cprint

symbols = ['tBTCUSD', 'tETHUSD', 'tSOLUSD', 'tBNBUSD', 'tDOGEUSD']
base_url = 'https://api-pub.bitfinex.com/v2'
trades_filename = 'bitfinex_trades.csv'

if not os.path.isfile(trades_filename):
    with open(trades_filename, 'w') as f:
        f.write('Event Time, Symbol, Trade ID, Price, Amount, USD Size, Side, Is Large\n')

class BitfinexTradeAggregator:
    def __init__(self):
        self.trade_buckets = {}
        self.base_url = base_url
        self.last_trade_ids = {}  # Track last seen trade IDs to avoid duplicates

    async def get_recent_trades(self, symbol, limit=50):
        """Get recent trades from Bitfinex API"""
        try:
            url = f"{self.base_url}/trades/{symbol}/hist?limit={limit}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                trades = []
                
                for trade in data:
                    # Bitfinex trade format: [ID, MTS, AMOUNT, PRICE]
                    trade_id, timestamp, amount, price = trade[0], trade[1], float(trade[2]), float(trade[3])
                    
                    # Skip if we've already processed this trade
                    if symbol in self.last_trade_ids and trade_id <= self.last_trade_ids[symbol]:
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
                        'symbol': symbol
                    })
                
                # Update last seen trade ID
                if trades:
                    self.last_trade_ids[symbol] = max(trade['id'] for trade in trades)
                
                return trades
                
        except Exception as e:
            print(f"Error getting trades for {symbol}: {e}")
            return []

    async def add_trade(self, symbol, second, usd_size, is_buyer_maker):
        """Add trade to aggregation bucket"""
        trade_key = (symbol, second, is_buyer_maker)
        self.trade_buckets[trade_key] = self.trade_buckets.get(trade_key, 0) + usd_size

    async def check_and_print_trades(self):
        """Check aggregated trades and print large ones"""
        timestamp_now = datetime.utcnow().strftime("%H:%M:%S")
        deletions = []
        
        for trade_key, usd_size in self.trade_buckets.items():
            symbol, second, is_sell = trade_key
            
            # Only process trades from previous seconds
            if second < timestamp_now and usd_size > 500000:
                symbol_display = symbol.replace('t', '').replace('USD', '')
                attrs = ['bold']
                back_color = 'on_blue' if not is_sell else 'on_magenta'
                trade_type = "BUY" if not is_sell else 'SELL'
                
                if usd_size > 3000000:
                    usd_size_display = usd_size / 1000000
                    output = f"\033[5m🚨 {trade_type} {symbol_display} {second} ${usd_size_display:.2f}M\033[0m"
                    attrs.append('blink')
                else:
                    usd_size_display = usd_size / 1000000
                    output = f"💰 {trade_type} {symbol_display} {second} ${usd_size_display:.2f}M"
                
                cprint(output, 'white', back_color, attrs=attrs)
                deletions.append(trade_key)

        for key in deletions:
            del self.trade_buckets[key]

async def bitfinex_trade_stream(symbol, aggregator):
    """Monitor trades for a specific symbol"""
    print(f"🏛️  Starting Bitfinex trade monitor for {symbol}")
    
    while True:
        try:
            # Get recent trades
            trades = await aggregator.get_recent_trades(symbol)
            
            current_time = int(time.time() * 1000)
            recent_threshold = current_time - (10 * 1000)  # Last 10 seconds
            
            for trade in trades:
                # Only process very recent trades
                if trade['timestamp'] > recent_threshold:
                    usd_size = trade['usd_size']
                    
                    # Process individual large trades immediately
                    if usd_size > 500000:
                        est = pytz.timezone('Europe/Bucharest')
                        trade_time = datetime.fromtimestamp(trade['timestamp'] / 1000, est).strftime('%H:%M:%S')
                        symbol_display = symbol.replace('t', '').replace('USD', '')
                        
                        attrs = ['bold']
                        back_color = 'on_blue' if trade['side'] == 'BUY' else 'on_magenta'
                        
                        if usd_size > 5000000:
                            usd_size_display = usd_size / 1000000
                            output = f"\033[5m🚨🚨 {trade['side']} {symbol_display} {trade_time} ${usd_size_display:.2f}M 🚨🚨\033[0m"
                            attrs.append('blink')
                        elif usd_size > 2000000:
                            usd_size_display = usd_size / 1000000
                            output = f"🚨 {trade['side']} {symbol_display} {trade_time} ${usd_size_display:.2f}M"
                        else:
                            usd_size_display = usd_size / 1000
                            output = f"💰 {trade['side']} {symbol_display} {trade_time} ${usd_size_display:.0f}K"
                        
                        cprint(output, 'white', back_color, attrs=attrs)
                        
                        # Log to CSV
                        with open(trades_filename, 'a') as f:
                            f.write(f"{trade_time}, {symbol}, {trade['id']}, {trade['price']}, {trade['amount']}, {usd_size}, {trade['side']}, True\n")
                    
                    # Add to aggregator for second-by-second analysis
                    second = datetime.fromtimestamp(trade['timestamp'] / 1000).strftime("%H:%M:%S")
                    is_sell = trade['side'] == 'SELL'
                    await aggregator.add_trade(symbol, second, usd_size, is_sell)
            
        except Exception as e:
            print(f"Error in trade stream for {symbol}: {e}")
            
        await asyncio.sleep(3)  # Check every 3 seconds

async def print_aggregated_trades_every_second(aggregator):
    """Print aggregated trades every second"""
    while True:
        try:
            await aggregator.check_and_print_trades()
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in aggregated trades: {e}")
            await asyncio.sleep(1)

async def display_startup_info():
    """Display startup information"""
    cprint("🏛️  Bitfinex Huge Trades Monitor", 'cyan', attrs=['bold'])
    cprint("=" * 50, 'cyan')
    print()
    cprint("📊 Monitoring large trades across major pairs", 'white')
    cprint("💰 >$500K: Large trades", 'white', 'on_blue')
    cprint("🚨 >$2M: Huge trades", 'white', 'on_magenta')
    cprint("🚨🚨 >$5M: Massive trades", 'white', 'on_red')
    print()
    cprint("Note: Bitfinex provides historical trade data", 'yellow')
    cprint("Real-time WebSocket feeds require authentication", 'yellow')
    print()

async def monitor_market_impact():
    """Monitor for potential market impact from large trades"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Could add logic here to correlate large trades with price movements
            # This would require price data correlation
            
        except Exception as e:
            print(f"Error in market impact monitor: {e}")

async def main():
    # Display startup info
    await display_startup_info()
    
    # Create trade aggregator
    trade_aggregator = BitfinexTradeAggregator()
    
    # Start monitoring tasks
    trade_tasks = [bitfinex_trade_stream(symbol, trade_aggregator) for symbol in symbols]
    aggregator_task = print_aggregated_trades_every_second(trade_aggregator)
    impact_task = monitor_market_impact()
    
    try:
        await asyncio.gather(*trade_tasks, aggregator_task, impact_task)
    except KeyboardInterrupt:
        cprint("\n🛑 Bitfinex huge trades monitor stopped", 'yellow')

if __name__ == "__main__":
    asyncio.run(main())
