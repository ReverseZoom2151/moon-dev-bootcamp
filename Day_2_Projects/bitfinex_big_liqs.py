import time
import asyncio, os, pytz, requests
from datetime import datetime
from termcolor import cprint

# Bitfinex doesn't have direct liquidation feeds like Binance
# We'll monitor for large orders and unusual price movements as proxies
symbols = ['tBTCUSD', 'tETHUSD', 'tSOLUSD', 'tDOGEUSD', 'tBNBUSD']
filename = 'bitfinex_bigliqs.csv'

if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(",".join([
            'symbol',
            'side',
            'order_type',
            'price',
            'amount',
            'timestamp',
            'usd_size',
            'estimated_liquidation'
        ]) + "\n")

class BitfinexLiquidationMonitor:
    def __init__(self):
        self.base_url = 'https://api-pub.bitfinex.com/v2'
        self.price_cache = {}
        self.order_cache = {}
        
    async def get_ticker(self, symbol):
        """Get current ticker data"""
        try:
            url = f"{self.base_url}/ticker/{symbol}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Bitfinex ticker format: [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, DAILY_CHANGE_RELATIVE, LAST_PRICE, VOLUME, HIGH, LOW]
                return {
                    'last_price': float(data[6]),
                    'volume': float(data[7]),
                    'daily_change_perc': float(data[5]) * 100,
                    'bid': float(data[0]),
                    'ask': float(data[2])
                }
        except Exception as e:
            print(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def get_order_book(self, symbol):
        """Get order book data to detect large orders"""
        try:
            url = f"{self.base_url}/book/{symbol}/P0"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Bitfinex order book format: [PRICE, COUNT, AMOUNT]
                # Positive amount = bid, negative amount = ask
                bids = []
                asks = []
                
                for entry in data:
                    price, count, amount = float(entry[0]), int(entry[1]), float(entry[2])
                    if amount > 0:
                        bids.append({'price': price, 'amount': amount, 'count': count})
                    else:
                        asks.append({'price': price, 'amount': abs(amount), 'count': count})
                
                return {'bids': bids[:20], 'asks': asks[:20]}
        except Exception as e:
            print(f"Error getting order book for {symbol}: {e}")
            return None
    
    async def get_recent_trades(self, symbol):
        """Get recent trades to detect large liquidation-like trades"""
        try:
            url = f"{self.base_url}/trades/{symbol}/hist?limit=100"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Bitfinex trades format: [ID, MTS, AMOUNT, PRICE]
                trades = []
                for trade in data:
                    trade_id, timestamp, amount, price = trade[0], trade[1], float(trade[2]), float(trade[3])
                    usd_size = abs(amount) * price
                    side = 'SELL' if amount < 0 else 'BUY'
                    
                    trades.append({
                        'id': trade_id,
                        'timestamp': timestamp,
                        'amount': abs(amount),
                        'price': price,
                        'side': side,
                        'usd_size': usd_size
                    })
                
                return trades
        except Exception as e:
            print(f"Error getting trades for {symbol}: {e}")
            return None
    
    async def detect_potential_liquidations(self, symbol):
        """Detect potential liquidations based on trade patterns and order book analysis"""
        try:
            # Get recent trades
            trades = await self.get_recent_trades(symbol)
            if not trades:
                return
            
            # Get order book
            order_book = await self.get_order_book(symbol)
            if not order_book:
                return
            
            # Get ticker for context
            ticker = await self.get_ticker(symbol)
            if not ticker:
                return
            
            current_time = int(time.time() * 1000)
            recent_threshold = current_time - (60 * 1000)  # Last 1 minute
            
            # Analyze recent large trades
            large_trades = []
            for trade in trades:
                if trade['timestamp'] > recent_threshold and trade['usd_size'] > 100000:
                    large_trades.append(trade)
            
            # Check for liquidation patterns
            for trade in large_trades:
                est = pytz.timezone('Europe/Bucharest')
                trade_time = datetime.fromtimestamp(trade['timestamp'] / 1000, est).strftime('%H:%M:%S')
                
                # Estimate if this could be a liquidation based on:
                # 1. Large size
                # 2. Rapid execution
                # 3. Price impact
                
                liquidation_probability = 0
                
                # Size factor
                if trade['usd_size'] > 500000:
                    liquidation_probability += 0.4
                elif trade['usd_size'] > 250000:
                    liquidation_probability += 0.3
                else:
                    liquidation_probability += 0.2
                
                # Check for multiple large trades in same direction (cascade effect)
                same_direction_trades = [t for t in large_trades 
                                       if t['side'] == trade['side'] and 
                                       abs(t['timestamp'] - trade['timestamp']) < 30000]  # Within 30 seconds
                
                if len(same_direction_trades) > 1:
                    liquidation_probability += 0.3
                
                # Check price impact vs order book depth
                if order_book['bids'] and order_book['asks']:
                    best_bid = order_book['bids'][0]['price']
                    best_ask = order_book['asks'][0]['price']
                    
                    if trade['side'] == 'SELL' and trade['price'] < best_bid * 0.999:
                        liquidation_probability += 0.2
                    elif trade['side'] == 'BUY' and trade['price'] > best_ask * 1.001:
                        liquidation_probability += 0.2
                
                # Display if high probability liquidation
                if liquidation_probability > 0.5:
                    symbol_display = symbol.replace('t', '').replace('USD', '')
                    liquidation_type = 'L LIQ' if trade['side'] == 'SELL' else 'S LIQ'
                    
                    usd_size_display = trade['usd_size']
                    color = 'blue' if trade['side'] == 'SELL' else 'magenta'
                    attrs = ['bold'] if trade['usd_size'] > 500000 else []
                    
                    if trade['usd_size'] > 1000000:
                        usd_size_display = trade['usd_size'] / 1000000
                        output = f"🚨 {liquidation_type} {symbol_display} {trade_time} ${usd_size_display:.2f}M (Est. {liquidation_probability:.0%})"
                        attrs.append('blink')
                    else:
                        output = f"{liquidation_type} {symbol_display} {trade_time} ${usd_size_display:.0f} (Est. {liquidation_probability:.0%})"
                    
                    cprint(output, 'white', f'on_{color}', attrs=attrs)
                    
                    # Log to CSV
                    with open(filename, 'a') as f:
                        f.write(f"{symbol},{trade['side']},MARKET,{trade['price']},{trade['amount']},{trade['timestamp']},{trade['usd_size']},{liquidation_probability}\n")
            
        except Exception as e:
            print(f"Error detecting liquidations for {symbol}: {e}")

async def monitor_liquidations():
    """Main monitoring loop"""
    monitor = BitfinexLiquidationMonitor()
    
    cprint("🏛️  Starting Bitfinex Big Liquidations Monitor", 'cyan', attrs=['bold'])
    cprint("Note: Bitfinex doesn't provide direct liquidation feeds", 'yellow')
    cprint("Monitoring large trades and unusual patterns as liquidation proxies", 'yellow')
    print()
    
    while True:
        try:
            # Monitor each symbol
            tasks = [monitor.detect_potential_liquidations(symbol) for symbol in symbols]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Wait before next check
            await asyncio.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            cprint("\n🛑 Bitfinex liquidation monitor stopped", 'yellow')
            break
        except Exception as e:
            cprint(f"Error in main loop: {e}", 'red')
            await asyncio.sleep(5)

async def main():
    await monitor_liquidations()

if __name__ == "__main__":
    asyncio.run(main())
