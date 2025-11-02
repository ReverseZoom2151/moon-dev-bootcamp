import time
import asyncio, os, pytz, requests
from datetime import datetime
from termcolor import cprint

# Bitfinex symbols to monitor
symbols = ['tBTCUSD', 'tETHUSD', 'tSOLUSD', 'tDOGEUSD', 'tBNBUSD']
base_url = 'https://api-pub.bitfinex.com/v2'
filename = 'bitfinex_liqs.csv'

if not os.path.isfile(filename):
    with open(filename, 'w') as f:
        f.write(",".join([
            'symbol',
            'side',
            'order_type',
            'time_in_force',
            'original_quantity',
            'price',
            'average_price',
            'order_status',
            'order_last_filled_quantity',
            'order_filled_accumulated_quantity',
            'order_trade_time',
            'usd_size',
            'liquidation_probability'
        ]) + "\n")

class BitfinexLiquidationDetector:
    def __init__(self):
        self.base_url = base_url
        self.price_history = {}
        self.volume_history = {}
        self.last_trade_ids = {}
        
    async def get_ticker(self, symbol):
        """Get current market data"""
        try:
            url = f"{self.base_url}/ticker/{symbol}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # [BID, BID_SIZE, ASK, ASK_SIZE, DAILY_CHANGE, DAILY_CHANGE_RELATIVE, LAST_PRICE, VOLUME, HIGH, LOW]
                return {
                    'last_price': float(data[6]),
                    'volume': float(data[7]),
                    'bid': float(data[0]),
                    'ask': float(data[2]),
                    'daily_change_perc': float(data[5]) * 100
                }
        except Exception as e:
            print(f"Error getting ticker for {symbol}: {e}")
        return None
    
    async def get_recent_trades(self, symbol, limit=100):
        """Get recent trades"""
        try:
            url = f"{self.base_url}/trades/{symbol}/hist?limit={limit}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                trades = []
                for trade in data:
                    # [ID, MTS, AMOUNT, PRICE]
                    trade_id, timestamp, amount, price = trade[0], trade[1], float(trade[2]), float(trade[3])
                    
                    # Skip already processed trades
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
                        'usd_size': usd_size
                    })
                
                if trades:
                    self.last_trade_ids[symbol] = max(trade['id'] for trade in trades)
                
                return trades
        except Exception as e:
            print(f"Error getting trades for {symbol}: {e}")
        return []
    
    async def calculate_liquidation_probability(self, trade, symbol, ticker):
        """Calculate probability that a trade is a liquidation"""
        probability = 0.0
        
        # Size-based scoring
        if trade['usd_size'] > 100000:
            probability += 0.3
        elif trade['usd_size'] > 50000:
            probability += 0.2
        elif trade['usd_size'] > 10000:
            probability += 0.1
        else:
            return 0.0  # Too small to be considered
        
        # Price impact analysis
        if ticker:
            mid_price = (ticker['bid'] + ticker['ask']) / 2
            price_deviation = abs(trade['price'] - mid_price) / mid_price
            
            if price_deviation > 0.005:  # >0.5% from mid
                probability += 0.2
            elif price_deviation > 0.002:  # >0.2% from mid
                probability += 0.1
        
        # Volume analysis (if available)
        if ticker and ticker['volume'] > 0:
            volume_ratio = trade['usd_size'] / (ticker['volume'] * ticker['last_price'])
            if volume_ratio > 0.01:  # >1% of daily volume
                probability += 0.2
        
        # Time-based clustering (rapid succession of similar trades)
        recent_trades = await self.get_recent_trades(symbol, 20)
        current_time = trade['timestamp']
        time_window = 30000  # 30 seconds
        
        similar_trades = [
            t for t in recent_trades 
            if (abs(t['timestamp'] - current_time) < time_window and 
                t['side'] == trade['side'] and 
                t['usd_size'] > 5000)
        ]
        
        if len(similar_trades) > 3:
            probability += 0.3
        elif len(similar_trades) > 1:
            probability += 0.1
        
        return min(probability, 1.0)  # Cap at 100%

async def bitfinex_liquidation_monitor(symbol):
    """Monitor potential liquidations for a symbol"""
    detector = BitfinexLiquidationDetector()
    print(f"🏛️  Starting liquidation monitor for {symbol}")
    
    while True:
        try:
            # Get market data
            ticker = await detector.get_ticker(symbol)
            if not ticker:
                await asyncio.sleep(5)
                continue
            
            # Get recent trades
            trades = await detector.get_recent_trades(symbol)
            current_time = int(time.time() * 1000)
            recent_threshold = current_time - (60 * 1000)  # Last 1 minute
            
            for trade in trades:
                if trade['timestamp'] > recent_threshold and trade['usd_size'] > 3000:
                    # Calculate liquidation probability
                    liq_prob = await detector.calculate_liquidation_probability(trade, symbol, ticker)
                    
                    if liq_prob > 0.3:  # 30% threshold for display
                        est = pytz.timezone('Europe/Bucharest')
                        time_est = datetime.fromtimestamp(trade['timestamp'] / 1000, est).strftime('%H:%M:%S')
                        
                        symbol_display = symbol.replace('t', '').replace('USD', '')[:4]
                        liquidation_type = 'L LIQ' if trade['side'] == 'SELL' else 'S LIQ'
                        
                        usd_size = trade['usd_size']
                        color = 'green' if trade['side'] == 'SELL' else 'red'
                        attrs = ['bold'] if usd_size > 10000 else []
                        
                        if liq_prob > 0.7:  # High probability
                            stars = '***'
                            attrs.append('blink')
                            output = f'{stars} {liquidation_type} {symbol_display} {time_est} ${usd_size:.0f} ({liq_prob:.0%})'
                        elif liq_prob > 0.5:  # Medium probability
                            stars = '**'
                            output = f'{stars} {liquidation_type} {symbol_display} {time_est} ${usd_size:.0f} ({liq_prob:.0%})'
                        else:  # Lower probability but still notable
                            stars = '*'
                            output = f'{stars} {liquidation_type} {symbol_display} {time_est} ${usd_size:.0f} ({liq_prob:.0%})'
                        
                        cprint(output, 'white', f'on_{color}', attrs=attrs)
                        
                        # Log to CSV
                        with open(filename, 'a') as f:
                            f.write(f"{symbol},{trade['side']},MARKET,,{trade['amount']},{trade['price']},,FILLED,{trade['amount']},{trade['amount']},{trade['timestamp']},{usd_size},{liq_prob}\n")
            
        except Exception as e:
            print(f"Error monitoring {symbol}: {e}")
            
        await asyncio.sleep(8)  # Check every 8 seconds

async def display_liquidation_info():
    """Display information about liquidation detection"""
    cprint("🏛️  Bitfinex Liquidation Detection System", 'cyan', attrs=['bold'])
    cprint("=" * 50, 'cyan')
    print()
    cprint("📊 Detecting potential liquidations through trade analysis", 'white')
    cprint("⭐ * = 30-50% probability", 'yellow')
    cprint("⭐ ** = 50-70% probability", 'orange')  
    cprint("⭐ *** = 70%+ probability", 'red', attrs=['bold'])
    print()
    cprint("🔍 Analysis factors:", 'white')
    cprint("  • Trade size relative to volume", 'cyan')
    cprint("  • Price impact vs order book", 'cyan')
    cprint("  • Time clustering patterns", 'cyan')
    cprint("  • Market stress indicators", 'cyan')
    print()
    cprint("Note: Bitfinex doesn't provide direct liquidation feeds", 'yellow')
    cprint("This system uses heuristics to estimate liquidation probability", 'yellow')
    print()

async def monitor_market_stress():
    """Monitor overall market stress indicators"""
    detector = BitfinexLiquidationDetector()
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            stress_levels = {}
            total_volume = 0
            total_volatility = 0
            
            for symbol in symbols:
                ticker = await detector.get_ticker(symbol)
                if ticker:
                    daily_change = abs(ticker['daily_change_perc'])
                    volume = ticker['volume'] * ticker['last_price']
                    
                    stress_levels[symbol] = {
                        'volatility': daily_change,
                        'volume': volume
                    }
                    
                    total_volume += volume
                    total_volatility += daily_change
            
            if stress_levels:
                avg_volatility = total_volatility / len(stress_levels)
                
                if avg_volatility > 15:  # High volatility across markets
                    cprint(f"⚠️  HIGH MARKET STRESS: {avg_volatility:.1f}% avg volatility", 'red', attrs=['bold'])
                elif avg_volatility > 8:
                    cprint(f"📈 Elevated market volatility: {avg_volatility:.1f}%", 'yellow')
                
        except Exception as e:
            print(f"Error in market stress monitor: {e}")

async def main():
    # Display startup info
    await display_liquidation_info()
    
    # Start monitoring tasks
    liquidation_tasks = [bitfinex_liquidation_monitor(symbol) for symbol in symbols]
    stress_task = monitor_market_stress()
    
    try:
        await asyncio.gather(*liquidation_tasks, stress_task)
    except KeyboardInterrupt:
        cprint("\n🛑 Bitfinex liquidation monitor stopped", 'yellow')

if __name__ == "__main__":
    asyncio.run(main())
