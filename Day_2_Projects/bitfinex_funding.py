import asyncio, requests
from datetime import datetime
from termcolor import cprint

symbols = ['tBTCUSD', 'tETHUSD', 'tSOLUSD', 'tDOGEUSD']
base_url = 'https://api-pub.bitfinex.com/v2'

shared_symbol_counter = {'count': 0}
print_lock = asyncio.Lock()

class BitfinexFundingMonitor:
    def __init__(self):
        self.base_url = base_url
        
    async def get_funding_stats(self, symbol):
        """Get funding statistics for a symbol"""
        try:
            # Get current funding rate
            url = f"{self.base_url}/stats1/{symbol}/funding.yield:p30:a365/last"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Bitfinex stats format: [MTS, VALUE]
                if data and len(data) >= 2:
                    annual_yield = float(data[1]) * 100  # Convert to percentage
                    return annual_yield
                    
        except Exception as e:
            print(f"Error getting funding for {symbol}: {e}")
            
        return None
    
    async def get_funding_history(self, symbol):
        """Get recent funding history for additional context"""
        try:
            url = f"{self.base_url}/stats1/{symbol}/funding.yield:p30:a365/hist?limit=10"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                rates = []
                for entry in data:
                    if len(entry) >= 2:
                        timestamp, rate = entry[0], float(entry[1]) * 100
                        rates.append({'timestamp': timestamp, 'rate': rate})
                return rates
                
        except Exception as e:
            print(f"Error getting funding history for {symbol}: {e}")
            
        return []

async def bitfinex_funding_stream(symbol, shared_counter):
    """Monitor funding rates for a symbol"""
    global print_lock
    monitor = BitfinexFundingMonitor()
    
    while True:
        try:
            async with print_lock:
                # Get current funding rate
                annual_yield = await monitor.get_funding_stats(symbol)
                
                if annual_yield is not None:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    symbol_display = symbol.replace('t', '').replace('USD', '')
                    
                    # Color coding based on funding rate levels
                    if annual_yield > 50:
                        text_color, back_color = 'white', 'on_red'
                        status = '🔥 EXTREME'
                    elif annual_yield > 30:
                        text_color, back_color = 'white', 'on_yellow'  
                        status = '⚡ HIGH'
                    elif annual_yield > 15:
                        text_color, back_color = 'black', 'on_cyan'
                        status = '📈 ELEVATED'
                    elif annual_yield > 5:
                        text_color, back_color = 'white', 'on_green'
                        status = '💰 POSITIVE'
                    elif annual_yield > 0:
                        text_color, back_color = 'white', 'on_light_green'
                        status = '✅ LOW+'
                    elif annual_yield > -10:
                        text_color, back_color = 'black', 'on_light_cyan'
                        status = '💎 NEGATIVE'
                    else:
                        text_color, back_color = 'white', 'on_blue'
                        status = '🌊 DEEP-'
                    
                    # Get recent trend
                    history = await monitor.get_funding_history(symbol)
                    trend = ""
                    if len(history) >= 2:
                        recent_rates = [h['rate'] for h in history[:3]]
                        if all(recent_rates[i] > recent_rates[i+1] for i in range(len(recent_rates)-1)):
                            trend = " 📈"
                        elif all(recent_rates[i] < recent_rates[i+1] for i in range(len(recent_rates)-1)):
                            trend = " 📉"
                        else:
                            trend = " ➡️"
                    
                    output = f"{status} {symbol_display} funding: {annual_yield:.2f}%{trend}"
                    cprint(output, text_color, back_color)
                    
                    shared_counter['count'] += 1
                    
                    if shared_counter['count'] >= len(symbols):
                        cprint(f"🏛️  {current_time} Bitfinex Annual Funding Update", 'white', 'on_black')
                        shared_counter['count'] = 0
                        print()  # Add spacing between cycles
                
                else:
                    print(f"⚠️  Failed to get funding data for {symbol}")
                    
        except Exception as e:
            print(f"Error in funding stream for {symbol}: {e}")
            
        await asyncio.sleep(10)  # Wait 10 seconds between updates

async def display_funding_info():
    """Display initial information about Bitfinex funding"""
    cprint("🏛️  Bitfinex Funding Rates Monitor", 'cyan', attrs=['bold'])
    cprint("=" * 50, 'cyan')
    print()
    cprint("📊 Monitoring annual funding yields for major pairs", 'white')
    cprint("🔥 >50%: Extreme rates", 'white', 'on_red')
    cprint("⚡ >30%: High rates", 'white', 'on_yellow')
    cprint("📈 >15%: Elevated rates", 'black', 'on_cyan')
    cprint("💰 >5%: Positive rates", 'white', 'on_green')
    cprint("✅ >0%: Low positive", 'white', 'on_light_green')
    cprint("💎 <0%: Negative rates", 'black', 'on_light_cyan')
    cprint("🌊 <-10%: Deep negative", 'white', 'on_blue')
    print()
    cprint("📈📉➡️ Trend indicators show recent direction", 'yellow')
    print()

async def monitor_funding_with_alerts():
    """Enhanced monitoring with alerts for extreme rates"""
    monitor = BitfinexFundingMonitor()
    alert_thresholds = {'extreme_high': 50, 'high': 30, 'extreme_low': -20}
    
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute for alerts
            
            for symbol in symbols:
                rate = await monitor.get_funding_stats(symbol)
                if rate is not None:
                    symbol_display = symbol.replace('t', '').replace('USD', '')
                    
                    if rate > alert_thresholds['extreme_high']:
                        cprint(f"🚨 FUNDING ALERT: {symbol_display} at {rate:.2f}% (EXTREME HIGH)", 'red', attrs=['bold', 'blink'])
                    elif rate > alert_thresholds['high']:
                        cprint(f"⚠️  FUNDING ALERT: {symbol_display} at {rate:.2f}% (HIGH)", 'yellow', attrs=['bold'])
                    elif rate < alert_thresholds['extreme_low']:
                        cprint(f"💎 FUNDING ALERT: {symbol_display} at {rate:.2f}% (EXTREME NEGATIVE)", 'blue', attrs=['bold'])
                        
        except Exception as e:
            print(f"Error in funding alerts: {e}")

async def main():
    # Display initial info
    await display_funding_info()
    
    # Start funding streams and alert system
    funding_tasks = [bitfinex_funding_stream(symbol, shared_symbol_counter) for symbol in symbols]
    alert_task = monitor_funding_with_alerts()
    
    try:
        await asyncio.gather(*funding_tasks, alert_task)
    except KeyboardInterrupt:
        cprint("\n🛑 Bitfinex funding monitor stopped", 'yellow')

if __name__ == "__main__":
    asyncio.run(main())
