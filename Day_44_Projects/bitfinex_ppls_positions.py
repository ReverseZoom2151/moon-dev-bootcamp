'''
BITFINEX PROFESSIONAL WHALE TRACKER - Institutional Large Holder Analysis System
Professional-grade tracking and analysis of institutional holders and significant trading activity on Bitfinex
'''

import os
import json
import time
import colorama
import argparse
import sys
import pandas as pd
import bitfinex_nice_funcs as bfx
from datetime import datetime
from tqdm import tqdm
from colorama import Fore

sys.path.append('..')

# Initialize professional colorama
colorama.init(autoreset=True)

# Professional pandas configuration
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

# ===== BITFINEX PROFESSIONAL CONFIGURATION =====
PROFESSIONAL_DATA_DIR = "bitfinex_professional_whale_data"
INSTITUTIONAL_THRESHOLD_USD = 250000  # $250k+ positions considered institutional
WHALE_THRESHOLD_USD = 1000000  # $1M+ positions considered whale-tier
PROFESSIONAL_TRADE_THRESHOLD_USD = 100000  # $100k+ trades flagged as professional
TOP_N_INSTITUTIONAL = 25
TOP_N_POSITIONS = 15
UPDATE_INTERVAL_MINUTES = 30

# Professional Bitfinex symbol universe for institutional analysis
PROFESSIONAL_SYMBOLS = [
    'tBTCUSD', 'tETHUSD', 'tLTCUSD', 'tXRPUSD', 'tEOSUSD',
    'tBCHUSD', 'tXLMUSD', 'tLINKUSD', 'tTRXUSD', 'tADAUSD',
    'tUNIUSD', 'tDOTUSD', 'tSOLUSD', 'tMATICUSD', 'tAVAXUSD'
]

def ensure_professional_data_dir():
    """Ensure the professional whale data directory exists."""
    if not os.path.exists(PROFESSIONAL_DATA_DIR):
        os.makedirs(PROFESSIONAL_DATA_DIR)
        print(f"📁 Created professional whale data directory: {PROFESSIONAL_DATA_DIR}")

def save_professional_whale_data(data, filename):
    """Save professional whale data with institutional-grade formatting."""
    try:
        ensure_professional_data_dir()
        filepath = os.path.join(PROFESSIONAL_DATA_DIR, filename)
        
        professional_data = {
            'metadata': {
                'analysis_type': 'institutional_whale_tracking',
                'exchange': 'bitfinex_professional',
                'timestamp': datetime.utcnow().isoformat(),
                'data_grade': 'institutional'
            },
            'data': data
        }
        
        with open(filepath, 'w') as f:
            json.dump(professional_data, f, indent=2, default=str)
        print(f"💾 Saved professional whale data to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving professional whale data: {e}")

def analyze_professional_order_flow(symbol, depth=100):
    """Professional institutional order flow analysis using advanced market microstructure."""
    try:
        print(f"🏛️ Analyzing institutional order flow for {symbol}...")
        
        order_book = bfx.get_professional_order_book(symbol, 'P0', depth)
        if not order_book:
            return []
        
        institutional_orders = []
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        # Analyze bid-side institutional activity
        for bid in bids:
            price, amount, count = bid[0], bid[1], bid[2]
            value_usd = price * amount
            
            if value_usd >= PROFESSIONAL_TRADE_THRESHOLD_USD:
                institutional_orders.append({
                    'symbol': symbol,
                    'side': 'BID',
                    'price': price,
                    'amount': amount,
                    'value_usd': value_usd,
                    'order_count': count,
                    'institutional_tier': classify_institutional_tier(value_usd),
                    'market_impact': calculate_market_impact(price, amount, bids, asks),
                    'timestamp': datetime.utcnow(),
                    'analysis_type': 'professional_order_book'
                })
        
        # Analyze ask-side institutional activity  
        for ask in asks:
            price, amount, count = ask[0], ask[1], ask[2]
            value_usd = price * amount
            
            if value_usd >= PROFESSIONAL_TRADE_THRESHOLD_USD:
                institutional_orders.append({
                    'symbol': symbol,
                    'side': 'ASK',
                    'price': price,
                    'amount': amount,
                    'value_usd': value_usd,
                    'order_count': count,
                    'institutional_tier': classify_institutional_tier(value_usd),
                    'market_impact': calculate_market_impact(price, amount, bids, asks),
                    'timestamp': datetime.utcnow(),
                    'analysis_type': 'professional_order_book'
                })
        
        institutional_orders.sort(key=lambda x: x['value_usd'], reverse=True)
        return institutional_orders
        
    except Exception as e:
        print(f"❌ Professional error analyzing order flow for {symbol}: {e}")
        return []

def classify_institutional_tier(value_usd):
    """Classify institutional tier based on professional thresholds."""
    if value_usd >= 10000000:
        return "Sovereign/Central Bank"
    elif value_usd >= 5000000:
        return "Major Institution"  
    elif value_usd >= 2000000:
        return "Investment Bank"
    elif value_usd >= 1000000:
        return "Hedge Fund"
    elif value_usd >= 500000:
        return "Family Office"
    elif value_usd >= 250000:
        return "Professional Trader"
    else:
        return "Sophisticated Retail"

def calculate_market_impact(price, amount, bids, asks):
    """Calculate professional market impact assessment."""
    try:
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else price
        threshold = mid_price * 0.01
        
        nearby_liquidity = 0
        for bid in bids:
            if abs(bid[0] - mid_price) <= threshold:
                nearby_liquidity += bid[1]
        
        for ask in asks:
            if abs(ask[0] - mid_price) <= threshold:
                nearby_liquidity += ask[1]
        
        if nearby_liquidity > 0:
            impact_ratio = amount / nearby_liquidity
            if impact_ratio >= 0.5:
                return "Extreme"
            elif impact_ratio >= 0.25:
                return "High"
            elif impact_ratio >= 0.1:
                return "Moderate"
            else:
                return "Low"
        
        return "Unknown"
        
    except Exception as e:
        return "Unknown"

def monitor_professional_institutional_flow():
    """Monitor institutional flow across professional symbol universe."""
    print(f"🏛️ Starting professional institutional flow monitoring for {len(PROFESSIONAL_SYMBOLS)} symbols...")
    
    all_institutional_data = []
    
    for symbol in tqdm(PROFESSIONAL_SYMBOLS, desc="Professional Analysis"):
        try:
            print(f"\n🔍 Analyzing {symbol}...")
            
            # Professional order flow analysis
            institutional_orders = analyze_professional_order_flow(symbol, 100)
            
            # Professional funding analysis
            funding_analysis = bfx.get_professional_funding_rate(symbol)
            
            symbol_data = {
                'symbol': symbol,
                'institutional_orders': institutional_orders,
                'funding_analysis': funding_analysis,
                'analysis_timestamp': datetime.utcnow(),
                'institutional_count': len([o for o in institutional_orders if o.get('institutional_tier') in ['Major Institution', 'Investment Bank', 'Sovereign/Central Bank']]),
                'total_institutional_volume': sum(o['value_usd'] for o in institutional_orders),
                'professional_grade': 'institutional' if len(institutional_orders) >= 5 else 'professional' if len(institutional_orders) >= 2 else 'standard'
            }
            
            all_institutional_data.append(symbol_data)
            time.sleep(1.0)
            
        except Exception as e:
            print(f"❌ Professional error monitoring {symbol}: {e}")
            continue
    
    return all_institutional_data

def create_professional_institutional_summary(institutional_data):
    """Create comprehensive institutional activity summary."""
    try:
        print("📊 Creating professional institutional flow summary...")
        
        total_institutional_orders = 0
        total_institutional_volume = 0
        top_institutions = []
        most_active_symbols = []
        institutional_tiers = {}
        
        for symbol_data in institutional_data:
            symbol = symbol_data['symbol']
            institutional_orders = symbol_data.get('institutional_orders', [])
            
            total_institutional_orders += len(institutional_orders)
            symbol_volume = symbol_data.get('total_institutional_volume', 0)
            total_institutional_volume += symbol_volume
            
            for order in institutional_orders:
                top_institutions.append({
                    'symbol': symbol,
                    'value_usd': order['value_usd'],
                    'institutional_tier': order['institutional_tier'],
                    'market_impact': order['market_impact'],
                    'side': order['side'],
                    'timestamp': order['timestamp']
                })
                
                tier = order['institutional_tier']
                institutional_tiers[tier] = institutional_tiers.get(tier, 0) + 1
            
            if institutional_orders:
                most_active_symbols.append({
                    'symbol': symbol,
                    'institutional_orders': len(institutional_orders),
                    'total_volume_usd': symbol_volume,
                    'professional_grade': symbol_data['professional_grade']
                })
        
        top_institutions.sort(key=lambda x: x['value_usd'], reverse=True)
        most_active_symbols.sort(key=lambda x: x['total_volume_usd'], reverse=True)
        
        summary = {
            'analysis_timestamp': datetime.utcnow(),
            'analysis_type': 'professional_institutional_flow',
            'total_symbols_analyzed': len(institutional_data),
            'total_institutional_orders': total_institutional_orders,
            'total_institutional_volume_usd': total_institutional_volume,
            'average_institutional_order_size': total_institutional_volume / max(total_institutional_orders, 1),
            'top_institutional_orders': top_institutions[:TOP_N_INSTITUTIONAL],
            'most_active_symbols': most_active_symbols[:TOP_N_POSITIONS],
            'institutional_tier_distribution': institutional_tiers
        }
        
        return summary
        
    except Exception as e:
        print(f"❌ Error creating professional institutional summary: {e}")
        return {}

def display_professional_institutional_summary(summary):
    """Display formatted professional institutional summary."""
    try:
        print(f"\n{Fore.CYAN}{'='*100}")
        print(f"{Fore.CYAN}🏛️ BITFINEX PROFESSIONAL INSTITUTIONAL FLOW ANALYSIS 🏛️")
        print(f"{Fore.CYAN}{'='*100}")
        
        print(f"\n{Fore.YELLOW}📊 INSTITUTIONAL OVERVIEW:")
        print(f"Symbols Analyzed: {summary.get('total_symbols_analyzed', 0)}")
        print(f"Total Institutional Orders: {summary.get('total_institutional_orders', 0):,}")
        print(f"Total Institutional Volume: ${summary.get('total_institutional_volume_usd', 0):,.2f}")
        print(f"Average Order Size: ${summary.get('average_institutional_order_size', 0):,.2f}")
        
        # Institutional tier distribution
        tier_distribution = summary.get('institutional_tier_distribution', {})
        if tier_distribution:
            print(f"\n{Fore.YELLOW}🏆 INSTITUTIONAL TIER DISTRIBUTION:")
            for tier, count in sorted(tier_distribution.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tier}: {count} orders")
        
        # Top institutional orders
        top_institutions = summary.get('top_institutional_orders', [])
        if top_institutions:
            print(f"\n{Fore.GREEN}🏛️ TOP INSTITUTIONAL ORDERS:")
            for i, order in enumerate(top_institutions[:15], 1):
                print(f"  {i:2d}. {order['symbol']:8s} | ${order['value_usd']:>15,.2f} | {order['institutional_tier']:20s} | {order['side']:4s} | {order['market_impact']:8s}")
        
        # Most active symbols
        active_symbols = summary.get('most_active_symbols', [])
        if active_symbols:
            print(f"\n{Fore.MAGENTA}📈 MOST ACTIVE PROFESSIONAL SYMBOLS:")
            for i, symbol_data in enumerate(active_symbols[:12], 1):
                print(f"  {i:2d}. {symbol_data['symbol']:8s} | Orders: {symbol_data['institutional_orders']:3d} | Volume: ${symbol_data['total_volume_usd']:>12,.0f} | Grade: {symbol_data['professional_grade']:12s}")
        
        print(f"\n{Fore.CYAN}{'='*100}\n")
        
    except Exception as e:
        print(f"❌ Error displaying professional summary: {e}")

def run_professional_institutional_analysis():
    """Run complete professional institutional analysis."""
    print(f"{Fore.CYAN}🏛️ Starting Bitfinex Professional Institutional Analysis System...")
    print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    try:
        if not bfx.test_professional_connection():
            print("❌ Failed to connect to Bitfinex professional API")
            return
        
        institutional_data = monitor_professional_institutional_flow()
        
        if not institutional_data:
            print("❌ No institutional data collected")
            return
        
        summary = create_professional_institutional_summary(institutional_data)
        display_professional_institutional_summary(summary)
        
        # Save results
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        save_professional_whale_data(institutional_data, f'institutional_analysis_detailed_{timestamp}.json')
        save_professional_whale_data(summary, f'institutional_analysis_summary_{timestamp}.json')
        
        print(f"\n✅ Bitfinex professional institutional analysis completed successfully!")
        print(f"📁 Professional results saved to: {PROFESSIONAL_DATA_DIR}")
        
        return institutional_data, summary
        
    except Exception as e:
        print(f"❌ Professional institutional analysis failed: {e}")
        return None, None

def main():
    """Main function to run Bitfinex professional institutional tracker."""
    parser = argparse.ArgumentParser(description='Bitfinex Professional Institutional Tracker')
    parser.add_argument('--symbols', nargs='+', default=PROFESSIONAL_SYMBOLS[:8])
    parser.add_argument('--institutional-threshold', type=int, default=INSTITUTIONAL_THRESHOLD_USD)
    parser.add_argument('--continuous', action='store_true')
    
    args = parser.parse_args()
    
    global PROFESSIONAL_SYMBOLS, INSTITUTIONAL_THRESHOLD_USD
    PROFESSIONAL_SYMBOLS = args.symbols
    INSTITUTIONAL_THRESHOLD_USD = args.institutional_threshold
    
    print(f"🏛️ Bitfinex Professional Configuration:")
    print(f"   Symbols: {', '.join(PROFESSIONAL_SYMBOLS)}")
    print(f"   Institutional Threshold: ${INSTITUTIONAL_THRESHOLD_USD:,}")
    
    if args.continuous:
        while True:
            try:
                run_professional_institutional_analysis()
                time.sleep(UPDATE_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user")
                break
    else:
        run_professional_institutional_analysis()

if __name__ == "__main__":
    main()
