'''
BINANCE WHALE TRACKER - Large Holder Analysis System
Track and analyze large holders and significant trading activity on Binance

This system monitors:
- Large balance holders 
- Significant trading activity
- Whale movements and patterns
- Risk metrics for large positions

Key Features:
- Real-time balance tracking
- Trade history analysis  
- Position risk assessment
- Portfolio concentration analysis
- Automated whale detection

Usage:
1. Configure API keys in dontshareconfig.py
2. Set whale detection thresholds
3. Run analysis to identify significant holders
4. Monitor for whale movements
'''

import os
import json
import time
import colorama
import pandas as pd
import argparse
import sys
import binance_nice_funcs as bnf
from datetime import datetime
from tqdm import tqdm
from colorama import Fore

sys.path.append('..')

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas display
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ===== BINANCE WHALE CONFIGURATION =====
DATA_DIR = "binance_whale_data"
WHALE_THRESHOLD_USD = 100000  # $100k+ positions considered whale
LARGE_TRADE_THRESHOLD_USD = 50000  # $50k+ trades flagged as significant
TOP_N_WHALES = 20
TOP_N_POSITIONS = 10
RISK_THRESHOLD_PERCENT = 80  # Flag positions with >80% concentration
UPDATE_INTERVAL_MINUTES = 15

# Binance-specific whale detection symbols
WHALE_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT',
    'XRPUSDT', 'LTCUSDT', 'LINKUSDT', 'BCHUSDT', 'XLMUSDT',
    'UNIUSDT', 'DOGEUSDT', 'SOLUSDT', 'MATICUSDT', 'AVAXUSDT'
]

def ensure_binance_data_dir():
    """Ensure the Binance whale data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 Created Binance whale data directory: {DATA_DIR}")

def save_binance_whale_data(data, filename):
    """Save Binance whale data to JSON file."""
    try:
        ensure_binance_data_dir()
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"💾 Saved Binance whale data to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving Binance whale data: {e}")

def load_binance_whale_data(filename):
    """Load Binance whale data from JSON file."""
    try:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        return None
    except Exception as e:
        print(f"❌ Error loading Binance whale data: {e}")
        return None

def get_binance_top_holders_estimate(symbol, limit=100):
    """
    Estimate large holders by analyzing order book depth and trading patterns.
    Since Binance doesn't expose wallet addresses, we analyze market microstructure.
    """
    try:
        print(f"🔍 Analyzing large holder patterns for {symbol}...")
        
        # Get deep order book
        order_book = bnf.get_order_book(symbol, 5000)
        if not order_book:
            return []
        
        # Analyze large orders in the book
        large_orders = []
        
        # Process bids (buy orders)
        for bid in order_book.get('bids', []):
            price = float(bid[0])
            quantity = float(bid[1])
            value_usd = price * quantity
            
            if value_usd >= LARGE_TRADE_THRESHOLD_USD:
                large_orders.append({
                    'symbol': symbol,
                    'side': 'BUY',
                    'price': price,
                    'quantity': quantity,
                    'value_usd': value_usd,
                    'timestamp': datetime.utcnow(),
                    'type': 'order_book_analysis'
                })
        
        # Process asks (sell orders)  
        for ask in order_book.get('asks', []):
            price = float(ask[0])
            quantity = float(ask[1])
            value_usd = price * quantity
            
            if value_usd >= LARGE_TRADE_THRESHOLD_USD:
                large_orders.append({
                    'symbol': symbol,
                    'side': 'SELL',
                    'price': price,
                    'quantity': quantity,
                    'value_usd': value_usd,
                    'timestamp': datetime.utcnow(),
                    'type': 'order_book_analysis'
                })
        
        # Sort by value
        large_orders.sort(key=lambda x: x['value_usd'], reverse=True)
        
        return large_orders[:limit]
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol} large holders: {e}")
        return []

def analyze_binance_recent_trades(symbol, limit=1000):
    """Analyze recent trades to identify whale activity patterns."""
    try:
        print(f"📊 Analyzing recent whale trades for {symbol}...")
        
        # Get recent trades
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        trades_data = bnf.make_request('/api/v3/aggTrades', params, signed=False)
        if not trades_data:
            return []
        
        whale_trades = []
        
        for trade in trades_data:
            price = float(trade['p'])
            quantity = float(trade['q'])
            value_usd = price * quantity
            
            if value_usd >= LARGE_TRADE_THRESHOLD_USD:
                whale_trades.append({
                    'symbol': symbol,
                    'trade_id': trade['a'],
                    'price': price,
                    'quantity': quantity,
                    'value_usd': value_usd,
                    'timestamp': pd.to_datetime(trade['T'], unit='ms'),
                    'is_buyer_maker': trade['m'],
                    'whale_tier': classify_whale_tier(value_usd)
                })
        
        return whale_trades
        
    except Exception as e:
        print(f"❌ Error analyzing recent trades for {symbol}: {e}")
        return []

def classify_whale_tier(value_usd):
    """Classify whale tier based on trade/position value."""
    if value_usd >= 1000000:
        return "Mega Whale"
    elif value_usd >= 500000:
        return "Large Whale" 
    elif value_usd >= 250000:
        return "Whale"
    elif value_usd >= 100000:
        return "Mini Whale"
    else:
        return "Large Trader"

def get_binance_volume_weighted_analysis(symbol, days=7):
    """Analyze volume-weighted patterns to identify whale accumulation/distribution."""
    try:
        print(f"📈 Analyzing volume patterns for {symbol} over {days} days...")
        
        # Get historical klines
        df = bnf.get_ohlcv(symbol, '1h', days * 24)
        if df.empty:
            return None
        
        # Calculate volume-weighted metrics
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
        df['cumulative_vwap'] = df['vwap'].expanding().mean()
        
        # Identify unusual volume spikes
        df['volume_ma'] = df['volume'].rolling(window=24).mean()
        df['volume_spike'] = df['volume'] / df['volume_ma']
        
        # Whale activity indicators
        recent_data = df.tail(24)  # Last 24 hours
        
        analysis = {
            'symbol': symbol,
            'avg_hourly_volume': recent_data['volume'].mean(),
            'max_volume_spike': recent_data['volume_spike'].max(),
            'volume_trend': 'increasing' if recent_data['volume'].iloc[-1] > recent_data['volume'].mean() else 'decreasing',
            'vwap_current': recent_data['cumulative_vwap'].iloc[-1],
            'price_vs_vwap': 'above' if recent_data['close'].iloc[-1] > recent_data['cumulative_vwap'].iloc[-1] else 'below',
            'whale_activity_score': calculate_whale_activity_score(recent_data),
            'analysis_timestamp': datetime.utcnow()
        }
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analyzing volume patterns for {symbol}: {e}")
        return None

def calculate_whale_activity_score(df):
    """Calculate a whale activity score based on volume and price patterns."""
    try:
        # Volume scoring
        volume_score = min(df['volume_spike'].max() / 2, 5)  # Cap at 5
        
        # Price movement scoring
        price_volatility = df['close'].std() / df['close'].mean()
        volatility_score = min(price_volatility * 100, 3)  # Cap at 3
        
        # Trading intensity scoring
        trades_intensity = len(df[df['volume'] > df['volume'].quantile(0.8)]) / len(df)
        intensity_score = min(trades_intensity * 10, 2)  # Cap at 2
        
        total_score = volume_score + volatility_score + intensity_score
        return round(total_score, 2)
        
    except Exception as e:
        print(f"❌ Error calculating whale activity score: {e}")
        return 0

def monitor_binance_whale_movements():
    """Monitor whale movements across multiple symbols."""
    print(f"🐋 Starting Binance whale monitoring for {len(WHALE_SYMBOLS)} symbols...")
    
    all_whale_data = []
    
    for symbol in tqdm(WHALE_SYMBOLS, desc="Analyzing symbols"):
        try:
            # Order book analysis
            large_orders = get_binance_top_holders_estimate(symbol, 20)
            
            # Recent trades analysis
            whale_trades = analyze_binance_recent_trades(symbol, 500)
            
            # Volume pattern analysis
            volume_analysis = get_binance_volume_weighted_analysis(symbol, 3)
            
            symbol_data = {
                'symbol': symbol,
                'large_orders': large_orders,
                'whale_trades': whale_trades,
                'volume_analysis': volume_analysis,
                'analysis_timestamp': datetime.utcnow(),
                'whale_count': len([t for t in whale_trades if t.get('whale_tier') in ['Whale', 'Large Whale', 'Mega Whale']])
            }
            
            all_whale_data.append(symbol_data)
            
            # Rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error monitoring {symbol}: {e}")
            continue
    
    return all_whale_data

def create_binance_whale_summary(whale_data):
    """Create comprehensive whale activity summary."""
    try:
        print("📊 Creating Binance whale activity summary...")
        
        # Aggregate data
        total_whale_trades = 0
        total_whale_volume = 0
        top_whales = []
        active_symbols = []
        
        for symbol_data in whale_data:
            symbol = symbol_data['symbol']
            whale_trades = symbol_data.get('whale_trades', [])
            volume_analysis = symbol_data.get('volume_analysis', {})
            
            # Count whale trades
            total_whale_trades += len(whale_trades)
            
            # Calculate volume
            symbol_volume = sum(trade['value_usd'] for trade in whale_trades)
            total_whale_volume += symbol_volume
            
            # Track top whales
            for trade in whale_trades:
                top_whales.append({
                    'symbol': symbol,
                    'value_usd': trade['value_usd'],
                    'whale_tier': trade['whale_tier'],
                    'timestamp': trade['timestamp']
                })
            
            # Track active symbols
            if len(whale_trades) > 0:
                active_symbols.append({
                    'symbol': symbol,
                    'whale_trades': len(whale_trades),
                    'total_volume_usd': symbol_volume,
                    'whale_activity_score': volume_analysis.get('whale_activity_score', 0) if volume_analysis else 0
                })
        
        # Sort and limit results
        top_whales.sort(key=lambda x: x['value_usd'], reverse=True)
        active_symbols.sort(key=lambda x: x['whale_activity_score'], reverse=True)
        
        summary = {
            'analysis_timestamp': datetime.utcnow(),
            'total_symbols_analyzed': len(whale_data),
            'total_whale_trades': total_whale_trades,
            'total_whale_volume_usd': total_whale_volume,
            'top_whale_trades': top_whales[:TOP_N_WHALES],
            'most_active_symbols': active_symbols[:TOP_N_POSITIONS],
            'whale_tier_distribution': {},
            'average_whale_trade_size': total_whale_volume / max(total_whale_trades, 1)
        }
        
        # Calculate whale tier distribution
        tier_counts = {}
        for whale in top_whales:
            tier = whale['whale_tier']
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        summary['whale_tier_distribution'] = tier_counts
        
        return summary
        
    except Exception as e:
        print(f"❌ Error creating whale summary: {e}")
        return {}

def display_binance_whale_summary(summary):
    """Display formatted whale activity summary."""
    try:
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}🐋 BINANCE WHALE ACTIVITY SUMMARY 🐋")
        print(f"{Fore.CYAN}{'='*80}")
        
        print(f"\n{Fore.YELLOW}📊 OVERVIEW:")
        print(f"Symbols Analyzed: {summary.get('total_symbols_analyzed', 0)}")
        print(f"Total Whale Trades: {summary.get('total_whale_trades', 0):,}")
        print(f"Total Whale Volume: ${summary.get('total_whale_volume_usd', 0):,.2f}")
        print(f"Average Trade Size: ${summary.get('average_whale_trade_size', 0):,.2f}")
        
        # Whale tier distribution
        distribution = summary.get('whale_tier_distribution', {})
        if distribution:
            print(f"\n{Fore.YELLOW}🏆 WHALE TIER DISTRIBUTION:")
            for tier, count in distribution.items():
                print(f"  {tier}: {count} trades")
        
        # Top whale trades
        top_whales = summary.get('top_whale_trades', [])
        if top_whales:
            print(f"\n{Fore.GREEN}🐋 TOP WHALE TRADES:")
            for i, whale in enumerate(top_whales[:10], 1):
                print(f"  {i:2d}. {whale['symbol']:8s} | ${whale['value_usd']:>12,.2f} | {whale['whale_tier']:12s} | {whale['timestamp']}")
        
        # Most active symbols
        active_symbols = summary.get('most_active_symbols', [])
        if active_symbols:
            print(f"\n{Fore.MAGENTA}📈 MOST ACTIVE SYMBOLS:")
            for i, symbol_data in enumerate(active_symbols[:10], 1):
                print(f"  {i:2d}. {symbol_data['symbol']:8s} | Trades: {symbol_data['whale_trades']:3d} | Volume: ${symbol_data['total_volume_usd']:>10,.0f} | Score: {symbol_data['whale_activity_score']:4.1f}")
        
        print(f"\n{Fore.CYAN}{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Error displaying summary: {e}")

def save_binance_whale_report(whale_data, summary):
    """Save comprehensive whale analysis report."""
    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed data
        save_binance_whale_data(whale_data, f'whale_analysis_detailed_{timestamp}.json')
        
        # Save summary
        save_binance_whale_data(summary, f'whale_analysis_summary_{timestamp}.json')
        
        # Save CSV report
        if summary.get('top_whale_trades'):
            df = pd.DataFrame(summary['top_whale_trades'])
            csv_path = os.path.join(DATA_DIR, f'whale_trades_report_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            print(f"📊 Saved whale trades CSV: {csv_path}")
        
        # Save active symbols CSV
        if summary.get('most_active_symbols'):
            df_symbols = pd.DataFrame(summary['most_active_symbols'])
            symbols_csv_path = os.path.join(DATA_DIR, f'active_symbols_report_{timestamp}.csv')
            df_symbols.to_csv(symbols_csv_path, index=False)
            print(f"📊 Saved active symbols CSV: {symbols_csv_path}")
        
    except Exception as e:
        print(f"❌ Error saving whale report: {e}")

def run_binance_whale_analysis():
    """Run complete Binance whale analysis."""
    print(f"{Fore.CYAN}🐋 Starting Binance Whale Analysis System...")
    print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    try:
        # Test connection
        if not bnf.test_connection():
            print("❌ Failed to connect to Binance API")
            return
        
        # Monitor whale movements
        whale_data = monitor_binance_whale_movements()
        
        if not whale_data:
            print("❌ No whale data collected")
            return
        
        # Create summary
        summary = create_binance_whale_summary(whale_data)
        
        # Display results
        display_binance_whale_summary(summary)
        
        # Save reports
        save_binance_whale_report(whale_data, summary)
        
        print(f"\n✅ Binance whale analysis completed successfully!")
        print(f"📁 Results saved to: {DATA_DIR}")
        
        return whale_data, summary
        
    except Exception as e:
        print(f"❌ Binance whale analysis failed: {e}")
        return None, None

def main():
    """Main function to run Binance whale tracker."""
    parser = argparse.ArgumentParser(description='Binance Whale Tracker')
    parser.add_argument('--symbols', nargs='+', default=WHALE_SYMBOLS[:5], 
                       help='Symbols to analyze (default: top 5)')
    parser.add_argument('--whale-threshold', type=int, default=WHALE_THRESHOLD_USD,
                       help=f'Whale threshold in USD (default: {WHALE_THRESHOLD_USD})')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously with updates')
    
    args = parser.parse_args()
    
    # Update global settings
    global WHALE_SYMBOLS, WHALE_THRESHOLD_USD
    WHALE_SYMBOLS = args.symbols
    WHALE_THRESHOLD_USD = args.whale_threshold
    
    print(f"🐋 Binance Whale Tracker Configuration:")
    print(f"   Symbols: {', '.join(WHALE_SYMBOLS)}")
    print(f"   Whale Threshold: ${WHALE_THRESHOLD_USD:,}")
    print(f"   Continuous Mode: {args.continuous}")
    
    if args.continuous:
        print(f"\n🔄 Starting continuous monitoring (updates every {UPDATE_INTERVAL_MINUTES} minutes)...")
        while True:
            try:
                run_binance_whale_analysis()
                print(f"\n😴 Sleeping for {UPDATE_INTERVAL_MINUTES} minutes...")
                time.sleep(UPDATE_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                print("\n🛑 Continuous monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in continuous mode: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    else:
        run_binance_whale_analysis()

if __name__ == "__main__":
    main()
