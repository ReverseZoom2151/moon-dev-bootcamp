"""
🌙 Moon Dev's Binance Position Tracker (API Version) 🚀
Built with love by Moon Dev 🌙 ✨

This script uses the Binance API to get futures positions data
and performs analysis to identify whale positions and liquidation risks.

Binance version of ppls_pos_3perc.py adapted for Binance futures trading.

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.
"""

import time
import pandas as pd
import colorama
import binance_nice_funcs as n
import argparse
import sys
import traceback
import random
import schedule
import logging
from pathlib import Path
from colorama import Fore, Style

# Add the project root to the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from binance_api import BinanceAPI

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas to display numbers with commas and no scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Configuration
CONFIG = {
    "TOP_N_POSITIONS": 10,
    "MIN_POSITION_VALUE": 25000,
    "WHALE_THRESHOLD": 250000,
    "LIQUIDATION_RISK_THRESHOLD": 5.0,  # % from liquidation price
    "DATA_REFRESH_MINUTES": 5,
    "WHALE_MINIMUM_BALANCE": 100000,
    "SAVE_CSV": True,
    "DISPLAY_COLORS": True
}

# --- Moon Dev Quotes ---
MOON_DEV_QUOTES = [
    "🌙 To the moon we go! 🚀",
    "🔥 Hunting liquidations like a pro! 💎",
    "📊 Data is the new oil, and we're drilling deep! ⛽",
    "🐋 Whale watching season is open! 🌊",
    "💰 Stack sats, not regrets! ₿",
    "🎯 Precision strikes only! No spray and pray! 🏹",
    "🚀 Rocket fuel: coffee and market data! ☕",
    "🌟 Diamond hands don't fold! 💎🙌"
]

def get_random_quote():
    """Return a random Moon Dev quote"""
    return random.choice(MOON_DEV_QUOTES)

def ensure_data_dir():
    """Ensure the data directory exists"""
    data_dir = Path("data/binance_positions")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_binance_balance(api, asset='USDT'):
    """Get balance for a specific asset from Binance"""
    try:
        account_info = api._make_request(f"{api.base_url.replace('fapi', 'api')}/api/v3/account", signed=True)
        if account_info and 'balances' in account_info:
            for balance in account_info['balances']:
                if balance['asset'] == asset:
                    return {
                        'asset': asset,
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
        return {'asset': asset, 'free': 0.0, 'locked': 0.0, 'total': 0.0}
    except Exception as e:
        logger.error(f"Error getting balance for {asset}: {str(e)}")
        return {'asset': asset, 'free': 0.0, 'locked': 0.0, 'total': 0.0}

def get_binance_symbol_price(symbol):
    """Get current price for a Binance symbol"""
    try:
        # Remove 'USDT' suffix if present for price lookup
        if symbol.endswith('USDT'):
            lookup_symbol = symbol
        else:
            lookup_symbol = f"{symbol}USDT"
        
        price_data = n.spot_price_and_symbol_info(lookup_symbol)
        if price_data:
            return price_data['price']
        return None
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {str(e)}")
        return None

def display_top_individual_positions(df: pd.DataFrame, n: int = CONFIG["TOP_N_POSITIONS"]):
    """
    Display top individual long and short positions
    """
    try:
        if df.empty:
            print(f"{Fore.RED}❌ No position data available{Style.RESET_ALL}")
            return

        # Separate longs and shorts
        longs_df = df[df['side'] == 'LONG'].nlargest(n, 'notional')
        shorts_df = df[df['side'] == 'SHORT'].nlargest(n, 'notional')

        print(f"\n{Fore.CYAN}🔥 TOP {n} LONG POSITIONS 🔥{Style.RESET_ALL}")
        print("=" * 100)
        if not longs_df.empty:
            for idx, row in longs_df.iterrows():
                risk_color = Fore.RED if row['liquidation_risk'] <= 5 else Fore.YELLOW if row['liquidation_risk'] <= 10 else Fore.GREEN
                print(f"{Fore.GREEN}📈 {row['symbol']:12} | "
                      f"${row['notional']:>12,.0f} | "
                      f"Entry: ${row['entry_price']:>8.2f} | "
                      f"Current: ${row['mark_price']:>8.2f} | "
                      f"PnL: {row['unrealized_pnl']:>+8.0f} | "
                      f"{risk_color}Risk: {row['liquidation_risk']:>5.1f}%{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}🩸 TOP {n} SHORT POSITIONS 🩸{Style.RESET_ALL}")
        print("=" * 100)
        if not shorts_df.empty:
            for idx, row in shorts_df.iterrows():
                risk_color = Fore.RED if row['liquidation_risk'] <= 5 else Fore.YELLOW if row['liquidation_risk'] <= 10 else Fore.GREEN
                print(f"{Fore.RED}📉 {row['symbol']:12} | "
                      f"${row['notional']:>12,.0f} | "
                      f"Entry: ${row['entry_price']:>8.2f} | "
                      f"Current: ${row['mark_price']:>8.2f} | "
                      f"PnL: {row['unrealized_pnl']:>+8.0f} | "
                      f"{risk_color}Risk: {row['liquidation_risk']:>5.1f}%{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"Error displaying top positions: {str(e)}")

def display_risk_metrics(df: pd.DataFrame):
    """
    Display metrics for positions closest to liquidation
    """
    try:
        if df.empty:
            print(f"{Fore.RED}❌ No position data available for risk analysis{Style.RESET_ALL}")
            return

        # Positions at high risk (within 10% of liquidation)
        high_risk = df[df['liquidation_risk'] <= 10.0].sort_values('liquidation_risk')
        
        if high_risk.empty:
            print(f"{Fore.GREEN}✅ No positions at high liquidation risk (>10%){Style.RESET_ALL}")
            return

        print(f"\n{Fore.RED}⚠️  POSITIONS AT LIQUIDATION RISK ⚠️{Style.RESET_ALL}")
        print("=" * 120)
        
        total_at_risk = high_risk['notional'].sum()
        print(f"{Fore.YELLOW}Total Value at Risk: ${total_at_risk:,.0f}{Style.RESET_ALL}")
        
        # Show top risky positions with account balance info
        for idx, row in high_risk.head(10).iterrows():
            risk_color = Fore.RED if row['liquidation_risk'] <= 3 else Fore.YELLOW
            
            # Get estimated account balance (simplified)
            estimated_balance = row['notional'] / (row.get('leverage', 10) * 0.8)  # Rough estimate
            
            print(f"{risk_color}🚨 {row['symbol']:12} | "
                  f"Size: ${row['notional']:>12,.0f} | "
                  f"Entry: ${row['entry_price']:>8.2f} | "
                  f"Liq: ${row['liquidation_price']:>8.2f} | "
                  f"Risk: {row['liquidation_risk']:>5.1f}% | "
                  f"Est.Bal: ${estimated_balance:>8,.0f} | "
                  f"Side: {row['side']}{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"Error displaying risk metrics: {str(e)}")

def save_liquidation_risk_to_csv(df: pd.DataFrame):
    """
    Save positions closest to liquidation to a CSV file
    """
    try:
        if df.empty:
            return
        
        data_dir = ensure_data_dir()
        
        # Get high-risk positions
        high_risk = df[df['liquidation_risk'] <= 15.0].sort_values('liquidation_risk')
        
        if high_risk.empty:
            logger.info("No high-risk positions to save")
            return
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = data_dir / f"binance_liquidation_risk_{timestamp}.csv"
        
        high_risk.to_csv(filename, index=False)
        logger.info(f"💾 Saved {len(high_risk)} high-risk positions to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving liquidation risk data: {str(e)}")

def process_positions(df, symbol_filter=None):
    """
    Process the position data into a more usable format, filtering positions below min value
    and optionally by symbol
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Filter by minimum position value
        df_filtered = df[df['notional'] >= CONFIG["MIN_POSITION_VALUE"]].copy()
        
        # Filter by symbol if specified
        if symbol_filter:
            df_filtered = df_filtered[df_filtered['symbol'].str.contains(symbol_filter, case=False)]
        
        # Calculate liquidation risk percentage
        df_filtered['liquidation_risk'] = df_filtered.apply(
            lambda row: abs((row['mark_price'] - row.get('liquidation_price', row['mark_price'] * 0.9)) / row['mark_price']) * 100
            if row.get('liquidation_price', 0) > 0 else 99.9, axis=1
        )
        
        # Add whale classification
        df_filtered['is_whale'] = df_filtered['notional'] >= CONFIG["WHALE_THRESHOLD"]
        
        return df_filtered.sort_values('notional', ascending=False)
        
    except Exception as e:
        logger.error(f"Error processing positions: {str(e)}")
        return pd.DataFrame()

def fetch_positions_from_api():
    """
    Fetch positions from Binance API
    """
    try:
        logger.info("🌙 Fetching positions from Binance API...")
        
        api = BinanceAPI()
        positions_df = api.get_positions()
        
        if positions_df.empty:
            logger.warning("No positions data received from Binance API")
            return pd.DataFrame()
        
        # Process the data
        processed_df = process_positions(positions_df)
        
        logger.info(f"✅ Processed {len(processed_df)} positions (${processed_df['notional'].sum():,.0f} total)")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error fetching positions from API: {str(e)}")
        return pd.DataFrame()

def display_market_metrics():
    """
    Display market metrics (funding rates) in a compact format
    """
    try:
        logger.info("📊 Fetching market metrics...")
        
        api = BinanceAPI()
        funding_df = api.get_funding_data()
        
        if funding_df.empty:
            print(f"{Fore.RED}❌ No funding data available{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}💰 FUNDING RATES (Top 10){Style.RESET_ALL}")
        print("=" * 80)
        
        # Show top funding rates
        top_funding = funding_df.nlargest(10, 'funding_rate_pct')
        
        for idx, row in top_funding.iterrows():
            funding_color = Fore.RED if row['funding_rate_pct'] > 0.1 else Fore.GREEN if row['funding_rate_pct'] < -0.1 else Fore.YELLOW
            print(f"{funding_color}{row['symbol']:15} | {row['funding_rate_pct']:>+7.4f}%{Style.RESET_ALL}")
        
    except Exception as e:
        logger.error(f"Error displaying market metrics: {str(e)}")

def bot():
    """Main function to run the position tracker (renamed from main to bot)"""
    try:
        print(f"\n{Fore.MAGENTA}🌙 Moon Dev's Binance Position Tracker 🚀{Style.RESET_ALL}")
        print("=" * 80)
        print(f"{Fore.CYAN}{get_random_quote()}{Style.RESET_ALL}")
        print("=" * 80)
        
        # Fetch position data
        positions_df = fetch_positions_from_api()
        
        if positions_df.empty:
            print(f"{Fore.RED}❌ No position data available. Make sure you have active positions.{Style.RESET_ALL}")
            return
        
        # Display analysis
        display_top_individual_positions(positions_df)
        display_risk_metrics(positions_df)
        display_market_metrics()
        
        # Save data if configured
        if CONFIG["SAVE_CSV"]:
            save_liquidation_risk_to_csv(positions_df)
        
        # Summary statistics
        total_positions = len(positions_df)
        total_value = positions_df['notional'].sum()
        whale_count = len(positions_df[positions_df['is_whale']])
        high_risk_count = len(positions_df[positions_df['liquidation_risk'] <= 10])
        
        print(f"\n{Fore.CYAN}📊 SUMMARY STATISTICS{Style.RESET_ALL}")
        print("=" * 50)
        print(f"Total Positions: {total_positions}")
        print(f"Total Value: ${total_value:,.0f}")
        print(f"Whale Positions: {whale_count}")
        print(f"High Risk Positions: {high_risk_count}")
        
        print(f"\n{Fore.GREEN}✅ Analysis complete! Next update in {CONFIG['DATA_REFRESH_MINUTES']} minutes.{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 Moon Dev's Position Tracker stopped by user{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Error in main bot function: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🌙 Moon Dev's Binance Position Tracker")
    parser.add_argument("--symbol", help="Filter by symbol (e.g., BTC, ETH)")
    parser.add_argument("--min-value", type=float, default=CONFIG["MIN_POSITION_VALUE"], 
                       help="Minimum position value to display")
    parser.add_argument("--refresh", type=int, default=CONFIG["DATA_REFRESH_MINUTES"],
                       help="Refresh interval in minutes")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG["MIN_POSITION_VALUE"] = args.min_value
    CONFIG["DATA_REFRESH_MINUTES"] = args.refresh
    
    if args.once:
        # Run once
        bot()
    else:
        # Initial run
        bot()
        
        # Schedule regular updates
        schedule.every(CONFIG["DATA_REFRESH_MINUTES"]).minutes.do(bot)
        
        print(f"\n{Fore.CYAN}🔄 Scheduled to run every {CONFIG['DATA_REFRESH_MINUTES']} minutes. Press Ctrl+C to stop.{Style.RESET_ALL}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(10)
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Binance Position Tracker stopped{Style.RESET_ALL}")
