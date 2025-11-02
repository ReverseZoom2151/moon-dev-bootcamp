"""
🌙 Moon Dev's Bitfinex Position Tracker (API Version) 🚀
Built with love by Moon Dev 🌙 ✨

This script uses the Bitfinex API to get margin and derivatives positions data
and performs institutional-grade analysis to identify whale positions and liquidation risks.

Bitfinex version of ppls_pos_3perc.py adapted for institutional Bitfinex trading.

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.
"""

import time
import pandas as pd
import colorama
import bitfinex_nice_funcs as n
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
from bitfinex_api import BitfinexAPI

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas to display numbers with commas and no scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Institutional Configuration
CONFIG = {
    "TOP_N_POSITIONS": 15,  # More positions for institutional analysis
    "MIN_POSITION_VALUE": 50000,  # Higher threshold for institutional trading
    "WHALE_THRESHOLD": 500000,  # Higher whale threshold for institutional market
    "INSTITUTIONAL_THRESHOLD": 2000000,  # New tier for institutional traders
    "LIQUIDATION_RISK_THRESHOLD": 3.0,  # Tighter risk threshold for margin trading
    "DATA_REFRESH_MINUTES": 3,  # More frequent updates for institutional trading
    "WHALE_MINIMUM_BALANCE": 250000,  # Higher minimum for institutional whales
    "SAVE_CSV": True,
    "DISPLAY_COLORS": True,
    "MARGIN_CALL_THRESHOLD": 20.0,  # % from margin call
    "PROFESSIONAL_MODE": True
}

# --- Moon Dev Institutional Quotes ---
MOON_DEV_QUOTES = [
    "🌙 Institutional grade moon missions! 🏛️🚀",
    "🔥 Professional liquidation hunting with style! 💼💎",
    "📊 Big data, bigger profits! Institutional edition! 🏢⛽",
    "🐋 Institutional whale watching - where titans trade! 🌊🏛️",
    "💰 Margin trading with institutional precision! ₿🎯",
    "🎯 Professional strikes only! No amateur hour! 🏹💼",
    "🚀 Rocket fuel: espresso and institutional data! ☕📈",
    "🌟 Institutional diamond hands never fold! 💎🙌🏛️"
]

def get_random_quote():
    """Return a random Moon Dev institutional quote"""
    return random.choice(MOON_DEV_QUOTES)

def ensure_data_dir():
    """Ensure the data directory exists"""
    data_dir = Path("data/bitfinex_positions")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_bitfinex_wallet_balance(api, currency='USD'):
    """Get wallet balance for a specific currency from Bitfinex"""
    try:
        wallets = api._make_request("/v2/auth/r/wallets", signed=True)
        
        if wallets:
            total_balance = 0
            margin_balance = 0
            exchange_balance = 0
            
            for wallet in wallets:
                # Format: [WALLET_TYPE, CURRENCY, BALANCE, ...]
                if len(wallet) >= 3 and wallet[1] == currency:
                    wallet_type = wallet[0]
                    balance = float(wallet[2])
                    
                    if wallet_type == 'exchange':
                        exchange_balance += balance
                    elif wallet_type == 'margin':
                        margin_balance += balance
                    
                    total_balance += balance
            
            return {
                'currency': currency,
                'total': total_balance,
                'exchange': exchange_balance,
                'margin': margin_balance,
                'available_margin': margin_balance * 3.3  # Approximate available margin
            }
        
        return {'currency': currency, 'total': 0.0, 'exchange': 0.0, 'margin': 0.0, 'available_margin': 0.0}
    except Exception as e:
        logger.error(f"Error getting wallet balance for {currency}: {str(e)}")
        return {'currency': currency, 'total': 0.0, 'exchange': 0.0, 'margin': 0.0, 'available_margin': 0.0}

def get_bitfinex_symbol_price(symbol):
    """Get current price for a Bitfinex symbol"""
    try:
        if not symbol.startswith('t'):
            symbol = f"t{symbol}"
        
        price_data = n.spot_price_and_symbol_info(symbol)
        if price_data:
            return price_data['price']
        return None
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {str(e)}")
        return None

def classify_trader_tier(notional_value):
    """Classify trader based on position size"""
    if notional_value >= CONFIG["INSTITUTIONAL_THRESHOLD"]:
        return "🏛️ INSTITUTIONAL", Fore.MAGENTA
    elif notional_value >= CONFIG["WHALE_THRESHOLD"]:
        return "🐋 WHALE", Fore.CYAN
    elif notional_value >= CONFIG["MIN_POSITION_VALUE"]:
        return "🐟 RETAIL", Fore.WHITE
    else:
        return "🦐 MICRO", Fore.LIGHTBLACK_EX

def display_top_individual_positions(df: pd.DataFrame, n: int = CONFIG["TOP_N_POSITIONS"]):
    """
    Display top individual long and short positions with institutional analysis
    """
    try:
        if df.empty:
            print(f"{Fore.RED}❌ No position data available{Style.RESET_ALL}")
            return

        # Separate longs and shorts
        longs_df = df[df['side'] == 'LONG'].nlargest(n, 'notional')
        shorts_df = df[df['side'] == 'SHORT'].nlargest(n, 'notional')

        print(f"\n{Fore.CYAN}📈 TOP {n} INSTITUTIONAL LONG POSITIONS 📈{Style.RESET_ALL}")
        print("=" * 130)
        if not longs_df.empty:
            for idx, row in longs_df.iterrows():
                margin_risk_color = Fore.RED if row.get('margin_risk', 99) <= 10 else Fore.YELLOW if row.get('margin_risk', 99) <= 25 else Fore.GREEN
                tier_name, tier_color = classify_trader_tier(row['notional'])
                
                print(f"{Fore.GREEN}📈 {row['symbol']:15} | "
                      f"{tier_color}{tier_name:15}{Style.RESET_ALL} | "
                      f"${row['notional']:>15,.0f} | "
                      f"Entry: ${row['base_price']:>10.4f} | "
                      f"Current: ${row.get('mark_price', row['base_price']):>10.4f} | "
                      f"PnL: {row['unrealized_pnl']:>+10.0f} | "
                      f"{margin_risk_color}Margin Risk: {row.get('margin_risk', 99):>6.1f}%{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}📉 TOP {n} INSTITUTIONAL SHORT POSITIONS 📉{Style.RESET_ALL}")
        print("=" * 130)
        if not shorts_df.empty:
            for idx, row in shorts_df.iterrows():
                margin_risk_color = Fore.RED if row.get('margin_risk', 99) <= 10 else Fore.YELLOW if row.get('margin_risk', 99) <= 25 else Fore.GREEN
                tier_name, tier_color = classify_trader_tier(row['notional'])
                
                print(f"{Fore.RED}📉 {row['symbol']:15} | "
                      f"{tier_color}{tier_name:15}{Style.RESET_ALL} | "
                      f"${row['notional']:>15,.0f} | "
                      f"Entry: ${row['base_price']:>10.4f} | "
                      f"Current: ${row.get('mark_price', row['base_price']):>10.4f} | "
                      f"PnL: {row['unrealized_pnl']:>+10.0f} | "
                      f"{margin_risk_color}Margin Risk: {row.get('margin_risk', 99):>6.1f}%{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"Error displaying top positions: {str(e)}")

def display_institutional_risk_metrics(df: pd.DataFrame):
    """
    Display institutional-grade risk metrics for margin positions
    """
    try:
        if df.empty:
            print(f"{Fore.RED}❌ No position data available for risk analysis{Style.RESET_ALL}")
            return

        # Positions at high margin risk
        high_risk = df[df['margin_risk'] <= CONFIG["MARGIN_CALL_THRESHOLD"]].sort_values('margin_risk')
        
        if high_risk.empty:
            print(f"{Fore.GREEN}✅ No positions at high margin risk (<{CONFIG['MARGIN_CALL_THRESHOLD']}%){Style.RESET_ALL}")
            return

        print(f"\n{Fore.RED}⚠️  INSTITUTIONAL POSITIONS AT MARGIN RISK ⚠️{Style.RESET_ALL}")
        print("=" * 140)
        
        total_at_risk = high_risk['notional'].sum()
        institutional_at_risk = len(high_risk[high_risk['notional'] >= CONFIG["INSTITUTIONAL_THRESHOLD"]])
        
        print(f"{Fore.YELLOW}Total Institutional Value at Risk: ${total_at_risk:,.0f}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Institutional Positions at Risk: {institutional_at_risk}{Style.RESET_ALL}")
        
        # Show top risky positions with detailed institutional analysis
        for idx, row in high_risk.head(15).iterrows():
            risk_color = Fore.RED if row['margin_risk'] <= 5 else Fore.YELLOW if row['margin_risk'] <= 15 else Fore.CYAN
            tier_name, tier_color = classify_trader_tier(row['notional'])
            
            # Estimate required margin and available margin
            required_margin = row['notional'] / 3.3  # Approximate for 3.3:1 leverage
            margin_ratio = (row['margin_risk'] / 100) * required_margin
            
            print(f"{risk_color}🚨 {row['symbol']:15} | "
                  f"{tier_color}{tier_name:15}{Style.RESET_ALL} | "
                  f"Size: ${row['notional']:>15,.0f} | "
                  f"Entry: ${row['base_price']:>10.4f} | "
                  f"Risk: {row['margin_risk']:>6.1f}% | "
                  f"Est.Margin: ${required_margin:>12,.0f} | "
                  f"Side: {row['side']:>5}{Style.RESET_ALL}")

        # Institutional summary metrics
        print(f"\n{Fore.MAGENTA}🏛️ INSTITUTIONAL RISK SUMMARY{Style.RESET_ALL}")
        print("=" * 80)
        
        institutional_positions = df[df['notional'] >= CONFIG["INSTITUTIONAL_THRESHOLD"]]
        whale_positions = df[(df['notional'] >= CONFIG["WHALE_THRESHOLD"]) & (df['notional'] < CONFIG["INSTITUTIONAL_THRESHOLD"])]
        
        print(f"Institutional Positions: {len(institutional_positions)} (${institutional_positions['notional'].sum():,.0f})")
        print(f"Whale Positions: {len(whale_positions)} (${whale_positions['notional'].sum():,.0f})")
        print(f"Average Position Size: ${df['notional'].mean():,.0f}")
        print(f"Largest Position: ${df['notional'].max():,.0f}")

    except Exception as e:
        logger.error(f"Error displaying institutional risk metrics: {str(e)}")

def save_institutional_risk_to_csv(df: pd.DataFrame):
    """
    Save institutional positions at margin risk to a CSV file
    """
    try:
        if df.empty:
            return
        
        data_dir = ensure_data_dir()
        
        # Get high-risk positions with focus on institutional
        high_risk = df[df['margin_risk'] <= CONFIG["MARGIN_CALL_THRESHOLD"]].sort_values('margin_risk')
        
        if high_risk.empty:
            logger.info("No high-risk institutional positions to save")
            return
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = data_dir / f"bitfinex_institutional_risk_{timestamp}.csv"
        
        # Add tier classification to CSV
        high_risk['trader_tier'] = high_risk['notional'].apply(lambda x: classify_trader_tier(x)[0])
        
        high_risk.to_csv(filename, index=False)
        logger.info(f"💾 Saved {len(high_risk)} high-risk institutional positions to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving institutional risk data: {str(e)}")

def process_positions(df, symbol_filter=None):
    """
    Process the position data with institutional-grade analysis
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Filter by minimum position value (higher for institutional)
        df_filtered = df[df['notional'] >= CONFIG["MIN_POSITION_VALUE"]].copy()
        
        # Filter by symbol if specified
        if symbol_filter:
            df_filtered = df_filtered[df_filtered['symbol'].str.contains(symbol_filter, case=False)]
        
        # Calculate margin risk (distance from margin call)
        df_filtered['margin_risk'] = df_filtered.apply(
            lambda row: max(0.1, abs((row.get('mark_price', row['base_price']) - 
                                    row.get('liquidation_price', row['base_price'] * 0.85)) / 
                                   row.get('mark_price', row['base_price'])) * 100)
            if row.get('liquidation_price', 0) > 0 else 50.0, axis=1
        )
        
        # Add institutional classifications
        df_filtered['is_whale'] = df_filtered['notional'] >= CONFIG["WHALE_THRESHOLD"]
        df_filtered['is_institutional'] = df_filtered['notional'] >= CONFIG["INSTITUTIONAL_THRESHOLD"]
        
        # Calculate position health score (0-100, higher is healthier)
        df_filtered['health_score'] = df_filtered.apply(
            lambda row: min(100, max(0, (row['margin_risk'] - 5) * 2)), axis=1
        )
        
        return df_filtered.sort_values('notional', ascending=False)
        
    except Exception as e:
        logger.error(f"Error processing institutional positions: {str(e)}")
        return pd.DataFrame()

def fetch_positions_from_api():
    """
    Fetch positions from Bitfinex API with institutional focus
    """
    try:
        logger.info("🌙 Fetching institutional positions from Bitfinex API...")
        
        api = BitfinexAPI()
        positions_df = api.get_positions()
        
        if positions_df.empty:
            logger.warning("No positions data received from Bitfinex API")
            return pd.DataFrame()
        
        # Add current market prices for better risk calculation
        for idx, row in positions_df.iterrows():
            current_price = get_bitfinex_symbol_price(row['symbol'])
            if current_price:
                positions_df.at[idx, 'mark_price'] = current_price
        
        # Process the data with institutional analysis
        processed_df = process_positions(positions_df)
        
        logger.info(f"✅ Processed {len(processed_df)} institutional positions (${processed_df['notional'].sum():,.0f} total)")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error fetching positions from API: {str(e)}")
        return pd.DataFrame()

def display_funding_and_derivatives_metrics():
    """
    Display institutional funding rates and derivatives metrics
    """
    try:
        logger.info("📊 Fetching institutional market metrics...")
        
        api = BitfinexAPI()
        funding_df = api.get_funding_data()
        oi_df = api.get_oi_data()
        
        if not funding_df.empty:
            print(f"\n{Fore.CYAN}💰 INSTITUTIONAL FUNDING RATES{Style.RESET_ALL}")
            print("=" * 90)
            
            # Show funding rates with institutional focus
            for idx, row in funding_df.iterrows():
                funding_color = Fore.RED if row['funding_rate_pct'] > 0.2 else Fore.GREEN if row['funding_rate_pct'] < -0.2 else Fore.YELLOW
                
                # Estimate daily funding cost for $1M position
                daily_cost = abs(row['funding_rate_pct'] * 1000000 / 100)
                
                print(f"{funding_color}{row['symbol']:20} | "
                      f"{row['funding_rate_pct']:>+8.4f}% | "
                      f"Daily Cost ($1M): ${daily_cost:>8,.0f}{Style.RESET_ALL}")
        
        if not oi_df.empty:
            print(f"\n{Fore.CYAN}📊 OPEN INTEREST ANALYSIS{Style.RESET_ALL}")
            print("=" * 70)
            
            total_oi = oi_df['open_interest'].sum()
            print(f"Total Open Interest: {total_oi:,.0f}")
            
            for idx, row in oi_df.iterrows():
                oi_pct = (row['open_interest'] / total_oi) * 100
                print(f"{row['symbol']:20} | {row['open_interest']:>15,.0f} | {oi_pct:>6.2f}%")
        
    except Exception as e:
        logger.error(f"Error displaying institutional market metrics: {str(e)}")

def bot():
    """Main institutional position tracker function"""
    try:
        print(f"\n{Fore.MAGENTA}🌙 Moon Dev's Institutional Bitfinex Tracker 🏛️🚀{Style.RESET_ALL}")
        print("=" * 90)
        print(f"{Fore.CYAN}{get_random_quote()}{Style.RESET_ALL}")
        print("=" * 90)
        
        # Fetch institutional position data
        positions_df = fetch_positions_from_api()
        
        if positions_df.empty:
            print(f"{Fore.RED}❌ No institutional position data available. Check API credentials.{Style.RESET_ALL}")
            return
        
        # Display institutional analysis
        display_top_individual_positions(positions_df)
        display_institutional_risk_metrics(positions_df)
        display_funding_and_derivatives_metrics()
        
        # Save data if configured
        if CONFIG["SAVE_CSV"]:
            save_institutional_risk_to_csv(positions_df)
        
        # Institutional summary statistics
        total_positions = len(positions_df)
        total_value = positions_df['notional'].sum()
        institutional_count = len(positions_df[positions_df['is_institutional']])
        whale_count = len(positions_df[positions_df['is_whale'] & ~positions_df['is_institutional']])
        high_risk_count = len(positions_df[positions_df['margin_risk'] <= CONFIG["MARGIN_CALL_THRESHOLD"]])
        avg_health = positions_df['health_score'].mean()
        
        print(f"\n{Fore.CYAN}📊 INSTITUTIONAL SUMMARY{Style.RESET_ALL}")
        print("=" * 60)
        print(f"Total Positions: {total_positions}")
        print(f"Total Value: ${total_value:,.0f}")
        print(f"Institutional Positions: {institutional_count}")
        print(f"Whale Positions: {whale_count}")
        print(f"High Risk Positions: {high_risk_count}")
        print(f"Average Health Score: {avg_health:.1f}/100")
        
        print(f"\n{Fore.GREEN}✅ Institutional analysis complete! Next update in {CONFIG['DATA_REFRESH_MINUTES']} minutes.{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 Institutional Bitfinex Tracker stopped by user{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Error in institutional bot function: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🌙 Moon Dev's Institutional Bitfinex Position Tracker")
    parser.add_argument("--symbol", help="Filter by symbol (e.g., BTC, ETH)")
    parser.add_argument("--min-value", type=float, default=CONFIG["MIN_POSITION_VALUE"], 
                       help="Minimum position value to display")
    parser.add_argument("--refresh", type=int, default=CONFIG["DATA_REFRESH_MINUTES"],
                       help="Refresh interval in minutes")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--professional", action="store_true", help="Enable professional mode")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG["MIN_POSITION_VALUE"] = args.min_value
    CONFIG["DATA_REFRESH_MINUTES"] = args.refresh
    CONFIG["PROFESSIONAL_MODE"] = args.professional
    
    if args.once:
        # Run once
        bot()
    else:
        # Initial run
        bot()
        
        # Schedule regular updates for institutional trading
        schedule.every(CONFIG["DATA_REFRESH_MINUTES"]).minutes.do(bot)
        
        print(f"\n{Fore.CYAN}🔄 Institutional tracker scheduled every {CONFIG['DATA_REFRESH_MINUTES']} minutes. Press Ctrl+C to stop.{Style.RESET_ALL}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(5)  # More frequent checks for institutional trading
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Institutional Bitfinex Position Tracker stopped{Style.RESET_ALL}")
