'''
🌙 Moon Dev's Bitfinex Professional Token Tracker 🚀
Built with love by Moon Dev 🌙 ✨

This script tracks trending tokens, new listings, and high-volume tokens on Bitfinex
using the Bitfinex API with professional features including funding rate analysis.

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.
'''

import os
import time
import pandas as pd
import colorama
import argparse
import traceback
import random
import schedule
import bitfinex_config as config  # Import the Bitfinex configuration file
from bitfinex_api_client import BitfinexAPI # Import the Bitfinex API client
from datetime import datetime, timedelta
from colorama import Fore
from bitfinex_display import ( # Import display functions and constants
    BITFINEX_BANNER, 
    BITFINEX_QUOTES, 
    display_trending_tokens, 
    display_new_listings, 
    display_possible_gems, 
    display_consistent_trending,
    display_volume_leaders,
    display_funding_rates
)

try:
    import dontshare as d  # Import API key from dontshare.py
    BITFINEX_API_KEY = getattr(d, 'BITFINEX_API_KEY', None)
    BITFINEX_API_SECRET = getattr(d, 'BITFINEX_API_SECRET', None)
except ImportError:
    print("Warning: dontshare.py not found. API keys not loaded.")
    BITFINEX_API_KEY = None
    BITFINEX_API_SECRET = None

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas to display numbers with commas and no scientific notation
pd.set_option('display.float_format', '{:.6f}'.format)  # Higher precision for Bitfinex
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)  # Wider display for more data

def get_random_quote():
    """Return a random Bitfinex professional quote"""
    return random.choice(BITFINEX_QUOTES)

def ensure_data_dir():
    """Ensure the data directory exists"""
    os.makedirs(config.DATA_DIR, exist_ok=True)

def save_data_to_csv(df, filename):
    """Save DataFrame to CSV file"""
    if not df.empty:
        ensure_data_dir()
        filepath = os.path.join(config.DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Professional data saved to: {filepath}")

def update_trending_history(df):
    """Update historical tracking of trending tokens"""
    if df.empty:
        return
    
    ensure_data_dir()
    
    # Add timestamp and professional metadata
    df_with_timestamp = df.copy()
    df_with_timestamp['timestamp'] = datetime.now()
    df_with_timestamp['date'] = datetime.now().strftime('%Y-%m-%d')
    df_with_timestamp['hour'] = datetime.now().strftime('%H')
    df_with_timestamp['exchange'] = 'Bitfinex'
    df_with_timestamp['data_source'] = 'Professional API'
    
    # Load existing history or create new
    if os.path.exists(config.HISTORY_FILE):
        try:
            history_df = pd.read_csv(config.HISTORY_FILE)
            # Convert timestamp column to datetime if it exists
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        except Exception as e:
            print(f"Error loading history file: {e}")
            history_df = pd.DataFrame()
    else:
        history_df = pd.DataFrame()
    
    # Combine with new data
    combined_df = pd.concat([history_df, df_with_timestamp], ignore_index=True)
    
    # Keep only recent data (last 30 days for professional analysis)
    cutoff_date = datetime.now() - timedelta(days=30)
    if 'timestamp' in combined_df.columns:
        combined_df = combined_df[pd.to_datetime(combined_df['timestamp']) > cutoff_date]
    
    # Save updated history
    try:
        combined_df.to_csv(config.HISTORY_FILE, index=False)
        print(f"Updated professional trending history with {len(df_with_timestamp)} tokens")
    except Exception as e:
        print(f"Error saving professional history: {e}")
    
    return combined_df

def bot():
    """Main function to run the Bitfinex professional token tracker"""
    try:
        print(BITFINEX_BANNER)
        print(f"{Fore.GREEN}🎯 {get_random_quote()}")
        print(f"{Fore.CYAN}Starting Bitfinex Professional Token Tracker...")
        print(f"{Fore.WHITE}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Professional Mode)")
        print(f"{Fore.MAGENTA}🏛️  Institutional-grade data analysis enabled")
        
        # Initialize API client
        api_client = BitfinexAPI(api_key=BITFINEX_API_KEY, api_secret=BITFINEX_API_SECRET)
        
        # Add professional API delay
        print(f"{Fore.YELLOW}⏱️  Applying professional rate limiting...")
        time.sleep(config.API_DELAY)
        
        # Fetch trending tokens
        print(f"\n{Fore.GREEN}📈 Fetching trending tokens from Bitfinex...")
        trending_df = api_client.fetch_trending_tokens()
        
        if not trending_df.empty:
            display_trending_tokens(trending_df)
            
            # Update historical tracking
            history_df = update_trending_history(trending_df)
            
            # Display consistent trending tokens
            if not history_df.empty:
                display_consistent_trending(history_df)
            
            # Display possible gems
            display_possible_gems(trending_df)
            
            # Save trending data with professional timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_data_to_csv(trending_df, f'bitfinex_trending_professional_{timestamp}.csv')
        else:
            print(f"{Fore.RED}❌ No trending data retrieved from Bitfinex")
        
        # Professional rate limiting
        time.sleep(config.API_DELAY)
        
        # Fetch new/emerging listings
        print(f"\n{Fore.GREEN}🆕 Fetching emerging tokens from Bitfinex...")
        new_listings_df = api_client.fetch_new_listings()
        
        if not new_listings_df.empty:
            display_new_listings(new_listings_df)
            
            # Save new listings data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_data_to_csv(new_listings_df, f'bitfinex_emerging_tokens_{timestamp}.csv')
        else:
            print(f"{Fore.RED}❌ No emerging tokens data available")
        
        # Professional rate limiting
        time.sleep(config.API_DELAY)
        
        # Fetch and display volume leaders
        print(f"\n{Fore.GREEN}📊 Fetching volume leaders from Bitfinex...")
        high_volume_df = api_client.fetch_high_volume_tokens()
        
        if not high_volume_df.empty:
            display_volume_leaders(high_volume_df)
            
            # Save volume data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_data_to_csv(high_volume_df, f'bitfinex_volume_leaders_professional_{timestamp}.csv')
        else:
            print(f"{Fore.RED}❌ No volume data available")
        
        # Professional feature: Funding rates analysis
        if config.INCLUDE_FUNDING_ANALYSIS:
            time.sleep(config.API_DELAY)
            print(f"\n{Fore.GREEN}💰 Fetching funding rates from Bitfinex (Professional Feature)...")
            funding_df = api_client.fetch_funding_rates()
            
            if not funding_df.empty:
                display_funding_rates(funding_df)
                
                # Save funding data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_data_to_csv(funding_df, f'bitfinex_funding_rates_{timestamp}.csv')
                
                # Professional insights
                high_funding = funding_df[funding_df['funding_rate'].abs() > config.FUNDING_RATE_THRESHOLD]
                if not high_funding.empty:
                    print(f"\n{Fore.YELLOW}🚨 PROFESSIONAL ALERT: {len(high_funding)} tokens with high funding rates detected!")
                    print(f"{Fore.YELLOW}💡 Consider arbitrage opportunities between spot and margin markets")
            else:
                print(f"{Fore.RED}❌ No funding rate data available")
        
        print(f"\n{Fore.GREEN}✅ Bitfinex Professional Token Tracker completed successfully!")
        print(f"{Fore.CYAN}💡 Professional data saved to: {config.DATA_DIR}")
        print(f"{Fore.MAGENTA}🏛️  Institutional analysis complete - {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ An error occurred in the Bitfinex professional bot: {e}")
        print(f"{Fore.RED}📝 Professional traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Parse command line arguments with professional options
    parser = argparse.ArgumentParser(description="🌙 Moon Dev's Bitfinex Professional Token Tracker")
    parser.add_argument('--interval', type=int, default=1800, help='Update interval in seconds (default: 1800s - 30 min for professional use)')
    parser.add_argument('--run-once', action='store_true', help='Run once and exit (no scheduling)')
    parser.add_argument('--save-data', action='store_true', help='Save data to CSV files', default=True)
    parser.add_argument('--gems-only', action='store_true', help='Focus on potential gems only')
    parser.add_argument('--funding-analysis', action='store_true', help='Include funding rate analysis', default=True)
    parser.add_argument('--professional-mode', action='store_true', help='Enable all professional features', default=True)
    parser.add_argument('--volume-threshold', type=float, help='Custom minimum volume threshold')
    parser.add_argument('--funding-threshold', type=float, help='Custom funding rate threshold for alerts')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.volume_threshold:
        config.MIN_VOLUME_24H = args.volume_threshold
        print(f"{Fore.CYAN}📊 Professional volume threshold set to: ${args.volume_threshold:,.0f}")
    
    if args.funding_threshold:
        config.FUNDING_RATE_THRESHOLD = args.funding_threshold
        print(f"{Fore.CYAN}💰 Professional funding rate threshold set to: {args.funding_threshold:.4f}%")
    
    if args.funding_analysis:
        config.INCLUDE_FUNDING_ANALYSIS = True
        print(f"{Fore.CYAN}💼 Professional funding analysis enabled")
    
    if args.professional_mode:
        config.PROFESSIONAL_MODE = True
        config.INCLUDE_FUNDING_ANALYSIS = True
        print(f"{Fore.MAGENTA}🏛️  Full professional mode activated")
    
    if args.run_once:
        # Run once and exit
        bot()
    else:
        # Schedule the bot to run at specified intervals (professional default: 30 minutes)
        schedule.every(args.interval).seconds.do(bot)
        
        print(f"{Fore.CYAN}🕒 Bitfinex Professional Token Tracker scheduled to run every {args.interval} seconds")
        print(f"{Fore.MAGENTA}🏛️  Professional mode - Conservative scheduling for institutional use")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop")
        
        # Initial run
        bot()
        
        # Keep the script running with professional monitoring
        try:
            while True:
                schedule.run_pending()
                time.sleep(120)  # Check every 2 minutes for professional use
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Bitfinex Professional Token Tracker stopped by user")
            print(f"{Fore.CYAN}📊 Professional session ended - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            print(f"\n{Fore.RED}❌ Unexpected professional error: {e}")
            print(f"{Fore.RED}📝 Professional traceback: {traceback.format_exc()}")
