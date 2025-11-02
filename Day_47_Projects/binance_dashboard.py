'''
🌙 Moon Dev's Binance Token Tracker 🚀
Built with love by Moon Dev 🌙 ✨

This script tracks trending tokens, new listings, and high-volume tokens on Binance
using the Binance API.

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
import binance_config as config  # Import the Binance configuration file
from binance_api_client import BinanceAPI # Import the Binance API client
from datetime import datetime, timedelta
from colorama import Fore
from binance_display import ( # Import display functions and constants
    BINANCE_BANNER, 
    BINANCE_QUOTES, 
    display_trending_tokens, 
    display_new_listings, 
    display_possible_gems, 
    display_consistent_trending,
    display_volume_leaders
)

try:
    import dontshare as d  # Import API key from dontshare.py
    BINANCE_API_KEY = getattr(d, 'BINANCE_API_KEY', None)
    BINANCE_API_SECRET = getattr(d, 'BINANCE_API_SECRET', None)
except ImportError:
    print("Warning: dontshare.py not found. API keys not loaded.")
    BINANCE_API_KEY = None
    BINANCE_API_SECRET = None

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas to display numbers with commas and no scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def get_random_quote():
    """Return a random Binance quote"""
    return random.choice(BINANCE_QUOTES)

def ensure_data_dir():
    """Ensure the data directory exists"""
    os.makedirs(config.DATA_DIR, exist_ok=True)

def save_data_to_csv(df, filename):
    """Save DataFrame to CSV file"""
    if not df.empty:
        ensure_data_dir()
        filepath = os.path.join(config.DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")

def update_trending_history(df):
    """Update historical tracking of trending tokens"""
    if df.empty:
        return
    
    ensure_data_dir()
    
    # Add timestamp
    df_with_timestamp = df.copy()
    df_with_timestamp['timestamp'] = datetime.now()
    df_with_timestamp['date'] = datetime.now().strftime('%Y-%m-%d')
    df_with_timestamp['hour'] = datetime.now().strftime('%H')
    
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
    
    # Keep only recent data (last 30 days)
    cutoff_date = datetime.now() - timedelta(days=30)
    if 'timestamp' in combined_df.columns:
        combined_df = combined_df[pd.to_datetime(combined_df['timestamp']) > cutoff_date]
    
    # Save updated history
    try:
        combined_df.to_csv(config.HISTORY_FILE, index=False)
        print(f"Updated trending history with {len(df_with_timestamp)} tokens")
    except Exception as e:
        print(f"Error saving history: {e}")
    
    return combined_df

def bot():
    """Main function to run the Binance token tracker"""
    try:
        print(BINANCE_BANNER)
        print(f"{Fore.GREEN}🎯 {get_random_quote()}")
        print(f"{Fore.CYAN}Starting Binance Token Tracker...")
        print(f"{Fore.WHITE}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize API client
        api_client = BinanceAPI(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
        
        # Fetch trending tokens
        print(f"\n{Fore.YELLOW}📈 Fetching trending tokens from Binance...")
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
            
            # Save trending data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_data_to_csv(trending_df, f'binance_trending_{timestamp}.csv')
        else:
            print(f"{Fore.RED}❌ No trending data retrieved from Binance")
        
        # Fetch new listings
        print(f"\n{Fore.YELLOW}🆕 Fetching new listings from Binance...")
        new_listings_df = api_client.fetch_new_listings()
        
        if not new_listings_df.empty:
            display_new_listings(new_listings_df)
            
            # Save new listings data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_data_to_csv(new_listings_df, f'binance_new_listings_{timestamp}.csv')
        else:
            print(f"{Fore.RED}❌ No new listings data available")
        
        # Fetch and display volume leaders
        print(f"\n{Fore.YELLOW}📊 Fetching volume leaders from Binance...")
        high_volume_df = api_client.fetch_high_volume_tokens()
        
        if not high_volume_df.empty:
            display_volume_leaders(high_volume_df)
            
            # Save volume data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_data_to_csv(high_volume_df, f'binance_volume_leaders_{timestamp}.csv')
        else:
            print(f"{Fore.RED}❌ No volume data available")
        
        print(f"\n{Fore.GREEN}✅ Binance Token Tracker completed successfully!")
        print(f"{Fore.CYAN}💡 Data saved to: {config.DATA_DIR}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ An error occurred in the Binance bot: {e}")
        print(f"{Fore.RED}📝 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="🌙 Moon Dev's Binance Token Tracker")
    parser.add_argument('--interval', type=int, default=3600, help='Update interval in seconds (default: 3600s)')
    parser.add_argument('--run-once', action='store_true', help='Run once and exit (no scheduling)')
    parser.add_argument('--save-data', action='store_true', help='Save data to CSV files', default=True)
    parser.add_argument('--gems-only', action='store_true', help='Focus on potential gems only')
    parser.add_argument('--volume-threshold', type=float, help='Custom minimum volume threshold')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.volume_threshold:
        config.MIN_VOLUME_24H = args.volume_threshold
    
    if args.run_once:
        # Run once and exit
        bot()
    else:
        # Schedule the bot to run at specified intervals
        schedule.every(args.interval).seconds.do(bot)
        
        print(f"{Fore.CYAN}🕒 Binance Token Tracker scheduled to run every {args.interval} seconds")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop")
        
        # Initial run
        bot()
        
        # Keep the script running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}👋 Binance Token Tracker stopped by user")
        except Exception as e:
            print(f"\n{Fore.RED}❌ Unexpected error: {e}")
            print(f"{Fore.RED}📝 Traceback: {traceback.format_exc()}")
