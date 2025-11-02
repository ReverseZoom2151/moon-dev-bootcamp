'''
🌙 Moon Dev's Solana Token Tracker 🚀
Built with love by Moon Dev 🌙 ✨

This script tracks trending tokens, new listings, and high-volume tokens on Solana
using the Birdeye API.

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.
'''

import os
import time
import pandas as pd
from datetime import datetime, timedelta
import colorama
from colorama import Fore
import argparse
import traceback
import random
import schedule
import dontshare as d  # Import API key from dontshare.py
import pytz
import config  # Import the new configuration file
from api_client import BirdeyeAPI # Import the new API client
from display import ( # Import display functions and constants
    MOON_DEV_BANNER, 
    MOON_DEV_QUOTES, 
    display_trending_tokens, 
    display_new_listings, 
    display_possible_gems, 
    display_consistent_trending
)

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas to display numbers with commas and no scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Moon Dev ASCII Art Banner
MOON_DEV_BANNER = f"""{Fore.CYAN}
  __  __  ___   ___  _  _    ___  ___ _   _ 
 |  \\/  |/ _ \\ / _ \\| \\| |  |   \\| __\\ \\ / /
 | |\\/| | (_) | (_) | .` |  | |) | _| \\ V / 
 |_|  |_|\\___/ \\___/|_|\\_|  |___/|___| \\_/  
                                             
{Fore.MAGENTA}🚀 Solana Token Tracker 🌙{Fore.RESET}
"""

# Fun Moon Dev quotes
MOON_DEV_QUOTES = [
    "Tracking tokens like a boss! 🪙",
    "New listings incoming... maybe! 🚀",
    "Moon Dev API - making data fun again! 🎯",
    "To the moon! 🌕 (not financial advice)",
    "Who needs sleep when you have tokens? 😴",
    "Tokens move markets, so track the tokens! 📊",
    "Moon Dev sees all the tokens... 👀",
    "Diamond hands or paper hands? 💎",
    "Watch the tokens, follow the money! 💰",
    "API-powered alpha at your fingertips! ✨"
]

# ===== CONFIGURATION =====
# Addresses to ignore in consistently trending display
IGNORE_LIST = [
    "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh",  # Wrapped BTC
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",   # Jupiter
    "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # Bonk
    "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",
    "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"   
]

DATA_DIR = "data/raw"  # Directory for saving data
HISTORY_FILE = "data/raw/trending_history.csv"  # File to track historical trending appearances
MIN_VOLUME_24H = 1000000  # Minimum 24h volume to consider
DISPLAY_TRENDING_TOKENS = 40  # Number of trending tokens to display (API limit: 20)
DISPLAY_NEW_LISTINGS = 40  # Number of new listings to display (API limit: 20)
GEMS_MAX_MARKET_CAP = 1000000  # Maximum market cap for possible gems ($1M)
TOP_CONSISTENT_TOKENS = 10  # Number of most consistent trending tokens to display
API_KEY = d.birdeye_api_key  # Get API key from dontshare.py

def get_random_quote():
    """Return a random Moon Dev quote"""
    return random.choice(MOON_DEV_QUOTES)

def ensure_data_dir():
    """Ensure the data directory exists"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    return True

def save_data_to_csv(df, filename):
    """Save DataFrame to CSV file"""
    if df.empty:
        return
    filepath = os.path.join(config.DATA_DIR, filename)
    df.to_csv(filepath, index=False)

def update_trending_history(df):
    """Update historical tracking of trending tokens"""
    try:
        # Create history DataFrame if it doesn't exist
        if os.path.exists(config.HISTORY_FILE):
            # Read CSV with appearances as integer type
            history_df = pd.read_csv(config.HISTORY_FILE, dtype={'appearances': int})
            # Add last_hour_counted column if it doesn't exist
            if 'last_hour_counted' not in history_df.columns:
                history_df['last_hour_counted'] = pd.NaT
            else:
                history_df['last_hour_counted'] = pd.to_datetime(history_df['last_hour_counted'])
        else:
            history_df = pd.DataFrame(columns=['address', 'symbol', 'name', 'appearances', 'last_seen', 'last_hour_counted'])
            history_df['appearances'] = history_df['appearances'].astype(int)
            history_df['last_hour_counted'] = pd.NaT
        
        # Get current timestamp in ET
        eastern = pytz.timezone('America/New_York')
        current_time = datetime.now(eastern)
        current_time_str = current_time.strftime('%Y-%m-%d %I:%M %p')
        
        # Update appearances for current trending tokens
        for _, row in df.iterrows():
            address = row.get('address')
            if address:
                # If token exists in history, update it
                if address in history_df['address'].values:
                    mask = history_df['address'] == address
                    last_counted = history_df.loc[mask, 'last_hour_counted'].iloc[0]
                    
                    # Only increment if it's been at least an hour since last count
                    if pd.isna(last_counted) or (current_time - last_counted) >= timedelta(hours=1):
                        history_df.loc[mask, 'appearances'] = history_df.loc[mask, 'appearances'].astype(int) + 1
                        print(f"🌙 Moon Dev counted {row.get('symbol', 'N/A')} as trending! 🚀")
                    
                    # Always update the timestamps
                    history_df.loc[mask, 'last_seen'] = current_time_str
                    history_df.loc[mask, 'last_hour_counted'] = current_time
                # If token is new, add it
                else:
                    new_row = {
                        'address': address,
                        'symbol': row.get('symbol', 'N/A'),
                        'name': row.get('name', 'N/A'),
                        'appearances': 1,
                        'last_seen': current_time_str,
                        'last_hour_counted': current_time
                    }
                    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
                    print(f"🌙 Moon Dev found new trending token: {row.get('symbol', 'N/A')} 🎯")
        
        # Ensure appearances is integer type before saving
        history_df['appearances'] = history_df['appearances'].astype(int)
        
        # Save updated history
        history_df.to_csv(config.HISTORY_FILE, index=False)
        return history_df
        
    except Exception as e:
        print(f"Error in update_trending_history: {str(e)}")
        print(f"📋 Stack trace:\n{traceback.format_exc()}")
        return pd.DataFrame()

def bot():
    """Main function to run the token tracker"""
    print(MOON_DEV_BANNER)
    print(f"{Fore.MAGENTA}{get_random_quote()}")
    
    ensure_data_dir()
    
    api = BirdeyeAPI(d.birdeye_api_key) # Initialize the API client

    new_listings_df = api.fetch_new_listings()
    if not new_listings_df.empty:
        display_new_listings(new_listings_df)
        save_data_to_csv(new_listings_df, "new_listings.csv")
    
    trending_df = api.fetch_trending_tokens()
    if not trending_df.empty:
        # Update historical tracking
        history_df = update_trending_history(trending_df)
        
        # Display all tables in order:
        # 1. New Listings (already displayed)
        # 2. Trending Tokens
        display_trending_tokens(trending_df)
        
        # 3. Consistently Trending
        if not history_df.empty:
            display_consistent_trending(history_df)
            
        # 4. Possible Gems (always last)
        display_possible_gems(trending_df)
    
    print(f"\n{Fore.MAGENTA}{get_random_quote()}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="🌙 Moon Dev's Solana Token Tracker")
    parser.add_argument('--interval', type=int, default=3600, help='Update interval in seconds (default: 3600s)')
    args = parser.parse_args()
    
    # Initial run
    bot()
    
    # Schedule the main function to run at specified interval
    schedule.every(args.interval).seconds.do(bot)
    
    while True:
        try:
            # Run pending scheduled tasks
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            print(f"{Fore.RED}Encountered an error: {e}")
            print(f"{Fore.RED}📋 Stack trace:\n{traceback.format_exc()}")
            # Wait before retrying to avoid rapid error logging
            time.sleep(10)
