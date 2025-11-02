"""
Solana Token Tracker Service
---
Integrates the functionality of the Day 47 Solana Token Tracker project,
including API client, display functions, and main tracking logic.
"""

import os
import time
import pandas as pd
import asyncio
import traceback
import random
import pytz
import requests
from autonomous_trading_system.backend.core.config import get_settings
from datetime import datetime, timedelta
from colorama import Fore, init

# Initialize colorama for terminal colors
init(autoreset=True)

# Configure pandas for better display
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

settings = get_settings()

# --- Display Content ---
MOON_DEV_BANNER = f"""{Fore.CYAN}
  __  __  ___   ___  _  _    ___  ___ _   _ 
 |  \/  |/ _ \ / _ \| \| |  |   \| __\ \ / /
 | |\/| | (_) | (_) | .` |  | |) | _| \ V / 
 |_|  |_|\___/ \___/|_|\_|  |___/|___| \_/  
                                             
{Fore.MAGENTA}🚀 Solana Token Tracker 🌙{Fore.RESET}
"""

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

# --- Birdeye API Client ---
class BirdeyeAPI:
    """A client for the Birdeye API."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://public-api.birdeye.so/defi"

    def _make_request(self, endpoint, params=None):
        headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": self.api_key
        }
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}Birdeye API request failed: {e}{Fore.RESET}")
            return None

    def fetch_trending_tokens(self):
        all_tokens = []
        offset = 0
        try:
            while len(all_tokens) < settings.SOLANA_TOKEN_TRACKER_DISPLAY_TRENDING_TOKENS:
                params = {
                    "sort_by": "rank",
                    "sort_type": "asc",
                    "offset": offset,
                    "limit": 20
                }
                data = self._make_request("/token_trending", params=params)
                if not data: break
                
                tokens = data.get('data', {}).get('tokens') or data.get('data', {}).get('items')
                if not tokens: break
                    
                all_tokens.extend(tokens)
                offset += 20
                time.sleep(0.5)
                
                if offset >= 100: break
            
            return pd.DataFrame(all_tokens[:settings.SOLANA_TOKEN_TRACKER_DISPLAY_TRENDING_TOKENS])
        except Exception as e:
            print(f"{Fore.RED}Error fetching trending tokens: {e}{Fore.RESET}")
            return pd.DataFrame()

    def fetch_new_listings(self):
        all_listings = []
        offset = 0
        try:
            while len(all_listings) < settings.SOLANA_TOKEN_TRACKER_DISPLAY_NEW_LISTINGS:
                params = {
                    "offset": offset,
                    "limit": 20,
                    "meme_platform_enabled": "false"
                }
                data = self._make_request("/v2/tokens/new_listing", params=params)
                if not data: break
                
                listings = data.get('data', {}).get('items')
                if not listings: break
                    
                all_listings.extend(listings)
                offset += 20
                time.sleep(0.5)
                
                if offset >= 100: break
            
            return pd.DataFrame(all_listings[:settings.SOLANA_TOKEN_TRACKER_DISPLAY_NEW_LISTINGS])
        except Exception as e:
            print(f"{Fore.RED}Error fetching new listings: {e}{Fore.RESET}")
            return pd.DataFrame()

# --- Display Functions ---
class TokenDisplay:
    """Handles the terminal display of token data."""

    @staticmethod
    def get_random_quote():
        return random.choice(MOON_DEV_QUOTES)

    @staticmethod
    def display_trending_tokens(df):
        if df.empty: return
        print(f"\n{Fore.CYAN}{'='*150}")
        print(f"{Fore.CYAN}🚀 TRENDING TOKENS ({len(df)} tokens) 🚀")
        print(f"{Fore.CYAN}{'='*150}")
        
        header = (f"{Fore.YELLOW}{'Rank':<5} | {'Symbol':<10} | {'Name':<20} | {'24h Change':>12} | "
                 f"{'Volume 24h':>15} | {'Market Cap':>15} | {'Contract Address':<44}")
        print(header)
        print(f"{Fore.CYAN}{'-'*150}")
        
        for _, row in df.head(settings.SOLANA_TOKEN_TRACKER_DISPLAY_TRENDING_TOKENS).iterrows():
            price_change_24h = row.get('price24hChangePercent', 0)
            price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
            print(f"{Fore.WHITE}{row.get('rank', 'N/A'):<5} | "
                  f"{Fore.YELLOW}{str(row.get('symbol', 'N/A')):<10.10} | "
                  f"{Fore.CYAN}{str(row.get('name', 'N/A')):<20.20} | "
                  f"{price_change_color}{row.get('price24hChangePercent', 0):+.2f}% | "
                  f"{Fore.MAGENTA}${row.get('volume24hUSD', 0):>14,.0f} | "
                  f"{Fore.BLUE}${row.get('marketcap', 0):>14,.0f} | "
                  f"{Fore.GREEN}{row.get('address', 'N/A'):<44}")

    @staticmethod
    def display_new_listings(df):
        if df.empty: return
        print(f"\n{Fore.CYAN}{'='*150}")
        print(f"{Fore.CYAN}🌟 NEW LISTINGS ON SOLANA ({len(df)} tokens) 🌟")
        print(f"{Fore.CYAN}{'='*150}")
        
        header = (f"{Fore.YELLOW}{'Symbol':<10} | {'Name':<20} | {'Liquidity':>12} | "
                 f"{'Listed At (ET)':<22} | {'Contract Address':<44}")
        print(header)
        print(f"{Fore.CYAN}{'-'*150}")
        
        utc, eastern = pytz.UTC, pytz.timezone('America/New_York')
        for _, row in df.iterrows():
            try:
                listed_time_str = "N/A"
                listed_time = row.get('liquidityAddedAt')
                if listed_time:
                    utc_time = utc.localize(datetime.strptime(listed_time, "%Y-%m-%dT%H:%M:%S"))
                    eastern_time = utc_time.astimezone(eastern)
                    listed_time_str = eastern_time.strftime('%Y-%m-%d %I:%M %p')
                
                print(f"{Fore.YELLOW}{str(row.get('symbol', 'N/A')):<10.10} | "
                      f"{Fore.CYAN}{str(row.get('name', 'N/A')):<20.20} | "
                      f"{Fore.GREEN}${row.get('liquidity', 0):>11,.2f} | "
                      f"{Fore.WHITE}{listed_time_str:<22} | "
                      f"{Fore.BLUE}{row.get('address', 'N/A'):<44}")
            except Exception:
                continue

    @staticmethod
    def display_possible_gems(df):
        gems_df = df[df['marketcap'].fillna(float('inf')) <= settings.SOLANA_TOKEN_TRACKER_GEMS_MAX_MARKET_CAP].copy()
        if gems_df.empty: return
        
        print(f"\n{Fore.CYAN}{'='*150}")
        print(f"{Fore.CYAN}💎 POSSIBLE GEMS (Market Cap < ${settings.SOLANA_TOKEN_TRACKER_GEMS_MAX_MARKET_CAP:,.0f}) 💎")
        print(f"{Fore.CYAN}{'='*150}")
        
        header = (f"{Fore.YELLOW}{'Rank':<5} | {'Symbol':<10} | {'Name':<20} | {'24h Change':>12} | "
                 f"{'Volume 24h':>15} | {'Market Cap':>15} | {'Contract Address':<44}")
        print(header)
        print(f"{Fore.CYAN}{'-'*150}")
        
        for _, row in gems_df.iterrows():
            price_change_24h = row.get('price24hChangePercent', 0)
            price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
            print(f"{Fore.WHITE}{row.get('rank', 'N/A'):<5} | "
                  f"{Fore.YELLOW}{str(row.get('symbol', 'N/A')):<10.10} | "
                  f"{Fore.CYAN}{str(row.get('name', 'N/A')):<20.20} | "
                  f"{price_change_color}{row.get('price24hChangePercent', 0):+.2f}% | "
                  f"{Fore.MAGENTA}${row.get('volume24hUSD', 0):>14,.0f} | "
                  f"{Fore.BLUE}${row.get('marketcap', 0):>14,.0f} | "
                  f"{Fore.GREEN}{row.get('address', 'N/A'):<44}")

    @staticmethod
    def display_consistent_trending(history_df):
        if history_df.empty: return
            
        print(f"\n{Fore.CYAN}{'='*110}")
        print(f"{Fore.CYAN}🏆 CONSISTENTLY TRENDING 🏆")
        print(f"{Fore.CYAN}{'='*110}")
        
        filtered_df = history_df[~history_df['address'].isin(settings.SOLANA_TOKEN_TRACKER_IGNORE_LIST)]
        top_tokens = filtered_df.nlargest(20, 'appearances')
        
        left_tokens = top_tokens.iloc[:settings.SOLANA_TOKEN_TRACKER_TOP_CONSISTENT_TOKENS]
        right_tokens = top_tokens.iloc[settings.SOLANA_TOKEN_TRACKER_TOP_CONSISTENT_TOKENS:settings.SOLANA_TOKEN_TRACKER_TOP_CONSISTENT_TOKENS*2]
        
        header = f"{Fore.YELLOW}{'Name':<25} | {'Contract Address':<44}    {'Name':<25} | {'Contract Address':<44}"
        print(header)
        print(f"{Fore.CYAN}{'-'*71}    {'-'*71}")
        
        for i in range(settings.SOLANA_TOKEN_TRACKER_TOP_CONSISTENT_TOKENS):
            left_str = " " * 71
            if i < len(left_tokens):
                row = left_tokens.iloc[i]
                display_name = f"{str(row.get('name', 'N/A'))[:20]} ({int(row.get('appearances', 0))})"
                left_str = f"{Fore.CYAN}{display_name:<25} | {Fore.WHITE}{row.get('address', 'N/A'):<44}"
            
            right_str = ""
            if i < len(right_tokens):
                row = right_tokens.iloc[i]
                display_name = f"{str(row.get('name', 'N/A'))[:20]} ({int(row.get('appearances', 0))})"
                right_str = f"{Fore.CYAN}{display_name:<25} | {Fore.WHITE}{row.get('address', 'N/A'):<44}"

            print(f"{left_str}    {right_str}")

# --- Main Service ---
class SolanaTokenTrackerService:
    """The main service for tracking Solana tokens."""
    def __init__(self):
        self.settings = get_settings()
        self.api = BirdeyeAPI(self.settings.BIRDEYE_API_KEY)
        self.display = TokenDisplay()
        self.data_dir = os.path.join(self.settings.BASE_DIR, self.settings.SOLANA_TOKEN_TRACKER_DATA_DIR)
        self.history_file = os.path.join(self.settings.BASE_DIR, self.settings.SOLANA_TOKEN_TRACKER_HISTORY_FILE)
        self.ensure_data_dir()
        
        # In-memory cache for API endpoints
        self.latest_trending_df = pd.DataFrame()
        self.latest_new_listings_df = pd.DataFrame()
        self.latest_history_df = pd.DataFrame()
        self.status = "initialized"
        self.last_run_timestamp = None

    def ensure_data_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)

    def save_data_to_csv(self, df, filename):
        if not df.empty:
            filepath = os.path.join(self.data_dir, filename)
            df.to_csv(filepath, index=False)

    def update_trending_history(self, df):
        try:
            if os.path.exists(self.history_file):
                history_df = pd.read_csv(self.history_file, dtype={'appearances': int})
                if 'last_hour_counted' not in history_df.columns:
                    history_df['last_hour_counted'] = pd.NaT
                else:
                    history_df['last_hour_counted'] = pd.to_datetime(history_df['last_hour_counted'])
            else:
                history_df = pd.DataFrame(columns=['address', 'symbol', 'name', 'appearances', 'last_seen', 'last_hour_counted'])
            
            eastern, current_time = pytz.timezone('America/New_York'), datetime.now(pytz.timezone('America/New_York'))
            
            for _, row in df.iterrows():
                address = row.get('address')
                if not address: continue

                mask = history_df['address'] == address
                if mask.any():
                    last_counted = history_df.loc[mask, 'last_hour_counted'].iloc[0]
                    if pd.isna(last_counted) or (current_time - last_counted) >= timedelta(hours=1):
                        history_df.loc[mask, 'appearances'] += 1
                        print(f"🌙 Moon Dev counted {row.get('symbol', 'N/A')} as trending! 🚀")
                    history_df.loc[mask, 'last_seen'] = current_time.strftime('%Y-%m-%d %I:%M %p')
                    history_df.loc[mask, 'last_hour_counted'] = current_time
                else:
                    new_row = pd.DataFrame([{'address': address, 'symbol': row.get('symbol', 'N/A'),
                                 'name': row.get('name', 'N/A'), 'appearances': 1,
                                 'last_seen': current_time.strftime('%Y-%m-%d %I:%M %p'),
                                 'last_hour_counted': current_time}])
                    history_df = pd.concat([history_df, new_row], ignore_index=True)
                    print(f"🌙 Moon Dev found new trending token: {row.get('symbol', 'N/A')} 🎯")
            
            history_df['appearances'] = history_df['appearances'].astype(int)
            history_df.to_csv(self.history_file, index=False)
            return history_df
            
        except Exception as e:
            print(f"{Fore.RED}Error in update_trending_history: {e}\n{traceback.format_exc()}{Fore.RESET}")
            return pd.DataFrame()

    def run_tracker_iteration(self):
        """Main function to run one iteration of the token tracker."""
        self.status = "running"
        print(MOON_DEV_BANNER)
        print(f"{Fore.MAGENTA}{self.display.get_random_quote()}{Fore.RESET}")
        
        new_listings_df = self.api.fetch_new_listings()
        if not new_listings_df.empty:
            self.display.display_new_listings(new_listings_df)
            self.save_data_to_csv(new_listings_df, "new_listings.csv")
            self.latest_new_listings_df = new_listings_df # Cache data
        
        trending_df = self.api.fetch_trending_tokens()
        if not trending_df.empty:
            history_df = self.update_trending_history(trending_df)
            self.display.display_trending_tokens(trending_df)
            
            # Cache data
            self.latest_trending_df = trending_df
            self.latest_history_df = history_df
            
            if history_df is not None and not history_df.empty:
                self.display.display_consistent_trending(history_df)
            self.display.display_possible_gems(trending_df)
        
        print(f"\n{Fore.MAGENTA}{self.display.get_random_quote()}{Fore.RESET}")
        self.status = "idle"
        self.last_run_timestamp = datetime.now(pytz.utc)

    async def run_as_background_task(self):
        """Runs the tracker in a loop as a background task."""
        self.status = "starting"
        while True:
            try:
                if self.settings.ENABLE_SOLANA_TOKEN_TRACKER:
                    self.run_tracker_iteration()
                else:
                    print("Solana Token Tracker is disabled in settings.")
                
                interval = self.settings.SOLANA_TOKEN_TRACKER_INTERVAL_SECONDS
                print(f"Tracker iteration finished. Sleeping for {interval} seconds.")
                await asyncio.sleep(interval)

            except Exception as e:
                self.status = f"error: {e}"
                print(f"{Fore.RED}An error occurred in the main tracker loop: {e}{Fore.RESET}")
                print(f"📋 Stack trace:\n{traceback.format_exc()}{Fore.RESET}")
                await asyncio.sleep(60)
