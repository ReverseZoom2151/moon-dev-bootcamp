'''Display functions for the Solana Token Tracker'''

from colorama import Fore, init
import pytz
from datetime import datetime
import config

# Initialize colorama for terminal colors
init(autoreset=True)

# Moon Dev ASCII Art Banner
MOON_DEV_BANNER = f"""{Fore.CYAN}
  __  __  ___   ___  _  _    ___  ___ _   _ 
 |  \/  |/ _ \ / _ \| \| |  |   \| __\ \ / /
 | |\/| | (_) | (_) | .` |  | |) | _| \ V / 
 |_|  |_|\___/ \___/|_|\_|  |___/|___| \_/  
                                             
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

def display_trending_tokens(df):
    """Display trending tokens in a beautiful format"""
    if df.empty:
        return
        
    print(f"\n{Fore.CYAN}{'='*150}")
    print(f"{Fore.CYAN}🚀 TRENDING TOKENS ({len(df)} tokens) 🚀")
    print(f"{Fore.CYAN}{'='*150}")
    
    header = (f"{Fore.YELLOW}{'Rank':<5} | {'Symbol':<10} | {'Name':<20} | {'24h Change':>12} | "
             f"{'Volume 24h':>15} | {'Market Cap':>15} | {'Contract Address':<44}")
    separator = f"{Fore.CYAN}{'-'*5}-+-{'-'*10}-+-{'-'*20}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*44}"
    
    print(header)
    print(separator)
    
    for _, row in df.head(config.DISPLAY_TRENDING_TOKENS).iterrows():
        try:
            rank = str(row.get('rank', 'N/A'))
            symbol = str(row.get('symbol', 'N/A'))[:10]
            name = str(row.get('name', 'N/A'))[:20]
            address = str(row.get('address', 'N/A'))
            
            try:
                price_change_24h = float(row.get('price24hChangePercent', 0))
                price_change_str = f"{price_change_24h:+.2f}%"
            except (ValueError, TypeError):
                price_change_str = "N/A"
            price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
            
            try:
                volume_24h = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
                
            try:
                market_cap = f"${float(row.get('marketcap', 0)):,.0f}"
            except (ValueError, TypeError):
                market_cap = "N/A"
            
            print(f"{Fore.WHITE}{rank:<5} | "
                  f"{Fore.YELLOW}{symbol:<10} | "
                  f"{Fore.CYAN}{name:<20} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.MAGENTA}{volume_24h:>15} | "
                  f"{Fore.BLUE}{market_cap:>15} | "
                  f"{Fore.GREEN}{address:<44}")
                  
        except Exception:
            continue

def display_new_listings(df):
    """Display new token listings in a beautiful format"""
    if df.empty:
        return
        
    print(f"\n{Fore.CYAN}{'='*150}")
    print(f"{Fore.CYAN}🌟 NEW LISTINGS ON SOLANA ({len(df)} tokens) 🌟")
    print(f"{Fore.CYAN}{'='*150}")
    
    header = (f"{Fore.YELLOW}{'Symbol':<10} | {'Name':<20} | {'Liquidity':>12} | "
             f"{'Listed At (ET)':<22} | {'Contract Address':<44}")
    separator = f"{Fore.CYAN}{'-'*10}-+-{'-'*20}-+-{'-'*12}-+-{'-'*22}-+-{'-'*44}"
    
    print(header)
    print(separator)
    
    utc = pytz.UTC
    eastern = pytz.timezone('America/New_York')
    
    for _, row in df.iterrows():
        try:
            symbol = str(row.get('symbol', 'N/A') or 'N/A')[:10]
            name = str(row.get('name', 'N/A') or 'N/A')[:20]
            address = str(row.get('address', 'N/A') or 'N/A')
            
            try:
                liquidity = float(row.get('liquidity', 0) or 0)
                liquidity_str = f"${liquidity:,.2f}"
            except (ValueError, TypeError):
                liquidity_str = "N/A"
            
            try:
                listed_time = row.get('liquidityAddedAt', 'N/A')
                if listed_time != 'N/A':
                    utc_time = datetime.strptime(listed_time, "%Y-%m-%dT%H:%M:%S")
                    utc_time = utc.localize(utc_time)
                    eastern_time = utc_time.astimezone(eastern)
                    listed_time = eastern_time.strftime('%Y-%m-%d %I:%M %p')
            except (ValueError, TypeError):
                listed_time = "N/A"
            
            print(f"{Fore.YELLOW}{symbol:<10} | "
                  f"{Fore.CYAN}{name:<20} | "
                  f"{Fore.GREEN}{liquidity_str:>12} | "
                  f"{Fore.WHITE}{listed_time:<22} | "
                  f"{Fore.BLUE}{address:<44}")
                  
        except Exception:
            continue

def display_possible_gems(df):
    """Display possible gem tokens (trending tokens with market cap under threshold)"""
    if df.empty:
        return
        
    # Filter for tokens under the market cap threshold
    gems_df = df[df['marketcap'].fillna(float('inf')) <= config.GEMS_MAX_MARKET_CAP].copy()
    
    if gems_df.empty:
        return
        
    print(f"\n{Fore.CYAN}{'='*150}")
    print(f"{Fore.CYAN}💎 POSSIBLE GEMS (Market Cap < ${config.GEMS_MAX_MARKET_CAP:,.0f}) 💎")
    print(f"{Fore.CYAN}{'='*150}")
    
    header = (f"{Fore.YELLOW}{'Rank':<5} | {'Symbol':<10} | {'Name':<20} | {'24h Change':>12} | "
             f"{'Volume 24h':>15} | {'Market Cap':>15} | {'Contract Address':<44}")
    separator = f"{Fore.CYAN}{'-'*5}-+-{'-'*10}-+-{'-'*20}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*44}"
    
    print(header)
    print(separator)
    
    for _, row in gems_df.iterrows():
        try:
            rank = str(row.get('rank', 'N/A'))
            symbol = str(row.get('symbol', 'N/A'))[:10]
            name = str(row.get('name', 'N/A'))[:20]
            address = str(row.get('address', 'N/A'))
            
            try:
                price_change_24h = float(row.get('price24hChangePercent', 0))
                price_change_str = f"{price_change_24h:+.2f}%"
            except (ValueError, TypeError):
                price_change_str = "N/A"
            price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
            
            try:
                volume_24h = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
                
            try:
                market_cap = f"${float(row.get('marketcap', 0)):,.0f}"
            except (ValueError, TypeError):
                market_cap = "N/A"
            
            print(f"{Fore.WHITE}{rank:<5} | "
                  f"{Fore.YELLOW}{symbol:<10} | "
                  f"{Fore.CYAN}{name:<20} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.MAGENTA}{volume_24h:>15} | "
                  f"{Fore.BLUE}{market_cap:>15} | "
                  f"{Fore.GREEN}{address:<44}")
                  
        except Exception:
            continue

def display_consistent_trending(history_df):
    """Display tokens that consistently appear in trending"""
    if history_df.empty:
        return
        
    print(f"\n{Fore.CYAN}{'='*110}")
    print(f"{Fore.CYAN}🏆 CONSISTENTLY TRENDING 🏆")
    print(f"{Fore.CYAN}{'='*110}")
    
    # Filter out ignored addresses and sort by appearances
    filtered_df = history_df[~history_df['address'].isin(config.IGNORE_LIST)]
    top_tokens = filtered_df.nlargest(20, 'appearances') # Fetch top 20 for two columns
    
    # Split into left and right columns (10 each)
    left_tokens = top_tokens.iloc[:config.TOP_CONSISTENT_TOKENS]
    right_tokens = top_tokens.iloc[config.TOP_CONSISTENT_TOKENS:config.TOP_CONSISTENT_TOKENS*2]
    
    # Create header for both columns
    header = f"{Fore.YELLOW}{'Name':<25} | {'Contract Address':<44}    "
    header += f"{Fore.YELLOW}{'Name':<25} | {'Contract Address':<44}"
    separator = f"{Fore.CYAN}{'-'*25}-+-{'-'*44}    {Fore.CYAN}{'-'*25}-+-{'-'*44}"
    
    print(header)
    print(separator)
    
    # Print both columns side by side, up to TOP_CONSISTENT_TOKENS rows
    for i in range(config.TOP_CONSISTENT_TOKENS):
        left_str = " " * 71 # Default to empty string
        right_str = " " * 71 # Default to empty string

        # Left column
        if i < len(left_tokens):
            left_row = left_tokens.iloc[i]
            left_name = str(left_row.get('name', 'N/A'))[:20]
            left_times = int(left_row.get('appearances', 0))
            left_address = str(left_row.get('address', 'N/A'))
            left_display_name = f"{left_name} ({left_times})"
            left_str = f"{Fore.CYAN}{left_display_name:<25} | {Fore.WHITE}{left_address:<44}"
            
        # Right column
        if i < len(right_tokens):
            right_row = right_tokens.iloc[i]
            right_name = str(right_row.get('name', 'N/A'))[:20]
            right_times = int(right_row.get('appearances', 0))
            right_address = str(right_row.get('address', 'N/A'))
            right_display_name = f"{right_name} ({right_times})"
            right_str = f"{Fore.CYAN}{right_display_name:<25} | {Fore.WHITE}{right_address:<44}"
            
        print(f"{left_str}    {right_str}") 