'''Display functions for the Binance Token Tracker'''

import binance_config as config
from colorama import Fore, init

# Initialize colorama for terminal colors
init(autoreset=True)

# Binance ASCII Art Banner
BINANCE_BANNER = f"""{Fore.YELLOW}
  ____  _                            
 |  _ \(_)                           
 | |_) |_ _ __   __ _ _ __   ___ ___  
 |  _ <| | '_ \ / _` | '_ \ / __/ _ \ 
 | |_) | | | | | (_| | | | | (_|  __/
 |____/|_|_| |_|\__,_|_| |_|\___\___|
                                     
{Fore.MAGENTA}🚀 Binance Token Tracker 🌙{Fore.RESET}
"""

# Fun Binance trading quotes
BINANCE_QUOTES = [
    "Tracking Binance like a boss! 🪙",
    "USDT pairs incoming... maybe! 🚀",
    "Binance API - making data fun again! 🎯",
    "To the moon on Binance! 🌕 (not financial advice)",
    "Who needs sleep when you have altcoins? 😴",
    "Altcoins move markets, so track the pairs! 📊",
    "Binance sees all the trades... 👀",
    "Diamond hands or paper hands? 💎",
    "Watch the pairs, follow the volume! 💰",
    "API-powered alpha from Binance! ✨"
]

def display_trending_tokens(df):
    """Display trending tokens in a beautiful format"""
    if df.empty:
        print(f"{Fore.RED}No trending tokens data available from Binance.")
        return
        
    print(f"\n{Fore.YELLOW}{'='*150}")
    print(f"{Fore.YELLOW}🚀 BINANCE TRENDING TOKENS ({len(df)} pairs) 🚀")
    print(f"{Fore.YELLOW}{'='*150}")
    
    header = (f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Name':<15} | {'Price':>12} | {'24h Change':>12} | "
             f"{'Volume 24h':>15} | {'Trades':>10} | {'High 24h':>12} | {'Low 24h':>12}")
    separator = f"{Fore.YELLOW}{'-'*5}-+-{'-'*12}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}"
    
    print(header)
    print(separator)
    
    for _, row in df.head(config.DISPLAY_TRENDING_TOKENS).iterrows():
        try:
            rank = str(row.get('rank', 'N/A'))
            symbol = str(row.get('symbol', 'N/A'))[:12]
            name = str(row.get('name', 'N/A'))[:15]
            
            try:
                price = f"${float(row.get('price', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                price = "N/A"
            
            try:
                price_change_24h = float(row.get('price24hChangePercent', 0))
                price_change_str = f"{price_change_24h:+.2f}%"
            except (ValueError, TypeError):
                price_change_str = "N/A"
                price_change_24h = 0
            price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
            
            try:
                volume_24h = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
            
            try:
                trades_24h = f"{int(row.get('trades24h', 0)):,}"
            except (ValueError, TypeError):
                trades_24h = "N/A"
            
            try:
                high_24h = f"${float(row.get('high24h', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                high_24h = "N/A"
                
            try:
                low_24h = f"${float(row.get('low24h', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                low_24h = "N/A"
            
            print(f"{Fore.WHITE}{rank:<5} | "
                  f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.CYAN}{name:<15} | "
                  f"{Fore.GREEN}{price:>12} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.MAGENTA}{volume_24h:>15} | "
                  f"{Fore.BLUE}{trades_24h:>10} | "
                  f"{Fore.GREEN}{high_24h:>12} | "
                  f"{Fore.RED}{low_24h:>12}")
                  
        except Exception:
            continue

def display_new_listings(df):
    """Display new token listings in a beautiful format"""
    if df.empty:
        print(f"{Fore.RED}No new listings data available from Binance.")
        return
        
    print(f"\n{Fore.YELLOW}{'='*140}")
    print(f"{Fore.YELLOW}🌟 NEW BINANCE LISTINGS ({len(df)} pairs) 🌟")
    print(f"{Fore.YELLOW}{'='*140}")
    
    header = (f"{Fore.CYAN}{'Symbol':<12} | {'Name':<15} | {'Price':>12} | {'24h Change':>12} | "
             f"{'Volume 24h':>15} | {'Trades':>10} | {'Status':<10}")
    separator = f"{Fore.YELLOW}{'-'*12}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*10}-+-{'-'*10}"
    
    print(header)
    print(separator)
    
    for _, row in df.iterrows():
        try:
            symbol = str(row.get('symbol', 'N/A'))[:12]
            name = str(row.get('name', 'N/A'))[:15]
            
            try:
                price = f"${float(row.get('price', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                price = "N/A"
            
            try:
                price_change_24h = float(row.get('price24hChangePercent', 0))
                price_change_str = f"{price_change_24h:+.2f}%"
            except (ValueError, TypeError):
                price_change_str = "N/A"
                price_change_24h = 0
            price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
            
            try:
                volume_24h = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
            
            try:
                trades_24h = f"{int(row.get('trades24h', 0)):,}"
            except (ValueError, TypeError):
                trades_24h = "N/A"
            
            status = str(row.get('status', 'TRADING'))[:10]
            status_color = Fore.GREEN if status == 'TRADING' else Fore.YELLOW
            
            print(f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.CYAN}{name:<15} | "
                  f"{Fore.GREEN}{price:>12} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.MAGENTA}{volume_24h:>15} | "
                  f"{Fore.BLUE}{trades_24h:>10} | "
                  f"{status_color}{status:<10}")
                  
        except Exception:
            continue

def display_possible_gems(df):
    """Display possible gem tokens (low price, high volume potential)"""
    if df.empty:
        print(f"{Fore.RED}No potential gems found on Binance.")
        return
        
    # Filter for tokens under the price threshold with good volume
    gems_df = df[
        (df['price'].fillna(float('inf')) <= config.GEMS_MAX_PRICE) & 
        (df['volume24hUSD'].fillna(0) >= config.GEMS_MIN_VOLUME)
    ].copy()
    
    if gems_df.empty:
        print(f"{Fore.RED}No gems found matching criteria (Price < ${config.GEMS_MAX_PRICE:.2f}, Volume > ${config.GEMS_MIN_VOLUME:,})")
        return
        
    print(f"\n{Fore.YELLOW}{'='*150}")
    print(f"{Fore.YELLOW}💎 POSSIBLE BINANCE GEMS (Price < ${config.GEMS_MAX_PRICE:.2f}) 💎")
    print(f"{Fore.YELLOW}{'='*150}")
    
    header = (f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Name':<15} | {'Price':>12} | {'24h Change':>12} | "
             f"{'Volume 24h':>15} | {'Trades':>10} | {'High 24h':>12} | {'Low 24h':>12}")
    separator = f"{Fore.YELLOW}{'-'*5}-+-{'-'*12}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}"
    
    print(header)
    print(separator)
    
    for _, row in gems_df.iterrows():
        try:
            rank = str(row.get('rank', 'N/A'))
            symbol = str(row.get('symbol', 'N/A'))[:12]
            name = str(row.get('name', 'N/A'))[:15]
            
            try:
                price = f"${float(row.get('price', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                price = "N/A"
            
            try:
                price_change_24h = float(row.get('price24hChangePercent', 0))
                price_change_str = f"{price_change_24h:+.2f}%"
            except (ValueError, TypeError):
                price_change_str = "N/A"
                price_change_24h = 0
            price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
            
            try:
                volume_24h = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
            
            try:
                trades_24h = f"{int(row.get('trades24h', 0)):,}"
            except (ValueError, TypeError):
                trades_24h = "N/A"
            
            try:
                high_24h = f"${float(row.get('high24h', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                high_24h = "N/A"
                
            try:
                low_24h = f"${float(row.get('low24h', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                low_24h = "N/A"
            
            print(f"{Fore.WHITE}{rank:<5} | "
                  f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.CYAN}{name:<15} | "
                  f"{Fore.GREEN}{price:>12} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.MAGENTA}{volume_24h:>15} | "
                  f"{Fore.BLUE}{trades_24h:>10} | "
                  f"{Fore.GREEN}{high_24h:>12} | "
                  f"{Fore.RED}{low_24h:>12}")
                  
        except Exception:
            continue

def display_consistent_trending(history_df):
    """Display tokens that consistently appear in trending"""
    if history_df.empty:
        print(f"{Fore.RED}No historical trending data available.")
        return
        
    print(f"\n{Fore.YELLOW}{'='*110}")
    print(f"{Fore.YELLOW}🏆 CONSISTENTLY TRENDING ON BINANCE 🏆")
    print(f"{Fore.YELLOW}{'='*110}")
    
    # Group by symbol and count appearances
    if 'symbol' in history_df.columns:
        symbol_counts = history_df['symbol'].value_counts().head(config.TOP_CONSISTENT_TOKENS)
        
        header = f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Appearances':<12} | {'Avg Price Change':<16}"
        separator = f"{Fore.YELLOW}{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*16}"
        
        print(header)
        print(separator)
        
        for i, (symbol, count) in enumerate(symbol_counts.items(), 1):
            try:
                # Calculate average price change for this symbol
                symbol_data = history_df[history_df['symbol'] == symbol]
                if 'price24hChangePercent' in symbol_data.columns:
                    avg_change = symbol_data['price24hChangePercent'].mean()
                    avg_change_str = f"{avg_change:+.2f}%"
                    avg_change_color = Fore.GREEN if avg_change >= 0 else Fore.RED
                else:
                    avg_change_str = "N/A"
                    avg_change_color = Fore.WHITE
                
                print(f"{Fore.WHITE}{i:<5} | "
                      f"{Fore.YELLOW}{symbol:<12} | "
                      f"{Fore.CYAN}{count:<12} | "
                      f"{avg_change_color}{avg_change_str:<16}")
                      
            except Exception:
                continue
    
    print(f"{Fore.YELLOW}{'='*110}")

def display_volume_leaders(df):
    """Display top volume trading pairs"""
    if df.empty:
        print(f"{Fore.RED}No volume data available.")
        return
    
    print(f"\n{Fore.YELLOW}{'='*120}")
    print(f"{Fore.YELLOW}📊 BINANCE VOLUME LEADERS 📊")
    print(f"{Fore.YELLOW}{'='*120}")
    
    # Sort by volume
    volume_sorted = df.sort_values('volume24hUSD', ascending=False)
    
    header = (f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Volume 24h':>15} | "
             f"{'Price':>12} | {'24h Change':>12} | {'Trades':>10}")
    separator = f"{Fore.YELLOW}{'-'*5}-+-{'-'*12}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}"
    
    print(header)
    print(separator)
    
    for i, (_, row) in enumerate(volume_sorted.head(20).iterrows(), 1):
        try:
            symbol = str(row.get('symbol', 'N/A'))[:12]
            
            try:
                volume_24h = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
            
            try:
                price = f"${float(row.get('price', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                price = "N/A"
            
            try:
                price_change_24h = float(row.get('price24hChangePercent', 0))
                price_change_str = f"{price_change_24h:+.2f}%"
            except (ValueError, TypeError):
                price_change_str = "N/A"
                price_change_24h = 0
            price_change_color = Fore.GREEN if price_change_24h >= 0 else Fore.RED
            
            try:
                trades_24h = f"{int(row.get('trades24h', 0)):,}"
            except (ValueError, TypeError):
                trades_24h = "N/A"
            
            print(f"{Fore.WHITE}{i:<5} | "
                  f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.MAGENTA}{volume_24h:>15} | "
                  f"{Fore.GREEN}{price:>12} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.BLUE}{trades_24h:>10}")
                  
        except Exception:
            continue
