'''Display functions for the Bitfinex Token Tracker'''

import bitfinex_config as config
from colorama import Fore, init

# Initialize colorama for terminal colors
init(autoreset=True)

# Bitfinex ASCII Art Banner
BITFINEX_BANNER = f"""{Fore.GREEN}
  ____  _ _    __ _                
 |  _ \(_) |  / _(_)               
 | |_) |_| |_| |_ _ _ __   _____  __
 |  _ <| | __| _| | '_ \ / _ \ \/ /
 | |_) | | |_| | | | | |  __/>  < 
 |____/|_|\__|_| |_|_| |_|\___/_/\_\
                                   
{Fore.MAGENTA}🚀 Bitfinex Professional Tracker 🌙{Fore.RESET}
"""

# Professional Bitfinex trading quotes
BITFINEX_QUOTES = [
    "Professional trading on Bitfinex! 📈",
    "Margin opportunities incoming... 🚀",
    "Bitfinex API - institutional grade data! 🎯",
    "Professional trading to the moon! 🌕",
    "Margin trading never sleeps! 😴",
    "Funding rates drive the markets! 📊",
    "Bitfinex - where professionals trade! 👔",
    "Leverage your positions wisely! ⚖️",
    "Watch the funding, follow the smart money! 💰",
    "Professional-grade alpha from Bitfinex! ✨"
]

def display_trending_tokens(df):
    """Display trending tokens in a beautiful format"""
    if df.empty:
        print(f"{Fore.RED}No trending tokens data available from Bitfinex.")
        return
        
    print(f"\n{Fore.GREEN}{'='*160}")
    print(f"{Fore.GREEN}🚀 BITFINEX TRENDING TOKENS ({len(df)} pairs) 🚀")
    print(f"{Fore.GREEN}{'='*160}")
    
    header = (f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Name':<15} | {'Price':>15} | {'24h Change':>12} | "
             f"{'Volume 24h':>15} | {'Volume':>12} | {'High 24h':>15} | {'Low 24h':>15} | {'Spread':>8}")
    separator = f"{Fore.GREEN}{'-'*5}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*12}-+-{'-'*15}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*8}"
    
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
                volume_24h_usd = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h_usd = "N/A"
            
            try:
                volume_24h = f"{float(row.get('volume24h', 0)):,.2f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
            
            try:
                high_24h = f"${float(row.get('high24h', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                high_24h = "N/A"
                
            try:
                low_24h = f"${float(row.get('low24h', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                low_24h = "N/A"
            
            # Calculate spread
            try:
                bid = float(row.get('bid', 0))
                ask = float(row.get('ask', 0))
                if bid > 0 and ask > 0:
                    spread = ((ask - bid) / bid) * 100
                    spread_str = f"{spread:.2f}%"
                else:
                    spread_str = "N/A"
            except (ValueError, TypeError):
                spread_str = "N/A"
            
            print(f"{Fore.WHITE}{rank:<5} | "
                  f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.CYAN}{name:<15} | "
                  f"{Fore.GREEN}{price:>15} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.MAGENTA}{volume_24h_usd:>15} | "
                  f"{Fore.BLUE}{volume_24h:>12} | "
                  f"{Fore.GREEN}{high_24h:>15} | "
                  f"{Fore.RED}{low_24h:>15} | "
                  f"{Fore.YELLOW}{spread_str:>8}")
                  
        except Exception:
            continue

def display_new_listings(df):
    """Display new token listings in a beautiful format"""
    if df.empty:
        print(f"{Fore.RED}No new listings data available from Bitfinex.")
        return
        
    print(f"\n{Fore.GREEN}{'='*140}")
    print(f"{Fore.GREEN}🌟 BITFINEX NEW/EMERGING TOKENS ({len(df)} pairs) 🌟")
    print(f"{Fore.GREEN}{'='*140}")
    
    header = (f"{Fore.CYAN}{'Symbol':<12} | {'Name':<15} | {'Price':>15} | {'24h Change':>12} | "
             f"{'Volume 24h':>15} | {'Volume':>12} | {'Status':<10}")
    separator = f"{Fore.GREEN}{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*12}-+-{'-'*15}-+-{'-'*12}-+-{'-'*10}"
    
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
                volume_24h_usd = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h_usd = "N/A"
            
            try:
                volume_24h = f"{float(row.get('volume24h', 0)):,.2f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
            
            status = str(row.get('status', 'TRADING'))[:10]
            status_color = Fore.GREEN if status == 'TRADING' else Fore.YELLOW
            
            print(f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.CYAN}{name:<15} | "
                  f"{Fore.GREEN}{price:>15} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.MAGENTA}{volume_24h_usd:>15} | "
                  f"{Fore.BLUE}{volume_24h:>12} | "
                  f"{status_color}{status:<10}")
                  
        except Exception:
            continue

def display_possible_gems(df):
    """Display possible gem tokens (low price, good volume potential)"""
    if df.empty:
        print(f"{Fore.RED}No potential gems found on Bitfinex.")
        return
        
    # Filter for tokens under the price threshold with good volume
    gems_df = df[
        (df['price'].fillna(float('inf')) <= config.GEMS_MAX_PRICE) & 
        (df['volume24hUSD'].fillna(0) >= config.GEMS_MIN_VOLUME)
    ].copy()
    
    if gems_df.empty:
        print(f"{Fore.RED}No gems found matching criteria (Price < ${config.GEMS_MAX_PRICE:.2f}, Volume > ${config.GEMS_MIN_VOLUME:,})")
        return
        
    print(f"\n{Fore.GREEN}{'='*160}")
    print(f"{Fore.GREEN}💎 POTENTIAL BITFINEX GEMS (Price < ${config.GEMS_MAX_PRICE:.2f}) 💎")
    print(f"{Fore.GREEN}{'='*160}")
    
    header = (f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Name':<15} | {'Price':>15} | {'24h Change':>12} | "
             f"{'Volume 24h':>15} | {'Volume':>12} | {'High 24h':>15} | {'Low 24h':>15} | {'Spread':>8}")
    separator = f"{Fore.GREEN}{'-'*5}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*12}-+-{'-'*15}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*8}"
    
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
                volume_24h_usd = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h_usd = "N/A"
            
            try:
                volume_24h = f"{float(row.get('volume24h', 0)):,.2f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
            
            try:
                high_24h = f"${float(row.get('high24h', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                high_24h = "N/A"
                
            try:
                low_24h = f"${float(row.get('low24h', 0)):,.{config.PRICE_PRECISION}f}"
            except (ValueError, TypeError):
                low_24h = "N/A"
            
            # Calculate spread
            try:
                bid = float(row.get('bid', 0))
                ask = float(row.get('ask', 0))
                if bid > 0 and ask > 0:
                    spread = ((ask - bid) / bid) * 100
                    spread_str = f"{spread:.2f}%"
                else:
                    spread_str = "N/A"
            except (ValueError, TypeError):
                spread_str = "N/A"
            
            print(f"{Fore.WHITE}{rank:<5} | "
                  f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.CYAN}{name:<15} | "
                  f"{Fore.GREEN}{price:>15} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.MAGENTA}{volume_24h_usd:>15} | "
                  f"{Fore.BLUE}{volume_24h:>12} | "
                  f"{Fore.GREEN}{high_24h:>15} | "
                  f"{Fore.RED}{low_24h:>15} | "
                  f"{Fore.YELLOW}{spread_str:>8}")
                  
        except Exception:
            continue

def display_consistent_trending(history_df):
    """Display tokens that consistently appear in trending"""
    if history_df.empty:
        print(f"{Fore.RED}No historical trending data available.")
        return
        
    print(f"\n{Fore.GREEN}{'='*120}")
    print(f"{Fore.GREEN}🏆 CONSISTENTLY TRENDING ON BITFINEX 🏆")
    print(f"{Fore.GREEN}{'='*120}")
    
    # Group by symbol and count appearances
    if 'symbol' in history_df.columns:
        symbol_counts = history_df['symbol'].value_counts().head(config.TOP_CONSISTENT_TOKENS)
        
        header = f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Appearances':<12} | {'Avg Price Change':<16} | {'Avg Volume':<15}"
        separator = f"{Fore.GREEN}{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*16}-+-{'-'*15}"
        
        print(header)
        print(separator)
        
        for i, (symbol, count) in enumerate(symbol_counts.items(), 1):
            try:
                # Calculate average price change and volume for this symbol
                symbol_data = history_df[history_df['symbol'] == symbol]
                
                if 'price24hChangePercent' in symbol_data.columns:
                    avg_change = symbol_data['price24hChangePercent'].mean()
                    avg_change_str = f"{avg_change:+.2f}%"
                    avg_change_color = Fore.GREEN if avg_change >= 0 else Fore.RED
                else:
                    avg_change_str = "N/A"
                    avg_change_color = Fore.WHITE
                
                if 'volume24hUSD' in symbol_data.columns:
                    avg_volume = symbol_data['volume24hUSD'].mean()
                    avg_volume_str = f"${avg_volume:,.0f}"
                else:
                    avg_volume_str = "N/A"
                
                print(f"{Fore.WHITE}{i:<5} | "
                      f"{Fore.YELLOW}{symbol:<12} | "
                      f"{Fore.CYAN}{count:<12} | "
                      f"{avg_change_color}{avg_change_str:<16} | "
                      f"{Fore.MAGENTA}{avg_volume_str:<15}")
                      
            except Exception:
                continue
    
    print(f"{Fore.GREEN}{'='*120}")

def display_volume_leaders(df):
    """Display top volume trading pairs"""
    if df.empty:
        print(f"{Fore.RED}No volume data available.")
        return
    
    print(f"\n{Fore.GREEN}{'='*140}")
    print(f"{Fore.GREEN}📊 BITFINEX VOLUME LEADERS 📊")
    print(f"{Fore.GREEN}{'='*140}")
    
    # Sort by volume
    volume_sorted = df.sort_values('volume24hUSD', ascending=False)
    
    header = (f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Volume 24h':>15} | "
             f"{'Price':>15} | {'24h Change':>12} | {'Volume':>12}")
    separator = f"{Fore.GREEN}{'-'*5}-+-{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}"
    
    print(header)
    print(separator)
    
    for i, (_, row) in enumerate(volume_sorted.head(20).iterrows(), 1):
        try:
            symbol = str(row.get('symbol', 'N/A'))[:12]
            
            try:
                volume_24h_usd = f"${float(row.get('volume24hUSD', 0)):,.0f}"
            except (ValueError, TypeError):
                volume_24h_usd = "N/A"
            
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
                volume_24h = f"{float(row.get('volume24h', 0)):,.2f}"
            except (ValueError, TypeError):
                volume_24h = "N/A"
            
            print(f"{Fore.WHITE}{i:<5} | "
                  f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.MAGENTA}{volume_24h_usd:>15} | "
                  f"{Fore.GREEN}{price:>15} | "
                  f"{price_change_color}{price_change_str:>12} | "
                  f"{Fore.BLUE}{volume_24h:>12}")
                  
        except Exception:
            continue

def display_funding_rates(funding_df):
    """Display funding rates for margin tokens"""
    if funding_df.empty:
        print(f"{Fore.RED}No funding rate data available.")
        return
    
    print(f"\n{Fore.GREEN}{'='*120}")
    print(f"{Fore.GREEN}💰 BITFINEX FUNDING RATES 💰")
    print(f"{Fore.GREEN}{'='*120}")
    
    # Sort by absolute funding rate
    funding_df['abs_rate'] = funding_df['funding_rate'].abs()
    funding_sorted = funding_df.sort_values('abs_rate', ascending=False)
    
    header = (f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Base Token':<12} | {'Funding Rate':>12} | "
             f"{'Rate Change 24h':>15} | {'Volume':>12}")
    separator = f"{Fore.GREEN}{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*12}"
    
    print(header)
    print(separator)
    
    for i, (_, row) in enumerate(funding_sorted.head(15).iterrows(), 1):
        try:
            symbol = str(row.get('symbol', 'N/A'))[:12]
            base_symbol = str(row.get('base_symbol', 'N/A'))[:12]
            
            try:
                funding_rate = float(row.get('funding_rate', 0))
                funding_rate_str = f"{funding_rate:+.4f}%"
            except (ValueError, TypeError):
                funding_rate_str = "N/A"
                funding_rate = 0
            funding_rate_color = Fore.GREEN if funding_rate >= 0 else Fore.RED
            
            try:
                rate_change = float(row.get('rate_change_24h', 0))
                rate_change_str = f"{rate_change:+.2f}%"
            except (ValueError, TypeError):
                rate_change_str = "N/A"
                rate_change = 0
            rate_change_color = Fore.GREEN if rate_change >= 0 else Fore.RED
            
            try:
                volume = f"{float(row.get('volume', 0)):,.2f}"
            except (ValueError, TypeError):
                volume = "N/A"
            
            print(f"{Fore.WHITE}{i:<5} | "
                  f"{Fore.YELLOW}{symbol:<12} | "
                  f"{Fore.CYAN}{base_symbol:<12} | "
                  f"{funding_rate_color}{funding_rate_str:>12} | "
                  f"{rate_change_color}{rate_change_str:>15} | "
                  f"{Fore.BLUE}{volume:>12}")
                  
        except Exception:
            continue
    
    print(f"{Fore.GREEN}{'='*120}")
    print(f"{Fore.YELLOW}💡 Professional Tip: High funding rates may indicate profitable arbitrage opportunities!")
    print(f"{Fore.YELLOW}📊 Monitor funding rate trends for institutional trading insights!")
