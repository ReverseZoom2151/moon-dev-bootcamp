"""
Market Display Utilities
========================
Day 47: Beautiful terminal display functions for market data.
"""

import pandas as pd
from colorama import Fore, Style, init
from typing import Optional, Dict

# Initialize colorama
init(autoreset=True)


class MarketDisplay:
    """Display utilities for market data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize market display.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.price_precision = self.config.get('price_precision', 8)
        self.volume_precision = self.config.get('volume_precision', 0)
    
    def display_trending_tokens(self, df: pd.DataFrame, exchange_name: str = "Exchange"):
        """Display trending tokens in a beautiful format."""
        if df.empty:
            print(f"{Fore.RED}No trending tokens data available from {exchange_name}.")
            return
        
        print(f"\n{Fore.YELLOW}{'='*150}")
        print(f"{Fore.YELLOW}🚀 {exchange_name.upper()} TRENDING TOKENS ({len(df)} tokens) 🚀")
        print(f"{Fore.YELLOW}{'='*150}")
        
        header = (f"{Fore.CYAN}{'Rank':<5} | {'Symbol':<12} | {'Name':<15} | {'Price':>12} | {'24h Change':>12} | "
                 f"{'Volume 24h':>15} | {'Trades':>10} | {'High 24h':>12} | {'Low 24h':>12}")
        separator = f"{Fore.YELLOW}{'-'*5}-+-{'-'*12}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}"
        
        print(header)
        print(separator)
        
        for _, row in df.head(self.config.get('display_trending_tokens', 40)).iterrows():
            try:
                rank = str(row.get('rank', 'N/A'))
                symbol = str(row.get('symbol', 'N/A'))[:12]
                name = str(row.get('name', 'N/A'))[:15]
                
                try:
                    price = f"${float(row.get('price', 0)):,.{self.price_precision}f}"
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
                    high_24h = f"${float(row.get('high24h', 0)):,.{self.price_precision}f}"
                except (ValueError, TypeError):
                    high_24h = "N/A"
                
                try:
                    low_24h = f"${float(row.get('low24h', 0)):,.{self.price_precision}f}"
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
    
    def display_new_listings(self, df: pd.DataFrame, exchange_name: str = "Exchange"):
        """Display new listings in a beautiful format."""
        if df.empty:
            print(f"{Fore.RED}No new listings data available from {exchange_name}.")
            return
        
        print(f"\n{Fore.YELLOW}{'='*140}")
        print(f"{Fore.YELLOW}🌟 NEW {exchange_name.upper()} LISTINGS ({len(df)} tokens) 🌟")
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
                    price = f"${float(row.get('price', 0)):,.{self.price_precision}f}"
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
    
    def display_volume_leaders(self, df: pd.DataFrame, exchange_name: str = "Exchange"):
        """Display volume leaders."""
        if df.empty:
            print(f"{Fore.RED}No volume data available.")
            return
        
        print(f"\n{Fore.YELLOW}{'='*120}")
        print(f"{Fore.YELLOW}📊 {exchange_name.upper()} VOLUME LEADERS 📊")
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
                    price = f"${float(row.get('price', 0)):,.{self.price_precision}f}"
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
    
    def display_possible_gems(self, df: pd.DataFrame, exchange_name: str = "Exchange"):
        """Display possible gem tokens."""
        if df.empty:
            print(f"{Fore.RED}No potential gems found on {exchange_name}.")
            return
        
        gems_max_price = self.config.get('gems_max_price', 1.0)
        gems_min_volume = self.config.get('gems_min_volume', 100000)
        
        # Filter for gems
        gems_df = df[
            (df['price'].fillna(float('inf')) <= gems_max_price) & 
            (df['volume24hUSD'].fillna(0) >= gems_min_volume)
        ].copy()
        
        if gems_df.empty:
            print(f"{Fore.RED}No gems found matching criteria (Price < ${gems_max_price:.2f}, Volume > ${gems_min_volume:,})")
            return
        
        print(f"\n{Fore.YELLOW}{'='*150}")
        print(f"{Fore.YELLOW}💎 POSSIBLE {exchange_name.upper()} GEMS (Price < ${gems_max_price:.2f}) 💎")
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
                    price = f"${float(row.get('price', 0)):,.{self.price_precision}f}"
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
                    high_24h = f"${float(row.get('high24h', 0)):,.{self.price_precision}f}"
                except (ValueError, TypeError):
                    high_24h = "N/A"
                
                try:
                    low_24h = f"${float(row.get('low24h', 0)):,.{self.price_precision}f}"
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
    
    def display_consistent_trending(self, history_df: pd.DataFrame, exchange_name: str = "Exchange"):
        """Display consistently trending tokens."""
        if history_df.empty:
            print(f"{Fore.RED}No historical trending data available.")
            return
        
        print(f"\n{Fore.YELLOW}{'='*110}")
        print(f"{Fore.YELLOW}🏆 CONSISTENTLY TRENDING ON {exchange_name.upper()} 🏆")
        print(f"{Fore.YELLOW}{'='*110}")
        
        # Group by symbol and count appearances
        if 'symbol' in history_df.columns:
            symbol_counts = history_df['symbol'].value_counts().head(self.config.get('top_consistent_tokens', 10))
            
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

