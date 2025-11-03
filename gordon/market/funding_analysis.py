"""
Funding Rate Analysis
====================
Day 47: Bitfinex funding rates analysis and arbitrage detection.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from colorama import Fore, init

init(autoreset=True)
logger = logging.getLogger(__name__)


class FundingRateAnalyzer:
    """Analyze funding rates for arbitrage opportunities."""
    
    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        """
        Initialize funding rate analyzer.
        
        Args:
            exchange_adapter: Exchange adapter instance (Bitfinex)
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        self.funding_rate_threshold = self.config.get('funding_rate_threshold', 0.01)  # 1%
        self.api_delay = self.config.get('api_delay', 1.0)
    
    async def fetch_funding_rates(self) -> pd.DataFrame:
        """
        Fetch funding rates from Bitfinex.
        
        Returns:
            DataFrame with funding rate data
        """
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return pd.DataFrame()
        
        try:
            import asyncio
            await asyncio.sleep(self.api_delay)
            
            # Get CCXT client from adapter
            ccxt_client = getattr(self.exchange_adapter, 'ccxt_client', None)
            if not ccxt_client:
                logger.warning("CCXT client not available")
                return pd.DataFrame()
            
            # Get tickers (Bitfinex uses 'f' prefix for funding tickers)
            tickers = await ccxt_client.fetch_tickers()
            
            funding_data = []
            for symbol, ticker in tickers.items():
                try:
                    # Bitfinex funding symbols start with 'f' or have ':USD' suffix
                    if not (symbol.startswith('f') or ':USD' in symbol):
                        continue
                    
                    # Skip if not USD funding
                    if not symbol.endswith('USD') and ':USD' not in symbol:
                        continue
                    
                    # Extract funding rate data
                    # Bitfinex funding rate is typically in the ticker
                    last_rate = float(ticker.get('last', 0))
                    daily_change_rel = float(ticker.get('percentage', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    
                    # Get base symbol
                    base_symbol = symbol.replace('f', '').replace(':USD', '').replace('/USD', '').replace('/', '')
                    
                    funding_info = {
                        'symbol': symbol,
                        'base_symbol': base_symbol,
                        'funding_rate': last_rate,
                        'rate_change_24h': daily_change_rel,
                        'volume': volume,
                        'address': symbol
                    }
                    
                    if abs(last_rate) > 0:  # Only include active funding
                        funding_data.append(funding_info)
                        
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Error processing funding ticker {symbol}: {e}")
                    continue
            
            return pd.DataFrame(funding_data) if funding_data else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return pd.DataFrame()
    
    def analyze_arbitrage_opportunities(self, funding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze funding rates for arbitrage opportunities.
        
        Args:
            funding_df: DataFrame with funding rate data
            
        Returns:
            DataFrame with arbitrage opportunities
        """
        if funding_df.empty:
            return pd.DataFrame()
        
        # Filter for high funding rates
        high_funding = funding_df[
            funding_df['funding_rate'].abs() > self.funding_rate_threshold
        ].copy()
        
        if high_funding.empty:
            return pd.DataFrame()
        
        # Sort by absolute funding rate (descending)
        high_funding = high_funding.sort_values(
            'funding_rate',
            key=lambda x: x.abs(),
            ascending=False
        )
        
        # Add opportunity score
        high_funding['opportunity_score'] = (
            high_funding['funding_rate'].abs() * 
            (high_funding['volume'] / high_funding['volume'].max() if high_funding['volume'].max() > 0 else 0)
        )
        
        return high_funding
    
    def display_funding_rates(self, funding_df: pd.DataFrame):
        """Display funding rates in a beautiful format."""
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

