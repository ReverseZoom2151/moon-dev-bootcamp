"""
Conversational Market Context Provider
======================================
Day 30: Provides real-time market data context for conversations.

Fetches and formats market data to inject into conversation prompts,
enabling Gordon to provide context-aware responses.
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketContextProvider:
    """
    Provides real-time market data context for conversations.
    
    Fetches market data from exchanges and formats it for inclusion
    in conversation prompts.
    """

    def __init__(self, exchange_adapter=None, default_symbols: Optional[List[str]] = None):
        """
        Initialize market context provider.
        
        Args:
            exchange_adapter: Exchange adapter instance (optional)
            default_symbols: Default symbols to fetch
        """
        self.exchange_adapter = exchange_adapter
        self.default_symbols = default_symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
    async def get_market_summary(self) -> str:
        """
        Get summary of key market data.
        
        Returns:
            Formatted market summary string
        """
        if not self.exchange_adapter:
            return "\n⚠️ Market data unavailable (no exchange connection)\n"
        
        try:
            summary_lines = [
                f"\n🟠 MARKET SNAPSHOT - {datetime.now().strftime('%H:%M:%S')} UTC 🟠"
            ]
            
            for symbol in self.default_symbols:
                try:
                    ticker = await self._get_ticker(symbol)
                    if ticker:
                        price = ticker.get('last', ticker.get('lastPrice', 0))
                        change_pct = ticker.get('percentage', ticker.get('priceChangePercent', 0))
                        volume = ticker.get('quoteVolume', ticker.get('volume', 0))
                        
                        change_indicator = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                        
                        summary_lines.append(
                            f"{change_indicator} {symbol}: ${price:,.8g} ({change_pct:+.2f}%) | "
                            f"Vol: {volume:,.0f}"
                        )
                except Exception as e:
                    logger.debug(f"Error fetching ticker for {symbol}: {e}")
                    continue
            
            summary_lines.append("="*70)
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return f"\n⚠️ Error fetching market summary: {e}\n"
    
    async def get_symbol_analysis(self, symbol: str) -> str:
        """
        Get detailed analysis for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Formatted symbol analysis string
        """
        if not self.exchange_adapter:
            return f"❌ Could not fetch data for {symbol} (no exchange connection)"
        
        try:
            ticker = await self._get_ticker(symbol)
            if not ticker:
                return f"❌ Could not fetch data for {symbol}"
            
            price = ticker.get('last', ticker.get('lastPrice', 0))
            change_24h = ticker.get('change', ticker.get('priceChange', 0))
            change_pct = ticker.get('percentage', ticker.get('priceChangePercent', 0))
            high_24h = ticker.get('high', ticker.get('highPrice', 0))
            low_24h = ticker.get('low', ticker.get('lowPrice', 0))
            volume = ticker.get('quoteVolume', ticker.get('volume', 0))
            
            analysis = [
                f"\n📊 {symbol.upper()} DETAILED ANALYSIS",
                f"{'='*40}",
                f"💰 Current Price: ${price:,.8g}",
                f"📈 24h Change: ${change_24h:+,.8g} ({change_pct:+.2f}%)",
                f"🔺 24h High: ${high_24h:,.8g}",
                f"🔻 24h Low: ${low_24h:,.8g}",
                f"📊 24h Volume: {volume:,.0f}",
            ]
            
            return "\n".join(analysis) + "\n"
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return f"❌ Error analyzing {symbol}: {e}"
    
    async def _get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker data from exchange."""
        try:
            if hasattr(self.exchange_adapter, 'fetch_ticker'):
                return await self.exchange_adapter.fetch_ticker(symbol)
            elif hasattr(self.exchange_adapter, 'get_ticker'):
                return await self.exchange_adapter.get_ticker(symbol)
            else:
                # Fallback: try calling directly
                ticker = self.exchange_adapter.fetch_ticker(symbol)
                if hasattr(ticker, '__await__'):
                    return await ticker
                return ticker
        except Exception as e:
            logger.debug(f"Error fetching ticker for {symbol}: {e}")
            return None


class QuickCommandHandler:
    """
    Handles quick command shortcuts for conversational interface.
    
    Commands:
    - price SYMBOL: Get current price
    - analyze SYMBOL: Get detailed analysis
    - market: Get market summary
    """
    
    def __init__(self, market_context: MarketContextProvider):
        """
        Initialize quick command handler.
        
        Args:
            market_context: Market context provider
        """
        self.market_context = market_context
    
    async def handle_command(self, prompt: str) -> Optional[str]:
        """
        Handle quick command shortcuts.
        
        Args:
            prompt: User prompt
            
        Returns:
            Modified prompt or None if not a command
        """
        if not prompt:
            return prompt
        
        lower_prompt = prompt.lower().strip()
        
        # Price command
        if lower_prompt.startswith('price '):
            symbol = lower_prompt.split('price ', 1)[1].strip().upper()
            ticker = await self.market_context._get_ticker(symbol)
            if ticker:
                price = ticker.get('last', ticker.get('lastPrice', 0))
                change_pct = ticker.get('percentage', ticker.get('priceChangePercent', 0))
                return f"Current {symbol} price is ${price:,.8g} ({change_pct:+.2f}% 24h). What would you like to know about this price action?"
            else:
                return f"Could not fetch price for {symbol}. Can you help me analyze this symbol anyway?"
        
        # Analyze command
        elif lower_prompt.startswith('analyze '):
            symbol = lower_prompt.split('analyze ', 1)[1].strip().upper()
            analysis = await self.market_context.get_symbol_analysis(symbol)
            return f"{analysis}\n\nBased on this data, what's your analysis or what should I focus on?"
        
        # Market command
        elif lower_prompt == 'market':
            summary = await self.market_context.get_market_summary()
            return f"{summary}\n\nWhat do you think about these market conditions? Any trading opportunities you see?"
        
        return prompt

