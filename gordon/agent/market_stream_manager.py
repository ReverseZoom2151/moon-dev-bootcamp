"""
WebSocket Market Stream Manager
================================
Day 30 Enhancement: Real-time market data streaming for conversational interface.

Provides WebSocket connections for live market updates that can be
injected into conversations for real-time context.
"""

import asyncio
import json
import logging
from typing import Dict, Optional, Callable, List
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class MarketStreamManager:
    """
    Manages WebSocket connections for real-time market data streaming.
    
    Provides live market updates that can be used in conversational context.
    """

    def __init__(
        self,
        exchange_adapter=None,
        symbols: Optional[List[str]] = None,
        max_cache_size: int = 100
    ):
        """
        Initialize market stream manager.
        
        Args:
            exchange_adapter: Exchange adapter with WebSocket support
            symbols: List of symbols to stream
            max_cache_size: Maximum cached updates per symbol
        """
        self.exchange_adapter = exchange_adapter
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.max_cache_size = max_cache_size
        
        # Cache for recent updates
        self.update_cache: Dict[str, deque] = {
            symbol: deque(maxlen=max_cache_size)
            for symbol in self.symbols
        }
        
        # WebSocket connections
        self.ws_connections: Dict[str, any] = {}
        self.is_streaming = False
        
        # Callbacks for updates
        self.update_callbacks: List[Callable] = []
        
    async def start_streaming(self):
        """Start WebSocket streaming for all symbols."""
        if not self.exchange_adapter:
            logger.warning("No exchange adapter available for streaming")
            return
        
        if self.is_streaming:
            logger.info("Already streaming")
            return
        
        try:
            self.is_streaming = True
            
            # Try to use exchange adapter's WebSocket if available
            if hasattr(self.exchange_adapter, 'watch_ticker'):
                for symbol in self.symbols:
                    await self._stream_symbol(symbol)
            elif hasattr(self.exchange_adapter, 'watch_tickers'):
                # Stream all symbols at once if supported
                await self._stream_all_symbols()
            else:
                logger.warning("Exchange adapter does not support WebSocket streaming")
                self.is_streaming = False
                
        except Exception as e:
            logger.error(f"Error starting WebSocket stream: {e}")
            self.is_streaming = False
    
    async def stop_streaming(self):
        """Stop all WebSocket streams."""
        try:
            for ws in self.ws_connections.values():
                if hasattr(ws, 'close'):
                    await ws.close()
            
            self.ws_connections.clear()
            self.is_streaming = False
            logger.info("WebSocket streaming stopped")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket stream: {e}")
    
    async def _stream_symbol(self, symbol: str):
        """Stream updates for a single symbol."""
        try:
            if hasattr(self.exchange_adapter, 'watch_ticker'):
                async def handle_update(update):
                    await self._handle_market_update(symbol, update)
                
                # Start watching ticker
                ws = await self.exchange_adapter.watch_ticker(symbol, handle_update)
                self.ws_connections[symbol] = ws
                logger.info(f"Started streaming for {symbol}")
                
        except Exception as e:
            logger.error(f"Error streaming {symbol}: {e}")
    
    async def _stream_all_symbols(self):
        """Stream updates for all symbols."""
        try:
            if hasattr(self.exchange_adapter, 'watch_tickers'):
                async def handle_update(update):
                    symbol = update.get('symbol', '')
                    await self._handle_market_update(symbol, update)
                
                ws = await self.exchange_adapter.watch_tickers(self.symbols, handle_update)
                self.ws_connections['all'] = ws
                logger.info(f"Started streaming for all symbols")
                
        except Exception as e:
            logger.error(f"Error streaming all symbols: {e}")
    
    async def _handle_market_update(self, symbol: str, update: Dict):
        """Handle incoming market update."""
        try:
            timestamp = datetime.now().isoformat()
            update['timestamp'] = timestamp
            
            # Cache the update
            if symbol in self.update_cache:
                self.update_cache[symbol].append(update)
            
            # Notify callbacks
            for callback in self.update_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol, update)
                    else:
                        callback(symbol, update)
                except Exception as e:
                    logger.error(f"Error in update callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling market update: {e}")
    
    def get_latest_updates(self, symbol: str, count: int = 10) -> List[Dict]:
        """
        Get latest market updates for a symbol.
        
        Args:
            symbol: Trading symbol
            count: Number of updates to retrieve
            
        Returns:
            List of recent updates
        """
        if symbol not in self.update_cache:
            return []
        
        updates = list(self.update_cache[symbol])
        return updates[-count:] if len(updates) > count else updates
    
    def get_market_summary_from_stream(self) -> str:
        """
        Generate market summary from cached stream data.
        
        Returns:
            Formatted market summary
        """
        if not self.is_streaming:
            return "⚠️ Market streaming not active"
        
        summary_lines = [
            f"\n🟢 LIVE MARKET STREAM - {datetime.now().strftime('%H:%M:%S')} UTC 🟢"
        ]
        
        for symbol in self.symbols:
            updates = self.get_latest_updates(symbol, count=1)
            if updates:
                latest = updates[-1]
                price = latest.get('last', latest.get('price', 0))
                change_pct = latest.get('percentage', latest.get('change', 0))
                
                summary_lines.append(
                    f"📊 {symbol}: ${price:,.8g} ({change_pct:+.2f}%)"
                )
        
        summary_lines.append("="*70)
        return "\n".join(summary_lines)
    
    def register_update_callback(self, callback: Callable):
        """
        Register callback for market updates.
        
        Args:
            callback: Callback function(symbol, update)
        """
        self.update_callbacks.append(callback)
    
    def unregister_update_callback(self, callback: Callable):
        """Unregister update callback."""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)

