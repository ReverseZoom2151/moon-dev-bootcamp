"""
Market Data Manager - Unified interface for market data from various sources
Provides OHLCV data, real-time prices, and market information
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from core.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Single market data point"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataManager:
    """
    Unified market data manager that aggregates data from multiple sources
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)  # 5-minute cache
        self.last_update = {}
        
        # Data sources
        self.data_sources = {}
        self.primary_source = "hyperliquid"  # Default primary source
        
        logger.info("🔧 Market Data Manager initialized")
    
    async def start(self):
        """Start the market data manager"""
        try:
            # Initialize data sources
            await self._initialize_data_sources()
            logger.info("✅ Market Data Manager started")
        except Exception as e:
            logger.error(f"❌ Failed to start Market Data Manager: {e}")
            raise
    
    async def stop(self):
        """Stop the market data manager"""
        try:
            # Cleanup data sources
            for source in self.data_sources.values():
                if hasattr(source, 'close'):
                    await source.close()
            logger.info("✅ Market Data Manager stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping Market Data Manager: {e}")
    
    async def _initialize_data_sources(self):
        """Initialize various data sources"""
        # For now, we'll use simulated data
        # In production, this would initialize real data sources
        logger.info("📊 Initializing data sources (simulation mode)")
    
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            cache_key = f"{symbol}_{timeframe}_{limit}"
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                logger.debug(f"📊 Using cached data for {symbol} {timeframe}")
                return self.cache[cache_key]
            
            # Generate simulated data for now
            data = await self._generate_simulated_data(symbol, timeframe, limit)
            
            if data is not None:
                # Cache the data
                self.cache[cache_key] = data
                self.last_update[cache_key] = datetime.utcnow()
                
                logger.debug(f"📊 Retrieved {len(data)} candles for {symbol} {timeframe}")
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting OHLCV data for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            # Get latest candle
            data = await self.get_ohlcv(symbol, "1m", 1)
            if data is not None and not data.empty:
                return float(data['close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"❌ Error getting current price for {symbol}: {e}")
            return None
    
    async def get_volume_data(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[pd.Series]:
        """Get volume data for a symbol"""
        try:
            data = await self.get_ohlcv(symbol, timeframe, limit)
            if data is not None:
                return data['volume']
            return None
        except Exception as e:
            logger.error(f"❌ Error getting volume data for {symbol}: {e}")
            return None
    
    async def _generate_simulated_data(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Generate simulated OHLCV data for testing"""
        try:
            # Base prices for different symbols
            base_prices = {
                "BTC": 45000,
                "ETH": 3000,
                "SOL": 100,
                "WIF": 2.5,
                "POPCAT": 1.2,
                "LINK": 15,
                "ADA": 0.5,
                "DOT": 8,
                "XRP": 0.6,
                "LTC": 80
            }
            
            base_price = base_prices.get(symbol, 100)
            
            # Generate timestamps
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=timeframe_minutes * limit)
            
            timestamps = pd.date_range(start=start_time, end=end_time, periods=limit)
            
            # Generate price data with some randomness
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            # Generate returns
            returns = np.random.normal(0, 0.02, limit)  # 2% volatility
            
            # Add some trend
            trend = np.linspace(-0.001, 0.001, limit)
            returns += trend
            
            # Calculate prices
            prices = [base_price]
            for i in range(1, limit):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(max(new_price, 0.01))  # Prevent negative prices
            
            # Generate OHLC from close prices
            data = []
            for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
                # Generate realistic OHLC
                volatility = abs(returns[i]) * close
                
                high = close + np.random.uniform(0, volatility)
                low = close - np.random.uniform(0, volatility)
                
                if i == 0:
                    open_price = close
                else:
                    open_price = prices[i-1]
                
                # Ensure OHLC relationships are valid
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                # Generate volume
                base_volume = 1000000 if symbol in ["BTC", "ETH"] else 100000
                volume = base_volume * (1 + np.random.uniform(-0.5, 2))  # Variable volume
                
                data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error generating simulated data: {e}")
            return None
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        return timeframe_map.get(timeframe, 60)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache or cache_key not in self.last_update:
            return False
        
        time_since_update = datetime.utcnow() - self.last_update[cache_key]
        return time_since_update < self.cache_duration
    
    async def get_market_summary(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market summary for multiple symbols"""
        summary = {}
        
        for symbol in symbols:
            try:
                current_price = await self.get_current_price(symbol)
                data = await self.get_ohlcv(symbol, "1d", 2)  # Today and yesterday
                
                if data is not None and len(data) >= 2:
                    price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
                    price_change_pct = (price_change / data['close'].iloc[-2]) * 100
                    
                    summary[symbol] = {
                        'current_price': current_price,
                        'price_change': price_change,
                        'price_change_pct': price_change_pct,
                        'volume_24h': data['volume'].iloc[-1],
                        'high_24h': data['high'].iloc[-1],
                        'low_24h': data['low'].iloc[-1]
                    }
                else:
                    summary[symbol] = {
                        'current_price': current_price,
                        'price_change': 0,
                        'price_change_pct': 0,
                        'volume_24h': 0,
                        'high_24h': current_price,
                        'low_24h': current_price
                    }
                    
            except Exception as e:
                logger.error(f"❌ Error getting market summary for {symbol}: {e}")
                summary[symbol] = {'error': str(e)}
        
        return summary
    
    async def get_orderbook(self, symbol: str, depth: int = 10) -> Optional[Dict[str, Any]]:
        """Get order book data for a symbol"""
        try:
            # Simulate order book data
            current_price = await self.get_current_price(symbol)
            if current_price is None:
                return None
            
            # Generate realistic bid/ask spread
            spread_pct = 0.001  # 0.1% spread
            spread = current_price * spread_pct
            
            bids = []
            asks = []
            
            # Generate bids (below current price)
            for i in range(depth):
                price = current_price - spread/2 - (i * spread * 0.1)
                size = np.random.uniform(0.1, 10)
                bids.append({'price': price, 'size': size})
            
            # Generate asks (above current price)
            for i in range(depth):
                price = current_price + spread/2 + (i * spread * 0.1)
                size = np.random.uniform(0.1, 10)
                asks.append({'price': price, 'size': size})
            
            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting order book for {symbol}: {e}")
            return None
    
    async def subscribe_to_price_updates(self, symbol: str, callback):
        """Subscribe to real-time price updates"""
        # In a real implementation, this would set up WebSocket connections
        logger.info(f"📊 Subscribed to price updates for {symbol}")
    
    async def unsubscribe_from_price_updates(self, symbol: str):
        """Unsubscribe from real-time price updates"""
        logger.info(f"📊 Unsubscribed from price updates for {symbol}")
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        return [
            "BTC", "ETH", "SOL", "WIF", "POPCAT", "LINK", "ADA", 
            "DOT", "XRP", "LTC", "BCH", "AVAX", "MATIC", "UNI"
        ]
    
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes"""
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of market data sources"""
        return {
            'status': 'healthy',
            'cache_size': len(self.cache),
            'supported_symbols': len(self.get_supported_symbols()),
            'last_update': max(self.last_update.values()) if self.last_update else None
        } 