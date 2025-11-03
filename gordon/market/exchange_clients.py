"""
Exchange Market Clients
======================
Day 47: Exchange-specific API clients for market data.
"""

import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class BaseMarketClient:
    """Base class for exchange market clients."""
    
    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        """
        Initialize market client.
        
        Args:
            exchange_adapter: Exchange adapter instance (CCXT-based)
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        # Default configuration
        self.min_volume_24h = self.config.get('min_volume_24h', 1000000)
        self.min_price_change = self.config.get('min_price_change_threshold', 5.0)
        self.display_limit = self.config.get('display_trending_tokens', 40)
        self.api_delay = self.config.get('api_delay', 0.2)
        
        # Ignore list
        self.ignore_list = self.config.get('ignore_list', [])
        
    async def fetch_trending_tokens(self) -> pd.DataFrame:
        """Fetch trending tokens. Must be implemented by subclasses."""
        raise NotImplementedError
    
    async def fetch_new_listings(self) -> pd.DataFrame:
        """Fetch new listings. Must be implemented by subclasses."""
        raise NotImplementedError
    
    async def fetch_high_volume_tokens(self) -> pd.DataFrame:
        """Fetch high volume tokens. Must be implemented by subclasses."""
        raise NotImplementedError


class BinanceMarketClient(BaseMarketClient):
    """Binance market client using CCXT."""
    
    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        super().__init__(exchange_adapter, config)
        
        # Binance-specific config
        self.focus_usdt_pairs = self.config.get('focus_usdt_pairs', True)
        self.gems_max_price = self.config.get('gems_max_price', 1.0)
        self.gems_min_volume = self.config.get('gems_min_volume', 100000)
        self.new_listing_days = self.config.get('new_listing_days', 7)
        self.min_new_listing_volume = self.config.get('min_new_listing_volume', 500000)
    
    async def fetch_trending_tokens(self) -> pd.DataFrame:
        """Fetch trending tokens from Binance."""
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return pd.DataFrame()
        
        try:
            await asyncio.sleep(self.api_delay)
            
            # Get CCXT client from adapter
            ccxt_client = getattr(self.exchange_adapter, 'ccxt_client', None)
            if not ccxt_client:
                logger.warning("CCXT client not available")
                return pd.DataFrame()
            
            # Get 24hr ticker statistics
            tickers = await ccxt_client.fetch_tickers()
            
            tokens = []
            for symbol, ticker in tickers.items():
                try:
                    # Filter by USDT pairs if configured
                    if self.focus_usdt_pairs and not symbol.endswith('/USDT'):
                        continue
                    
                    # Skip ignored symbols
                    if any(ignored in symbol for ignored in self.ignore_list):
                        continue
                    
                    # Extract ticker data
                    price = float(ticker.get('last', 0))
                    price_change_pct = float(ticker.get('percentage', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    high_24h = float(ticker.get('high', 0))
                    low_24h = float(ticker.get('low', 0))
                    count = int(ticker.get('count', 0))
                    
                    # Filter by minimum volume and price change
                    if volume < self.min_volume_24h or abs(price_change_pct) < self.min_price_change:
                        continue
                    
                    token = {
                        'symbol': symbol,
                        'name': symbol.replace('/USDT', '').replace('/', ''),
                        'price': price,
                        'price24hChangePercent': price_change_pct,
                        'volume24hUSD': volume,
                        'priceChange24h': (price - ticker.get('open', price)) if ticker.get('open') else 0,
                        'high24h': high_24h,
                        'low24h': low_24h,
                        'trades24h': count,
                        'address': symbol,
                        'rank': 0
                    }
                    
                    tokens.append(token)
                    
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Error processing ticker {symbol}: {e}")
                    continue
            
            # Sort by price change percentage (descending)
            tokens.sort(key=lambda x: x['price24hChangePercent'], reverse=True)
            
            # Add rank
            for i, token in enumerate(tokens):
                token['rank'] = i + 1
            
            return pd.DataFrame(tokens[:self.display_limit]) if tokens else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching trending tokens from Binance: {e}")
            return pd.DataFrame()
    
    async def fetch_new_listings(self) -> pd.DataFrame:
        """Fetch new listings from Binance."""
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return pd.DataFrame()
        
        try:
            await asyncio.sleep(self.api_delay)
            
            # Get CCXT client from adapter
            ccxt_client = getattr(self.exchange_adapter, 'ccxt_client', None)
            if not ccxt_client:
                logger.warning("CCXT client not available")
                return pd.DataFrame()
            
            # Get exchange info
            markets = await ccxt_client.load_markets()
            
            # Get tickers for volume data
            tickers = await ccxt_client.fetch_tickers()
            
            new_listings = []
            for symbol, market_info in markets.items():
                try:
                    # Filter by USDT pairs
                    if self.focus_usdt_pairs and not symbol.endswith('/USDT'):
                        continue
                    
                    if market_info.get('active') != True:
                        continue
                    
                    # Skip ignored symbols
                    if any(ignored in symbol for ignored in self.ignore_list):
                        continue
                    
                    # Get ticker data
                    ticker = tickers.get(symbol)
                    if not ticker:
                        continue
                    
                    volume = float(ticker.get('quoteVolume', 0))
                    price = float(ticker.get('last', 0))
                    price_change_pct = float(ticker.get('percentage', 0))
                    count = int(ticker.get('count', 0))
                    
                    # Filter by volume threshold
                    if volume < self.min_new_listing_volume:
                        continue
                    
                    listing = {
                        'symbol': symbol,
                        'name': symbol.replace('/USDT', '').replace('/', ''),
                        'volume24hUSD': volume,
                        'price': price,
                        'price24hChangePercent': price_change_pct,
                        'trades24h': count,
                        'address': symbol,
                        'listed_time': 'Recent',
                        'status': 'TRADING'
                    }
                    
                    new_listings.append(listing)
                    
                except (ValueError, KeyError, TypeError):
                    continue
            
            # Sort by volume (descending)
            new_listings.sort(key=lambda x: x['volume24hUSD'], reverse=True)
            
            return pd.DataFrame(new_listings[:self.config.get('display_new_listings', 30)]) if new_listings else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching new listings from Binance: {e}")
            return pd.DataFrame()
    
    async def fetch_high_volume_tokens(self) -> pd.DataFrame:
        """Fetch high volume tokens from Binance."""
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return pd.DataFrame()
        
        try:
            await asyncio.sleep(self.api_delay)
            
            # Get CCXT client from adapter
            ccxt_client = getattr(self.exchange_adapter, 'ccxt_client', None)
            if not ccxt_client:
                logger.warning("CCXT client not available")
                return pd.DataFrame()
            
            tickers = await ccxt_client.fetch_tickers()
            
            tokens = []
            for symbol, ticker in tickers.items():
                try:
                    if self.focus_usdt_pairs and not symbol.endswith('/USDT'):
                        continue
                    
                    if any(ignored in symbol for ignored in self.ignore_list):
                        continue
                    
                    price = float(ticker.get('last', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    price_change_pct = float(ticker.get('percentage', 0))
                    count = int(ticker.get('count', 0))
                    
                    if volume < self.min_volume_24h:
                        continue
                    
                    token = {
                        'symbol': symbol,
                        'name': symbol.replace('/USDT', '').replace('/', ''),
                        'price': price,
                        'volume24hUSD': volume,
                        'price24hChangePercent': price_change_pct,
                        'trades24h': count,
                        'address': symbol
                    }
                    
                    tokens.append(token)
                    
                except (ValueError, KeyError, TypeError):
                    continue
            
            # Sort by volume (descending)
            tokens.sort(key=lambda x: x['volume24hUSD'], reverse=True)
            
            return pd.DataFrame(tokens[:50]) if tokens else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching high volume tokens from Binance: {e}")
            return pd.DataFrame()


class BitfinexMarketClient(BaseMarketClient):
    """Bitfinex market client using CCXT."""
    
    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        super().__init__(exchange_adapter, config)
        
        # Bitfinex-specific config
        self.focus_usd_pairs = self.config.get('focus_usd_pairs', True)
        self.gems_max_price = self.config.get('gems_max_price', 5.0)
        self.gems_min_volume = self.config.get('gems_min_volume', 50000)
        self.min_new_listing_volume = self.config.get('min_new_listing_volume', 10000)
        self.max_new_listing_volume = self.config.get('max_new_listing_volume', 1000000)
        self.include_funding = self.config.get('include_funding_analysis', True)
    
    async def fetch_trending_tokens(self) -> pd.DataFrame:
        """Fetch trending tokens from Bitfinex."""
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return pd.DataFrame()
        
        try:
            await asyncio.sleep(self.api_delay)
            
            # Get CCXT client from adapter
            ccxt_client = getattr(self.exchange_adapter, 'ccxt_client', None)
            if not ccxt_client:
                logger.warning("CCXT client not available")
                return pd.DataFrame()
            
            # Get tickers
            tickers = await ccxt_client.fetch_tickers()
            
            tokens = []
            for symbol, ticker in tickers.items():
                try:
                    # Filter by USD pairs
                    if self.focus_usd_pairs and not symbol.endswith('/USD'):
                        continue
                    
                    # Skip margin pairs (starting with 'f')
                    if ':USD' in symbol or symbol.startswith('f'):
                        continue
                    
                    # Skip ignored symbols
                    if any(ignored in symbol for ignored in self.ignore_list):
                        continue
                    
                    # Extract ticker data
                    price = float(ticker.get('last', 0))
                    price_change_pct = float(ticker.get('percentage', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    high_24h = float(ticker.get('high', 0))
                    low_24h = float(ticker.get('low', 0))
                    bid = float(ticker.get('bid', 0))
                    ask = float(ticker.get('ask', 0))
                    
                    # Calculate USD volume
                    volume_usd = volume * price if volume > 0 else 0
                    
                    # Filter by minimum volume and price change
                    if volume_usd < self.min_volume_24h or abs(price_change_pct) < self.min_price_change:
                        continue
                    
                    token = {
                        'symbol': symbol,
                        'name': symbol.replace('/USD', '').replace('/', ''),
                        'price': price,
                        'price24hChangePercent': price_change_pct,
                        'priceChange24h': (price - ticker.get('open', price)) if ticker.get('open') else 0,
                        'volume24hUSD': volume_usd,
                        'volume24h': volume,
                        'high24h': high_24h,
                        'low24h': low_24h,
                        'bid': bid,
                        'ask': ask,
                        'address': symbol,
                        'rank': 0
                    }
                    
                    tokens.append(token)
                    
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Error processing ticker {symbol}: {e}")
                    continue
            
            # Sort by price change percentage (descending)
            tokens.sort(key=lambda x: x['price24hChangePercent'], reverse=True)
            
            # Add rank
            for i, token in enumerate(tokens):
                token['rank'] = i + 1
            
            return pd.DataFrame(tokens[:self.display_limit]) if tokens else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching trending tokens from Bitfinex: {e}")
            return pd.DataFrame()
    
    async def fetch_new_listings(self) -> pd.DataFrame:
        """Fetch new listings from Bitfinex."""
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return pd.DataFrame()
        
        try:
            await asyncio.sleep(self.api_delay)
            
            # Get CCXT client from adapter
            ccxt_client = getattr(self.exchange_adapter, 'ccxt_client', None)
            if not ccxt_client:
                logger.warning("CCXT client not available")
                return pd.DataFrame()
            
            # Get markets
            markets = await ccxt_client.load_markets()
            
            # Get tickers
            tickers = await ccxt_client.fetch_tickers()
            
            new_listings = []
            for symbol, market_info in markets.items():
                try:
                    # Filter by USD pairs
                    if self.focus_usd_pairs and not symbol.endswith('/USD'):
                        continue
                    
                    if ':USD' in symbol or symbol.startswith('f'):
                        continue
                    
                    if market_info.get('active') != True:
                        continue
                    
                    # Skip ignored symbols
                    if any(ignored in symbol for ignored in self.ignore_list):
                        continue
                    
                    # Get ticker data
                    ticker = tickers.get(symbol)
                    if not ticker:
                        continue
                    
                    price = float(ticker.get('last', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    volume_usd = volume * price if volume > 0 else 0
                    price_change_pct = float(ticker.get('percentage', 0))
                    volume_base = float(ticker.get('baseVolume', 0))
                    
                    # Filter by volume range (potentially new tokens)
                    if (volume_usd < self.min_new_listing_volume or 
                        volume_usd > self.max_new_listing_volume):
                        continue
                    
                    listing = {
                        'symbol': symbol,
                        'name': symbol.replace('/USD', '').replace('/', ''),
                        'price': price,
                        'volume24hUSD': volume_usd,
                        'volume24h': volume_base,
                        'price24hChangePercent': price_change_pct,
                        'address': symbol,
                        'listed_time': 'Unknown',
                        'status': 'TRADING'
                    }
                    
                    new_listings.append(listing)
                    
                except (ValueError, KeyError, TypeError):
                    continue
            
            # Sort by volume (ascending) to get potentially newer/smaller tokens
            new_listings.sort(key=lambda x: x['volume24hUSD'])
            
            return pd.DataFrame(new_listings[:self.config.get('display_new_listings', 30)]) if new_listings else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching new listings from Bitfinex: {e}")
            return pd.DataFrame()
    
    async def fetch_high_volume_tokens(self) -> pd.DataFrame:
        """Fetch high volume tokens from Bitfinex."""
        if not self.exchange_adapter:
            logger.warning("Exchange adapter not available")
            return pd.DataFrame()
        
        try:
            await asyncio.sleep(self.api_delay)
            
            # Get CCXT client from adapter
            ccxt_client = getattr(self.exchange_adapter, 'ccxt_client', None)
            if not ccxt_client:
                logger.warning("CCXT client not available")
                return pd.DataFrame()
            
            tickers = await ccxt_client.fetch_tickers()
            
            tokens = []
            for symbol, ticker in tickers.items():
                try:
                    if self.focus_usd_pairs and not symbol.endswith('/USD'):
                        continue
                    
                    if ':USD' in symbol or symbol.startswith('f'):
                        continue
                    
                    if any(ignored in symbol for ignored in self.ignore_list):
                        continue
                    
                    price = float(ticker.get('last', 0))
                    volume = float(ticker.get('quoteVolume', 0))
                    volume_usd = volume * price if volume > 0 else 0
                    price_change_pct = float(ticker.get('percentage', 0))
                    volume_base = float(ticker.get('baseVolume', 0))
                    
                    if volume_usd < self.min_volume_24h:
                        continue
                    
                    token = {
                        'symbol': symbol,
                        'name': symbol.replace('/USD', '').replace('/', ''),
                        'price': price,
                        'volume24hUSD': volume_usd,
                        'volume24h': volume_base,
                        'price24hChangePercent': price_change_pct,
                        'address': symbol
                    }
                    
                    tokens.append(token)
                    
                except (ValueError, KeyError, TypeError):
                    continue
            
            # Sort by volume (descending)
            tokens.sort(key=lambda x: x['volume24hUSD'], reverse=True)
            
            return pd.DataFrame(tokens[:50]) if tokens else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching high volume tokens from Bitfinex: {e}")
            return pd.DataFrame()

