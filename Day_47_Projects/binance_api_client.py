'''Binance API Client for Token Tracking'''

import requests
import time
import pandas as pd
import hmac
import hashlib
import urllib.parse
import binance_config as config
from datetime import datetime, timedelta

class BinanceAPI:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com/api/v3"
        self.fapi_url = "https://fapi.binance.com/fapi/v1"

    def _get_signature(self, params):
        """Generate HMAC SHA256 signature for authenticated requests"""
        if not self.api_secret:
            return None
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, endpoint, params=None, authenticated=False, use_fapi=False):
        """Make API request to Binance"""
        if params is None:
            params = {}
        
        base_url = self.fapi_url if use_fapi else self.base_url
        
        if authenticated and self.api_key:
            params['timestamp'] = int(time.time() * 1000)
            if self.api_secret:
                params['signature'] = self._get_signature(params)
            headers = {'X-MBX-APIKEY': self.api_key}
        else:
            headers = {}
        
        url = f"{base_url}{endpoint}"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_trending_tokens(self):
        """Fetch top performing tokens by 24h price change"""
        try:
            # Get 24hr ticker statistics
            data = self._make_request("/ticker/24hr")
            
            # Filter and process the data
            tokens = []
            for ticker in data:
                try:
                    symbol = ticker['symbol']
                    # Focus on USDT pairs for consistency
                    if not symbol.endswith('USDT'):
                        continue
                    
                    # Skip stablecoins and known non-tokens
                    if any(stable in symbol for stable in config.IGNORE_LIST):
                        continue
                    
                    token = {
                        'symbol': symbol,
                        'name': symbol.replace('USDT', ''),  # Use base symbol as name
                        'price': float(ticker['lastPrice']),
                        'price24hChangePercent': float(ticker['priceChangePercent']),
                        'volume24hUSD': float(ticker['quoteVolume']),  # Already in USDT
                        'priceChange24h': float(ticker['priceChange']),
                        'high24h': float(ticker['highPrice']),
                        'low24h': float(ticker['lowPrice']),
                        'trades24h': int(ticker['count']),
                        'address': symbol,  # Use symbol as identifier
                        'rank': 0  # Will be set during sorting
                    }
                    
                    # Only include tokens with significant volume
                    if token['volume24hUSD'] >= config.MIN_VOLUME_24H:
                        tokens.append(token)
                        
                except (ValueError, KeyError) as e:
                    continue
            
            # Sort by price change percentage (descending)
            tokens.sort(key=lambda x: x['price24hChangePercent'], reverse=True)
            
            # Add rank
            for i, token in enumerate(tokens):
                token['rank'] = i + 1
            
            return pd.DataFrame(tokens[:config.DISPLAY_TRENDING_TOKENS]) if tokens else pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trending tokens from Binance: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while fetching trending tokens: {e}")
            return pd.DataFrame()

    def fetch_new_listings(self):
        """Fetch recently added trading pairs"""
        try:
            # Get exchange info to find recent listings
            exchange_info = self._make_request("/exchangeInfo")
            
            # Get current time
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=config.NEW_LISTING_DAYS)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            # Get 24hr ticker for volume data
            ticker_data = self._make_request("/ticker/24hr")
            ticker_dict = {t['symbol']: t for t in ticker_data}
            
            new_listings = []
            for symbol_info in exchange_info['symbols']:
                try:
                    symbol = symbol_info['symbol']
                    
                    # Focus on USDT pairs
                    if not symbol.endswith('USDT') or symbol_info['status'] != 'TRADING':
                        continue
                    
                    # Skip known stablecoins
                    if any(stable in symbol for stable in config.IGNORE_LIST):
                        continue
                    
                    # Check if it's a recent listing (using orderTypes as proxy for recent activity)
                    # This is an approximation since Binance doesn't provide listing timestamps
                    ticker = ticker_dict.get(symbol)
                    if not ticker:
                        continue
                    
                    # Consider high volume new pairs as potential new listings
                    volume_24h = float(ticker['quoteVolume'])
                    price_change = float(ticker['priceChangePercent'])
                    
                    listing = {
                        'symbol': symbol,
                        'name': symbol.replace('USDT', ''),
                        'volume24hUSD': volume_24h,
                        'price': float(ticker['lastPrice']),
                        'price24hChangePercent': price_change,
                        'trades24h': int(ticker['count']),
                        'address': symbol,
                        'listed_time': 'Recent',  # Placeholder since exact listing time not available
                        'status': symbol_info['status']
                    }
                    
                    # Only include pairs with reasonable volume
                    if volume_24h >= config.MIN_NEW_LISTING_VOLUME:
                        new_listings.append(listing)
                        
                except (ValueError, KeyError):
                    continue
            
            # Sort by 24h volume (descending) as proxy for interest in new listings
            new_listings.sort(key=lambda x: x['volume24hUSD'], reverse=True)
            
            return pd.DataFrame(new_listings[:config.DISPLAY_NEW_LISTINGS]) if new_listings else pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching new listings from Binance: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while fetching new listings: {e}")
            return pd.DataFrame()

    def fetch_high_volume_tokens(self):
        """Fetch tokens with highest 24h trading volume"""
        try:
            data = self._make_request("/ticker/24hr")
            
            tokens = []
            for ticker in data:
                try:
                    symbol = ticker['symbol']
                    if not symbol.endswith('USDT'):
                        continue
                    
                    if any(stable in symbol for stable in config.IGNORE_LIST):
                        continue
                    
                    token = {
                        'symbol': symbol,
                        'name': symbol.replace('USDT', ''),
                        'price': float(ticker['lastPrice']),
                        'volume24hUSD': float(ticker['quoteVolume']),
                        'price24hChangePercent': float(ticker['priceChangePercent']),
                        'trades24h': int(ticker['count']),
                        'address': symbol
                    }
                    
                    if token['volume24hUSD'] >= config.MIN_VOLUME_24H:
                        tokens.append(token)
                        
                except (ValueError, KeyError):
                    continue
            
            # Sort by volume
            tokens.sort(key=lambda x: x['volume24hUSD'], reverse=True)
            
            return pd.DataFrame(tokens[:50]) if tokens else pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching high volume tokens: {e}")
            return pd.DataFrame()

    def get_symbol_info(self, symbol):
        """Get detailed information about a specific symbol"""
        try:
            # Get ticker info
            ticker = self._make_request(f"/ticker/24hr", params={'symbol': symbol})
            
            # Get exchange info for symbol details
            exchange_info = self._make_request("/exchangeInfo")
            symbol_info = None
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    symbol_info = s
                    break
            
            if symbol_info:
                return {
                    'symbol': symbol,
                    'status': symbol_info['status'],
                    'price': float(ticker['lastPrice']),
                    'volume24h': float(ticker['quoteVolume']),
                    'priceChange24h': float(ticker['priceChangePercent']),
                    'high24h': float(ticker['highPrice']),
                    'low24h': float(ticker['lowPrice'])
                }
            
            return None
            
        except Exception as e:
            print(f"Error getting symbol info for {symbol}: {e}")
            return None
