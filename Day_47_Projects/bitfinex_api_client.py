'''Bitfinex API Client for Token Tracking'''

import requests
import time
import pandas as pd
import hmac
import hashlib
import json
import bitfinex_config as config

class BitfinexAPI:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-pub.bitfinex.com/v2"
        self.auth_url = "https://api.bitfinex.com/v2/auth"

    def _get_signature(self, path, body):
        """Generate HMAC SHA384 signature for authenticated requests"""
        if not self.api_secret:
            return None
        nonce = str(int(time.time() * 1000000))
        message = f"/api/v2/{path}{nonce}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha384
        ).hexdigest()
        return signature, nonce

    def _make_request(self, endpoint, params=None, authenticated=False):
        """Make API request to Bitfinex"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        if authenticated and self.api_key and self.api_secret:
            # For authenticated requests (not used in this public data client)
            body = json.dumps(params)
            signature, nonce = self._get_signature(endpoint, body)
            headers = {
                'bfx-nonce': nonce,
                'bfx-apikey': self.api_key,
                'bfx-signature': signature,
                'content-type': 'application/json'
            }
            response = requests.post(url, headers=headers, data=body)
        else:
            # Public requests
            response = requests.get(url, params=params)
        
        response.raise_for_status()
        return response.json()

    def fetch_trending_tokens(self):
        """Fetch trending tokens based on 24h volume and price change"""
        try:
            # Get ticker data for all symbols
            tickers = self._make_request("/tickers", params={"symbols": "ALL"})
            
            tokens = []
            for ticker in tickers:
                try:
                    if len(ticker) < 11:  # Skip invalid tickers
                        continue
                    
                    symbol = ticker[0]  # Symbol
                    
                    # Focus on USD pairs for consistency
                    if not symbol.endswith('USD') or symbol.startswith('f'):
                        continue
                    
                    # Skip known stablecoins and major pairs if configured
                    if any(ignored in symbol.upper() for ignored in config.IGNORE_LIST):
                        continue
                    
                    # Extract ticker data
                    bid = float(ticker[1]) if ticker[1] else 0
                    ask = float(ticker[3]) if ticker[3] else 0
                    daily_change = float(ticker[5]) if ticker[5] else 0
                    daily_change_rel = float(ticker[6]) if ticker[6] else 0
                    last_price = float(ticker[7]) if ticker[7] else 0
                    volume = float(ticker[8]) if ticker[8] else 0
                    high = float(ticker[9]) if ticker[9] else 0
                    low = float(ticker[10]) if ticker[10] else 0
                    
                    # Calculate USD volume
                    volume_usd = volume * last_price
                    
                    token = {
                        'symbol': symbol,
                        'name': symbol.replace('USD', ''),
                        'price': last_price,
                        'price24hChangePercent': daily_change_rel * 100,  # Convert to percentage
                        'priceChange24h': daily_change,
                        'volume24hUSD': volume_usd,
                        'volume24h': volume,
                        'high24h': high,
                        'low24h': low,
                        'bid': bid,
                        'ask': ask,
                        'address': symbol,
                        'rank': 0
                    }
                    
                    # Only include tokens with significant volume
                    if volume_usd >= config.MIN_VOLUME_24H and last_price > 0:
                        tokens.append(token)
                        
                except (ValueError, IndexError, TypeError):
                    continue
            
            # Sort by price change percentage (descending)
            tokens.sort(key=lambda x: x['price24hChangePercent'], reverse=True)
            
            # Add rank
            for i, token in enumerate(tokens):
                token['rank'] = i + 1
            
            return pd.DataFrame(tokens[:config.DISPLAY_TRENDING_TOKENS]) if tokens else pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trending tokens from Bitfinex: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while fetching trending tokens: {e}")
            return pd.DataFrame()

    def fetch_new_listings(self):
        """Fetch potentially new or less known trading pairs based on volume patterns"""
        try:
            # Get all available symbols
            symbols_data = self._make_request("/conf/pub:list:pair:exchange")
            
            if not symbols_data or len(symbols_data) == 0:
                return pd.DataFrame()
            
            symbols = symbols_data[0] if isinstance(symbols_data[0], list) else []
            
            # Filter for USD pairs
            usd_symbols = [s for s in symbols if s.endswith('USD') and not s.startswith('f')]
            
            # Get ticker data for these symbols
            symbol_list = ','.join([f't{s}' for s in usd_symbols[:100]])  # Limit to avoid API limits
            tickers = self._make_request("/tickers", params={"symbols": symbol_list})
            
            new_listings = []
            for ticker in tickers:
                try:
                    if len(ticker) < 11:
                        continue
                    
                    symbol = ticker[0]  # Remove 't' prefix
                    if symbol.startswith('t'):
                        symbol = symbol[1:]
                    
                    # Skip major pairs and stablecoins
                    if any(ignored in symbol.upper() for ignored in config.IGNORE_LIST):
                        continue
                    
                    last_price = float(ticker[7]) if ticker[7] else 0
                    volume = float(ticker[8]) if ticker[8] else 0
                    daily_change_rel = float(ticker[6]) if ticker[6] else 0
                    
                    volume_usd = volume * last_price
                    
                    listing = {
                        'symbol': symbol,
                        'name': symbol.replace('USD', ''),
                        'price': last_price,
                        'volume24hUSD': volume_usd,
                        'price24hChangePercent': daily_change_rel * 100,
                        'volume24h': volume,
                        'address': symbol,
                        'listed_time': 'Unknown',  # Bitfinex doesn't provide listing timestamps
                        'status': 'TRADING'
                    }
                    
                    # Filter for reasonable volume but not too high (potentially new)
                    if (volume_usd >= config.MIN_NEW_LISTING_VOLUME and 
                        volume_usd <= config.MAX_NEW_LISTING_VOLUME and 
                        last_price > 0):
                        new_listings.append(listing)
                        
                except (ValueError, IndexError, TypeError):
                    continue
            
            # Sort by volume (ascending) to get potentially newer/smaller tokens
            new_listings.sort(key=lambda x: x['volume24hUSD'])
            
            return pd.DataFrame(new_listings[:config.DISPLAY_NEW_LISTINGS]) if new_listings else pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching new listings from Bitfinex: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while fetching new listings: {e}")
            return pd.DataFrame()

    def fetch_high_volume_tokens(self):
        """Fetch tokens with highest 24h trading volume"""
        try:
            tickers = self._make_request("/tickers", params={"symbols": "ALL"})
            
            tokens = []
            for ticker in tickers:
                try:
                    if len(ticker) < 11:
                        continue
                    
                    symbol = ticker[0]
                    if not symbol.endswith('USD') or symbol.startswith('f'):
                        continue
                    
                    if any(ignored in symbol.upper() for ignored in config.IGNORE_LIST):
                        continue
                    
                    last_price = float(ticker[7]) if ticker[7] else 0
                    volume = float(ticker[8]) if ticker[8] else 0
                    daily_change_rel = float(ticker[6]) if ticker[6] else 0
                    
                    volume_usd = volume * last_price
                    
                    token = {
                        'symbol': symbol,
                        'name': symbol.replace('USD', ''),
                        'price': last_price,
                        'volume24hUSD': volume_usd,
                        'price24hChangePercent': daily_change_rel * 100,
                        'volume24h': volume,
                        'address': symbol
                    }
                    
                    if volume_usd >= config.MIN_VOLUME_24H and last_price > 0:
                        tokens.append(token)
                        
                except (ValueError, IndexError, TypeError):
                    continue
            
            # Sort by volume (descending)
            tokens.sort(key=lambda x: x['volume24hUSD'], reverse=True)
            
            return pd.DataFrame(tokens[:50]) if tokens else pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching high volume tokens: {e}")
            return pd.DataFrame()

    def get_symbol_info(self, symbol):
        """Get detailed information about a specific symbol"""
        try:
            # Add 't' prefix for trading symbols
            if not symbol.startswith('t'):
                symbol = f"t{symbol}"
            
            ticker = self._make_request(f"/ticker/{symbol}")
            
            if ticker and len(ticker) >= 11:
                return {
                    'symbol': symbol,
                    'price': float(ticker[6]) if ticker[6] else 0,  # Last price
                    'volume24h': float(ticker[7]) if ticker[7] else 0,  # Volume
                    'priceChange24h': float(ticker[5]) * 100 if ticker[5] else 0,  # Daily change %
                    'high24h': float(ticker[8]) if ticker[8] else 0,  # Daily high
                    'low24h': float(ticker[9]) if ticker[9] else 0,  # Daily low
                    'bid': float(ticker[0]) if ticker[0] else 0,  # Bid
                    'ask': float(ticker[2]) if ticker[2] else 0   # Ask
                }
            
            return None
            
        except Exception as e:
            print(f"Error getting symbol info for {symbol}: {e}")
            return None

    def fetch_funding_rates(self):
        """Fetch funding rates for margin tokens"""
        try:
            # Get funding tickers (starts with 'f')
            tickers = self._make_request("/tickers", params={"symbols": "ALL"})
            
            funding_data = []
            for ticker in tickers:
                try:
                    if len(ticker) < 11 or not ticker[0].startswith('f'):
                        continue
                    
                    symbol = ticker[0]  # Funding symbol
                    
                    # Skip if not USD funding
                    if not symbol.endswith('USD'):
                        continue
                    
                    daily_change_rel = float(ticker[6]) if ticker[6] else 0
                    last_rate = float(ticker[7]) if ticker[7] else 0
                    volume = float(ticker[8]) if ticker[8] else 0
                    
                    funding_info = {
                        'symbol': symbol,
                        'base_symbol': symbol.replace('f', '').replace('USD', ''),
                        'funding_rate': last_rate,
                        'rate_change_24h': daily_change_rel * 100,
                        'volume': volume,
                        'address': symbol
                    }
                    
                    if abs(last_rate) > 0:  # Only include active funding
                        funding_data.append(funding_info)
                        
                except (ValueError, IndexError, TypeError):
                    continue
            
            return pd.DataFrame(funding_data) if funding_data else pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching funding rates: {e}")
            return pd.DataFrame()
