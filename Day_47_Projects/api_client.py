'''Birdeye API Client'''

import requests
import time
import pandas as pd
import config

class BirdeyeAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://public-api.birdeye.so/defi"

    def _make_request(self, endpoint, params=None):
        headers = {
            "accept": "application/json",
            "x-chain": "solana",
            "X-API-KEY": self.api_key
        }
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()

    def fetch_trending_tokens(self):
        all_tokens = []
        offset = 0
        try:
            while len(all_tokens) < config.DISPLAY_TRENDING_TOKENS:
                params = {
                    "sort_by": "rank",
                    "sort_type": "asc",
                    "offset": offset,
                    "limit": 20
                }
                data = self._make_request("/token_trending", params=params)
                
                tokens = []
                if 'data' in data and 'tokens' in data['data']:
                    tokens = data['data']['tokens']
                elif 'data' in data and 'items' in data['data']:
                    tokens = data['data']['items']
                    
                if not tokens:
                    break
                    
                all_tokens.extend(tokens)
                offset += 20
                time.sleep(0.5) # Respect API rate limits
                
                if offset >= 100 or len(all_tokens) >= config.DISPLAY_TRENDING_TOKENS:
                    break
            
            return pd.DataFrame(all_tokens[:config.DISPLAY_TRENDING_TOKENS]) if all_tokens else pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trending tokens: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while fetching trending tokens: {e}")
            return pd.DataFrame()

    def fetch_new_listings(self):
        all_listings = []
        offset = 0
        try:
            while len(all_listings) < config.DISPLAY_NEW_LISTINGS:
                params = {
                    "offset": offset,
                    "limit": 20,
                    "meme_platform_enabled": "false"
                }
                data = self._make_request("/v2/tokens/new_listing", params=params)
                
                if 'data' in data and 'items' in data['data']:
                    listings = data['data']['items']
                    if not listings:
                        break
                        
                    all_listings.extend(listings)
                    offset += 20
                    time.sleep(0.5) # Respect API rate limits
                    
                    if offset >= 100 or len(all_listings) >= config.DISPLAY_NEW_LISTINGS:
                        break
                else:
                    break # No data or items found
            
            return pd.DataFrame(all_listings[:config.DISPLAY_NEW_LISTINGS]) if all_listings else pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching new listings: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while fetching new listings: {e}")
            return pd.DataFrame() 