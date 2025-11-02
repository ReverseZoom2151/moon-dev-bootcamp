"""
🚀 Binance Trading API Handler
Comprehensive Binance API wrapper with data fetching and analysis capabilities

Disclaimer: This is not financial advice and there is no guarantee of any kind. Use at your own risk.

Quick Start Guide:
-----------------
1. Install required packages:
   ```
   pip install requests pandas python-dotenv
   ```

2. Create a .env file or configure dontshareconfig.py:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   ```

3. Basic Usage:
   ```python
   from binance_api import BinanceAPI
   
   # Initialize
   api = BinanceAPI()
   
   # Get data
   liquidations = api.get_liquidation_data('BTCUSDT', limit=1000)
   funding = api.get_funding_data()
   oi = api.get_oi_data('BTCUSDT')
   ```

Available Methods:
----------------
- get_liquidation_data(symbol, limit=None): Get liquidation/trade data for a symbol
- get_funding_data(): Get current funding rates (futures only)
- get_token_addresses(): Get all available trading symbols
- get_oi_data(symbol): Get open interest data for a symbol (futures)
- get_oi_total(): Get total open interest across major symbols
- get_whale_addresses(): Get large holders analysis
- get_recent_transactions(symbol): Get recent large trades
- get_agg_positions(symbol): Get aggregated position data
- get_order_book_depth(symbol): Get detailed order book analysis
"""

import os
import sys
import time
import hmac
import hashlib
import urllib.parse
import requests
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

# Add parent directory for config imports
sys.path.append('..')
try:
    from dontshareconfig import binance_api_key, binance_api_secret
except ImportError:
    binance_api_key = os.getenv('BINANCE_API_KEY')
    binance_api_secret = os.getenv('BINANCE_SECRET_KEY')

# Setup logging
SCRIPT_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
api_logger = logging.getLogger(__name__)

class BinanceAPI:
    """Comprehensive Binance API wrapper for trading data analysis."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 data_dir: Optional[Path | str] = None,
                 testnet: bool = False):
        """Initialize the Binance API handler.
        
        Args:
            api_key: Binance API key
            api_secret: Binance secret key  
            data_dir: Directory to store downloaded data
            testnet: Use Binance testnet endpoints
        """
        # Setup data directory
        _data_dir_str = data_dir or os.getenv('BINANCE_DATA_DIR')
        if _data_dir_str:
            self.base_dir = Path(_data_dir_str).resolve()
        else:
            self.base_dir = SCRIPT_DIR / "data" / "binance_api"
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        api_logger.info(f"Binance data directory: {self.base_dir}")
        
        # Setup API credentials
        self.api_key = api_key or binance_api_key
        self.api_secret = api_secret or binance_api_secret
        
        # Setup endpoints
        if testnet:
            self.spot_base_url = "https://testnet.binance.vision/api"
            self.futures_base_url = "https://testnet.binancefuture.com"
        else:
            self.spot_base_url = "https://api.binance.com/api"
            self.futures_base_url = "https://fapi.binance.com"
        
        api_logger.info(f"Binance API endpoints: Spot={self.spot_base_url}, Futures={self.futures_base_url}")
        
        # Setup session
        self.session = requests.Session()
        self.max_retries = 3
        self.rate_limit_delay = 0.1  # seconds between requests
        
        if not self.api_key:
            api_logger.warning("Binance API key not provided. Public endpoints only.")
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests."""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: Dict = None, 
                     signed: bool = False, futures: bool = False) -> Optional[Dict]:
        """Make authenticated request to Binance API."""
        if params is None:
            params = {}
        
        base_url = self.futures_base_url if futures else self.spot_base_url
        url = f"{base_url}{endpoint}"
        
        headers = {}
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        
        if signed and self.api_secret:
            params['timestamp'] = int(time.time() * 1000)
            query_string = urllib.parse.urlencode(params)
            signature = self._generate_signature(query_string)
            params['signature'] = signature
        
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)  # Rate limiting
                response = self.session.get(url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    api_logger.warning(f"Rate limit hit, waiting...")
                    time.sleep(60)
                    continue
                else:
                    api_logger.error(f"API error {response.status_code}: {response.text}")
                    if attempt == self.max_retries - 1:
                        return None
                    
            except requests.exceptions.RequestException as e:
                api_logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        return None
    
    def get_liquidation_data(self, symbol: str = 'BTCUSDT', limit: Optional[int] = 1000) -> Optional[pd.DataFrame]:
        """Get recent trade data (liquidations equivalent) for a symbol."""
        try:
            api_logger.info(f"Fetching trade data for {symbol}")
            
            # Get recent trades (closest to liquidation data)
            data = self._make_request('/v3/aggTrades', {
                'symbol': symbol,
                'limit': min(limit or 1000, 1000)
            })
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            if df.empty:
                return df
            
            # Process trade data
            df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
            df['price'] = df['p'].astype(float)
            df['quantity'] = df['q'].astype(float)
            df['value_usd'] = df['price'] * df['quantity']
            df['is_buyer_maker'] = df['m']
            df['side'] = df['m'].apply(lambda x: 'SELL' if x else 'BUY')
            
            # Add liquidation-like analysis
            df['large_trade'] = df['value_usd'] > df['value_usd'].quantile(0.9)
            df['whale_tier'] = pd.cut(df['value_usd'], 
                                    bins=[0, 10000, 50000, 100000, float('inf')],
                                    labels=['Retail', 'Large', 'Whale', 'Mega Whale'])
            
            # Save to file
            filename = f"binance_trades_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved trade data to {filepath}")
            
            return df[['timestamp', 'price', 'quantity', 'value_usd', 'side', 'large_trade', 'whale_tier']]
            
        except Exception as e:
            api_logger.error(f"Error fetching liquidation data: {e}")
            return None
    
    def get_funding_data(self) -> Optional[pd.DataFrame]:
        """Get current funding rates for futures."""
        try:
            api_logger.info("Fetching funding rate data")
            
            data = self._make_request('/fapi/v1/premiumIndex', futures=True)
            if not data:
                return None
            
            df = pd.DataFrame(data)
            if df.empty:
                return df
            
            # Process funding data
            df['funding_rate_pct'] = (df['lastFundingRate'].astype(float) * 100)
            df['funding_time'] = pd.to_datetime(df['nextFundingTime'], unit='ms')
            df['mark_price'] = df['markPrice'].astype(float)
            df['index_price'] = df['indexPrice'].astype(float)
            
            # Save to file
            filename = f"binance_funding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved funding data to {filepath}")
            
            return df[['symbol', 'funding_rate_pct', 'funding_time', 'mark_price', 'index_price']]
            
        except Exception as e:
            api_logger.error(f"Error fetching funding data: {e}")
            return None
    
    def get_token_addresses(self) -> Optional[pd.DataFrame]:
        """Get all available trading symbols (token addresses equivalent)."""
        try:
            api_logger.info("Fetching exchange information")
            
            data = self._make_request('/v3/exchangeInfo')
            if not data or 'symbols' not in data:
                return None
            
            symbols_data = data['symbols']
            df = pd.DataFrame(symbols_data)
            
            # Filter active symbols
            df = df[df['status'] == 'TRADING']
            
            # Process symbol data  
            df['base_asset'] = df['baseAsset']
            df['quote_asset'] = df['quoteAsset']
            df['is_spot_trading'] = df['permissions'].apply(lambda x: 'SPOT' in x)
            df['is_margin_trading'] = df['permissions'].apply(lambda x: 'MARGIN' in x)
            
            # Save to file
            filename = f"binance_symbols_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved symbols data to {filepath}")
            
            return df[['symbol', 'base_asset', 'quote_asset', 'is_spot_trading', 'is_margin_trading']]
            
        except Exception as e:
            api_logger.error(f"Error fetching token addresses: {e}")
            return None
    
    def get_oi_data(self, symbol: str = 'BTCUSDT') -> Optional[pd.DataFrame]:
        """Get open interest data for a futures symbol."""
        try:
            api_logger.info(f"Fetching open interest data for {symbol}")
            
            # Get current open interest
            oi_data = self._make_request('/fapi/v1/openInterest', {'symbol': symbol}, futures=True)
            
            # Get historical OI
            hist_data = self._make_request('/futures/data/openInterestHist', {
                'symbol': symbol,
                'period': '1h',
                'limit': 100
            }, futures=True)
            
            if not oi_data:
                return None
            
            # Process current OI
            current_oi = {
                'symbol': symbol,
                'open_interest': float(oi_data['openInterest']),
                'timestamp': datetime.utcnow()
            }
            
            # Process historical data if available
            if hist_data:
                df_hist = pd.DataFrame(hist_data)
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'], unit='ms')
                df_hist['open_interest'] = df_hist['sumOpenInterest'].astype(float)
                df_hist['open_interest_value'] = df_hist['sumOpenInterestValue'].astype(float)
                
                # Add current data
                df_current = pd.DataFrame([current_oi])
                df = pd.concat([df_hist, df_current], ignore_index=True)
            else:
                df = pd.DataFrame([current_oi])
            
            # Save to file
            filename = f"binance_oi_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved OI data to {filepath}")
            
            return df[['timestamp', 'symbol', 'open_interest']].tail(100)
            
        except Exception as e:
            api_logger.error(f"Error fetching OI data: {e}")
            return None
    
    def get_oi_total(self) -> Optional[pd.DataFrame]:
        """Get total open interest across major symbols."""
        try:
            api_logger.info("Fetching total open interest data")
            
            major_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
            all_oi_data = []
            
            for symbol in major_symbols:
                oi_data = self.get_oi_data(symbol)
                if oi_data is not None:
                    latest = oi_data.iloc[-1]
                    all_oi_data.append({
                        'symbol': symbol,
                        'open_interest': latest['open_interest'],
                        'timestamp': latest['timestamp']
                    })
            
            if not all_oi_data:
                return None
            
            df = pd.DataFrame(all_oi_data)
            
            # Calculate totals
            total_oi = df['open_interest'].sum()
            df['oi_percentage'] = (df['open_interest'] / total_oi) * 100
            
            # Save to file
            filename = f"binance_total_oi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved total OI data to {filepath}")
            
            return df
            
        except Exception as e:
            api_logger.error(f"Error fetching total OI data: {e}")
            return None
    
    def get_whale_addresses(self) -> Optional[pd.DataFrame]:
        """Get large holder analysis from order book depth."""
        try:
            api_logger.info("Analyzing whale activity from order book depth")
            
            major_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            whale_data = []
            
            for symbol in major_symbols:
                depth_data = self._make_request('/v3/depth', {'symbol': symbol, 'limit': 1000})
                
                if depth_data:
                    # Analyze large orders in order book
                    for bid in depth_data['bids']:
                        price, quantity = float(bid[0]), float(bid[1])
                        value = price * quantity
                        
                        if value > 50000:  # $50k+ orders
                            whale_data.append({
                                'symbol': symbol,
                                'side': 'BID',
                                'price': price,
                                'quantity': quantity,
                                'value_usd': value,
                                'whale_tier': 'Whale' if value > 100000 else 'Large Trader',
                                'timestamp': datetime.utcnow()
                            })
                    
                    for ask in depth_data['asks']:
                        price, quantity = float(ask[0]), float(ask[1])
                        value = price * quantity
                        
                        if value > 50000:  # $50k+ orders
                            whale_data.append({
                                'symbol': symbol,
                                'side': 'ASK',
                                'price': price,
                                'quantity': quantity,
                                'value_usd': value,
                                'whale_tier': 'Whale' if value > 100000 else 'Large Trader',
                                'timestamp': datetime.utcnow()
                            })
            
            if not whale_data:
                return None
            
            df = pd.DataFrame(whale_data)
            df = df.sort_values('value_usd', ascending=False)
            
            # Save to file
            filename = f"binance_whales_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved whale analysis to {filepath}")
            
            return df.head(100)
            
        except Exception as e:
            api_logger.error(f"Error fetching whale addresses: {e}")
            return None
    
    def get_recent_transactions(self, symbol: str = 'BTCUSDT') -> Optional[pd.DataFrame]:
        """Get recent large transactions for a symbol."""
        try:
            api_logger.info(f"Fetching recent transactions for {symbol}")
            
            data = self._make_request('/v3/aggTrades', {
                'symbol': symbol,
                'limit': 500
            })
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
            df['price'] = df['p'].astype(float)
            df['quantity'] = df['q'].astype(float)
            df['value_usd'] = df['price'] * df['quantity']
            df['side'] = df['m'].apply(lambda x: 'SELL' if x else 'BUY')
            
            # Filter for large transactions
            large_threshold = df['value_usd'].quantile(0.95)
            df_large = df[df['value_usd'] >= large_threshold]
            
            # Save to file
            filename = f"binance_large_txns_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df_large.to_csv(filepath, index=False)
            api_logger.info(f"Saved large transactions to {filepath}")
            
            return df_large[['timestamp', 'price', 'quantity', 'value_usd', 'side']]
            
        except Exception as e:
            api_logger.error(f"Error fetching recent transactions: {e}")
            return None
    
    def get_agg_positions(self, symbol: str = 'BTCUSDT') -> Optional[pd.DataFrame]:
        """Get aggregated position data from long/short ratios."""
        try:
            api_logger.info(f"Fetching position data for {symbol}")
            
            # Get long/short ratio
            ratio_data = self._make_request('/futures/data/globalLongShortAccountRatio', {
                'symbol': symbol,
                'period': '1h',
                'limit': 100
            }, futures=True)
            
            if not ratio_data:
                return None
            
            df = pd.DataFrame(ratio_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['long_short_ratio'] = df['longShortRatio'].astype(float)
            df['long_account'] = df['longAccount'].astype(float)
            df['short_account'] = df['shortAccount'].astype(float)
            
            # Calculate aggregated metrics
            df['total_accounts'] = df['long_account'] + df['short_account']
            df['long_percentage'] = (df['long_account'] / df['total_accounts']) * 100
            df['short_percentage'] = (df['short_account'] / df['total_accounts']) * 100
            
            # Save to file
            filename = f"binance_positions_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved position data to {filepath}")
            
            return df[['timestamp', 'long_short_ratio', 'long_percentage', 'short_percentage']]
            
        except Exception as e:
            api_logger.error(f"Error fetching aggregated positions: {e}")
            return None
    
    def get_order_book_depth(self, symbol: str = 'BTCUSDT') -> Optional[pd.DataFrame]:
        """Get detailed order book analysis."""
        try:
            api_logger.info(f"Fetching order book depth for {symbol}")
            
            data = self._make_request('/v3/depth', {'symbol': symbol, 'limit': 1000})
            
            if not data:
                return None
            
            # Process bids and asks
            bids_data = []
            asks_data = []
            
            for bid in data['bids']:
                price, quantity = float(bid[0]), float(bid[1])
                bids_data.append({
                    'price': price,
                    'quantity': quantity,
                    'value_usd': price * quantity,
                    'side': 'BID'
                })
            
            for ask in data['asks']:
                price, quantity = float(ask[0]), float(ask[1])
                asks_data.append({
                    'price': price,
                    'quantity': quantity,
                    'value_usd': price * quantity,
                    'side': 'ASK'
                })
            
            # Combine and analyze
            all_orders = bids_data + asks_data
            df = pd.DataFrame(all_orders)
            
            # Add market depth analysis
            df['cumulative_volume'] = df.groupby('side')['quantity'].cumsum()
            df['cumulative_value'] = df.groupby('side')['value_usd'].cumsum()
            df['timestamp'] = datetime.utcnow()
            
            # Save to file
            filename = f"binance_depth_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved order book depth to {filepath}")
            
            return df
            
        except Exception as e:
            api_logger.error(f"Error fetching order book depth: {e}")
            return None

if __name__ == "__main__":
    # Test the Binance API
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    api = BinanceAPI()
    
    print("🚀 Testing Binance API...")
    
    # Test liquidation data
    print("\n📊 Testing trade data...")
    liquidations = api.get_liquidation_data('BTCUSDT', limit=100)
    if liquidations is not None:
        print(f"✅ Got {len(liquidations)} trade records")
        print(liquidations.head())
    
    # Test funding data
    print("\n💰 Testing funding data...")
    funding = api.get_funding_data()
    if funding is not None:
        print(f"✅ Got funding data for {len(funding)} symbols")
        print(funding.head())
    
    # Test whale analysis
    print("\n🐋 Testing whale analysis...")
    whales = api.get_whale_addresses()
    if whales is not None:
        print(f"✅ Found {len(whales)} whale orders")
        print(whales.head())
    
    print("\n✅ Binance API testing completed!")
