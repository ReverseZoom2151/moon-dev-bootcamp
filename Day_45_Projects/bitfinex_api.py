"""
🏛️ Bitfinex Professional Trading API Handler
Institutional-grade Bitfinex API wrapper with advanced data analysis capabilities

Disclaimer: This is not financial advice and there is no guarantee of any kind. Use at your own risk.

Quick Start Guide:
-----------------
1. Install required packages:
   ```
   pip install requests pandas python-dotenv
   ```

2. Create a .env file or configure dontshareconfig.py:
   ```
   BITFINEX_API_KEY=your_api_key_here
   BITFINEX_SECRET_KEY=your_secret_key_here
   ```

3. Basic Usage:
   ```python
   from bitfinex_api import BitfinexAPI
   
   # Initialize
   api = BitfinexAPI()
   
   # Get data
   liquidations = api.get_liquidation_data('tBTCUSD', limit=1000)
   funding = api.get_funding_data()
   oi = api.get_oi_data('tBTCUSD')
   ```

Available Methods:
----------------
- get_liquidation_data(symbol, limit=None): Get trade/liquidation data for a symbol
- get_funding_data(): Get current funding rates and stats
- get_token_addresses(): Get all available trading pairs
- get_oi_data(symbol): Get open interest equivalent (position data)
- get_oi_total(): Get total position data across major symbols
- get_whale_addresses(): Get large holder analysis from order books
- get_recent_transactions(symbol): Get recent large trades
- get_agg_positions(symbol): Get aggregated position data
- get_order_book_depth(symbol): Get detailed order book analysis
"""

import os
import sys
import time
import hmac
import hashlib
import json
import requests
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory for config imports
sys.path.append('..')
try:
    from dontshareconfig import bitfinex_api_key, bitfinex_api_secret
except ImportError:
    bitfinex_api_key = os.getenv('BITFINEX_API_KEY')
    bitfinex_api_secret = os.getenv('BITFINEX_SECRET_KEY')

# Setup logging
SCRIPT_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
api_logger = logging.getLogger(__name__)

class BitfinexAPI:
    """Professional Bitfinex API wrapper for institutional trading analysis."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 data_dir: Optional[Path | str] = None):
        """Initialize the Bitfinex API handler."""
        
        # Setup data directory
        _data_dir_str = data_dir or os.getenv('BITFINEX_DATA_DIR')
        if _data_dir_str:
            self.base_dir = Path(_data_dir_str).resolve()
        else:
            self.base_dir = SCRIPT_DIR / "data" / "bitfinex_api"
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
        api_logger.info(f"Bitfinex data directory: {self.base_dir}")
        
        # Setup API credentials
        self.api_key = api_key or bitfinex_api_key
        self.api_secret = api_secret or bitfinex_api_secret
        
        # Setup endpoints
        self.base_url = "https://api-pub.bitfinex.com/v2"
        self.auth_url = "https://api.bitfinex.com/v2/auth"
        
        api_logger.info(f"Bitfinex API endpoint: {self.base_url}")
        
        # Setup session
        self.session = requests.Session()
        self.max_retries = 3
        self.rate_limit_delay = 0.2
        
        if not self.api_key:
            api_logger.warning("Bitfinex API key not provided. Public endpoints only.")
    
    def _generate_signature(self, path: str, nonce: str, body: str = "") -> str:
        """Generate HMAC SHA384 signature for authenticated requests."""
        message = f'/v2/auth{path}{nonce}{body}'
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha384
        ).hexdigest()
        return signature
    
    def _make_request(self, endpoint: str, params: Dict = None, 
                     authenticated: bool = False) -> Optional[Any]:
        """Make request to Bitfinex API."""
        if params is None:
            params = {}
        
        if authenticated:
            url = f"{self.auth_url}{endpoint}"
            nonce = str(int(time.time() * 1000000))
            body = json.dumps(params) if params else ""
            signature = self._generate_signature(endpoint, nonce, body)
            
            headers = {
                'bfx-nonce': nonce,
                'bfx-apikey': self.api_key,
                'bfx-signature': signature,
                'content-type': 'application/json'
            }
            
            for attempt in range(self.max_retries):
                try:
                    time.sleep(self.rate_limit_delay)
                    response = self.session.post(url, data=body, headers=headers, timeout=30)
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        api_logger.error(f"Auth API error {response.status_code}: {response.text}")
                        if attempt == self.max_retries - 1:
                            return None
                        time.sleep(2 ** attempt)
                        
                except requests.exceptions.RequestException as e:
                    api_logger.error(f"Auth request error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
        else:
            url = f"{self.base_url}{endpoint}"
            
            for attempt in range(self.max_retries):
                try:
                    time.sleep(self.rate_limit_delay)
                    response = self.session.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 429:
                        api_logger.warning("Rate limit hit, waiting...")
                        time.sleep(60)
                        continue
                    else:
                        api_logger.error(f"Public API error {response.status_code}: {response.text}")
                        if attempt == self.max_retries - 1:
                            return None
                        
                except requests.exceptions.RequestException as e:
                    api_logger.error(f"Public request error (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                
        return None
    
    def get_liquidation_data(self, symbol: str = 'tBTCUSD', limit: Optional[int] = 1000) -> Optional[pd.DataFrame]:
        """Get recent trade data (liquidations equivalent) for a symbol."""
        try:
            api_logger.info(f"Fetching trade data for {symbol}")
            
            # Get recent trades
            data = self._make_request(f'/trades/{symbol}/hist', {
                'limit': min(limit or 1000, 5000)
            })
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=['ID', 'Timestamp', 'Amount', 'Price'])
            if df.empty:
                return df
            
            # Process trade data
            df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df['price'] = df['Price'].astype(float)
            df['amount'] = df['Amount'].astype(float)
            df['quantity'] = df['amount'].abs()
            df['value_usd'] = df['price'] * df['quantity']
            df['side'] = df['amount'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
            
            # Add institutional analysis
            df['large_trade'] = df['value_usd'] > df['value_usd'].quantile(0.9)
            df['institutional_tier'] = pd.cut(df['value_usd'], 
                                            bins=[0, 25000, 100000, 500000, float('inf')],
                                            labels=['Retail', 'Professional', 'Institutional', 'Sovereign'])
            
            # Save to file
            filename = f"bitfinex_trades_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved institutional trade data to {filepath}")
            
            return df[['timestamp', 'price', 'quantity', 'value_usd', 'side', 'large_trade', 'institutional_tier']]
            
        except Exception as e:
            api_logger.error(f"Error fetching liquidation data: {e}")
            return None
    
    def get_funding_data(self) -> Optional[pd.DataFrame]:
        """Get current funding rates and comprehensive funding statistics."""
        try:
            api_logger.info("Fetching professional funding rate data")
            
            # Get funding rates for major symbols
            major_symbols = ['fUSD', 'fBTC', 'fETH', 'fEUR', 'fGBP']
            funding_data = []
            
            for symbol in major_symbols:
                try:
                    # Get current funding stats
                    stats = self._make_request(f'/stats1/{symbol}:30d:FUNDING/last')
                    
                    if stats:
                        funding_data.append({
                            'symbol': symbol,
                            'funding_rate_daily': float(stats[1]) if len(stats) > 1 else 0,
                            'timestamp': pd.to_datetime(stats[0], unit='ms') if len(stats) > 0 else datetime.utcnow()
                        })
                
                except Exception as e:
                    api_logger.warning(f"Error fetching funding for {symbol}: {e}")
                    continue
            
            if not funding_data:
                return None
            
            df = pd.DataFrame(funding_data)
            
            # Calculate professional metrics
            if len(df) > 1:
                df['funding_rate_annualized'] = df['funding_rate_daily'] * 365
                df['funding_premium'] = df['funding_rate_daily'] - df['funding_rate_daily'].median()
                df['institutional_demand'] = df['funding_rate_daily'] > df['funding_rate_daily'].quantile(0.75)
            
            # Save to file
            filename = f"bitfinex_funding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved professional funding data to {filepath}")
            
            return df
            
        except Exception as e:
            api_logger.error(f"Error fetching funding data: {e}")
            return None
    
    def get_token_addresses(self) -> Optional[pd.DataFrame]:
        """Get all available trading pairs."""
        try:
            api_logger.info("Fetching trading pairs information")
            
            # Get trading symbols
            symbols = self._make_request('/conf/pub:list:pair:exchange')
            
            if not symbols or not isinstance(symbols[0], list):
                return None
            
            # Process symbols
            trading_pairs = []
            for symbol in symbols[0]:
                if symbol.startswith('t'):  # Trading pairs
                    base_quote = symbol[1:]  # Remove 't' prefix
                    if len(base_quote) >= 6:
                        base = base_quote[:3]
                        quote = base_quote[3:]
                        
                        trading_pairs.append({
                            'symbol': symbol,
                            'base_currency': base,
                            'quote_currency': quote,
                            'is_trading': True,
                            'is_margin': True,
                            'is_funding': base in ['USD', 'BTC', 'ETH', 'EUR']
                        })
            
            df = pd.DataFrame(trading_pairs)
            
            # Save to file
            filename = f"bitfinex_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved trading pairs to {filepath}")
            
            return df
            
        except Exception as e:
            api_logger.error(f"Error fetching token addresses: {e}")
            return None
    
    def get_oi_data(self, symbol: str = 'tBTCUSD') -> Optional[pd.DataFrame]:
        """Get position data (open interest equivalent) for a symbol."""
        try:
            api_logger.info(f"Fetching position data for {symbol}")
            
            # Get position stats
            data = self._make_request(f'/stats1/{symbol}:1D:long/hist', {'limit': 100})
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=['Timestamp', 'Value'])
            df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df['long_positions'] = df['Value'].astype(float)
            
            # Get short positions
            short_data = self._make_request(f'/stats1/{symbol}:1D:short/hist', {'limit': 100})
            if short_data:
                df_short = pd.DataFrame(short_data, columns=['Timestamp', 'Value'])
                df_short['short_positions'] = df_short['Value'].astype(float)
                df = df.merge(df_short[['Timestamp', 'short_positions']], on='Timestamp', how='left')
            
            # Calculate institutional metrics
            df['net_positions'] = df.get('long_positions', 0) - df.get('short_positions', 0)
            df['total_positions'] = df.get('long_positions', 0) + df.get('short_positions', 0)
            
            # Save to file
            filename = f"bitfinex_positions_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved position data to {filepath}")
            
            return df[['timestamp', 'long_positions', 'short_positions', 'net_positions']]
            
        except Exception as e:
            api_logger.error(f"Error fetching OI data: {e}")
            return None
    
    def get_oi_total(self) -> Optional[pd.DataFrame]:
        """Get total position data across major symbols."""
        try:
            api_logger.info("Fetching total institutional position data")
            
            major_symbols = ['tBTCUSD', 'tETHUSD', 'tLTCUSD', 'tXRPUSD']
            all_position_data = []
            
            for symbol in major_symbols:
                position_data = self.get_oi_data(symbol)
                if position_data is not None and not position_data.empty:
                    latest = position_data.iloc[-1]
                    all_position_data.append({
                        'symbol': symbol,
                        'long_positions': latest.get('long_positions', 0),
                        'short_positions': latest.get('short_positions', 0),
                        'net_positions': latest.get('net_positions', 0),
                        'timestamp': latest['timestamp']
                    })
            
            if not all_position_data:
                return None
            
            df = pd.DataFrame(all_position_data)
            
            # Save to file
            filename = f"bitfinex_total_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved total position data to {filepath}")
            
            return df
            
        except Exception as e:
            api_logger.error(f"Error fetching total OI data: {e}")
            return None
    
    def get_whale_addresses(self) -> Optional[pd.DataFrame]:
        """Get institutional holder analysis from order book depth."""
        try:
            api_logger.info("Analyzing institutional activity from order books")
            
            major_symbols = ['tBTCUSD', 'tETHUSD', 'tLTCUSD']
            institutional_data = []
            
            for symbol in major_symbols:
                book_data = self._make_request(f'/book/{symbol}/P0', {'len': 100})
                
                if book_data:
                    # Analyze large orders in order book
                    for order in book_data:
                        price, count, amount = float(order[0]), int(order[1]), float(order[2])
                        value = abs(price * amount)
                        
                        if value > 100000:  # $100k+ institutional orders
                            institutional_data.append({
                                'symbol': symbol,
                                'side': 'BID' if amount > 0 else 'ASK',
                                'price': price,
                                'amount': abs(amount),
                                'value_usd': value,
                                'order_count': count,
                                'timestamp': datetime.utcnow()
                            })
            
            if not institutional_data:
                return None
            
            df = pd.DataFrame(institutional_data)
            df = df.sort_values('value_usd', ascending=False)
            
            # Save to file
            filename = f"bitfinex_institutional_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved institutional analysis to {filepath}")
            
            return df.head(50)
            
        except Exception as e:
            api_logger.error(f"Error fetching whale addresses: {e}")
            return None
    
    def get_recent_transactions(self, symbol: str = 'tBTCUSD') -> Optional[pd.DataFrame]:
        """Get recent large institutional transactions."""
        try:
            api_logger.info(f"Fetching institutional transactions for {symbol}")
            
            data = self._make_request(f'/trades/{symbol}/hist', {'limit': 1000})
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=['ID', 'Timestamp', 'Amount', 'Price'])
            df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df['price'] = df['Price'].astype(float)
            df['amount'] = df['Amount'].astype(float)
            df['quantity'] = df['amount'].abs()
            df['value_usd'] = df['price'] * df['quantity']
            df['side'] = df['amount'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
            
            # Filter for institutional transactions
            institutional_threshold = df['value_usd'].quantile(0.9)
            df_institutional = df[df['value_usd'] >= institutional_threshold]
            
            # Save to file
            filename = f"bitfinex_institutional_txns_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df_institutional.to_csv(filepath, index=False)
            api_logger.info(f"Saved institutional transactions to {filepath}")
            
            return df_institutional[['timestamp', 'price', 'quantity', 'value_usd', 'side']]
            
        except Exception as e:
            api_logger.error(f"Error fetching recent transactions: {e}")
            return None
    
    def get_agg_positions(self, symbol: str = 'tBTCUSD') -> Optional[pd.DataFrame]:
        """Get aggregated professional position data."""
        try:
            api_logger.info(f"Fetching aggregated positions for {symbol}")
            
            # Get long positions over time
            long_data = self._make_request(f'/stats1/{symbol}:1h:long/hist', {'limit': 168})
            short_data = self._make_request(f'/stats1/{symbol}:1h:short/hist', {'limit': 168})
            
            if not long_data or not short_data:
                return None
            
            # Process data
            df_long = pd.DataFrame(long_data, columns=['Timestamp', 'LongValue'])
            df_long['timestamp'] = pd.to_datetime(df_long['Timestamp'], unit='ms')
            
            df_short = pd.DataFrame(short_data, columns=['Timestamp', 'ShortValue'])
            df_short['timestamp'] = pd.to_datetime(df_short['Timestamp'], unit='ms')
            
            # Merge data
            df = pd.merge(df_long, df_short, on='timestamp', how='inner')
            df['long_positions'] = df['LongValue'].astype(float)
            df['short_positions'] = df['ShortValue'].astype(float)
            
            # Calculate metrics
            df['net_positions'] = df['long_positions'] - df['short_positions']
            df['total_positions'] = df['long_positions'] + df['short_positions']
            df['long_percentage'] = (df['long_positions'] / df['total_positions']) * 100
            
            # Save to file
            filename = f"bitfinex_agg_positions_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved aggregated positions to {filepath}")
            
            return df[['timestamp', 'long_percentage', 'net_positions']].tail(100)
            
        except Exception as e:
            api_logger.error(f"Error fetching aggregated positions: {e}")
            return None
    
    def get_order_book_depth(self, symbol: str = 'tBTCUSD') -> Optional[pd.DataFrame]:
        """Get detailed institutional order book analysis."""
        try:
            api_logger.info(f"Fetching institutional order book for {symbol}")
            
            book_data = self._make_request(f'/book/{symbol}/P0', {'len': 100})
            
            if not book_data:
                return None
            
            # Process order book
            orders_data = []
            for order in book_data:
                price, count, amount = float(order[0]), int(order[1]), float(order[2])
                side = 'BID' if amount > 0 else 'ASK'
                volume = abs(amount)
                value_usd = price * volume
                
                orders_data.append({
                    'price': price,
                    'count': count,
                    'volume': volume,
                    'value_usd': value_usd,
                    'side': side,
                    'timestamp': datetime.utcnow()
                })
            
            df = pd.DataFrame(orders_data)
            
            # Add analysis
            df['large_order'] = df['value_usd'] > df['value_usd'].quantile(0.8)
            df['cumulative_volume'] = df.groupby('side')['volume'].cumsum()
            
            # Save to file
            filename = f"bitfinex_depth_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.base_dir / filename
            df.to_csv(filepath, index=False)
            api_logger.info(f"Saved order book depth to {filepath}")
            
            return df
            
        except Exception as e:
            api_logger.error(f"Error fetching order book depth: {e}")
            return None

if __name__ == "__main__":
    # Test the Bitfinex API
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    api = BitfinexAPI()
    
    print("🏛️ Testing Bitfinex API...")
    
    # Test trade data
    print("\n📊 Testing institutional trade data...")
    liquidations = api.get_liquidation_data('tBTCUSD', limit=100)
    if liquidations is not None:
        print(f"✅ Got {len(liquidations)} institutional trade records")
        print(liquidations.head())
    
    # Test funding data
    print("\n💰 Testing professional funding data...")
    funding = api.get_funding_data()
    if funding is not None:
        print(f"✅ Got funding data for {len(funding)} currencies")
        print(funding.head())
    
    # Test institutional analysis
    print("\n🏛️ Testing institutional analysis...")
    whales = api.get_whale_addresses()
    if whales is not None:
        print(f"✅ Found {len(whales)} institutional orders")
        print(whales.head())
    
    print("\n✅ Bitfinex API testing completed!")
