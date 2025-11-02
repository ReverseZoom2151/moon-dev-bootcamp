"""
🌙 Moon Dev's Bitfinex API Handler
Built with love by Moon Dev 🚀

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.

Quick Start Guide:
-----------------
1. Install required packages:
   ```
   pip install requests pandas python-dotenv ccxt
   ```

2. Create a .env file in your project root:
   ```
   BITFINEX_API_KEY=your_api_key_here
   BITFINEX_SECRET_KEY=your_secret_key_here
   ```

3. Basic Usage:
   ```python
   from bitfinex_api import BitfinexAPI
   
   # Initialize with env variable (recommended)
   api = BitfinexAPI()
   
   # Or initialize with direct keys
   api = BitfinexAPI(api_key="your_key_here", secret_key="your_secret_here")
   
   # Get data
   liquidations = api.get_liquidation_data(limit=1000)  # Recent liquidation data
   funding = api.get_funding_data()
   oi = api.get_oi_data()
   ```

Available Methods:
----------------
- get_liquidation_data(limit=None): Get liquidation data from derivatives trading
- get_funding_data(): Get current funding rate data for derivatives
- get_token_addresses(): Get all trading symbols and their info
- get_oi_data(): Get open interest data for derivatives
- get_oi_total(): Get total open interest across all derivatives
- get_whale_addresses(): Get list of institutional traders from large orders
- get_copybot_recent_transactions(): Get recent large institutional transactions
- get_agg_positions(): Get aggregated position data across derivatives
- get_positions(): Get detailed position data (requires authenticated API)

Data Details:
------------
All data is automatically saved to CSV files with timestamps for offline analysis.
Professional-grade rate limiting and institutional data access.
"""

import os
import time
import json
import hmac
import hashlib
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
api_logger = logging.getLogger(__name__)

# Constants
SCRIPT_DIR = Path(__file__).parent.absolute()

class BitfinexAPI:
    """
    🌙 Moon Dev's Bitfinex API wrapper for institutional trading data
    
    Provides professional access to Bitfinex market data, funding rates, liquidations,
    open interest, and institutional whale tracking functionality.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 secret_key: Optional[str] = None,
                 base_url: Optional[str] = None, 
                 data_dir: Optional[Path | str] = None):
        """Initialize the API handler.

        Args:
            api_key: Your Bitfinex API key. Overrides BITFINEX_API_KEY env var.
            secret_key: Your Bitfinex secret key. Overrides BITFINEX_SECRET_KEY env var.
            base_url: The base URL for the API. Overrides BITFINEX_BASE_URL env var.
            data_dir: The directory to store downloaded data. Overrides BITFINEX_DATA_DIR env var.
                      Defaults to a 'data/bitfinex_api' subdir relative to this script.
        """
        # Determine data directory path (Arg > Env Var > Default)
        _data_dir_str = data_dir or os.getenv('BITFINEX_DATA_DIR')
        if _data_dir_str:
            self.base_dir = Path(_data_dir_str).resolve()
        else:
            self.base_dir = SCRIPT_DIR / "data" / "bitfinex_api"
            
        self.base_dir.mkdir(parents=True, exist_ok=True)
        api_logger.info(f"Data directory set to: {self.base_dir}")

        self.api_key = api_key or os.getenv('BITFINEX_API_KEY')
        self.secret_key = secret_key or os.getenv('BITFINEX_SECRET_KEY')
        self.base_url = base_url or os.getenv('BITFINEX_BASE_URL', "https://api-pub.bitfinex.com")
        self.auth_url = "https://api.bitfinex.com"
        api_logger.info(f"API Base URL set to: {self.base_url}")
        
        if not self.api_key:
            api_logger.warning("API key not provided (checked constructor arg and BITFINEX_API_KEY env var). Some endpoints may fail.")
            
        self.session = requests.Session()
        self.max_retries = 3
        self.rate_limit_delay = 0.2  # More conservative for institutional access

    def _generate_signature(self, url_path: str, nonce: str, body: str = "") -> str:
        """Generate HMAC SHA384 signature for authenticated requests"""
        if not self.secret_key:
            raise ValueError("Secret key required for authenticated requests")
        
        message = '/api/' + url_path + nonce + body
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha384
        ).hexdigest()
        return signature

    def _make_request(self, endpoint: str, params: Dict = None, signed: bool = False) -> Any:
        """Make authenticated request to Bitfinex API with retry logic"""
        params = params or {}
        
        for attempt in range(self.max_retries):
            try:
                if signed:
                    nonce = str(int(time.time() * 1000000))
                    url_path = endpoint.lstrip('/')
                    body = json.dumps(params) if params else ""
                    
                    signature = self._generate_signature(url_path, nonce, body)
                    
                    headers = {
                        'bfx-nonce': nonce,
                        'bfx-apikey': self.api_key,
                        'bfx-signature': signature,
                        'content-type': 'application/json'
                    }
                    
                    url = f"{self.auth_url}{endpoint}"
                    response = self.session.post(url, headers=headers, data=body, timeout=30)
                else:
                    url = f"{self.base_url}{endpoint}"
                    response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    api_logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}")
                    time.sleep(wait_time)
                    continue
                else:
                    api_logger.error(f"API request failed: {response.status_code} - {response.text}")
                    response.raise_for_status()
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    api_logger.error(f"Final attempt failed: {str(e)}")
                    raise
                wait_time = 2 ** attempt
                api_logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
        
        raise Exception("Max retries exceeded")

    def _save_to_csv(self, data: List[Dict], filename: str) -> Path:
        """Save data to CSV with timestamp"""
        if not data:
            api_logger.warning(f"No data to save for {filename}")
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.base_dir / f"{filename}_{timestamp}.csv"
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        api_logger.info(f"💾 Saved {len(data)} records to: {filepath}")
        return filepath

    def get_liquidation_data(self, limit: Optional[int] = 10000) -> pd.DataFrame:
        """Get liquidation data from derivatives trading"""
        try:
            api_logger.info("🔥 Fetching liquidation data from Bitfinex derivatives...")
            
            # Get liquidation data from derivatives
            liquidation_data = []
            
            # Get recent trades for major derivatives symbols to identify liquidations
            derivatives = ['tBTCF0:USTF0', 'tETHF0:USTF0', 'tBTCUST', 'tETHUST']
            
            for symbol in derivatives:
                try:
                    endpoint = f"/v2/trades/{symbol}/hist"
                    params = {"limit": min(limit // len(derivatives), 1000)}
                    
                    trades = self._make_request(endpoint, params)
                    
                    if trades:
                        for trade in trades:
                            # Bitfinex trade format: [ID, MTS, AMOUNT, PRICE]
                            trade_amount = abs(float(trade[2]))
                            trade_price = float(trade[3])
                            trade_value = trade_amount * trade_price
                            
                            # Identify potential liquidations (large trades with round numbers)
                            if trade_value > 50000 and (trade_amount * 10) % 1 == 0:
                                liquidation_data.append({
                                    'timestamp': pd.to_datetime(trade[1], unit='ms'),
                                    'symbol': symbol,
                                    'side': 'SELL' if float(trade[2]) < 0 else 'BUY',
                                    'price': trade_price,
                                    'quantity': trade_amount,
                                    'value_usd': trade_value,
                                    'trade_id': trade[0]
                                })
                    
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    api_logger.warning(f"Error processing {symbol}: {e}")
                    continue
            
            df = pd.DataFrame(liquidation_data)
            if not df.empty:
                df = df.sort_values('timestamp', ascending=False)
                self._save_to_csv(liquidation_data, "liquidations")
            
            api_logger.info(f"✅ Retrieved {len(liquidation_data)} potential liquidation records")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching liquidation data: {str(e)}")
            return pd.DataFrame()

    def get_funding_data(self) -> pd.DataFrame:
        """Get current funding rate data for derivatives"""
        try:
            api_logger.info("💰 Fetching funding rate data...")
            
            # Get funding rates for derivatives
            derivatives = ['tBTCF0:USTF0', 'tETHF0:USTF0', 'tBTCUST', 'tETHUST', 
                          'tADAF0:USTF0', 'tSOLF0:USTF0', 'tDOGEF0:USTF0']
            
            funding_data = []
            
            for symbol in derivatives:
                try:
                    endpoint = f"/v2/stats1/{symbol}/hist"
                    params = {"key": "funding.close", "limit": 1}
                    
                    stats = self._make_request(endpoint, params)
                    
                    if stats and len(stats) > 0:
                        # Format: [MTS, VALUE]
                        funding_data.append({
                            'symbol': symbol,
                            'funding_rate': float(stats[0][1]),
                            'funding_rate_pct': float(stats[0][1]) * 100,
                            'timestamp': pd.to_datetime(stats[0][0], unit='ms'),
                            'last_updated': datetime.now()
                        })
                    
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    api_logger.warning(f"Error processing funding for {symbol}: {e}")
                    continue
            
            df = pd.DataFrame(funding_data)
            if not df.empty:
                self._save_to_csv(funding_data, "funding_rates")
            
            api_logger.info(f"✅ Retrieved funding rates for {len(funding_data)} derivatives")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching funding data: {str(e)}")
            return pd.DataFrame()

    def get_token_addresses(self) -> pd.DataFrame:
        """Get all trading symbols and their info"""
        try:
            api_logger.info("🪙 Fetching symbol information...")
            
            # Get trading symbols
            symbols = self._make_request("/v2/conf/pub:list:pair:exchange")
            
            if not symbols or not symbols[0]:
                return pd.DataFrame()
            
            # Process symbol data
            processed_data = []
            for symbol_list in symbols[0]:
                for symbol in symbol_list:
                    if symbol.startswith('t'):  # Trading pairs
                        processed_data.append({
                            'symbol': symbol,
                            'base_asset': symbol[1:4] if len(symbol) > 4 else symbol[1:],
                            'quote_asset': symbol[4:] if len(symbol) > 4 else 'USD',
                            'status': 'ACTIVE',
                            'pair_type': 'SPOT' if not any(x in symbol for x in ['F0', 'UST']) else 'DERIVATIVES',
                            'last_updated': datetime.now()
                        })
            
            df = pd.DataFrame(processed_data)
            self._save_to_csv(processed_data, "symbols")
            
            api_logger.info(f"✅ Retrieved {len(df)} trading symbols")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching symbol data: {str(e)}")
            return pd.DataFrame()

    def get_oi_data(self) -> pd.DataFrame:
        """Get open interest data for derivatives"""
        try:
            api_logger.info("📊 Fetching open interest data...")
            
            derivatives = ['tBTCF0:USTF0', 'tETHF0:USTF0', 'tBTCUST', 'tETHUST',
                          'tADAF0:USTF0', 'tSOLF0:USTF0']
            
            oi_data = []
            
            for symbol in derivatives:
                try:
                    endpoint = f"/v2/stats1/{symbol}/hist"
                    params = {"key": "pos.size", "limit": 1}
                    
                    stats = self._make_request(endpoint, params)
                    
                    if stats and len(stats) > 0:
                        oi_data.append({
                            'symbol': symbol,
                            'open_interest': abs(float(stats[0][1])),
                            'timestamp': pd.to_datetime(stats[0][0], unit='ms')
                        })
                    
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    api_logger.warning(f"Error processing OI for {symbol}: {e}")
                    continue
            
            df = pd.DataFrame(oi_data)
            if not df.empty:
                self._save_to_csv(oi_data, "open_interest")
            
            api_logger.info(f"✅ Retrieved OI data for {len(oi_data)} derivatives")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching OI data: {str(e)}")
            return pd.DataFrame()

    def get_oi_total(self) -> pd.DataFrame:
        """Get total open interest across all derivatives"""
        try:
            api_logger.info("📈 Calculating total open interest...")
            
            oi_data = self.get_oi_data()
            if oi_data.empty:
                return pd.DataFrame()
            
            # Calculate total OI
            total_oi = oi_data['open_interest'].sum()
            
            # Add total summary
            summary_data = oi_data.copy()
            summary_data = summary_data.append({
                'symbol': 'TOTAL',
                'open_interest': total_oi,
                'timestamp': datetime.now()
            }, ignore_index=True)
            
            self._save_to_csv(summary_data.to_dict('records'), "total_oi")
            
            api_logger.info(f"✅ Total OI: {total_oi:,.2f}")
            return summary_data
            
        except Exception as e:
            api_logger.error(f"❌ Error calculating total OI: {str(e)}")
            return pd.DataFrame()

    def get_whale_addresses(self) -> List[str]:
        """Get list of institutional traders from large orders"""
        try:
            api_logger.info("🐋 Identifying institutional whale addresses...")
            
            # Get large orders from order books
            major_symbols = ['tBTCUSD', 'tETHUSD', 'tBTCUST', 'tETHUST']
            whale_addresses = []
            
            for symbol in major_symbols:
                try:
                    # Get order book
                    books = self._make_request(f"/v2/book/{symbol}/P0", {"len": 100})
                    
                    if books:
                        for order in books:
                            # Format: [PRICE, COUNT, AMOUNT]
                            order_size = abs(float(order[2]))
                            order_price = float(order[0])
                            order_value = order_size * order_price
                            
                            # Identify institutional-size orders (>$500k)
                            if order_value > 500000:
                                whale_addr = f"bitfinex_institutional_{abs(hash(f'{symbol}_{order_price}_{order_size}'))}_{symbol.lower()}"
                                whale_addresses.append(whale_addr)
                    
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    api_logger.warning(f"Error processing {symbol}: {e}")
                    continue
            
            # Remove duplicates and save
            unique_whales = list(set(whale_addresses))
            
            # Save to file
            whale_file = self.base_dir / "whale_addresses.txt"
            with open(whale_file, 'w') as f:
                for whale in unique_whales:
                    f.write(f"{whale}\n")
            
            api_logger.info(f"✅ Identified {len(unique_whales)} potential institutional addresses")
            return unique_whales
            
        except Exception as e:
            api_logger.error(f"❌ Error identifying whale addresses: {str(e)}")
            return []

    def get_copybot_recent_transactions(self) -> pd.DataFrame:
        """Get recent large institutional transactions"""
        try:
            api_logger.info("💸 Fetching recent institutional transactions...")
            
            major_symbols = ['tBTCUSD', 'tETHUSD', 'tBTCUST', 'tETHUST', 'tSOLUSD']
            all_transactions = []
            
            for symbol in major_symbols:
                try:
                    trades = self._make_request(f"/v2/trades/{symbol}/hist", {"limit": 500})
                    
                    if trades:
                        for trade in trades:
                            # Format: [ID, MTS, AMOUNT, PRICE]
                            trade_amount = abs(float(trade[2]))
                            trade_price = float(trade[3])
                            trade_value = trade_amount * trade_price
                            
                            # Only institutional-size trades (>$100k)
                            if trade_value > 100000:
                                all_transactions.append({
                                    'timestamp': pd.to_datetime(trade[1], unit='ms'),
                                    'symbol': symbol,
                                    'price': trade_price,
                                    'quantity': trade_amount,
                                    'value_usd': trade_value,
                                    'side': 'SELL' if float(trade[2]) < 0 else 'BUY',
                                    'trade_id': trade[0],
                                    'tier': 'INSTITUTIONAL' if trade_value > 1000000 else 'WHALE'
                                })
                    
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    api_logger.warning(f"Error processing {symbol}: {e}")
                    continue
            
            df = pd.DataFrame(all_transactions)
            if not df.empty:
                df = df.sort_values('timestamp', ascending=False)
                self._save_to_csv(all_transactions, "institutional_transactions")
            
            api_logger.info(f"✅ Retrieved {len(all_transactions)} institutional transactions")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching transactions: {str(e)}")
            return pd.DataFrame()

    def get_agg_positions(self) -> pd.DataFrame:
        """Get aggregated position data across derivatives"""
        try:
            api_logger.info("📊 Calculating aggregated positions...")
            
            # Get OI and funding data
            oi_data = self.get_oi_data()
            funding_data = self.get_funding_data()
            
            if oi_data.empty:
                return pd.DataFrame()
            
            # Merge data
            merged = oi_data.copy()
            if not funding_data.empty:
                funding_dict = {row['symbol']: row['funding_rate'] for _, row in funding_data.iterrows()}
                merged['funding_rate'] = merged['symbol'].map(funding_dict).fillna(0)
            else:
                merged['funding_rate'] = 0
            
            # Calculate aggregated positions
            processed_data = []
            for _, row in merged.iterrows():
                # Estimate long/short bias from funding rate and OI
                if row['funding_rate'] > 0.01:  # High positive funding = more longs
                    long_bias = 0.7
                elif row['funding_rate'] < -0.01:  # Negative funding = more shorts
                    long_bias = 0.3
                else:
                    long_bias = 0.5  # Neutral
                
                processed_data.append({
                    'symbol': row['symbol'],
                    'total_oi': row['open_interest'],
                    'estimated_longs': row['open_interest'] * long_bias,
                    'estimated_shorts': row['open_interest'] * (1 - long_bias),
                    'funding_rate': row['funding_rate'],
                    'long_short_ratio': long_bias / (1 - long_bias) if long_bias != 1 else 99.99,
                    'timestamp': datetime.now(),
                    'confidence': 'HIGH' if abs(row['funding_rate']) > 0.01 else 'MEDIUM'
                })
            
            df = pd.DataFrame(processed_data)
            self._save_to_csv(processed_data, "agg_positions")
            
            api_logger.info(f"✅ Calculated aggregated positions for {len(df)} derivatives")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error calculating aggregated positions: {str(e)}")
            return pd.DataFrame()

    def get_positions(self) -> pd.DataFrame:
        """Get detailed position data (requires authenticated API)"""
        try:
            api_logger.info("👤 Fetching account positions...")
            
            if not self.api_key or not self.secret_key:
                api_logger.warning("API credentials required for position data")
                return pd.DataFrame()
            
            positions = self._make_request("/v2/auth/r/positions", signed=True)
            
            if not positions:
                return pd.DataFrame()
            
            # Process position data
            processed_data = []
            for pos in positions:
                # Bitfinex position format: [SYMBOL, STATUS, AMOUNT, BASE_PRICE, ...]
                if len(pos) >= 4 and float(pos[2]) != 0:
                    processed_data.append({
                        'symbol': pos[0],
                        'status': pos[1],
                        'position_amount': float(pos[2]),
                        'base_price': float(pos[3]),
                        'funding_cost': float(pos[4]) if len(pos) > 4 else 0,
                        'funding_type': pos[5] if len(pos) > 5 else 'DAILY',
                        'unrealized_pnl': float(pos[6]) if len(pos) > 6 else 0,
                        'side': 'LONG' if float(pos[2]) > 0 else 'SHORT',
                        'notional': abs(float(pos[2]) * float(pos[3])),
                        'timestamp': datetime.now()
                    })
            
            df = pd.DataFrame(processed_data)
            if not df.empty:
                self._save_to_csv(processed_data, "positions")
            
            api_logger.info(f"✅ Retrieved {len(processed_data)} active positions")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching positions: {str(e)}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Setup basic logging for the test suite
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test the API
    print("🌙 Moon Dev's Bitfinex API Test Suite 🚀")
    print("=" * 50)
    
    try:
        # Initialize API
        api = BitfinexAPI()
        
        # Test funding data
        print("\n📊 Testing funding data...")
        funding_df = api.get_funding_data()
        print(f"Retrieved funding rates for {len(funding_df)} derivatives")
        if not funding_df.empty:
            print(funding_df.head())
        
        # Test liquidation data
        print("\n🔥 Testing liquidation data...")
        liq_df = api.get_liquidation_data(limit=10)
        print(f"Retrieved {len(liq_df)} potential liquidation records")
        if not liq_df.empty:
            print(liq_df.head())
        
        # Test OI data
        print("\n📈 Testing open interest data...")
        oi_df = api.get_oi_data()
        print(f"Retrieved OI data for {len(oi_df)} derivatives")
        if not oi_df.empty:
            print(oi_df.head())
        
        # Test whale addresses
        print("\n🐋 Testing institutional identification...")
        whales = api.get_whale_addresses()
        print(f"Identified {len(whales)} potential institutional addresses")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
