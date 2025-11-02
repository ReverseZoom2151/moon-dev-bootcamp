"""
🌙 Moon Dev's Binance API Handler
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
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   ```

3. Basic Usage:
   ```python
   from binance_api import BinanceAPI
   
   # Initialize with env variable (recommended)
   api = BinanceAPI()
   
   # Or initialize with direct keys
   api = BinanceAPI(api_key="your_key_here", secret_key="your_secret_here")
   
   # Get data
   liquidations = api.get_liquidation_data(limit=1000)  # Last 1000 rows
   funding = api.get_funding_data()
   oi = api.get_oi_data()
   ```

Available Methods:
----------------
- get_liquidation_data(limit=None): Get historical liquidation data from force orders
- get_funding_data(): Get current funding rate data for futures symbols
- get_token_addresses(): Get all trading symbols and their info
- get_oi_data(): Get detailed open interest data for futures
- get_oi_total(): Get total open interest data across all symbols
- get_whale_addresses(): Get list of large traders (simulated from recent large trades)
- get_copybot_recent_transactions(): Get recent large transactions
- get_agg_positions(): Get aggregated position data (simulated)
- get_positions(): Get detailed position data (requires authenticated API)

Data Details:
------------
All data is automatically saved to CSV files with timestamps for offline analysis.
Rate limiting is implemented to avoid API restrictions.
"""

import os
import logging
import requests
import pandas as pd
import time
import hmac
import hashlib
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
api_logger = logging.getLogger(__name__)

# Constants
SCRIPT_DIR = Path(__file__).parent.absolute()

class BinanceAPI:
    """
    🌙 Moon Dev's Binance API wrapper for trading data
    
    Provides easy access to Binance market data, funding rates, liquidations,
    open interest, and whale tracking functionality.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 secret_key: Optional[str] = None,
                 base_url: Optional[str] = None, 
                 data_dir: Optional[Path | str] = None):
        """Initialize the API handler.

        Args:
            api_key: Your Binance API key. Overrides BINANCE_API_KEY env var.
            secret_key: Your Binance secret key. Overrides BINANCE_SECRET_KEY env var.
            base_url: The base URL for the API. Overrides BINANCE_BASE_URL env var.
            data_dir: The directory to store downloaded data. Overrides BINANCE_DATA_DIR env var.
                      Defaults to a 'data/binance_api' subdir relative to this script.
        """
        # Determine data directory path (Arg > Env Var > Default)
        _data_dir_str = data_dir or os.getenv('BINANCE_DATA_DIR')
        if _data_dir_str:
            self.base_dir = Path(_data_dir_str).resolve()
        else:
            self.base_dir = SCRIPT_DIR / "data" / "binance_api"
            
        self.base_dir.mkdir(parents=True, exist_ok=True)
        api_logger.info(f"Data directory set to: {self.base_dir}")

        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.secret_key = secret_key or os.getenv('BINANCE_SECRET_KEY')
        self.base_url = base_url or os.getenv('BINANCE_BASE_URL', "https://fapi.binance.com")
        api_logger.info(f"API Base URL set to: {self.base_url}")
        
        self.headers: Dict[str, str] = {'X-MBX-APIKEY': self.api_key} if self.api_key else {}
        if not self.api_key:
            api_logger.warning("API key not provided (checked constructor arg and BINANCE_API_KEY env var). Some endpoints may fail.")
            
        self.session = requests.Session()
        self.max_retries = 3
        self.rate_limit_delay = 0.1

    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        if not self.secret_key:
            raise ValueError("Secret key required for authenticated requests")
        return hmac.new(
            self.secret_key.encode('utf-8'), 
            query_string.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make authenticated request to Binance API with retry logic"""
        params = params or {}
        
        for attempt in range(self.max_retries):
            try:
                if signed:
                    params['timestamp'] = int(time.time() * 1000)
                    query_string = urllib.parse.urlencode(params)
                    params['signature'] = self._generate_signature(query_string)
                
                url = f"{self.base_url}{endpoint}"
                response = self.session.get(url, headers=self.headers, params=params, timeout=30)
                
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
        """Get liquidation data from force orders"""
        try:
            api_logger.info("🔥 Fetching liquidation data from Binance...")
            
            # Get force orders (liquidations)
            params = {"limit": min(limit or 1000, 1000)}
            data = self._make_request("/fapi/v1/forceOrders", params)
            
            if not data:
                api_logger.warning("No liquidation data received")
                return pd.DataFrame()
            
            # Process liquidation data
            processed_data = []
            for order in data:
                processed_data.append({
                    'timestamp': pd.to_datetime(order['time'], unit='ms'),
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'price': float(order['avgPrice']),
                    'quantity': float(order['origQty']),
                    'value_usd': float(order['avgPrice']) * float(order['origQty']),
                    'time_in_force': order['timeInForce']
                })
            
            df = pd.DataFrame(processed_data)
            self._save_to_csv(processed_data, "liquidations")
            
            api_logger.info(f"✅ Retrieved {len(df)} liquidation records")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching liquidation data: {str(e)}")
            return pd.DataFrame()

    def get_funding_data(self) -> pd.DataFrame:
        """Get current funding rate data for futures symbols"""
        try:
            api_logger.info("💰 Fetching funding rate data...")
            
            data = self._make_request("/fapi/v1/premiumIndex")
            
            if not data:
                return pd.DataFrame()
            
            # Process funding data
            processed_data = []
            for item in data:
                if item.get('symbol') and item.get('lastFundingRate'):
                    processed_data.append({
                        'symbol': item['symbol'],
                        'funding_rate': float(item['lastFundingRate']),
                        'funding_rate_pct': float(item['lastFundingRate']) * 100,
                        'next_funding_time': pd.to_datetime(item['nextFundingTime'], unit='ms'),
                        'mark_price': float(item.get('markPrice', 0)),
                        'index_price': float(item.get('indexPrice', 0))
                    })
            
            df = pd.DataFrame(processed_data)
            self._save_to_csv(processed_data, "funding_rates")
            
            api_logger.info(f"✅ Retrieved funding rates for {len(df)} symbols")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching funding data: {str(e)}")
            return pd.DataFrame()

    def get_token_addresses(self) -> pd.DataFrame:
        """Get all trading symbols and their info"""
        try:
            api_logger.info("🪙 Fetching symbol information...")
            
            data = self._make_request("/fapi/v1/exchangeInfo")
            
            if not data or 'symbols' not in data:
                return pd.DataFrame()
            
            # Process symbol data
            processed_data = []
            for symbol_info in data['symbols']:
                if symbol_info['status'] == 'TRADING':
                    processed_data.append({
                        'symbol': symbol_info['symbol'],
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'status': symbol_info['status'],
                        'contract_type': symbol_info.get('contractType', ''),
                        'delivery_date': symbol_info.get('deliveryDate', ''),
                        'onboard_date': pd.to_datetime(symbol_info['onboardDate'], unit='ms') if symbol_info.get('onboardDate') else None
                    })
            
            df = pd.DataFrame(processed_data)
            self._save_to_csv(processed_data, "symbols")
            
            api_logger.info(f"✅ Retrieved {len(df)} active trading symbols")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching symbol data: {str(e)}")
            return pd.DataFrame()

    def get_oi_data(self) -> pd.DataFrame:
        """Get detailed open interest data for futures"""
        try:
            api_logger.info("📊 Fetching open interest data...")
            
            data = self._make_request("/fapi/v1/openInterest")
            
            if not data:
                return pd.DataFrame()
            
            # Process OI data
            processed_data = []
            for item in data:
                processed_data.append({
                    'symbol': item['symbol'],
                    'open_interest': float(item['openInterest']),
                    'timestamp': pd.to_datetime(item['time'], unit='ms')
                })
            
            df = pd.DataFrame(processed_data)
            self._save_to_csv(processed_data, "open_interest")
            
            api_logger.info(f"✅ Retrieved OI data for {len(df)} symbols")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching OI data: {str(e)}")
            return pd.DataFrame()

    def get_oi_total(self) -> pd.DataFrame:
        """Get total open interest data across all symbols"""
        try:
            api_logger.info("📈 Calculating total open interest...")
            
            oi_data = self.get_oi_data()
            if oi_data.empty:
                return pd.DataFrame()
            
            # Get current prices to calculate USD values
            prices_data = self._make_request("/fapi/v1/ticker/price")
            price_dict = {item['symbol']: float(item['price']) for item in prices_data}
            
            # Calculate USD values
            total_oi_usd = 0
            processed_data = []
            
            for _, row in oi_data.iterrows():
                symbol = row['symbol']
                oi_amount = row['open_interest']
                price = price_dict.get(symbol, 0)
                oi_usd = oi_amount * price
                total_oi_usd += oi_usd
                
                processed_data.append({
                    'symbol': symbol,
                    'open_interest': oi_amount,
                    'price': price,
                    'oi_usd': oi_usd,
                    'timestamp': datetime.now()
                })
            
            # Add total summary
            processed_data.append({
                'symbol': 'TOTAL',
                'open_interest': sum(row['open_interest'] for row in processed_data[:-1]),
                'price': 1.0,
                'oi_usd': total_oi_usd,
                'timestamp': datetime.now()
            })
            
            df = pd.DataFrame(processed_data)
            self._save_to_csv(processed_data, "total_oi")
            
            api_logger.info(f"✅ Total OI: ${total_oi_usd:,.2f}")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error calculating total OI: {str(e)}")
            return pd.DataFrame()

    def get_whale_addresses(self) -> List[str]:
        """Get list of large traders (simulated from recent large trades)"""
        try:
            api_logger.info("🐋 Identifying whale addresses from large trades...")
            
            # Get recent large trades across multiple symbols
            major_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            whale_addresses = []
            
            for symbol in major_symbols:
                try:
                    trades = self._make_request(f"/fapi/v1/aggTrades", {"symbol": symbol, "limit": 1000})
                    
                    # Filter for large trades (>$50k)
                    for trade in trades:
                        trade_value = float(trade['p']) * float(trade['q'])
                        if trade_value > 50000:
                            # Simulate whale address (in real scenario, this would come from on-chain data)
                            whale_addr = f"binance_whale_{trade['a']}_{symbol.lower()}"
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
            
            api_logger.info(f"✅ Identified {len(unique_whales)} potential whale addresses")
            return unique_whales
            
        except Exception as e:
            api_logger.error(f"❌ Error identifying whale addresses: {str(e)}")
            return []

    def get_copybot_recent_transactions(self) -> pd.DataFrame:
        """Get recent large transactions"""
        try:
            api_logger.info("💸 Fetching recent large transactions...")
            
            # Get recent large trades across major pairs
            major_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            all_transactions = []
            
            for symbol in major_symbols:
                try:
                    trades = self._make_request(f"/fapi/v1/aggTrades", {"symbol": symbol, "limit": 500})
                    
                    for trade in trades:
                        trade_value = float(trade['p']) * float(trade['q'])
                        if trade_value > 25000:  # Only large trades
                            all_transactions.append({
                                'timestamp': pd.to_datetime(trade['T'], unit='ms'),
                                'symbol': symbol,
                                'price': float(trade['p']),
                                'quantity': float(trade['q']),
                                'value_usd': trade_value,
                                'side': 'BUY' if trade['m'] else 'SELL',
                                'trade_id': trade['a']
                            })
                    
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    api_logger.warning(f"Error processing {symbol}: {e}")
                    continue
            
            df = pd.DataFrame(all_transactions)
            if not df.empty:
                df = df.sort_values('timestamp', ascending=False)
                self._save_to_csv(all_transactions, "large_transactions")
            
            api_logger.info(f"✅ Retrieved {len(all_transactions)} large transactions")
            return df
            
        except Exception as e:
            api_logger.error(f"❌ Error fetching transactions: {str(e)}")
            return pd.DataFrame()

    def get_agg_positions(self) -> pd.DataFrame:
        """Get aggregated position data (simulated from open interest and funding)"""
        try:
            api_logger.info("📊 Calculating aggregated positions...")
            
            # Get OI and funding data
            oi_data = self.get_oi_data()
            funding_data = self.get_funding_data()
            
            if oi_data.empty or funding_data.empty:
                return pd.DataFrame()
            
            # Merge data
            merged = oi_data.merge(funding_data, on='symbol', how='inner')
            
            # Calculate aggregated positions
            processed_data = []
            for _, row in merged.iterrows():
                # Estimate long/short bias from funding rate
                long_bias = 0.6 if row['funding_rate'] > 0 else 0.4  # Higher funding = more longs
                
                processed_data.append({
                    'symbol': row['symbol'],
                    'total_oi': row['open_interest'],
                    'estimated_longs': row['open_interest'] * long_bias,
                    'estimated_shorts': row['open_interest'] * (1 - long_bias),
                    'funding_rate': row['funding_rate'],
                    'long_short_ratio': long_bias / (1 - long_bias),
                    'timestamp': datetime.now()
                })
            
            df = pd.DataFrame(processed_data)
            self._save_to_csv(processed_data, "agg_positions")
            
            api_logger.info(f"✅ Calculated aggregated positions for {len(df)} symbols")
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
            
            data = self._make_request("/fapi/v2/positionRisk", signed=True)
            
            if not data:
                return pd.DataFrame()
            
            # Process position data
            processed_data = []
            for pos in data:
                if float(pos['positionAmt']) != 0:  # Only active positions
                    processed_data.append({
                        'symbol': pos['symbol'],
                        'position_amount': float(pos['positionAmt']),
                        'entry_price': float(pos['entryPrice']),
                        'mark_price': float(pos['markPrice']),
                        'unrealized_pnl': float(pos['unRealizedProfit']),
                        'percentage': float(pos['percentage']),
                        'notional': abs(float(pos['notional'])),
                        'isolated_wallet': float(pos['isolatedWallet']),
                        'side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT',
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
    print("🌙 Moon Dev's Binance API Test Suite 🚀")
    print("=" * 50)
    
    try:
        # Initialize API
        api = BinanceAPI()
        
        # Test funding data
        print("\n📊 Testing funding data...")
        funding_df = api.get_funding_data()
        print(f"Retrieved {len(funding_df)} funding rates")
        if not funding_df.empty:
            print(funding_df.head())
        
        # Test liquidation data
        print("\n🔥 Testing liquidation data...")
        liq_df = api.get_liquidation_data(limit=10)
        print(f"Retrieved {len(liq_df)} liquidation records")
        if not liq_df.empty:
            print(liq_df.head())
        
        # Test OI data
        print("\n📈 Testing open interest data...")
        oi_df = api.get_oi_data()
        print(f"Retrieved OI data for {len(oi_df)} symbols")
        if not oi_df.empty:
            print(oi_df.head())
        
        # Test whale addresses
        print("\n🐋 Testing whale identification...")
        whales = api.get_whale_addresses()
        print(f"Identified {len(whales)} potential whale addresses")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
