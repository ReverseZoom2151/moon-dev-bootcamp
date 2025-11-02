# binance_rrs_data_fetcher.py
import requests
import pandas as pd
import logging
import time
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from binance_rrs_config import BINANCE_API_KEY, BINANCE_SECRET_KEY, REQUEST_DELAY, TIMEFRAME_MAPPING

# Setup logger for this module
logger = logging.getLogger(__name__)

class BinanceDataFetcher:
    """Handles data fetching from Binance API for RRS analysis."""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.api_key = BINANCE_API_KEY
        self.secret_key = BINANCE_SECRET_KEY
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': self.api_key})
    
    def _create_signature(self, query_string: str) -> str:
        """Create HMAC SHA256 signature for authenticated requests."""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def get_server_time(self) -> int:
        """Get Binance server time for synchronization."""
        try:
            response = self.session.get(f"{self.base_url}/api/v3/time", timeout=10)
            response.raise_for_status()
            return response.json()['serverTime']
        except Exception as e:
            logger.warning(f"Failed to get server time, using local time: {e}")
            return int(time.time() * 1000)
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules."""
        try:
            response = self.session.get(f"{self.base_url}/api/v3/exchangeInfo", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get exchange info: {e}")
            return {}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is actively trading on Binance."""
        try:
            exchange_info = self.get_exchange_info()
            for symbol_info in exchange_info.get('symbols', []):
                if (symbol_info['symbol'] == symbol.upper() and 
                    symbol_info['status'] == 'TRADING'):
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to validate symbol {symbol}: {e}")
            return False

def fetch_data(symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Fetches OHLCV data from Binance API for a given symbol and time range.

    Args:
        symbol: The trading pair symbol (e.g., 'BTCUSDT').
        timeframe: The timeframe string (e.g., '1m', '5m', '1h', '1d').
        start_time: The start datetime object (UTC).
        end_time: The end datetime object (UTC).

    Returns:
        A pandas DataFrame containing the OHLCV data, or an empty DataFrame if fetching fails.
    """
    fetcher = BinanceDataFetcher()
    
    # Validate symbol
    if not fetcher.validate_symbol(symbol):
        logger.error(f"Symbol {symbol} is not valid or not trading")
        return pd.DataFrame()
    
    # Map timeframe to Binance interval
    interval = TIMEFRAME_MAPPING.get(timeframe)
    if not interval:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return pd.DataFrame()
    
    # Convert times to milliseconds
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)
    
    logger.info(f"Fetching Binance data for {symbol} [{timeframe}]")
    logger.debug(f"Time range (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Binance API endpoint for klines (candlestick data)
        url = f"{fetcher.base_url}/api/v3/klines"
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'startTime': start_time_ms,
            'endTime': end_time_ms,
            'limit': 1000  # Maximum allowed by Binance
        }
        
        all_data = []
        current_start = start_time_ms
        
        # Handle data pagination if needed
        while current_start < end_time_ms:
            params['startTime'] = current_start
            
            logger.debug(f"Requesting data from {datetime.fromtimestamp(current_start/1000)}")
            
            response = fetcher.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.info(f"No more data available for {symbol}")
                break
            
            all_data.extend(data)
            
            # Update start time for next batch
            current_start = data[-1][6] + 1  # Close time + 1ms
            
            # Rate limiting
            time.sleep(REQUEST_DELAY)
            
            # Break if we have enough data or hit API limits
            if len(all_data) >= 10000:  # Reasonable limit
                logger.info(f"Reached data limit for {symbol}")
                break
        
        if not all_data:
            logger.warning(f"No data retrieved for {symbol} in the specified time range")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['close_timestamp'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        
        # Drop unnecessary columns
        df = df.drop(['open_time', 'close_time', 'ignore', 'taker_buy_base_asset_volume', 
                     'taker_buy_quote_asset_volume'], axis=1)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove any duplicate timestamps
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
        logger.debug(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching data for {symbol}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        logger.error(f"Data parsing error for {symbol}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def get_24hr_ticker_stats(symbol: str) -> Dict:
    """Get 24hr ticker statistics for additional market insights."""
    fetcher = BinanceDataFetcher()
    
    try:
        url = f"{fetcher.base_url}/api/v3/ticker/24hr"
        params = {'symbol': symbol.upper()}
        
        response = fetcher.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'symbol': data['symbol'],
            'price_change': float(data['priceChange']),
            'price_change_percent': float(data['priceChangePercent']),
            'weighted_avg_price': float(data['weightedAvgPrice']),
            'prev_close_price': float(data['prevClosePrice']),
            'last_price': float(data['lastPrice']),
            'bid_price': float(data['bidPrice']),
            'ask_price': float(data['askPrice']),
            'open_price': float(data['openPrice']),
            'high_price': float(data['highPrice']),
            'low_price': float(data['lowPrice']),
            'volume': float(data['volume']),
            'quote_volume': float(data['quoteVolume']),
            'open_time': pd.to_datetime(int(data['openTime']), unit='ms', utc=True),
            'close_time': pd.to_datetime(int(data['closeTime']), unit='ms', utc=True),
            'count': int(data['count'])
        }
    except Exception as e:
        logger.error(f"Failed to get 24hr stats for {symbol}: {e}")
        return {}

def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol."""
    fetcher = BinanceDataFetcher()
    
    try:
        url = f"{fetcher.base_url}/api/v3/ticker/price"
        params = {'symbol': symbol.upper()}
        
        response = fetcher.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return float(data['price'])
    except Exception as e:
        logger.error(f"Failed to get current price for {symbol}: {e}")
        return None

def get_top_volume_symbols(limit: int = 50) -> List[str]:
    """Get top volume symbols for dynamic symbol discovery."""
    fetcher = BinanceDataFetcher()
    
    try:
        url = f"{fetcher.base_url}/api/v3/ticker/24hr"
        
        response = fetcher.session.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Filter USDT pairs and sort by quote volume
        usdt_pairs = [
            item for item in data 
            if item['symbol'].endswith('USDT') and float(item['quoteVolume']) > 0
        ]
        
        # Sort by quote volume (descending)
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        
        # Return top symbols
        return [item['symbol'] for item in usdt_pairs[:limit]]
        
    except Exception as e:
        logger.error(f"Failed to get top volume symbols: {e}")
        return []

if __name__ == "__main__":
    # Test the data fetcher
    logging.basicConfig(level=logging.INFO)
    
    # Test fetching BTC data for last 7 days
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    print("Testing Binance data fetcher...")
    
    # Test fetch data
    btc_data = fetch_data('BTCUSDT', '1h', start_time, end_time)
    print(f"BTC data points: {len(btc_data)}")
    
    if not btc_data.empty:
        print(btc_data.head())
        print(f"Date range: {btc_data['timestamp'].min()} to {btc_data['timestamp'].max()}")
    
    # Test 24hr stats
    btc_stats = get_24hr_ticker_stats('BTCUSDT')
    print(f"BTC 24hr stats: {btc_stats}")
    
    # Test current price
    btc_price = get_current_price('BTCUSDT')
    print(f"BTC current price: {btc_price}")
    
    # Test top volume symbols
    top_symbols = get_top_volume_symbols(10)
    print(f"Top 10 volume symbols: {top_symbols}")
    
    print("✅ Binance data fetcher test complete")
