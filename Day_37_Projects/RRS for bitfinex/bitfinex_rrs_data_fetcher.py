# bitfinex_rrs_data_fetcher.py
import requests
import pandas as pd
import logging
import time
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from bitfinex_rrs_config import (BITFINEX_API_KEY, BITFINEX_SECRET_KEY, REQUEST_DELAY, TIMEFRAME_MAPPING, MIN_VOLUME_THRESHOLD)

# Setup logger for this module
logger = logging.getLogger(__name__)

class BitfinexProfessionalDataFetcher:
    """Handles professional data fetching from Bitfinex API for RRS analysis."""
    
    def __init__(self):
        self.base_url = "https://api.bitfinex.com"
        self.api_key = BITFINEX_API_KEY
        self.secret_key = BITFINEX_SECRET_KEY
        self.session = requests.Session()
        
    def _get_nonce(self) -> str:
        """Generate nonce for authenticated requests."""
        return str(int(time.time() * 1000000))
    
    def _create_auth_headers(self, path: str, nonce: str, body: str = "") -> Dict[str, str]:
        """Create professional authentication headers for Bitfinex API."""
        message = f"/api/{path}{nonce}{body}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha384
        ).hexdigest()
        
        return {
            'X-BFX-APIKEY': self.api_key,
            'X-BFX-PAYLOAD': base64.b64encode(body.encode('utf-8')).decode('utf-8'),
            'X-BFX-SIGNATURE': signature,
            'Content-Type': 'application/json'
        }
    
    def get_symbol_details(self, symbol: str) -> Optional[Dict]:
        """Get professional trading details for a symbol."""
        try:
            symbol_formatted = f"t{symbol.upper()}"
            response = self.session.get(f"{self.base_url}/v2/conf/pub:info:pair", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                symbols_info = data[0]
                
                for pair_info in symbols_info:
                    if isinstance(pair_info, list) and len(pair_info) > 0:
                        if pair_info[0] == symbol_formatted:
                            return {
                                'symbol': pair_info[0],
                                'min_order_size': pair_info[3] if len(pair_info) > 3 else 0.001,
                                'max_order_size': pair_info[4] if len(pair_info) > 4 else 1000000,
                                'price_precision': pair_info[5] if len(pair_info) > 5 else 5
                            }
            return None
        except Exception as e:
            logger.error(f"Failed to get symbol details for {symbol}: {e}")
            return None
    
    def get_professional_ticker(self, symbol: str) -> Optional[Dict]:
        """Get professional ticker data with enhanced metrics."""
        try:
            symbol_formatted = f"t{symbol.upper()}"
            response = self.session.get(f"{self.base_url}/v2/ticker/{symbol_formatted}", timeout=10)
            response.raise_for_status()
            
            ticker_data = response.json()
            if isinstance(ticker_data, list) and len(ticker_data) >= 10:
                return {
                    'symbol': symbol,
                    'bid': float(ticker_data[0]),
                    'bid_size': float(ticker_data[1]),
                    'ask': float(ticker_data[2]),
                    'ask_size': float(ticker_data[3]),
                    'daily_change': float(ticker_data[4]),
                    'daily_change_perc': float(ticker_data[5]),
                    'last_price': float(ticker_data[6]),
                    'volume': float(ticker_data[7]),
                    'high': float(ticker_data[8]),
                    'low': float(ticker_data[9])
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get professional ticker for {symbol}: {e}")
            return None

def fetch_data(symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Fetches professional OHLCV data from Bitfinex API for a given symbol and time range.

    Args:
        symbol: The trading pair symbol (e.g., 'btcusd').
        timeframe: The timeframe string (e.g., '1m', '5m', '1h', '1d').
        start_time: The start datetime object (UTC).
        end_time: The end datetime object (UTC).

    Returns:
        A pandas DataFrame containing professional OHLCV data, or empty DataFrame if fails.
    """
    fetcher = BitfinexProfessionalDataFetcher()
    
    # Validate and format symbol
    symbol = symbol.lower()
    symbol_formatted = f"t{symbol.upper()}"
    
    # Map timeframe to Bitfinex interval
    interval = TIMEFRAME_MAPPING.get(timeframe)
    if not interval:
        logger.error(f"Unsupported professional timeframe: {timeframe}")
        return pd.DataFrame()
    
    # Convert times to milliseconds
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)
    
    logger.info(f"Fetching Bitfinex professional data for {symbol_formatted} [{timeframe}]")
    logger.debug(f"Time range (UTC): {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Bitfinex professional candles endpoint
        url = f"{fetcher.base_url}/v2/candles/trade:{interval}:{symbol_formatted}/hist"
        
        all_data = []
        current_end = end_time_ms
        current_start = start_time_ms
        
        # Handle data pagination (Bitfinex limits to 10,000 candles per request)
        while current_start < current_end:
            params = {
                'start': current_start,
                'end': min(current_end, current_start + (10000 * _get_interval_ms(interval))),
                'sort': 1,  # Sort ascending by timestamp
                'limit': 10000
            }
            
            logger.debug(f"Professional request: {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(params['end']/1000)}")
            
            response = fetcher.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.info(f"No more professional data available for {symbol_formatted}")
                break
            
            # Bitfinex returns [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME] format
            formatted_data = []
            for candle in data:
                if len(candle) >= 6:
                    formatted_data.append({
                        'timestamp': candle[0],  # MTS (millisecond timestamp)
                        'open': float(candle[1]),
                        'close': float(candle[2]), 
                        'high': float(candle[3]),
                        'low': float(candle[4]),
                        'volume': float(candle[5])
                    })
            
            all_data.extend(formatted_data)
            
            # Update for next batch
            if data:
                current_start = max([candle[0] for candle in data]) + 1
            else:
                break
            
            # Professional rate limiting
            time.sleep(REQUEST_DELAY)
            
            # Safety break to avoid infinite loops
            if len(all_data) >= 50000:  # Professional data limit
                logger.info(f"Reached professional data limit for {symbol_formatted}")
                break
        
        if not all_data:
            logger.warning(f"No professional data retrieved for {symbol_formatted}")
            return pd.DataFrame()
        
        # Convert to professional DataFrame
        df = pd.DataFrame(all_data)
        
        # Convert timestamps to datetime (Bitfinex uses milliseconds)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Sort by timestamp and remove duplicates
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Professional data validation
        df = df[df['volume'] > 0]  # Remove zero volume candles
        df = df.dropna()  # Remove any NaN values
        
        # Filter by minimum volume threshold for professional analysis
        daily_volume_usd = df['volume'] * df['close']
        if timeframe in ['1d', '1w'] and daily_volume_usd.mean() < MIN_VOLUME_THRESHOLD:
            logger.warning(f"Professional volume threshold not met for {symbol_formatted}: {daily_volume_usd.mean():.0f}")
        
        logger.info(f"Successfully fetched {len(df)} professional data points for {symbol_formatted}")
        logger.debug(f"Professional data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Add professional metadata
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['data_source'] = 'bitfinex_professional'
        df['fetch_timestamp'] = pd.Timestamp.utcnow()
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Professional network error for {symbol_formatted}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        logger.error(f"Professional data parsing error for {symbol_formatted}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected professional error for {symbol_formatted}: {e}")
        return pd.DataFrame()

def _get_interval_ms(interval: str) -> int:
    """Convert timeframe to milliseconds for pagination calculations."""
    interval_map = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
        '1D': 24 * 60 * 60 * 1000,
        '7D': 7 * 24 * 60 * 60 * 1000
    }
    return interval_map.get(interval, 60 * 60 * 1000)  # Default to 1 hour

def get_professional_funding_rates(symbol: str) -> Optional[Dict]:
    """Get professional funding rate data for margin analysis."""
    fetcher = BitfinexProfessionalDataFetcher()
    
    try:
        # Funding rates are only available for specific symbols
        funding_symbol = f"f{symbol.upper()}"
        url = f"{fetcher.base_url}/v2/stats1/{funding_symbol}/funding.size/1D/hist"
        
        response = fetcher.session.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and len(data) > 0:
            latest = data[0]  # Most recent funding data
            return {
                'symbol': symbol,
                'funding_size': float(latest[1]) if len(latest) > 1 else 0,
                'timestamp': pd.to_datetime(latest[0], unit='ms', utc=True) if len(latest) > 0 else None
            }
        return None
    except Exception as e:
        logger.error(f"Failed to get professional funding rates for {symbol}: {e}")
        return None

def get_professional_market_overview(limit: int = 50) -> List[Dict]:
    """Get professional market overview for top trading pairs."""
    fetcher = BitfinexProfessionalDataFetcher()
    
    try:
        # Get all tickers
        response = fetcher.session.get(f"{fetcher.base_url}/v2/tickers", 
                                     params={'symbols': 'ALL'}, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        professional_pairs = []
        
        for ticker in data:
            if (len(ticker) >= 11 and 
                ticker[0].startswith('t') and 
                ticker[0].endswith('USD')):  # Focus on USD pairs for professional analysis
                
                symbol = ticker[0][1:].lower()  # Remove 't' prefix and lowercase
                volume_usd = float(ticker[7]) * float(ticker[6])  # volume * last_price
                
                if volume_usd > MIN_VOLUME_THRESHOLD:  # Professional volume filter
                    professional_pairs.append({
                        'symbol': symbol,
                        'last_price': float(ticker[6]),
                        'volume': float(ticker[7]),
                        'volume_usd': volume_usd,
                        'daily_change_perc': float(ticker[5]),
                        'high': float(ticker[8]),
                        'low': float(ticker[9])
                    })
        
        # Sort by volume (descending) and return top pairs
        professional_pairs.sort(key=lambda x: x['volume_usd'], reverse=True)
        return professional_pairs[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get professional market overview: {e}")
        return []

def get_professional_order_book(symbol: str, precision: str = "P0") -> Optional[Dict]:
    """Get professional order book data for liquidity analysis."""
    fetcher = BitfinexProfessionalDataFetcher()
    
    try:
        symbol_formatted = f"t{symbol.upper()}"
        url = f"{fetcher.base_url}/v2/book/{symbol_formatted}/{precision}"
        
        response = fetcher.session.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        bids = []
        asks = []
        
        for entry in data:
            if len(entry) >= 3:
                price = float(entry[0])
                count = int(entry[1])
                amount = float(entry[2])
                
                if amount > 0:
                    bids.append({'price': price, 'count': count, 'amount': amount})
                else:
                    asks.append({'price': price, 'count': count, 'amount': abs(amount)})
        
        # Sort bids descending, asks ascending
        bids.sort(key=lambda x: x['price'], reverse=True)
        asks.sort(key=lambda x: x['price'])
        
        return {
            'symbol': symbol,
            'bids': bids[:25],  # Top 25 bids
            'asks': asks[:25],  # Top 25 asks
            'bid_liquidity': sum(b['amount'] * b['price'] for b in bids[:10]),
            'ask_liquidity': sum(a['amount'] * a['price'] for a in asks[:10]),
            'spread': (asks[0]['price'] - bids[0]['price']) / bids[0]['price'] if bids and asks else 0,
            'timestamp': pd.Timestamp.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get professional order book for {symbol}: {e}")
        return None

if __name__ == "__main__":
    # Test the professional data fetcher
    logging.basicConfig(level=logging.INFO)
    
    # Test professional fetching for BTC
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    print("Testing Bitfinex professional data fetcher...")
    
    # Test professional data fetch
    btc_data = fetch_data('btcusd', '1h', start_time, end_time)
    print(f"BTC professional data points: {len(btc_data)}")
    
    if not btc_data.empty:
        print("Professional data sample:")
        print(btc_data.head())
        print(f"Professional data range: {btc_data['timestamp'].min()} to {btc_data['timestamp'].max()}")
    
    # Test professional ticker
    fetcher = BitfinexProfessionalDataFetcher()
    btc_ticker = fetcher.get_professional_ticker('btcusd')
    print(f"BTC professional ticker: {btc_ticker}")
    
    # Test professional market overview
    market_overview = get_professional_market_overview(5)
    print(f"Top 5 professional pairs: {[p['symbol'] for p in market_overview]}")
    
    # Test professional order book
    order_book = get_professional_order_book('btcusd')
    if order_book:
        print(f"BTC professional spread: {order_book['spread']:.4%}")
        print(f"Bid liquidity: ${order_book['bid_liquidity']:,.0f}")
    
    print("✅ Bitfinex professional data fetcher test complete")
