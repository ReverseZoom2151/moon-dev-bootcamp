'''
BINANCE VERSION - EARLY TRADERS ANALYZER
Find early traders and large volume participants for specific trading pairs on Binance
Analyzes historical trades to identify significant market participants

To run:
1. Put your Binance API key and secret in dontshareconfig.py
2. Set the SYMBOL you want to analyze (e.g., 'BTCUSDT')
3. Configure date range and trade size filters
4. Run the file

Examples:
BTCUSDT - Bitcoin/USDT pair
ETHUSDT - Ethereum/USDT pair
ADAUSDT - Cardano/USDT pair
'''

import requests
import pandas as pd
import time
import logging
import hmac
import hashlib
import urllib.parse
import sys
sys.path.append('..')
from dontshareconfig import binance_api_key, binance_api_secret
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

# --- Configuration ---

@dataclass(frozen=True)
class BinanceConfig:
    """Binance script configuration."""
    SYMBOL: str = "BTCUSDT"  # Trading pair to analyze
    # Dates in MM-DD-YYYY format
    START_DATE_STR: str = "01-01-2024"
    END_DATE_STR: str = "12-31-2024"
    MIN_TRADE_SIZE_USD: float = 1000.0  # Minimum trade size in USD
    MAX_TRADE_SIZE_USD: float = 1000000.0  # Maximum trade size in USD
    OUTPUT_DIR: Path = Path("binance_output_data")
    API_BASE_URL: str = "https://api.binance.com"
    EXPLORER_BASE_URL: str = "https://www.binance.com/en/trade/"
    # API fetch settings
    FETCH_LIMIT: int = 1000  # Binance max is 1000
    MAX_REQUESTS: int = 100  # Safety limit for API calls
    RETRY_DELAY_SECONDS: int = 2
    MAX_CONSECUTIVE_ERRORS: int = 3
    MAX_EMPTY_BATCHES: int = 3
    # Binance API settings
    API_KEY: str = binance_api_key
    API_SECRET: str = binance_api_secret

CONFIG = BinanceConfig()

# Setup Logging
def setup_logging():
    """Configure logging for the application."""
    log_level = logging.INFO
    
    # Create output directory if it doesn't exist
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(CONFIG.OUTPUT_DIR / 'binance_early_traders.log')
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

def generate_binance_signature(query_string: str, api_secret: str) -> str:
    """Generate HMAC SHA256 signature for Binance API."""
    return hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def parse_date_range(start_str: str, end_str: str) -> Tuple[datetime, datetime]:
    """Parses date strings into timezone-aware UTC datetime objects."""
    start_dt = datetime.strptime(start_str, "%m-%d-%Y").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%m-%d-%Y").replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)
    return start_dt, end_dt

def make_binance_api_request(endpoint: str, params: Dict[str, Any] = None, signed: bool = True) -> Optional[Dict]:
    """Makes a request to the Binance API with proper authentication."""
    try:
        url = f"{CONFIG.API_BASE_URL}{endpoint}"
        
        if params is None:
            params = {}
        
        headers = {
            'X-MBX-APIKEY': CONFIG.API_KEY
        }
        
        if signed:
            # Add timestamp
            params['timestamp'] = int(time.time() * 1000)
            
            # Create query string
            query_string = urllib.parse.urlencode(params)
            
            # Generate signature
            signature = generate_binance_signature(query_string, CONFIG.API_SECRET)
            params['signature'] = signature
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            logger.warning("Rate limit hit, waiting...")
            time.sleep(60)  # Wait 1 minute for rate limit
            return None
        else:
            logger.error(f"Binance API error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def get_symbol_price_usdt(symbol: str, timestamp: int) -> float:
    """Get approximate USDT price for a symbol at a given timestamp."""
    try:
        # For USDT pairs, return 1.0
        if symbol.endswith('USDT'):
            return 1.0
        
        # For BTC pairs, get BTC price in USDT
        if symbol.endswith('BTC'):
            btc_price_data = make_binance_api_request('/api/v3/klines', {
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'startTime': timestamp - (3600 * 1000),  # 1 hour before
                'endTime': timestamp + (3600 * 1000),   # 1 hour after
                'limit': 1
            }, signed=False)
            
            if btc_price_data and len(btc_price_data) > 0:
                # Return close price
                return float(btc_price_data[0][4])
        
        # Default fallback - try to get current price
        ticker_data = make_binance_api_request('/api/v3/ticker/price', {
            'symbol': f"{symbol}USDT"
        }, signed=False)
        
        if ticker_data:
            return float(ticker_data['price'])
        
        return 1.0  # Fallback
        
    except Exception as e:
        logger.warning(f"Error getting USDT price for {symbol}: {e}")
        return 1.0

def process_binance_trade(trade: Dict[str, Any], symbol: str, start_dt: datetime, end_dt: datetime) -> Optional[Dict[str, Any]]:
    """Processes a single Binance trade, calculating USD value and checking filters."""
    try:
        trade_time_ms = trade.get('time')
        if trade_time_ms is None:
            logger.warning("Trade missing 'time'. Skipping.")
            return None
            
        trade_time = datetime.fromtimestamp(trade_time_ms / 1000, tz=timezone.utc)
        
        # Check date range first
        if not (start_dt <= trade_time <= end_dt):
            return None
        
        price = float(trade.get('price', 0))
        qty = float(trade.get('qty', 0))
        quote_qty = float(trade.get('quoteQty', 0))  # This is already in quote currency
        buyer = trade.get('buyer', 'Unknown')
        seller = trade.get('seller', 'Unknown')
        trade_id = trade.get('id')
        is_buyer_maker = trade.get('isBuyerMaker', False)
        
        if not all([price > 0, qty > 0, trade_id]):
            logger.warning(f"Trade missing essential fields at {trade_time}. Skipping.")
            return None
        
        # Calculate USD value
        if symbol.endswith('USDT'):
            trade_size_usd = quote_qty  # Already in USDT
        else:
            # Get USDT conversion rate
            usdt_rate = get_symbol_price_usdt(symbol.replace(symbol[-3:], ''), trade_time_ms)
            trade_size_usd = quote_qty * usdt_rate
        
        # Check trade size filter
        if not (CONFIG.MIN_TRADE_SIZE_USD <= trade_size_usd <= CONFIG.MAX_TRADE_SIZE_USD):
            return None
        
        # Extract base and quote assets
        if symbol.endswith('USDT'):
            base_asset = symbol[:-4]
            quote_asset = 'USDT'
        elif symbol.endswith('BTC'):
            base_asset = symbol[:-3]
            quote_asset = 'BTC'
        else:
            base_asset = symbol[:-3]
            quote_asset = symbol[-3:]
        
        # Determine the significant trader (larger volume participant)
        significant_trader = buyer if is_buyer_maker else seller
        trade_type = "BUY" if is_buyer_maker else "SELL"
        
        explorer_link = f"{CONFIG.EXPLORER_BASE_URL}{symbol}"
        
        return {
            'Timestamp': trade_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'Trader': significant_trader,
            'Explorer Link': explorer_link,
            'Trade Type': trade_type,
            'Base Asset': base_asset,
            'Quote Asset': quote_asset,
            'Price': price,
            'Quantity': qty,
            'Quote Quantity': quote_qty,
            'USD Value': trade_size_usd,
            'Trade ID': trade_id,
            'Is Buyer Maker': is_buyer_maker
        }
        
    except Exception as e:
        logger.exception(f"Unexpected error processing Binance trade: {trade}. Error: {e}")
        return None

def fetch_and_process_binance_trades(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetches Binance trades and processes them based on filters."""
    all_processed_trades: List[Dict[str, Any]] = []
    
    # Convert to milliseconds for Binance API
    start_time_ms = int(start_dt.timestamp() * 1000)
    end_time_ms = int(end_dt.timestamp() * 1000)
    
    from_id = 0
    total_trades_fetched = 0
    consecutive_errors = 0
    consecutive_empty_batches = 0
    requests_made = 0
    
    logger.info(f"Starting Binance trade fetch for {symbol}...")
    
    while requests_made < CONFIG.MAX_REQUESTS:
        # Binance historical trades endpoint
        params = {
            'symbol': symbol,
            'limit': CONFIG.FETCH_LIMIT
        }
        
        if from_id > 0:
            params['fromId'] = from_id
        
        logger.info(f"Fetching Binance trades from ID {from_id}...")
        
        data = make_binance_api_request('/api/v3/historicalTrades', params, signed=False)
        requests_made += 1
        
        if not data:
            consecutive_errors += 1
            logger.warning(f"Binance API request failed. Error count: {consecutive_errors}/{CONFIG.MAX_CONSECUTIVE_ERRORS}")
            if consecutive_errors >= CONFIG.MAX_CONSECUTIVE_ERRORS:
                logger.error(f"Reached maximum consecutive errors. Stopping.")
                break
            time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            continue
        else:
            consecutive_errors = 0
        
        trades = data if isinstance(data, list) else []
        total_trades_fetched += len(trades)
        
        if not trades:
            consecutive_empty_batches += 1
            logger.info(f"No trades in batch. Empty batch count: {consecutive_empty_batches}/{CONFIG.MAX_EMPTY_BATCHES}")
            if consecutive_empty_batches >= CONFIG.MAX_EMPTY_BATCHES:
                logger.info("Reached maximum empty batches. End of data.")
                break
            from_id += CONFIG.FETCH_LIMIT
            continue
        else:
            consecutive_empty_batches = 0
        
        # Process trades in this batch
        batch_processed = 0
        earliest_time_in_batch = None
        latest_time_in_batch = None
        
        for trade in trades:
            processed_trade = process_binance_trade(trade, symbol, start_dt, end_dt)
            
            # Track time range
            trade_time = trade.get('time')
            if trade_time:
                if earliest_time_in_batch is None or trade_time < earliest_time_in_batch:
                    earliest_time_in_batch = trade_time
                if latest_time_in_batch is None or trade_time > latest_time_in_batch:
                    latest_time_in_batch = trade_time
            
            if processed_trade:
                all_processed_trades.append(processed_trade)
                batch_processed += 1
            
            # Update from_id for next batch
            trade_id = trade.get('id')
            if trade_id and trade_id > from_id:
                from_id = trade_id
        
        logger.info(f"Batch processed: {batch_processed} qualifying trades out of {len(trades)} total")
        
        # Check if we've gone beyond our time range
        if earliest_time_in_batch and earliest_time_in_batch > end_time_ms:
            logger.info("Reached trades beyond target date range. Stopping.")
            break
        
        time.sleep(0.1)  # Small delay to avoid rate limits
    
    logger.info(f"Total Binance trades fetched: {total_trades_fetched}")
    logger.info(f"Total qualifying trades: {len(all_processed_trades)}")
    
    if all_processed_trades:
        df = pd.DataFrame(all_processed_trades)
        # Sort by timestamp
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()

def save_binance_trades(df: pd.DataFrame, output_path: Path):
    """Saves the Binance DataFrame to a CSV file."""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} Binance trades to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Binance trades: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    start_timer = time.time()
    
    logger.info("🚀 Starting Binance Early Traders Analysis")
    logger.info(f"📊 Symbol: {CONFIG.SYMBOL}")
    logger.info(f"📅 Date range: {CONFIG.START_DATE_STR} to {CONFIG.END_DATE_STR}")
    logger.info(f"💰 Trade size range: ${CONFIG.MIN_TRADE_SIZE_USD:,.2f} - ${CONFIG.MAX_TRADE_SIZE_USD:,.2f}")
    
    try:
        # Parse dates
        start_dt, end_dt = parse_date_range(CONFIG.START_DATE_STR, CONFIG.END_DATE_STR)
        logger.info(f"Parsed date range: {start_dt} to {end_dt}")
        
        # Fetch and process trades
        trades_df = fetch_and_process_binance_trades(CONFIG.SYMBOL, start_dt, end_dt)
        
        if not trades_df.empty:
            # Generate output filename
            output_filename = f"binance_early_traders_{CONFIG.SYMBOL}_{CONFIG.START_DATE_STR.replace('-', '')}_{CONFIG.END_DATE_STR.replace('-', '')}.csv"
            output_path = CONFIG.OUTPUT_DIR / output_filename
            
            # Save results
            save_binance_trades(trades_df, output_path)
            
            # Summary statistics
            total_trades = len(trades_df)
            total_volume = trades_df['USD Value'].sum()
            avg_trade_size = trades_df['USD Value'].mean()
            
            logger.info("✅ Binance Analysis Complete!")
            logger.info(f"📈 Total qualifying trades: {total_trades:,}")
            logger.info(f"💰 Total volume: ${total_volume:,.2f}")
            logger.info(f"📊 Average trade size: ${avg_trade_size:,.2f}")
            logger.info(f"📄 Results saved to: {output_path}")
            
        else:
            logger.warning("⚠️ No qualifying Binance trades found")
            
    except Exception as e:
        logger.error(f"❌ Fatal error in Binance analysis: {e}", exc_info=True)
        raise
    
    finally:
        end_timer = time.time()
        logger.info(f"⏱️ Total execution time: {end_timer - start_timer:.2f} seconds")
