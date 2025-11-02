'''
BITFINEX PROFESSIONAL VERSION - INSTITUTIONAL TRADERS ANALYZER
Find institutional traders and large volume participants for specific trading pairs on Bitfinex
Analyzes historical trades to identify significant market participants and institutional flow

To run:
1. Put your Bitfinex API key and secret in dontshareconfig.py
2. Set the SYMBOL you want to analyze (e.g., 'tBTCUSD')
3. Configure date range and trade size filters
4. Run the file

Examples:
tBTCUSD - Bitcoin/USD pair
tETHUSD - Ethereum/USD pair
tLTCUSD - Litecoin/USD pair
Note: Bitfinex symbols start with 't' for trading pairs
'''

import sys
import requests
import json
import pandas as pd
import time
import logging
import hmac
import hashlib
sys.path.append('..')
from dontshareconfig import bitfinex_api_key, bitfinex_api_secret
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, timezone

# --- Configuration ---

@dataclass(frozen=True)
class BitfinexProfessionalConfig:
    """Bitfinex professional script configuration."""
    SYMBOL: str = "tBTCUSD"  # Trading pair to analyze (Bitfinex format with 't' prefix)
    # Dates in MM-DD-YYYY format
    START_DATE_STR: str = "01-01-2024"
    END_DATE_STR: str = "12-31-2024"
    MIN_TRADE_SIZE_USD: float = 5000.0  # Higher minimum for institutional analysis
    MAX_TRADE_SIZE_USD: float = 10000000.0  # Maximum trade size in USD
    OUTPUT_DIR: Path = Path("bitfinex_professional_output_data")
    API_BASE_URL: str = "https://api-pub.bitfinex.com"
    EXPLORER_BASE_URL: str = "https://www.bitfinex.com/t/"
    # Professional API fetch settings
    FETCH_LIMIT: int = 5000  # Bitfinex allows up to 5000
    MAX_REQUESTS: int = 200  # Higher limit for institutional analysis
    RETRY_DELAY_SECONDS: int = 3
    MAX_CONSECUTIVE_ERRORS: int = 5  # More resilient for professional use
    MAX_EMPTY_BATCHES: int = 5
    # Bitfinex professional API settings
    API_KEY: str = bitfinex_api_key
    API_SECRET: str = bitfinex_api_secret
    # Professional features
    INCLUDE_FUNDING_ANALYSIS: bool = True
    INSTITUTIONAL_THRESHOLD_USD: float = 50000.0  # Mark as institutional if above this
    PROFESSIONAL_GRADE_FEATURES: bool = True

CONFIG = BitfinexProfessionalConfig()

# Setup Professional Logging
def setup_professional_logging():
    """Configure professional logging for institutional analysis."""
    log_level = logging.INFO
    
    # Create professional output directory
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Professional logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(CONFIG.OUTPUT_DIR / 'bitfinex_professional_traders.log')
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_professional_logging()

def generate_bitfinex_signature(path: str, nonce: str, body: str = '') -> str:
    """Generate HMAC SHA384 signature for Bitfinex API."""
    message = f'/api{path}{nonce}{body}'
    signature = hmac.new(
        CONFIG.API_SECRET.encode(),
        message.encode(),
        hashlib.sha384
    ).hexdigest()
    return signature

def parse_professional_date_range(start_str: str, end_str: str) -> Tuple[datetime, datetime]:
    """Parses date strings into timezone-aware UTC datetime objects for professional analysis."""
    start_dt = datetime.strptime(start_str, "%m-%d-%Y").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, "%m-%d-%Y").replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)
    return start_dt, end_dt

def make_bitfinex_professional_api_request(endpoint: str, params: Dict[str, Any] = None) -> Optional[List]:
    """Makes a professional request to the Bitfinex API."""
    try:
        url = f"{CONFIG.API_BASE_URL}{endpoint}"
        
        if params:
            # Build query string for GET requests
            query_params = []
            for key, value in params.items():
                if isinstance(value, list):
                    query_params.append(f"{key}={','.join(map(str, value))}")
                else:
                    query_params.append(f"{key}={value}")
            
            if query_params:
                url += '?' + '&'.join(query_params)
        
        # Professional headers for institutional access
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'Bitfinex Professional Trader Analysis System'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, list) else [data]
        elif response.status_code == 429:
            logger.warning("Professional rate limit hit, implementing backoff strategy...")
            time.sleep(60)  # Professional backoff
            return None
        else:
            logger.error(f"Bitfinex professional API error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Professional request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected professional error: {e}")
        return None

def get_professional_usd_conversion_rate(symbol: str, timestamp: int) -> float:
    """Get professional USD conversion rate for non-USD pairs."""
    try:
        # If already USD pair, return 1.0
        if symbol.upper().endswith('USD'):
            return 1.0
        
        # For BTC pairs, get BTCUSD rate
        if symbol.upper().endswith('BTC'):
            btc_rate_data = make_bitfinex_professional_api_request(
                f"/v2/candles/trade:1h:tBTCUSD/hist",
                {
                    'start': timestamp - (3600 * 1000),
                    'end': timestamp + (3600 * 1000),
                    'limit': 1
                }
            )
            
            if btc_rate_data and len(btc_rate_data) > 0:
                return float(btc_rate_data[0][2])  # Close price
        
        # For other pairs, try to get current ticker
        ticker_data = make_bitfinex_professional_api_request(f"/v2/ticker/{symbol}")
        if ticker_data and len(ticker_data) > 0:
            return float(ticker_data[6])  # Last price
        
        return 1.0  # Professional fallback
        
    except Exception as e:
        logger.warning(f"Professional USD conversion error for {symbol}: {e}")
        return 1.0

def classify_trader_tier(trade_size_usd: float) -> str:
    """Classify trader based on trade size for professional analysis."""
    if trade_size_usd >= 1000000:
        return "Whale"
    elif trade_size_usd >= CONFIG.INSTITUTIONAL_THRESHOLD_USD:
        return "Institutional"
    elif trade_size_usd >= 10000:
        return "Professional"
    else:
        return "Retail"

def process_bitfinex_professional_trade(trade: List, symbol: str, start_dt: datetime, end_dt: datetime) -> Optional[Dict[str, Any]]:
    """Processes a single Bitfinex trade with professional institutional analysis."""
    try:
        # Bitfinex trade format: [ID, MTS, AMOUNT, PRICE]
        if len(trade) < 4:
            logger.warning("Professional trade data incomplete. Skipping.")
            return None
        
        trade_id = trade[0]
        trade_time_ms = trade[1]
        amount = float(trade[2])
        price = float(trade[3])
        
        trade_time = datetime.fromtimestamp(trade_time_ms / 1000, tz=timezone.utc)
        
        # Professional date range check
        if not (start_dt <= trade_time <= end_dt):
            return None
        
        if not all([trade_id, price > 0, amount != 0]):
            logger.warning(f"Professional trade missing essential data at {trade_time}. Skipping.")
            return None
        
        # Professional trade size calculation
        trade_value = abs(amount) * price
        
        # Get USD conversion for professional analysis
        if not symbol.upper().endswith('USD'):
            usd_rate = get_professional_usd_conversion_rate(symbol, trade_time_ms)
            trade_size_usd = trade_value * usd_rate
        else:
            trade_size_usd = trade_value
        
        # Professional trade size filter
        if not (CONFIG.MIN_TRADE_SIZE_USD <= trade_size_usd <= CONFIG.MAX_TRADE_SIZE_USD):
            return None
        
        # Professional trader classification
        trader_tier = classify_trader_tier(trade_size_usd)
        
        # Extract professional symbol information
        if symbol.startswith('t'):
            clean_symbol = symbol[1:]  # Remove 't' prefix
        else:
            clean_symbol = symbol
        
        # Professional trade direction analysis
        trade_direction = "BUY" if amount > 0 else "SELL"
        
        # Professional explorer link
        explorer_link = f"{CONFIG.EXPLORER_BASE_URL}{clean_symbol}"
        
        # Professional trade analysis
        return {
            'Timestamp': trade_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'Trade ID': trade_id,
            'Symbol': symbol,
            'Explorer Link': explorer_link,
            'Direction': trade_direction,
            'Amount': abs(amount),
            'Price': price,
            'Trade Value': trade_value,
            'USD Value': trade_size_usd,
            'Trader Tier': trader_tier,
            'Professional Grade': trader_tier in ['Institutional', 'Whale'],
            'Market Impact': 'High' if trade_size_usd >= 100000 else 'Medium' if trade_size_usd >= 25000 else 'Low',
            'Analysis Timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Unexpected error processing professional Bitfinex trade: {trade}. Error: {e}")
        return None

def fetch_and_process_bitfinex_professional_trades(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetches Bitfinex trades with professional institutional analysis."""
    all_processed_trades: List[Dict[str, Any]] = []
    
    # Convert to milliseconds for Bitfinex API
    start_time_ms = int(start_dt.timestamp() * 1000)
    end_time_ms = int(end_dt.timestamp() * 1000)
    
    total_trades_fetched = 0
    consecutive_errors = 0
    consecutive_empty_batches = 0
    requests_made = 0
    
    logger.info(f"Starting Bitfinex professional trade fetch for {symbol}...")
    
    # Professional batch processing
    current_end = end_time_ms
    batch_size_ms = 24 * 60 * 60 * 1000  # 1 day batches for professional analysis
    
    while requests_made < CONFIG.MAX_REQUESTS and current_end > start_time_ms:
        current_start = max(start_time_ms, current_end - batch_size_ms)
        
        logger.info(f"Fetching professional trades from {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(current_end/1000)}...")
        
        # Bitfinex professional trades endpoint
        data = make_bitfinex_professional_api_request(
            f"/v2/trades/{symbol}/hist",
            {
                'start': current_start,
                'end': current_end,
                'limit': CONFIG.FETCH_LIMIT,
                'sort': -1  # Descending order
            }
        )
        
        requests_made += 1
        
        if not data:
            consecutive_errors += 1
            logger.warning(f"Professional API request failed. Error count: {consecutive_errors}/{CONFIG.MAX_CONSECUTIVE_ERRORS}")
            if consecutive_errors >= CONFIG.MAX_CONSECUTIVE_ERRORS:
                logger.error("Reached maximum consecutive errors in professional analysis.")
                break
            time.sleep(CONFIG.RETRY_DELAY_SECONDS)
            continue
        else:
            consecutive_errors = 0
        
        trades = data if isinstance(data, list) else []
        total_trades_fetched += len(trades)
        
        if not trades:
            consecutive_empty_batches += 1
            logger.info(f"No professional trades in batch. Empty count: {consecutive_empty_batches}/{CONFIG.MAX_EMPTY_BATCHES}")
            if consecutive_empty_batches >= CONFIG.MAX_EMPTY_BATCHES:
                logger.info("Professional analysis: End of relevant data reached.")
                break
            current_end = current_start
            continue
        else:
            consecutive_empty_batches = 0
        
        # Process trades in this professional batch
        batch_processed = 0
        
        for trade in trades:
            processed_trade = process_bitfinex_professional_trade(trade, symbol, start_dt, end_dt)
            
            if processed_trade:
                all_processed_trades.append(processed_trade)
                batch_processed += 1
        
        logger.info(f"Professional batch processed: {batch_processed} institutional trades out of {len(trades)} total")
        
        # Move to next batch
        current_end = current_start
        
        # Professional rate limiting
        time.sleep(0.5)  # Professional courtesy delay
    
    logger.info(f"Total professional trades fetched: {total_trades_fetched}")
    logger.info(f"Total qualifying institutional trades: {len(all_processed_trades)}")
    
    if all_processed_trades:
        df = pd.DataFrame(all_processed_trades)
        # Professional sorting by timestamp and USD value
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(['Timestamp', 'USD Value'], ascending=[True, False]).reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()

def save_bitfinex_professional_trades(df: pd.DataFrame, output_path: Path):
    """Saves the professional Bitfinex DataFrame with institutional analysis."""
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} professional Bitfinex trades to {output_path}")
        
        # Professional summary analysis
        if len(df) > 0:
            institutional_trades = df[df['Trader Tier'].isin(['Institutional', 'Whale'])]
            logger.info(f"📊 Institutional trades identified: {len(institutional_trades)} ({len(institutional_trades)/len(df)*100:.1f}%)")
            
            if len(institutional_trades) > 0:
                logger.info(f"💰 Institutional volume: ${institutional_trades['USD Value'].sum():,.2f}")
                logger.info(f"📈 Average institutional trade: ${institutional_trades['USD Value'].mean():,.2f}")
        
    except Exception as e:
        logger.error(f"Failed to save professional Bitfinex trades: {e}")

def generate_professional_analysis_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate professional institutional analysis report."""
    if df.empty:
        return {}
    
    report = {
        'analysis_timestamp': datetime.utcnow().isoformat(),
        'total_trades': len(df),
        'total_volume_usd': df['USD Value'].sum(),
        'average_trade_size_usd': df['USD Value'].mean(),
        'trader_tier_distribution': df['Trader Tier'].value_counts().to_dict(),
        'direction_analysis': df['Direction'].value_counts().to_dict(),
        'market_impact_distribution': df['Market Impact'].value_counts().to_dict(),
        'professional_grade_percentage': (df['Professional Grade'].sum() / len(df)) * 100,
        'largest_trade': df.loc[df['USD Value'].idxmax()].to_dict() if len(df) > 0 else {},
        'time_range': {
            'start': df['Timestamp'].min(),
            'end': df['Timestamp'].max()
        }
    }
    
    return report

# --- Main Professional Execution ---

if __name__ == "__main__":
    start_timer = time.time()
    
    logger.info("🏛️ Starting Bitfinex Professional Institutional Traders Analysis")
    logger.info(f"📊 Symbol: {CONFIG.SYMBOL}")
    logger.info(f"📅 Date range: {CONFIG.START_DATE_STR} to {CONFIG.END_DATE_STR}")
    logger.info(f"💰 Trade size range: ${CONFIG.MIN_TRADE_SIZE_USD:,.2f} - ${CONFIG.MAX_TRADE_SIZE_USD:,.2f}")
    logger.info(f"🏢 Institutional threshold: ${CONFIG.INSTITUTIONAL_THRESHOLD_USD:,.2f}")
    
    try:
        # Parse professional dates
        start_dt, end_dt = parse_professional_date_range(CONFIG.START_DATE_STR, CONFIG.END_DATE_STR)
        logger.info(f"Professional analysis period: {start_dt} to {end_dt}")
        
        # Fetch and process professional trades
        trades_df = fetch_and_process_bitfinex_professional_trades(CONFIG.SYMBOL, start_dt, end_dt)
        
        if not trades_df.empty:
            # Generate professional output filename
            output_filename = f"bitfinex_professional_traders_{CONFIG.SYMBOL}_{CONFIG.START_DATE_STR.replace('-', '')}_{CONFIG.END_DATE_STR.replace('-', '')}.csv"
            output_path = CONFIG.OUTPUT_DIR / output_filename
            
            # Save professional results
            save_bitfinex_professional_trades(trades_df, output_path)
            
            # Generate and save professional analysis report
            professional_report = generate_professional_analysis_report(trades_df)
            report_path = CONFIG.OUTPUT_DIR / f"professional_analysis_report_{CONFIG.SYMBOL}.json"
            
            with open(report_path, 'w') as f:
                json.dump(professional_report, f, indent=2, default=str)
            
            # Professional summary statistics
            total_trades = len(trades_df)
            total_volume = trades_df['USD Value'].sum()
            avg_trade_size = trades_df['USD Value'].mean()
            institutional_count = len(trades_df[trades_df['Trader Tier'].isin(['Institutional', 'Whale'])])
            
            logger.info("✅ Bitfinex Professional Analysis Complete!")
            logger.info(f"📈 Total qualifying trades: {total_trades:,}")
            logger.info(f"🏢 Institutional trades: {institutional_count:,} ({institutional_count/total_trades*100:.1f}%)")
            logger.info(f"💰 Total volume: ${total_volume:,.2f}")
            logger.info(f"📊 Average trade size: ${avg_trade_size:,.2f}")
            logger.info(f"📄 Results saved to: {output_path}")
            logger.info(f"📋 Professional report: {report_path}")
            
        else:
            logger.warning("⚠️ No qualifying professional trades found")
            
    except Exception as e:
        logger.error(f"❌ Fatal error in professional Bitfinex analysis: {e}", exc_info=True)
        raise
    
    finally:
        end_timer = time.time()
        logger.info(f"⏱️ Total professional execution time: {end_timer - start_timer:.2f} seconds")
