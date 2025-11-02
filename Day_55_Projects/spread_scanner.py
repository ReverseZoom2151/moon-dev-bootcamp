import requests
import pandas as pd
import time
import json
import threading
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from termcolor import colored

# Configuration Constants
class Config:
    """Configuration constants for the spread scanner"""
    MARKETS_TO_SCAN = 10
    MAX_SPREAD_FILTER = 85.0
    MIN_INVESTMENT_THRESHOLD = 100.0
    API_TIMEOUT = 10
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0
    THREAD_POOL_SIZE = 5
    CACHE_DURATION = 30  # seconds
    
    # API URLs
    GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"
    CLOB_API_URL = "https://clob.polymarket.com"

@dataclass
class OpportunityData:
    """Data class for spread opportunities"""
    market: str
    token: str
    bid: float
    ask: float
    spread: float
    spread_pct: float
    min_size: float
    max_investment: float
    volume_24h: float
    market_id: str
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.profit_for_100 = (100 / self.ask * self.spread) if self.ask > 0 else 0
        self.roi_for_100 = (self.profit_for_100 / 100) * 100 if self.profit_for_100 > 0 else 0
        self.can_invest_100 = self.max_investment >= 100

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spread_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIClient:
    """Centralized API client with retry logic and rate limiting"""
    
    def __init__(self):
        self._session = requests.Session()
        self._session.timeout = Config.API_TIMEOUT
        self._cache = {}
        self._cache_timestamps = {}
        self._lock = threading.Lock()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[key]) < Config.CACHE_DURATION
    
    def get_with_retry(self, url: str, params: Dict = None, use_cache: bool = True) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and caching"""
        cache_key = f"{url}_{str(params)}"
        
        # Check cache first
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {url}")
            return self._cache[cache_key]
        
        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                response = self._session.get(url, params=params)
                response.raise_for_status()
                
                # Cache successful response
                if use_cache:
                    with self._lock:
                        self._cache[cache_key] = response
                        self._cache_timestamps[cache_key] = time.time()
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{Config.RETRY_ATTEMPTS}): {e}")
                if attempt < Config.RETRY_ATTEMPTS - 1:
                    time.sleep(Config.RETRY_DELAY * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All retry attempts failed for {url}")
                    return None
        
        return None

def get_token_id(market_id: str, choice: Optional[str] = None) -> Union[str, List[str]]:
    """
    Get token IDs for a specific market
    
    Args:
        market_id: The market ID from Polymarket
        choice: Either 'yes' or 'no' (case insensitive). If None, returns all token IDs
    
    Returns:
        Single token ID string if choice specified, otherwise [market_id, yes_token_id, no_token_id]
    """
    logger.info(f"Fetching token ID(s) for market {market_id}{f', {choice}' if choice else ' (all tokens)'}")
    
    try:
        api_client = APIClient()
        params = {'closed': 'false', 'limit': 1000}
        
        response = api_client.get_with_retry(Config.GAMMA_API_URL, params)
        if not response:
            logger.error(f"Failed to fetch markets data")
            return '' if choice else ['', '', '']
        
        df = pd.DataFrame(response.json())
        
        # Find the specific market
        market_row = df[df['id'].astype(str) == str(market_id)]
        
        if market_row.empty:
            try:
                market_row = df[df['id'] == int(market_id)]
            except (ValueError, TypeError):
                pass
        
        if market_row.empty:
            logger.error(f"Market ID {market_id} not found")
            return '' if choice else ['', '', '']
        
        clob_token_ids = market_row['clobTokenIds'].iloc[0]
        
        if pd.isna(clob_token_ids):
            logger.error(f"No token IDs found for market {market_id}")
            return '' if choice else ['', '', '']
        
        token_list = json.loads(clob_token_ids)
        
        if len(token_list) < 2:
            logger.error(f"Invalid token list for market {market_id}")
            return '' if choice else ['', '', '']
        
        yes_token_id, no_token_id = token_list[0], token_list[1]
        
        if choice is None:
            logger.info(f"Found all tokens for market {market_id}")
            return [market_id, yes_token_id, no_token_id]
        
        choice = choice.lower().strip()
        if choice not in ['yes', 'no']:
            logger.error(f"Invalid choice '{choice}', must be 'yes' or 'no'")
            return ''
        
        token_id = yes_token_id if choice == 'yes' else no_token_id
        logger.info(f"Found {choice.upper()} token: {token_id[:20]}...")
        return token_id
        
    except Exception as e:
        logger.error(f"Error getting token ID: {e}")
        return '' if choice else ['', '', '']

class SpreadScanner:
    """Enhanced spread scanner with improved performance and error handling"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.markets_to_scan: List[Dict] = []
        self.opportunities: List[OpportunityData] = []
        self.api_client = APIClient()
        self.last_scan_time: Optional[datetime] = None
        self.scan_count = 0
        
        logger.info("💰 MOON DEV's Enhanced Spread Scanner initialized! 🌙")
        logger.info(f"📊 Will monitor top {self.config.MARKETS_TO_SCAN} markets")
        logger.info(f"🚫 Filtering spreads over {self.config.MAX_SPREAD_FILTER}%")
        logger.info(f"💵 Minimum investment threshold: ${self.config.MIN_INVESTMENT_THRESHOLD}")
        
    def load_top_markets(self, count: Optional[int] = None) -> bool:
        """Load top markets by volume from LIVE Polymarket API"""
        count = count or self.config.MARKETS_TO_SCAN
        logger.info(f"Loading top {count} markets by volume...")
        
        try:
            params = {
                'closed': 'false',
                'order': 'volume24hr',
                'ascending': 'false',
                'limit': count * 3  # Get extra to filter out problematic ones
            }
            
            response = self.api_client.get_with_retry(self.config.GAMMA_API_URL, params)
            if not response:
                logger.error("Failed to fetch markets data")
                return False
            
            markets_data = response.json()
            if not markets_data:
                logger.error("No markets data received")
                return False
            
            logger.info(f"Received {len(markets_data)} markets from API")
            
            # Filter and process markets
            valid_markets = []
            
            for market in markets_data:
                if not self._is_valid_market(market):
                    continue
                
                # Shorten question for display
                question = market['question']
                if len(question) > 50:
                    question = question[:47] + "..."
                
                valid_markets.append({
                    'id': market['id'],
                    'question': question,
                    'volume24hr': float(market.get('volume24hr', 0)),
                    'clobTokenIds': market['clobTokenIds']
                })
                
                if len(valid_markets) >= count:
                    break
            
            self.markets_to_scan = valid_markets
            logger.info(f"Loaded {len(self.markets_to_scan)} valid markets")
            
            self._log_markets_summary()
            return True
                
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return False
    
    def _is_valid_market(self, market: Dict) -> bool:
        """Check if a market is valid for scanning"""
        required_fields = ['id', 'question', 'volume24hr', 'clobTokenIds']
        if not all(key in market for key in required_fields):
            return False
        
        # Must have token IDs
        if pd.isna(market.get('clobTokenIds')):
            return False
        
        # Must have meaningful volume
        volume = float(market.get('volume24hr', 0))
        return volume >= 100  # Skip very low volume markets
    
    def _log_markets_summary(self):
        """Log summary of loaded markets"""
        if not self.markets_to_scan:
            return
            
        logger.info("Markets to scan:")
        for i, market in enumerate(self.markets_to_scan, 1):
            logger.info(f"  {i}. {market['question']} (${market['volume24hr']:,.0f} 24h vol)")
            logger.info("💡 Make sure you have internet connection!")
    
    def get_spread_data(self, token_id: str) -> Optional[Dict]:
        """Get spread data for a token with improved error handling"""
        try:
            url = f"{self.config.CLOB_API_URL}/book"
            params = {'token_id': token_id}
            
            response = self.api_client.get_with_retry(url, params, use_cache=False)  # Don't cache order book data
            if not response:
                logger.warning(f"Failed to get spread data for token {token_id[:20]}...")
                return None
            
            data = response.json()
            asks = data.get('asks', [])
            bids = data.get('bids', [])
            
            if not asks or not bids:
                logger.debug(f"No asks or bids for token {token_id[:20]}...")
                return None
            
            # Best bid (highest price buyers will pay - last element)
            best_bid = float(bids[-1]['price'])
            best_bid_size = float(bids[-1]['size'])
            
            # Best ask (lowest price sellers will accept - last element)
            best_ask = float(asks[-1]['price'])
            best_ask_size = float(asks[-1]['size'])
            
            # Calculate spread
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_size': best_bid_size,
                'ask_size': best_ask_size,
                'spread': spread,
                'spread_pct': spread_pct
            }
            
        except Exception as e:
            logger.error(f"Error getting spread data for token {token_id[:20]}...: {e}")
            return None
            
    def scan_market_spreads(self, market: Dict) -> List[OpportunityData]:
        """Scan spreads for both YES and NO tokens in a market with improved token handling"""
        logger.debug(f"Scanning: {market['question']}")
        
        # Get token IDs for this market
        token_data = get_token_id(market['id'])
        if len(token_data) != 3:
            logger.warning(f"Could not get token IDs for market {market['id']}")
            return []
        
        _, yes_token_id, no_token_id = token_data
        
        # Get spread data for both tokens using parallel processing
        with ThreadPoolExecutor(max_workers=2) as executor:
            yes_future = executor.submit(self.get_spread_data, yes_token_id)
            no_future = executor.submit(self.get_spread_data, no_token_id)
            
            yes_data = yes_future.result()
            no_data = no_future.result()
        
        # Check if market is essentially resolved (either token over 98%)
        market_resolved = False
        if yes_data and (yes_data['best_ask'] > 0.98 or yes_data['best_bid'] > 0.98):
            logger.info(f"Market essentially resolved - YES token over 98% (Bid: ${yes_data['best_bid']:.3f}, Ask: ${yes_data['best_ask']:.3f})")
            market_resolved = True
        elif no_data and (no_data['best_ask'] > 0.98 or no_data['best_bid'] > 0.98):
            logger.info(f"Market essentially resolved - NO token over 98% (Bid: ${no_data['best_bid']:.3f}, Ask: ${no_data['best_ask']:.3f})")
            market_resolved = True
            
        if market_resolved:
            return []  # Skip entire market
        
        opportunities = []
        
        # Process YES token opportunity
        if yes_data:
            opp = self._create_opportunity_data(market, 'YES', yes_data)
            if opp:
                opportunities.append(opp)
        
        # Process NO token opportunity        
        if no_data:
            opp = self._create_opportunity_data(market, 'NO', no_data)
            if opp:
                opportunities.append(opp)
            
        return opportunities
    
    def _create_opportunity_data(self, market: Dict, token_type: str, spread_data: Dict) -> Optional[OpportunityData]:
        """Create OpportunityData object from market and spread data"""
        try:
            profit_per_share = spread_data['spread']
            min_size = min(spread_data['bid_size'], spread_data['ask_size'])
            max_investment = spread_data['best_ask'] * min_size  # Use ask price for max investment calculation
            
            # Filter out poor opportunities
            if (profit_per_share <= 0 or 
                max_investment < self.config.MIN_INVESTMENT_THRESHOLD or 
                spread_data['spread_pct'] > self.config.MAX_SPREAD_FILTER):
                return None
            
            return OpportunityData(
                market=market['question'],
                token=token_type,
                bid=spread_data['best_bid'],
                ask=spread_data['best_ask'],
                spread=spread_data['spread'],
                spread_pct=spread_data['spread_pct'],
                min_size=min_size,
                max_investment=max_investment,
                volume_24h=market['volume24hr'],
                market_id=str(market['id'])
            )
            
        except Exception as e:
            logger.error(f"Error creating opportunity data: {e}")
            return None
    
    def scan_all_markets(self) -> bool:
        """Scan all markets and collect opportunities using parallel processing"""
        if not self.markets_to_scan:
            logger.warning("No markets to scan")
            return False
            
        logger.info(f"Scanning {len(self.markets_to_scan)} markets for spread opportunities...")
        self.opportunities = []
        self.scan_count += 1
        self.last_scan_time = datetime.now()
        
        # Use ThreadPoolExecutor for parallel market scanning
        with ThreadPoolExecutor(max_workers=self.config.THREAD_POOL_SIZE) as executor:
            future_to_market = {executor.submit(self.scan_market_spreads, market): market for market in self.markets_to_scan}
            
            for future in as_completed(future_to_market):
                market = future_to_market[future]
                try:
                    market_opportunities = future.result(timeout=30)  # 30 second timeout per market
                    self.opportunities.extend(market_opportunities)
                except Exception as e:
                    logger.error(f"Error scanning market {market.get('question', 'Unknown')}: {e}")
        
        logger.info(f"Found {len(self.opportunities)} total opportunities")
        return True
    
    def print_opportunities_table(self):
        """Print opportunities in a clean, enhanced table format"""
        if not self.opportunities:
            print("❌ No opportunities found!")
            return
        
        # Filter and sort opportunities
        filtered_opps = [opp for opp in self.opportunities 
                        if opp.spread_pct <= self.config.MAX_SPREAD_FILTER]
        sorted_opps = sorted(filtered_opps, key=lambda x: x.spread_pct, reverse=True)
        
        if not sorted_opps:
            print(f"❌ No realistic opportunities found after filtering spreads over {self.config.MAX_SPREAD_FILTER}%!")
            return
        
        # Calculate dynamic column widths
        max_market_length = max(len(opp.market) for opp in sorted_opps[:15]) if sorted_opps else 50
        market_width = max(max_market_length + 2, 50)
        table_width = market_width + 75
        
        # Print header
        print(f"\n💎 Enhanced Spread Scanner Results (Scan #{self.scan_count})")
        print(f"🔥 Found {len(sorted_opps)} realistic opportunities")
        if len(self.opportunities) > len(sorted_opps):
            print(f"🚫 Filtered out {len(self.opportunities) - len(sorted_opps)} unrealistic opportunities")
        
        print("=" * table_width)
        header = f"{'#':<2} {'Market':<{market_width}} {'Tok':<3} {'Bid':<6} {'Ask':<6} {'Spr%':<5} {'MaxUSD':<7} {'Prof':<6} {'$100P':<6} {'ROI%':<5}"
        print(header)
        print("-" * table_width)
        
        # Print top opportunities
        for i, opp in enumerate(sorted_opps[:15], 1):
            color = self._get_opportunity_color(opp)
            line = self._format_opportunity_line(i, opp, market_width)
            print(colored(line, color))
            
        print("=" * table_width)
        
        # Print enhanced summary statistics
        self._print_opportunity_statistics(sorted_opps)
        self._print_tips_and_legends()
    
    def _get_opportunity_color(self, opp: OpportunityData) -> str:
        """Determine color coding for opportunity based on quality"""
        if opp.spread_pct > 5.0 and opp.can_invest_100:
            return 'green'  # Excellent opportunities
        elif opp.spread_pct > 2.0:
            return 'yellow'  # Good opportunities
        else:
            return 'white'  # Modest opportunities
    
    def _format_opportunity_line(self, rank: int, opp: OpportunityData, market_width: int) -> str:
        """Format a single opportunity line for the table"""
        max_profit_dollars = opp.spread * opp.min_size
        
        return (f"{rank:<2} {opp.market:<{market_width}} {opp.token:<3} "
                f"${opp.bid:<5.3f} ${opp.ask:<5.3f} {opp.spread_pct:>4.1f}% "
                f"${opp.max_investment:<6,.0f} ${max_profit_dollars:<5.2f} "
                f"${opp.profit_for_100:<5.2f} {opp.roi_for_100:>4.1f}%")
    
    def _print_opportunity_statistics(self, opportunities: List[OpportunityData]):
        """Print enhanced summary statistics"""
        if not opportunities:
            return
            
        best_opp = opportunities[0]
        avg_spread = sum(opp.spread_pct for opp in opportunities) / len(opportunities)
        big_opportunities = [opp for opp in opportunities if opp.max_investment >= 1000]
        
        print(f"\n📊 Enhanced Analysis:")
        print(f"🏆 Best Spread: {best_opp.token} on {best_opp.market[:30]} ({best_opp.spread_pct:.1f}% spread)")
        print(f"📈 Average Spread: {avg_spread:.1f}%")
        print(f"💰 Total Opportunities: {len(opportunities)}")
        print(f"🚀 High-Volume Opportunities ($1000+): {len(big_opportunities)}")
        
        # Show best ROI opportunity for $100 investment
        sorted_by_roi = sorted(opportunities, key=lambda x: x.roi_for_100, reverse=True)
        if sorted_by_roi and sorted_by_roi[0].can_invest_100:
            best_roi = sorted_by_roi[0]
            print(f"💎 Best $100 ROI: {best_roi.token} on {best_roi.market[:30]} ({best_roi.roi_for_100:.1f}% return)")
            
        # Scan timing info
        if self.last_scan_time:
            print(f"🕰 Last scan: {self.last_scan_time.strftime('%H:%M:%S')} (Scan #{self.scan_count})")
    
    def _print_tips_and_legends(self):
        """Print helpful tips and color legends"""
        print(f"\n💡 Tips & Legends:")
        print(f"   🟢 Green = Excellent (>5% spread + $100+ investable)")
        print(f"   🟡 Yellow = Good opportunities (>2% spread)")
        print(f"   ⚪ White = Modest opportunities")
        print(f"   💵 MaxUSD = Maximum investable at ask price")
        print(f"   🚀 Prof = Total profit from max investment")
        print(f"   💰 $100P = Expected profit on $100 investment")
        print(f"   📈 ROI% = Return percentage on $100 investment")

    def start_continuous_scan(self, delay: int = 30):
        """Start continuous spread scanning with enhanced features"""
        logger.info(f"Starting continuous spread scan (every {delay}s)...")
        print(f"🎯 Enhanced Spread Scanner - Continuous Mode (every {delay}s)")
        print(f"🚀 Press Ctrl+C to stop\n")
        
        try:
            while True:
                scan_start = time.time()
                print(f"\n🕐 Scan at {time.strftime('%H:%M:%S')}")
                
                # Perform market scan
                if self.scan_all_markets():
                    self.print_opportunities_table()
                else:
                    logger.error("Failed to scan markets, retrying next cycle...")
                
                scan_duration = time.time() - scan_start
                logger.info(f"Scan completed in {scan_duration:.2f}s")
                
                print(f"\n⏰ Waiting {delay}s before next scan...")
                time.sleep(delay)
                
        except KeyboardInterrupt:
            logger.info("Spread scanner stopped by user")
            print("\n🌙 Enhanced Spread Scanner stopped by user!")
        except Exception as e:
            logger.error(f"Unexpected error in continuous scan: {e}")
            print(f"\n❌ Unexpected error occurred: {e}")

def main():
    """Enhanced main function with better error handling and user interaction"""
    print("💰 Enhanced Polymarket Spread Scanner! 🌙")
    print(f"📈 Monitoring top {Config.MARKETS_TO_SCAN} markets by volume")
    print(f"🚫 Filtering spreads over {Config.MAX_SPREAD_FILTER}%")
    print(f"💵 Minimum investment threshold: ${Config.MIN_INVESTMENT_THRESHOLD}")
    print("=" * 60)
    
    try:
        # Initialize scanner with configuration
        scanner = SpreadScanner()
        
        # Load top markets
        print("\n🔍 Loading top markets from Polymarket API...")
        if not scanner.load_top_markets():
            print("❌ Failed to load markets! Check your internet connection.")
            return 1
        
        if not scanner.markets_to_scan:
            print("❌ No valid markets found to scan!")
            return 1
        
        print(f"\n✅ Successfully loaded {len(scanner.markets_to_scan)} markets")
        print("💡 Looking for profitable bid-ask spread opportunities...")
        
        # Perform initial scan
        print("\n🔍 Performing initial market scan...")
        scanner.scan_all_markets()
        scanner.print_opportunities_table()
        
        # Ask user if they want continuous monitoring
        print("\n" + "=" * 60)
        response = input("🚀 Start continuous monitoring? (y/n): ").lower().strip()
        
        if response in ('y', 'yes'):
            delay = input("\n⏰ Scan interval in seconds (default 30): ").strip()
            try:
                delay = int(delay) if delay else 30
                delay = max(10, delay)  # Minimum 10 seconds
            except ValueError:
                delay = 30
                
            scanner.start_continuous_scan(delay)
        else:
            print("🌙 Single scan completed. Goodbye!")
            
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"\n❌ Fatal error occurred: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 