"""
Bitfinex New Token Scanner

This script monitors Bitfinex for new token listings and trading pair additions.
Since Bitfinex is a centralized exchange, we monitor their API for new trading 
pairs rather than blockchain events.
"""

import asyncio
import json
import csv
import aiohttp
import os
import logging
import platform
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from Day_26_Projects.bitfinex_config import API_KEY, API_SECRET
except ImportError:
    logger.warning("config_bitfinex not found, using empty API credentials")
    API_KEY = ""
    API_SECRET = ""

# Bitfinex API URLs
BITFINEX_API_URL = "https://api.bitfinex.com"
BITFINEX_WS_URL = "wss://api.bitfinex.com/ws/2"

# CSV Configuration
DEFAULT_CSV_FILE = "./new_bitfinex_tokens.csv"
CSV_FILE = os.environ.get("BITFINEXSCANNER_CSV_FILE", DEFAULT_CSV_FILE)

# Monitoring Configuration
CHECK_INTERVAL = 300  # Check every 5 minutes for new listings
KNOWN_PAIRS_FILE = "./known_bitfinex_pairs.json"

def configure_event_loop():
    """Configure event loop for Windows compatibility."""
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logger.info("Configured Windows to use SelectorEventLoop")

class BitfinexScanner:
    def __init__(self):
        self.session = None
        self.known_pairs = set()
        self.last_check = datetime.utcnow() - timedelta(hours=24)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.load_known_pairs()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def load_known_pairs(self):
        """Load previously known trading pairs from file."""
        try:
            if os.path.exists(KNOWN_PAIRS_FILE):
                with open(KNOWN_PAIRS_FILE, 'r') as f:
                    data = json.load(f)
                    self.known_pairs = set(data.get('pairs', []))
                    if 'last_check' in data:
                        self.last_check = datetime.fromisoformat(data['last_check'])
                logger.info(f"Loaded {len(self.known_pairs)} known trading pairs")
            else:
                # Initialize with current pairs
                await self.initialize_known_pairs()
        except Exception as e:
            logger.error(f"Error loading known pairs: {e}")
            await self.initialize_known_pairs()
    
    async def save_known_pairs(self):
        """Save known trading pairs to file."""
        try:
            data = {
                'pairs': list(self.known_pairs),
                'last_check': self.last_check.isoformat()
            }
            with open(KNOWN_PAIRS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving known pairs: {e}")
    
    async def initialize_known_pairs(self):
        """Initialize known pairs with current symbols."""
        try:
            logger.info("Initializing known trading pairs...")
            symbols = await self.get_symbols()
            if symbols:
                # Filter for trading symbols (not funding symbols)
                trading_symbols = [s for s in symbols if not s.endswith('F0')]
                self.known_pairs = set(trading_symbols)
                logger.info(f"Initialized with {len(self.known_pairs)} trading pairs")
                await self.save_known_pairs()
        except Exception as e:
            logger.error(f"Error initializing known pairs: {e}")
    
    async def get_symbols(self):
        """Get current trading symbols from Bitfinex API."""
        try:
            async with self.session.get(f"{BITFINEX_API_URL}/v1/symbols") as response:
                if response.status == 200:
                    symbols = await response.json()
                    return symbols
                else:
                    logger.error(f"Failed to get symbols: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return None
    
    async def get_symbol_details(self):
        """Get detailed information about trading pairs."""
        try:
            async with self.session.get(f"{BITFINEX_API_URL}/v1/symbols_details") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get symbol details: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching symbol details: {e}")
            return None
    
    async def check_for_new_pairs(self):
        """Check for new trading pairs."""
        try:
            logger.info("Checking for new trading pairs...")
            current_symbols = await self.get_symbols()
            
            if not current_symbols:
                return []
            
            # Filter trading symbols only
            current_pairs = set([s for s in current_symbols if not s.endswith('F0')])
            
            new_pairs = current_pairs - self.known_pairs
            
            if new_pairs:
                logger.info(f"Found {len(new_pairs)} new trading pairs: {new_pairs}")
                
                # Get detailed info for new pairs
                symbol_details = await self.get_symbol_details()
                new_tokens = []
                
                for pair in new_pairs:
                    token_info = await self.process_new_pair(pair, symbol_details)
                    if token_info:
                        new_tokens.append(token_info)
                
                # Update known pairs
                self.known_pairs.update(new_pairs)
                self.last_check = datetime.utcnow()
                await self.save_known_pairs()
                
                return new_tokens
            else:
                logger.info("No new trading pairs found")
                self.last_check = datetime.utcnow()
                await self.save_known_pairs()
                return []
                
        except Exception as e:
            logger.error(f"Error checking for new pairs: {e}")
            return []
    
    async def process_new_pair(self, symbol, symbol_details):
        """Process a new trading pair and extract relevant information."""
        try:
            # Find details for this symbol
            details = None
            if symbol_details:
                for detail in symbol_details:
                    if detail.get('pair', '').lower() == symbol.lower():
                        details = detail
                        break
            
            # Extract base and quote currencies
            # Bitfinex symbols are like 'btcusd', 'ethusd', etc.
            if symbol.endswith('usd'):
                base_asset = symbol[:-3].upper()
                quote_asset = 'USD'
            elif symbol.endswith('usdt'):
                base_asset = symbol[:-4].upper()
                quote_asset = 'USDT'
            elif symbol.endswith('btc'):
                base_asset = symbol[:-3].upper()
                quote_asset = 'BTC'
            elif symbol.endswith('eth'):
                base_asset = symbol[:-3].upper()
                quote_asset = 'ETH'
            else:
                # Try to guess based on common patterns
                base_asset = symbol[:3].upper()
                quote_asset = symbol[3:].upper()
            
            # Get current ticker info
            ticker_info = await self.get_ticker_info(symbol)
            
            return {
                'symbol': symbol.upper(),
                'baseAsset': base_asset,
                'quoteAsset': quote_asset,
                'price': ticker_info.get('last_price') if ticker_info else 'N/A',
                'volume': ticker_info.get('volume') if ticker_info else 'N/A',
                'high': ticker_info.get('high') if ticker_info else 'N/A',
                'low': ticker_info.get('low') if ticker_info else 'N/A',
                'minimum_order_size': details.get('minimum_order_size') if details else 'N/A',
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error processing new pair {symbol}: {e}")
            return None
    
    async def get_ticker_info(self, symbol):
        """Get current ticker info for a symbol."""
        try:
            async with self.session.get(f"{BITFINEX_API_URL}/v1/pubticker/{symbol}") as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.debug(f"Could not get ticker info for {symbol}: {e}")
            return None

def ensure_csv_file_exists():
    """Ensure the CSV file exists with proper headers."""
    try:
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            with open(CSV_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Symbol", "Base Asset", "Quote Asset", "Price", 
                    "Volume 24h", "High 24h", "Low 24h", "Min Order Size",
                    "Time Found", "Epoch Time", "Bitfinex Link", 
                    "CoinGecko Link", "CoinMarketCap Link"
                ])
            logger.info(f"Created CSV file: {CSV_FILE}")
    except IOError as e:
        logger.error(f"Error creating CSV file: {e}")

def save_to_csv(token_info):
    """Save new token information to CSV."""
    try:
        with open(CSV_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            
            symbol = token_info['symbol']
            base_asset = token_info['baseAsset']
            time_found = token_info['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            epoch_time = int(token_info['timestamp'].timestamp())
            
            bitfinex_link = f"https://trading.bitfinex.com/t/{symbol}"
            coingecko_link = f"https://www.coingecko.com/en/coins/{base_asset.lower()}"
            coinmarketcap_link = f"https://coinmarketcap.com/currencies/{base_asset.lower()}/"
            
            writer.writerow([
                symbol,
                base_asset,
                token_info['quoteAsset'],
                token_info['price'],
                token_info['volume'],
                token_info['high'],
                token_info['low'],
                token_info['minimum_order_size'],
                time_found,
                epoch_time,
                bitfinex_link,
                coingecko_link,
                coinmarketcap_link
            ])
            
        logger.info(f"Saved new token: {symbol} ({base_asset})")
    except IOError as e:
        logger.error(f"Error writing to CSV: {e}")

async def heartbeat():
    """Periodic heartbeat to confirm the script is running."""
    counter = 0
    while True:
        await asyncio.sleep(60)  # Every minute
        counter += 1
        logger.info(f"Bitfinex Scanner running... (heartbeat #{counter})")

async def main():
    """Main function to run the scanner."""
    configure_event_loop()
    ensure_csv_file_exists()
    
    logger.info(f"Starting Bitfinex Token Scanner")
    logger.info(f"Output CSV: {CSV_FILE}")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
    
    async with BitfinexScanner() as scanner:
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat())
        
        try:
            while True:
                try:
                    # Check for new trading pairs
                    new_tokens = await scanner.check_for_new_pairs()
                    
                    # Save any new tokens found
                    for token_info in new_tokens:
                        save_to_csv(token_info)
                        logger.info(f"New token detected: {token_info['baseAsset']} "
                                  f"({token_info['symbol']}) - Price: {token_info['price']}")
                    
                    # Wait for next check
                    logger.debug(f"Waiting {CHECK_INTERVAL} seconds before next check...")
                    await asyncio.sleep(CHECK_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    logger.info("Continuing after error...")
                    await asyncio.sleep(30)  # Wait 30 seconds after error
                    
        except KeyboardInterrupt:
            logger.info("Scanner stopped by user")
        finally:
            heartbeat_task.cancel()
            logger.info("Scanner shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
