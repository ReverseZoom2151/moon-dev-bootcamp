"""
Binance New Token Scanner

This script monitors Binance for new token listings and trading pair additions.
Since Binance is a centralized exchange, we monitor their announcements and API
for new trading pairs rather than blockchain events.
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
    from Day_26_Projects.binance_config import API_KEY, API_SECRET
except ImportError:
    logger.warning("config_binance not found, using empty API credentials")
    API_KEY = ""
    API_SECRET = ""

# Binance API URLs
BINANCE_API_URL = "https://api.binance.com/api/v3"
BINANCE_STREAM_URL = "wss://stream.binance.com:9443/ws"

# CSV Configuration
DEFAULT_CSV_FILE = "./new_binance_tokens.csv"
CSV_FILE = os.environ.get("BINANCESCANNER_CSV_FILE", DEFAULT_CSV_FILE)

# Monitoring Configuration
CHECK_INTERVAL = 300  # Check every 5 minutes for new listings
KNOWN_PAIRS_FILE = "./known_binance_pairs.json"

def configure_event_loop():
    """Configure event loop for Windows compatibility."""
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logger.info("Configured Windows to use SelectorEventLoop")

class BinanceScanner:
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
        """Initialize known pairs with current exchange info."""
        try:
            logger.info("Initializing known trading pairs...")
            exchange_info = await self.get_exchange_info()
            if exchange_info:
                current_pairs = {symbol['symbol'] for symbol in exchange_info.get('symbols', [])}
                self.known_pairs = current_pairs
                logger.info(f"Initialized with {len(self.known_pairs)} trading pairs")
                await self.save_known_pairs()
        except Exception as e:
            logger.error(f"Error initializing known pairs: {e}")
    
    async def get_exchange_info(self):
        """Get current exchange info from Binance API."""
        try:
            async with self.session.get(f"{BINANCE_API_URL}/exchangeInfo") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get exchange info: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return None
    
    async def check_for_new_pairs(self):
        """Check for new trading pairs."""
        try:
            logger.info("Checking for new trading pairs...")
            exchange_info = await self.get_exchange_info()
            
            if not exchange_info:
                return []
            
            current_pairs = {symbol['symbol'] for symbol in exchange_info.get('symbols', [])
                           if symbol.get('status') == 'TRADING'}
            
            new_pairs = current_pairs - self.known_pairs
            
            if new_pairs:
                logger.info(f"Found {len(new_pairs)} new trading pairs: {new_pairs}")
                
                # Get detailed info for new pairs
                new_tokens = []
                for pair in new_pairs:
                    for symbol_info in exchange_info['symbols']:
                        if symbol_info['symbol'] == pair:
                            token_info = await self.process_new_pair(symbol_info)
                            if token_info:
                                new_tokens.append(token_info)
                            break
                
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
    
    async def process_new_pair(self, symbol_info):
        """Process a new trading pair and extract relevant information."""
        try:
            symbol = symbol_info['symbol']
            base_asset = symbol_info['baseAsset']
            quote_asset = symbol_info['quoteAsset']
            
            # Skip if it's just a new quote asset pairing for existing token
            if quote_asset not in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']:
                return None
            
            # Get current price
            price_info = await self.get_price_info(symbol)
            
            return {
                'symbol': symbol,
                'baseAsset': base_asset,
                'quoteAsset': quote_asset,
                'status': symbol_info.get('status'),
                'price': price_info.get('price') if price_info else 'N/A',
                'volume': price_info.get('volume') if price_info else 'N/A',
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error processing new pair {symbol_info.get('symbol', 'unknown')}: {e}")
            return None
    
    async def get_price_info(self, symbol):
        """Get current price and volume info for a symbol."""
        try:
            async with self.session.get(f"{BINANCE_API_URL}/ticker/24hr", 
                                      params={'symbol': symbol}) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.debug(f"Could not get price info for {symbol}: {e}")
            return None
    
    async def monitor_announcements(self):
        """Monitor Binance announcements for new listings (simplified version)."""
        # Note: This is a basic implementation. For production use, you might want to 
        # monitor Binance's official announcement channels or RSS feeds
        try:
            # Check if there are any announcements in the last 24 hours
            # This is a placeholder - you could integrate with Binance's announcement API
            # or scrape their announcement page if available
            logger.debug("Checking for announcements...")
            # Placeholder implementation
            return []
        except Exception as e:
            logger.error(f"Error monitoring announcements: {e}")
            return []

def ensure_csv_file_exists():
    """Ensure the CSV file exists with proper headers."""
    try:
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            with open(CSV_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Symbol", "Base Asset", "Quote Asset", "Status", 
                    "Price", "Volume 24h", "Time Found", "Epoch Time", 
                    "Binance Link", "CoinGecko Link", "CoinMarketCap Link"
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
            
            binance_link = f"https://www.binance.com/en/trade/{symbol}"
            coingecko_link = f"https://www.coingecko.com/en/coins/{base_asset.lower()}"
            coinmarketcap_link = f"https://coinmarketcap.com/currencies/{base_asset.lower()}/"
            
            writer.writerow([
                symbol,
                base_asset,
                token_info['quoteAsset'],
                token_info['status'],
                token_info['price'],
                token_info['volume'],
                time_found,
                epoch_time,
                binance_link,
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
        logger.info(f"Binance Scanner running... (heartbeat #{counter})")

async def main():
    """Main function to run the scanner."""
    configure_event_loop()
    ensure_csv_file_exists()
    
    logger.info(f"Starting Binance Token Scanner")
    logger.info(f"Output CSV: {CSV_FILE}")
    logger.info(f"Check interval: {CHECK_INTERVAL} seconds")
    
    async with BinanceScanner() as scanner:
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
