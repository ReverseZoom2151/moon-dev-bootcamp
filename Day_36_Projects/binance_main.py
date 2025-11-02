"""Monitors target Twitter accounts for cryptocurrency mentions and new token listings
and executes buy orders via Binance Spot API when opportunities are detected.
"""

import pandas as pd
import time
import logging
import sys
import os
import re
from typing import List
from datetime import datetime

# Patch httpx before importing twikit - Required for certain environments/proxies
try:
    import httpx
    original_client = httpx.Client
    def patched_client(*args, **kwargs):
        # Attempt to remove proxy, handle if not present
        kwargs.pop('proxy', None)
        return original_client(*args, **kwargs)
    httpx.Client = patched_client
    logging.debug("httpx.Client patched to ignore proxy settings.")
except ImportError:
    logging.warning("httpx not found, skipping patch. Twikit might have issues with proxies.")
except Exception as e:
    logging.warning(f"Failed to patch httpx: {e}")

# Now import twikit and other dependencies
try:
    from twikit import Client as TwitterClient
    from twikit.errors import TooManyRequests
except ImportError:
    logging.error("twikit library not found. Please install it: pip install twikit")
    sys.exit(1)

try:
    import schedule
except ImportError:
    logging.error("schedule library not found. Please install it: pip install schedule")
    sys.exit(1)

# Local imports
import config
import dontshare as ds
from binance_utils import market_buy, get_current_price, validate_symbol_format, search_new_listings

# --- Logging Setup ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binance_twitter_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
TWEET_LOG_FILE = "binance_tweet_log.csv"
TARGET_ACCOUNTS = config.TARGET_ACCOUNTS  # List of Twitter usernames to monitor
CHECK_INTERVAL_MINUTES = config.CHECK_INTERVAL_MINUTES  # How often to check for new tweets
USDT_BUY_AMOUNT = config.USDT_SIZE  # Amount of USDT to spend per buy
MAX_SLIPPAGE = config.MAX_SLIPPAGE  # Maximum acceptable slippage for market orders

# Track processed tweets to avoid duplicates
processed_tweets = set()
tweet_log_df = pd.DataFrame()

# --- Regex Patterns for Token Detection ---
# Patterns to detect potential token symbols and trading opportunities
TOKEN_PATTERNS = [
    r'\$([A-Z]{2,6})\b',  # $BTC, $ETH pattern
    r'\b([A-Z]{2,6})USDT\b',  # BTCUSDT pattern  
    r'\b([A-Z]{2,6})/USDT\b',  # BTC/USDT pattern
    r'#([A-Za-z0-9]{2,10})',  # Hashtag tokens
]

# Keywords that suggest new listing or trading opportunity
OPPORTUNITY_KEYWORDS = [
    'new listing', 'now live', 'trading starts', 'available now',
    'just listed', 'spot trading', 'binance listing', 'trading pair',
    'pump', 'moon', 'gem', 'breakout', 'bullish', 'buy signal'
]

def load_tweet_log():
    """Loads the tweet log CSV if it exists."""
    global tweet_log_df, processed_tweets
    
    try:
        if os.path.exists(TWEET_LOG_FILE):
            tweet_log_df = pd.read_csv(TWEET_LOG_FILE)
            processed_tweets = set(tweet_log_df['tweet_id'].astype(str))
            logger.info(f"📊 Loaded {len(tweet_log_df)} processed tweets from log")
        else:
            tweet_log_df = pd.DataFrame(columns=[
                'timestamp', 'tweet_id', 'username', 'content', 
                'detected_tokens', 'action_taken', 'buy_success'
            ])
            logger.info("📊 Created new tweet log")
    except Exception as e:
        logger.error(f"❌ Failed to load tweet log: {e}")
        tweet_log_df = pd.DataFrame()
        processed_tweets = set()

def save_tweet_log(new_data):
    """Appends new data to the tweet log DataFrame and saves to CSV."""
    global tweet_log_df
    
    try:
        new_row = pd.DataFrame([new_data])
        tweet_log_df = pd.concat([tweet_log_df, new_row], ignore_index=True)
        tweet_log_df.to_csv(TWEET_LOG_FILE, index=False)
        logger.debug(f"💾 Saved tweet log with {len(tweet_log_df)} entries")
    except Exception as e:
        logger.error(f"❌ Failed to save tweet log: {e}")

def initialize_twitter_client():
    """Initializes and authenticates the Twitter client."""
    try:
        client = TwitterClient('en-US')
        
        # Check if we have credentials
        if not hasattr(ds, 'twitter_username') or not hasattr(ds, 'twitter_email') or not hasattr(ds, 'twitter_password'):
            logger.error("❌ Twitter credentials not found in dontshare.py")
            return None
        
        logger.info("🔐 Logging into Twitter...")
        client.login(
            auth_info_1=ds.twitter_username,
            auth_info_2=ds.twitter_email, 
            password=ds.twitter_password
        )
        
        logger.info("✅ Twitter authentication successful")
        return client
        
    except Exception as e:
        logger.error(f"❌ Twitter authentication failed: {e}")
        return None

def extract_tokens_from_text(text: str) -> List[str]:
    """Extract potential token symbols from tweet text."""
    tokens = set()
    text_upper = text.upper()
    
    # Apply regex patterns
    for pattern in TOKEN_PATTERNS:
        matches = re.findall(pattern, text_upper)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]  # Extract from tuple if regex has groups
            if len(match) >= 2 and len(match) <= 6:
                # Add USDT suffix for Binance trading pairs
                if not match.endswith('USDT'):
                    tokens.add(f"{match}USDT")
                else:
                    tokens.add(match)
    
    return list(tokens)

def check_opportunity_keywords(text: str) -> bool:
    """Check if tweet contains opportunity keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in OPPORTUNITY_KEYWORDS)

def execute_buy_strategy(symbol: str) -> bool:
    """Attempts to buy the token aggressively based on config settings."""
    try:
        logger.info(f"🎯 Executing buy strategy for {symbol}")
        
        # Validate symbol format
        if not validate_symbol_format(symbol):
            logger.warning(f"⚠️ Invalid symbol format: {symbol}")
            return False
        
        # Get current price for validation
        current_price = get_current_price(symbol)
        if not current_price:
            logger.warning(f"⚠️ Could not get price for {symbol} - may not be listed")
            return False
        
        logger.info(f"💰 Current price for {symbol}: {current_price} USDT")
        
        # Execute market buy
        logger.info(f"🚀 Executing market buy: {USDT_BUY_AMOUNT} USDT worth of {symbol}")
        
        order_result = market_buy(symbol, USDT_BUY_AMOUNT, MAX_SLIPPAGE)
        
        if order_result:
            logger.info(f"✅ Successfully bought {symbol}!")
            logger.info(f"📋 Order ID: {order_result.get('orderId')}")
            logger.info(f"📋 Executed Qty: {order_result.get('executedQty')}")
            return True
        else:
            logger.error(f"❌ Failed to execute buy for {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Buy strategy failed for {symbol}: {e}")
        return False

def analyze_tweet_sentiment(text: str) -> str:
    """Simple sentiment analysis based on keywords."""
    bullish_words = ['moon', 'pump', 'bull', 'bullish', 'buy', 'gem', 'rocket', '🚀', '📈']
    bearish_words = ['dump', 'bear', 'bearish', 'sell', 'crash', 'drop', '📉']
    
    text_lower = text.lower()
    
    bullish_count = sum(1 for word in bullish_words if word in text_lower)
    bearish_count = sum(1 for word in bearish_words if word in text_lower)
    
    if bullish_count > bearish_count:
        return 'bullish'
    elif bearish_count > bullish_count:
        return 'bearish'
    else:
        return 'neutral'

def check_new_tweets(client: TwitterClient):
    """Checks for new tweets from target accounts and triggers actions."""
    global processed_tweets
    
    try:
        logger.info(f"🔍 Checking tweets from {len(TARGET_ACCOUNTS)} accounts...")
        
        for username in TARGET_ACCOUNTS:
            try:
                logger.debug(f"🔍 Checking tweets from @{username}")
                
                # Get user and their recent tweets
                user = client.get_user_by_screen_name(username)
                tweets = client.get_user_tweets(user.id, count=10)  # Check last 10 tweets
                
                new_tweets_count = 0
                
                for tweet in tweets:
                    tweet_id = str(tweet.id)
                    
                    # Skip if already processed
                    if tweet_id in processed_tweets:
                        continue
                    
                    new_tweets_count += 1
                    tweet_text = tweet.text
                    tweet_time = tweet.created_at
                    
                    logger.info(f"📱 New tweet from @{username}: {tweet_text[:100]}...")
                    
                    # Extract potential tokens
                    detected_tokens = extract_tokens_from_text(tweet_text)
                    has_opportunity = check_opportunity_keywords(tweet_text)
                    sentiment = analyze_tweet_sentiment(tweet_text)
                    
                    action_taken = "none"
                    buy_success = False
                    
                    # Decide whether to take action
                    if detected_tokens and (has_opportunity or sentiment == 'bullish'):
                        logger.info(f"🎯 Opportunity detected! Tokens: {detected_tokens}")
                        logger.info(f"📊 Has opportunity keywords: {has_opportunity}")
                        logger.info(f"📊 Sentiment: {sentiment}")
                        
                        # Try to buy the first valid token
                        for token in detected_tokens:
                            if execute_buy_strategy(token):
                                action_taken = f"bought_{token}"
                                buy_success = True
                                break
                            else:
                                action_taken = f"attempted_{token}"
                        
                        if buy_success:
                            logger.info(f"✅ Successfully executed buy based on @{username}'s tweet!")
                        else:
                            logger.warning(f"⚠️ Failed to execute any buys from @{username}'s tweet")
                    
                    elif detected_tokens:
                        logger.info(f"📊 Tokens detected but no strong signal: {detected_tokens}")
                        action_taken = f"detected_{','.join(detected_tokens)}"
                    
                    # Log the tweet
                    tweet_data = {
                        'timestamp': datetime.now().isoformat(),
                        'tweet_id': tweet_id,
                        'username': username,
                        'content': tweet_text,
                        'detected_tokens': ','.join(detected_tokens),
                        'action_taken': action_taken,
                        'buy_success': buy_success
                    }
                    
                    save_tweet_log(tweet_data)
                    processed_tweets.add(tweet_id)
                
                if new_tweets_count > 0:
                    logger.info(f"📊 Processed {new_tweets_count} new tweets from @{username}")
                else:
                    logger.debug(f"📊 No new tweets from @{username}")
                
                # Small delay between users to avoid rate limits
                time.sleep(2)
                
            except TooManyRequests as e:
                logger.warning(f"⚠️ Rate limited while checking @{username}. Waiting...")
                time.sleep(60)  # Wait 1 minute
                
            except Exception as e:
                logger.error(f"❌ Error checking tweets from @{username}: {e}")
                continue
        
        logger.info("✅ Tweet check completed")
        
    except Exception as e:
        logger.error(f"❌ Error in tweet checking: {e}")

def check_new_listings():
    """Check Binance for new listings and potential opportunities."""
    try:
        logger.info("🔍 Checking for new Binance listings...")
        
        new_pairs = search_new_listings()
        usdt_pairs = [pair['symbol'] for pair in new_pairs if pair['symbol'].endswith('USDT')]
        
        logger.info(f"📊 Found {len(usdt_pairs)} USDT trading pairs")
        
        # You could implement logic here to buy newly listed tokens
        # For now, just log them for monitoring
        for pair in usdt_pairs[-10:]:  # Log last 10
            logger.debug(f"📊 Available pair: {pair}")
            
    except Exception as e:
        logger.error(f"❌ Error checking new listings: {e}")

def main_loop():
    """Main operational loop, scheduled to run periodically."""
    logger.info("🔄 Running main monitoring loop...")
    
    try:
        # Initialize Twitter client
        client = initialize_twitter_client()
        if not client:
            logger.error("❌ Could not initialize Twitter client")
            return
        
        # Check tweets
        check_new_tweets(client)
        
        # Check new listings (optional)
        check_new_listings()
        
        logger.info("✅ Main loop completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in main loop: {e}")

if __name__ == "__main__":
    logger.info(f"🚀 Binance Twitter Monitor Initializing..." )
    logger.info(f"   Configuration: Max Slippage={MAX_SLIPPAGE*100}%, Buy Amount={USDT_BUY_AMOUNT} USDT" )
    logger.info(f"   Target Accounts: {TARGET_ACCOUNTS}")
    logger.info(f"   Check Interval: {CHECK_INTERVAL_MINUTES} minutes")
    
    try:
        # Load existing tweet log
        load_tweet_log()
        
        # Schedule periodic checks
        schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(main_loop)
        
        logger.info("🎯 Binance Twitter monitoring started! Press Ctrl+C to stop.")
        logger.info("📊 Monitoring for token mentions and trading opportunities...")
        
        # Initial run
        main_loop()
        
        # Keep running scheduled tasks
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        logger.info("🛑 Binance Twitter monitor stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
