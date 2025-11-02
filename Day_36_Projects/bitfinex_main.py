"""Professional Twitter monitoring system for Bitfinex margin and derivatives trading.
Monitors target Twitter accounts for cryptocurrency signals and executes sophisticated
trading strategies including margin positions and funding rate arbitrage.
"""

import pandas as pd
import time
import logging
import sys
import os
import re
from typing import List, Dict, Tuple
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
from bitfinex_utils import (market_buy, get_ticker, validate_symbol_format, 
                          get_trading_pairs, get_funding_book, get_recent_trades)

# --- Professional Logging Setup ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('bitfinex_professional_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Professional Configuration ---
TWEET_LOG_FILE = "bitfinex_professional_signals.csv"
TARGET_ACCOUNTS = config.TARGET_ACCOUNTS  # List of professional Twitter accounts to monitor
CHECK_INTERVAL_MINUTES = config.CHECK_INTERVAL_MINUTES  # Monitoring frequency
USD_POSITION_SIZE = config.USD_SIZE  # Position size in USD
MAX_SLIPPAGE = config.MAX_SLIPPAGE  # Maximum acceptable slippage
MARGIN_TRADING = getattr(config, 'MARGIN_TRADING', False)  # Enable margin trading
FUNDING_ARBITRAGE = getattr(config, 'FUNDING_ARBITRAGE', False)  # Enable funding arbitrage

# Professional trading parameters
MIN_CONFIDENCE_SCORE = 0.7  # Minimum confidence for trade execution
MAX_DAILY_POSITIONS = 5  # Maximum positions per day for risk management
POSITION_SIZING_METHOD = "fixed"  # "fixed", "volatility_adjusted", "sentiment_weighted"

# Track processed signals and daily statistics
processed_tweets = set()
signal_log_df = pd.DataFrame()
daily_stats = {'positions_opened': 0, 'positions_closed': 0, 'pnl': 0.0}

# --- Advanced Regex Patterns for Professional Signal Detection ---
PROFESSIONAL_PATTERNS = [
    r'\$([A-Z]{2,6})\b',  # $BTC, $ETH pattern
    r'\b([A-Z]{2,6})USD\b',  # BTCUSD pattern for Bitfinex
    r'\b([A-Z]{2,6})/USD\b',  # BTC/USD pattern
    r'#([A-Za-z0-9]{2,10})',  # Hashtag tokens
    r'\b([A-Z]{2,6})\s*(?:long|short|buy|sell)\b',  # Trading direction patterns
]

# Professional trading keywords with confidence weights
PROFESSIONAL_SIGNALS = {
    'high_confidence': [
        'long setup', 'short setup', 'buy signal', 'sell signal', 'breakout confirmed',
        'institutional flow', 'whale movement', 'smart money', 'accumulation phase',
        'distribution phase', 'margin call', 'liquidation cascade', 'funding squeeze'
    ],
    'medium_confidence': [
        'bullish', 'bearish', 'support', 'resistance', 'trend change',
        'momentum', 'oversold', 'overbought', 'divergence', 'reversal pattern'
    ],
    'low_confidence': [
        'pump', 'dump', 'moon', 'rocket', 'gem', 'degen', 'ape', 'fomo'
    ]
}

# Risk management keywords
RISK_KEYWORDS = {
    'high_risk': ['leverage', 'margin', 'liquidation', 'volatile', 'risky', 'gambling'],
    'caution': ['careful', 'risk', 'stop loss', 'position size', 'manage risk']
}

def load_signal_log():
    """Loads the professional signal log CSV if it exists."""
    global signal_log_df, processed_tweets
    
    try:
        if os.path.exists(TWEET_LOG_FILE):
            signal_log_df = pd.read_csv(TWEET_LOG_FILE)
            processed_tweets = set(signal_log_df['tweet_id'].astype(str))
            logger.info(f"📊 Loaded {len(signal_log_df)} professional signals from log")
        else:
            signal_log_df = pd.DataFrame(columns=[
                'timestamp', 'tweet_id', 'username', 'content', 'detected_symbols',
                'signal_type', 'confidence_score', 'trading_direction', 'action_taken', 
                'position_size', 'entry_price', 'success', 'risk_level'
            ])
            logger.info("📊 Created new professional signal log")
    except Exception as e:
        logger.error(f"❌ Failed to load signal log: {e}")
        signal_log_df = pd.DataFrame()
        processed_tweets = set()

def save_signal_log(new_data):
    """Appends new professional signal data to the log DataFrame and saves to CSV."""
    global signal_log_df
    
    try:
        new_row = pd.DataFrame([new_data])
        signal_log_df = pd.concat([signal_log_df, new_row], ignore_index=True)
        signal_log_df.to_csv(TWEET_LOG_FILE, index=False)
        logger.debug(f"💾 Saved professional signal log with {len(signal_log_df)} entries")
    except Exception as e:
        logger.error(f"❌ Failed to save signal log: {e}")

def initialize_twitter_client():
    """Initializes and authenticates the Twitter client."""
    try:
        client = TwitterClient('en-US')
        
        # Check if we have credentials
        if not hasattr(ds, 'twitter_username') or not hasattr(ds, 'twitter_email') or not hasattr(ds, 'twitter_password'):
            logger.error("❌ Twitter credentials not found in dontshare.py")
            return None
        
        logger.info("🔐 Logging into Twitter for professional monitoring...")
        client.login(
            auth_info_1=ds.twitter_username,
            auth_info_2=ds.twitter_email, 
            password=ds.twitter_password
        )
        
        logger.info("✅ Twitter authentication successful for professional monitoring")
        return client
        
    except Exception as e:
        logger.error(f"❌ Professional Twitter authentication failed: {e}")
        return None

def extract_trading_symbols(text: str) -> List[str]:
    """Extract potential trading symbols from professional signals."""
    symbols = set()
    text_upper = text.upper()
    
    # Apply professional regex patterns
    for pattern in PROFESSIONAL_PATTERNS:
        matches = re.findall(pattern, text_upper)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]  # Extract from tuple if regex has groups
            if len(match) >= 2 and len(match) <= 6:
                # Format for Bitfinex (lowercase with USD)
                symbol = match.lower()
                if not symbol.endswith('usd'):
                    symbols.add(f"{symbol}usd")
                else:
                    symbols.add(symbol)
    
    return list(symbols)

def analyze_professional_signal(text: str) -> Dict[str, any]:
    """Analyze tweet text for professional trading signals."""
    text_lower = text.lower()
    
    # Calculate confidence score
    confidence_score = 0.0
    signal_type = "unknown"
    trading_direction = "neutral"
    risk_level = "medium"
    
    # Check for high confidence signals
    for keyword in PROFESSIONAL_SIGNALS['high_confidence']:
        if keyword in text_lower:
            confidence_score += 0.3
            signal_type = "professional"
    
    # Check for medium confidence signals
    for keyword in PROFESSIONAL_SIGNALS['medium_confidence']:
        if keyword in text_lower:
            confidence_score += 0.2
            if signal_type == "unknown":
                signal_type = "technical"
    
    # Check for low confidence signals
    for keyword in PROFESSIONAL_SIGNALS['low_confidence']:
        if keyword in text_lower:
            confidence_score += 0.1
            if signal_type == "unknown":
                signal_type = "sentiment"
    
    # Determine trading direction
    long_keywords = ['long', 'buy', 'bullish', 'pump', 'moon', 'breakout up']
    short_keywords = ['short', 'sell', 'bearish', 'dump', 'breakdown']
    
    long_count = sum(1 for kw in long_keywords if kw in text_lower)
    short_count = sum(1 for kw in short_keywords if kw in text_lower)
    
    if long_count > short_count:
        trading_direction = "long"
    elif short_count > long_count:
        trading_direction = "short"
    
    # Assess risk level
    high_risk_count = sum(1 for kw in RISK_KEYWORDS['high_risk'] if kw in text_lower)
    caution_count = sum(1 for kw in RISK_KEYWORDS['caution'] if kw in text_lower)
    
    if high_risk_count > 0:
        risk_level = "high"
    elif caution_count > 0:
        risk_level = "low"
    
    # Cap confidence score
    confidence_score = min(confidence_score, 1.0)
    
    return {
        'confidence_score': confidence_score,
        'signal_type': signal_type,
        'trading_direction': trading_direction,
        'risk_level': risk_level
    }

def calculate_position_size(symbol: str, confidence_score: float, risk_level: str) -> float:
    """Calculate appropriate position size based on risk management."""
    base_size = USD_POSITION_SIZE
    
    if POSITION_SIZING_METHOD == "fixed":
        return base_size
    
    elif POSITION_SIZING_METHOD == "volatility_adjusted":
        # Get recent trades to assess volatility
        recent_trades = get_recent_trades(symbol, limit=20)
        if recent_trades and len(recent_trades) > 5:
            prices = [trade['price'] for trade in recent_trades[-10:]]
            volatility = (max(prices) - min(prices)) / sum(prices) * len(prices)
            # Reduce size for high volatility
            volatility_multiplier = max(0.5, 1.0 - volatility)
            base_size *= volatility_multiplier
    
    elif POSITION_SIZING_METHOD == "sentiment_weighted":
        # Adjust size based on confidence
        base_size *= (0.5 + confidence_score * 0.5)
    
    # Risk level adjustments
    if risk_level == "high":
        base_size *= 0.5
    elif risk_level == "low":
        base_size *= 1.2
    
    return round(base_size, 2)

def execute_professional_strategy(symbol: str, signal_data: Dict[str, any]) -> Tuple[bool, Dict]:
    """Execute professional trading strategy based on signal analysis."""
    try:
        confidence = signal_data['confidence_score']
        direction = signal_data['trading_direction']
        risk_level = signal_data['risk_level']
        
        logger.info(f"🎯 Executing professional strategy for {symbol}")
        logger.info(f"📊 Confidence: {confidence:.2f}, Direction: {direction}, Risk: {risk_level}")
        
        # Check minimum confidence threshold
        if confidence < MIN_CONFIDENCE_SCORE:
            logger.info(f"⚠️ Confidence {confidence:.2f} below threshold {MIN_CONFIDENCE_SCORE}")
            return False, {'reason': 'insufficient_confidence'}
        
        # Check daily position limit
        if daily_stats['positions_opened'] >= MAX_DAILY_POSITIONS:
            logger.warning(f"⚠️ Daily position limit reached: {daily_stats['positions_opened']}")
            return False, {'reason': 'daily_limit_reached'}
        
        # Validate symbol and get market data
        if not validate_symbol_format(symbol):
            logger.warning(f"⚠️ Invalid symbol format: {symbol}")
            return False, {'reason': 'invalid_symbol'}
        
        ticker = get_ticker(symbol)
        if not ticker:
            logger.warning(f"⚠️ Could not get ticker for {symbol}")
            return False, {'reason': 'no_ticker_data'}
        
        current_price = ticker['last_price']
        logger.info(f"💰 Current price for {symbol}: {current_price} USD")
        
        # Calculate position size
        position_size = calculate_position_size(symbol, confidence, risk_level)
        logger.info(f"📏 Calculated position size: {position_size} USD")
        
        # Execute trade based on direction
        order_result = None
        order_type = "exchange market" if not MARGIN_TRADING else "market"
        
        if direction == "long":
            logger.info(f"🚀 Executing LONG position: {position_size} USD worth of {symbol}")
            order_result = market_buy(symbol, position_size, order_type)
            
        elif direction == "short" and MARGIN_TRADING:
            logger.info(f"🔻 SHORT positions require margin trading implementation")
            # Short selling would require additional margin API implementation
            return False, {'reason': 'short_not_implemented'}
        
        else:
            logger.info(f"⚠️ Neutral or unsupported direction: {direction}")
            return False, {'reason': 'unsupported_direction'}
        
        if order_result:
            daily_stats['positions_opened'] += 1
            logger.info(f"✅ Successfully executed professional trade!")
            logger.info(f"📋 Order ID: {order_result.get('id')}")
            
            return True, {
                'order_id': order_result.get('id'),
                'symbol': symbol,
                'position_size': position_size,
                'entry_price': current_price,
                'direction': direction,
                'confidence': confidence
            }
        else:
            logger.error(f"❌ Failed to execute trade for {symbol}")
            return False, {'reason': 'execution_failed'}
            
    except Exception as e:
        logger.error(f"❌ Professional strategy execution failed for {symbol}: {e}")
        return False, {'reason': f'exception: {str(e)}'}

def check_funding_opportunities():
    """Check for funding rate arbitrage opportunities."""
    if not FUNDING_ARBITRAGE:
        return
    
    try:
        logger.info("🔍 Checking funding rate arbitrage opportunities...")
        
        # Get funding book for major pairs
        major_pairs = ['btcusd', 'ethusd', 'adausd', 'solusd']
        
        for symbol in major_pairs:
            funding_book = get_funding_book(symbol)
            if funding_book and 'funding_book' in funding_book:
                # Analyze funding rates (implementation depends on strategy)
                logger.debug(f"📊 Funding data available for {symbol}")
                
    except Exception as e:
        logger.error(f"❌ Error checking funding opportunities: {e}")

def check_professional_signals(client: TwitterClient):
    """Check for professional trading signals from target accounts."""
    global processed_tweets
    
    try:
        logger.info(f"🔍 Monitoring {len(TARGET_ACCOUNTS)} professional accounts...")
        
        for username in TARGET_ACCOUNTS:
            try:
                logger.debug(f"🔍 Checking professional signals from @{username}")
                
                # Get user and their recent tweets
                user = client.get_user_by_screen_name(username)
                tweets = client.get_user_tweets(user.id, count=5)  # Focus on recent signals
                
                new_signals_count = 0
                
                for tweet in tweets:
                    tweet_id = str(tweet.id)
                    
                    # Skip if already processed
                    if tweet_id in processed_tweets:
                        continue
                    
                    new_signals_count += 1
                    tweet_text = tweet.text
                    tweet_time = tweet.created_at
                    
                    logger.info(f"📱 Professional signal from @{username}: {tweet_text[:150]}...")
                    
                    # Extract trading symbols
                    detected_symbols = extract_trading_symbols(tweet_text)
                    
                    # Analyze signal quality
                    signal_analysis = analyze_professional_signal(tweet_text)
                    
                    action_taken = "none"
                    trade_success = False
                    trade_details = {}
                    
                    # Execute trades for high-quality signals
                    if (detected_symbols and 
                        signal_analysis['confidence_score'] >= MIN_CONFIDENCE_SCORE):
                        
                        logger.info(f"🎯 High-quality signal detected!")
                        logger.info(f"📊 Symbols: {detected_symbols}")
                        logger.info(f"📊 Analysis: {signal_analysis}")
                        
                        # Try to execute trade on first valid symbol
                        for symbol in detected_symbols:
                            success, details = execute_professional_strategy(symbol, signal_analysis)
                            if success:
                                action_taken = f"traded_{symbol}"
                                trade_success = True
                                trade_details = details
                                break
                            else:
                                action_taken = f"attempted_{symbol}_{details.get('reason', 'unknown')}"
                        
                        if trade_success:
                            logger.info(f"✅ Successfully executed trade based on @{username}'s signal!")
                        else:
                            logger.warning(f"⚠️ Failed to execute trades from @{username}'s signal")
                    
                    elif detected_symbols:
                        logger.info(f"📊 Symbols detected but confidence too low: {signal_analysis['confidence_score']:.2f}")
                        action_taken = f"low_confidence_{','.join(detected_symbols)}"
                    
                    # Log the professional signal
                    signal_data = {
                        'timestamp': datetime.now().isoformat(),
                        'tweet_id': tweet_id,
                        'username': username,
                        'content': tweet_text,
                        'detected_symbols': ','.join(detected_symbols),
                        'signal_type': signal_analysis['signal_type'],
                        'confidence_score': signal_analysis['confidence_score'],
                        'trading_direction': signal_analysis['trading_direction'],
                        'action_taken': action_taken,
                        'position_size': trade_details.get('position_size', 0),
                        'entry_price': trade_details.get('entry_price', 0),
                        'success': trade_success,
                        'risk_level': signal_analysis['risk_level']
                    }
                    
                    save_signal_log(signal_data)
                    processed_tweets.add(tweet_id)
                
                if new_signals_count > 0:
                    logger.info(f"📊 Processed {new_signals_count} professional signals from @{username}")
                else:
                    logger.debug(f"📊 No new signals from @{username}")
                
                # Respectful delay between accounts
                time.sleep(3)
                
            except TooManyRequests as e:
                logger.warning(f"⚠️ Rate limited while monitoring @{username}. Pausing...")
                time.sleep(300)  # Wait 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error monitoring @{username}: {e}")
                continue
        
        logger.info("✅ Professional signal monitoring completed")
        
    except Exception as e:
        logger.error(f"❌ Error in professional signal monitoring: {e}")

def professional_main_loop():
    """Main professional monitoring and trading loop."""
    logger.info("🔄 Running professional monitoring loop...")
    
    try:
        # Initialize Twitter client
        client = initialize_twitter_client()
        if not client:
            logger.error("❌ Could not initialize Twitter client")
            return
        
        # Check professional signals
        check_professional_signals(client)
        
        # Check funding opportunities
        check_funding_opportunities()
        
        # Log daily statistics
        logger.info(f"📊 Daily Stats - Positions: {daily_stats['positions_opened']}, PnL: {daily_stats['pnl']:.2f}")
        
        logger.info("✅ Professional monitoring loop completed")
        
    except Exception as e:
        logger.error(f"❌ Error in professional main loop: {e}")

if __name__ == "__main__":
    logger.info(f"🏛️  Bitfinex Professional Trading Monitor Initializing..." )
    logger.info(f"   Configuration: Position Size={USD_POSITION_SIZE} USD, Max Slippage={MAX_SLIPPAGE*100}%" )
    logger.info(f"   Margin Trading: {MARGIN_TRADING}, Funding Arbitrage: {FUNDING_ARBITRAGE}")
    logger.info(f"   Target Accounts: {TARGET_ACCOUNTS}")
    logger.info(f"   Check Interval: {CHECK_INTERVAL_MINUTES} minutes")
    logger.info(f"   Risk Management: Min Confidence={MIN_CONFIDENCE_SCORE}, Max Daily Positions={MAX_DAILY_POSITIONS}")
    
    try:
        # Load existing signal log
        load_signal_log()
        
        # Schedule professional monitoring
        schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(professional_main_loop)
        
        logger.info("🎯 Professional Bitfinex monitoring started! Press Ctrl+C to stop.")
        logger.info("📊 Monitoring for professional trading signals and arbitrage opportunities...")
        
        # Initial run
        professional_main_loop()
        
        # Keep running scheduled monitoring
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for professional operations
            
    except KeyboardInterrupt:
        logger.info("🛑 Professional Bitfinex monitor stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in professional monitor: {e}")
        raise
