"""
Twitter Sentiment Collector Service
Integrates Day_28_Projects/twitter.py functionality for real Twitter sentiment analysis
"""

import asyncio
import csv
import logging
import os
import re
import httpx
import numpy as np
from datetime import datetime, timezone, timedelta
from random import randint
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from fake_useragent import UserAgent
from textblob import TextBlob
from twikit import Client, TooManyRequests
from twikit.errors import Forbidden
from core.config import get_settings

logger = logging.getLogger(__name__)

# Create a single UserAgent instance
ua_generator = UserAgent()

# Patch httpx.Client and httpx.AsyncClient to remove proxy settings
original_client = httpx.Client
original_async_client = httpx.AsyncClient

def patched_client(*args, **kwargs):
    kwargs.pop('proxy', None)
    return original_client(*args, **kwargs)

def patched_async_client(*args, **kwargs):
    kwargs.pop('proxy', None)
    return original_async_client(*args, **kwargs)

httpx.Client = patched_client
httpx.AsyncClient = patched_async_client

# Patch twikit's AsyncClient reference
import twikit.client.client as twclient_mod
twclient_mod.AsyncClient = patched_async_client

@dataclass
class TweetData:
    """Data structure for collected tweets"""
    tweet_id: str
    user_name: str
    text: str
    created_at: datetime
    retweet_count: int
    favorite_count: int
    reply_count: int
    sentiment_score: float
    confidence: float
    crypto_mentions: List[str]
    timestamp_collected: datetime

@dataclass
class TwitterSentimentResult:
    """Aggregated Twitter sentiment result"""
    query: str
    total_tweets: int
    average_sentiment: float
    confidence: float
    mention_count: int
    engagement_score: float
    top_mentions: List[str]
    collection_time: datetime
    raw_tweets: List[TweetData]

class TwitterSentimentCollector:
    """Enterprise Twitter Sentiment Collector"""
    
    def __init__(self, config=None):
        self.settings = config or get_settings()
        
        # Twitter credentials (to be configured in settings)
        self.username = getattr(self.settings, 'TWITTER_USERNAME', None)
        self.email = getattr(self.settings, 'TWITTER_EMAIL', None)
        self.password = getattr(self.settings, 'TWITTER_PASSWORD', None)
        
        if not all([self.username, self.email, self.password]):
            logger.warning("⚠️ Twitter credentials not configured. Twitter sentiment collection will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
        
        # Configuration
        self.cookies_file = "./data/twitter_cookies.json"
        self.csv_file = "./data/twitter_sentiment.csv"
        self.minimum_tweets = getattr(self.settings, 'TWITTER_MINIMUM_TWEETS', 50)
        self.search_style = getattr(self.settings, 'TWITTER_SEARCH_STYLE', 'latest')
        self.collection_interval = getattr(self.settings, 'TWITTER_COLLECTION_INTERVAL', 1800)
        
        # Ignore list for filtering spam
        self.ignore_list = [
            't.co', 'discord', 'join', 'telegram', 'discount', 'pay',
            'airdrop', 'giveaway', 'free', 'win', 'contest', 'promotion'
        ]
        
        # Crypto queries
        self.crypto_queries = [
            'bitcoin OR btc',
            'ethereum OR eth', 
            'solana OR sol',
            'crypto OR cryptocurrency'
        ]
        
        # Token patterns
        self.token_patterns = [
            r'\$([A-Z]{2,10})',
            r'#([A-Z]{2,10})',
            r'\b([A-Z]{2,10})\s*(?:coin|token|crypto)\b'
        ]
        
        # State
        self.client: Optional[Client] = None
        self.is_running = False
        self.collected_tweets = []
        self.sentiment_cache = {}
        
        os.makedirs("./data", exist_ok=True)
        logger.info("🐦 Twitter Sentiment Collector initialized")
    
    async def start(self):
        """Start the Twitter sentiment collection service"""
        if not self.enabled:
            logger.warning("⚠️ Twitter sentiment collector disabled (missing credentials)")
            return
        
        self.is_running = True
        logger.info("🚀 Starting Twitter sentiment collection...")
        
        try:
            self.client = await self._load_client()
            self._create_csv_file()
            await self._collection_loop()
        except Exception as e:
            logger.error(f"❌ Error starting Twitter sentiment collector: {e}")
            self.is_running = False
    
    async def stop(self):
        """Stop the service"""
        self.is_running = False
        logger.info("🛑 Twitter sentiment collector stopped")
    
    async def _load_client(self) -> Client:
        """Load Twitter client with authentication"""
        client = Client()
        
        if os.path.exists(self.cookies_file):
            logger.info(f"🍪 Loading cookies from {self.cookies_file}")
            client.load_cookies(self.cookies_file)
            
            try:
                user = await client.get_user_by_screen_name(self.username)
                logger.info(f"✅ Cookie verification successful (User: {user.name})")
                return client
            except Exception as e:
                logger.warning(f"⚠️ Cookie verification failed: {e}")
                try:
                    os.remove(self.cookies_file)
                except OSError:
                    pass
        
        logger.info("🔐 Logging in to Twitter...")
        await client.login(
            auth_info_1=self.username,
            auth_info_2=self.email,
            password=self.password
        )
        logger.info(f"💾 Saving cookies to {self.cookies_file}")
        client.save_cookies(self.cookies_file)
        return client
    
    def _create_csv_file(self):
        """Create CSV file with headers"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "tweet_id", "user_name", "text", "created_at", "retweet_count",
                    "favorite_count", "reply_count", "sentiment_score", "confidence",
                    "crypto_mentions", "timestamp_collected", "query"
                ])
            logger.info(f"📄 Created CSV file: {self.csv_file}")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.is_running:
            try:
                for query in self.crypto_queries:
                    if not self.is_running:
                        break
                    
                    logger.info(f"🔍 Collecting tweets for: '{query}'")
                    result = await self._collect_tweets_for_query(query)
                    
                    if result:
                        await self._process_sentiment_result(result)
                    
                    await asyncio.sleep(randint(30, 60))
                
                logger.info(f"⏰ Waiting {self.collection_interval} seconds...")
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in collection loop: {e}")
                await asyncio.sleep(300)
    
    async def _collect_tweets_for_query(self, query: str) -> Optional[TwitterSentimentResult]:
        """Collect tweets for specific query"""
        try:
            tweets_data = []
            tweet_count = 0
            tweets = None
            
            while tweet_count < self.minimum_tweets:
                try:
                    tweets = await self._get_next_tweets(tweets, query)
                    
                    if not tweets:
                        break
                    
                    for tweet in tweets:
                        if self._should_ignore_tweet(tweet.text):
                            continue
                        
                        sentiment_data = self._analyze_tweet_sentiment(tweet.text)
                        crypto_mentions = self._extract_crypto_mentions(tweet.text)
                        
                        tweet_data = TweetData(
                            tweet_id=tweet.id,
                            user_name=tweet.user.name,
                            text=tweet.text,
                            created_at=tweet.created_at,
                            retweet_count=tweet.retweet_count,
                            favorite_count=tweet.favorite_count,
                            reply_count=tweet.reply_count,
                            sentiment_score=sentiment_data['sentiment_score'],
                            confidence=sentiment_data['confidence'],
                            crypto_mentions=crypto_mentions,
                            timestamp_collected=datetime.now(timezone.utc)
                        )
                        
                        tweets_data.append(tweet_data)
                        tweet_count += 1
                        self._save_tweet_to_csv(tweet_data, query)
                        
                        if tweet_count >= self.minimum_tweets:
                            break
                    
                    logger.info(f"📊 Collected {tweet_count}/{self.minimum_tweets} tweets")
                    
                except TooManyRequests as e:
                    await self._handle_rate_limit(e)
                except Forbidden as e:
                    await self._handle_forbidden_error(e)
                except Exception as e:
                    logger.error(f"❌ Error collecting tweets: {e}")
                    break
            
            if tweets_data:
                return self._create_sentiment_result(query, tweets_data)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in tweet collection: {e}")
            return None
    
    async def _get_next_tweets(self, tweets, query: str):
        """Get next tweets with rate limiting"""
        ua_string = ua_generator.random
        self.client.http.headers["User-Agent"] = ua_string
        
        delay = randint(2, 6) if tweets is None else randint(5, 13)
        await asyncio.sleep(delay)
        
        if tweets is None:
            return await self.client.search_tweet(query, product=self.search_style)
        else:
            return await tweets.next()
    
    def _should_ignore_tweet(self, text: str) -> bool:
        """Check if tweet should be ignored"""
        text_lower = text.lower()
        return any(word in text_lower for word in self.ignore_list)
    
    def _analyze_tweet_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        try:
            text_hash = hash(text)
            if text_hash in self.sentiment_cache:
                return self.sentiment_cache[text_hash]
            
            blob = TextBlob(text.lower())
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            result = {
                'sentiment_score': polarity,
                'confidence': 1 - subjectivity,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
            
            self.sentiment_cache[text_hash] = result
            return result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment: {e}")
            return {'sentiment_score': 0, 'confidence': 0, 'polarity': 0, 'subjectivity': 0}
    
    def _extract_crypto_mentions(self, text: str) -> List[str]:
        """Extract crypto mentions from text"""
        mentions = []
        
        for pattern in self.token_patterns:
            matches = re.findall(pattern, text.upper())
            mentions.extend(matches)
        
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol',
            'cardano', 'ada', 'polkadot', 'dot', 'chainlink', 'link'
        ]
        
        text_lower = text.lower()
        for keyword in crypto_keywords:
            if keyword in text_lower:
                mentions.append(keyword.upper())
        
        return list(set(mentions))
    
    def _save_tweet_to_csv(self, tweet_data: TweetData, query: str):
        """Save tweet to CSV"""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    tweet_data.tweet_id,
                    tweet_data.user_name,
                    tweet_data.text,
                    tweet_data.created_at.isoformat(),
                    tweet_data.retweet_count,
                    tweet_data.favorite_count,
                    tweet_data.reply_count,
                    tweet_data.sentiment_score,
                    tweet_data.confidence,
                    ','.join(tweet_data.crypto_mentions),
                    tweet_data.timestamp_collected.isoformat(),
                    query
                ])
        except Exception as e:
            logger.error(f"❌ Error saving tweet: {e}")
    
    def _create_sentiment_result(self, query: str, tweets_data: List[TweetData]) -> TwitterSentimentResult:
        """Create aggregated sentiment result"""
        try:
            if not tweets_data:
                return None
            
            total_tweets = len(tweets_data)
            sentiments = [t.sentiment_score for t in tweets_data]
            confidences = [t.confidence for t in tweets_data]
            
            total_weight = sum(confidences)
            if total_weight > 0:
                average_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / total_weight
            else:
                average_sentiment = np.mean(sentiments)
            
            confidence = np.mean(confidences)
            
            engagements = []
            for tweet in tweets_data:
                engagement = tweet.retweet_count + tweet.favorite_count + tweet.reply_count
                engagements.append(engagement)
            
            engagement_score = np.mean(engagements) if engagements else 0
            
            all_mentions = []
            for tweet in tweets_data:
                all_mentions.extend(tweet.crypto_mentions)
            
            mention_counts = {}
            for mention in all_mentions:
                mention_counts[mention] = mention_counts.get(mention, 0) + 1
            
            top_mentions = sorted(mention_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_mentions = [mention for mention, count in top_mentions]
            
            return TwitterSentimentResult(
                query=query,
                total_tweets=total_tweets,
                average_sentiment=average_sentiment,
                confidence=confidence,
                mention_count=len(all_mentions),
                engagement_score=engagement_score,
                top_mentions=top_mentions,
                collection_time=datetime.now(timezone.utc),
                raw_tweets=tweets_data
            )
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment result: {e}")
            return None
    
    async def _process_sentiment_result(self, result: TwitterSentimentResult):
        """Process sentiment result"""
        try:
            logger.info(f"📊 TWITTER SENTIMENT: {result.query}")
            logger.info(f"   📈 Sentiment: {result.average_sentiment:.3f}")
            logger.info(f"   🎯 Confidence: {result.confidence:.3f}")
            logger.info(f"   📱 Tweets: {result.total_tweets}")
            logger.info(f"   🔥 Top: {', '.join(result.top_mentions[:5])}")
            
            self.collected_tweets.append(result)
            
            # Keep only recent results
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            self.collected_tweets = [
                r for r in self.collected_tweets 
                if r.collection_time > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"❌ Error processing result: {e}")
    
    async def _handle_rate_limit(self, e: TooManyRequests):
        """Handle rate limiting"""
        try:
            rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
            wait_time = (rate_limit_reset - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.warning(f"⏰ Rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 5)
            else:
                await asyncio.sleep(60)
        except Exception:
            logger.warning("⏰ Rate limit: waiting 5 minutes")
            await asyncio.sleep(300)
    
    async def _handle_forbidden_error(self, e: Forbidden):
        """Handle 403 errors"""
        logger.error(f"🚫 Access forbidden: {e}")
        try:
            os.remove(self.cookies_file)
            logger.info("🗑️ Deleted cookies, will re-authenticate")
        except OSError:
            pass
        
        await asyncio.sleep(60)
        self.client = await self._load_client()
    
    async def get_recent_sentiment(self, hours: int = 24) -> List[TwitterSentimentResult]:
        """Get recent sentiment results"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [r for r in self.collected_tweets if r.collection_time > cutoff_time]
        except Exception as e:
            logger.error(f"❌ Error getting recent sentiment: {e}")
            return []
    
    async def get_sentiment_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get sentiment for specific symbol"""
        try:
            recent_results = await self.get_recent_sentiment(hours=6)
            
            symbol_tweets = []
            for result in recent_results:
                for tweet in result.raw_tweets:
                    if symbol.upper() in tweet.crypto_mentions:
                        symbol_tweets.append(tweet)
            
            if not symbol_tweets:
                return None
            
            sentiments = [t.sentiment_score for t in symbol_tweets]
            confidences = [t.confidence for t in symbol_tweets]
            
            total_weight = sum(confidences)
            if total_weight > 0:
                avg_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / total_weight
            else:
                avg_sentiment = np.mean(sentiments)
            
            return {
                'symbol': symbol,
                'sentiment_score': avg_sentiment,
                'confidence': np.mean(confidences),
                'mention_count': len(symbol_tweets),
                'total_engagement': sum(t.retweet_count + t.favorite_count + t.reply_count for t in symbol_tweets),
                'timestamp': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment for {symbol}: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'enabled': self.enabled,
            'running': self.is_running,
            'total_results': len(self.collected_tweets),
            'csv_file': self.csv_file,
            'minimum_tweets': self.minimum_tweets,
            'search_style': self.search_style,
            'collection_interval': self.collection_interval
        } 