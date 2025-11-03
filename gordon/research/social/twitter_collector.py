"""
Twitter Data Collector
======================
Day 28: Enhanced Twitter data collection with sentiment analysis integration.

Features:
- Twitter authentication with cookie persistence
- Tweet collection with filtering
- Rate limit handling
- Integration with sentiment analysis
"""

import asyncio
import csv
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from random import randint
import httpx
from twikit import Client, TooManyRequests
from twikit.errors import Forbidden
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)

# Patch httpx to remove proxy settings which can cause issues
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

# Create a single UserAgent instance
ua_generator = UserAgent()


class TwitterCollector:
    """
    Twitter data collector with authentication and rate limiting.
    
    Collects tweets based on search queries and provides them for sentiment analysis.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Twitter collector.
        
        Args:
            config: Configuration dictionary with:
                - username: Twitter username
                - email: Twitter email
                - password: Twitter password
                - cookies_file: Path to cookies file
                - ignore_list: List of words to ignore in tweets
        """
        self.config = config or {}
        self.username = self.config.get('username')
        self.email = self.config.get('email')
        self.password = self.config.get('password')
        self.cookies_file = self.config.get('cookies_file', 'cookies.json')
        self.ignore_list = self.config.get('ignore_list', [
            't.co', 'discord', 'join', 'telegram', 'discount', 'pay'
        ])
        self.client: Optional[Client] = None

    async def _load_client(self) -> Client:
        """Load and verify Twitter client with cookies or login."""
        if self.client:
            return self.client
        
        client = Client()
        
        if os.path.exists(self.cookies_file):
            logger.info(f"Loading cookies from {self.cookies_file}")
            client.load_cookies(self.cookies_file)
            
            # Verify cookies are still valid
            try:
                user = await client.get_user_by_screen_name(self.username)
                logger.info(f"Cookie verification successful (User: {user.name})")
                self.client = client
                return client
            except Exception as e:
                logger.warning(f"Cookie verification failed: {e}")
                logger.info("Removing invalid cookies and re-authenticating...")
                os.remove(self.cookies_file)
        
        # Perform fresh login if no cookies or verification failed
        if not self.username or not self.email or not self.password:
            raise ValueError("Twitter credentials not configured")
        
        logger.info("No valid cookies found. Logging in...")
        await client.login(
            auth_info_1=self.username,
            auth_info_2=self.email,
            password=self.password
        )
        logger.info(f"Saving new cookies to {self.cookies_file}")
        client.save_cookies(self.cookies_file)
        self.client = client
        return client

    def _should_ignore_tweet(self, text: str) -> bool:
        """
        Check if a tweet should be ignored based on the ignore list.
        
        Args:
            text: The tweet text to check
            
        Returns:
            True if the tweet should be ignored, False otherwise
        """
        return any(word.lower() in text.lower() for word in self.ignore_list)

    async def _get_next_tweets(
        self,
        tweets,
        query: str,
        search_style: str = 'latest'
    ):
        """Get next tweets with random user-agent and delay."""
        client = await self._load_client()
        
        # Rotate user agent
        ua_string = ua_generator.random
        client.http.headers["User-Agent"] = ua_string
        logger.debug(f"Using User-Agent: {ua_string[:60]}...")
        
        # Delay logic
        delay = randint(2, 6) if tweets is None else randint(5, 13)
        logger.info(f"Waiting {delay} seconds before next request...")
        await asyncio.sleep(delay)
        
        if tweets is None:
            logger.info(f"Starting new search for '{query}' ({search_style})")
            return await client.search_tweet(query, product=search_style)
        else:
            logger.info("Getting next page of results")
            return await tweets.next()

    def _tweet_to_dict(self, tweet, tweet_count: int) -> Dict[str, Any]:
        """
        Convert tweet object to dictionary.
        
        Args:
            tweet: Tweet object
            tweet_count: Running count of tweets
            
        Returns:
            Dictionary with tweet data
        """
        return {
            'tweet_count': tweet_count,
            'user_name': tweet.user.name,
            'text': tweet.text,
            'created_at': tweet.created_at,
            'retweet_count': tweet.retweet_count,
            'favorite_count': tweet.favorite_count,
            'reply_count': tweet.reply_count,
            'timestamp': datetime.now().isoformat()
        }

    async def collect_tweets(
        self,
        query: str,
        minimum_tweets: int = 100,
        search_style: str = 'latest',
        on_tweet: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect tweets based on search query.
        
        Args:
            query: Search query
            minimum_tweets: Minimum number of tweets to collect
            search_style: 'latest' or 'top'
            on_tweet: Optional callback function for each tweet
            
        Returns:
            List of tweet dictionaries
        """
        await self._load_client()
        
        tweets_collected = []
        tweet_count = 0
        tweets = None
        
        logger.info(f"Starting collection of at least {minimum_tweets} tweets for query: {query}")
        
        while tweet_count < minimum_tweets:
            try:
                tweets = await self._get_next_tweets(tweets, query, search_style)
                
                if not tweets:
                    logger.info("No more tweets available")
                    break
                
                # Process tweets
                for tweet in tweets:
                    if self._should_ignore_tweet(tweet.text):
                        logger.debug(f"Ignoring tweet: {tweet.text[:50]}...")
                        continue
                    
                    tweet_count += 1
                    tweet_dict = self._tweet_to_dict(tweet, tweet_count)
                    tweets_collected.append(tweet_dict)
                    
                    # Call callback if provided
                    if on_tweet:
                        on_tweet(tweet_dict)
                    
                    logger.debug(f"Collected tweet #{tweet_count}: {tweet.text[:50]}...")
                    
                    if tweet_count >= minimum_tweets:
                        break
                
                logger.info(f"Collected {tweet_count}/{minimum_tweets} tweets")
                
                if tweet_count >= minimum_tweets:
                    break
                    
            except TooManyRequests as e:
                # Handle rate limiting
                try:
                    rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                    wait_time = (rate_limit_reset - datetime.now()).total_seconds()
                    if wait_time > 0:
                        logger.warning(
                            f"Rate limit reached. Waiting until {rate_limit_reset} "
                            f"({wait_time:.1f} seconds)"
                        )
                        await asyncio.sleep(wait_time + 5)
                    else:
                        logger.warning("Rate limit reached but reset time is in the past. Waiting 60 seconds.")
                        await asyncio.sleep(60)
                except (AttributeError, TypeError, ValueError) as err:
                    logger.error(f"Error parsing rate limit: {err}")
                    logger.warning("Rate limit reached. Waiting 5 minutes before retrying.")
                    await asyncio.sleep(300)
                    
            except Forbidden as e:
                # Handle forbidden errors
                logger.error(f"Access forbidden (403) when searching tweets: {e}")
                try:
                    os.remove(self.cookies_file)
                    logger.info("Deleted cookies file. Will re-authenticate on next iteration.")
                except OSError:
                    logger.warning("Failed to delete cookies file.")
                await asyncio.sleep(60)
                self.client = None
                await self._load_client()
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error during collection: {e}", exc_info=True)
                logger.warning("Waiting 30 seconds before retrying...")
                await asyncio.sleep(30)
        
        logger.info(f"Collection complete. Collected {tweet_count} tweets")
        return tweets_collected

    def save_to_csv(self, tweets: List[Dict[str, Any]], output_file: str):
        """
        Save tweets to CSV file.
        
        Args:
            tweets: List of tweet dictionaries
            output_file: Path to CSV file
        """
        if not tweets:
            logger.warning("No tweets to save")
            return
        
        # Create CSV file with headers
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'tweet_count',
                'user_name',
                'text',
                'created_at',
                'retweet_count',
                'favorite_count',
                'reply_count',
                'timestamp'
            ])
            
            for tweet in tweets:
                writer.writerow([
                    tweet.get('tweet_count'),
                    tweet.get('user_name'),
                    tweet.get('text'),
                    tweet.get('created_at'),
                    tweet.get('retweet_count'),
                    tweet.get('favorite_count'),
                    tweet.get('reply_count'),
                    tweet.get('timestamp')
                ])
        
        logger.info(f"Saved {len(tweets)} tweets to {output_file}")

    async def search_symbol_tweets(
        self,
        symbol: str,
        minimum_tweets: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for tweets about a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC', 'SOL', 'AAPL')
            minimum_tweets: Minimum number of tweets to collect
            
        Returns:
            List of tweet dictionaries
        """
        # Create search query for symbol
        query = f"${symbol} OR {symbol}"
        return await self.collect_tweets(query, minimum_tweets)
    
    async def get_user_tweets(
        self,
        username: str,
        count: int = 10,
        since_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tweets from a specific user account.
        
        Args:
            username: Twitter username (without @)
            count: Number of tweets to fetch
            since_id: Only return tweets newer than this ID
            
        Returns:
            List of tweet dictionaries
        """
        await self._load_client()
        
        try:
            # Get user by screen name
            user = await self.client.get_user_by_screen_name(username)
            
            if not user:
                logger.warning(f"User {username} not found")
                return []
            
            # Get user tweets
            tweets_result = await user.get_tweets('Tweets', count=count)
            
            if not tweets_result:
                logger.debug(f"No tweets found for {username}")
                return []
            
            tweets_collected = []
            tweet_count = 0
            
            for tweet in tweets_result:
                # Skip if we've reached since_id
                if since_id and str(tweet.id) == since_id:
                    break
                
                # Skip if tweet should be ignored
                if self._should_ignore_tweet(tweet.text):
                    continue
                
                tweet_count += 1
                tweet_dict = {
                    'id': str(tweet.id),
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'user_name': tweet.user.name,
                    'user_screen_name': tweet.user.screen_name,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'reply_count': tweet.reply_count,
                    'timestamp': datetime.now().isoformat()
                }
                tweets_collected.append(tweet_dict)
                
                # Stop if we've reached since_id
                if since_id and str(tweet.id) == since_id:
                    break
            
            logger.info(f"Collected {len(tweets_collected)} tweets from @{username}")
            return tweets_collected
            
        except TooManyRequests as e:
            logger.warning(f"Rate limit reached while fetching tweets from {username}")
            raise
        except Forbidden as e:
            logger.error(f"Access forbidden (403) for user {username}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching tweets from {username}: {e}")
            return []
    
    async def get_user_tweets_by_id(
        self,
        user_id: str,
        count: int = 10,
        since_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tweets from a user by their Twitter ID.
        
        Args:
            user_id: Twitter user ID
            count: Number of tweets to fetch
            since_id: Only return tweets newer than this ID
            
        Returns:
            List of tweet dictionaries
        """
        await self._load_client()
        
        try:
            # Get user by ID
            user = await self.client.get_user_by_id(user_id)
            
            if not user:
                logger.warning(f"User with ID {user_id} not found")
                return []
            
            # Get user tweets
            tweets_result = await user.get_tweets('Tweets', count=count)
            
            if not tweets_result:
                logger.debug(f"No tweets found for user ID {user_id}")
                return []
            
            tweets_collected = []
            
            for tweet in tweets_result:
                # Skip if we've reached since_id
                if since_id and str(tweet.id) == since_id:
                    break
                
                # Skip if tweet should be ignored
                if self._should_ignore_tweet(tweet.text):
                    continue
                
                tweet_dict = {
                    'id': str(tweet.id),
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'user_name': tweet.user.name,
                    'user_screen_name': tweet.user.screen_name,
                    'user_id': str(tweet.user.id),
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count,
                    'reply_count': tweet.reply_count,
                    'timestamp': datetime.now().isoformat()
                }
                tweets_collected.append(tweet_dict)
                
                # Stop if we've reached since_id
                if since_id and str(tweet.id) == since_id:
                    break
            
            logger.info(f"Collected {len(tweets_collected)} tweets from user ID {user_id}")
            return tweets_collected
            
        except TooManyRequests as e:
            logger.warning(f"Rate limit reached while fetching tweets from user ID {user_id}")
            raise
        except Forbidden as e:
            logger.error(f"Access forbidden (403) for user ID {user_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching tweets from user ID {user_id}: {e}")
            return []
    
    async def monitor_accounts(
        self,
        accounts: Dict[str, str],
        latest_tweet_ids: Optional[Dict[str, str]] = None,
        count_per_account: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Monitor multiple Twitter accounts for new tweets.
        
        Args:
            accounts: Dictionary mapping account_id -> username
            latest_tweet_ids: Dictionary mapping account_id -> latest processed tweet ID
            count_per_account: Number of recent tweets to check per account
            
        Returns:
            Dictionary mapping account_id -> list of new tweets
        """
        if latest_tweet_ids is None:
            latest_tweet_ids = {}
        
        new_tweets = {}
        
        for account_id, username in accounts.items():
            since_id = latest_tweet_ids.get(account_id)
            
            try:
                tweets = await self.get_user_tweets_by_id(
                    account_id,
                    count=count_per_account,
                    since_id=since_id
                )
                
                if tweets:
                    new_tweets[account_id] = tweets
                    logger.info(f"Found {len(tweets)} new tweets from {username}")
                
            except Exception as e:
                logger.error(f"Error monitoring {username}: {e}")
                continue
        
        return new_tweets

