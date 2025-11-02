#!/usr/bin/env python3
"""
Twitter sentiment analysis tool for collecting tweets based on a search query.
Rate limited to 5k requests per month.
"""

import httpx
from datetime import datetime
import csv
from random import randint
import os
import argparse
import logging
from typing import List, Dict, Any
import asyncio
from twikit import Client, TooManyRequests
from twikit.errors import Forbidden
from fake_useragent import UserAgent

# Create a single UserAgent instance
ua_generator = UserAgent()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Patch httpx.Client and httpx.AsyncClient to remove proxy settings which can cause issues
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

# Additionally patch twikit's AsyncClient reference
import twikit.client.client as twclient_mod
twclient_mod.AsyncClient = patched_async_client

# Import credentials from separate file
import dontshare as d

# Default configuration
DEFAULT_CONFIG = {
    "username": d.user_name,
    "email": d.email,
    "password": d.password,
    "query": "solana",
    "minimum_tweets": 100,
    "search_style": "latest",  # Options: 'latest' or 'top'
    "ignore_list": ['t.co', 'discord', 'join', 'telegram', 'discount', 'pay'],
    "output_file": "tweets.csv",
    "cookies_file": "cookies.json"
}

async def load_client(config: Dict[str, Any]) -> Client:
    """Load and verify Twitter client with cookies or login"""
    client = Client()
    cookies_file = config["cookies_file"]
    
    if os.path.exists(cookies_file):
        logger.info(f"Loading cookies from {cookies_file}")
        client.load_cookies(cookies_file)
        
        # Verify cookies are still valid
        try:
            user = await client.get_user_by_screen_name(config["username"])
            logger.info(f"Cookie verification successful (User: {user.name})")
            return client
        except Exception as e:
            logger.warning(f"Cookie verification failed: {e}")
            logger.info("Removing invalid cookies and re-authenticating...")
            os.remove(cookies_file)
    
    # Perform fresh login if no cookies or verification failed
    logger.info("No valid cookies found. Logging in...")
    await client.login(
        auth_info_1=config["username"],
        auth_info_2=config["email"],
        password=config["password"]
    )
    logger.info(f"Saving new cookies to {cookies_file}")
    client.save_cookies(cookies_file)
    return client

def should_ignore_tweet(text: str, ignore_list: List[str]) -> bool:
    """
    Check if a tweet should be ignored based on the ignore list.
    
    Args:
        text: The tweet text to check
        ignore_list: List of words that should cause a tweet to be ignored
        
    Returns:
        True if the tweet should be ignored, False otherwise
    """
    return any(word.lower() in text.lower() for word in ignore_list)

def create_csv_file(filename: str) -> None:
    """
    Create a new CSV file with headers for tweet data.
    
    Args:
        filename: Path to the CSV file
    """
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "tweet_count", 
            "user_name", 
            "text", 
            "created_at", 
            "retweet_count", 
            "favorite_count", 
            "reply_count"
        ])
    logger.info(f"Created output file: {filename}")

async def get_next_tweets(tweets, client: Client, config: Dict[str, Any], query: str, search_style: str) -> Any:
    """Get next tweets with random user-agent"""
    # Rotate user agent using .random property from the global generator
    ua_string = ua_generator.random
    client.http.headers["User-Agent"] = ua_string
    logger.debug(f"Using User-Agent: {ua_string[:60]}...")
    
    # Existing delay logic
    delay = randint(2, 6) if tweets is None else randint(5, 13)
    logger.info(f"Waiting {delay} seconds before next request...")
    await asyncio.sleep(delay)
    
    if tweets is None:
        logger.info(f"Starting new search for '{query}' ({search_style})")
        return await client.search_tweet(query, product=search_style)
    else:
        logger.info("Getting next page of results")
        return await tweets.next()

def save_tweet(tweet, tweet_count: int, output_file: str) -> None:
    """
    Save a tweet to the CSV file.
    
    Args:
        tweet: Tweet object
        tweet_count: Running count of tweets saved
        output_file: Path to the CSV file
    """
    tweet_data = [
        tweet_count,
        tweet.user.name,
        tweet.text,
        tweet.created_at,
        tweet.retweet_count,
        tweet.favorite_count,
        tweet.reply_count
    ]
    
    with open(output_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(tweet_data)
    
    logger.debug(f"Saved tweet #{tweet_count}: {tweet.text[:50]}...")

async def collect_tweets(config: Dict[str, Any]) -> None:
    """
    Main async function to collect tweets based on the provided configuration.
    
    Args:
        config: Dictionary containing configuration values
    """
    # Initialize the client
    client = await load_client(config) # Use await
    
    # Create output CSV file (this function is synchronous)
    create_csv_file(config["output_file"])
    
    tweet_count = 0
    tweets = None
    
    logger.info(f"Starting collection of at least {config['minimum_tweets']} tweets")
    
    # Main collection loop
    while tweet_count < config["minimum_tweets"]:
        try:
            tweets = await get_next_tweets(
                tweets, 
                client, 
                config,
                config["query"], 
                config["search_style"]
            )
            
            if not tweets:
                logger.info("No more tweets available")
                break
            
            # Process tweets using a standard synchronous for loop
            for tweet in tweets:
                if should_ignore_tweet(tweet.text, config["ignore_list"]):
                    logger.debug(f"Ignoring tweet: {tweet.text[:50]}...")
                    continue
                
                tweet_count += 1
                save_tweet(tweet, tweet_count, config["output_file"])
                
                if tweet_count >= config["minimum_tweets"]:
                    break
            
            logger.info(f"Collected {tweet_count}/{config['minimum_tweets']} tweets")
            
            if tweet_count >= config["minimum_tweets"]:
                 break
                 
        except TooManyRequests as e:
            # Handle HTTP 429 Too Many Requests (rate limiting)
            try:
                rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
                wait_time = (rate_limit_reset - datetime.now()).total_seconds()
                if wait_time > 0:
                    logger.warning(f"Rate limit reached. Waiting until {rate_limit_reset} ({wait_time:.1f} seconds)")
                    await asyncio.sleep(wait_time + 5)
                else:
                    logger.warning("Rate limit reached but reset time is in the past. Waiting 60 seconds.")
                    await asyncio.sleep(60)
            except (AttributeError, TypeError, ValueError) as err:
                logger.error(f"Error parsing rate limit: {err}")
                logger.warning("Rate limit reached. Waiting 5 minutes before retrying.")
                await asyncio.sleep(300)
            except Forbidden as e:
                # Handle HTTP 403 Forbidden errors by clearing session cookies and re-authenticating
                logger.error(f"Access forbidden (403) when searching tweets: {e}")
                try:
                    os.remove(config["cookies_file"])
                    logger.info("Deleted cookies file. Will re-authenticate on next iteration.")
                except OSError:
                    logger.warning("Failed to delete cookies file.")
                # Pause before re-authentication
                await asyncio.sleep(60)
                # Re-load the client for a new session
                client = await load_client(config)
                continue
        except Exception as e:
            logger.error(f"Unexpected error during collection: {e}", exc_info=True)
            logger.warning("Waiting 30 seconds before retrying...")
            await asyncio.sleep(30)
    
    logger.info(f"Collection complete. Saved {tweet_count} tweets to {config['output_file']}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Twitter sentiment analysis tool")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--minimum-tweets", type=int, help="Minimum number of tweets to collect")
    parser.add_argument("--search-style", choices=["latest", "top"], help="Search style (latest or top)")
    parser.add_argument("--output-file", type=str, help="Output CSV filename")
    return parser.parse_args()

async def main(): # Make main async
    """Main async entry point of the script."""
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Update with command-line arguments if provided
    args = parse_arguments()
    for key, value in vars(args).items():
        if value is not None:
            config[key.replace("-", "_")] = value
    
    try:
        await collect_tweets(config) # Use await
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    # Use asyncio.run to execute the async main function
    exit_code = asyncio.run(main())
    exit(exit_code)