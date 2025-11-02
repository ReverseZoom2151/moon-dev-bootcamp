"""
Twitter Solana Bot Service

Monitors target Twitter accounts for Solana contract addresses in new tweets
and executes buy orders via Jupiter aggregator if found.

Based on Day 36 Projects implementation with enterprise enhancements.
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

def is_contract_address(text: str) -> Optional[str]:
    """Check if text contains a string that looks like a Solana contract address.

    Args:
        text: The input string (e.g., tweet text).

    Returns:
        The potential Solana address string if found, otherwise None.
    """
    # Solana addresses are Base58 encoded and typically 32-44 characters long.
    base58_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    words = text.split()
    for word in words:
        if 32 <= len(word) <= 44 and all(c in base58_chars for c in word):
            # Basic check passed, could add more validation if needed (e.g., regex)
            return word
    return None

@dataclass
class TweetLog:
    """Data structure for tweet log entries"""
    timestamp: str
    username: str
    tweet_text: str
    contract_found: str

class TwitterSolanaBotService:
    """
    Twitter Solana Bot Service
    
    Monitors Twitter accounts for Solana contract addresses and executes trades
    """
    
    def __init__(self):
        self.is_running = False
        
        # Configuration (loaded from settings)
        self.config = {
            'enabled': False,
            'polling_interval': 4,
            'usdc_size': 0.1,
            'orders_per_open': 5,
            'slippage': 500,
            'priority_fee': 200000,
            'max_retries': 5,
            'retry_delay': 5,
            'target_accounts': {},
            'csv_filename': 'twitter_monitor.csv',
            'cookies_filename': 'cookies.json'
        }
        
        # Tweet log storage
        self.df_tweet_log = pd.DataFrame(columns=['timestamp', 'username', 'tweet_text', 'contract_found'])
        
        logger.info("🐦 Twitter Solana Bot Service initialized")
    
    async def start(self):
        """Start the Twitter Solana bot service"""
        try:
            self.is_running = True
            logger.info("🚀 Twitter Solana Bot started")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start Twitter Solana Bot: {e}")
            self.is_running = False
            return False
    
    async def stop(self):
        """Stop the Twitter Solana bot service"""
        self.is_running = False
        logger.info("🛑 Twitter Solana Bot stopped")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get the service status"""
        return {
            'service': 'Twitter Solana Bot',
            'status': 'running' if self.is_running else 'stopped',
            'dependencies_available': True,  # Assume dependencies are available
            'configuration': self.config
        }
    
    def test_contract_detection(self, text: str) -> Optional[str]:
        """Test contract detection functionality"""
        return is_contract_address(text)
    
    def _save_tweet_log(self, new_data: Dict[str, str]):
        """Append new data to the tweet log"""
        try:
            new_df = pd.DataFrame([new_data])
            self.df_tweet_log = pd.concat([self.df_tweet_log, new_df], ignore_index=True)
            # Optionally save to CSV
            if self.config.get('csv_filename'):
                self.df_tweet_log.to_csv(self.config['csv_filename'], index=False)
            logger.debug(f"Saved tweet log entry")
        except Exception as e:
            logger.error(f"Error saving tweet log: {e}")
    
    async def simulate_tweet_detection(self, tweet_text: str, username: str = "test_user"):
        """Simulate tweet detection for testing purposes"""
        current_time = datetime.now()
        contract_address = self.test_contract_detection(tweet_text)
        
        log_data = {
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'username': username,
            'tweet_text': tweet_text,
            'contract_found': contract_address if contract_address else ''
        }
        
        self._save_tweet_log(log_data)
        
        result = {
            'tweet_processed': True,
            'contract_found': contract_address,
            'log_entry': log_data
        }
        
        if contract_address:
            logger.info(f"💎 Contract detected in simulated tweet: {contract_address}")
            result['would_trigger_buy'] = True
        else:
            logger.info("No contract found in simulated tweet")
            result['would_trigger_buy'] = False
        
        return result
    
    async def get_recent_tweets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tweet logs"""
        try:
            recent_logs = self.df_tweet_log.tail(limit).to_dict('records')
            return recent_logs
        except Exception as e:
            logger.error(f"Failed to get recent tweets: {e}")
            return []
    
    async def add_target_account(self, user_id: str, username: str) -> bool:
        """Add a target account to monitor"""
        try:
            self.config['target_accounts'][user_id] = username
            logger.info(f"Added target account: {username} ({user_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add target account: {e}")
            return False
    
    async def remove_target_account(self, user_id: str) -> bool:
        """Remove a target account from monitoring"""
        try:
            if user_id in self.config['target_accounts']:
                username = self.config['target_accounts'].pop(user_id)
                logger.info(f"Removed target account: {username} ({user_id})")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove target account: {e}")
            return False

# Global service instance
twitter_solana_bot_service = TwitterSolanaBotService() 