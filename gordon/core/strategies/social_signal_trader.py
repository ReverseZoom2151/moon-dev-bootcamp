"""
Social Signal Trading Automation
=================================
Day 36: Twitter monitoring and automated trading based on social signals.
"""

import pandas as pd
import logging
import re
import time
import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SocialSignalDetector:
    """
    Detects trading opportunities from social media signals.
    
    Monitors Twitter for token mentions, contract addresses, and trading signals.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize social signal detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Token patterns
        self.token_patterns = self.config.get('token_patterns', [
            r'\$([A-Z]{2,6})\b',  # $BTC, $ETH pattern
            r'\b([A-Z]{2,6})USDT\b',  # BTCUSDT pattern
            r'\b([A-Z]{2,6})/USDT\b',  # BTC/USDT pattern
            r'#([A-Za-z0-9]{2,10})',  # Hashtag tokens
        ])
        
        # Opportunity keywords
        self.opportunity_keywords = self.config.get('opportunity_keywords', [
            'new listing', 'now live', 'trading starts', 'available now',
            'just listed', 'spot trading', 'binance listing', 'trading pair',
            'pump', 'moon', 'gem', 'breakout', 'bullish', 'buy signal'
        ])
        
        # Solana contract pattern
        self.solana_address_pattern = re.compile(
            r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
        )
        
        self.processed_tweets: Set[str] = set()
    
    def extract_tokens(self, text: str) -> List[str]:
        """Extract potential token symbols from text."""
        tokens = set()
        text_upper = text.upper()
        
        # Apply regex patterns
        for pattern in self.token_patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if len(match) >= 2 and len(match) <= 6:
                    # Add USDT suffix for Binance trading pairs
                    if not match.endswith('USDT'):
                        tokens.add(f"{match}USDT")
                    else:
                        tokens.add(match)
        
        return list(tokens)
    
    def extract_solana_contract(self, text: str) -> Optional[str]:
        """Extract Solana contract address from text."""
        matches = self.solana_address_pattern.findall(text)
        if matches:
            # Return first match that looks like a valid Solana address
            for match in matches:
                if 32 <= len(match) <= 44:
                    return match
        return None
    
    def check_opportunity_keywords(self, text: str) -> bool:
        """Check if text contains opportunity keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.opportunity_keywords)
    
    def detect_opportunities(
        self,
        tweet_text: str,
        tweet_id: str
    ) -> Optional[Dict[str, any]]:
        """
        Detect trading opportunities from tweet text.
        
        Args:
            tweet_text: Tweet content
            tweet_id: Unique tweet identifier
            
        Returns:
            Dictionary with opportunity details or None
        """
        if tweet_id in self.processed_tweets:
            return None
        
        opportunity = {
            'tweet_id': tweet_id,
            'text': tweet_text,
            'timestamp': datetime.now().isoformat(),
            'tokens': [],
            'solana_contract': None,
            'has_opportunity_keywords': False,
            'confidence': 0.0
        }
        
        # Extract tokens
        tokens = self.extract_tokens(tweet_text)
        opportunity['tokens'] = tokens
        
        # Extract Solana contract
        solana_contract = self.extract_solana_contract(tweet_text)
        opportunity['solana_contract'] = solana_contract
        
        # Check opportunity keywords
        has_keywords = self.check_opportunity_keywords(tweet_text)
        opportunity['has_opportunity_keywords'] = has_keywords
        
        # Calculate confidence score
        confidence = 0.0
        if tokens:
            confidence += 0.3
        if solana_contract:
            confidence += 0.4
        if has_keywords:
            confidence += 0.3
        
        opportunity['confidence'] = confidence
        
        # Only return if confidence threshold met
        min_confidence = self.config.get('min_confidence', 0.5)
        if confidence >= min_confidence:
            self.processed_tweets.add(tweet_id)
            return opportunity
        
        return None


class SocialSignalExecutor:
    """
    Executes trades based on social signals.
    
    Handles automated trading when opportunities are detected.
    Supports exchange-specific execution logic (Binance, Bitfinex).
    Note: Solana contract addresses are detected but not executed (require Jupiter/DEX access).
    """
    
    def __init__(
        self,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize social signal executor.
        
        Args:
            exchange_adapter: Exchange adapter for trading
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or {}
        
        self.max_trade_amount = self.config.get('max_trade_amount', 100)  # USDT
        self.max_slippage = self.config.get('max_slippage', 500)  # Basis points
        self.enabled = self.config.get('enabled', False)
        
        # Exchange-specific settings
        self.exchange_type = self.config.get('exchange_type', 'binance').lower()
        self.orders_per_opportunity = self.config.get('orders_per_opportunity', 1)
        
        # Confidence scoring (from Day 36 Bitfinex logic)
        self.min_confidence_score = self.config.get('min_confidence_score', 0.7)
        self.max_daily_positions = self.config.get('max_daily_positions', 5)
        self.position_sizing_method = self.config.get('position_sizing_method', 'fixed')
        
        self.trade_log: List[Dict] = []
        self.daily_stats = {'positions_opened': 0, 'positions_closed': 0, 'pnl': 0.0}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol format."""
        if not symbol or len(symbol) < 3:
            return False
        
        # Binance format: BTCUSDT
        if self.exchange_type == 'binance':
            if not symbol.endswith('USDT'):
                return False
        
        # Bitfinex format: btcusd (lowercase)
        elif self.exchange_type == 'bitfinex':
            if not symbol.endswith('usd'):
                return False
        
        # Solana contract addresses cannot be traded on CEX - skip them
        # (We only support Binance/Bitfinex which use symbol pairs like SOLUSDT)
        if len(symbol) >= 32 and len(symbol) <= 44:
            # Looks like a Solana contract address - not supported on CEX
            return False
        
        return True
    
    def calculate_position_size(
        self,
        symbol: str,
        confidence_score: float,
        risk_level: str = 'medium'
    ) -> float:
        """
        Calculate position size based on confidence and risk.
        
        Enhanced with Day 36 Bitfinex position sizing logic.
        """
        base_size = self.max_trade_amount
        
        if self.position_sizing_method == 'fixed':
            return base_size
        
        elif self.position_sizing_method == 'volatility_adjusted':
            # Could fetch volatility data here
            # For now, use confidence-based adjustment
            volatility_multiplier = max(0.5, 1.0 - (confidence_score * 0.5))
            base_size *= volatility_multiplier
        
        elif self.position_sizing_method == 'sentiment_weighted':
            # Adjust size based on confidence
            base_size *= (0.5 + confidence_score * 0.5)
        
        # Risk level adjustments
        if risk_level == 'high':
            base_size *= 0.5
        elif risk_level == 'low':
            base_size *= 1.2
        
        return round(base_size, 2)
    
    def execute_trade(
        self,
        symbol: str,
        opportunity: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Execute trade for detected opportunity.
        
        Enhanced with Day 36 exchange-specific execution logic.
        
        Args:
            symbol: Trading symbol
            opportunity: Opportunity details
            
        Returns:
            Trade execution result
        """
        if not self.enabled:
            logger.warning("Automated trading is disabled")
            return {'success': False, 'reason': 'disabled'}
        
        # Check minimum confidence threshold (Day 36 Bitfinex logic)
        confidence = opportunity.get('confidence', 0.0)
        if confidence < self.min_confidence_score:
            logger.info(
                f"Confidence {confidence:.2f} below threshold {self.min_confidence_score}"
            )
            return {'success': False, 'reason': 'insufficient_confidence'}
        
        # Check daily position limit
        if self.daily_stats['positions_opened'] >= self.max_daily_positions:
            logger.warning(f"Daily position limit reached: {self.daily_stats['positions_opened']}")
            return {'success': False, 'reason': 'daily_limit_reached'}
        
        if not self.validate_symbol(symbol):
            logger.warning(f"Invalid symbol format: {symbol}")
            return {'success': False, 'reason': 'invalid_symbol'}
        
        try:
            # Exchange-specific execution
            if self.exchange_type == 'binance':
                return self._execute_binance_trade(symbol, opportunity)
            elif self.exchange_type == 'bitfinex':
                return self._execute_bitfinex_trade(symbol, opportunity)
            else:
                return self._execute_generic_trade(symbol, opportunity)
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _execute_binance_trade(
        self,
        symbol: str,
        opportunity: Dict[str, any]
    ) -> Dict[str, any]:
        """Execute trade on Binance (Day 36 logic)."""
        logger.info(f"Executing Binance trade for {symbol}")
        
        # Get current price
        if hasattr(self.exchange_adapter, 'fetch_ticker'):
            ticker = self.exchange_adapter.fetch_ticker(symbol)
            current_price = ticker.get('last', 0)
        else:
            logger.error("Exchange adapter does not support price fetching")
            return {'success': False, 'reason': 'exchange_not_supported'}
        
        if not current_price:
            logger.warning(f"Could not get price for {symbol}")
            return {'success': False, 'reason': 'price_unavailable'}
        
        # Calculate position size
        position_size = self.calculate_position_size(
            symbol,
            opportunity.get('confidence', 0.0),
            opportunity.get('risk_level', 'medium')
        )
        
        # Execute multiple orders if configured (Day 36 aggressive buying)
        successful_orders = []
        for i in range(self.orders_per_opportunity):
            try:
                quantity = position_size / current_price
                
                if hasattr(self.exchange_adapter, 'create_market_buy_order'):
                    order = self.exchange_adapter.create_market_buy_order(
                        symbol=symbol,
                        amount=quantity
                    )
                    
                    successful_orders.append({
                        'order_id': order.get('id'),
                        'quantity': quantity,
                        'price': current_price
                    })
                    
                    logger.info(f"✅ Order {i+1}/{self.orders_per_opportunity} executed")
                    
                    # Small delay between orders
                    import time
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Order {i+1} failed: {e}")
                continue
        
        if successful_orders:
            self.daily_stats['positions_opened'] += len(successful_orders)
            
            trade_result = {
                'success': True,
                'symbol': symbol,
                'orders': successful_orders,
                'amount_usdt': position_size,
                'opportunity': opportunity
            }
            
            self.trade_log.append(trade_result)
            logger.info(f"✅ Binance trade executed successfully: {symbol}")
            
            return trade_result
        else:
            return {'success': False, 'reason': 'all_orders_failed'}
    
    def _execute_bitfinex_trade(
        self,
        symbol: str,
        opportunity: Dict[str, any]
    ) -> Dict[str, any]:
        """Execute trade on Bitfinex (Day 36 logic)."""
        logger.info(f"Executing Bitfinex trade for {symbol}")
        
        # Bitfinex uses lowercase symbols (e.g., btcusd)
        symbol_lower = symbol.lower()
        if symbol_lower.endswith('usdt'):
            symbol_lower = symbol_lower.replace('usdt', 'usd')
        
        # Similar to Binance but with Bitfinex-specific formatting
        return self._execute_generic_trade(symbol_lower, opportunity)
    
    def _execute_generic_trade(
        self,
        symbol: str,
        opportunity: Dict[str, any]
    ) -> Dict[str, any]:
        """Execute trade using generic exchange adapter."""
        if hasattr(self.exchange_adapter, 'fetch_ticker'):
            ticker = self.exchange_adapter.fetch_ticker(symbol)
            current_price = ticker.get('last', 0)
        else:
            logger.error("Exchange adapter does not support price fetching")
            return {'success': False, 'reason': 'exchange_not_supported'}
        
        if not current_price:
            logger.warning(f"Could not get price for {symbol}")
            return {'success': False, 'reason': 'price_unavailable'}
        
        position_size = self.calculate_position_size(
            symbol,
            opportunity.get('confidence', 0.0),
            opportunity.get('risk_level', 'medium')
        )
        
        quantity = position_size / current_price
        
        if hasattr(self.exchange_adapter, 'create_market_buy_order'):
            order = self.exchange_adapter.create_market_buy_order(
                symbol=symbol,
                amount=quantity
            )
            
            trade_result = {
                'success': True,
                'symbol': symbol,
                'order_id': order.get('id'),
                'quantity': quantity,
                'price': current_price,
                'amount_usdt': position_size,
                'opportunity': opportunity
            }
            
            self.trade_log.append(trade_result)
            self.daily_stats['positions_opened'] += 1
            logger.info(f"Trade executed successfully: {symbol}")
            
            return trade_result
        else:
            logger.error("Exchange adapter does not support market orders")
            return {'success': False, 'reason': 'order_not_supported'}
    
    def should_execute(self, opportunity: Dict[str, any]) -> bool:
        """Determine if trade should be executed based on opportunity."""
        if not self.enabled:
            return False
        
        confidence = opportunity.get('confidence', 0.0)
        min_confidence = self.config.get('min_execution_confidence', 0.6)
        
        return confidence >= min_confidence


class SocialSignalTrader:
    """
    Complete social signal trading system.
    
    Combines detection and execution for automated trading.
    Day 38 Enhanced: Auto-discovers accounts to follow based on early buyer analysis.
    """
    
    def __init__(
        self,
        twitter_collector=None,
        exchange_adapter=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize social signal trader.
        
        Args:
            twitter_collector: Twitter collector instance
            exchange_adapter: Exchange adapter for trading
            config: Configuration dictionary
        """
        self.twitter_collector = twitter_collector
        self.detector = SocialSignalDetector(config)
        self.executor = SocialSignalExecutor(exchange_adapter, config)
        self.config = config or {}
        
        # Initialize trader intelligence (Day 38)
        self.trader_intelligence = None
        if exchange_adapter:
            try:
                from ...research.trader_intelligence import TraderIntelligenceManager
                self.trader_intelligence = TraderIntelligenceManager(
                    exchange_adapter=exchange_adapter,
                    config=self.config.get('trader_intelligence', {})
                )
            except ImportError:
                logger.warning("Trader intelligence module not available")
        
        self.log_file = Path(self.config.get('log_file', './social_trading_log.csv'))
        self.trade_log_df = pd.DataFrame()
    
    def load_trade_log(self):
        """Load trade log from CSV."""
        if self.log_file.exists():
            try:
                self.trade_log_df = pd.read_csv(self.log_file)
                logger.info(f"Loaded {len(self.trade_log_df)} trade log entries")
            except Exception as e:
                logger.error(f"Error loading trade log: {e}")
                self.trade_log_df = pd.DataFrame()
    
    def save_trade_log(self, entry: Dict):
        """Save trade log entry to CSV."""
        try:
            new_row = pd.DataFrame([entry])
            self.trade_log_df = pd.concat([self.trade_log_df, new_row], ignore_index=True)
            self.trade_log_df.to_csv(self.log_file, index=False)
            logger.debug(f"Saved trade log entry")
        except Exception as e:
            logger.error(f"Error saving trade log: {e}")
    
    async def monitor_and_trade(self, target_accounts: List[str]):
        """
        Monitor Twitter accounts and execute trades when opportunities detected.
        
        Args:
            target_accounts: List of Twitter usernames to monitor
        """
        if not self.twitter_collector:
            logger.error("Twitter collector not available")
            return
        
        logger.info(f"Monitoring {len(target_accounts)} Twitter accounts...")
        
        for account in target_accounts:
            try:
                # Get tweets from user account
                tweets = await self.twitter_collector.get_user_tweets(
                    account,
                    count=10
                )
                
                # Process each tweet
                for tweet in tweets:
                    tweet_id = str(tweet.get('id', ''))
                    tweet_text = tweet.get('text', '')
                    
                    # Detect opportunities
                    opportunity = self.detector.detect_opportunities(
                        tweet_text,
                        tweet_id
                    )
                    
                    if opportunity:
                        logger.info(f"Opportunity detected: {opportunity}")
                        
                        # Check if should execute
                        if self.executor.should_execute(opportunity):
                            # Execute trades for detected tokens
                            for token in opportunity.get('tokens', []):
                                trade_result = self.executor.execute_trade(
                                    token,
                                    opportunity
                                )
                                
                                # Log trade
                                log_entry = {
                                    'timestamp': datetime.now().isoformat(),
                                    'account': account,
                                    'tweet_id': tweet_id,
                                    'token': token,
                                    'success': trade_result.get('success', False),
                                    'confidence': opportunity.get('confidence', 0.0)
                                }
                                self.save_trade_log(log_entry)
                                
                                if trade_result.get('success'):
                                    logger.info(f"Trade executed: {token}")
                        
                        # Handle Solana contract if present (log only, cannot trade on CEX)
                        solana_contract = opportunity.get('solana_contract')
                        if solana_contract:
                            logger.info(f"ℹ️ Solana contract detected: {solana_contract[:8]}... (not tradable on CEX - requires Jupiter)")
                            # Note: Solana contract addresses cannot be traded on Binance/Bitfinex
                            # They require Jupiter/DEX access which is not implemented
                
                # Rate limiting
                await asyncio.sleep(self.config.get('polling_interval', 4))
                
            except Exception as e:
                logger.error(f"Error monitoring {account}: {e}")
    
    async def monitor_accounts_dict(
        self,
        accounts: Dict[str, str],
        latest_tweet_ids: Optional[Dict[str, str]] = None
    ):
        """
        Monitor Twitter accounts by ID and execute trades when opportunities detected.
        
        Args:
            accounts: Dictionary mapping account_id -> username
            latest_tweet_ids: Dictionary mapping account_id -> latest processed tweet ID
        """
        if not self.twitter_collector:
            logger.error("Twitter collector not available")
            return
        
        if latest_tweet_ids is None:
            latest_tweet_ids = {}
        
        logger.info(f"Monitoring {len(accounts)} Twitter accounts...")
        
        # Use TwitterCollector's monitor_accounts method
        new_tweets = await self.twitter_collector.monitor_accounts(
            accounts,
            latest_tweet_ids,
            count_per_account=5
        )
        
        # Process new tweets
        for account_id, tweets in new_tweets.items():
            username = accounts.get(account_id, account_id)
            
            for tweet in tweets:
                tweet_id = str(tweet.get('id', ''))
                tweet_text = tweet.get('text', '')
                
                # Detect opportunities
                opportunity = self.detector.detect_opportunities(
                    tweet_text,
                    tweet_id
                )
                
                if opportunity:
                    logger.info(f"Opportunity detected from {username}: {opportunity}")
                    
                    # Check if should execute
                    if self.executor.should_execute(opportunity):
                        # Execute trades for detected tokens
                        for token in opportunity.get('tokens', []):
                            trade_result = self.executor.execute_trade(
                                token,
                                opportunity
                            )
                            
                            # Log trade
                            log_entry = {
                                'timestamp': datetime.now().isoformat(),
                                'account': username,
                                'account_id': account_id,
                                'tweet_id': tweet_id,
                                'token': token,
                                'success': trade_result.get('success', False),
                                'confidence': opportunity.get('confidence', 0.0)
                            }
                            self.save_trade_log(log_entry)
                            
                            if trade_result.get('success'):
                                logger.info(f"Trade executed: {token}")
                    
                    # Handle Solana contract if present (log only, cannot trade on CEX)
                    solana_contract = opportunity.get('solana_contract')
                    if solana_contract:
                        logger.info(f"ℹ️ Solana contract detected: {solana_contract[:8]}... (not tradable on CEX - requires Jupiter)")
                        # Note: Solana contract addresses cannot be traded on Binance/Bitfinex
                        # They require Jupiter/DEX access which is not implemented
        
        # Return updated latest_tweet_ids
        updated_ids = latest_tweet_ids.copy()
        for account_id, tweets in new_tweets.items():
            if tweets:
                # Update with the most recent tweet ID
                updated_ids[account_id] = tweets[0].get('id', '')
        
        return updated_ids
    
    async def discover_accounts_to_follow(
        self,
        symbol: str,
        max_accounts: int = 20
    ) -> List[str]:
        """
        Discover accounts to follow based on early buyer analysis (Day 38).
        
        Args:
            symbol: Trading symbol to analyze
            max_accounts: Maximum accounts to discover
            
        Returns:
            List of account IDs/usernames to follow
        """
        if not self.trader_intelligence:
            logger.warning("Trader intelligence not available")
            return []
        
        logger.info(f"Discovering accounts to follow for {symbol}")
        
        # Get accounts from trader intelligence
        accounts = self.trader_intelligence.get_accounts_to_follow(
            symbol=symbol,
            max_accounts=max_accounts
        )
        
        logger.info(f"Discovered {len(accounts)} accounts to follow")
        
        return accounts

