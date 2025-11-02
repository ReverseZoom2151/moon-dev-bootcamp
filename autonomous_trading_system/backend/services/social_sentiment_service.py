"""
Social Sentiment Service
Based on Day_42_Projects TikTok agent and sentiment analysis implementation
"""

import asyncio
import logging
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Represents sentiment data from social media"""
    platform: str
    symbol: str
    sentiment_score: float  # -1 to 1
    confidence: float      # 0 to 1
    mention_count: int
    engagement_score: float
    timestamp: datetime
    raw_text: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class TrendingToken:
    """Represents a trending token from social media"""
    symbol: str
    address: Optional[str]
    platform: str
    mention_count: int
    sentiment_score: float
    engagement_rate: float
    first_seen: datetime
    last_updated: datetime
    keywords: List[str]


class SocialSentimentService:
    """
    Comprehensive social sentiment analysis service
    
    Features:
    - Multi-platform sentiment aggregation
    - Real-time trend detection
    - Token mention tracking
    - Sentiment momentum analysis
    - Alpha extraction from social signals
    """
    
    def __init__(self, config):
        self.config = config
        
        # Platform configurations
        self.tiktok_enabled = config.get('TIKTOK_SENTIMENT_ENABLED', False)
        self.twitter_enabled = config.get('TWITTER_SENTIMENT_ENABLED', False)
        self.reddit_enabled = config.get('REDDIT_SENTIMENT_ENABLED', False)
        
        # Sentiment thresholds
        self.sentiment_threshold = config.get('SENTIMENT_THRESHOLD', 0.7)
        self.social_volume_threshold = config.get('SOCIAL_VOLUME_THRESHOLD', 100)
        self.confidence_threshold = config.get('CONFIDENCE_THRESHOLD', 0.6)
        
        # Data storage
        self.sentiment_history = {}
        self.trending_tokens = {}
        self.platform_weights = {
            'tiktok': 0.4,
            'twitter': 0.35,
            'reddit': 0.25
        }
        
        # Crypto keywords for detection
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'solana', 'sol', 'trading', 'pump', 'dump', 'moon', 'bullish', 'bearish',
            'defi', 'nft', 'altcoin', 'hodl', 'diamond hands', 'paper hands',
            'to the moon', 'rug pull', 'whale', 'degen', 'ape', 'fomo'
        ]
        
        # Token patterns
        self.token_patterns = [
            r'\$([A-Z]{2,10})',  # $BTC, $ETH format
            r'#([A-Z]{2,10})',   # #BTC, #ETH format
            r'\b([A-Z]{2,10})\b(?=\s*(coin|token|crypto))',  # BTC coin, ETH token
        ]
        
        logger.info(f"🔍 Social Sentiment Service initialized")
        logger.info(f"📱 Platforms: TikTok={self.tiktok_enabled}, Twitter={self.twitter_enabled}, Reddit={self.reddit_enabled}")
    
    async def start(self):
        """Start the social sentiment service"""
        try:
            logger.info("🚀 Starting Social Sentiment Service...")
            
            # Start background tasks for each enabled platform
            tasks = []
            
            if self.tiktok_enabled:
                tasks.append(asyncio.create_task(self._tiktok_monitor()))
            
            if self.twitter_enabled:
                tasks.append(asyncio.create_task(self._twitter_monitor()))
            
            if self.reddit_enabled:
                tasks.append(asyncio.create_task(self._reddit_monitor()))
            
            # Start sentiment aggregation task
            tasks.append(asyncio.create_task(self._sentiment_aggregator()))
            
            # Start trend detection task
            tasks.append(asyncio.create_task(self._trend_detector()))
            
            logger.info(f"✅ Started {len(tasks)} social sentiment monitoring tasks")
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Error starting social sentiment service: {e}")
    
    async def _tiktok_monitor(self):
        """Monitor TikTok for crypto sentiment (based on Day_42 implementation)"""
        try:
            logger.info("📱 Starting TikTok sentiment monitoring...")
            
            while True:
                try:
                    # Simulate TikTok data collection
                    # In production, this would use the Day_42 TikTok scraper
                    tiktok_data = await self._simulate_tiktok_data()
                    
                    for data in tiktok_data:
                        await self._process_sentiment_data(data)
                    
                    # Wait before next collection
                    await asyncio.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"❌ Error in TikTok monitoring: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in TikTok monitor: {e}")
    
    async def _twitter_monitor(self):
        """Monitor Twitter for crypto sentiment"""
        try:
            logger.info("🐦 Starting Twitter sentiment monitoring...")
            
            # Try to use enhanced Twitter collector if available
            if hasattr(self, 'twitter_collector') and self.twitter_collector.enabled:
                logger.info("🔗 Using enhanced Twitter sentiment collector")
                while True:
                    try:
                        # Get recent sentiment from enhanced collector
                        recent_results = await self.twitter_collector.get_recent_sentiment(hours=1)
                        
                        for result in recent_results:
                            # Convert to SentimentData format
                            for mention in result.top_mentions[:5]:  # Top 5 mentions
                                sentiment_data = SentimentData(
                                    platform='twitter',
                                    symbol=mention,
                                    sentiment_score=result.average_sentiment,
                                    confidence=result.confidence,
                                    mention_count=result.mention_count,
                                    engagement_score=result.engagement_score / 100,  # Normalize
                                    timestamp=result.collection_time,
                                    raw_text=f"Twitter sentiment for {mention}",
                                    metadata={'source': 'enhanced_twitter_collector', 'query': result.query}
                                )
                                await self._process_sentiment_data(sentiment_data)
                        
                        await asyncio.sleep(300)  # 5 minutes
                        
                    except Exception as e:
                        logger.error(f"❌ Error in enhanced Twitter monitoring: {e}")
                        await asyncio.sleep(60)
            else:
                # Fallback to simulated data
                logger.info("📊 Using simulated Twitter data (enhanced collector not available)")
                while True:
                    try:
                        # Simulate Twitter data collection
                        twitter_data = await self._simulate_twitter_data()
                        
                        for data in twitter_data:
                            await self._process_sentiment_data(data)
                        
                        await asyncio.sleep(180)  # 3 minutes
                        
                    except Exception as e:
                        logger.error(f"❌ Error in Twitter monitoring: {e}")
                        await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in Twitter monitor: {e}")
    
    async def _reddit_monitor(self):
        """Monitor Reddit for crypto sentiment"""
        try:
            logger.info("🔴 Starting Reddit sentiment monitoring...")
            
            while True:
                try:
                    # Simulate Reddit data collection
                    reddit_data = await self._simulate_reddit_data()
                    
                    for data in reddit_data:
                        await self._process_sentiment_data(data)
                    
                    await asyncio.sleep(240)  # 4 minutes
                    
                except Exception as e:
                    logger.error(f"❌ Error in Reddit monitoring: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in Reddit monitor: {e}")
    
    async def _simulate_tiktok_data(self) -> List[SentimentData]:
        """Simulate TikTok sentiment data (replace with real scraper)"""
        try:
            # Simulate realistic TikTok sentiment data
            symbols = ['BTC', 'ETH', 'SOL', 'WIF', 'POPCAT', 'BONK']
            data = []
            
            for symbol in symbols:
                # Generate realistic sentiment
                base_sentiment = np.random.normal(0, 0.4)
                
                # TikTok tends to be more bullish and volatile
                tiktok_bias = 0.2
                sentiment_score = np.clip(base_sentiment + tiktok_bias, -1, 1)
                
                mention_count = np.random.randint(10, 200)
                engagement_score = np.random.uniform(0.4, 0.9)
                confidence = min(mention_count / 50, 1.0)
                
                data.append(SentimentData(
                    platform='tiktok',
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    mention_count=mention_count,
                    engagement_score=engagement_score,
                    timestamp=datetime.utcnow(),
                    raw_text=f"TikTok mentions about {symbol}",
                    metadata={'source': 'tiktok_scraper'}
                ))
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error simulating TikTok data: {e}")
            return []
    
    async def _simulate_twitter_data(self) -> List[SentimentData]:
        """Simulate Twitter sentiment data"""
        try:
            symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
            data = []
            
            for symbol in symbols:
                # Twitter sentiment tends to be more analytical
                sentiment_score = np.random.normal(0, 0.3)
                sentiment_score = np.clip(sentiment_score, -1, 1)
                
                mention_count = np.random.randint(50, 500)
                engagement_score = np.random.uniform(0.3, 0.8)
                confidence = min(mention_count / 100, 1.0)
                
                data.append(SentimentData(
                    platform='twitter',
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    mention_count=mention_count,
                    engagement_score=engagement_score,
                    timestamp=datetime.utcnow(),
                    raw_text=f"Twitter discussions about {symbol}",
                    metadata={'source': 'twitter_api'}
                ))
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error simulating Twitter data: {e}")
            return []
    
    async def _simulate_reddit_data(self) -> List[SentimentData]:
        """Simulate Reddit sentiment data"""
        try:
            symbols = ['BTC', 'ETH', 'SOL', 'LINK', 'UNI']
            data = []
            
            for symbol in symbols:
                # Reddit tends to be more technical and less emotional
                sentiment_score = np.random.normal(0, 0.25)
                sentiment_score = np.clip(sentiment_score, -1, 1)
                
                mention_count = np.random.randint(20, 150)
                engagement_score = np.random.uniform(0.5, 0.9)
                confidence = min(mention_count / 75, 1.0)
                
                data.append(SentimentData(
                    platform='reddit',
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    mention_count=mention_count,
                    engagement_score=engagement_score,
                    timestamp=datetime.utcnow(),
                    raw_text=f"Reddit discussions about {symbol}",
                    metadata={'source': 'reddit_api'}
                ))
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error simulating Reddit data: {e}")
            return []
    
    async def _process_sentiment_data(self, data: SentimentData):
        """Process and store sentiment data"""
        try:
            symbol = data.symbol
            
            # Initialize symbol history if not exists
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = {
                    'tiktok': [],
                    'twitter': [],
                    'reddit': []
                }
            
            # Add to platform-specific history
            platform_history = self.sentiment_history[symbol][data.platform]
            platform_history.append(data)
            
            # Keep only recent data (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.sentiment_history[symbol][data.platform] = [
                d for d in platform_history if d.timestamp > cutoff_time
            ]
            
            # Check for trending tokens
            await self._check_trending_token(data)
            
        except Exception as e:
            logger.error(f"❌ Error processing sentiment data: {e}")
    
    async def _check_trending_token(self, data: SentimentData):
        """Check if a token is trending based on sentiment data"""
        try:
            symbol = data.symbol
            
            # Check if token meets trending criteria
            if (data.mention_count >= self.social_volume_threshold and
                abs(data.sentiment_score) >= self.sentiment_threshold and
                data.confidence >= self.confidence_threshold):
                
                if symbol not in self.trending_tokens:
                    self.trending_tokens[symbol] = TrendingToken(
                        symbol=symbol,
                        address=None,  # Would be populated from token database
                        platform=data.platform,
                        mention_count=data.mention_count,
                        sentiment_score=data.sentiment_score,
                        engagement_rate=data.engagement_score,
                        first_seen=data.timestamp,
                        last_updated=data.timestamp,
                        keywords=self._extract_keywords(data.raw_text or "")
                    )
                    
                    logger.info(f"🔥 NEW TRENDING TOKEN: {symbol} on {data.platform}")
                    logger.info(f"   Sentiment: {data.sentiment_score:.2f}, Mentions: {data.mention_count}")
                else:
                    # Update existing trending token
                    trending = self.trending_tokens[symbol]
                    trending.mention_count += data.mention_count
                    trending.sentiment_score = (trending.sentiment_score + data.sentiment_score) / 2
                    trending.last_updated = data.timestamp
            
        except Exception as e:
            logger.error(f"❌ Error checking trending token: {e}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        try:
            if not text:
                return []
            
            text_lower = text.lower()
            keywords = []
            
            # Extract crypto-related keywords
            for keyword in self.crypto_keywords:
                if keyword in text_lower:
                    keywords.append(keyword)
            
            # Extract token symbols
            for pattern in self.token_patterns:
                matches = re.findall(pattern, text.upper())
                keywords.extend(matches)
            
            return list(set(keywords))
            
        except Exception as e:
            logger.error(f"❌ Error extracting keywords: {e}")
            return []
    
    async def _sentiment_aggregator(self):
        """Aggregate sentiment data across platforms"""
        try:
            logger.info("📊 Starting sentiment aggregation task...")
            
            while True:
                try:
                    await self._aggregate_platform_sentiment()
                    await asyncio.sleep(60)  # Aggregate every minute
                    
                except Exception as e:
                    logger.error(f"❌ Error in sentiment aggregation: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in sentiment aggregator: {e}")
    
    async def _aggregate_platform_sentiment(self):
        """Aggregate sentiment across all platforms for each symbol"""
        try:
            for symbol in self.sentiment_history:
                platform_sentiments = {}
                
                # Get recent sentiment for each platform
                for platform in ['tiktok', 'twitter', 'reddit']:
                    recent_data = [
                        d for d in self.sentiment_history[symbol][platform]
                        if d.timestamp > datetime.utcnow() - timedelta(hours=1)
                    ]
                    
                    if recent_data:
                        # Calculate weighted average sentiment
                        total_weight = sum(d.confidence * d.mention_count for d in recent_data)
                        if total_weight > 0:
                            weighted_sentiment = sum(
                                d.sentiment_score * d.confidence * d.mention_count 
                                for d in recent_data
                            ) / total_weight
                            
                            platform_sentiments[platform] = {
                                'sentiment': weighted_sentiment,
                                'mentions': sum(d.mention_count for d in recent_data),
                                'confidence': sum(d.confidence for d in recent_data) / len(recent_data)
                            }
                
                # Calculate overall sentiment
                if platform_sentiments:
                    overall_sentiment = self._calculate_overall_sentiment(platform_sentiments)
                    
                    # Store aggregated sentiment
                    if not hasattr(self, 'aggregated_sentiment'):
                        self.aggregated_sentiment = {}
                    
                    self.aggregated_sentiment[symbol] = {
                        'overall_sentiment': overall_sentiment,
                        'platform_breakdown': platform_sentiments,
                        'timestamp': datetime.utcnow()
                    }
            
        except Exception as e:
            logger.error(f"❌ Error aggregating platform sentiment: {e}")
    
    def _calculate_overall_sentiment(self, platform_sentiments: Dict) -> Dict:
        """Calculate overall sentiment from platform sentiments"""
        try:
            total_weighted_sentiment = 0
            total_weight = 0
            total_mentions = 0
            
            for platform, data in platform_sentiments.items():
                weight = self.platform_weights.get(platform, 0.33)
                mention_weight = data['mentions'] * data['confidence']
                
                total_weighted_sentiment += data['sentiment'] * weight * mention_weight
                total_weight += weight * mention_weight
                total_mentions += data['mentions']
            
            if total_weight > 0:
                overall_sentiment = total_weighted_sentiment / total_weight
            else:
                overall_sentiment = 0
            
            return {
                'sentiment_score': overall_sentiment,
                'total_mentions': total_mentions,
                'confidence': min(total_mentions / 100, 1.0),
                'platforms_count': len(platform_sentiments)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating overall sentiment: {e}")
            return {'sentiment_score': 0, 'total_mentions': 0, 'confidence': 0, 'platforms_count': 0}
    
    async def _trend_detector(self):
        """Detect emerging trends and sentiment momentum"""
        try:
            logger.info("📈 Starting trend detection task...")
            
            while True:
                try:
                    await self._detect_sentiment_trends()
                    await asyncio.sleep(300)  # Check trends every 5 minutes
                    
                except Exception as e:
                    logger.error(f"❌ Error in trend detection: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in trend detector: {e}")
    
    async def _detect_sentiment_trends(self):
        """Detect sentiment trends and momentum"""
        try:
            if not hasattr(self, 'aggregated_sentiment'):
                return
            
            for symbol, data in self.aggregated_sentiment.items():
                # Get historical sentiment for trend analysis
                historical_sentiment = self._get_historical_sentiment(symbol, hours=6)
                
                if len(historical_sentiment) >= 3:
                    # Calculate sentiment momentum
                    recent_scores = [s['sentiment_score'] for s in historical_sentiment[-3:]]
                    momentum = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                    
                    # Detect significant momentum changes
                    if abs(momentum) > 0.1:  # Significant momentum threshold
                        trend_type = "BULLISH" if momentum > 0 else "BEARISH"
                        logger.info(f"🚨 SENTIMENT TREND DETECTED: {symbol} - {trend_type}")
                        logger.info(f"   Momentum: {momentum:.3f}, Current: {data['overall_sentiment']['sentiment_score']:.3f}")
                        
                        # Store trend data
                        if not hasattr(self, 'sentiment_trends'):
                            self.sentiment_trends = {}
                        
                        self.sentiment_trends[symbol] = {
                            'trend_type': trend_type,
                            'momentum': momentum,
                            'current_sentiment': data['overall_sentiment']['sentiment_score'],
                            'confidence': data['overall_sentiment']['confidence'],
                            'detected_at': datetime.utcnow()
                        }
            
        except Exception as e:
            logger.error(f"❌ Error detecting sentiment trends: {e}")
    
    def _get_historical_sentiment(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get historical sentiment data for a symbol"""
        try:
            if not hasattr(self, 'aggregated_sentiment') or symbol not in self.aggregated_sentiment:
                return []
            
            # This is a simplified version - in production, you'd store historical data
            # For now, return current data as historical
            return [self.aggregated_sentiment[symbol]['overall_sentiment']]
            
        except Exception as e:
            logger.error(f"❌ Error getting historical sentiment: {e}")
            return []
    
    async def get_symbol_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get current sentiment data for a symbol"""
        try:
            if hasattr(self, 'aggregated_sentiment') and symbol in self.aggregated_sentiment:
                return self.aggregated_sentiment[symbol]
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting symbol sentiment: {e}")
            return None
    
    async def get_trending_tokens(self) -> List[TrendingToken]:
        """Get list of currently trending tokens"""
        try:
            # Filter trending tokens by recency (last 2 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=2)
            
            trending = [
                token for token in self.trending_tokens.values()
                if token.last_updated > cutoff_time
            ]
            
            # Sort by mention count and sentiment strength
            trending.sort(
                key=lambda t: t.mention_count * abs(t.sentiment_score),
                reverse=True
            )
            
            return trending[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"❌ Error getting trending tokens: {e}")
            return []
    
    async def get_sentiment_signals(self) -> List[Dict]:
        """Get trading signals based on sentiment analysis"""
        try:
            signals = []
            
            if not hasattr(self, 'sentiment_trends'):
                return signals
            
            for symbol, trend_data in self.sentiment_trends.items():
                # Generate signal based on sentiment trend
                if trend_data['confidence'] >= self.confidence_threshold:
                    signal_strength = abs(trend_data['momentum']) * trend_data['confidence']
                    
                    if signal_strength >= 0.5:  # Minimum signal strength
                        signals.append({
                            'symbol': symbol,
                            'action': 'buy' if trend_data['trend_type'] == 'BULLISH' else 'sell',
                            'confidence': trend_data['confidence'],
                            'sentiment_score': trend_data['current_sentiment'],
                            'momentum': trend_data['momentum'],
                            'signal_strength': signal_strength,
                            'source': 'social_sentiment',
                            'timestamp': trend_data['detected_at']
                        })
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment signals: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup service resources"""
        try:
            # Clear data structures
            self.sentiment_history.clear()
            self.trending_tokens.clear()
            
            if hasattr(self, 'aggregated_sentiment'):
                self.aggregated_sentiment.clear()
            
            if hasattr(self, 'sentiment_trends'):
                self.sentiment_trends.clear()
            
            logger.info("🧹 Social sentiment service cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up social sentiment service: {e}") 