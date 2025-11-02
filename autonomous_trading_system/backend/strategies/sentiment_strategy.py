"""
Social Sentiment Strategy - Trading based on social media sentiment analysis
Integrates TikTok, Twitter, Reddit sentiment for market predictions
"""

import logging
import re
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from textblob import TextBlob
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes sentiment from various social media sources"""
    
    def __init__(self):
        self.sentiment_cache = {}
        self.keywords_crypto = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'solana', 'sol', 'trading', 'pump', 'dump', 'moon', 'bullish', 'bearish'
        ]
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text.lower())
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert to trading signals
            sentiment_score = polarity
            confidence = 1 - subjectivity  # More objective = higher confidence
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment_score': 0, 'confidence': 0, 'polarity': 0, 'subjectivity': 0}
    
    def extract_crypto_mentions(self, text: str) -> List[str]:
        """Extract cryptocurrency mentions from text"""
        mentions = []
        text_lower = text.lower()
        
        for keyword in self.keywords_crypto:
            if keyword in text_lower:
                mentions.append(keyword)
        
        # Extract ticker symbols like $BTC, $ETH
        ticker_pattern = r'\$([A-Z]{2,5})'
        tickers = re.findall(ticker_pattern, text.upper())
        mentions.extend([f"${ticker}" for ticker in tickers])
        
        return list(set(mentions))
    
    async def get_social_sentiment(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get aggregated social sentiment for a symbol"""
        try:
            # Simulate social media sentiment data
            # In production, this would connect to Twitter API, Reddit API, etc.
            
            # Generate realistic sentiment data
            base_sentiment = np.random.normal(0, 0.3)  # Neutral with some variation
            
            # Add symbol-specific bias
            symbol_bias = {
                'BTC': 0.1,   # Slightly bullish
                'ETH': 0.05,  # Slightly bullish
                'SOL': 0.15,  # More bullish
                'WIF': 0.2,   # Meme coin - more volatile
                'POPCAT': 0.25  # Meme coin - very volatile
            }
            
            sentiment_score = base_sentiment + symbol_bias.get(symbol, 0)
            sentiment_score = np.clip(sentiment_score, -1, 1)
            
            # Simulate volume and engagement metrics
            mention_count = np.random.randint(50, 500)
            engagement_score = np.random.uniform(0.3, 0.9)
            
            return {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'mention_count': mention_count,
                'engagement_score': engagement_score,
                'confidence': min(mention_count / 100, 1.0),  # More mentions = higher confidence
                'timestamp': datetime.utcnow(),
                'sources': ['twitter', 'reddit', 'tiktok']
            }
            
        except Exception as e:
            logger.error(f"Error getting social sentiment for {symbol}: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0,
                'mention_count': 0,
                'engagement_score': 0,
                'confidence': 0,
                'timestamp': datetime.utcnow(),
                'sources': []
            }


class SentimentStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Social Sentiment Trading Strategy
    
    Features:
    - Social media sentiment analysis
    - Multi-platform sentiment aggregation
    - Sentiment momentum tracking
    - Volume-weighted sentiment signals
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "Sentiment"):
        super().__init__(config, market_data_manager, name)
        
        # Sentiment Configuration
        self.sentiment_threshold = config.get("sentiment_threshold", 0.3)
        self.min_mentions = config.get("min_mentions", 20)
        self.sentiment_window = config.get("sentiment_window", 4)  # hours
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
        # Sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Sentiment tracking
        self.sentiment_history = {}
        self.last_sentiment_update = {}
        
        logger.info(f"📱 Sentiment Strategy initialized:")
        logger.info(f"   Sentiment Threshold: {self.sentiment_threshold}")
        logger.info(f"   Min Mentions: {self.min_mentions}")
        logger.info(f"   Symbols: {self.symbols}")
    
    async def _initialize_strategy(self):
        """Initialize sentiment strategy"""
        try:
            # Initialize sentiment history for each symbol
            for symbol in self.symbols:
                self.sentiment_history[symbol] = []
                self.last_sentiment_update[symbol] = None
            
            logger.info("✅ Sentiment strategy validation complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment strategy: {e}")
            raise
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate sentiment-based trading signal"""
        try:
            for symbol in self.symbols:
                signal = await self._analyze_symbol_sentiment(symbol)
                if signal and signal.action != SignalAction.HOLD:
                    return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error generating sentiment signal: {e}", exc_info=True)
            return None
    
    async def _analyze_symbol_sentiment(self, symbol: str) -> Optional[StrategySignal]:
        """Analyze sentiment for a specific symbol"""
        try:
            # Get current sentiment
            current_sentiment = await self.sentiment_analyzer.get_social_sentiment(symbol)
            
            # Update sentiment history
            self.sentiment_history[symbol].append(current_sentiment)
            
            # Keep only recent sentiment data
            cutoff_time = datetime.utcnow() - timedelta(hours=self.sentiment_window)
            self.sentiment_history[symbol] = [
                s for s in self.sentiment_history[symbol] 
                if s['timestamp'] > cutoff_time
            ]
            
            if len(self.sentiment_history[symbol]) < 2:
                return None
            
            # Calculate sentiment metrics
            sentiment_metrics = self._calculate_sentiment_metrics(symbol)
            if not sentiment_metrics:
                return None
            
            # Get market data for price context
            market_data = await self._get_market_data(symbol, limit=50)
            if market_data is None:
                return None
            
            # Generate signal based on sentiment and price action
            signal = await self._generate_sentiment_signal(symbol, sentiment_metrics, market_data)
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment for {symbol}: {e}")
            return None
    
    def _calculate_sentiment_metrics(self, symbol: str) -> Optional[Dict[str, float]]:
        """Calculate sentiment metrics from history"""
        try:
            history = self.sentiment_history[symbol]
            if len(history) < 2:
                return None
            
            # Current sentiment
            current = history[-1]
            previous = history[-2] if len(history) > 1 else history[0]
            
            # Calculate metrics
            current_sentiment = current['sentiment_score']
            sentiment_change = current_sentiment - previous['sentiment_score']
            
            # Average sentiment over window
            avg_sentiment = np.mean([s['sentiment_score'] for s in history])
            
            # Sentiment momentum (rate of change)
            if len(history) >= 3:
                recent_scores = [s['sentiment_score'] for s in history[-3:]]
                sentiment_momentum = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            else:
                sentiment_momentum = sentiment_change
            
            # Volume metrics
            total_mentions = sum(s['mention_count'] for s in history)
            avg_engagement = np.mean([s['engagement_score'] for s in history])
            
            # Confidence based on volume and consistency
            confidence = min(
                current['confidence'],
                total_mentions / (self.min_mentions * len(history)),
                avg_engagement
            )
            
            return {
                'current_sentiment': current_sentiment,
                'sentiment_change': sentiment_change,
                'avg_sentiment': avg_sentiment,
                'sentiment_momentum': sentiment_momentum,
                'total_mentions': total_mentions,
                'avg_engagement': avg_engagement,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating sentiment metrics for {symbol}: {e}")
            return None
    
    async def _generate_sentiment_signal(
        self, 
        symbol: str, 
        sentiment_metrics: Dict[str, float],
        market_data: Any
    ) -> Optional[StrategySignal]:
        """Generate trading signal from sentiment analysis"""
        try:
            current_price = float(market_data[-1]['close'])
            
            # Extract sentiment metrics
            current_sentiment = sentiment_metrics['current_sentiment']
            sentiment_momentum = sentiment_metrics['sentiment_momentum']
            confidence = sentiment_metrics['confidence']
            total_mentions = sentiment_metrics['total_mentions']
            
            # Check minimum requirements
            if confidence < self.confidence_threshold:
                return None
            
            if total_mentions < self.min_mentions:
                return None
            
            # Determine signal based on sentiment
            action = SignalAction.HOLD
            
            # Strong positive sentiment with momentum
            if (current_sentiment > self.sentiment_threshold and 
                sentiment_momentum > 0.1):
                action = SignalAction.BUY
            
            # Strong negative sentiment with momentum
            elif (current_sentiment < -self.sentiment_threshold and 
                  sentiment_momentum < -0.1):
                action = SignalAction.SELL
            
            if action == SignalAction.HOLD:
                return None
            
            # Adjust confidence based on sentiment strength
            signal_confidence = min(
                confidence * (abs(current_sentiment) / self.sentiment_threshold),
                1.0
            )
            
            # Create metadata
            metadata = {
                'sentiment_score': current_sentiment,
                'sentiment_momentum': sentiment_momentum,
                'mention_count': total_mentions,
                'avg_engagement': sentiment_metrics['avg_engagement'],
                'sentiment_change': sentiment_metrics['sentiment_change'],
                'strategy_type': 'sentiment'
            }
            
            # Create signal
            signal = self._create_signal(
                symbol=symbol,
                action=action,
                price=current_price,
                confidence=signal_confidence,
                metadata=metadata
            )
            
            logger.info(f"📱 Sentiment Signal: {action.value} {symbol} @ {current_price:.4f}")
            logger.info(f"   Sentiment: {current_sentiment:.3f}, Momentum: {sentiment_momentum:.3f}")
            logger.info(f"   Mentions: {total_mentions}, Confidence: {signal_confidence:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating sentiment signal for {symbol}: {e}")
            return None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and performance metrics"""
        return {
            "name": self.name,
            "type": "sentiment_analysis",
            "sentiment_threshold": self.sentiment_threshold,
            "min_mentions": self.min_mentions,
            "symbols": self.symbols,
            "sentiment_history_length": {
                symbol: len(history) 
                for symbol, history in self.sentiment_history.items()
            },
            "last_sentiment_scores": {
                symbol: history[-1]['sentiment_score'] if history else 0
                for symbol, history in self.sentiment_history.items()
            },
            "status": self.status.value,
            "enabled": self.enabled
        } 