"""
Sentiment Analyzer
==================
Sentiment analysis for tweets and social media content.

Features:
- Multiple sentiment analysis methods (OpenAI, VADER, TextBlob)
- Sentiment scoring and classification
- Symbol/token extraction from tweets
- Aggregate sentiment metrics
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    SentimentIntensityAnalyzer = None

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    TextBlob = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class SentimentAnalyzer:
    """
    Sentiment analyzer for social media content.
    
    Supports multiple analysis methods and provides aggregate sentiment scores.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            config: Configuration dictionary with:
                - openai_api_key: OpenAI API key (optional)
                - method: 'vader', 'textblob', 'openai', or 'composite'
        """
        self.config = config or {}
        self.method = self.config.get('method', 'composite')
        self.openai_api_key = self.config.get('openai_api_key')
        
        # Initialize analyzers
        self.vader_analyzer = None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        self.openai_client = None
        if OPENAI_AVAILABLE and self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)

    def _analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.vader_analyzer:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        scores = self.vader_analyzer.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'neutral': scores['neu'],
            'negative': scores['neg']
        }

    def _analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not TEXTBLOB_AVAILABLE:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    async def _analyze_openai(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenAI.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        if not self.openai_client:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analyzer. Analyze the sentiment of the following text and respond with JSON: {\"sentiment\": \"positive\"|\"negative\"|\"neutral\", \"confidence\": 0.0-1.0, \"reasoning\": \"brief explanation\"}"
                    },
                    {
                        "role": "user",
                        "content": text[:500]  # Limit length
                    }
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            content = response.choices[0].message.content
            # Try to parse JSON response
            import json
            try:
                result = json.loads(content)
                return result
            except:
                # Fallback if not valid JSON
                sentiment_map = {
                    'positive': 1.0,
                    'negative': -1.0,
                    'neutral': 0.0
                }
                sentiment = 'neutral'
                for key in sentiment_map:
                    if key in content.lower():
                        sentiment = key
                        break
                
                return {
                    'sentiment': sentiment,
                    'confidence': 0.7,
                    'reasoning': content
                }
                
        except Exception as e:
            logger.error(f"Error in OpenAI sentiment analysis: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5}

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using configured method.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'method': self.method
            }
        
        results = {}
        
        # Run VADER analysis
        if VADER_AVAILABLE:
            vader_results = self._analyze_vader(text)
            results['vader'] = vader_results
        
        # Run TextBlob analysis
        if TEXTBLOB_AVAILABLE:
            textblob_results = self._analyze_textblob(text)
            results['textblob'] = textblob_results
        
        # Determine overall sentiment
        if self.method == 'composite':
            sentiment_score = self._compute_composite_sentiment(results)
        elif self.method == 'vader' and 'vader' in results:
            sentiment_score = results['vader']['compound']
        elif self.method == 'textblob' and 'textblob' in results:
            sentiment_score = results['textblob']['polarity']
        else:
            # Default to VADER if available
            if 'vader' in results:
                sentiment_score = results['vader']['compound']
            elif 'textblob' in results:
                sentiment_score = results['textblob']['polarity']
            else:
                sentiment_score = 0.0
        
        # Classify sentiment
        if sentiment_score > 0.05:
            sentiment = 'positive'
        elif sentiment_score < -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'confidence': abs(sentiment_score),
            'method': self.method,
            'details': results
        }

    async def analyze_async(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment asynchronously (supports OpenAI).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if self.method == 'openai' and self.openai_client:
            openai_result = await self._analyze_openai(text)
            return {
                'sentiment': openai_result.get('sentiment', 'neutral'),
                'score': self._sentiment_to_score(openai_result.get('sentiment', 'neutral')),
                'confidence': openai_result.get('confidence', 0.5),
                'method': 'openai',
                'details': openai_result
            }
        else:
            return self.analyze(text)

    def _compute_composite_sentiment(self, results: Dict[str, Any]) -> float:
        """
        Compute composite sentiment score from multiple methods.
        
        Args:
            results: Dictionary with analysis results from different methods
            
        Returns:
            Composite sentiment score (-1 to 1)
        """
        scores = []
        weights = []
        
        if 'vader' in results:
            scores.append(results['vader']['compound'])
            weights.append(0.6)  # VADER is good for social media
        
        if 'textblob' in results:
            scores.append(results['textblob']['polarity'])
            weights.append(0.4)
        
        if not scores:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return composite

    def _sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment string to numeric score."""
        sentiment_map = {
            'positive': 0.5,
            'negative': -0.5,
            'neutral': 0.0
        }
        return sentiment_map.get(sentiment.lower(), 0.0)

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        return [self.analyze(text) for text in texts]

    def get_aggregate_sentiment(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute aggregate sentiment from multiple analyses.
        
        Args:
            analyses: List of sentiment analysis results
            
        Returns:
            Dictionary with aggregate metrics
        """
        if not analyses:
            return {
                'average_score': 0.0,
                'sentiment': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_count': 0
            }
        
        scores = [a.get('score', 0.0) for a in analyses]
        sentiments = [a.get('sentiment', 'neutral') for a in analyses]
        
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        positive_count = sum(1 for s in sentiments if s == 'positive')
        negative_count = sum(1 for s in sentiments if s == 'negative')
        neutral_count = sum(1 for s in sentiments if s == 'neutral')
        
        # Determine overall sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            overall_sentiment = 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'average_score': average_score,
            'sentiment': overall_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_count': len(analyses),
            'positive_pct': positive_count / len(analyses) * 100,
            'negative_pct': negative_count / len(analyses) * 100,
            'neutral_pct': neutral_count / len(analyses) * 100
        }

    def extract_symbols(self, text: str) -> List[str]:
        """
        Extract trading symbols from text (e.g., $BTC, SOL, etc.).
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted symbols
        """
        symbols = []
        
        # Pattern for $SYMBOL format
        dollar_pattern = r'\$([A-Z]{2,10})'
        matches = re.findall(dollar_pattern, text)
        symbols.extend(matches)
        
        # Pattern for common crypto symbols (case-insensitive)
        crypto_pattern = r'\b(BTC|ETH|SOL|USDT|USDC|BNB|ADA|DOGE|XRP|MATIC|AVAX|DOT|LINK|UNI|LTC|ATOM|FIL|TRX|ETC|XLM)\b'
        matches = re.findall(crypto_pattern, text, re.IGNORECASE)
        symbols.extend([m.upper() for m in matches])
        
        # Remove duplicates and return
        return list(set(symbols))

