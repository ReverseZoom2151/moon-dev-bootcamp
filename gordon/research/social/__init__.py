"""
Social Media Research Module
============================
Day 28: Enhanced Twitter integration with sentiment analysis.

Combines Twitter data collection with sentiment analysis for trading insights.
"""

from .twitter_collector import TwitterCollector
from .sentiment_analyzer import SentimentAnalyzer

__all__ = [
    'TwitterCollector',
    'SentimentAnalyzer',
]

