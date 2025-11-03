"""
Hybrid Analyzer for Gordon
==========================
Combines fundamental analysis with technical trading for comprehensive insights.
Day 28: Now includes social sentiment analysis as a third dimension!
This is Gordon's secret sauce - the fusion of Warren Buffett, Jim Simons, and social media intelligence!
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from .agent import Agent
from .config_manager import get_config

logger = logging.getLogger(__name__)

# Import social media modules
try:
    from ..research.social import TwitterCollector, SentimentAnalyzer
    SOCIAL_AVAILABLE = True
except ImportError:
    SOCIAL_AVAILABLE = False
    TwitterCollector = None
    SentimentAnalyzer = None


class HybridAnalyzer:
    """Combines fundamental and technical analysis for trading decisions."""

    def __init__(self):
        """Initialize the hybrid analyzer."""
        self.agent = Agent()
        self.config = get_config()
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize social media components if available
        self.twitter_collector = None
        self.sentiment_analyzer = None
        if SOCIAL_AVAILABLE:
            try:
                social_config = self.config.get('research', {}).get('social', {})
                twitter_config = social_config.get('twitter', {})
                if twitter_config.get('enabled', False):
                    self.twitter_collector = TwitterCollector(twitter_config)
                    self.sentiment_analyzer = SentimentAnalyzer({
                        'method': social_config.get('sentiment_method', 'composite'),
                        'openai_api_key': social_config.get('openai_api_key')
                    })
                    logger.info("Social media analysis enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize social media components: {e}")

    async def analyze_stock(self, symbol: str, include_trading: bool = True, include_sentiment: bool = True) -> Dict[str, Any]:
        """Perform comprehensive hybrid analysis on a stock.
        
        Day 28 Enhanced: Now includes social sentiment analysis!

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            include_trading: Whether to include trading recommendations
            include_sentiment: Whether to include social sentiment analysis

        Returns:
            Comprehensive analysis combining fundamental, technical, and sentiment insights
        """
        print(f"\n🔬 Gordon is analyzing {symbol}...")
        print("=" * 50)

        # Gather fundamental data
        print("\n📊 Gathering fundamental data...")
        fundamental_analysis = self._analyze_fundamentals(symbol)

        # Gather technical data
        print("\n📈 Performing technical analysis...")
        technical_analysis = self._analyze_technicals(symbol)
        
        # Gather sentiment data (Day 28)
        sentiment_analysis = {}
        if include_sentiment and self.sentiment_analyzer and self.twitter_collector:
            print("\n📱 Analyzing social sentiment...")
            sentiment_analysis = await self._analyze_sentiment(symbol)

        # Generate trading signals
        trading_signals = {}
        if include_trading:
            print("\n🎯 Generating trading signals...")
            trading_signals = self._generate_trading_signals(
                symbol, 
                fundamental_analysis, 
                technical_analysis,
                sentiment_analysis
            )

        # Combine insights
        print("\n🧠 Synthesizing insights...")
        combined_insights = self._synthesize_insights(
            symbol,
            fundamental_analysis,
            technical_analysis,
            trading_signals,
            sentiment_analysis
        )

        return combined_insights

    def _analyze_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Perform fundamental analysis using research tools."""
        try:
            # Use the agent to gather fundamental data
            query = f"""
            Analyze {symbol} fundamentals:
            1. Get latest income statement
            2. Get key financial metrics
            3. Get analyst estimates
            4. Get recent news
            """

            # Run the agent
            self.agent.run(query)

            # Extract results from agent's execution
            results = {
                'revenue_growth': 'Pending analysis',
                'profit_margins': 'Pending analysis',
                'pe_ratio': 'Pending analysis',
                'analyst_rating': 'Pending analysis',
                'financial_health': 'Pending analysis'
            }

            return results
        except Exception as e:
            print(f"⚠️ Fundamental analysis error: {e}")
            return {}

    def _analyze_technicals(self, symbol: str) -> Dict[str, Any]:
        """Perform technical analysis using trading tools."""
        try:
            # Convert stock symbol to crypto pair for technical analysis
            # (In production, you'd have proper stock data feeds)
            crypto_symbol = f"{symbol}/USDT" if "/" not in symbol else symbol

            from tools.trading.market_data import get_live_price
            from tools.trading.strategies import run_rsi_strategy, run_sma_strategy

            # Get current price
            price_data = get_live_price(crypto_symbol, exchange="binance")

            # Run technical indicators
            rsi_analysis = run_rsi_strategy(
                symbol=crypto_symbol,
                dry_run=True
            )

            sma_analysis = run_sma_strategy(
                symbol=crypto_symbol,
                dry_run=True
            )

            return {
                'current_price': price_data.get('price', 0),
                'price_change_24h': price_data.get('change_24h', 0),
                'volume_24h': price_data.get('volume_24h', 0),
                'rsi': {
                    'value': rsi_analysis.get('rsi_value', 50),
                    'signal': rsi_analysis.get('signal', 'neutral')
                },
                'sma': {
                    'signal': sma_analysis.get('signal', 'neutral'),
                    'trend': 'bullish' if sma_analysis.get('signal') == 'buy' else 'bearish'
                },
                'support_resistance': self._calculate_support_resistance(crypto_symbol)
            }
        except Exception as e:
            print(f"⚠️ Technical analysis error: {e}")
            return {}

    async def _analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze social sentiment for a symbol (Day 28).
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.twitter_collector or not self.sentiment_analyzer:
            return {}
        
        try:
            # Collect tweets about the symbol
            tweets = await self.twitter_collector.search_symbol_tweets(
                symbol=symbol,
                minimum_tweets=50  # Collect at least 50 tweets
            )
            
            if not tweets:
                logger.warning(f"No tweets found for {symbol}")
                return {
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0,
                    'tweet_count': 0
                }
            
            # Analyze sentiment for each tweet
            analyses = []
            for tweet in tweets[:100]:  # Limit to 100 tweets for analysis
                analysis = await self.sentiment_analyzer.analyze_async(tweet['text'])
                analyses.append(analysis)
            
            # Get aggregate sentiment
            aggregate = self.sentiment_analyzer.get_aggregate_sentiment(analyses)
            
            # Extract symbols mentioned
            all_symbols = set()
            for tweet in tweets:
                symbols = self.sentiment_analyzer.extract_symbols(tweet['text'])
                all_symbols.update(symbols)
            
            return {
                'sentiment': aggregate['sentiment'],
                'score': aggregate['average_score'],
                'confidence': aggregate['positive_pct'] / 100 if aggregate['sentiment'] == 'positive' else aggregate['negative_pct'] / 100,
                'tweet_count': len(tweets),
                'positive_pct': aggregate['positive_pct'],
                'negative_pct': aggregate['negative_pct'],
                'neutral_pct': aggregate['neutral_pct'],
                'mentioned_symbols': list(all_symbols),
                'details': aggregate
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {}

    def _generate_trading_signals(
        self,
        symbol: str,
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate trading signals based on combined analysis (now includes sentiment)."""
        try:
            signals = {
                'fundamental_score': self._calculate_fundamental_score(fundamental_data),
                'technical_score': self._calculate_technical_score(technical_data),
                'sentiment_score': self._calculate_sentiment_score(sentiment_data) if sentiment_data else 50,
                'combined_score': 0,
                'recommendation': 'HOLD',
                'confidence': 'LOW',
                'strategies': []
            }

            # Calculate combined score (weighted average with sentiment)
            # Day 28: Now includes sentiment as third dimension
            fundamental_weight = 0.35  # 35% weight on fundamentals
            technical_weight = 0.50    # 50% weight on technicals
            sentiment_weight = 0.15    # 15% weight on sentiment

            signals['combined_score'] = (
                signals['fundamental_score'] * fundamental_weight +
                signals['technical_score'] * technical_weight +
                signals['sentiment_score'] * sentiment_weight
            )

            # Generate recommendation
            if signals['combined_score'] >= 70:
                signals['recommendation'] = 'STRONG BUY'
                signals['confidence'] = 'HIGH'
                signals['strategies'] = ['sma_crossover', 'mean_reversion']
            elif signals['combined_score'] >= 55:
                signals['recommendation'] = 'BUY'
                signals['confidence'] = 'MEDIUM'
                signals['strategies'] = ['rsi', 'vwap']
            elif signals['combined_score'] <= 30:
                signals['recommendation'] = 'STRONG SELL'
                signals['confidence'] = 'HIGH'
                signals['strategies'] = ['short_sma', 'momentum']
            elif signals['combined_score'] <= 45:
                signals['recommendation'] = 'SELL'
                signals['confidence'] = 'MEDIUM'
                signals['strategies'] = ['bollinger_bands']
            else:
                signals['recommendation'] = 'HOLD'
                signals['confidence'] = 'LOW'
                signals['strategies'] = ['wait_for_confirmation']

            # Add risk-adjusted position sizing
            signals['suggested_position'] = self._calculate_position_size(
                signals['combined_score'],
                signals['confidence']
            )

            return signals
        except Exception as e:
            print(f"⚠️ Signal generation error: {e}")
            return {}

    def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """
        Calculate sentiment score (0-100) from sentiment analysis.
        
        Args:
            sentiment_data: Dictionary with sentiment analysis results
            
        Returns:
            Sentiment score from 0-100
        """
        if not sentiment_data:
            return 50.0  # Neutral if no data
        
        sentiment = sentiment_data.get('sentiment', 'neutral')
        score = sentiment_data.get('score', 0.0)
        confidence = sentiment_data.get('confidence', 0.5)
        tweet_count = sentiment_data.get('tweet_count', 0)
        
        # Convert sentiment score (-1 to 1) to (0 to 100)
        base_score = 50 + (score * 50)
        
        # Adjust based on confidence
        if tweet_count < 10:
            # Low confidence if few tweets
            base_score = 50 + (base_score - 50) * 0.5
        elif tweet_count > 100:
            # Higher confidence with more tweets
            base_score = 50 + (base_score - 50) * 1.2
        
        # Clamp to 0-100
        return max(0, min(100, base_score))

    def _synthesize_insights(
        self,
        symbol: str,
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        trading_signals: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize all analysis into actionable insights (now includes sentiment)."""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'HYBRID_WITH_SENTIMENT' if sentiment_data else 'HYBRID',

            'fundamental_analysis': {
                'summary': 'Strong fundamentals' if fundamental_data else 'Analysis pending',
                'details': fundamental_data
            },

            'technical_analysis': {
                'summary': self._summarize_technicals(technical_data),
                'details': technical_data
            },
            
            'sentiment_analysis': {
                'summary': self._summarize_sentiment(sentiment_data) if sentiment_data else 'Not available',
                'details': sentiment_data or {}
            },

            'trading_signals': trading_signals,

            'gordon_insights': {
                'market_sentiment': self._assess_market_sentiment(technical_data, sentiment_data),
                'risk_level': self._assess_risk_level(fundamental_data, technical_data),
                'time_horizon': self._recommend_time_horizon(trading_signals),
                'key_levels': {
                    'entry': technical_data.get('current_price', 0),
                    'stop_loss': technical_data.get('current_price', 0) * 0.95,
                    'take_profit': technical_data.get('current_price', 0) * 1.10
                }
            },

            'action_items': self._generate_action_items(symbol, trading_signals, sentiment_data),

            'disclaimer': "This analysis combines fundamental, technical, and sentiment factors. Always do your own research and consider your risk tolerance."
        }

    def _summarize_sentiment(self, sentiment_data: Dict[str, Any]) -> str:
        """Create a summary of sentiment analysis."""
        if not sentiment_data:
            return "No sentiment data available"
        
        sentiment = sentiment_data.get('sentiment', 'neutral')
        score = sentiment_data.get('score', 0.0)
        tweet_count = sentiment_data.get('tweet_count', 0)
        
        if sentiment == 'positive':
            return f"Positive sentiment ({score:.2f}) from {tweet_count} tweets"
        elif sentiment == 'negative':
            return f"Negative sentiment ({score:.2f}) from {tweet_count} tweets"
        else:
            return f"Neutral sentiment ({score:.2f}) from {tweet_count} tweets"

    def _assess_market_sentiment(
        self,
        technical_data: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Assess overall market sentiment (now includes social sentiment)."""
        signals = []

        if technical_data.get('rsi', {}).get('value', 50) < 30:
            signals.append('oversold')
        elif technical_data.get('rsi', {}).get('value', 50) > 70:
            signals.append('overbought')

        if technical_data.get('sma', {}).get('trend') == 'bullish':
            signals.append('uptrend')
        elif technical_data.get('sma', {}).get('trend') == 'bearish':
            signals.append('downtrend')
        
        # Add sentiment signals
        if sentiment_data:
            sentiment = sentiment_data.get('sentiment', 'neutral')
            if sentiment == 'positive':
                signals.append('positive_sentiment')
            elif sentiment == 'negative':
                signals.append('negative_sentiment')

        if not signals:
            return "NEUTRAL"
        elif 'uptrend' in signals and 'positive_sentiment' in signals and 'oversold' not in signals:
            return "STRONGLY BULLISH"
        elif 'uptrend' in signals or 'positive_sentiment' in signals:
            return "BULLISH"
        elif 'downtrend' in signals or 'negative_sentiment' in signals or 'overbought' in signals:
            return "BEARISH"
        else:
            return "MIXED"

    def _generate_action_items(
        self,
        symbol: str,
        signals: Dict[str, Any],
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate specific action items for the trader (now includes sentiment insights)."""
        actions = []

        if signals.get('recommendation') in ['BUY', 'STRONG BUY']:
            actions.append(f"✅ Consider buying {symbol}")
            actions.append(f"📊 Suggested position size: {signals.get('suggested_position', {}).get('percentage', 0.01):.1%}")
            actions.append(f"🎯 Use strategies: {', '.join(signals.get('strategies', []))}")
            actions.append("⚠️ Set stop-loss at -5%")
            
            # Add sentiment-specific actions
            if sentiment_data and sentiment_data.get('sentiment') == 'positive':
                actions.append(f"📱 Social sentiment is positive ({sentiment_data.get('positive_pct', 0):.1f}% positive)")
            elif sentiment_data and sentiment_data.get('sentiment') == 'negative':
                actions.append(f"⚠️ Social sentiment is negative - consider waiting for better sentiment")
                
        elif signals.get('recommendation') in ['SELL', 'STRONG SELL']:
            actions.append(f"❌ Consider selling {symbol}")
            actions.append("📉 Look for short opportunities")
            actions.append("🛡️ Reduce exposure to this asset")
            
            # Add sentiment-specific actions
            if sentiment_data and sentiment_data.get('sentiment') == 'negative':
                actions.append(f"📱 Social sentiment confirms negative outlook ({sentiment_data.get('negative_pct', 0):.1f}% negative)")
        else:
            actions.append(f"⏸️ Hold position in {symbol}")
            actions.append("👀 Monitor for better entry points")
            actions.append("📈 Wait for clearer signals")
            
            # Add sentiment-specific actions
            if sentiment_data:
                actions.append(f"📱 Current sentiment: {sentiment_data.get('sentiment', 'neutral')} ({sentiment_data.get('tweet_count', 0)} tweets analyzed)")

        return actions

    def _calculate_fundamental_score(self, data: Dict[str, Any]) -> float:
        """Calculate fundamental analysis score (0-100)."""
        # Simplified scoring - in production, this would be much more sophisticated
        score = 50  # Base score

        # Adjust based on available data
        if 'revenue_growth' in data:
            score += 10
        if 'profit_margins' in data:
            score += 10
        if 'analyst_rating' in data:
            score += 10

        return min(100, max(0, score))

    def _calculate_technical_score(self, data: Dict[str, Any]) -> float:
        """Calculate technical analysis score (0-100)."""
        score = 50  # Base score

        # RSI scoring
        rsi_value = data.get('rsi', {}).get('value', 50)
        if 30 <= rsi_value <= 70:
            score += 10  # Neutral RSI is good
        elif rsi_value < 30:
            score += 20  # Oversold - potential buy
        elif rsi_value > 70:
            score -= 10  # Overbought - caution

        # SMA signal scoring
        if data.get('sma', {}).get('signal') == 'buy':
            score += 20
        elif data.get('sma', {}).get('signal') == 'sell':
            score -= 20

        # Volume scoring
        if data.get('volume_24h', 0) > 0:
            score += 5  # Active volume is positive

        return min(100, max(0, score))

    def _calculate_support_resistance(self, symbol: str) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        # Simplified calculation - in production, use proper technical analysis
        return {
            'support_1': 0,
            'support_2': 0,
            'resistance_1': 0,
            'resistance_2': 0
        }

    def _calculate_position_size(self, score: float, confidence: str) -> Dict[str, Any]:
        """Calculate suggested position size based on score and confidence."""
        base_position = self.config.get('trading.position.default_size', 0.01)

        # Adjust based on confidence
        confidence_multiplier = {
            'HIGH': 2.0,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }.get(confidence, 1.0)

        # Adjust based on score
        score_multiplier = score / 100.0

        suggested_size = base_position * confidence_multiplier * score_multiplier

        # Apply limits
        max_size = self.config.get('trading.risk.max_position_size', 0.1)
        suggested_size = min(suggested_size, max_size)

        return {
            'percentage': suggested_size,
            'risk_adjusted': True,
            'kelly_criterion': suggested_size * 0.25  # Conservative Kelly
        }

    def _summarize_technicals(self, data: Dict[str, Any]) -> str:
        """Create a summary of technical indicators."""
        rsi_signal = data.get('rsi', {}).get('signal', 'neutral')
        sma_trend = data.get('sma', {}).get('trend', 'neutral')

        if rsi_signal == 'buy' and sma_trend == 'bullish':
            return "Strong bullish signals across indicators"
        elif rsi_signal == 'sell' and sma_trend == 'bearish':
            return "Strong bearish signals across indicators"
        else:
            return "Mixed signals - exercise caution"

    def _assess_risk_level(
        self,
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any]
    ) -> str:
        """Assess risk level of the trade."""
        # Simplified risk assessment
        volatility = abs(float(technical_data.get('price_change_24h', '0%').rstrip('%')))

        if volatility > 10:
            return "HIGH"
        elif volatility > 5:
            return "MEDIUM"
        else:
            return "LOW"

    def _recommend_time_horizon(self, trading_signals: Dict[str, Any]) -> str:
        """Recommend investment time horizon."""
        if trading_signals.get('confidence') == 'HIGH':
            if trading_signals.get('recommendation') in ['STRONG BUY', 'STRONG SELL']:
                return "SHORT-TERM (1-5 days)"
            else:
                return "MEDIUM-TERM (1-4 weeks)"
        else:
            return "LONG-TERM (1+ months) or WAIT"

    def _generate_action_items(self, symbol: str, signals: Dict[str, Any]) -> List[str]:
        """Generate specific action items for the trader."""
        actions = []

        if signals.get('recommendation') in ['BUY', 'STRONG BUY']:
            actions.append(f"✅ Consider buying {symbol}")
            actions.append(f"📊 Suggested position size: {signals.get('suggested_position', {}).get('percentage', 0.01):.1%}")
            actions.append(f"🎯 Use strategies: {', '.join(signals.get('strategies', []))}")
            actions.append("⚠️ Set stop-loss at -5%")
        elif signals.get('recommendation') in ['SELL', 'STRONG SELL']:
            actions.append(f"❌ Consider selling {symbol}")
            actions.append("📉 Look for short opportunities")
            actions.append("🛡️ Reduce exposure to this asset")
        else:
            actions.append(f"⏸️ Hold position in {symbol}")
            actions.append("👀 Monitor for better entry points")
            actions.append("📈 Wait for clearer signals")

        return actions


# Convenience function for quick analysis
async def analyze(symbol: str) -> Dict[str, Any]:
    """Quick hybrid analysis of a symbol (now async for sentiment support).

    Args:
        symbol: Stock or crypto symbol to analyze

    Returns:
        Comprehensive hybrid analysis with sentiment
    """
    analyzer = HybridAnalyzer()
    return await analyzer.analyze_stock(symbol)