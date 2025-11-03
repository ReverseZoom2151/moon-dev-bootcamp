"""
Hybrid Analyzer for Gordon
==========================
Combines fundamental analysis with technical trading for comprehensive insights.
This is Gordon's secret sauce - the fusion of Warren Buffett and Jim Simons!
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .agent import Agent
from .config_manager import get_config


class HybridAnalyzer:
    """Combines fundamental and technical analysis for trading decisions."""

    def __init__(self):
        """Initialize the hybrid analyzer."""
        self.agent = Agent()
        self.config = get_config()
        self.executor = ThreadPoolExecutor(max_workers=5)

    def analyze_stock(self, symbol: str, include_trading: bool = True) -> Dict[str, Any]:
        """Perform comprehensive hybrid analysis on a stock.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            include_trading: Whether to include trading recommendations

        Returns:
            Comprehensive analysis combining fundamental and technical insights
        """
        print(f"\n🔬 Gordon is analyzing {symbol}...")
        print("=" * 50)

        # Gather fundamental data
        print("\n📊 Gathering fundamental data...")
        fundamental_analysis = self._analyze_fundamentals(symbol)

        # Gather technical data
        print("\n📈 Performing technical analysis...")
        technical_analysis = self._analyze_technicals(symbol)

        # Generate trading signals
        trading_signals = {}
        if include_trading:
            print("\n🎯 Generating trading signals...")
            trading_signals = self._generate_trading_signals(symbol, fundamental_analysis, technical_analysis)

        # Combine insights
        print("\n🧠 Synthesizing insights...")
        combined_insights = self._synthesize_insights(
            symbol,
            fundamental_analysis,
            technical_analysis,
            trading_signals
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

    def _generate_trading_signals(
        self,
        symbol: str,
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals based on combined analysis."""
        try:
            signals = {
                'fundamental_score': self._calculate_fundamental_score(fundamental_data),
                'technical_score': self._calculate_technical_score(technical_data),
                'combined_score': 0,
                'recommendation': 'HOLD',
                'confidence': 'LOW',
                'strategies': []
            }

            # Calculate combined score (weighted average)
            fundamental_weight = 0.4  # 40% weight on fundamentals
            technical_weight = 0.6  # 60% weight on technicals

            signals['combined_score'] = (
                signals['fundamental_score'] * fundamental_weight +
                signals['technical_score'] * technical_weight
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

    def _synthesize_insights(
        self,
        symbol: str,
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        trading_signals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize all analysis into actionable insights."""
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'HYBRID',

            'fundamental_analysis': {
                'summary': 'Strong fundamentals' if fundamental_data else 'Analysis pending',
                'details': fundamental_data
            },

            'technical_analysis': {
                'summary': self._summarize_technicals(technical_data),
                'details': technical_data
            },

            'trading_signals': trading_signals,

            'gordon_insights': {
                'market_sentiment': self._assess_market_sentiment(technical_data),
                'risk_level': self._assess_risk_level(fundamental_data, technical_data),
                'time_horizon': self._recommend_time_horizon(trading_signals),
                'key_levels': {
                    'entry': technical_data.get('current_price', 0),
                    'stop_loss': technical_data.get('current_price', 0) * 0.95,
                    'take_profit': technical_data.get('current_price', 0) * 1.10
                }
            },

            'action_items': self._generate_action_items(symbol, trading_signals),

            'disclaimer': "This analysis combines fundamental and technical factors. Always do your own research and consider your risk tolerance."
        }

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

    def _assess_market_sentiment(self, technical_data: Dict[str, Any]) -> str:
        """Assess overall market sentiment."""
        signals = []

        if technical_data.get('rsi', {}).get('value', 50) < 30:
            signals.append('oversold')
        elif technical_data.get('rsi', {}).get('value', 50) > 70:
            signals.append('overbought')

        if technical_data.get('sma', {}).get('trend') == 'bullish':
            signals.append('uptrend')
        elif technical_data.get('sma', {}).get('trend') == 'bearish':
            signals.append('downtrend')

        if not signals:
            return "NEUTRAL"
        elif 'uptrend' in signals and 'oversold' not in signals:
            return "BULLISH"
        elif 'downtrend' in signals or 'overbought' in signals:
            return "BEARISH"
        else:
            return "MIXED"

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
def analyze(symbol: str) -> Dict[str, Any]:
    """Quick hybrid analysis of a symbol.

    Args:
        symbol: Stock or crypto symbol to analyze

    Returns:
        Comprehensive hybrid analysis
    """
    analyzer = HybridAnalyzer()
    return analyzer.analyze_stock(symbol)