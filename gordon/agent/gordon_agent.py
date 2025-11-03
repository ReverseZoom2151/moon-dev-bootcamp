"""
Gordon Agent - Unified Financial Research & Trading System
===========================================================
Combines fundamental financial analysis with technical trading capabilities.
The complete AI trading assistant with both research and execution.
"""

from typing import Dict, Any, Optional
import asyncio  

# Research Agent imports (financial analysis)
from .agent import Agent as ResearchAgent

# Our trading system imports
from .core.strategy_manager import StrategyManager
from .core.risk.base_manager import RiskManager
from .core.position_manager import PositionManager
from .core.algo_orders import AlgoOrderManager
from .core.streams.aggregator import StreamAggregator
from .backtesting.backtest_main import ComprehensiveBacktester
from .exchanges.factory import ExchangeFactory

# Utilities
from .utils.logger import Logger
from .utils.ui import show_progress


class GordonAgent:
    """
    Gordon - The complete trading and research agent.
    Combines fundamental analysis with technical trading.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize Gordon with both research and trading capabilities."""
        self.logger = Logger()
        self.config = config or {}

        # Initialize Gordon's research capabilities
        self.research_agent = ResearchAgent(
            max_steps=20,
            max_steps_per_task=5
        )

        # Initialize our trading system
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager()
        self.algo_order_manager = AlgoOrderManager()
        self.stream_aggregator = StreamAggregator()
        self.backtester = ComprehensiveBacktester()

        # Exchange connections
        self.exchanges = {}
        self._setup_exchanges()

        self.logger._log("Gordon initialized with research and trading capabilities")

    def _setup_exchanges(self):
        """Setup exchange connections."""
        exchange_configs = self.config.get('exchanges', {})
        for exchange_name, exchange_config in exchange_configs.items():
            try:
                self.exchanges[exchange_name] = ExchangeFactory.create(
                    exchange_name,
                    exchange_config
                )
                self.logger._log(f"Connected to {exchange_name}")
            except Exception as e:
                self.logger._log(f"Failed to connect to {exchange_name}: {e}")

    # ========== UNIFIED INTERFACE ==========

    async def run(self, query: str) -> str:
        """
        Main entry point - handles both research and trading queries.

        Examples:
            - "Analyze Apple's financials and execute a trade if fundamentals are strong"
            - "What's Tesla's debt situation and should I open a position?"
            - "Run RSI strategy on BTC with risk management"
            - "Backtest SMA strategy on ETH for the last month"
        """
        self.logger._log(f"Processing query: {query}")

        # Determine query type
        query_type = self._classify_query(query)

        if query_type == "research":
            return await self.research(query)
        elif query_type == "trading":
            return await self.trade(query)
        elif query_type == "hybrid":
            return await self.research_and_trade(query)
        elif query_type == "backtest":
            return await self.backtest(query)
        else:
            return await self.general_assistance(query)

    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()

        # Keywords for different query types
        research_keywords = ['analyze', 'financial', 'revenue', 'earnings', 'debt',
                           'balance sheet', 'income statement', 'cash flow', '10-k', '10-q']
        trading_keywords = ['buy', 'sell', 'trade', 'position', 'execute', 'order',
                          'strategy', 'rsi', 'sma', 'vwap', 'bollinger']
        backtest_keywords = ['backtest', 'historical', 'test strategy', 'performance']

        has_research = any(kw in query_lower for kw in research_keywords)
        has_trading = any(kw in query_lower for kw in trading_keywords)
        has_backtest = any(kw in query_lower for kw in backtest_keywords)

        if has_backtest:
            return "backtest"
        elif has_research and has_trading:
            return "hybrid"
        elif has_research:
            return "research"
        elif has_trading:
            return "trading"
        else:
            return "general"

    # ========== RESEARCH CAPABILITIES (from Gordon) ==========

    @show_progress("Conducting financial research...", "Research complete")
    async def research(self, query: str) -> str:
        """Perform financial research using Gordon's capabilities."""
        try:
            # Use Gordon's agent for research
            result = await self.research_agent.run(query)
            return result
        except Exception as e:
            self.logger._log(f"Research failed: {e}")
            return f"Research error: {str(e)}"

    # ========== TRADING CAPABILITIES (our system) ==========

    @show_progress("Executing trading operation...", "Trading complete")
    async def trade(self, query: str) -> str:
        """Execute trading operations."""
        try:
            # Parse trading intent
            trading_params = self._parse_trading_query(query)

            # Check risk management
            if not self.risk_manager.check_trade_allowed(trading_params):
                return "Trade rejected by risk management"

            # Select strategy
            strategy = self.strategy_manager.get_strategy(trading_params.get('strategy'))
            if not strategy:
                return "No suitable strategy found"

            # Execute trade
            signal = await strategy.execute()
            if signal:
                result = await self._execute_trade(signal)
                return f"Trade executed: {result}"
            else:
                return "No trading signal generated"

        except Exception as e:
            self.logger._log(f"Trading failed: {e}")
            return f"Trading error: {str(e)}"

    # ========== HYBRID OPERATIONS ==========

    @show_progress("Analyzing fundamentals and technicals...", "Analysis complete")
    async def research_and_trade(self, query: str) -> str:
        """
        Combine fundamental research with technical trading.
        This is Gordon's unique capability!
        """
        try:
            # Extract symbol from query
            symbol = self._extract_symbol(query)
            if not symbol:
                return "Could not identify symbol to analyze"

            # Step 1: Fundamental Analysis
            self.logger._log(f"Performing fundamental analysis for {symbol}")
            fundamental_query = f"Analyze {symbol}'s financial health, revenue growth, and profitability"
            fundamental_result = await self.research(fundamental_query)

            # Step 2: Score fundamentals
            fundamental_score = self._score_fundamentals(fundamental_result)
            self.logger._log(f"Fundamental score: {fundamental_score}/10")

            # Step 3: Technical Analysis
            self.logger._log(f"Performing technical analysis for {symbol}")
            technical_signals = await self._run_technical_analysis(symbol)
            technical_score = self._score_technicals(technical_signals)
            self.logger._log(f"Technical score: {technical_score}/10")

            # Step 4: Combined Decision
            combined_score = (fundamental_score * 0.4 + technical_score * 0.6)
            self.logger._log(f"Combined score: {combined_score}/10")

            # Step 5: Execute if criteria met
            if combined_score >= 7:
                self.logger._log("Criteria met - executing trade")
                trade_result = await self._execute_smart_trade(
                    symbol,
                    fundamental_score,
                    technical_score
                )

                return f"""
📊 Gordon's Analysis Complete:

**Fundamental Score**: {fundamental_score}/10
{fundamental_result[:500]}...

**Technical Score**: {technical_score}/10
- RSI: {technical_signals.get('rsi', 'N/A')}
- SMA Signal: {technical_signals.get('sma', 'N/A')}
- VWAP Position: {technical_signals.get('vwap', 'N/A')}

**Combined Score**: {combined_score:.1f}/10

**Action Taken**: {trade_result}
"""
            else:
                return f"""
📊 Gordon's Analysis Complete:

**Fundamental Score**: {fundamental_score}/10
**Technical Score**: {technical_score}/10
**Combined Score**: {combined_score:.1f}/10

**Decision**: No trade - criteria not met (need 7.0+ score)
Consider waiting for better entry conditions.
"""

        except Exception as e:
            self.logger._log(f"Hybrid operation failed: {e}")
            return f"Analysis error: {str(e)}"

    # ========== BACKTESTING ==========

    @show_progress("Running backtest...", "Backtest complete")
    async def backtest(self, query: str) -> str:
        """Run backtesting operations."""
        try:
            # Parse backtest parameters
            params = self._parse_backtest_query(query)

            # Run backtest
            results = self.backtester.run_strategy(
                params['strategy'],
                symbol=params.get('symbol', 'BTCUSDT'),
                **params.get('settings', {})
            )

            return f"""
📈 Backtest Results:

Strategy: {params['strategy']}
Symbol: {params.get('symbol', 'BTCUSDT')}
Total Return: {results.get('total_return', 0):.2f}%
Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
Max Drawdown: {results.get('max_drawdown', 0):.2f}%
Win Rate: {results.get('win_rate', 0):.2f}%
"""

        except Exception as e:
            self.logger._log(f"Backtest failed: {e}")
            return f"Backtest error: {str(e)}"

    # ========== HELPER METHODS ==========

    def _extract_symbol(self, query: str) -> Optional[str]:
        """Extract trading symbol from query."""
        # Common crypto symbols
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'DOT', 'AVAX']

        # Common stock symbols
        stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA']

        query_upper = query.upper()
        for symbol in crypto_symbols + stock_symbols:
            if symbol in query_upper:
                # Return with appropriate suffix for crypto
                if symbol in crypto_symbols:
                    return f"{symbol}USDT"
                return symbol

        return None

    def _score_fundamentals(self, fundamental_result: str) -> float:
        """Score fundamental analysis results (0-10)."""
        # Simplified scoring based on keywords
        positive_keywords = ['strong', 'growth', 'profitable', 'increasing',
                           'healthy', 'positive', 'robust', 'excellent']
        negative_keywords = ['weak', 'declining', 'loss', 'decreasing',
                           'poor', 'negative', 'concerning', 'risky']

        result_lower = fundamental_result.lower()
        positive_count = sum(1 for kw in positive_keywords if kw in result_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in result_lower)

        # Calculate score
        score = 5 + positive_count - negative_count
        return max(0, min(10, score))

    async def _run_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Run technical analysis on symbol."""
        results = {}

        # Run multiple strategies
        for strategy_name in ['RSI', 'SMA', 'VWAP']:
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if strategy:
                signal = await strategy.analyze(symbol)
                results[strategy_name.lower()] = signal

        return results

    def _score_technicals(self, signals: Dict[str, Any]) -> float:
        """Score technical signals (0-10)."""
        score = 5.0  # Start neutral

        # Check each signal
        for signal_name, signal_value in signals.items():
            if signal_value == 'BUY' or signal_value == 'BULLISH':
                score += 1.5
            elif signal_value == 'SELL' or signal_value == 'BEARISH':
                score -= 1.5

        return max(0, min(10, score))

    async def _execute_smart_trade(self, symbol: str,
                                  fundamental_score: float,
                                  technical_score: float) -> str:
        """Execute trade with smart position sizing based on confidence."""
        # Calculate position size based on scores
        base_size = self.config.get('base_position_size', 0.01)
        confidence_multiplier = (fundamental_score + technical_score) / 20
        position_size = base_size * confidence_multiplier

        # Create order
        order = {
            'symbol': symbol,
            'side': 'BUY',
            'size': position_size,
            'type': 'MARKET',
            'metadata': {
                'fundamental_score': fundamental_score,
                'technical_score': technical_score,
                'agent': 'Gordon'
            }
        }

        # Execute through algo order manager
        result = await self.algo_order_manager.execute_order(order)

        return f"Executed {position_size:.4f} {symbol} position (Confidence: {confidence_multiplier*100:.0f}%)"

    def _parse_trading_query(self, query: str) -> Dict[str, Any]:
        """Parse trading parameters from query."""
        params = {}

        # Extract strategy
        strategies = ['RSI', 'SMA', 'VWAP', 'Bollinger', 'Breakout']
        for strategy in strategies:
            if strategy.lower() in query.lower():
                params['strategy'] = strategy
                break

        # Extract symbol
        params['symbol'] = self._extract_symbol(query)

        # Extract action
        if 'buy' in query.lower():
            params['action'] = 'BUY'
        elif 'sell' in query.lower():
            params['action'] = 'SELL'

        return params

    def _parse_backtest_query(self, query: str) -> Dict[str, Any]:
        """Parse backtest parameters from query."""
        params = {}

        # Extract strategy
        strategies = ['sma_crossover', 'rsi', 'vwap', 'bollinger', 'mean_reversion']
        for strategy in strategies:
            if strategy.replace('_', ' ').lower() in query.lower():
                params['strategy'] = strategy
                break

        # Extract symbol
        params['symbol'] = self._extract_symbol(query)

        # Extract time period (simplified)
        if 'month' in query.lower():
            params['settings'] = {'days': 30}
        elif 'year' in query.lower():
            params['settings'] = {'days': 365}

        return params

    async def _execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade signal."""
        # Select exchange
        exchange_name = signal.get('exchange', 'binance')
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not connected")

        exchange = self.exchanges[exchange_name]

        # Execute order
        result = await exchange.create_order(
            symbol=signal['symbol'],
            side=signal['action'],
            order_type='MARKET',
            amount=signal['size']
        )

        # Track position
        self.position_manager.add_position(result)

        return result

    async def general_assistance(self, query: str) -> str:
        """Handle general queries."""
        return f"""
🤖 Gordon here! I can help you with:

📊 **Financial Research**:
- Company financial analysis
- Income statements, balance sheets
- SEC filings (10-K, 10-Q, 8-K)
- Analyst estimates

📈 **Trading Operations**:
- Technical analysis (RSI, SMA, VWAP, Bollinger)
- Strategy execution
- Risk management
- Position tracking

🔬 **Backtesting**:
- Historical strategy performance
- Optimization
- Multiple timeframes

💡 **Hybrid Analysis**:
- Combine fundamentals + technicals
- Smart position sizing
- Data-driven decisions

Try asking:
- "Analyze Apple's financials and trade if strong"
- "Run RSI strategy on BTC"
- "Backtest SMA strategy on ETH"

Your query: "{query}"
"""


# ========== CLI ENTRY POINT ==========

async def main():
    """Main CLI entry point for Gordon."""
    print("""
    ╔══════════════════════════════════════════╗
    ║           GORDON TRADING AGENT           ║
    ║   Financial Research + Technical Trading ║
    ║          Powered by AI & Data            ║
    ╚══════════════════════════════════════════╝
    """)

    # Initialize Gordon
    config = {
        'exchanges': {
            'binance': {
                'api_key': os.getenv('BINANCE_API_KEY'),
                'api_secret': os.getenv('BINANCE_API_SECRET')
            }
        },
        'base_position_size': 0.01
    }

    gordon = GordonAgent(config)

    print("Gordon is ready! Type 'help' for commands or 'exit' to quit.\n")

    while True:
        try:
            query = input("Gordon> ").strip()

            if query.lower() == 'exit':
                print("Goodbye! Happy trading! 📈")
                break
            elif query.lower() == 'help':
                result = await gordon.general_assistance("")
                print(result)
            else:
                result = await gordon.run(query)
                print(f"\n{result}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy trading! 📈")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import os
    asyncio.run(main())