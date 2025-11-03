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
from gordon.core.strategy_manager import StrategyManager
from gordon.core.risk_manager import RiskManager
from gordon.core.position_manager import PositionManager
# Note: AlgorithmicTrader requires orchestrator, will be initialized when needed
from gordon.core.streams.aggregator import StreamAggregator
from gordon.backtesting.backtest_main import ComprehensiveBacktester
from gordon.exchanges.factory import ExchangeFactory
from gordon.core.event_bus import EventBus
from gordon.config.config_manager import ConfigManager

# Utilities
from gordon.utilities.ui import Logger, show_progress


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
        # Create event bus and config manager for managers that require them
        event_bus = EventBus()
        config_manager = ConfigManager()
        
        self.strategy_manager = StrategyManager(event_bus, config=self.config)
        self.risk_manager = RiskManager(event_bus, config_manager, demo_mode=True)
        self.position_manager = PositionManager()
        
        # AlgorithmicTrader requires orchestrator - we'll create a simple adapter
        # For now, initialize it with None and set it up later if needed
        # AlgorithmicTrader needs orchestrator for exchange operations, so we'll defer initialization
        self.algo_order_manager = None  # Will be initialized when needed with proper orchestrator
        
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
        elif query_type == "whale_tracking":
            return await self.handle_whale_tracking(query)
        elif query_type == "position_sizing":
            return await self.handle_position_sizing(query)
        elif query_type == "ml_indicators":
            return await self.handle_ml_indicators(query)
        elif query_type == "rrs_analysis":
            return await self.handle_rrs_analysis(query)
        elif query_type == "trader_intelligence":
            return await self.handle_trader_intelligence(query)
        elif query_type == "liquidation_hunter":
            return await self.handle_liquidation_hunter(query)
        elif query_type == "orderbook_analysis":
            return await self.handle_orderbook_analysis(query)
        elif query_type == "conversation":
            return await self.handle_conversation_commands(query)
        elif query_type == "system":
            return await self.handle_system_commands(query)
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
        whale_keywords = ['whale', 'large position', 'track whale', 'whale position', 
                         'institutional', 'big holder', 'whale tracker']
        sizing_keywords = ['position size', 'calculate size', 'size position', 'position sizing',
                          'how much', 'how many', 'position amount', 'leverage']
        ml_keywords = ['ml indicator', 'discover indicator', 'evaluate indicator', 'top indicator',
                      'indicator evaluation', 'indicator ranking', 'indicator discovery']
        rrs_keywords = ['rrs', 'relative rotation', 'rotation strength', 'rrs analysis', 'rrs ranking']
        trader_intel_keywords = ['early buyer', 'trader intelligence', 'find accounts', 'institutional trader',
                                'smart money', 'whale address', 'accounts to follow']
        conversation_keywords = ['search conversation', 'export conversation', 'conversation analytics',
                                'list users', 'switch user', 'conversation history']
        system_keywords = ['status', 'config', 'risk', 'help', 'clear']
        liquidation_keywords = ['liquidation hunter', 'liquidation risk', 'liquidation cascade', 'liquidation analysis',
                               'whale positions', 'liquidation data', 'moondev']
        orderbook_keywords = ['order book', 'orderbook', 'whale orders', 'order book depth', 'spread analysis',
                             'market depth', 'bid ask']

        has_research = any(kw in query_lower for kw in research_keywords)
        has_trading = any(kw in query_lower for kw in trading_keywords)
        has_backtest = any(kw in query_lower for kw in backtest_keywords)
        has_whale = any(kw in query_lower for kw in whale_keywords)
        has_sizing = any(kw in query_lower for kw in sizing_keywords)
        has_ml = any(kw in query_lower for kw in ml_keywords)
        has_rrs = any(kw in query_lower for kw in rrs_keywords)
        has_trader_intel = any(kw in query_lower for kw in trader_intel_keywords)
        has_conversation = any(kw in query_lower for kw in conversation_keywords)
        has_system = any(kw in query_lower for kw in system_keywords)
        has_liquidation = any(kw in query_lower for kw in liquidation_keywords)
        has_orderbook = any(kw in query_lower for kw in orderbook_keywords)

        if has_backtest:
            return "backtest"
        elif has_whale:
            return "whale_tracking"
        elif has_sizing:
            return "position_sizing"
        elif has_ml:
            return "ml_indicators"
        elif has_rrs:
            return "rrs_analysis"
        elif has_trader_intel:
            return "trader_intelligence"
        elif has_liquidation:
            return "liquidation_hunter"
        elif has_orderbook:
            return "orderbook_analysis"
        elif has_conversation:
            return "conversation"
        elif has_system:
            return "system"
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

    # ========== WHALE TRACKING (Day 44) ==========

    @show_progress("Tracking whale positions...", "Whale tracking complete")
    async def handle_whale_tracking(self, query: str) -> str:
        """Handle whale tracking queries in natural language."""
        try:
            from gordon.core.utilities import WhaleTrackingManager
            from gordon.exchanges.factory import ExchangeFactory
            import yaml
            from pathlib import Path
            
            # Get config
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            # Get exchange adapter
            exchange_name = 'binance'  # Default
            exchange_config = config.get('exchanges', {}).get(exchange_name, {})
            exchange_adapter = ExchangeFactory.create_exchange(
                exchange_name,
                exchange_config,
                event_bus=None
            )
            
            # Initialize manager
            manager = WhaleTrackingManager(
                exchange_adapter=exchange_adapter,
                config=config.get('whale_tracking', {})
            )
            
            # Parse query to extract symbol and min value
            symbol = self._extract_symbol(query)
            
            # Extract minimum value from query
            min_value = None
            import re
            value_matches = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', query)
            if value_matches:
                # Convert to float (remove commas)
                min_value = float(value_matches[-1].replace(',', ''))
            
            # Track whales
            results = await manager.track_whales(symbol=symbol, min_value_usd=min_value)
            
            # Generate response
            report = manager.get_whale_summary_report(results)
            
            response = f"""
🐋 **Whale Position Tracking Results**

{report}

"""
            
            # Add top positions if available
            if not results['top_positions'].empty:
                top_df = results['top_positions'].head(10)
                response += "**Top Positions:**\n"
                for _, row in top_df.iterrows():
                    response += f"- {row['symbol']}: ${row['position_value_usd']:,.2f} ({row['pnl_percent']:+.2f}%) - {row['whale_tier']}\n"
            
            return response
            
        except Exception as e:
            self.logger._log(f"Whale tracking failed: {e}")
            return f"Whale tracking error: {str(e)}"

    # ========== POSITION SIZING (Day 44) ==========

    @show_progress("Calculating position size...", "Position sizing complete")
    async def handle_position_sizing(self, query: str) -> str:
        """Handle position sizing queries in natural language."""
        try:
            from gordon.core.utilities import PositionSizingHelper
            import yaml
            from pathlib import Path
            import re
            
            # Get config
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            sizing = PositionSizingHelper(config.get('position_sizing', {}))
            
            query_lower = query.lower()
            
            # Extract numbers from query
            numbers = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', query)
            numbers = [float(n.replace(',', '')) for n in numbers]
            
            # Extract leverage
            leverage_match = re.search(r'(\d+)x?\s*(?:leverage|lev)', query_lower)
            leverage = int(leverage_match.group(1)) if leverage_match else sizing.default_leverage
            
            # Determine method based on query
            if 'balance' in query_lower or 'account' in query_lower:
                # Balance-based sizing
                balance = numbers[0] if numbers else 10000
                price = numbers[1] if len(numbers) > 1 else None
                
                if price:
                    leverage, size = sizing.calculate_size_from_balance_percent(
                        balance, price, leverage
                    )
                    return f"""
📏 **Position Size Calculation (Balance-Based)**

Account Balance: ${balance:,.2f}
Price: ${price:,.2f}
Leverage: {leverage}x

**Recommended Position Size:** {size}

This uses 95% of your balance with {leverage}x leverage.
"""
                else:
                    return "Please specify both balance and price. Example: 'Calculate position size for $10k balance at $50k price'"
            
            elif 'risk' in query_lower or 'stop loss' in query_lower or 'stop' in query_lower:
                # Risk-based sizing
                balance = numbers[0] if numbers else 10000
                entry = numbers[1] if len(numbers) > 1 else None
                stop = numbers[2] if len(numbers) > 2 else None
                
                if entry and stop:
                    leverage, size = sizing.calculate_size_from_risk_percent(
                        balance, entry, stop, leverage=leverage
                    )
                    risk_pct = ((entry - stop) / entry) * 100
                    return f"""
📏 **Position Size Calculation (Risk-Based)**

Account Balance: ${balance:,.2f}
Entry Price: ${entry:,.2f}
Stop Loss Price: ${stop:,.2f}
Risk Per Trade: {risk_pct:.2f}%
Leverage: {leverage}x

**Recommended Position Size:** {size}

This ensures you risk 2% of your account if stop loss is hit.
"""
                else:
                    return "Please specify balance, entry price, and stop loss. Example: 'Calculate position size with $10k balance, entry $50k, stop $48k'"
            
            elif any(kw in query_lower for kw in ['usd', 'dollar', '$']):
                # USD-based sizing
                amount = numbers[0] if numbers else 1000
                price = numbers[1] if len(numbers) > 1 else None
                
                if price:
                    leverage, size = sizing.calculate_size_from_usd_amount(
                        amount, price, leverage
                    )
                    return f"""
📏 **Position Size Calculation (USD-Based)**

USD Amount: ${amount:,.2f}
Price: ${price:,.2f}
Leverage: {leverage}x

**Recommended Position Size:** {size}

This calculates size for a fixed ${amount:,.2f} position with {leverage}x leverage.
"""
                else:
                    return "Please specify both USD amount and price. Example: 'Calculate position size for $1000 at $50k price'"
            
            else:
                # Generic response
                return """
📏 **Position Sizing Help**

I can calculate position sizes using:
1. **Balance-based**: "Calculate position size for $10k balance at $50k price"
2. **Risk-based**: "Calculate position size with $10k balance, entry $50k, stop $48k"
3. **USD-based**: "Calculate position size for $1000 at $50k price"

You can also specify leverage: "with 2x leverage" or "at 5x leverage"

Examples:
- "How much should I buy with $10k balance at $50k price?"
- "Calculate position size for $1000 at $50k with 2x leverage"
- "What's my position size with $10k balance, entry $50k, stop $48k?"
"""
            
        except Exception as e:
            self.logger._log(f"Position sizing failed: {e}")
            return f"Position sizing error: {str(e)}"

    # ========== ML INDICATORS (Day 33) ==========

    @show_progress("Processing ML indicator request...", "Complete")
    async def handle_ml_indicators(self, query: str) -> str:
        """Handle ML indicator queries in natural language."""
        try:
            from gordon.ml import MLIndicatorManager
            import yaml
            from pathlib import Path
            import re
            
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            ml_manager = MLIndicatorManager(config)
            query_lower = query.lower()
            
            # Discover indicators
            if any(kw in query_lower for kw in ['discover', 'list', 'available', 'what indicators']):
                indicators = ml_manager.discover_indicators()
                response = "🔍 **Available Indicators**\n\n"
                if indicators:
                    response += f"Found {len(indicators)} indicator categories:\n"
                    for category, ind_list in indicators.items():
                        response += f"\n**{category}**: {', '.join(ind_list[:10])}"
                        if len(ind_list) > 10:
                            response += f" ... and {len(ind_list) - 10} more"
                return response
            
            # Evaluate indicators
            elif any(kw in query_lower for kw in ['evaluate', 'test', 'performance']):
                symbol = self._extract_symbol(query) or 'BTCUSDT'
                from gordon.backtesting.data.fetcher import DataFetcher
                data_fetcher = DataFetcher()
                df = data_fetcher.fetch_for_backtesting_lib(symbol, '1h', limit=2000)
                
                if df.empty:
                    return f"Could not fetch data for {symbol}"
                
                results = ml_manager.evaluate_indicators(df, symbol)
                response = f"📊 **ML Indicator Evaluation for {symbol}**\n\n"
                if results:
                    for model_name, metrics in results.items():
                        response += f"**{model_name}**:\n"
                        response += f"  MSE: {metrics.get('mse', 'N/A')}\n"
                        response += f"  R2: {metrics.get('r2', 'N/A')}\n"
                        if metrics.get('top_features'):
                            response += f"  Top Features: {', '.join(metrics['top_features'][:5])}\n\n"
                return response
            
            # Top indicators
            elif any(kw in query_lower for kw in ['top', 'best', 'ranking', 'rank']):
                numbers = re.findall(r'\d+', query)
                top_n = int(numbers[0]) if numbers else 10
                
                rankings = ml_manager.get_top_indicators(top_n=top_n)
                if not rankings:
                    return "No ranking data found. Run indicator evaluation first."
                
                response = f"🏆 **Top {top_n} Indicators**\n\n"
                for metric, df in rankings.items():
                    if not df.empty:
                        response += f"**Ranked by {metric}**:\n"
                        response += df.head(10).to_string(index=False) + "\n\n"
                return response
            
            # Loop indicators
            elif any(kw in query_lower for kw in ['loop', 'generation', 'evolve']):
                symbol = self._extract_symbol(query) or 'BTCUSDT'
                numbers = re.findall(r'\d+', query)
                generations = int(numbers[0]) if numbers else 5
                
                from gordon.backtesting.data.fetcher import DataFetcher
                data_fetcher = DataFetcher()
                df = data_fetcher.fetch_for_backtesting_lib(symbol, '1h', limit=2000)
                
                if df.empty:
                    return f"Could not fetch data for {symbol}"
                
                results = ml_manager.run_indicator_looping(df)
                return f"✅ Completed {len(results)} generations of indicator evaluation for {symbol}"
            
            else:
                return """
📊 **ML Indicator Analysis Help**

I can help with:
- "Discover available indicators" - List all indicators
- "Evaluate indicators for BTC" - Test indicator performance
- "Top 10 indicators" - Show best performing indicators
- "Run indicator loop for ETH" - Multi-generation evaluation

Examples:
- "What indicators are available?"
- "Evaluate indicators for BTCUSDT"
- "Show me the top 20 indicators"
- "Run 5 generations of indicator evaluation for ETH"
"""
            
        except Exception as e:
            self.logger._log(f"ML indicator handling failed: {e}")
            return f"ML indicator error: {str(e)}"

    # ========== RRS ANALYSIS (Day 37) ==========

    @show_progress("Running RRS analysis...", "Complete")
    async def handle_rrs_analysis(self, query: str) -> str:
        """Handle RRS analysis queries in natural language."""
        try:
            from gordon.research.rrs import RRSManager
            from gordon.backtesting.data.fetcher import DataFetcher
            import yaml
            from pathlib import Path
            import re
            
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            data_fetcher = DataFetcher()
            rrs_manager = RRSManager(data_fetcher=data_fetcher, config=config.get('rrs', {}))
            
            query_lower = query.lower()
            
            # Extract timeframe
            timeframe = '1h'
            for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
                if tf in query_lower:
                    timeframe = tf
                    break
            
            # Extract benchmark
            benchmark = self._extract_symbol(query) or 'BTCUSDT'
            
            # Single symbol analysis
            if any(kw in query_lower for kw in ['analyze', 'analysis', 'rrs for', 'rrs of']):
                symbol = self._extract_symbol(query) or 'BTCUSDT'
                rrs_result = rrs_manager.analyze_symbol(symbol, timeframe=timeframe)
                
                if rrs_result is not None and not rrs_result.empty:
                    latest = rrs_result.iloc[-1]
                    return f"""
📊 **RRS Analysis for {symbol}**

Current RRS: {latest['smoothed_rrs']:.4f}
RRS Momentum: {latest['rrs_momentum']:.4f}
RRS Trend: {latest['rrs_trend']:.4f}
Risk-Adjusted RRS: {latest['risk_adjusted_rrs']:.4f}
Outperformance Ratio: {latest['outperformance_ratio']:.2%}
Volume Ratio: {latest['volume_ratio']:.2f}
"""
                return f"Could not generate RRS analysis for {symbol}"
            
            # Rankings
            elif any(kw in query_lower for kw in ['ranking', 'rank', 'compare', 'top']):
                # Extract symbols
                symbols = []
                common_symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'DOT', 'AVAX']
                for sym in common_symbols:
                    if sym in query.upper():
                        symbols.append(f"{sym}USDT")
                
                if not symbols:
                    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
                
                rrs_results = rrs_manager.analyze_multiple_symbols(symbols, timeframe=timeframe, benchmark=benchmark)
                
                if rrs_results:
                    signals_df = rrs_manager.generate_rankings_and_signals(rrs_results, timeframe)
                    if not signals_df.empty:
                        response = "🏆 **RRS Rankings**\n\n"
                        response += signals_df.head(10)[['rank', 'symbol', 'current_rrs', 'primary_signal', 'signal_confidence']].to_string(index=False)
                        return response
                
                return "Could not generate rankings"
            
            # Signals
            elif any(kw in query_lower for kw in ['signal', 'buy signal', 'sell signal', 'strong buy']):
                signal_type = 'STRONG_BUY'
                if 'strong buy' in query_lower:
                    signal_type = 'STRONG_BUY'
                elif 'buy' in query_lower:
                    signal_type = 'BUY'
                elif 'strong sell' in query_lower:
                    signal_type = 'STRONG_SELL'
                elif 'sell' in query_lower:
                    signal_type = 'SELL'
                
                symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
                rrs_results = rrs_manager.analyze_multiple_symbols(symbols, timeframe=timeframe)
                
                if rrs_results:
                    signals_df = rrs_manager.generate_rankings_and_signals(rrs_results, timeframe)
                    if not signals_df.empty:
                        top_signals = rrs_manager.signal_generator.get_top_signals(signals_df, signal_type, top_n=10)
                        if not top_signals.empty:
                            response = f"🎯 **{signal_type} Signals**\n\n"
                            response += top_signals[['symbol', 'current_rrs', 'signal_confidence', 'risk_level']].to_string(index=False)
                            return response
                
                return f"No {signal_type} signals found"
            
            else:
                return """
📊 **RRS Analysis Help**

I can help with:
- "RRS analysis for BTC" - Analyze single symbol
- "RRS rankings" - Compare multiple symbols
- "Show STRONG_BUY signals" - Get trading signals

Examples:
- "Analyze RRS for ETHUSDT"
- "Compare RRS rankings for BTC, ETH, SOL"
- "Show me strong buy signals"
"""
            
        except Exception as e:
            self.logger._log(f"RRS analysis failed: {e}")
            return f"RRS analysis error: {str(e)}"

    # ========== TRADER INTELLIGENCE (Day 38) ==========

    @show_progress("Analyzing trader intelligence...", "Complete")
    async def handle_trader_intelligence(self, query: str) -> str:
        """Handle trader intelligence queries in natural language."""
        try:
            from gordon.research.trader_intelligence import TraderIntelligenceManager
            from gordon.exchanges.factory import ExchangeFactory
            import yaml
            from pathlib import Path
            import re
            
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            exchange_config = config.get('exchanges', {}).get('binance', {})
            exchange_adapter = ExchangeFactory.create_exchange(
                'binance',
                exchange_config,
                event_bus=None
            )
            
            manager = TraderIntelligenceManager(
                exchange_adapter=exchange_adapter,
                config=config.get('trader_intelligence', {})
            )
            
            query_lower = query.lower()
            
            # Extract symbol
            symbol = self._extract_symbol(query) or 'BTCUSDT'
            
            # Extract days
            numbers = re.findall(r'\d+', query)
            lookback_days = int(numbers[0]) if numbers and any(kw in query_lower for kw in ['day', 'week', 'month']) else 30
            
            # Find accounts to follow
            if any(kw in query_lower for kw in ['find account', 'accounts to follow', 'who to follow', 'follow']):
                max_accounts = int(numbers[0]) if numbers else 20
                accounts = manager.get_accounts_to_follow(symbol, max_accounts=max_accounts)
                
                if accounts:
                    response = f"✅ **Found {len(accounts)} accounts to follow for {symbol}**\n\n"
                    for i, account in enumerate(accounts[:10], 1):
                        response += f"{i}. {account}\n"
                    return response
                return "No accounts found matching criteria"
            
            # Institutional traders
            elif any(kw in query_lower for kw in ['institutional', 'institution', 'whale trader']):
                results = manager.analyze_symbol(symbol, lookback_days=lookback_days)
                
                if not results['classified_trades'].empty:
                    institutional = results['classified_trades'][
                        results['classified_trades']['is_professional']
                    ]
                    
                    if not institutional.empty:
                        trader_profiles = results['trader_profiles']
                        institutional_profiles = trader_profiles[
                            trader_profiles['trader_classification'].isin(['Institutional', 'Whale'])
                        ]
                        
                        response = f"🏛️ **Institutional Traders for {symbol}**\n\n"
                        response += f"Total Institutional Trades: {len(institutional)}\n"
                        response += f"Total Volume: ${institutional['usd_value'].sum():,.2f}\n\n"
                        response += "**Top Institutional Traders:**\n"
                        response += institutional_profiles.head(10)[['rank', 'trader', 'trader_classification', 'total_volume', 'trade_count']].to_string(index=False)
                        return response
                
                return f"No institutional trades found for {symbol}"
            
            # General trader analysis
            else:
                results = manager.analyze_symbol(symbol, lookback_days=lookback_days)
                
                if not results['trader_profiles'].empty:
                    response = f"📊 **Trader Intelligence for {symbol}**\n\n"
                    response += f"Total Trades: {len(results['trades'])}\n"
                    response += f"Unique Traders: {len(results['trader_profiles'])}\n"
                    response += f"Institutional/Whale Trades: {results['classified_trades']['is_professional'].sum()}\n\n"
                    response += "**Top 10 Traders:**\n"
                    response += results['top_traders'].head(10)[['rank', 'trader', 'total_volume', 'trade_count']].to_string(index=False)
                    
                    if not results['early_buyers'].empty:
                        response += "\n\n**Top 5 Early Buyers:**\n"
                        response += results['early_buyers'].head(5)[['trader', 'timestamp', 'usd_value']].to_string(index=False)
                    
                    return response
                
                return f"No trader data found for {symbol}"
            
        except Exception as e:
            self.logger._log(f"Trader intelligence failed: {e}")
            return f"Trader intelligence error: {str(e)}"

    # ========== CONVERSATION COMMANDS (Day 30) ==========

    async def handle_conversation_commands(self, query: str) -> str:
        """Handle conversation management queries in natural language."""
        try:
            query_lower = query.lower()
            
            # Search conversations
            if any(kw in query_lower for kw in ['search', 'find', 'look for']):
                search_query = query.replace('search', '').replace('conversation', '').replace('find', '').strip()
                if not search_query:
                    return "Please specify what to search for. Example: 'Search conversations for Bitcoin'"
                
                from gordon.agent.conversation_search import ConversationSearcher
                import yaml
                from pathlib import Path
                
                config_path = Path(__file__).parent.parent.parent / 'config.yaml'
                config = {}
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                
                memory_dir = config.get('conversation', {}).get('memory_dir', './conversation_memory')
                searcher = ConversationSearcher(memory_dir)
                results = searcher.search(search_query)
                
                if results:
                    response = f"🔍 **Found {len(results)} matching conversations**\n\n"
                    for i, result in enumerate(results[:5], 1):
                        response += f"{i}. {result.get('preview', '')[:100]}...\n"
                    return response
                return f"No conversations found matching '{search_query}'"
            
            # Export conversations
            elif any(kw in query_lower for kw in ['export', 'save', 'download']):
                format_type = 'json'
                if 'csv' in query_lower:
                    format_type = 'csv'
                elif 'txt' in query_lower or 'text' in query_lower:
                    format_type = 'txt'
                
                from gordon.agent.conversation_export import export_all_conversations
                import yaml
                from pathlib import Path
                
                config_path = Path(__file__).parent.parent.parent / 'config.yaml'
                config = {}
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                
                memory_dir = config.get('conversation', {}).get('memory_dir', './conversation_memory')
                export_all_conversations(memory_dir, format_type)
                return f"✅ Conversations exported to {format_type} format"
            
            # Analytics
            elif any(kw in query_lower for kw in ['analytics', 'stats', 'statistics', 'summary']):
                from gordon.agent.conversation_analytics import ConversationAnalytics
                import yaml
                from pathlib import Path
                
                config_path = Path(__file__).parent.parent.parent / 'config.yaml'
                config = {}
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                
                memory_dir = config.get('conversation', {}).get('memory_dir', './conversation_memory')
                analytics = ConversationAnalytics(memory_dir)
                report = analytics.generate_report()
                
                response = "📊 **Conversation Analytics**\n\n"
                response += f"Total Messages: {report.get('total_messages', 0)}\n"
                response += f"Total Conversations: {report.get('total_conversations', 0)}\n"
                response += f"Active Days: {report.get('active_days', 0)}\n"
                return response
            
            # List users
            elif any(kw in query_lower for kw in ['list user', 'users', 'all users']):
                from gordon.agent.multi_user_manager import MultiUserConversationManager
                import yaml
                from pathlib import Path
                
                config_path = Path(__file__).parent.parent.parent / 'config.yaml'
                config = {}
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                
                memory_dir = config.get('conversation', {}).get('memory_dir', './conversation_memory')
                manager = MultiUserConversationManager(memory_dir)
                users = manager.list_users()
                
                if users:
                    response = "👥 **Users**\n\n"
                    for user_id, stats in users.items():
                        response += f"- {user_id}: {stats.get('message_count', 0)} messages\n"
                    return response
                return "No users found"
            
            else:
                return """
💬 **Conversation Management Help**

I can help with:
- "Search conversations for Bitcoin" - Search conversation history
- "Export conversations as CSV" - Export conversations
- "Show conversation analytics" - Get conversation statistics
- "List all users" - Show all conversation users

Examples:
- "Find conversations about ETH"
- "Export my conversations"
- "Show me conversation stats"
"""
            
        except Exception as e:
            self.logger._log(f"Conversation command failed: {e}")
            return f"Conversation error: {str(e)}"

    # ========== SYSTEM COMMANDS ==========

    async def handle_system_commands(self, query: str) -> str:
        """Handle system command queries in natural language."""
        try:
            query_lower = query.lower()
            
            # Status
            if 'status' in query_lower:
                return """
✅ **System Status**:
• Research Agent: Online
• Trading System: Online
• Risk Manager: Active
• Exchanges: Connected
• Strategies: 10+ loaded
"""
            
            # Config
            elif 'config' in query_lower:
                from gordon.agent.config_manager import get_config
                config = get_config()
                return f"⚙️ **Configuration**\n\n{config.display() if hasattr(config, 'display') else 'Config loaded'}"
            
            # Risk
            elif 'risk' in query_lower:
                from gordon.agent.config_manager import get_config
                config = get_config()
                return f"""
🛡️ **Risk Management Settings**:
• Max Position Size: {config.get('trading.risk.max_position_size', 0.1):.1%}
• Max Drawdown: {config.get('trading.risk.max_drawdown', 0.2):.1%}
• Daily Loss Limit: {config.get('trading.risk.daily_loss_limit', 0.05):.1%}
• Risk per Trade: {config.get('trading.risk.risk_per_trade', 0.02):.1%}
• Dry Run Mode: {'✅ ON' if config.is_dry_run() else '❌ OFF'}
"""
            
            # Help
            elif 'help' in query_lower:
                return await self.general_assistance(query)
            
            else:
                return "System command recognized but not handled. Try 'status', 'config', or 'risk'."
            
        except Exception as e:
            self.logger._log(f"System command failed: {e}")
            return f"System command error: {str(e)}"

    # ========== LIQUIDATION HUNTER (Day 45) ==========

    @show_progress("Running liquidation hunter analysis...", "Complete")
    async def handle_liquidation_hunter(self, query: str) -> str:
        """Handle liquidation hunter queries in natural language."""
        try:
            from gordon.core.strategies.liquidation_hunter_strategy import LiquidationHunterStrategy
            from gordon.research.data_providers.moondev_api import MoonDevAPI
            import yaml
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            query_lower = query.lower()
            
            # Extract symbol
            symbol = self._extract_symbol(query) or 'BTCUSDT'
            
            # Moon Dev API data requests
            if any(kw in query_lower for kw in ['liquidation data', 'funding data', 'oi data', 'positions data', 'whale addresses']):
                api = MoonDevAPI()
                
                if 'liquidation' in query_lower:
                    limit = 1000  # Default
                    import re
                    numbers = re.findall(r'\d+', query)
                    if numbers:
                        limit = int(numbers[0])
                    
                    data = api.get_liquidation_data(limit=limit)
                    if data is not None and not data.empty:
                        return f"✅ Fetched {len(data)} liquidation records\n\nPreview:\n{data.head(10).to_string()}"
                    return "Could not fetch liquidation data"
                
                elif 'funding' in query_lower:
                    data = api.get_funding_data()
                    if data is not None and not data.empty:
                        return f"✅ Funding Data\n\n{data.head(10).to_string()}"
                    return "Could not fetch funding data"
                
                elif 'open interest' in query_lower or 'oi' in query_lower:
                    data = api.get_oi_data()
                    if data is not None and not data.empty:
                        return f"✅ Open Interest Data\n\n{data.head(10).to_string()}"
                    return "Could not fetch OI data"
                
                elif 'positions' in query_lower:
                    data = api.get_positions_hlp()
                    if data is not None and not data.empty:
                        return f"✅ Fetched {len(data)} positions\n\n{data.head(10).to_string()}"
                    return "Could not fetch positions data"
                
                elif 'whale address' in query_lower:
                    addresses = api.get_whale_addresses()
                    if addresses:
                        return f"✅ Fetched {len(addresses)} whale addresses\n\nTop 10:\n" + "\n".join([f"{i}. {addr}" for i, addr in enumerate(addresses[:10], 1)])
                    return "Could not fetch whale addresses"
            
            # Liquidation hunter analysis
            else:
                strategy_config = config.get('liquidation_hunter', {})
                strategy_config['symbol'] = symbol
                strategy = LiquidationHunterStrategy(strategy_config)
                
                # Run analysis
                analysis = strategy._analyze_whale_positions()
                
                # Get recent liquidations
                liq_data = strategy._get_recent_liquidations(symbol)
                
                # Check trade signal
                should_enter, reason = strategy._should_enter_trade(
                    liq_data['long_liq_amount'],
                    liq_data['short_liq_amount']
                )
                
                response = f"""
🎯 **Liquidation Hunter Analysis for {symbol}**

**Market Bias:** {analysis.get('bias', 'None')}
**Long Liquidations (3% move):** ${analysis.get('long_liquidations', 0):,.2f}
**Short Liquidations (3% move):** ${analysis.get('short_liquidations', 0):,.2f}

**Recent Liquidations ({strategy.liquidation_lookback_minutes} min):**
- Long: ${liq_data['long_liq_amount']:,.2f}
- Short: ${liq_data['short_liq_amount']:,.2f}

**Trade Signal:** {'✅ ENTER' if should_enter else '⏸️ WAIT'}
{reason}

**Total Positions Analyzed:** {analysis.get('total_positions', 0)}
"""
                return response
            
        except Exception as e:
            self.logger._log(f"Liquidation hunter failed: {e}")
            return f"Liquidation hunter error: {str(e)}"

    # ========== ORDER BOOK ANALYSIS (Day 45) ==========

    @show_progress("Analyzing order book...", "Complete")
    async def handle_orderbook_analysis(self, query: str) -> str:
        """Handle order book analysis queries in natural language."""
        try:
            from gordon.core.utilities import OrderBookAnalyzer
            from gordon.exchanges.factory import ExchangeFactory
            import yaml
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            # Extract symbol
            symbol = self._extract_symbol(query) or 'BTCUSDT'
            
            # Get exchange adapter
            exchange_name = 'binance'  # Default
            exchange_config = config.get('exchanges', {}).get(exchange_name, {})
            exchange_adapter = ExchangeFactory.create_exchange(
                exchange_name,
                exchange_config,
                event_bus=None
            )
            
            # Get order book
            orderbook = await exchange_adapter.get_order_book(symbol)
            if not orderbook:
                return f"Could not fetch order book for {symbol}"
            
            analyzer = OrderBookAnalyzer()
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            current_price = float(orderbook.get('last', 0))
            
            if current_price == 0:
                return f"Could not get current price for {symbol}"
            
            analysis = analyzer.analyze_order_book(bids, asks, current_price)
            
            response = f"""
📊 **Order Book Analysis for {symbol}**

**Current Price:** ${current_price:,.2f}

**Spread Metrics:**
- Best Bid: ${analysis['spread_metrics']['best_bid']:,.2f}
- Best Ask: ${analysis['spread_metrics']['best_ask']:,.2f}
- Spread: ${analysis['spread_metrics']['spread']:,.2f} ({analysis['spread_metrics']['spread_percent']:.4f}%)

**Depth Metrics:**
- Bid Depth: ${analysis['depth_metrics']['bid_depth']:,.2f}
- Ask Depth: ${analysis['depth_metrics']['ask_depth']:,.2f}
- Depth Imbalance: {analysis['depth_metrics']['depth_imbalance']:.2%}
- Bid/Ask Ratio: {analysis['depth_metrics']['bid_ask_ratio']:.2f}

**Whale Analysis:**
- Bias: {analysis['whale_analysis']['bias']}
- Strength: {analysis['whale_analysis']['strength']:.2%}
- Whale Bids: ${analysis['whale_analysis']['whale_bids_value']:,.2f}
- Whale Asks: ${analysis['whale_analysis']['whale_asks_value']:,.2f}
- Total Whale Value: ${analysis['whale_analysis']['total_whale_value']:,.2f}
"""
            return response
            
        except Exception as e:
            self.logger._log(f"Order book analysis failed: {e}")
            return f"Order book analysis error: {str(e)}"

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

        # Execute order using exchange directly (algo_order_manager requires orchestrator)
        # For now, use exchange directly instead of algo_order_manager
        # Default to first available exchange or 'binance'
        exchange = 'binance'  # Default exchange
        exchange_instance = self.exchanges.get(exchange)
        if not exchange_instance and self.exchanges:
            # Use first available exchange
            exchange = list(self.exchanges.keys())[0]
            exchange_instance = self.exchanges[exchange]
        
        if not exchange_instance:
            return f"No exchange connected. Please connect an exchange first."
        
        # Execute trade through exchange
        try:
            result = await exchange_instance.create_order(
                symbol=symbol,
                side='buy',
                type='market',
                amount=position_size
            )
            return f"Executed {position_size:.4f} {symbol} position (Confidence: {confidence_multiplier*100:.0f}%)"
        except Exception as e:
            return f"Failed to execute trade: {e}"

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

🐋 **Whale Tracking** (Day 44):
- Track large positions and whales
- Monitor institutional traders
- Analyze position movements
- Identify top holders

📏 **Position Sizing** (Day 44):
- Calculate position sizes from balance
- Risk-based position sizing
- Leverage calculations
- ATR-based sizing

🤖 **ML Indicator Analysis** (Day 33):
- Discover available indicators
- Evaluate indicator performance
- Top indicator rankings
- Multi-generation evaluation

📊 **RRS Analysis** (Day 37):
- Relative Rotation Strength analysis
- Symbol rankings
- Trading signals
- Multi-timeframe comparison

🔍 **Trader Intelligence** (Day 38):
- Early buyer identification
- Institutional trader analysis
- Accounts to follow discovery
- Smart money tracking

🎯 **Liquidation Hunter** (Day 45):
- Real-time liquidation cascade hunting
- Whale position analysis
- Moon Dev API integration
- Automated trade signals

📊 **Order Book Analysis** (Day 45):
- Whale order detection
- Spread analysis
- Depth imbalance detection
- Market impact estimation

💬 **Conversation Management** (Day 30):
- Search conversation history
- Export conversations
- Conversation analytics
- Multi-user support

💡 **Hybrid Analysis**:
- Combine fundamentals + technicals
- Smart position sizing
- Data-driven decisions

Try asking:
- "Analyze Apple's financials and trade if strong"
- "Run RSI strategy on BTC"
- "Backtest SMA strategy on ETH"
- "Track whale positions for Bitcoin"
- "Calculate position size for $10k balance at $50k price"
- "How much should I buy with 2x leverage?"
- "Show me large positions on BTCUSDT"
- "Evaluate indicators for BTCUSDT"
- "Show me top 10 indicators"
- "RRS analysis for ETH"
- "Find accounts to follow for Bitcoin"
- "Search conversations for ETH"
- "Run liquidation hunter for BTC"
- "Analyze order book for BTCUSDT"
- "Show liquidation data"
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