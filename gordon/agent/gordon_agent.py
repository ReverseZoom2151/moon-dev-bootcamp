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
        
        # Initialize RL Manager for fundamental-level RL integration
        try:
            from gordon.core.rl import RLManager
            from gordon.core.rl.performance_monitor import RLPerformanceMonitor
            from gordon.core.rl.ab_testing import RLABTesting
            
            # Initialize performance monitor and A/B testing
            self.rl_performance_monitor = RLPerformanceMonitor()
            self.rl_ab_testing = RLABTesting(config=self.config)
            
            self.rl_manager = RLManager(config=self.config)
            if self.rl_manager.enabled:
                self.logger._log("🤖 RL Manager initialized - RL integrated at fundamental level")
                # Note: RL components will be initialized on first use or via explicit initialization
            else:
                self.rl_manager = None
                self.logger._log("RL Manager disabled in config")
        except Exception as e:
            self.rl_manager = None
            self.rl_performance_monitor = None
            self.rl_ab_testing = None
            self.logger._log(f"RL Manager not available: {e}")
        
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

        # Determine query type - try LLM first for semantic understanding
        try:
            query_type = await self._classify_query_llm(query)
            if not query_type or query_type == "general":
                # If LLM says general or failed, fall back to keyword matching
                query_type = self._classify_query(query)
            else:
                self.logger._log(f"LLM classified query as: {query_type}")
        except Exception as e:
            self.logger._log(f"LLM classification failed, using keyword matching: {e}")
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
        elif query_type == "market_dashboard":
            return await self.handle_market_dashboard(query)
        elif query_type == "conversation":
            return await self.handle_conversation_commands(query)
        elif query_type == "system":
            return await self.handle_system_commands(query)
        elif query_type == "position_management":
            return await self.handle_position_management(query)
        elif query_type == "ma_reversal":
            return await self.handle_ma_reversal(query)
        elif query_type == "enhanced_sd":
            return await self.handle_enhanced_sd(query)
        elif query_type == "quick_buysell":
            return await self.handle_quick_buysell(query)
        else:
            return await self.general_assistance(query)

    async def _classify_query_llm(self, query: str) -> str:
        """Use LLM to classify query type with semantic understanding."""
        try:
            from .model import call_llm
            
            classification_prompt = """You are a command classifier for Gordon, a trading and research agent.

Analyze the user's query and classify it into ONE of these categories based on intent:

- "research": Financial research, company analysis, SEC filings, earnings
- "trading": Execute trades, run strategies (RSI, SMA, VWAP), buy/sell orders
- "hybrid": Both research and trading combined
- "backtest": Historical testing, strategy performance, optimization
- "whale_tracking": Track large positions, whales, institutional traders, multi-address tracking, liquidation risk, aggregate positions
- "position_sizing": Calculate position sizes, how much to buy, leverage calculations
- "ml_indicators": ML indicator evaluation, discovery, ranking, top indicators
- "rrs_analysis": Relative Rotation Strength, RRS rankings, RRS signals
- "trader_intelligence": Early buyers, trader analysis, accounts to follow, institutional traders
- "liquidation_hunter": Liquidation analysis, liquidation cascades, Moon Dev API data
- "orderbook_analysis": Order book depth, whale orders, spread analysis, market depth
- "market_dashboard": Trending tokens, new listings, volume leaders, funding rates, market overview
- "position_management": Close positions based on PnL, chunk closing, easy entry, track positions, check PnL
- "ma_reversal": MA reversal strategy, dual moving average crossover, 2x MA reversal
- "enhanced_sd": Enhanced supply/demand zones, SD zone strategy, zone trading
- "quick_buysell": Quick buy/sell execution, rapid trading, instant execution, file monitoring
- "conversation": Search conversations, export conversations, conversation analytics, list users
- "system": System status, config, risk settings, help
- "general": General questions, clarification, or unclear intent

Respond with ONLY the category name, nothing else.

User Query: "{query}"

Category:"""
            
            import asyncio
            loop = asyncio.get_event_loop()
            llm_response = await loop.run_in_executor(
                None,
                lambda: call_llm(
                    prompt=query,
                    system_prompt=classification_prompt.format(query=query),
                    model="gpt-4o-mini"
                )
            )
            
            # Extract category from LLM response
            response_text = str(llm_response.content) if hasattr(llm_response, 'content') else str(llm_response)
            category = response_text.strip().lower()
            
            # Validate category
            valid_categories = [
                "research", "trading", "hybrid", "backtest", "whale_tracking",
                "position_sizing", "ml_indicators", "rrs_analysis", "trader_intelligence",
                "liquidation_hunter", "orderbook_analysis", "market_dashboard",
                "position_management", "ma_reversal", "enhanced_sd", "quick_buysell",
                "conversation", "system", "general"
            ]
            
            if category in valid_categories:
                return category
            
            # Fallback: check if category is mentioned in response
            for valid_cat in valid_categories:
                if valid_cat in category:
                    return valid_cat
            
            return "general"
            
        except Exception as e:
            self.logger._log(f"LLM classification failed: {e}, falling back to keyword matching")
            return None  # Signal to fall back to keyword matching

    def _classify_query(self, query: str) -> str:
        """Classify the type of query using LLM first, then keyword fallback."""
        # Try LLM classification first (async, but we'll handle sync fallback)
        # For now, use hybrid approach: keyword matching for speed, LLM for ambiguous cases
        query_lower = query.lower()

        # Keywords for different query types
        research_keywords = ['analyze', 'financial', 'revenue', 'earnings', 'debt',
                           'balance sheet', 'income statement', 'cash flow', '10-k', '10-q']
        trading_keywords = ['buy', 'sell', 'trade', 'position', 'execute', 'order',
                          'strategy', 'rsi', 'sma', 'vwap', 'bollinger']
        backtest_keywords = ['backtest', 'historical', 'test strategy', 'performance']
        whale_keywords = ['whale', 'large position', 'track whale', 'whale position', 
                         'institutional', 'big holder', 'whale tracker', 'multi-address',
                         'multiple whale', 'aggregate position', 'liquidation risk',
                         'distance to liquidation', 'positions closest to liquidation']
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
        market_dashboard_keywords = ['market dashboard', 'trending tokens', 'new listings', 'volume leaders',
                                    'trending', 'new tokens', 'volume', 'funding rates', 'market tracker',
                                    'gems', 'possible gems', 'consistent trending']
        # Day 48: Position Management
        position_mgmt_keywords = ['pnl close', 'close position pnl', 'close based on pnl', 'take profit', 'stop loss',
                                 'chunk close', 'close in chunks', 'close gradually', 'sell in chunks',
                                 'easy entry', 'average in', 'accumulate position', 'build position',
                                 'track position', 'watch position', 'monitor position', 'start tracking',
                                 'check pnl', 'pnl status', 'profit loss status', 'current pnl']
        # Day 49: MA Reversal
        ma_reversal_keywords = ['ma reversal', '2x ma', 'dual moving average', 'ma crossover reversal',
                               'reversal strategy', 'run ma reversal', 'execute ma reversal']
        # Day 50: Enhanced Supply/Demand
        enhanced_sd_keywords = ['enhanced supply demand', 'enhanced sd', 'supply demand zone strategy',
                               'sd zone strategy', 'zone strategy', 'enhanced zone']
        # Day 51: Quick Buy/Sell
        quick_buysell_keywords = ['quick buy', 'quick sell', 'rapid buy', 'rapid sell', 'instant buy', 'instant sell',
                                 'quick execution', 'qbs', 'quick-buy', 'quick-sell', 'file monitor', 'token file']

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
        has_market_dashboard = any(kw in query_lower for kw in market_dashboard_keywords)
        has_position_mgmt = any(kw in query_lower for kw in position_mgmt_keywords)
        has_ma_reversal = any(kw in query_lower for kw in ma_reversal_keywords)
        has_enhanced_sd = any(kw in query_lower for kw in enhanced_sd_keywords)
        has_quick_buysell = any(kw in query_lower for kw in quick_buysell_keywords)

        if has_backtest:
            return "backtest"
        elif has_quick_buysell:
            return "quick_buysell"
        elif has_position_mgmt:
            return "position_management"
        elif has_ma_reversal:
            return "ma_reversal"
        elif has_enhanced_sd:
            return "enhanced_sd"
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
        elif has_market_dashboard:
            return "market_dashboard"
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
        """
        Execute trading operations with RL-integrated decision making.
        RL is embedded at the fundamental level for:
        - Strategy selection
        - Signal aggregation
        - Position sizing optimization
        - Market regime detection
        - Risk parameter optimization
        
        Features:
        - Performance monitoring for all RL calls
        - Timeout protection to prevent hanging
        - A/B testing to compare RL vs baseline
        - Fail-fast option to disable RL on errors
        """
        import uuid
        import asyncio
        
        trade_id = str(uuid.uuid4())[:8]  # Generate unique trade ID
        symbol = None
        
        try:
            # Parse trading intent
            trading_params = self._parse_trading_query(query)
            symbol = trading_params.get('symbol', 'BTCUSDT')
            
            # Get RL config
            rl_config = self.config.get('rl', {})
            fail_fast = rl_config.get('fail_fast', False)
            timeout_ms = rl_config.get('timeout_ms', 100)  # Default 100ms timeout
            timeout_seconds = timeout_ms / 1000.0
            
            # A/B testing: determine if RL should be used
            use_rl = True
            if self.rl_ab_testing:
                use_rl = self.rl_ab_testing.should_use_rl(trade_id)
            
            # Baseline decisions (non-RL)
            baseline_decisions = {
                'regime': None,
                'risk_params': None,
                'signal_aggregation': None,
                'position_size_multiplier': 1.0
            }
            
            # ========== RL-ENHANCED DECISION MAKING (with monitoring & timeouts) ==========
            
            # 1. Detect market regime using RL
            market_regime = None
            market_context = await self._get_market_context(symbol)
            
            if use_rl and self.rl_manager and self.rl_manager.enabled and self.rl_manager.regime_detector:
                try:
                    price_data = await self._get_price_data(symbol)
                    
                    # RL call with timeout and monitoring
                    regime_result = await self._safe_rl_call(
                        component='regime_detector',
                        operation='detect_regime',
                        call_func=self.rl_manager.detect_regime,
                        call_args=(market_context, price_data),
                        timeout=timeout_seconds,
                        fail_fast=fail_fast
                    )
                    
                    if regime_result:
                        market_regime = regime_result.get('regime_name')
                        self.logger._log(f"🤖 RL Detected Market Regime: {market_regime} (Confidence: {regime_result.get('confidence', 0):.2%})")
                        baseline_decisions['regime'] = market_regime
                except Exception as e:
                    self.logger._log(f"RL regime detection failed: {e}")
                    if fail_fast:
                        return f"Trade aborted: RL regime detection failed (fail_fast enabled)"
            
            # 2. Optimize risk parameters using RL
            rl_risk = None
            if use_rl and self.rl_manager and self.rl_manager.enabled and self.rl_manager.risk_optimizer:
                try:
                    portfolio_context = await self._get_portfolio_context()
                    performance_context = await self._get_performance_context()
                    
                    # RL call with timeout and monitoring
                    rl_risk = await self._safe_rl_call(
                        component='risk_optimizer',
                        operation='optimize_risk',
                        call_func=self.rl_manager.optimize_risk,
                        call_args=(portfolio_context, market_context, performance_context),
                        timeout=timeout_seconds,
                        fail_fast=fail_fast
                    )
                    
                    if rl_risk:
                        self.logger._log(f"🤖 RL Optimized Risk: {rl_risk.get('risk_level_name')} (Risk/Trade: {rl_risk.get('risk_per_trade', 0):.2%})")
                        trading_params['rl_risk_params'] = rl_risk
                        baseline_decisions['risk_params'] = rl_risk
                except Exception as e:
                    self.logger._log(f"RL risk optimization failed: {e}")
                    if fail_fast:
                        return f"Trade aborted: RL risk optimization failed (fail_fast enabled)"
            
            # 3. Check risk management (may use RL-optimized parameters)
            if not self.risk_manager.check_trade_allowed(trading_params):
                return "Trade rejected by risk management"
            
            # 4. Select strategy (may use RL meta-strategy selection)
            strategy = self.strategy_manager.get_strategy(trading_params.get('strategy'))
            if not strategy:
                return "No suitable strategy found"
            
            # 5. Execute strategy and get signal
            signal = await strategy.execute()
            if not signal:
                return "No trading signal generated"
            
            # 6. Aggregate signals if multiple strategies exist (RL-enhanced)
            aggregated_signal = signal.copy()
            if use_rl and self.rl_manager and self.rl_manager.enabled and self.rl_manager.signal_aggregator:
                try:
                    # Get signals from multiple strategies if available
                    all_signals = {strategy.name: signal} if hasattr(strategy, 'name') else {'strategy': signal}
                    
                    # RL call with timeout and monitoring
                    aggregated = await self._safe_rl_call(
                        component='signal_aggregator',
                        operation='aggregate_signals',
                        call_func=self.rl_manager.aggregate_signals,
                        call_args=(all_signals, market_context),
                        timeout=timeout_seconds,
                        fail_fast=fail_fast
                    )
                    
                    if aggregated:
                        aggregated_signal.update(aggregated)
                        self.logger._log(f"🤖 RL Aggregated Signal: {aggregated.get('action')} (Confidence: {aggregated.get('confidence', 0):.2%})")
                        baseline_decisions['signal_aggregation'] = aggregated
                except Exception as e:
                    self.logger._log(f"RL signal aggregation failed: {e}")
                    if fail_fast:
                        return f"Trade aborted: RL signal aggregation failed (fail_fast enabled)"
            
            # 7. Optimize position size using RL
            optimized_size = signal.get('size', self.config.get('base_position_size', 0.01))
            size_multiplier = 1.0
            
            if use_rl and self.rl_manager and self.rl_manager.enabled and self.rl_manager.position_sizer:
                try:
                    signal_context = {
                        'confidence': aggregated_signal.get('confidence', 0.5),
                        'strength': aggregated_signal.get('strength', 0.5),
                        'strategy_performance': 0.1,  # Should be calculated from actual performance
                        'recent_win_rate': 0.6  # Should be calculated from actual win rate
                    }
                    portfolio_context = await self._get_portfolio_context()
                    
                    # RL call with timeout and monitoring
                    size_optimization = await self._safe_rl_call(
                        component='position_sizer',
                        operation='optimize_position_size',
                        call_func=self.rl_manager.optimize_position_size,
                        call_args=(signal_context, market_context, portfolio_context),
                        timeout=timeout_seconds,
                        fail_fast=fail_fast
                    )
                    
                    if size_optimization:
                        base_size = self.config.get('base_position_size', 0.01)
                        size_multiplier = size_optimization.get('size_multiplier', 1.0)
                        optimized_size = base_size * size_multiplier
                        aggregated_signal['size'] = optimized_size
                        aggregated_signal['rl_size_multiplier'] = size_multiplier
                        self.logger._log(f"🤖 RL Optimized Position Size: {optimized_size:.4f} (Multiplier: {size_multiplier:.2f}x)")
                        baseline_decisions['position_size_multiplier'] = size_multiplier
                except Exception as e:
                    self.logger._log(f"RL position sizing failed: {e}")
                    if fail_fast:
                        return f"Trade aborted: RL position sizing failed (fail_fast enabled)"
            
            # Record A/B test decision
            if self.rl_ab_testing:
                rl_decision = {
                    'regime': market_regime,
                    'risk_params': rl_risk,
                    'position_size_multiplier': size_multiplier,
                    'confidence': aggregated_signal.get('confidence', 0.5)
                }
                self.rl_ab_testing.record_decision(
                    trade_id=trade_id,
                    symbol=symbol,
                    rl_decision=rl_decision,
                    baseline_decision=baseline_decisions,
                    rl_used=use_rl
                )
            
            # 8. Execute trade with RL-optimized parameters
            result = await self._execute_trade(aggregated_signal)
            
            # Build response with RL insights
            response = f"Trade executed: {result}"
            if market_regime:
                response += f"\n📊 Market Regime: {market_regime}"
            if aggregated_signal.get('rl_size_multiplier'):
                response += f"\n🤖 RL Position Size Multiplier: {aggregated_signal['rl_size_multiplier']:.2f}x"
            if use_rl:
                response += f"\n🤖 RL Group: {'Enabled' if use_rl else 'Disabled'} (A/B Test)"
            else:
                response += f"\n📊 Baseline Group: Enabled (A/B Test)"
            
            # Log performance summary if available
            if self.rl_performance_monitor:
                summary = self.rl_performance_monitor.get_summary()
                self.logger._log(summary)
            
            return response
            
        except Exception as e:
            self.logger._log(f"Trading failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Trading error: {str(e)}"
    
    async def _safe_rl_call(
        self,
        component: str,
        operation: str,
        call_func,
        call_args: tuple = (),
        timeout: float = 0.1,
        fail_fast: bool = False
    ):
        """
        Safely execute an RL call with timeout protection and performance monitoring.
        
        Args:
            component: RL component name (e.g., 'regime_detector')
            operation: Operation name (e.g., 'detect_regime')
            call_func: Function to call (must be synchronous or async)
            call_args: Arguments to pass to call_func
            timeout: Timeout in seconds
            fail_fast: Whether to raise exception on failure
            
        Returns:
            Result from RL call or None if failed/timed out
        """
        import asyncio
        import time
        
        start_time = time.time()
        success = False
        error = None
        result = None
        
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(call_func):
                result = await asyncio.wait_for(call_func(*call_args), timeout=timeout)
            else:
                # For synchronous functions, run in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: call_func(*call_args)),
                    timeout=timeout
                )
            success = True
            
        except asyncio.TimeoutError:
            error = f"Timeout after {timeout*1000:.0f}ms"
            self.logger.warning(f"RL {component}.{operation} timed out after {timeout*1000:.0f}ms")
            if fail_fast:
                raise TimeoutError(error)
                
        except Exception as e:
            error = str(e)
            self.logger.warning(f"RL {component}.{operation} failed: {e}")
            if fail_fast:
                raise
        
        finally:
            # Record performance metrics
            latency_ms = (time.time() - start_time) * 1000
            if self.rl_performance_monitor:
                self.rl_performance_monitor.record_call(
                    component_name=component,
                    operation=operation,
                    latency_ms=latency_ms,
                    success=success,
                    error=error
                )
        
        return result

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

    # ========== WHALE TRACKING (Day 44 + Day 46) ==========

    @show_progress("Tracking whale positions...", "Whale tracking complete")
    async def handle_whale_tracking(self, query: str) -> str:
        """Handle whale tracking queries in natural language. Enhanced with Day 46 features."""
        try:
            from gordon.core.utilities import WhaleTrackingManager
            from gordon.exchanges.factory import ExchangeFactory
            import yaml
            import pandas as pd
            from pathlib import Path
            import re
            
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
            value_matches = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', query)
            if value_matches:
                # Convert to float (remove commas)
                min_value = float(value_matches[-1].replace(',', ''))
            
            query_lower = query.lower()
            
            # Day 46: Multi-address tracking
            if any(kw in query_lower for kw in ['multi-address', 'multiple whale', 'multiple address', 'track multiple']):
                results = await manager.track_multi_address_whales(
                    symbol=symbol,
                    min_value_usd=min_value,
                    export_csv=True
                )
                
                if results['positions'].empty:
                    return "❌ No positions found for tracked addresses. Make sure you have whale addresses configured in whale_addresses.txt"
                
                response = f"""
🐋🐋 MULTI-ADDRESS WHALE TRACKING RESULTS

✅ Found {len(results['positions'])} positions from {len(results['positions']['address'].unique())} addresses

"""
                # Add aggregated report
                agg_report = manager.get_aggregated_positions_report(results['positions'])
                response += agg_report
                
                # Add liquidation risk if available
                if not results['liquidation_risk'].empty:
                    liq_report = await manager.get_liquidation_risk_report(
                        results['positions'],
                        threshold_pct=config.get('whale_tracking', {}).get('multi_address_tracking', {}).get('liquidation_risk_threshold', 3.0)
                    )
                    response += f"\n\n{liq_report}"
                
                # Show saved files
                saved_files = results.get('saved_files', {})
                if saved_files:
                    response += "\n\n💾 CSV Files Saved:"
                    for key, path in saved_files.items():
                        if path:
                            response += f"\n   • {key}: {path}"
                
                return response
            
            # Day 46: Liquidation risk analysis
            elif any(kw in query_lower for kw in ['liquidation risk', 'distance to liquidation', 'closest to liquidation', 'liq risk']):
                # Get positions first
                results = await manager.track_multi_address_whales(
                    symbol=symbol,
                    export_csv=False
                )
                
                positions_df = results.get('positions', pd.DataFrame())
                if positions_df.empty:
                    # Fallback to single tracking
                    single_results = await manager.track_whales(symbol=symbol, min_value_usd=min_value)
                    positions_df = single_results.get('whale_positions', pd.DataFrame())
                
                if positions_df.empty:
                    return "❌ No positions found for liquidation risk analysis"
                
                # Extract threshold from query
                threshold = 3.0
                threshold_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', query)
                if threshold_matches:
                    threshold = float(threshold_matches[-1])
                
                report = await manager.get_liquidation_risk_report(
                    positions_df,
                    threshold_pct=threshold,
                    top_n=20
                )
                
                return f"""
💥 LIQUIDATION RISK ANALYSIS

{report}
"""
            
            # Day 46: Aggregated positions
            elif any(kw in query_lower for kw in ['aggregate', 'aggregated', 'group by coin', 'positions by coin']):
                results = await manager.track_multi_address_whales(
                    symbol=symbol,
                    export_csv=False
                )
                
                positions_df = results.get('positions', pd.DataFrame())
                if positions_df.empty:
                    single_results = await manager.track_whales(symbol=symbol, min_value_usd=min_value)
                    positions_df = single_results.get('whale_positions', pd.DataFrame())
                
                if positions_df.empty:
                    return "❌ No positions found for aggregation"
                
                report = manager.get_aggregated_positions_report(positions_df)
                
                return f"""
📊 AGGREGATED POSITIONS REPORT

{report}
"""
            
            # Standard whale tracking (Day 44)
            else:
                results = await manager.track_whales(symbol=symbol, min_value_usd=min_value)
                
                # Generate response
                report = manager.get_whale_summary_report(results)
                
                response = f"""
🐋 WHALE POSITION TRACKING

{report}
"""
                
                if not results['top_positions'].empty:
                    response += "\n📊 Top Positions:\n"
                    response += results['top_positions'][['symbol', 'position_value_usd', 'pnl_percent', 'whale_tier']].head(10).to_string(index=False)
                
                return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Whale tracking error: {str(e)}"

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

    # ========== MARKET DASHBOARD (Day 47) ==========

    @show_progress("Fetching market data...", "Complete")
    async def handle_market_dashboard(self, query: str) -> str:
        """
        Handle market dashboard queries (Day 47).
        
        Supports:
        - Trending tokens
        - New listings
        - Volume leaders
        - Funding rates (Bitfinex)
        - Market dashboard full analysis
        """
        query_lower = query.lower()
        
        # Extract exchange name
        exchange = 'binance'  # default
        if 'bitfinex' in query_lower:
            exchange = 'bitfinex'
        elif 'binance' in query_lower:
            exchange = 'binance'
        
        try:
            from gordon.market import MarketDashboard, FundingRateAnalyzer
            from gordon.exchanges.factory import ExchangeFactory
            import yaml
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            exchange_config = config.get('exchanges', {}).get(exchange, {})
            exchange_adapter = ExchangeFactory.create_exchange(
                exchange,
                exchange_config,
                event_bus=None
            )
            
            await exchange_adapter.initialize()
            
            market_config = config.get('market_dashboard', {}).get(exchange, {})
            
            # Determine what to fetch based on query
            if 'trending' in query_lower or 'trending tokens' in query_lower:
                dashboard = MarketDashboard(
                    exchange_adapter=exchange_adapter,
                    exchange_name=exchange,
                    config=market_config
                )
                trending_df = await dashboard.market_client.fetch_trending_tokens()
                
                if not trending_df.empty:
                    top_5 = trending_df.head(5)
                    response = f"🚀 Top 5 Trending Tokens on {exchange.upper()}:\n"
                    for i, (_, row) in enumerate(top_5.iterrows(), 1):
                        symbol = row.get('symbol', 'N/A')
                        change = row.get('price24hChangePercent', 0)
                        volume = row.get('volume24hUSD', 0)
                        response += f"{i}. {symbol}: {change:+.2f}% (${volume:,.0f} volume)\n"
                    return response
                else:
                    return f"❌ No trending tokens found on {exchange.upper()}"
            
            elif 'new listing' in query_lower or 'new token' in query_lower:
                dashboard = MarketDashboard(
                    exchange_adapter=exchange_adapter,
                    exchange_name=exchange,
                    config=market_config
                )
                listings_df = await dashboard.market_client.fetch_new_listings()
                
                if not listings_df.empty:
                    top_5 = listings_df.head(5)
                    response = f"🌟 Top 5 New Listings on {exchange.upper()}:\n"
                    for i, (_, row) in enumerate(top_5.iterrows(), 1):
                        symbol = row.get('symbol', 'N/A')
                        volume = row.get('volume24hUSD', 0)
                        response += f"{i}. {symbol}: ${volume:,.0f} volume\n"
                    return response
                else:
                    return f"❌ No new listings found on {exchange.upper()}"
            
            elif 'volume' in query_lower and 'leader' in query_lower:
                dashboard = MarketDashboard(
                    exchange_adapter=exchange_adapter,
                    exchange_name=exchange,
                    config=market_config
                )
                volume_df = await dashboard.market_client.fetch_high_volume_tokens()
                
                if not volume_df.empty:
                    top_5 = volume_df.head(5)
                    response = f"📊 Top 5 Volume Leaders on {exchange.upper()}:\n"
                    for i, (_, row) in enumerate(top_5.iterrows(), 1):
                        symbol = row.get('symbol', 'N/A')
                        volume = row.get('volume24hUSD', 0)
                        response += f"{i}. {symbol}: ${volume:,.0f}\n"
                    return response
                else:
                    return f"❌ No volume data found on {exchange.upper()}"
            
            elif 'funding rate' in query_lower and exchange == 'bitfinex':
                analyzer = FundingRateAnalyzer(
                    exchange_adapter=exchange_adapter,
                    config=market_config
                )
                funding_df = await analyzer.fetch_funding_rates()
                
                if not funding_df.empty:
                    arbitrage_df = analyzer.analyze_arbitrage_opportunities(funding_df)
                    if not arbitrage_df.empty:
                        top_5 = arbitrage_df.head(5)
                        response = f"💰 Top 5 Funding Rate Opportunities on Bitfinex:\n"
                        for i, (_, row) in enumerate(top_5.iterrows(), 1):
                            symbol = row.get('base_symbol', 'N/A')
                            rate = row.get('funding_rate', 0)
                            response += f"{i}. {symbol}: {rate:+.4f}%\n"
                        return response
                    else:
                        return "❌ No high funding rate opportunities found"
                else:
                    return "❌ No funding rate data available"
            
            else:
                # Full dashboard analysis
                dashboard = MarketDashboard(
                    exchange_adapter=exchange_adapter,
                    exchange_name=exchange,
                    config=market_config
                )
                results = await dashboard.run_full_analysis(export_csv=False)
                summary = dashboard.get_summary_report(results)
                return summary
        
        except Exception as e:
            self.logger._log(f"Market dashboard analysis failed: {e}")
            return f"Market dashboard error: {str(e)}"

    # ========== QUICK BUY/SELL (Day 51) ==========

    @show_progress("Processing quick buy/sell request...", "Complete")
    async def handle_quick_buysell(self, query: str) -> str:
        """Handle Day 51 Quick Buy/Sell queries in natural language."""
        try:
            import yaml
            from pathlib import Path
            import re
            from gordon.exchanges.factory import ExchangeFactory
            
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
            config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            
            query_lower = query.lower()
            
            # Extract symbol
            symbol = self._extract_symbol(query)
            
            # Extract USD amount
            numbers = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', query)
            usd_amount = float(numbers[0].replace(',', '')) if numbers else None
            
            # Get exchange adapter
            qbs_config = config.get('quick_buysell', {})
            exchange_name = qbs_config.get('exchange', 'binance')
            exchange_config = config.get('exchanges', {}).get(exchange_name, {})
            exchange_adapter = ExchangeFactory.create_exchange(
                exchange_name,
                exchange_config,
                event_bus=None
            )
            await exchange_adapter.initialize()
            
            from gordon.core.strategies import QuickBuySellStrategy
            strategy = QuickBuySellStrategy(
                exchange_adapter=exchange_adapter,
                config=qbs_config
            )
            
            # Determine action
            if any(kw in query_lower for kw in ['buy', 'purchase', 'acquire']):
                # Quick buy
                if not symbol:
                    return "Please specify a symbol. Example: 'Quick buy BTCUSDT for $10'"
                
                amount = usd_amount or qbs_config.get('usd_size', 10.0)
                await strategy.quick_buy_manual(symbol, amount)
                
                return f"✅ Quick buy executed: {symbol} for ${amount:.2f}"
            
            elif any(kw in query_lower for kw in ['sell', 'close', 'exit']):
                # Quick sell
                if not symbol:
                    return "Please specify a symbol. Example: 'Quick sell BTCUSDT'"
                
                await strategy.quick_sell_manual(symbol)
                
                return f"✅ Quick sell executed: {symbol}"
            
            elif any(kw in query_lower for kw in ['monitor', 'watch', 'start', 'file']):
                # Start monitoring
                await strategy.initialize(qbs_config)
                await strategy.start_monitoring()
                
                file_path = qbs_config.get('token_file_path', './token_addresses.txt')
                return f"""
✅ Quick Buy/Sell monitor started!

📁 Monitoring file: {file_path}

To buy: Add token symbol to file (e.g., "BTCUSDT")
To sell: Add token symbol + 'x' to file (e.g., "BTCUSDT x")

The monitor will execute trades instantly when tokens are added.
Press Ctrl+C to stop monitoring.
"""
            
            elif any(kw in query_lower for kw in ['add', 'write', 'append']):
                # Add to file
                if not symbol:
                    return "Please specify a symbol. Example: 'Add BTCUSDT to quick file'"
                
                command = 'SELL' if 'sell' in query_lower else 'BUY'
                
                from gordon.core.utilities import add_token_to_file
                file_path = qbs_config.get('token_file_path', './token_addresses.txt')
                add_token_to_file(file_path, symbol, command)
                
                return f"✅ Added {symbol} ({command}) to {file_path}"
            
            else:
                return """
⚡ **Quick Buy/Sell Help** (Day 51)

I can help with:
- "Quick buy BTCUSDT for $10" - Instant market buy
- "Quick sell ETHUSDT" - Instant market sell
- "Start quick buy/sell monitor" - Monitor file for rapid execution
- "Add BTCUSDT to quick file" - Add token to monitoring file

**File Format:**
- Token symbol only = BUY (e.g., "BTCUSDT")
- Token symbol + 'x' = SELL (e.g., "BTCUSDT x")

**Examples:**
- "Buy Bitcoin quickly for $10"
- "Sell my Ethereum position instantly"
- "Start monitoring token file"
- "Add SOLUSDT to file for buying"
"""
                
        except Exception as e:
            self.logger._log(f"Quick buy/sell failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Quick buy/sell error: {str(e)}"

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
    
    async def _get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Get market context for RL components."""
        try:
            # Try to get actual market data
            exchange = self.exchanges.get('binance') or (list(self.exchanges.values())[0] if self.exchanges else None)
            if exchange:
                # Get recent price data
                ticker = await exchange.get_ticker(symbol)
                # Get 24h price change
                price_change_24h = ticker.get('change24h', 0.0) if ticker else 0.0
                # Get volume ratio (simplified)
                volume_ratio = 1.0  # Should be calculated from historical volume
                
                return {
                    'volatility': abs(price_change_24h) if price_change_24h else 0.02,
                    'trend': 1.0 if price_change_24h > 0 else -1.0 if price_change_24h < 0 else 0.0,
                    'volume_ratio': volume_ratio,
                    'price_change_24h': price_change_24h,
                    'rsi': 50.0,  # Should be calculated from actual indicators
                    'macd': 0.0,  # Should be calculated
                    'bb_position': 0.5,  # Should be calculated
                    'spread': 0.001,
                    'liquidity': 1.0,
                    'regime': 0.5,
                    'fear_greed_index': 50.0,
                    'correlation': 0.0
                }
        except Exception as e:
            self.logger._log(f"Could not get market context: {e}")
        
        # Fallback to default values
        return {
            'volatility': 0.02,
            'trend': 0.0,
            'volume_ratio': 1.0,
            'price_change_24h': 0.0,
            'rsi': 50.0,
            'macd': 0.0,
            'bb_position': 0.5,
            'spread': 0.001,
            'liquidity': 1.0,
            'regime': 0.5,
            'fear_greed_index': 50.0,
            'correlation': 0.0
        }
    
    async def _get_price_data(self, symbol: str) -> Dict[str, Any]:
        """Get price data for regime detection."""
        try:
            exchange = self.exchanges.get('binance') or (list(self.exchanges.values())[0] if self.exchanges else None)
            if exchange:
                # Get recent OHLCV data
                # Simplified - should get actual historical data
                ticker = await exchange.get_ticker(symbol)
                current_price = ticker.get('last', 0.0) if ticker else 0.0
                
                return {
                    'price_change_1h': 0.001,  # Should be calculated
                    'price_change_24h': ticker.get('change24h', 0.0) if ticker else 0.0,
                    'price_change_7d': 0.05,  # Should be calculated
                    'high_low_ratio': 1.02  # Should be calculated
                }
        except Exception as e:
            self.logger._log(f"Could not get price data: {e}")
        
        # Fallback
        return {
            'price_change_1h': 0.001,
            'price_change_24h': 0.0,
            'price_change_7d': 0.05,
            'high_low_ratio': 1.02
        }
    
    async def _get_portfolio_context(self) -> Dict[str, Any]:
        """Get portfolio context for RL components."""
        try:
            # Get positions from position manager
            positions = self.position_manager.get_all_positions() if hasattr(self.position_manager, 'get_all_positions') else []
            
            # Get balance from risk manager
            risk_metrics = self.risk_manager.get_risk_metrics() if hasattr(self.risk_manager, 'get_risk_metrics') else {}
            
            balance = risk_metrics.get('current_balance', 10000.0)
            drawdown = risk_metrics.get('current_drawdown', 0.0)
            position_count = len(positions) if positions else 0
            
            return {
                'balance': balance,
                'drawdown': drawdown,
                'drawdown_pct': drawdown / balance if balance > 0 else 0.0,
                'leverage': risk_metrics.get('max_leverage', 1.0),
                'position_count': position_count,
                'margin_used': 0.0,  # Should be calculated
                'available_balance': balance * 0.5,  # Simplified
                'correlation': 0.3,  # Should be calculated
                'max_position_size': risk_metrics.get('max_position_size', 0.1)
            }
        except Exception as e:
            self.logger._log(f"Could not get portfolio context: {e}")
        
        # Fallback
        return {
            'balance': 10000.0,
            'drawdown': 0.0,
            'drawdown_pct': 0.0,
            'leverage': 1.0,
            'position_count': 0,
            'margin_used': 0.0,
            'available_balance': 5000.0,
            'correlation': 0.3,
            'max_position_size': 0.1
        }
    
    async def _get_performance_context(self) -> Dict[str, Any]:
        """Get performance context for RL components."""
        try:
            risk_metrics = self.risk_manager.get_risk_metrics() if hasattr(self.risk_manager, 'get_risk_metrics') else {}
            
            return {
                'win_rate': 0.5,  # Should be calculated from actual trades
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0.0),
                'total_pnl': risk_metrics.get('total_pnl', 0.0),
                'total_pnl_pct': risk_metrics.get('total_pnl_pct', 0.0),
                'daily_pnl': risk_metrics.get('daily_pnl', 0.0),
                'daily_pnl_pct': risk_metrics.get('daily_pnl_pct', 0.0),
                'consecutive_losses': 0  # Should be tracked
            }
        except Exception as e:
            self.logger._log(f"Could not get performance context: {e}")
        
        # Fallback
        return {
            'win_rate': 0.5,
            'sharpe_ratio': 0.0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'daily_pnl': 0.0,
            'daily_pnl_pct': 0.0,
            'consecutive_losses': 0
        }

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
        """Handle general queries using LLM for true semantic understanding."""
        try:
            from .model import call_llm
            
            # Build comprehensive context about all available commands
            command_context = """
You are Gordon, a comprehensive trading and research agent. You can execute commands in these categories:

**POSITION MANAGEMENT (Day 48)**:
- Close positions when profit/loss thresholds are reached (take profit/stop loss)
- Close large positions in smaller chunks to minimize market impact
- Easy entry strategies (averaging into positions at good prices)
- Track positions for automatic PnL monitoring
- Check current profit/loss status

**POSITION MANAGEMENT (Day 48)**:
- Close positions when profit/loss thresholds are reached (take profit/stop loss)
- Close large positions in smaller chunks to minimize market impact
- Easy entry strategies (averaging into positions at good prices)
- Track positions for automatic PnL monitoring
- Check current profit/loss status

**QUICK BUY/SELL (Day 51)**:
- Instant market buy/sell execution
- File-based rapid execution system
- Monitor text file for token symbols
- Perfect for arbitrage opportunities and new token launches

**STRATEGIES**:
- MA Reversal Strategy (Day 49): Dual moving average crossover reversal
- Enhanced Supply/Demand Strategy (Day 50): Trend-based zone trading
- RSI, SMA, VWAP, Bollinger Bands strategies
- Liquidation Hunter Strategy
- Market Making Strategy

**MARKET ANALYSIS**:
- Whale tracking (multi-address, liquidation risk, position aggregation)
- Position sizing calculations
- Order book depth analysis
- Market dashboard (trending tokens, new listings, volume leaders)
- RRS (Relative Rotation Strength) analysis
- ML indicator evaluation and ranking
- Trader intelligence (early buyers, institutional traders)

**RESEARCH**:
- Financial statement analysis
- SEC filings (10-K, 10-Q, 8-K)
- Company fundamentals
- Analyst estimates

**BACKTESTING**:
- Historical strategy performance
- Parameter optimization
- Multiple timeframes

**CONVERSATION**:
- Search conversation history
- Export conversations
- Conversation analytics
- Multi-user support

**SYSTEM**:
- System status
- Configuration
- Risk settings

User Query: "{query}"

Your task:
1. If the query matches any command category above, respond with: EXECUTE: [category] [original_query]
2. If the query is asking what you can do or for help, provide a helpful response listing capabilities
3. If the query doesn't match any command but is trading/research related, provide helpful guidance

Be flexible in interpretation - understand synonyms, paraphrasing, and different phrasings. For example:
- "What's my profit on BTC?" = EXECUTE: position_management Check PnL for BTCUSDT
- "Sell my Bitcoin position gradually" = EXECUTE: position_management Close BTCUSDT in chunks
- "I want to get into Ethereum at a good price" = EXECUTE: position_management Easy entry for ETHUSDT
- "Show me the biggest holders" = EXECUTE: whale_tracking Track whale positions
- "How much Bitcoin should I buy?" = EXECUTE: position_sizing Calculate position size
- "Quick buy BTCUSDT for $10" = EXECUTE: quick_buysell Quick buy BTCUSDT for $10
- "Instant sell ETHUSDT" = EXECUTE: quick_buysell Quick sell ETHUSDT
- "Start monitoring token file" = EXECUTE: quick_buysell Start quick buy/sell monitor

Respond naturally and helpfully."""
            
            import asyncio
            loop = asyncio.get_event_loop()
            llm_response = await loop.run_in_executor(
                None,
                lambda: call_llm(
                    prompt=query,
                    system_prompt=command_context.format(query=query),
                    model="gpt-4o-mini"
                )
            )
            
            response_text = str(llm_response.content) if hasattr(llm_response, 'content') else str(llm_response)
            
            # Check if LLM wants to execute a command
            if "EXECUTE:" in response_text.upper():
                # Parse command execution
                parts = response_text.split("EXECUTE:")[1].strip().split("\n")[0].strip()
                command_parts = parts.split(" ", 1)
                category = command_parts[0] if command_parts else None
                enhanced_query = command_parts[1] if len(command_parts) > 1 else query
                
                # Re-classify and route
                if category:
                    # Map to handler
                    if category == "position_management":
                        return await self.handle_position_management(enhanced_query)
                    elif category == "ma_reversal":
                        return await self.handle_ma_reversal(enhanced_query)
                    elif category == "enhanced_sd":
                        return await self.handle_enhanced_sd(enhanced_query)
                    elif category == "quick_buysell":
                        return await self.handle_quick_buysell(enhanced_query)
                    elif category == "whale_tracking":
                        return await self.handle_whale_tracking(enhanced_query)
                    elif category == "position_sizing":
                        return await self.handle_position_sizing(enhanced_query)
                    elif category == "ml_indicators":
                        return await self.handle_ml_indicators(enhanced_query)
                    elif category == "rrs_analysis":
                        return await self.handle_rrs_analysis(enhanced_query)
                    elif category == "trader_intelligence":
                        return await self.handle_trader_intelligence(enhanced_query)
                    elif category == "liquidation_hunter":
                        return await self.handle_liquidation_hunter(enhanced_query)
                    elif category == "orderbook_analysis":
                        return await self.handle_orderbook_analysis(enhanced_query)
                    elif category == "market_dashboard":
                        return await self.handle_market_dashboard(enhanced_query)
                    elif category == "conversation":
                        return await self.handle_conversation_commands(enhanced_query)
                    elif category == "system":
                        return await self.handle_system_commands(enhanced_query)
                    elif category == "research":
                        return await self.research(enhanced_query)
                    elif category == "trading":
                        return await self.trade(enhanced_query)
                    elif category == "backtest":
                        return await self.backtest(enhanced_query)
                    elif category == "hybrid":
                        return await self.research_and_trade(enhanced_query)
                
                # If category not recognized, try re-running with enhanced query
                return await self.run(enhanced_query)
            
            # Otherwise return LLM's helpful response
            return response_text
            
        except Exception as e:
            self.logger._log(f"LLM assistance failed: {e}")
            # Fallback to static help
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

🐋 **Whale Tracking** (Day 44 + Day 46 Enhanced):
- Track large positions and whales
- Monitor institutional traders
- Analyze position movements
- Identify top holders
- **Day 46: Multi-address tracking**
- **Day 46: Liquidation risk analysis (3% threshold)**
- **Day 46: Position aggregation by coin**
- **Day 46: CSV export and reporting**

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
- "Track multiple whale addresses" (Day 46)
- "Show liquidation risk analysis" (Day 46)
- "Aggregate positions by coin" (Day 46)
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
- "Close BTCUSDT position based on PnL" (Day 48)
- "Close ETHUSDT in chunks" (Day 48)
- "Easy entry for SOLUSDT" (Day 48)
- "Track BTCUSDT position entry $50000 quantity 0.1" (Day 48)
- "Check PnL for ETHUSDT" (Day 48)
- "Run MA reversal strategy for BTC" (Day 49)
- "Execute enhanced supply/demand strategy for ETH" (Day 50)
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