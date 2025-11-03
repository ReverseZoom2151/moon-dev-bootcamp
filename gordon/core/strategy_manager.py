"""
Strategy Manager (Refactored)
==============================
Orchestrates trading strategies and manages their execution.
Enhanced with Phase 1 lifecycle management and performance tracking.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import importlib
import inspect

from .event_bus import EventBus
from .config import StrategyConfigs
from .caching import CacheManager
from .strategies.base import BaseStrategy

# Import enhanced strategy engine
from .strategy_engine import EnhancedStrategyEngine, StrategyInstance, StrategyStatus

# Import RL components
try:
    from .rl import RLManager
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    RLManager = None


class StrategyManager:
    """
    Manages and orchestrates trading strategies.
    Refactored to focus only on orchestration, with strategies in separate modules.
    Enhanced with Phase 1 lifecycle management.
    """

    def __init__(self, event_bus: EventBus, config: Optional[Dict] = None):
        """
        Initialize strategy manager.

        Args:
            event_bus: Event bus for communication
            config: Configuration dictionary
        """
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config = config or {}
        self.configs = StrategyConfigs(self.config.get('config_file'))

        # Enhanced strategy engine (Phase 1)
        self.enhanced_engine = EnhancedStrategyEngine(event_bus, config)

        # Legacy strategy registry (for backward compatibility)
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[str] = []

        # Execution management
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False

        # Cache manager
        self.cache = CacheManager(max_size=1000, default_ttl=300)

        # Exchange connections
        self.exchange_connections = {}
        
        # Initialize RL manager if available
        if RL_AVAILABLE:
            try:
                self.rl_manager = RLManager(config=self.config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize RL Manager: {e}")
                self.rl_manager = None
        else:
            self.rl_manager = None

        # Register built-in strategies
        self._register_built_in_strategies()

        # Subscribe to events
        self._setup_event_handlers()
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all registered strategies (from enhanced engine)."""
        return self.enhanced_engine.get_all_strategies()
    
    def get_strategy_status(self, name: str) -> str:
        """Get strategy status (from enhanced engine)."""
        return self.enhanced_engine.get_strategy_status(name)
    
    def get_strategy_metrics(self, name: str) -> Dict[str, Any]:
        """Get strategy performance metrics (from enhanced engine)."""
        return self.enhanced_engine.get_strategy_metrics(name)

    def _register_built_in_strategies(self):
        """Register all built-in strategies from the strategies folder."""
        strategy_modules = [
            'day6_sma_strategy',
            'day7_rsi_strategy',
            'day8_vwap_strategy',
            'day9_vwma_strategy',
            'day10_bollinger_strategy',
            'day10_volume_strategy',
            'day11_breakout_strategy',
            'day11_supply_demand_strategy',
            'day12_engulfing_strategy',
            'day12_vwap_probability_strategy'
        ]

        for module_name in strategy_modules:
            try:
                # Import the strategy module
                module = importlib.import_module(f'.strategies.{module_name}', package='gordon.core')

                # Find the strategy class (looks for class inheriting from BaseStrategy)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseStrategy) and obj != BaseStrategy:
                        # Create instance and register
                        strategy_instance = obj()
                        strategy_name = module_name.replace('_strategy', '').replace('day', 'd')
                        # Register in both enhanced engine and legacy registry
                        self.register_strategy(strategy_name, strategy_instance)
                        self.logger.info(f"Registered strategy: {strategy_name} ({name})")
                        break

            except Exception as e:
                self.logger.error(f"Failed to load strategy {module_name}: {e}")

    def _setup_event_handlers(self):
        """Setup event handlers for strategy management."""
        self.event_bus.on("market_update", self._handle_market_update)
        self.event_bus.on("position_update", self._handle_position_update)
        self.event_bus.on("risk_alert", self._handle_risk_alert)
        self.event_bus.on("strategy_signal", self._handle_strategy_signal)

    async def _handle_market_update(self, event: Dict):
        """Handle market update events."""
        # Forward to all active strategies
        for strategy_name in self.active_strategies:
            if strategy_name in self.strategies:
                try:
                    await self.strategies[strategy_name].on_market_update(event)
                except Exception as e:
                    self.logger.error(f"Error in {strategy_name}.on_market_update: {e}")

    async def _handle_position_update(self, event: Dict):
        """Handle position update events."""
        # Forward to all active strategies
        for strategy_name in self.active_strategies:
            if strategy_name in self.strategies:
                try:
                    strategy = self.strategies[strategy_name]
                    if hasattr(strategy, 'on_position_update'):
                        await strategy.on_position_update(event)
                except Exception as e:
                    self.logger.error(f"Error in {strategy_name}.on_position_update: {e}")

    async def _handle_risk_alert(self, event: Dict):
        """Handle risk alert events."""
        self.logger.warning(f"Risk alert received: {event}")

        # Pause strategies if critical risk
        if event.get('severity') == 'critical':
            await self.pause_all_strategies()

    async def _handle_strategy_signal(self, signal: Dict):
        """Handle trading signals from strategies."""
        self.logger.info(f"Strategy signal received: {signal}")

        # Emit signal for execution
        await self.event_bus.emit("execute_signal", signal)

    def register_strategy(self, name: str, strategy: BaseStrategy):
        """
        Register a strategy.
        Delegates to enhanced engine for better tracking.

        Args:
            name: Strategy name
            strategy: Strategy instance
        """
        # Register in enhanced engine
        self.enhanced_engine.register_strategy(name, strategy, enabled=True)
        
        # Also register in legacy registry for backward compatibility
        self.strategies[name] = strategy
        self.logger.info(f"Strategy registered: {name}")

    def unregister_strategy(self, name: str):
        """
        Unregister a strategy.

        Args:
            name: Strategy name
        """
        if name in self.strategies:
            # Stop if running
            if name in self.active_strategies:
                asyncio.create_task(self.stop_strategy(name))

            del self.strategies[name]
            self.logger.info(f"Strategy unregistered: {name}")

    async def start_strategy(self, name: str, config: Optional[Dict] = None):
        """
        Start a strategy.
        Delegates to enhanced engine.

        Args:
            name: Strategy name
            config: Optional configuration override
        """
        # Use enhanced engine
        await self.enhanced_engine.start_strategy(name, config)
        
        # Also update legacy registry
        if name not in self.active_strategies:
            self.active_strategies.append(name)

    async def stop_strategy(self, name: str):
        """
        Stop a strategy.
        Delegates to enhanced engine.

        Args:
            name: Strategy name
        """
        # Use enhanced engine
        await self.enhanced_engine.stop_strategy(name)
        
        # Also update legacy registry
        if name in self.active_strategies:
            self.active_strategies.remove(name)

    async def _run_strategy(self, name: str):
        """
        Run strategy execution loop.

        Args:
            name: Strategy name
        """
        strategy = self.strategies[name]

        while name in self.active_strategies:
            try:
                if not strategy.is_paused:
                    # Execute strategy
                    signal = await strategy.execute()

                    if signal and strategy.validate_signal(signal):
                        # Add metadata
                        signal['strategy'] = name
                        signal['timestamp'] = datetime.now().isoformat()

                        # Emit signal
                        await self._handle_strategy_signal(signal)

                # Wait for next execution
                await asyncio.sleep(strategy.interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in strategy {name}: {e}")
                await asyncio.sleep(10)  # Brief pause on error

    async def pause_strategy(self, name: str):
        """
        Pause a strategy.

        Args:
            name: Strategy name
        """
        if name in self.strategies:
            await self.strategies[name].pause()

    async def resume_strategy(self, name: str):
        """
        Resume a strategy.

        Args:
            name: Strategy name
        """
        if name in self.strategies:
            await self.strategies[name].resume()

    async def pause_all_strategies(self):
        """Pause all active strategies."""
        for name in self.active_strategies:
            await self.pause_strategy(name)

    async def resume_all_strategies(self):
        """Resume all active strategies."""
        for name in self.active_strategies:
            await self.resume_strategy(name)

    async def start_all_strategies(self):
        """Start all registered strategies."""
        for name in list(self.strategies.keys()):
            if name not in self.active_strategies:
                await self.start_strategy(name)

    async def stop_all_strategies(self):
        """Stop all active strategies."""
        for name in list(self.active_strategies):
            await self.stop_strategy(name)

    def set_exchange_connection(self, exchange: str, connection: Any):
        """
        Set exchange connection.

        Args:
            exchange: Exchange name
            connection: Exchange connection object
        """
        self.exchange_connections[exchange] = connection

        # Update all strategies
        for strategy in self.strategies.values():
            strategy.set_exchange_connection(exchange, connection)

    def get_strategy_status(self, name: str) -> Optional[Dict]:
        """
        Get strategy status.

        Args:
            name: Strategy name

        Returns:
            Strategy status or None
        """
        if name not in self.strategies:
            return None

        strategy = self.strategies[name]
        status = strategy.get_status()
        status['active'] = name in self.active_strategies
        status['has_task'] = name in self.execution_tasks

        return status

    def get_all_statuses(self) -> Dict[str, Dict]:
        """
        Get status of all strategies.

        Returns:
            Dictionary of strategy statuses
        """
        statuses = {}
        for name in self.strategies:
            statuses[name] = self.get_strategy_status(name)

        return statuses
    
    def get_strategy(self, name: Optional[str] = None) -> Optional[BaseStrategy]:
        """
        Get a strategy by name.
        If no name provided and RL is enabled, use RL to select best strategy.

        Args:
            name: Strategy name (optional if RL enabled)

        Returns:
            Strategy instance or None
        """
        if name:
            return self.strategies.get(name)
        
        # If RL enabled and no name provided, use RL to select
        if self.rl_manager and self.rl_manager.enabled:
            # Get market context (simplified - should be enhanced)
            market_context = self._get_market_context()
            selection = self.rl_manager.select_strategy(market_context)
            if selection:
                selected_name = selection.get('strategy')
                return self.strategies.get(selected_name)
        
        return None
    
    def _get_market_context(self) -> Dict[str, Any]:
        """Get market context for RL components."""
        # Simplified - should be enhanced with actual market data
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
    
    async def aggregate_signals(self, signals: Dict[str, Dict], market_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Aggregate signals from multiple strategies using RL if available.
        
        Args:
            signals: Dictionary mapping strategy names to signals
            market_context: Market context (optional)
            
        Returns:
            Aggregated signal
        """
        if self.rl_manager and self.rl_manager.enabled and self.rl_manager.signal_aggregator:
            if market_context is None:
                market_context = self._get_market_context()
            return self.rl_manager.aggregate_signals(signals, market_context) or {}
        
        # Fallback: simple aggregation
        buy_signals = [s for s in signals.values() if s.get('action') == 'BUY']
        sell_signals = [s for s in signals.values() if s.get('action') == 'SELL']
        
        if buy_signals:
            avg_confidence = sum(s.get('confidence', 0.5) for s in buy_signals) / len(buy_signals)
            return {'action': 'BUY', 'confidence': avg_confidence}
        elif sell_signals:
            avg_confidence = sum(s.get('confidence', 0.5) for s in sell_signals) / len(sell_signals)
            return {'action': 'SELL', 'confidence': avg_confidence}
        else:
            return {'action': 'HOLD', 'confidence': 0.5}

    async def start(self):
        """Start the strategy manager."""
        self.is_running = True
        self.logger.info("Strategy Manager started")

        # Start configured strategies
        auto_start = self.config.get('auto_start_strategies', [])
        for strategy_name in auto_start:
            if strategy_name in self.strategies:
                await self.start_strategy(strategy_name)

    async def stop(self):
        """Stop the strategy manager."""
        self.is_running = False

        # Stop all strategies
        await self.stop_all_strategies()

        # Clear cache
        self.cache.clear()

        self.logger.info("Strategy Manager stopped")

    def __repr__(self):
        """String representation."""
        return (f"StrategyManager(strategies={len(self.strategies)}, "
                f"active={len(self.active_strategies)})")