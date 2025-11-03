"""
Strategy Manager (Refactored)
==============================
Orchestrates trading strategies and manages their execution.
All strategy implementations have been moved to the strategies/ folder.
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


class StrategyManager:
    """
    Manages and orchestrates trading strategies.
    Refactored to focus only on orchestration, with strategies in separate modules.
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

        # Strategy registry
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: List[str] = []

        # Execution management
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False

        # Cache manager
        self.cache = CacheManager(max_size=1000, default_ttl=300)

        # Exchange connections
        self.exchange_connections = {}

        # Register built-in strategies
        self._register_built_in_strategies()

        # Subscribe to events
        self._setup_event_handlers()

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

        Args:
            name: Strategy name
            strategy: Strategy instance
        """
        if not isinstance(strategy, BaseStrategy):
            raise ValueError(f"Strategy must inherit from BaseStrategy")

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

        Args:
            name: Strategy name
            config: Optional configuration override
        """
        if name not in self.strategies:
            raise ValueError(f"Strategy {name} not registered")

        if name in self.active_strategies:
            self.logger.warning(f"Strategy {name} already running")
            return

        strategy = self.strategies[name]

        # Get configuration
        strategy_config = config or self.configs.get(name.replace('d', 'day'))

        # Initialize strategy
        await strategy.initialize(strategy_config)

        # Set exchange connections
        for exchange, connection in self.exchange_connections.items():
            strategy.set_exchange_connection(exchange, connection)

        # Create execution task
        task = asyncio.create_task(self._run_strategy(name))
        self.execution_tasks[name] = task

        # Add to active strategies
        self.active_strategies.append(name)

        self.logger.info(f"Strategy started: {name}")

        # Emit event
        await self.event_bus.emit("strategy_started", {
            "strategy": name,
            "config": strategy_config,
            "timestamp": datetime.now().isoformat()
        })

    async def stop_strategy(self, name: str):
        """
        Stop a strategy.

        Args:
            name: Strategy name
        """
        if name not in self.active_strategies:
            self.logger.warning(f"Strategy {name} not running")
            return

        # Cancel execution task
        if name in self.execution_tasks:
            self.execution_tasks[name].cancel()
            try:
                await self.execution_tasks[name]
            except asyncio.CancelledError:
                pass
            del self.execution_tasks[name]

        # Remove from active strategies
        self.active_strategies.remove(name)

        # Cleanup strategy
        strategy = self.strategies[name]
        await strategy.cleanup()

        self.logger.info(f"Strategy stopped: {name}")

        # Emit event
        await self.event_bus.emit("strategy_stopped", {
            "strategy": name,
            "timestamp": datetime.now().isoformat()
        })

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