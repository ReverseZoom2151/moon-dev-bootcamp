"""
Main Exchange Orchestrator
==========================
Central coordinator for all trading operations across multiple exchanges.
Consolidates functionality from Days 2-56 of the ATC Bootcamp.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
from pathlib import Path

from .event_bus import EventBus
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .strategy_manager import StrategyManager
from ..exchanges.factory import ExchangeFactory
from ..utilities.logger import setup_logger
from ..config.config_manager import ConfigManager


class ExchangeOrchestrator:
    """
    Main orchestrator that coordinates all trading activities across exchanges.

    Features:
    - Multi-exchange support (Binance, Bitfinex, HyperLiquid, etc.)
    - Strategy management and execution
    - Risk management and position tracking
    - Event-driven architecture
    - Unified logging and monitoring
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Exchange Orchestrator."""
        self.logger = setup_logger("ExchangeOrchestrator")
        self.logger.info("Initializing Exchange Orchestrator...")

        # Core components
        self.config_manager = ConfigManager(config_path)
        self.event_bus = EventBus()
        self.exchanges: Dict[str, Any] = {}
        self.active_strategies: Dict[str, Any] = {}

        # Managers
        self.position_manager = PositionManager(self.event_bus)
        self.risk_manager = RiskManager(self.event_bus, self.config_manager)
        self.strategy_manager = StrategyManager(self.event_bus)

        # State tracking
        self.is_running = False
        self.start_time = None
        self.metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_pnl": 0.0,
            "active_positions": 0
        }

        # Register event handlers
        self._register_event_handlers()

        self.logger.info("Exchange Orchestrator initialized successfully")

    def _register_event_handlers(self):
        """Register internal event handlers."""
        self.event_bus.subscribe("trade_executed", self._on_trade_executed)
        self.event_bus.subscribe("position_opened", self._on_position_opened)
        self.event_bus.subscribe("position_closed", self._on_position_closed)
        self.event_bus.subscribe("strategy_signal", self._on_strategy_signal)
        self.event_bus.subscribe("risk_alert", self._on_risk_alert)

    async def initialize_exchange(self, exchange_name: str, credentials: Optional[Dict] = None):
        """
        Initialize a specific exchange connection.

        Args:
            exchange_name: Name of the exchange (binance, bitfinex, hyperliquid, etc.)
            credentials: Optional credentials dict (if not in config)
        """
        try:
            self.logger.info(f"Initializing {exchange_name} exchange...")

            # Get credentials from config if not provided
            if credentials is None:
                credentials = self.config_manager.get_exchange_credentials(exchange_name)

            # Create exchange instance using factory
            exchange = ExchangeFactory.create_exchange(
                exchange_name,
                credentials,
                self.event_bus
            )

            # Initialize the exchange
            await exchange.initialize()

            # Store the exchange instance
            self.exchanges[exchange_name] = exchange

            self.logger.info(f"{exchange_name} exchange initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize {exchange_name}: {str(e)}")
            return False

    async def initialize_all_exchanges(self):
        """Initialize all configured exchanges."""
        exchanges_config = self.config_manager.get_exchanges_config()

        tasks = []
        for exchange_name, config in exchanges_config.items():
            if config.get("enabled", False):
                tasks.append(self.initialize_exchange(exchange_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if r is True)
        self.logger.info(f"Initialized {successful}/{len(tasks)} exchanges")

        return successful > 0

    def register_strategy(self, strategy_name: str, strategy_instance: Any):
        """
        Register a trading strategy.

        Args:
            strategy_name: Unique name for the strategy
            strategy_instance: Instance of the strategy class
        """
        try:
            self.strategy_manager.register_strategy(strategy_name, strategy_instance)
            self.active_strategies[strategy_name] = strategy_instance
            self.logger.info(f"Registered strategy: {strategy_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register strategy {strategy_name}: {str(e)}")
            return False

    def load_strategy_from_day(self, day: int, strategy_name: str):
        """
        Load a strategy from a specific day's project.

        Args:
            day: Day number (2-56)
            strategy_name: Name of the strategy file
        """
        try:
            # Dynamic import of strategy from Day projects
            module_path = f"Day_{day}_Projects.{strategy_name}"
            self.logger.info(f"Loading strategy from {module_path}")

            # Import and instantiate the strategy
            strategy = self.strategy_manager.load_strategy_module(module_path)

            if strategy:
                self.register_strategy(f"day{day}_{strategy_name}", strategy)
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to load strategy from Day {day}: {str(e)}")
            return False

    async def start(self):
        """Start the orchestrator and all components."""
        if self.is_running:
            self.logger.warning("Orchestrator is already running")
            return

        self.logger.info("Starting Exchange Orchestrator...")
        self.start_time = datetime.now()
        self.is_running = True

        # Initialize all exchanges
        await self.initialize_all_exchanges()

        # Start managers
        await self.position_manager.start()
        await self.risk_manager.start()
        await self.strategy_manager.start()

        # Start event processing
        asyncio.create_task(self._process_events())

        # Start metrics collection
        asyncio.create_task(self._collect_metrics())

        self.logger.info("Exchange Orchestrator started successfully")

        # Emit start event
        await self.event_bus.emit("orchestrator_started", {
            "timestamp": datetime.now().isoformat(),
            "exchanges": list(self.exchanges.keys()),
            "strategies": list(self.active_strategies.keys())
        })

    async def stop(self):
        """Stop the orchestrator and all components."""
        if not self.is_running:
            self.logger.warning("Orchestrator is not running")
            return

        self.logger.info("Stopping Exchange Orchestrator...")
        self.is_running = False

        # Stop all strategies
        await self.strategy_manager.stop_all()

        # Close all positions if configured
        if self.config_manager.get("close_positions_on_stop", False):
            await self.position_manager.close_all_positions()

        # Stop managers
        await self.position_manager.stop()
        await self.risk_manager.stop()
        await self.strategy_manager.stop()

        # Disconnect from exchanges
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.disconnect()
                self.logger.info(f"Disconnected from {exchange_name}")
            except Exception as e:
                self.logger.error(f"Error disconnecting from {exchange_name}: {str(e)}")

        # Save final metrics
        self._save_metrics()

        self.logger.info("Exchange Orchestrator stopped")

        # Emit stop event
        await self.event_bus.emit("orchestrator_stopped", {
            "timestamp": datetime.now().isoformat(),
            "runtime": str(datetime.now() - self.start_time) if self.start_time else "0",
            "metrics": self.metrics
        })

    async def execute_trade(self, exchange: str, symbol: str, side: str,
                          amount: float, order_type: str = "market",
                          price: Optional[float] = None, **kwargs):
        """
        Execute a trade on a specific exchange.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: Buy or sell
            amount: Trade amount
            order_type: Market, limit, etc.
            price: Price for limit orders
            **kwargs: Additional parameters
        """
        if exchange not in self.exchanges:
            self.logger.error(f"Exchange {exchange} not initialized")
            return None

        try:
            # Check risk limits
            if not await self.risk_manager.check_trade_allowed(
                exchange, symbol, side, amount, price
            ):
                self.logger.warning(f"Trade rejected by risk manager")
                return None

            # Execute the trade
            exchange_instance = self.exchanges[exchange]
            order = await exchange_instance.place_order(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price,
                **kwargs
            )

            # Track the trade
            if order:
                self.metrics["total_trades"] += 1
                await self.event_bus.emit("trade_executed", {
                    "exchange": exchange,
                    "order": order,
                    "timestamp": datetime.now().isoformat()
                })

            return order

        except Exception as e:
            self.logger.error(f"Failed to execute trade: {str(e)}")
            self.metrics["failed_trades"] += 1
            return None

    async def get_market_data(self, exchange: str, symbol: str, timeframe: str = "1h",
                            limit: int = 100):
        """Get market data from a specific exchange."""
        if exchange not in self.exchanges:
            self.logger.error(f"Exchange {exchange} not initialized")
            return None

        try:
            exchange_instance = self.exchanges[exchange]
            return await exchange_instance.get_ohlcv(symbol, timeframe, limit)
        except Exception as e:
            self.logger.error(f"Failed to get market data: {str(e)}")
            return None

    async def get_positions(self, exchange: Optional[str] = None):
        """Get current positions from one or all exchanges."""
        if exchange:
            if exchange not in self.exchanges:
                return []
            return await self.position_manager.get_positions(exchange)
        else:
            # Get positions from all exchanges
            all_positions = []
            for exchange_name in self.exchanges:
                positions = await self.position_manager.get_positions(exchange_name)
                all_positions.extend(positions)
            return all_positions

    async def get_balance(self, exchange: str, asset: Optional[str] = None):
        """Get balance from a specific exchange."""
        if exchange not in self.exchanges:
            self.logger.error(f"Exchange {exchange} not initialized")
            return None

        try:
            exchange_instance = self.exchanges[exchange]
            return await exchange_instance.get_balance(asset)
        except Exception as e:
            self.logger.error(f"Failed to get balance: {str(e)}")
            return None

    async def _process_events(self):
        """Background task to process events."""
        while self.is_running:
            try:
                # Process events from the event bus
                await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
            except Exception as e:
                self.logger.error(f"Error processing events: {str(e)}")

    async def _collect_metrics(self):
        """Background task to collect metrics."""
        while self.is_running:
            try:
                # Update metrics every minute
                await asyncio.sleep(60)

                # Collect position metrics
                positions = await self.get_positions()
                self.metrics["active_positions"] = len(positions)

                # Calculate total PnL
                total_pnl = sum(p.get("pnl", 0) for p in positions)
                self.metrics["total_pnl"] = total_pnl

                # Emit metrics event
                await self.event_bus.emit("metrics_updated", self.metrics)

            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")

    def _save_metrics(self):
        """Save metrics to file."""
        try:
            metrics_file = Path("gordon/logs/metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)

            with open(metrics_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "runtime": str(datetime.now() - self.start_time) if self.start_time else "0",
                    "metrics": self.metrics
                }, f, indent=2)

            self.logger.info(f"Metrics saved to {metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")

    # Event handlers
    async def _on_trade_executed(self, event_data: Dict):
        """Handle trade execution events."""
        self.metrics["successful_trades"] += 1
        self.logger.info(f"Trade executed: {event_data}")

    async def _on_position_opened(self, event_data: Dict):
        """Handle position opened events."""
        self.logger.info(f"Position opened: {event_data}")

    async def _on_position_closed(self, event_data: Dict):
        """Handle position closed events."""
        pnl = event_data.get("pnl", 0)
        self.metrics["total_pnl"] += pnl
        self.logger.info(f"Position closed with PnL: {pnl}")

    async def _on_strategy_signal(self, event_data: Dict):
        """Handle strategy signals."""
        strategy = event_data.get("strategy")
        signal = event_data.get("signal")
        self.logger.info(f"Strategy signal from {strategy}: {signal}")

        # Execute trade based on signal if auto-trading is enabled
        if self.config_manager.get("auto_trading", False):
            await self.execute_trade(**signal)

    async def _on_risk_alert(self, event_data: Dict):
        """Handle risk alerts."""
        alert_type = event_data.get("type")
        message = event_data.get("message")
        self.logger.warning(f"Risk alert ({alert_type}): {message}")

        # Take action based on alert type
        if alert_type == "max_drawdown":
            # Stop all trading
            await self.strategy_manager.stop_all()
        elif alert_type == "position_limit":
            # Pause new positions
            await self.strategy_manager.pause_all()

    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        return {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else "0",
            "exchanges": list(self.exchanges.keys()),
            "active_strategies": list(self.active_strategies.keys()),
            "metrics": self.metrics
        }