"""
Strategy Engine - Core component that manages and executes all trading strategies
Consolidates strategies from all day projects into a unified execution engine
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from core.config import get_settings
from strategies.base_strategy import BaseStrategy, StrategySignal, StrategyStatus
from strategies.mean_reversion import MeanReversionStrategy
from strategies.bollinger_bands import BollingerBandsStrategy
from strategies.supply_demand import SupplyDemandStrategy
from strategies.vwap_strategy import VWAPStrategy
from strategies.stochrsi_strategy import StochRSIStrategy
from strategies.liquidation_strategy import LiquidationStrategy
from strategies.ml_lstm_strategy import LSTMStrategy
from strategies.rl_dqn_strategy import DQNStrategy
from strategies.arbitrage_strategy import ArbitrageStrategy
from strategies.sentiment_strategy import SentimentStrategy
from strategies.genetic_optimizer import GeneticOptimizerStrategy
from strategies.rrs_strategy import RRSStrategy
from strategies.indicator_optimizer import IndicatorOptimizerStrategy
from strategies.regime_detection import RegimeDetectionStrategy

logger = logging.getLogger(__name__)

class EngineStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class StrategyInstance:
    """Represents a strategy instance with its configuration"""
    strategy: BaseStrategy
    enabled: bool = True
    last_signal: Optional[StrategySignal] = None
    last_execution: Optional[datetime] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0
            }


class StrategyEngine:
    """
    Core strategy engine that manages all trading strategies
    """
    
    def __init__(self, market_data_manager, portfolio_manager, risk_manager, config):
        self.market_data_manager = market_data_manager
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.config = config
        self.settings = get_settings()
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.running = False
        self.main_task: Optional[asyncio.Task] = None
        self.strategy_tasks: Dict[str, asyncio.Task] = {}
        
        # Strategy class mapping
        self.strategy_classes = {
            "mean_reversion": MeanReversionStrategy,
            "bollinger_bands": BollingerBandsStrategy,
            "supply_demand": SupplyDemandStrategy,
            "vwap": VWAPStrategy,
            "stochrsi": StochRSIStrategy,
            "liquidation": LiquidationStrategy,
            "lstm_ml": LSTMStrategy,
            "dqn_rl": DQNStrategy,
            "arbitrage": ArbitrageStrategy,
            "sentiment": SentimentStrategy,
            "genetic_optimizer": GeneticOptimizerStrategy,
            "rrs": RRSStrategy,
            "indicator_optimizer": IndicatorOptimizerStrategy,
            "regime_detection": RegimeDetectionStrategy
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.total_signals_generated = 0
        self.total_trades_executed = 0
        
        logger.info("🎯 Strategy Engine initialized")
    
    async def _initialize_strategies(self):
        """Initialize all enabled strategies"""
        try:
            strategy_configs = self.config.get_strategy_configs()
            
            for strategy_name, strategy_config in strategy_configs.items():
                if strategy_name in self.strategy_classes:
                    try:
                        strategy_class = self.strategy_classes[strategy_name]
                        strategy = strategy_class(
                            config=strategy_config,
                            market_data_manager=self.market_data_manager,
                            name=strategy_name
                        )
                        
                        await strategy.initialize()
                        self.strategies[strategy_name] = strategy
                        
                        logger.info(f"✅ Initialized {strategy_name} strategy")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize {strategy_name}: {e}")
                        continue
                else:
                    logger.warning(f"⚠️ Unknown strategy type: {strategy_name}")
            
            logger.info(f"🎯 Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategies: {e}")
            raise
    
    async def start(self):
        """Start the strategy engine"""
        if self.running:
            logger.warning("Strategy engine is already running")
            return
        
        logger.info("🚀 Starting Strategy Engine...")
        self.running = True
        
        try:
            # Initialize strategies first
            await self._initialize_strategies()
            
            # Start market data manager
            await self.market_data_manager.start()
            
            # Start enabled strategies
            for name, strategy in self.strategies.items():
                await self._start_strategy(name, strategy)
            
            # Start main engine loop
            self.main_task = asyncio.create_task(self._main_loop())
            
            logger.info("✅ Strategy Engine started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Strategy Engine: {e}")
            self.running = False
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the strategy engine"""
        if not self.running:
            return
        
        logger.info("🛑 Stopping Strategy Engine...")
        self.running = False
        
        # Cancel main task
        if self.main_task and not self.main_task.done():
            self.main_task.cancel()
            try:
                await self.main_task
            except asyncio.CancelledError:
                pass
        
        # Stop all strategy tasks
        for name, task in self.strategy_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop market data manager
        await self.market_data_manager.stop()
        
        self.strategies.clear()
        logger.info("✅ Strategy Engine stopped")
    
    async def _main_loop(self):
        """Main engine loop"""
        logger.info("🔄 Strategy Engine main loop started")
        
        while self.running:
            try:
                # Process strategy signals
                await self._process_strategy_signals()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("🔄 Strategy Engine main loop stopped")
    
    async def _start_strategy(self, name: str, strategy: BaseStrategy):
        """Start a specific strategy"""
        try:
            await strategy.initialize()
            task = asyncio.create_task(self._strategy_loop(name, strategy))
            self.strategy_tasks[name] = task
            logger.info(f"✅ Started strategy: {name}")
        except Exception as e:
            logger.error(f"❌ Failed to start strategy {name}: {e}")
            strategy.status = StrategyStatus.DISABLED
    
    async def _strategy_loop(self, name: str, strategy: BaseStrategy):
        """Individual strategy execution loop"""
        logger.info(f"🔄 Starting strategy loop for: {name}")
        
        while strategy.status == StrategyStatus.ACTIVE and self.running:
            try:
                # Generate signal
                signal = await strategy.generate_signal()
                
                if signal and signal.action != "HOLD":
                    signal.strategy_name = name
                    self.total_signals_generated += 1
                    
                    logger.info(f"📊 Signal from {name}: {signal.action} {signal.symbol} @ {signal.price}")
                    
                    # Execute signal through portfolio manager
                    if await self._should_execute_signal(signal):
                        await self._execute_signal(signal)
                
                # Wait for next iteration
                await asyncio.sleep(getattr(strategy, 'execution_interval', 30))
                
            except asyncio.CancelledError:
                logger.info(f"🔄 Strategy loop cancelled for: {name}")
                break
            except Exception as e:
                logger.error(f"❌ Error in strategy loop {name}: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retrying
        
        logger.info(f"🔄 Strategy loop stopped for: {name}")
        strategy.status = StrategyStatus.DISABLED
    
    async def _should_execute_signal(self, signal: StrategySignal) -> bool:
        """Determine if a signal should be executed"""
        # Check if running
        if not self.running:
            return False
        
        # Check risk limits
        if not await self.risk_manager.check_position_risk(signal.symbol, signal.price):
            logger.warning(f"⚠️ Risk check failed for {signal.symbol}")
            return False
        
        # Check confidence threshold
        if signal.confidence < 0.5:  # Minimum confidence threshold
            logger.debug(f"⚠️ Signal confidence too low: {signal.confidence}")
            return False
        
        return True
    
    async def _execute_signal(self, signal: StrategySignal):
        """Execute a trading signal"""
        try:
            logger.info(f"🚀 Executing signal: {signal.action} {signal.symbol} @ {signal.price}")
            
            # Execute through portfolio manager
            execution_result = await self.portfolio_manager.execute_trade(signal)
            
            if execution_result:
                self.total_trades_executed += 1
                
                # Update strategy performance metrics
                strategy_name = getattr(signal, 'strategy_name', 'unknown')
                if strategy_name not in self.performance_metrics:
                    self.performance_metrics[strategy_name] = {
                        "total_trades": 0,
                        "winning_trades": 0,
                        "total_pnl": 0,
                        "win_rate": 0
                    }
                
                self.performance_metrics[strategy_name]["total_trades"] += 1
                
                logger.info(f"✅ Executed trade successfully")
            else:
                logger.warning(f"❌ Trade execution failed")
                
        except Exception as e:
            logger.error(f"❌ Error executing signal: {e}", exc_info=True)
    
    async def _process_strategy_signals(self):
        """Process and coordinate signals from multiple strategies"""
        # This could include signal filtering, conflict resolution, etc.
        pass
    
    async def _update_performance_metrics(self):
        """Update performance metrics for all strategies"""
        for name, strategy in self.strategies.items():
            if strategy.status == StrategyStatus.ACTIVE:
                # Update metrics from portfolio manager
                strategy_trades = await self.portfolio_manager.get_strategy_trades(name)
                if strategy_trades:
                    # Calculate performance metrics
                    winning_trades = sum(1 for trade in strategy_trades if trade.pnl > 0)
                    total_pnl = sum(trade.pnl for trade in strategy_trades)
                    
                    self.performance_metrics[name] = {
                        "total_trades": len(strategy_trades),
                        "winning_trades": winning_trades,
                        "total_pnl": total_pnl,
                        "win_rate": winning_trades / len(strategy_trades) if strategy_trades else 0
                    }
    
    # Public API methods
    
    async def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a specific strategy"""
        if strategy_name not in self.strategies:
            return False
        
        strategy = self.strategies[strategy_name]
        if strategy.status == StrategyStatus.DISABLED:
            strategy.status = StrategyStatus.ACTIVE
            if self.running:
                await self._start_strategy(strategy_name, strategy)
        
        return True
    
    async def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a specific strategy"""
        if strategy_name not in self.strategies:
            return False
        
        strategy = self.strategies[strategy_name]
        strategy.status = StrategyStatus.DISABLED
        
        # Cancel strategy task
        if strategy_name in self.strategy_tasks:
            task = self.strategy_tasks[strategy_name]
            if not task.done():
                task.cancel()
        
        return True
    
    async def get_strategy_status(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific strategy"""
        if strategy_name not in self.strategies:
            return None
        
        strategy = self.strategies[strategy_name]
        return {
            "name": strategy_name,
            "enabled": strategy.status == StrategyStatus.ACTIVE,
            "status": strategy.status.value,
            "last_signal": strategy.last_signal.dict() if strategy.last_signal else None,
            "last_execution": strategy.last_execution.isoformat() if strategy.last_execution else None,
            "performance": self.performance_metrics.get(strategy_name, {})
        }
    
    async def get_all_strategies_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all strategies"""
        result = {}
        for name in self.strategies:
            result[name] = await self.get_strategy_status(name)
        return result
    
    @property
    def is_running(self) -> bool:
        """Check if the engine is running"""
        return self.running
    
    @property
    def uptime(self) -> Optional[timedelta]:
        """Get engine uptime"""
        if self.start_time:
            return datetime.utcnow() - self.start_time
        return None
    
    @property
    def available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return list(self.strategies.keys()) 