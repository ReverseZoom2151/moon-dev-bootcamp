"""
Enhanced Strategy Engine (Phase 1)
===================================
Enhanced strategy manager with lifecycle management and performance tracking.
Based on autonomous_trading_system/services/strategy_engine.py
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .event_bus import EventBus
from .strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy execution status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StrategySignal:
    """Trading signal from strategy."""
    action: str  # 'BUY', 'SELL', 'HOLD'
    symbol: str
    size: float
    confidence: float  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyInstance:
    """Strategy instance with configuration and metrics."""
    strategy: BaseStrategy
    enabled: bool = True
    status: StrategyStatus = StrategyStatus.STOPPED
    last_signal: Optional[StrategySignal] = None
    last_execution: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=lambda: {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0
    })
    error_count: int = 0
    last_error: Optional[str] = None


class EnhancedStrategyEngine:
    """
    Enhanced strategy engine with lifecycle management and performance tracking.
    
    Features:
    - Strategy lifecycle management (start/stop/pause/resume)
    - Performance metrics tracking
    - Strategy status monitoring
    - Error handling and recovery
    - Signal generation and validation
    """
    
    def __init__(self, event_bus: EventBus, config: Optional[Dict] = None):
        """
        Initialize enhanced strategy engine.
        
        Args:
            event_bus: Event bus for communication
            config: Configuration dictionary
        """
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Strategy instances with enhanced tracking
        self.strategy_instances: Dict[str, StrategyInstance] = {}
        
        # Execution management
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
        # Performance tracking
        self.total_signals_generated = 0
        self.total_trades_executed = 0
        
        logger.info("Enhanced Strategy Engine initialized")
    
    def register_strategy(self, name: str, strategy: BaseStrategy, enabled: bool = True):
        """
        Register a strategy with enhanced tracking.
        
        Args:
            name: Strategy name
            strategy: Strategy instance
            enabled: Whether strategy is enabled by default
        """
        if not isinstance(strategy, BaseStrategy):
            raise ValueError(f"Strategy must inherit from BaseStrategy")
        
        instance = StrategyInstance(
            strategy=strategy,
            enabled=enabled,
            status=StrategyStatus.STOPPED
        )
        
        self.strategy_instances[name] = instance
        self.logger.info(f"Strategy registered: {name} (enabled={enabled})")
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get strategy instance."""
        if name in self.strategy_instances:
            return self.strategy_instances[name].strategy
        return None
    
    def get_all_strategies(self) -> Dict[str, BaseStrategy]:
        """Get all registered strategies."""
        return {name: instance.strategy for name, instance in self.strategy_instances.items()}
    
    def get_strategy_status(self, name: str) -> str:
        """Get strategy status."""
        if name in self.strategy_instances:
            return self.strategy_instances[name].status.value
        return "not_found"
    
    def get_strategy_metrics(self, name: str) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        if name in self.strategy_instances:
            instance = self.strategy_instances[name]
            return {
                **instance.performance_metrics,
                "status": instance.status.value,
                "enabled": instance.enabled,
                "error_count": instance.error_count,
                "last_error": instance.last_error,
                "last_execution": instance.last_execution.isoformat() if instance.last_execution else None
            }
        return {}
    
    async def start_strategy(self, name: str, config: Optional[Dict] = None):
        """Start a strategy."""
        if name not in self.strategy_instances:
            raise ValueError(f"Strategy {name} not registered")
        
        instance = self.strategy_instances[name]
        
        if instance.status == StrategyStatus.RUNNING:
            self.logger.warning(f"Strategy {name} already running")
            return
        
        try:
            instance.status = StrategyStatus.STARTING
            
            # Initialize strategy
            strategy_config = config or instance.strategy.config
            await instance.strategy.initialize(strategy_config)
            
            # Create execution task
            task = asyncio.create_task(self._run_strategy(name))
            self.execution_tasks[name] = task
            
            instance.status = StrategyStatus.RUNNING
            instance.enabled = True
            
            self.logger.info(f"Strategy started: {name}")
            
            await self.event_bus.emit("strategy_started", {
                "strategy": name,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            instance.status = StrategyStatus.ERROR
            instance.last_error = str(e)
            instance.error_count += 1
            self.logger.error(f"Failed to start strategy {name}: {e}")
            raise
    
    async def stop_strategy(self, name: str):
        """Stop a strategy."""
        if name not in self.strategy_instances:
            raise ValueError(f"Strategy {name} not registered")
        
        instance = self.strategy_instances[name]
        
        if instance.status == StrategyStatus.STOPPED:
            self.logger.warning(f"Strategy {name} already stopped")
            return
        
        try:
            instance.status = StrategyStatus.STOPPING
            
            # Cancel execution task
            if name in self.execution_tasks:
                self.execution_tasks[name].cancel()
                try:
                    await self.execution_tasks[name]
                except asyncio.CancelledError:
                    pass
                del self.execution_tasks[name]
            
            # Cleanup strategy
            await instance.strategy.cleanup()
            
            instance.status = StrategyStatus.STOPPED
            instance.enabled = False
            
            self.logger.info(f"Strategy stopped: {name}")
            
            await self.event_bus.emit("strategy_stopped", {
                "strategy": name,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            instance.status = StrategyStatus.ERROR
            instance.last_error = str(e)
            self.logger.error(f"Failed to stop strategy {name}: {e}")
            raise
    
    async def pause_strategy(self, name: str):
        """Pause a strategy."""
        if name in self.strategy_instances:
            instance = self.strategy_instances[name]
            await instance.strategy.pause()
            instance.status = StrategyStatus.PAUSED
            self.logger.info(f"Strategy paused: {name}")
    
    async def resume_strategy(self, name: str):
        """Resume a strategy."""
        if name in self.strategy_instances:
            instance = self.strategy_instances[name]
            await instance.strategy.resume()
            if instance.status == StrategyStatus.PAUSED:
                instance.status = StrategyStatus.RUNNING
            self.logger.info(f"Strategy resumed: {name}")
    
    async def _run_strategy(self, name: str):
        """Run strategy execution loop."""
        instance = self.strategy_instances[name]
        strategy = instance.strategy
        
        while instance.status == StrategyStatus.RUNNING and instance.enabled:
            try:
                if not strategy.is_paused:
                    # Execute strategy
                    signal_dict = await strategy.execute()
                    
                    if signal_dict:
                        # Create signal object
                        signal = StrategySignal(
                            action=signal_dict.get('action', 'HOLD'),
                            symbol=signal_dict.get('symbol', ''),
                            size=signal_dict.get('size', 0.0),
                            confidence=signal_dict.get('confidence', 0.5),
                            metadata=signal_dict.get('metadata', {}),
                            timestamp=datetime.now()
                        )
                        
                        # Update instance
                        instance.last_signal = signal
                        instance.last_execution = datetime.now()
                        
                        # Update metrics
                        if signal.action in ['BUY', 'SELL']:
                            self.total_signals_generated += 1
                        
                        # Emit signal
                        await self.event_bus.emit("strategy_signal", {
                            "strategy": name,
                            "signal": {
                                "action": signal.action,
                                "symbol": signal.symbol,
                                "size": signal.size,
                                "confidence": signal.confidence,
                                "metadata": signal.metadata
                            },
                            "timestamp": signal.timestamp.isoformat()
                        })
                
                # Wait for next execution
                await asyncio.sleep(strategy.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                instance.error_count += 1
                instance.last_error = str(e)
                self.logger.error(f"Error in strategy {name}: {e}")
                await asyncio.sleep(10)  # Brief pause on error
    
    def update_strategy_metrics(self, name: str, trade_result: Dict[str, Any]):
        """
        Update strategy performance metrics after a trade.
        
        Args:
            name: Strategy name
            trade_result: Trade result with 'pnl', 'success', etc.
        """
        if name not in self.strategy_instances:
            return
        
        instance = self.strategy_instances[name]
        metrics = instance.performance_metrics
        
        metrics["total_trades"] += 1
        self.total_trades_executed += 1
        
        if trade_result.get('success', False):
            pnl = trade_result.get('pnl', 0.0)
            pnl_pct = trade_result.get('pnl_pct', 0.0)
            
            if pnl > 0:
                metrics["winning_trades"] += 1
                metrics["avg_win"] = (
                    (metrics["avg_win"] * (metrics["winning_trades"] - 1) + pnl) / 
                    metrics["winning_trades"]
                )
            else:
                metrics["losing_trades"] += 1
                metrics["avg_loss"] = (
                    (metrics["avg_loss"] * (metrics["losing_trades"] - 1) + pnl) / 
                    metrics["losing_trades"]
                )
            
            metrics["total_pnl"] += pnl
            metrics["total_pnl_pct"] += pnl_pct
            
            # Update win rate
            if metrics["total_trades"] > 0:
                metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
            
            # Update max drawdown
            if pnl_pct < metrics["max_drawdown"]:
                metrics["max_drawdown"] = pnl_pct

