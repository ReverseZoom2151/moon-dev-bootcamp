"""
RL Manager
==========
Manages all RL components and coordinates their interactions.
"""

import logging
from typing import Dict, Any, List, Optional

from .meta_strategy_selector import MetaStrategySelector
from .signal_aggregator import SignalAggregator
from .risk_optimizer import RiskOptimizer
from .position_sizer import PositionSizeOptimizer
from .regime_detector import MarketRegimeDetector
from .portfolio_allocator import PortfolioAllocator

logger = logging.getLogger(__name__)


class RLManager:
    """
    Central manager for all RL components.
    Coordinates interactions and provides unified interface.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RL manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Check if RL is enabled
        self.enabled = self.config.get('rl', {}).get('enabled', False)
        
        if not self.enabled:
            self.logger.info("RL components disabled in config")
            self.meta_selector = None
            self.signal_aggregator = None
            self.risk_optimizer = None
            self.position_sizer = None
            self.regime_detector = None
            self.portfolio_allocator = None
            return
        
        # Get available strategies
        available_strategies = self.config.get('rl', {}).get('available_strategies', [
            'SMA', 'RSI', 'VWAP', 'Bollinger', 'Breakout', 'MeanReversion', 'DQN', 'MarketMaker'
        ])
        
        # Initialize components
        rl_config = self.config.get('rl', {})
        
        # Meta-Strategy Selector
        meta_config = rl_config.get('meta_strategy_selection', {})
        if meta_config.get('enabled', False):
            self.meta_selector = MetaStrategySelector(
                available_strategies=available_strategies,
                config={**rl_config, **meta_config}
            )
        else:
            self.meta_selector = None
        
        # Signal Aggregator
        signal_config = rl_config.get('signal_aggregation', {})
        if signal_config.get('enabled', False):
            strategy_names = signal_config.get('strategies', available_strategies)
            self.signal_aggregator = SignalAggregator(
                strategy_names=strategy_names,
                config={**rl_config, **signal_config}
            )
        else:
            self.signal_aggregator = None
        
        # Risk Optimizer
        risk_config = rl_config.get('risk_optimization', {})
        if risk_config.get('enabled', False):
            self.risk_optimizer = RiskOptimizer(
                config={**rl_config, **risk_config}
            )
        else:
            self.risk_optimizer = None
        
        # Position Size Optimizer
        position_config = rl_config.get('position_sizing', {})
        if position_config.get('enabled', False):
            base_size = position_config.get('base_size', 0.01)
            self.position_sizer = PositionSizeOptimizer(
                base_size=base_size,
                config={**rl_config, **position_config}
            )
        else:
            self.position_sizer = None
        
        # Market Regime Detector
        regime_config = rl_config.get('regime_detection', {})
        if regime_config.get('enabled', False):
            self.regime_detector = MarketRegimeDetector(
                config={**rl_config, **regime_config}
            )
        else:
            self.regime_detector = None
        
        # Portfolio Allocator
        portfolio_config = rl_config.get('portfolio_allocation', {})
        if portfolio_config.get('enabled', False):
            symbols = portfolio_config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
            self.portfolio_allocator = PortfolioAllocator(
                symbols=symbols,
                config={**rl_config, **portfolio_config}
            )
        else:
            self.portfolio_allocator = None
        
        self.logger.info("RL Manager initialized")
    
    def select_strategy(self, market_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select best strategy using RL meta-selector."""
        if not self.meta_selector:
            return None
        
        return self.meta_selector.select_strategy(market_context)
    
    def aggregate_signals(self, signals: Dict[str, Dict], market_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Aggregate signals using RL aggregator."""
        if not self.signal_aggregator:
            return None
        
        return self.signal_aggregator.aggregate_signals(signals, market_context)
    
    def optimize_risk(self, portfolio_context: Dict[str, Any], market_context: Dict[str, Any],
                     performance_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize risk parameters using RL optimizer."""
        if not self.risk_optimizer:
            return None
        
        return self.risk_optimizer.optimize_risk(portfolio_context, market_context, performance_context)
    
    def optimize_position_size(self, signal: Dict[str, Any], market_context: Dict[str, Any],
                               portfolio_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize position size using RL optimizer."""
        if not self.position_sizer:
            return None
        
        return self.position_sizer.optimize_size(signal, market_context, portfolio_context)
    
    def detect_regime(self, market_context: Dict[str, Any], price_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect market regime using RL detector."""
        if not self.regime_detector:
            return None
        
        return self.regime_detector.detect_regime(market_context, price_data)
    
    def optimize_allocation(self, portfolio_context: Dict[str, Any],
                           market_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize portfolio allocation using RL allocator."""
        if not self.portfolio_allocator:
            return None
        
        return self.portfolio_allocator.optimize_allocation(portfolio_context, market_context)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all RL components."""
        return {
            'enabled': self.enabled,
            'meta_selector': self.meta_selector.get_status() if self.meta_selector else None,
            'signal_aggregator': self.signal_aggregator.get_status() if self.signal_aggregator else None,
            'risk_optimizer': self.risk_optimizer.get_status() if self.risk_optimizer else None,
            'position_sizer': self.position_sizer.get_status() if self.position_sizer else None,
            'regime_detector': self.regime_detector.get_status() if self.regime_detector else None,
            'portfolio_allocator': self.portfolio_allocator.get_status() if self.portfolio_allocator else None,
        }

