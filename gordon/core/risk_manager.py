"""
Risk Manager
============
Manages risk across all trading operations.

Enhanced with Day 5 risk management functionality from:
- Day_5_Projects/binance_5_risk.py (demo mode, settings persistence, monitoring loop)
- Day_5_Projects/binance_5_risk_mgmt_hl.py (account minimum balance checks)
- Day_5_Projects/binance_nice_funcs.py (position helpers, PnL calculations)

REFACTORED: Now uses modular risk management components from gordon.core.risk
- PositionSizer: Position sizing and calculations
- LeverageManager: Leverage control and stop loss/take profit
- DrawdownCalculator: Drawdown tracking and alerts
- PnLCalculator: PnL tracking and analysis
- AccountMonitor: Account balance and exposure monitoring
- RiskLimits: Risk limit enforcement and kill switches
"""

import asyncio
from typing import Dict, Optional, Any, Tuple
import logging
import json
import os

# Import modular risk management components
from .risk import (
    PositionSizer,
    LeverageManager,
    DrawdownCalculator,
    PnLCalculator,
    AccountMonitor,
    RiskLimits
)

# Import RL components
try:
    from .rl import RLManager
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    RLManager = None


class RiskManager:
    """
    Centralized risk management across all exchanges.

    Consolidates risk management from:
    - Day_5_Projects (5_risk.py, 5_risk_mgmt_hl.py)
    - Position sizing and leverage management
    - Daily loss limits and drawdown protection

    REFACTORED: Now delegates to modular components for focused responsibilities.
    """

    def __init__(self, event_bus: Any, config_manager: Any, demo_mode: bool = False):
        """Initialize risk manager with modular components."""
        self.event_bus = event_bus
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Day 5: Demo mode support
        self.demo_mode = demo_mode
        if demo_mode:
            self.logger.info("⚠️ Risk Manager running in DEMO MODE")

        # Load risk configuration
        self.risk_config = config_manager.get_risk_config()

        # Day 5: Load persisted settings if available
        self.settings_file = "risk_manager_settings.json"
        saved_settings = self.load_settings()
        self.risk_config.update(saved_settings)

        # Initialize modular components
        self.position_sizer = PositionSizer(event_bus, config_manager, demo_mode)
        self.leverage_manager = LeverageManager(event_bus, config_manager, demo_mode)
        self.drawdown_calculator = DrawdownCalculator(event_bus, config_manager, demo_mode)
        self.pnl_calculator = PnLCalculator(event_bus, config_manager, demo_mode)
        self.account_monitor = AccountMonitor(event_bus, config_manager, demo_mode)
        self.risk_limits = RiskLimits(event_bus, config_manager, demo_mode)

        # Initialize RL manager if available
        if RL_AVAILABLE:
            try:
                # Get config from config_manager
                config = {}
                if hasattr(config_manager, 'config'):
                    config = config_manager.config
                self.rl_manager = RLManager(config=config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize RL Manager: {e}")
                self.rl_manager = None
        else:
            self.rl_manager = None

        # Quick access to commonly used values (for backward compatibility)
        self.max_positions = self.risk_limits.max_positions
        self.max_position_size = self.risk_limits.max_position_size
        self.max_leverage = self.leverage_manager.max_leverage
        self.max_drawdown_percent = self.drawdown_calculator.max_drawdown_percent
        self.risk_per_trade_percent = self.position_sizer.risk_per_trade_percent
        self.daily_loss_limit = self.pnl_calculator.daily_loss_limit
        self.max_risk_per_position = self.risk_limits.max_risk_per_position
        self.default_target_percent = self.pnl_calculator.default_target_percent
        self.default_max_loss_percent = self.pnl_calculator.default_max_loss_percent
        self.account_minimum_balance = self.account_monitor.account_minimum_balance
        self.monitoring_interval = self.account_monitor.monitoring_interval

        # Tracking (delegated to components, but keeping references for backward compatibility)
        self.daily_pnl = self.pnl_calculator.daily_pnl
        self.peak_balance = self.drawdown_calculator.peak_balance
        self.current_balance = self.drawdown_calculator.current_balance
        self.current_positions = self.risk_limits.current_positions
        self.is_trading_allowed = self.risk_limits.is_trading_allowed

        # Exchange connections - shared across all components
        self.exchange_connections = {}

        # Subscribe to events
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup event handlers."""
        self.event_bus.subscribe("position_opened", self._on_position_opened)
        self.event_bus.subscribe("position_closed", self._on_position_closed)
        self.event_bus.subscribe("pnl_update", self._on_pnl_update)

    async def start(self):
        """Start risk manager."""
        self.logger.info("Risk manager started")

        # Start monitoring
        asyncio.create_task(self._monitor_risk())

    async def stop(self):
        """Stop risk manager."""
        self.logger.info("Risk manager stopped")

    async def check_trade_allowed(self, exchange: str, symbol: str,
                                 side: str, amount: float,
                                 price: Optional[float] = None) -> bool:
        """
        Check if a trade is allowed based on risk parameters.
        Delegates to RiskLimits, PnLCalculator, and DrawdownCalculator.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: Buy or sell
            amount: Trade amount
            price: Trade price

        Returns:
            Whether trade is allowed
        """
        # Check basic risk limits (delegated to RiskLimits)
        if not await self.risk_limits.check_trade_allowed(exchange, symbol, side, amount, price):
            return False

        # Check daily loss limit (delegated to PnLCalculator)
        if self.pnl_calculator.is_daily_loss_limit_exceeded():
            self.logger.warning(f"Daily loss limit reached: ${self.pnl_calculator.get_daily_pnl()}")
            await self.event_bus.emit("risk_limit_reached", {
                "type": "daily_loss",
                "loss": self.pnl_calculator.get_daily_pnl(),
                "limit": self.daily_loss_limit
            })
            return False

        # Check drawdown (delegated to DrawdownCalculator)
        if self.drawdown_calculator.is_drawdown_exceeded():
            drawdown = self.drawdown_calculator.calculate_current_drawdown()
            self.logger.warning(f"Max drawdown exceeded: {drawdown:.2f}%")
            await self.event_bus.emit("risk_limit_reached", {
                "type": "max_drawdown",
                "drawdown": drawdown,
                "limit": self.max_drawdown_percent
            })
            return False

        return True

    def calculate_position_size(self, balance: float, stop_loss_percent: float) -> float:
        """
        Calculate position size based on risk management rules.
        Delegates to PositionSizer.

        Args:
            balance: Account balance
            stop_loss_percent: Stop loss percentage

        Returns:
            Position size
        """
        return self.position_sizer.calculate_position_size(balance, stop_loss_percent)

    def calculate_stop_loss(self, entry_price: float, side: str,
                          volatility: float = None) -> float:
        """
        Calculate stop loss price.
        Delegates to LeverageManager.

        Args:
            entry_price: Entry price
            side: Buy or sell
            volatility: Optional volatility measure

        Returns:
            Stop loss price
        """
        return self.leverage_manager.calculate_stop_loss(entry_price, side, volatility)

    def calculate_take_profit(self, entry_price: float, side: str,
                            risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price.
        Delegates to LeverageManager.

        Args:
            entry_price: Entry price
            side: Buy or sell
            risk_reward_ratio: Risk/reward ratio

        Returns:
            Take profit price
        """
        return self.leverage_manager.calculate_take_profit(entry_price, side, risk_reward_ratio)

    async def _monitor_risk(self):
        """Monitor risk metrics continuously. Delegates to modular components."""
        while True:
            try:
                # Check drawdown (delegated to DrawdownCalculator)
                await self.drawdown_calculator.check_drawdown_alerts()

                # Disable trading if drawdown exceeded
                if self.drawdown_calculator.is_drawdown_exceeded():
                    self.risk_limits.disable_trading("max_drawdown_exceeded")

                # Check daily loss (delegated to PnLCalculator)
                await self.pnl_calculator.check_daily_loss_alert()

                # Record drawdown history
                self.drawdown_calculator.add_to_history()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error monitoring risk: {e}")
                await asyncio.sleep(60)

    async def _on_position_opened(self, event: Dict):
        """Handle position opened event. Delegates to RiskLimits."""
        self.risk_limits.increment_position_count()
        self.current_positions = self.risk_limits.current_positions

    async def _on_position_closed(self, event: Dict):
        """Handle position closed event. Delegates to RiskLimits and PnLCalculator."""
        self.risk_limits.decrement_position_count()
        self.current_positions = self.risk_limits.current_positions

        # Update PnL
        pnl = event.get("data", {}).get("pnl", 0)
        self.pnl_calculator.update_daily_pnl(pnl)

    async def _on_pnl_update(self, event: Dict):
        """Handle PnL update event. Delegates to DrawdownCalculator."""
        balance = event.get("data", {}).get("balance", 0)

        self.drawdown_calculator.update_balance(balance)
        self.current_balance = self.drawdown_calculator.current_balance
        self.peak_balance = self.drawdown_calculator.peak_balance

    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics with advanced analytics.
        Aggregates metrics from all modular components and calculates advanced metrics.
        """
        # Get metrics from each component
        drawdown_metrics = self.drawdown_calculator.get_drawdown_metrics()
        pnl_metrics = self.pnl_calculator.get_pnl_metrics()
        risk_limits_metrics = self.risk_limits.get_risk_limits()

        # Calculate advanced metrics
        current_drawdown = drawdown_metrics["current_drawdown_percent"]
        max_drawdown = drawdown_metrics["max_drawdown_percent"]
        daily_pnl = pnl_metrics["daily_pnl"]
        
        # Calculate VaR (Value at Risk) - 95% confidence
        var_95 = self._calculate_var_95()
        
        # Calculate Sharpe Ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Calculate volatility
        volatility = self._calculate_volatility()
        
        # Calculate risk score (0-100, higher is riskier)
        risk_score = self._calculate_risk_score(
            current_drawdown, max_drawdown, daily_pnl, var_95, volatility
        )

        # Combine into unified metrics dict
        metrics = {
            # Risk limits
            "current_positions": risk_limits_metrics["current_positions"],
            "max_positions": risk_limits_metrics["max_positions"],
            "is_trading_allowed": risk_limits_metrics["is_trading_allowed"],

            # Balance and drawdown
            "current_balance": drawdown_metrics["current_balance"],
            "peak_balance": drawdown_metrics["peak_balance"],
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,

            # PnL
            "daily_pnl": daily_pnl,
            "daily_loss_limit": pnl_metrics["daily_loss_limit"],
            "weekly_pnl": pnl_metrics["weekly_pnl"],
            "monthly_pnl": pnl_metrics["monthly_pnl"],

            # Position sizing
            "risk_per_trade_percent": self.risk_per_trade_percent,
            "max_leverage": self.max_leverage,
            "max_position_size": self.max_position_size,
            
            # Advanced metrics (Phase 1)
            "var_95": var_95,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "risk_score": risk_score
        }
        
        # Add RL-optimized risk parameters if available
        if self.rl_manager and self.rl_manager.enabled and self.rl_manager.risk_optimizer:
            try:
                portfolio_context = {
                    'balance': metrics['current_balance'],
                    'drawdown': current_drawdown,
                    'drawdown_pct': current_drawdown,
                    'leverage': self.max_leverage,
                    'position_count': metrics['current_positions'],
                    'margin_used': 0.0,  # Should be calculated
                    'available_balance': metrics['current_balance'] * 0.5  # Simplified
                }
                market_context = {
                    'volatility': volatility,
                    'trend': 0.0,  # Should be calculated
                    'volume_ratio': 1.0,
                    'price_change_24h': daily_pnl / metrics['current_balance'] if metrics['current_balance'] > 0 else 0.0,
                    'fear_greed_index': 50.0
                }
                performance_context = {
                    'win_rate': 0.5,  # Should be calculated
                    'sharpe_ratio': sharpe_ratio,
                    'total_pnl': daily_pnl,
                    'total_pnl_pct': current_drawdown,
                    'daily_pnl': daily_pnl,
                    'daily_pnl_pct': daily_pnl / metrics['current_balance'] if metrics['current_balance'] > 0 else 0.0,
                    'consecutive_losses': 0
                }
                
                rl_risk = self.rl_manager.optimize_risk(portfolio_context, market_context, performance_context)
                if rl_risk:
                    metrics['rl_optimized_risk'] = rl_risk
            except Exception as e:
                self.logger.warning(f"Failed to get RL risk optimization: {e}")
        
        return metrics
    
    def optimize_risk_with_rl(self) -> Optional[Dict]:
        """Optimize risk parameters using RL if available."""
        if not self.rl_manager or not self.rl_manager.enabled or not self.rl_manager.risk_optimizer:
            return None
        
        metrics = self.get_risk_metrics()
        
        portfolio_context = {
            'balance': metrics['current_balance'],
            'drawdown': metrics['current_drawdown'],
            'drawdown_pct': metrics['current_drawdown'],
            'leverage': self.max_leverage,
            'position_count': metrics['current_positions'],
            'margin_used': 0.0,
            'available_balance': metrics['current_balance'] * 0.5
        }
        
        market_context = {
            'volatility': metrics.get('volatility', 0.02),
            'trend': 0.0,
            'volume_ratio': 1.0,
            'price_change_24h': metrics['daily_pnl'] / metrics['current_balance'] if metrics['current_balance'] > 0 else 0.0,
            'fear_greed_index': 50.0
        }
        
        performance_context = {
            'win_rate': 0.5,
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'total_pnl': metrics['daily_pnl'],
            'total_pnl_pct': metrics['current_drawdown'],
            'daily_pnl': metrics['daily_pnl'],
            'daily_pnl_pct': metrics['daily_pnl'] / metrics['current_balance'] if metrics['current_balance'] > 0 else 0.0,
            'consecutive_losses': 0
        }
        
        return self.rl_manager.optimize_risk(portfolio_context, market_context, performance_context)
    
    def _calculate_var_95(self) -> Optional[float]:
        """
        Calculate Value at Risk (VaR) at 95% confidence level.
        
        Returns:
            VaR value or None if insufficient data
        """
        try:
            import numpy as np
            
            # Get PnL history from drawdown calculator
            pnl_history = self.drawdown_calculator.drawdown_history
            
            if len(pnl_history) < 20:  # Need at least 20 data points
                return None
            
            # Extract PnL values
            pnl_values = [point.get('pnl', 0) for point in pnl_history[-252:]]  # Last year
            
            if not pnl_values or len(pnl_values) < 20:
                return None
            
            # Calculate VaR using historical simulation method
            # VaR at 95% = 5th percentile of returns
            sorted_pnl = sorted(pnl_values)
            var_index = int(len(sorted_pnl) * 0.05)
            
            if var_index >= len(sorted_pnl):
                var_index = len(sorted_pnl) - 1
            
            var_95 = abs(sorted_pnl[var_index]) if sorted_pnl[var_index] < 0 else 0
            
            return var_95
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate VaR: {e}")
            return None
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> Optional[float]:
        """
        Calculate Sharpe Ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            Sharpe ratio or None if insufficient data
        """
        try:
            import numpy as np
            
            # Get PnL history
            pnl_history = self.drawdown_calculator.drawdown_history
            
            if len(pnl_history) < 20:
                return None
            
            # Extract returns (daily PnL percentages)
            returns = []
            for i in range(1, len(pnl_history)):
                prev_balance = pnl_history[i-1].get('balance', 0)
                curr_balance = pnl_history[i].get('balance', 0)
                
                if prev_balance > 0:
                    daily_return = (curr_balance - prev_balance) / prev_balance
                    returns.append(daily_return)
            
            if len(returns) < 20:
                return None
            
            # Calculate Sharpe Ratio
            # Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return None
            
            # Annualize (assuming daily returns)
            annual_mean = mean_return * 252
            annual_std = std_return * np.sqrt(252)
            
            sharpe = (annual_mean - risk_free_rate) / annual_std if annual_std > 0 else None
            
            return sharpe
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate Sharpe ratio: {e}")
            return None
    
    def _calculate_volatility(self) -> Optional[float]:
        """
        Calculate portfolio volatility (annualized).
        
        Returns:
            Volatility as percentage or None if insufficient data
        """
        try:
            import numpy as np
            
            # Get PnL history
            pnl_history = self.drawdown_calculator.drawdown_history
            
            if len(pnl_history) < 20:
                return None
            
            # Extract returns
            returns = []
            for i in range(1, len(pnl_history)):
                prev_balance = pnl_history[i-1].get('balance', 0)
                curr_balance = pnl_history[i].get('balance', 0)
                
                if prev_balance > 0:
                    daily_return = (curr_balance - prev_balance) / prev_balance
                    returns.append(daily_return)
            
            if len(returns) < 20:
                return None
            
            # Calculate annualized volatility
            std_return = np.std(returns)
            annual_volatility = std_return * np.sqrt(252) * 100  # Convert to percentage
            
            return annual_volatility
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate volatility: {e}")
            return None
    
    def _calculate_risk_score(self, current_drawdown: float, max_drawdown: float,
                             daily_pnl: float, var_95: Optional[float],
                             volatility: Optional[float]) -> float:
        """
        Calculate overall risk score (0-100, higher is riskier).
        
        Args:
            current_drawdown: Current drawdown percentage
            max_drawdown: Maximum drawdown percentage
            daily_pnl: Daily PnL
            var_95: Value at Risk at 95%
            volatility: Portfolio volatility
        
        Returns:
            Risk score from 0-100
        """
        score = 0.0
        
        # Drawdown component (40% weight)
        drawdown_score = min(current_drawdown / self.max_drawdown_percent * 100, 100)
        score += drawdown_score * 0.4
        
        # Daily loss component (30% weight)
        if self.daily_loss_limit > 0:
            loss_ratio = abs(min(daily_pnl, 0)) / abs(self.daily_loss_limit)
            loss_score = min(loss_ratio * 100, 100)
            score += loss_score * 0.3
        
        # VaR component (20% weight)
        if var_95 is not None and self.current_balance > 0:
            var_ratio = var_95 / self.current_balance
            var_score = min(var_ratio * 100 * 10, 100)  # Scale appropriately
            score += var_score * 0.2
        
        # Volatility component (10% weight)
        if volatility is not None:
            # High volatility = higher risk
            vol_score = min(volatility / 50 * 100, 100)  # Normalize to 50% volatility = 100
            score += vol_score * 0.1
        
        return min(score, 100.0)  # Cap at 100

    def reset_daily_limits(self):
        """
        Reset daily limits (call at start of new trading day).
        Delegates to PnLCalculator and RiskLimits.
        """
        # Reset daily PnL
        self.pnl_calculator.reset_daily_pnl()

        # Re-enable trading if it was disabled due to daily loss
        if not self.risk_limits.is_trading_allowed:
            self.risk_limits.enable_trading()
            self.is_trading_allowed = self.risk_limits.is_trading_allowed

    # ========== Day 5 Specific Methods ==========

    def save_settings(self, settings: Dict = None):
        """
        Save risk manager settings to file (Day 5 feature).

        Args:
            settings: Optional settings dict, otherwise saves current config
        """
        try:
            if settings is None:
                settings = {
                    "max_positions": self.max_positions,
                    "max_position_size": self.max_position_size,
                    "max_leverage": self.max_leverage,
                    "max_drawdown_percent": self.max_drawdown_percent,
                    "risk_per_trade_percent": self.risk_per_trade_percent,
                    "daily_loss_limit": self.daily_loss_limit,
                    "max_risk_per_position": self.max_risk_per_position,
                    "default_target_percent": self.default_target_percent,
                    "default_max_loss_percent": self.default_max_loss_percent,
                    "account_minimum_balance": self.account_minimum_balance,
                    "monitoring_interval": self.monitoring_interval
                }

            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            self.logger.info(f"Risk settings saved to {self.settings_file}")

        except Exception as e:
            self.logger.error(f"Error saving risk settings: {e}")

    def load_settings(self) -> Dict:
        """
        Load risk manager settings from file (Day 5 feature).

        Returns:
            Saved settings or empty dict
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                self.logger.info(f"Risk settings loaded from {self.settings_file}")
                return settings
            return {}
        except Exception as e:
            self.logger.error(f"Error loading risk settings: {e}")
            return {}

    def set_exchange_connection(self, exchange: str, connection: Any):
        """
        Set exchange connection for direct position queries (Day 5 feature).
        Shares connection with all modular components.

        Args:
            exchange: Exchange name
            connection: Exchange connection object
        """
        self.exchange_connections[exchange] = connection

        # Share connection with all components
        self.position_sizer.set_exchange_connection(exchange, connection)
        self.leverage_manager.set_exchange_connection(exchange, connection)
        self.drawdown_calculator.set_exchange_connection(exchange, connection)
        self.pnl_calculator.set_exchange_connection(exchange, connection)
        self.account_monitor.set_exchange_connection(exchange, connection)
        self.risk_limits.set_exchange_connection(exchange, connection)

    async def get_position_details(self, exchange: str, symbol: str) -> Tuple[bool, float, float, float, bool]:
        """
        Get detailed position information (Day 5 feature from binance_nice_funcs).
        Delegates to PnLCalculator.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Tuple of (has_position, size, entry_price, pnl_percent, is_long)
        """
        return await self.pnl_calculator.get_position_details(exchange, symbol)

    async def check_pnl_limits(self, exchange: str, symbol: str,
                              target_percent: Optional[float] = None,
                              max_loss_percent: Optional[float] = None) -> bool:
        """
        Check if position PnL exceeds limits and needs closing (Day 5 feature).
        Delegates to PnLCalculator.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            target_percent: Take profit target (uses default if None)
            max_loss_percent: Stop loss limit (uses default if None)

        Returns:
            True if position was closed, False otherwise
        """
        return await self.pnl_calculator.check_pnl_limits(exchange, symbol, target_percent, max_loss_percent)

    async def check_account_minimum(self, exchange: str) -> bool:
        """
        Check if account balance is below minimum (Day 5 feature from binance_5_risk_mgmt_hl).
        Delegates to AccountMonitor.

        Args:
            exchange: Exchange name

        Returns:
            True if below minimum, False otherwise
        """
        return await self.account_monitor.check_account_minimum(exchange)

    async def position_size_kill_switch(self, exchange: str):
        """
        Kill positions that exceed maximum risk (Day 5 feature).
        Delegates to RiskLimits.

        Args:
            exchange: Exchange name
        """
        await self.risk_limits.position_size_kill_switch(exchange)

    async def start_day5_monitoring(self, exchange: str, symbol: str,
                                   target_percent: Optional[float] = None,
                                   max_loss_percent: Optional[float] = None,
                                   check_interval: Optional[int] = None):
        """
        Start Day 5 style monitoring loop for a position.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            target_percent: Take profit target
            max_loss_percent: Stop loss limit
            check_interval: Check interval in seconds
        """
        interval = check_interval or self.monitoring_interval

        self.logger.info(f"Starting Day 5 monitoring for {symbol} on {exchange}")
        self.logger.info(f"Target: {target_percent or self.default_target_percent}%, "
                        f"Max Loss: {max_loss_percent or self.default_max_loss_percent}%, "
                        f"Interval: {interval}s")

        while self.is_trading_allowed:
            try:
                # Check position size limits
                await self.position_size_kill_switch(exchange)

                # Check PnL limits
                closed = await self.check_pnl_limits(exchange, symbol, target_percent, max_loss_percent)

                if closed:
                    self.logger.info(f"Position {symbol} closed due to PnL limits")
                    break

                # Check account minimum
                below_min = await self.check_account_minimum(exchange)
                if below_min:
                    self.logger.warning("Account below minimum, triggering emergency stop")
                    await self.event_bus.emit("kill_switch_triggered", {
                        "exchange": exchange,
                        "symbol": symbol,
                        "reason": "account_minimum"
                    })
                    break

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Error in Day 5 monitoring: {e}")
                await asyncio.sleep(interval)

    async def execute_kill_switch(self, exchange: str, symbol: str):
        """
        Execute kill switch to close position immediately (Day 5 feature).
        Delegates to RiskLimits.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
        """
        return await self.risk_limits.execute_kill_switch(exchange, symbol, "manual")

    # ========== Day 5 Helper Methods from binance_nice_funcs.py ==========

    async def get_ask_bid(self, exchange: str, symbol: str) -> Tuple[float, float, Dict]:
        """
        Get current ask and bid prices (Day 5 feature from binance_nice_funcs.py).
        Delegates to AccountMonitor.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Tuple of (ask_price, bid_price, order_book)
        """
        return await self.account_monitor.get_ask_bid(exchange, symbol)

    async def get_precision_info(self, exchange: str, symbol: str) -> Tuple[int, int]:
        """
        Get size and price decimal precision for a symbol (Day 5 feature).
        Delegates to AccountMonitor.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Tuple of (size_decimals, price_decimals)
        """
        return await self.account_monitor.get_precision_info(exchange, symbol)

    async def place_limit_order_with_retry(self, exchange: str, symbol: str, side: str,
                                          size: float, price: float,
                                          reduce_only: bool = False,
                                          max_retries: int = 3) -> Optional[Dict]:
        """
        Place limit order with retry logic (Day 5 feature).
        Delegates to AccountMonitor.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            price: Limit price
            reduce_only: Whether order reduces position only
            max_retries: Maximum retry attempts

        Returns:
            Order result or None if failed
        """
        return await self.account_monitor.place_limit_order_with_retry(
            exchange, symbol, side, size, price, reduce_only, max_retries
        )

    async def cancel_all_orders_for_symbol(self, exchange: str, symbol: str) -> bool:
        """
        Cancel all open orders for a symbol (Day 5 feature from binance_nice_funcs.py).
        Delegates to AccountMonitor.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Success status
        """
        return await self.account_monitor.cancel_all_orders_for_symbol(exchange, symbol)

    async def get_account_value(self, exchange: str) -> float:
        """
        Get total account value (Day 5 feature from binance_nice_funcs.py acct_bal).
        Delegates to AccountMonitor.

        Args:
            exchange: Exchange name

        Returns:
            Account value in USD
        """
        return await self.account_monitor.get_account_value(exchange)

    async def execute_gradual_close(self, exchange: str, symbol: str,
                                   total_size: float, chunks: int = 5,
                                   delay_seconds: int = 5) -> bool:
        """
        Close position gradually in chunks (Day 5 enhancement).
        Delegates to RiskLimits.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            total_size: Total position size to close
            chunks: Number of chunks to split the close into
            delay_seconds: Delay between chunks

        Returns:
            Success status
        """
        return await self.risk_limits.execute_gradual_close(
            exchange, symbol, total_size, chunks, delay_seconds
        )

    async def monitor_and_close_on_pnl(self, exchange: str, symbol: str,
                                      target: float = None, max_loss: float = None,
                                      check_interval: int = 60) -> bool:
        """
        Monitor position and close based on PnL thresholds (Day 5 main feature from binance_nice_funcs.py).

        This is a high-level convenience method that coordinates multiple components.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            target: Target profit percentage
            max_loss: Maximum loss percentage (negative)
            check_interval: Check interval in seconds

        Returns:
            True if position closed, False otherwise
        """
        target = target or self.default_target_percent
        max_loss = max_loss or self.default_max_loss_percent

        self.logger.info(f"Starting PnL monitor for {symbol}: Target={target}%, MaxLoss={max_loss}%")

        while self.is_trading_allowed:
            try:
                has_position, size, _, pnl_percent, is_long = await self.get_position_details(exchange, symbol)

                if not has_position:
                    self.logger.info(f"No position to monitor for {symbol}")
                    return False

                self.logger.info(f"{symbol} PnL: {pnl_percent:.2f}% (Target: {target}%, Stop: {max_loss}%)")

                # Check if PnL triggers close
                if pnl_percent >= target:
                    self.logger.info(f"Target reached for {symbol}: {pnl_percent:.2f}% >= {target}%")
                    await self.execute_gradual_close(exchange, symbol, abs(size))
                    return True

                elif pnl_percent <= max_loss:
                    self.logger.warning(f"Stop loss triggered for {symbol}: {pnl_percent:.2f}% <= {max_loss}%")
                    # Fast close on stop loss (single chunk)
                    await self.execute_gradual_close(exchange, symbol, abs(size), chunks=1)
                    return True

                await asyncio.sleep(check_interval)

            except Exception as e:
                self.logger.error(f"Error monitoring {symbol}: {e}")
                await asyncio.sleep(check_interval)

        return False