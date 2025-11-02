"""
Risk Manager - Handles risk assessment, position sizing, and risk limits
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from dataclasses import dataclass
from core.config import Settings
from strategies.base_strategy import StrategySignal

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for the portfolio"""
    max_drawdown: float
    current_drawdown: float
    var_95: float  # Value at Risk 95%
    sharpe_ratio: float
    volatility: float
    risk_score: float  # 0-100 scale


class RiskManager:
    """
    Manages trading risk, position sizing, and risk limits
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Risk limits
        self.max_drawdown = settings.MAX_DRAWDOWN
        self.max_daily_loss = settings.MAX_DAILY_LOSS
        self.position_size_risk = settings.POSITION_SIZE_RISK
        
        # Risk tracking
        self.daily_pnl_history = []
        self.portfolio_values = []
        self.max_portfolio_value = 100000.0  # Starting value
        self.current_drawdown = 0.0
        
        # Risk state
        self.is_active = True
        self.risk_override = False
        
        logger.info("🔧 Risk Manager initialized")
        logger.info(f"   Max Drawdown: {self.max_drawdown * 100}%")
        logger.info(f"   Max Daily Loss: {self.max_daily_loss * 100}%")
        logger.info(f"   Position Size Risk: {self.position_size_risk * 100}%")
    
    async def approve_trade(self, signal: StrategySignal) -> bool:
        """Approve or reject a trading signal based on risk assessment"""
        try:
            if not self.is_active:
                logger.warning("⚠️ Risk manager is inactive")
                return False
            
            if self.risk_override:
                logger.info("🔓 Risk override active - approving trade")
                return True
            
            # Check drawdown limits
            if self.current_drawdown > self.max_drawdown:
                logger.warning(f"⚠️ Max drawdown exceeded: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}")
                return False
            
            # Check daily loss limits
            daily_loss = await self._get_daily_loss()
            if daily_loss < -self.max_daily_loss:
                logger.warning(f"⚠️ Daily loss limit exceeded: {daily_loss:.2%}")
                return False
            
            # Check signal confidence
            if signal.confidence < 0.3:  # Minimum confidence threshold
                logger.warning(f"⚠️ Signal confidence too low: {signal.confidence:.2f}")
                return False
            
            # Check symbol-specific risk
            if not await self._check_symbol_risk(signal.symbol):
                return False
            
            logger.debug(f"✅ Trade approved: {signal.action.value} {signal.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in trade approval: {e}")
            return False
    
    async def calculate_position_size(self, signal: StrategySignal) -> float:
        """Calculate appropriate position size based on risk parameters"""
        try:
            # Base position size from settings
            base_size = self.settings.DEFAULT_POSITION_SIZE
            
            # Adjust based on confidence
            confidence_multiplier = signal.confidence
            
            # Adjust based on current risk level
            risk_score = await self.get_risk_score()
            risk_multiplier = max(0.1, 1 - (risk_score / 100))  # Reduce size as risk increases
            
            # Adjust based on volatility (if available)
            volatility_multiplier = 1.0  # Default, would be calculated from price data
            
            # Calculate final size
            position_size = base_size * confidence_multiplier * risk_multiplier * volatility_multiplier
            
            # Apply maximum limits
            max_size = self.settings.MAX_POSITION_SIZE
            position_size = min(position_size, max_size)
            
            # Ensure minimum size
            min_size = 10.0  # Minimum $10 position
            position_size = max(position_size, min_size)
            
            logger.debug(f"📊 Position size calculated: ${position_size:.2f} "
                        f"(confidence: {confidence_multiplier:.2f}, risk: {risk_multiplier:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return self.settings.DEFAULT_POSITION_SIZE
    
    async def _check_symbol_risk(self, symbol: str) -> bool:
        """Check symbol-specific risk factors"""
        try:
            # Check if symbol is in high-risk category
            high_risk_symbols = ["MEME", "SHIB", "DOGE"]  # Example high-risk symbols
            if any(risk_symbol in symbol.upper() for risk_symbol in high_risk_symbols):
                logger.warning(f"⚠️ High-risk symbol detected: {symbol}")
                return False
            
            # Check for recent volatility spikes
            # In a real implementation, this would analyze recent price data
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking symbol risk for {symbol}: {e}")
            return True
    
    async def _get_daily_loss(self) -> float:
        """Get current daily loss percentage"""
        try:
            # In a real implementation, this would calculate from actual portfolio data
            # For now, simulate based on recent history
            if self.daily_pnl_history:
                return self.daily_pnl_history[-1] if self.daily_pnl_history else 0.0
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error getting daily loss: {e}")
            return 0.0
    
    async def update_portfolio_value(self, current_value: float):
        """Update portfolio value for risk calculations"""
        try:
            self.portfolio_values.append({
                'value': current_value,
                'timestamp': datetime.utcnow()
            })
            
            # Update max portfolio value
            if current_value > self.max_portfolio_value:
                self.max_portfolio_value = current_value
            
            # Calculate current drawdown
            self.current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
            
            # Keep only recent values (last 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            self.portfolio_values = [
                pv for pv in self.portfolio_values 
                if pv['timestamp'] > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio value: {e}")
    
    async def get_max_drawdown(self) -> float:
        """Get maximum drawdown"""
        return self.current_drawdown
    
    async def get_current_drawdown(self) -> float:
        """Get current drawdown"""
        return self.current_drawdown
    
    async def get_current_exposure(self) -> float:
        """Get current portfolio exposure"""
        # In a real implementation, this would calculate total exposure
        # For now, return a simulated value
        return 0.5  # 50% exposure
    
    async def get_risk_score(self) -> float:
        """Get overall risk score (0-100)"""
        try:
            risk_factors = []
            
            # Drawdown factor
            drawdown_factor = min(self.current_drawdown / self.max_drawdown, 1.0) * 30
            risk_factors.append(drawdown_factor)
            
            # Daily loss factor
            daily_loss = await self._get_daily_loss()
            daily_loss_factor = min(abs(daily_loss) / self.max_daily_loss, 1.0) * 25
            risk_factors.append(daily_loss_factor)
            
            # Volatility factor (simulated)
            volatility_factor = 20  # Base volatility score
            risk_factors.append(volatility_factor)
            
            # Exposure factor
            exposure = await self.get_current_exposure()
            exposure_factor = exposure * 25
            risk_factors.append(exposure_factor)
            
            # Calculate weighted risk score
            risk_score = sum(risk_factors)
            return min(risk_score, 100.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating risk score: {e}")
            return 50.0  # Default medium risk
    
    async def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        try:
            return RiskMetrics(
                max_drawdown=self.current_drawdown,
                current_drawdown=self.current_drawdown,
                var_95=0.05,  # Simulated VaR
                sharpe_ratio=1.2,  # Simulated Sharpe ratio
                volatility=0.15,  # Simulated volatility
                risk_score=await self.get_risk_score()
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 50)
    
    async def emergency_stop(self):
        """Trigger emergency stop"""
        logger.warning("🚨 EMERGENCY STOP TRIGGERED BY RISK MANAGER")
        self.is_active = False
    
    async def resume_trading(self):
        """Resume trading after emergency stop"""
        logger.info("▶️ Trading resumed by risk manager")
        self.is_active = True
    
    def set_risk_override(self, enabled: bool):
        """Enable/disable risk override"""
        self.risk_override = enabled
        status = "enabled" if enabled else "disabled"
        logger.warning(f"🔓 Risk override {status}")
    
    async def validate_portfolio_limits(self, portfolio_value: float, positions: Dict) -> bool:
        """Validate portfolio against risk limits"""
        try:
            # Update portfolio value
            await self.update_portfolio_value(portfolio_value)
            
            # Check drawdown
            if self.current_drawdown > self.max_drawdown:
                logger.error(f"❌ Portfolio drawdown limit exceeded: {self.current_drawdown:.2%}")
                return False
            
            # Check position concentration
            if positions:
                max_position_value = max(
                    abs(pos.size * pos.current_price) for pos in positions.values()
                )
                concentration = max_position_value / portfolio_value
                
                if concentration > 0.3:  # Max 30% in single position
                    logger.warning(f"⚠️ High position concentration: {concentration:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating portfolio limits: {e}")
            return False
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        return {
            'is_active': self.is_active,
            'risk_override': self.risk_override,
            'current_drawdown': self.current_drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'daily_loss_limit': self.max_daily_loss,
            'risk_score': await self.get_risk_score(),
            'portfolio_values_count': len(self.portfolio_values)
        } 