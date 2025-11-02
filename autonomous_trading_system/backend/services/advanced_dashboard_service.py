"""
Advanced Dashboard Service
Based on Day_47_Projects dashboard.py implementation
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class TokenMetrics:
    """Token performance metrics"""
    symbol: str
    address: Optional[str]
    current_price: float
    price_change_24h: float
    volume_24h: float
    market_cap: Optional[float]
    liquidity: Optional[float]
    holders_count: Optional[int]
    trending_score: float
    sentiment_score: float
    last_updated: datetime


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    active_positions: int
    last_updated: datetime


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot data"""
    total_value: float
    daily_pnl: float
    daily_pnl_pct: float
    total_positions: int
    cash_balance: float
    allocated_capital: float
    risk_exposure: float
    timestamp: datetime


@dataclass
class TradingAlert:
    """Trading alert data"""
    alert_id: str
    alert_type: str  # 'signal', 'risk', 'performance', 'system'
    severity: str    # 'low', 'medium', 'high', 'critical'
    title: str
    message: str
    symbol: Optional[str]
    strategy: Optional[str]
    timestamp: datetime
    acknowledged: bool = False


class AdvancedDashboardService:
    """
    Advanced Dashboard Service for real-time trading system monitoring
    
    Features:
    - Real-time token tracking and discovery
    - Strategy performance monitoring
    - Portfolio analytics and visualization
    - Risk metrics and alerts
    - Trading activity feed
    - Market sentiment dashboard
    """
    
    def __init__(self, config, strategy_engine, portfolio_manager, risk_manager, market_data_manager):
        self.config = config
        self.strategy_engine = strategy_engine
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.market_data_manager = market_data_manager
        
        # Dashboard configuration
        self.update_interval = config.get('DASHBOARD_UPDATE_INTERVAL', 5)  # seconds
        self.max_alerts = config.get('DASHBOARD_MAX_ALERTS', 100)
        self.token_tracking_enabled = config.get('TOKEN_SCANNER_ENABLED', True)
        
        # Data storage
        self.token_metrics = {}
        self.strategy_performance = {}
        self.portfolio_history = []
        self.trading_alerts = []
        self.market_overview = {}
        
        # Performance tracking
        self.dashboard_stats = {
            'total_updates': 0,
            'last_update': None,
            'active_connections': 0,
            'data_points_collected': 0
        }
        
        # Trending tokens tracking (from Day_47)
        self.trending_history = {}
        self.new_listings = []
        self.possible_gems = []
        
        logger.info(f"📊 Advanced Dashboard Service initialized")
        logger.info(f"🔄 Update interval: {self.update_interval}s")
    
    async def start(self):
        """Start the dashboard service"""
        try:
            logger.info("🚀 Starting Advanced Dashboard Service...")
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._token_tracker()),
                asyncio.create_task(self._strategy_monitor()),
                asyncio.create_task(self._portfolio_tracker()),
                asyncio.create_task(self._risk_monitor()),
                asyncio.create_task(self._market_scanner()),
                asyncio.create_task(self._alert_manager())
            ]
            
            logger.info(f"✅ Started {len(tasks)} dashboard monitoring tasks")
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"❌ Error starting dashboard service: {e}")
    
    async def _token_tracker(self):
        """Track token metrics and performance"""
        try:
            logger.info("📈 Starting token tracking...")
            
            while True:
                try:
                    await self._update_token_metrics()
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Error in token tracking: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in token tracker: {e}")
    
    async def _update_token_metrics(self):
        """Update token metrics from various sources"""
        try:
            # Get active symbols from strategy engine
            active_symbols = []
            if hasattr(self.strategy_engine, 'active_strategies'):
                for strategy in self.strategy_engine.active_strategies.values():
                    if hasattr(strategy, 'symbols'):
                        active_symbols.extend(strategy.symbols)
            
            # Add default symbols if none active
            if not active_symbols:
                active_symbols = ['BTC', 'ETH', 'SOL', 'WIF', 'POPCAT']
            
            # Update metrics for each symbol
            for symbol in set(active_symbols):
                await self._update_symbol_metrics(symbol)
            
            self.dashboard_stats['total_updates'] += 1
            self.dashboard_stats['last_update'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Error updating token metrics: {e}")
    
    async def _update_symbol_metrics(self, symbol: str):
        """Update metrics for a specific symbol"""
        try:
            # Get current price data
            current_data = await self.market_data_manager.get_current_price(symbol)
            if not current_data:
                return
            
            # Get historical data for calculations
            historical_data = await self.market_data_manager.get_historical_data(
                symbol, timeframe='1h', limit=24
            )
            
            if historical_data is None or len(historical_data) < 2:
                return
            
            # Calculate metrics
            current_price = current_data.get('price', 0)
            price_24h_ago = historical_data['close'].iloc[0] if len(historical_data) > 0 else current_price
            price_change_24h = ((current_price - price_24h_ago) / price_24h_ago * 100) if price_24h_ago > 0 else 0
            
            volume_24h = historical_data['volume'].sum() if 'volume' in historical_data.columns else 0
            
            # Calculate trending score (simplified)
            trending_score = self._calculate_trending_score(symbol, historical_data)
            
            # Get sentiment score (if available)
            sentiment_score = 0.0
            if hasattr(self.strategy_engine, 'social_sentiment_service'):
                sentiment_data = await self.strategy_engine.social_sentiment_service.get_symbol_sentiment(symbol)
                if sentiment_data:
                    sentiment_score = sentiment_data.get('overall_sentiment', {}).get('sentiment_score', 0.0)
            
            # Create token metrics
            self.token_metrics[symbol] = TokenMetrics(
                symbol=symbol,
                address=None,  # Would be populated from token database
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                market_cap=None,  # Would be calculated from supply data
                liquidity=None,   # Would be fetched from DEX data
                holders_count=None,  # Would be fetched from blockchain data
                trending_score=trending_score,
                sentiment_score=sentiment_score,
                last_updated=datetime.utcnow()
            )
            
            # Update trending history (Day_47 style)
            await self._update_trending_history(symbol, trending_score)
            
        except Exception as e:
            logger.error(f"❌ Error updating metrics for {symbol}: {e}")
    
    def _calculate_trending_score(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate trending score based on price and volume action"""
        try:
            if len(data) < 5:
                return 0.0
            
            # Price momentum (30%)
            recent_prices = data['close'].tail(5)
            price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Volume trend (40%)
            recent_volume = data['volume'].tail(5) if 'volume' in data.columns else pd.Series([1] * 5)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volatility (30%)
            returns = data['close'].pct_change().tail(10)
            volatility = returns.std() if len(returns) > 1 else 0
            
            # Combine scores
            trending_score = (
                price_momentum * 0.3 +
                min(volume_ratio - 1, 2) * 0.4 +  # Cap volume boost at 2x
                min(volatility * 10, 1) * 0.3     # Cap volatility contribution
            )
            
            return max(0, min(trending_score, 1))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"❌ Error calculating trending score for {symbol}: {e}")
            return 0.0
    
    async def _update_trending_history(self, symbol: str, trending_score: float):
        """Update trending history for consistent tracking"""
        try:
            current_time = datetime.utcnow()
            
            if symbol not in self.trending_history:
                self.trending_history[symbol] = {
                    'appearances': 0,
                    'last_seen': current_time,
                    'last_hour_counted': None,
                    'trending_scores': []
                }
            
            # Add current trending score
            self.trending_history[symbol]['trending_scores'].append({
                'score': trending_score,
                'timestamp': current_time
            })
            
            # Keep only last 24 hours of scores
            cutoff_time = current_time - timedelta(hours=24)
            self.trending_history[symbol]['trending_scores'] = [
                s for s in self.trending_history[symbol]['trending_scores']
                if s['timestamp'] > cutoff_time
            ]
            
            # Count as trending if score is high enough
            if trending_score > 0.7:  # Trending threshold
                last_counted = self.trending_history[symbol]['last_hour_counted']
                
                # Only count once per hour
                if not last_counted or (current_time - last_counted) >= timedelta(hours=1):
                    self.trending_history[symbol]['appearances'] += 1
                    self.trending_history[symbol]['last_hour_counted'] = current_time
                    logger.info(f"🔥 {symbol} is trending! Score: {trending_score:.2f}")
                
                self.trending_history[symbol]['last_seen'] = current_time
            
        except Exception as e:
            logger.error(f"❌ Error updating trending history for {symbol}: {e}")
    
    async def _strategy_monitor(self):
        """Monitor strategy performance"""
        try:
            logger.info("📊 Starting strategy monitoring...")
            
            while True:
                try:
                    await self._update_strategy_performance()
                    await asyncio.sleep(self.update_interval * 2)  # Update every 10 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error in strategy monitoring: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in strategy monitor: {e}")
    
    async def _update_strategy_performance(self):
        """Update performance metrics for all strategies"""
        try:
            if not hasattr(self.strategy_engine, 'active_strategies'):
                return
            
            for strategy_name, strategy in self.strategy_engine.active_strategies.items():
                try:
                    # Get strategy stats
                    stats = await strategy.get_strategy_stats()
                    
                    if stats and 'error' not in stats:
                        # Calculate additional metrics
                        win_rate = (stats.get('winning_trades', 0) / max(stats.get('total_trades', 1), 1)) * 100
                        
                        self.strategy_performance[strategy_name] = StrategyPerformance(
                            strategy_name=strategy_name,
                            total_trades=stats.get('total_trades', 0),
                            winning_trades=stats.get('winning_trades', 0),
                            win_rate=win_rate,
                            total_pnl=stats.get('total_pnl_pct', 0),
                            daily_pnl=stats.get('daily_pnl', 0),
                            max_drawdown=stats.get('max_drawdown', 0),
                            sharpe_ratio=stats.get('sharpe_ratio', 0),
                            active_positions=stats.get('active_positions', 0),
                            last_updated=datetime.utcnow()
                        )
                
                except Exception as e:
                    logger.error(f"❌ Error updating performance for {strategy_name}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error updating strategy performance: {e}")
    
    async def _portfolio_tracker(self):
        """Track portfolio metrics and history"""
        try:
            logger.info("💼 Starting portfolio tracking...")
            
            while True:
                try:
                    await self._update_portfolio_snapshot()
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Error in portfolio tracking: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in portfolio tracker: {e}")
    
    async def _update_portfolio_snapshot(self):
        """Update portfolio snapshot"""
        try:
            # Get portfolio data
            total_value = await self.portfolio_manager.get_total_value()
            daily_pnl = await self.portfolio_manager.get_daily_pnl()
            positions = await self.portfolio_manager.get_open_positions()
            
            # Calculate metrics
            daily_pnl_pct = (daily_pnl / total_value * 100) if total_value > 0 else 0
            cash_balance = await self.portfolio_manager.get_cash_balance()
            allocated_capital = total_value - cash_balance
            
            # Get risk exposure
            risk_exposure = await self.risk_manager.get_current_exposure()
            
            # Create snapshot
            snapshot = PortfolioSnapshot(
                total_value=total_value,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                total_positions=len(positions),
                cash_balance=cash_balance,
                allocated_capital=allocated_capital,
                risk_exposure=risk_exposure,
                timestamp=datetime.utcnow()
            )
            
            # Add to history
            self.portfolio_history.append(snapshot)
            
            # Keep only last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.portfolio_history = [
                s for s in self.portfolio_history if s.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio snapshot: {e}")
    
    async def _risk_monitor(self):
        """Monitor risk metrics and generate alerts"""
        try:
            logger.info("⚠️ Starting risk monitoring...")
            
            while True:
                try:
                    await self._check_risk_alerts()
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Error in risk monitoring: {e}")
                    await asyncio.sleep(30)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in risk monitor: {e}")
    
    async def _check_risk_alerts(self):
        """Check for risk-related alerts"""
        try:
            # Check drawdown
            max_drawdown = await self.risk_manager.get_max_drawdown()
            if max_drawdown > 0.1:  # 10% drawdown threshold
                await self._create_alert(
                    alert_type='risk',
                    severity='high' if max_drawdown > 0.15 else 'medium',
                    title='High Drawdown Alert',
                    message=f'Portfolio drawdown: {max_drawdown:.1%}',
                    symbol=None,
                    strategy=None
                )
            
            # Check risk exposure
            risk_exposure = await self.risk_manager.get_current_exposure()
            if risk_exposure > 0.8:  # 80% exposure threshold
                await self._create_alert(
                    alert_type='risk',
                    severity='medium',
                    title='High Risk Exposure',
                    message=f'Current exposure: {risk_exposure:.1%}',
                    symbol=None,
                    strategy=None
                )
            
        except Exception as e:
            logger.error(f"❌ Error checking risk alerts: {e}")
    
    async def _market_scanner(self):
        """Scan market for opportunities and alerts"""
        try:
            logger.info("🔍 Starting market scanning...")
            
            while True:
                try:
                    await self._scan_market_opportunities()
                    await asyncio.sleep(self.update_interval * 6)  # Every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error in market scanning: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in market scanner: {e}")
    
    async def _scan_market_opportunities(self):
        """Scan for market opportunities"""
        try:
            # Find possible gems (low market cap, high trending score)
            possible_gems = []
            
            for symbol, metrics in self.token_metrics.items():
                if (metrics.trending_score > 0.6 and 
                    abs(metrics.price_change_24h) > 10 and  # Significant price movement
                    metrics.volume_24h > 100000):  # Decent volume
                    
                    possible_gems.append({
                        'symbol': symbol,
                        'trending_score': metrics.trending_score,
                        'price_change_24h': metrics.price_change_24h,
                        'volume_24h': metrics.volume_24h
                    })
            
            # Sort by trending score
            possible_gems.sort(key=lambda x: x['trending_score'], reverse=True)
            self.possible_gems = possible_gems[:10]  # Keep top 10
            
            # Generate alerts for top opportunities
            for gem in possible_gems[:3]:  # Alert for top 3
                await self._create_alert(
                    alert_type='signal',
                    severity='medium',
                    title='Possible Gem Detected',
                    message=f"{gem['symbol']}: {gem['price_change_24h']:+.1f}% (24h), Trending: {gem['trending_score']:.2f}",
                    symbol=gem['symbol'],
                    strategy=None
                )
            
        except Exception as e:
            logger.error(f"❌ Error scanning market opportunities: {e}")
    
    async def _alert_manager(self):
        """Manage alerts and notifications"""
        try:
            logger.info("🔔 Starting alert management...")
            
            while True:
                try:
                    await self._cleanup_old_alerts()
                    await asyncio.sleep(60)  # Cleanup every minute
                    
                except Exception as e:
                    logger.error(f"❌ Error in alert management: {e}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in alert manager: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        try:
            # Remove alerts older than 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.trading_alerts = [
                alert for alert in self.trading_alerts
                if alert.timestamp > cutoff_time
            ]
            
            # Limit total alerts
            if len(self.trading_alerts) > self.max_alerts:
                self.trading_alerts = self.trading_alerts[-self.max_alerts:]
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up alerts: {e}")
    
    async def _create_alert(self, alert_type: str, severity: str, title: str, message: str, 
                           symbol: Optional[str] = None, strategy: Optional[str] = None):
        """Create a new trading alert"""
        try:
            alert = TradingAlert(
                alert_id=f"{alert_type}_{datetime.utcnow().timestamp()}",
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                symbol=symbol,
                strategy=strategy,
                timestamp=datetime.utcnow(),
                acknowledged=False
            )
            
            self.trading_alerts.append(alert)
            
            # Log high severity alerts
            if severity in ['high', 'critical']:
                logger.warning(f"🚨 {severity.upper()} ALERT: {title} - {message}")
            
        except Exception as e:
            logger.error(f"❌ Error creating alert: {e}")
    
    # API Methods for Dashboard Frontend
    
    async def get_dashboard_overview(self) -> Dict:
        """Get complete dashboard overview"""
        try:
            return {
                'portfolio': await self.get_portfolio_summary(),
                'strategies': await self.get_strategy_summary(),
                'tokens': await self.get_token_overview(),
                'alerts': await self.get_recent_alerts(),
                'market': await self.get_market_overview(),
                'stats': self.dashboard_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting dashboard overview: {e}")
            return {}
    
    async def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary data"""
        try:
            if not self.portfolio_history:
                return {}
            
            latest = self.portfolio_history[-1]
            
            return {
                'total_value': latest.total_value,
                'daily_pnl': latest.daily_pnl,
                'daily_pnl_pct': latest.daily_pnl_pct,
                'total_positions': latest.total_positions,
                'cash_balance': latest.cash_balance,
                'allocated_capital': latest.allocated_capital,
                'risk_exposure': latest.risk_exposure,
                'last_updated': latest.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio summary: {e}")
            return {}
    
    async def get_strategy_summary(self) -> List[Dict]:
        """Get strategy performance summary"""
        try:
            return [asdict(perf) for perf in self.strategy_performance.values()]
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy summary: {e}")
            return []
    
    async def get_token_overview(self) -> List[Dict]:
        """Get token metrics overview"""
        try:
            tokens = []
            for metrics in self.token_metrics.values():
                token_dict = asdict(metrics)
                token_dict['last_updated'] = metrics.last_updated.isoformat()
                tokens.append(token_dict)
            
            # Sort by trending score
            tokens.sort(key=lambda x: x['trending_score'], reverse=True)
            return tokens
            
        except Exception as e:
            logger.error(f"❌ Error getting token overview: {e}")
            return []
    
    async def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts"""
        try:
            recent_alerts = sorted(
                self.trading_alerts, 
                key=lambda x: x.timestamp, 
                reverse=True
            )[:limit]
            
            alerts = []
            for alert in recent_alerts:
                alert_dict = asdict(alert)
                alert_dict['timestamp'] = alert.timestamp.isoformat()
                alerts.append(alert_dict)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error getting recent alerts: {e}")
            return []
    
    async def get_market_overview(self) -> Dict:
        """Get market overview data"""
        try:
            return {
                'trending_tokens': [
                    {
                        'symbol': symbol,
                        'appearances': data['appearances'],
                        'last_seen': data['last_seen'].isoformat(),
                        'avg_trending_score': np.mean([s['score'] for s in data['trending_scores']]) if data['trending_scores'] else 0
                    }
                    for symbol, data in self.trending_history.items()
                    if data['appearances'] > 0
                ],
                'possible_gems': self.possible_gems,
                'new_listings': self.new_listings
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting market overview: {e}")
            return {}
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.trading_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error acknowledging alert: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup dashboard service"""
        try:
            # Clear data structures
            self.token_metrics.clear()
            self.strategy_performance.clear()
            self.portfolio_history.clear()
            self.trading_alerts.clear()
            self.trending_history.clear()
            
            logger.info("🧹 Advanced dashboard service cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up dashboard service: {e}") 