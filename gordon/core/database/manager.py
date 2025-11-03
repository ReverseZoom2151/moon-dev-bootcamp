"""
Database Manager
================
Manages database connections and operations for persistent storage.
Similar structure to ConversationMemory but using SQLAlchemy for structured data.
"""

import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import Base, Trade, Position, StrategyMetric, RiskMetric, PerformanceSnapshot

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for persistent storage of trades, positions, and metrics.
    
    Similar to ConversationMemory:
    - Directory-based storage (database directory)
    - Easy initialization and cleanup
    - Automatic table creation
    - Session management
    
    Features:
    - SQLite by default (can use PostgreSQL via config)
    - Auto-creates tables on initialization
    - Session management for thread safety
    - Query helpers for common operations
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        database_dir: str = "./database",
        database_file: str = "gordon.db",
        echo: bool = False
    ):
        """
        Initialize database manager.
        
        Args:
            database_url: Full database URL (e.g., 'sqlite:///gordon.db' or 'postgresql://...')
            database_dir: Directory to store database files (for SQLite)
            database_file: Database filename (for SQLite)
            echo: Enable SQLAlchemy echo for debugging
        """
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine database URL
        if database_url:
            self.database_url = database_url
        else:
            # Default to SQLite
            db_path = self.database_dir / database_file
            self.database_url = f"sqlite:///{db_path}"
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            echo=echo,
            connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
        
        # Create tables
        self._create_tables()
        
        logger.info(f"Database initialized: {self.database_url}")
    
    def _create_tables(self):
        """Create all tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
            logger.debug("Database tables created/verified")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")
    
    # ========== Trade Operations ==========
    
    def save_trade(self, trade_data: Dict[str, Any]) -> Optional[Trade]:
        """Save a trade to database."""
        session = self.get_session()
        try:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            logger.debug(f"Trade saved: {trade.trade_id}")
            return trade
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error saving trade: {e}")
            return None
        finally:
            session.close()
    
    def get_trades(
        self,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Query trades with filters."""
        session = self.get_session()
        try:
            query = session.query(Trade)
            
            if exchange:
                query = query.filter(Trade.exchange == exchange)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if strategy_name:
                query = query.filter(Trade.strategy_name == strategy_name)
            if start_date:
                query = query.filter(Trade.timestamp >= start_date)
            if end_date:
                query = query.filter(Trade.timestamp <= end_date)
            
            query = query.order_by(Trade.timestamp.desc()).limit(limit)
            return query.all()
        finally:
            session.close()
    
    # ========== Position Operations ==========
    
    def save_position(self, position_data: Dict[str, Any]) -> Optional[Position]:
        """Save or update a position."""
        session = self.get_session()
        try:
            # Check if position already exists
            existing = session.query(Position).filter(
                Position.exchange == position_data['exchange'],
                Position.symbol == position_data['symbol'],
                Position.is_open == True
            ).first()
            
            if existing:
                # Update existing position
                for key, value in position_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
                session.commit()
                session.refresh(existing)
                logger.debug(f"Position updated: {existing.symbol}")
                return existing
            else:
                # Create new position
                position = Position(**position_data)
                session.add(position)
                session.commit()
                session.refresh(position)
                logger.debug(f"Position saved: {position.symbol}")
                return position
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error saving position: {e}")
            return None
        finally:
            session.close()
    
    def close_position(self, exchange: str, symbol: str, close_data: Dict[str, Any]) -> Optional[Position]:
        """Close a position."""
        session = self.get_session()
        try:
            position = session.query(Position).filter(
                Position.exchange == exchange,
                Position.symbol == symbol,
                Position.is_open == True
            ).first()
            
            if position:
                position.is_open = False
                position.is_closed = True
                position.closed_at = datetime.utcnow()
                
                # Update with close data
                for key, value in close_data.items():
                    if hasattr(position, key):
                        setattr(position, key, value)
                
                session.commit()
                session.refresh(position)
                logger.debug(f"Position closed: {position.symbol}")
                return position
            return None
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error closing position: {e}")
            return None
        finally:
            session.close()
    
    def get_open_positions(
        self,
        exchange: Optional[str] = None,
        strategy_name: Optional[str] = None
    ) -> List[Position]:
        """Get all open positions."""
        session = self.get_session()
        try:
            query = session.query(Position).filter(Position.is_open == True)
            
            if exchange:
                query = query.filter(Position.exchange == exchange)
            if strategy_name:
                query = query.filter(Position.strategy_name == strategy_name)
            
            return query.all()
        finally:
            session.close()
    
    def get_positions(
        self,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Position]:
        """Query positions with filters."""
        session = self.get_session()
        try:
            query = session.query(Position)
            
            if exchange:
                query = query.filter(Position.exchange == exchange)
            if symbol:
                query = query.filter(Position.symbol == symbol)
            if strategy_name:
                query = query.filter(Position.strategy_name == strategy_name)
            if start_date:
                query = query.filter(Position.opened_at >= start_date)
            if end_date:
                query = query.filter(Position.opened_at <= end_date)
            
            query = query.order_by(Position.opened_at.desc()).limit(limit)
            return query.all()
        finally:
            session.close()
    
    # ========== Strategy Metrics Operations ==========
    
    def save_strategy_metrics(self, strategy_name: str, metrics: Dict[str, Any]) -> Optional[StrategyMetric]:
        """Save strategy performance metrics."""
        session = self.get_session()
        try:
            metric_data = {
                'strategy_name': strategy_name,
                'timestamp': datetime.utcnow(),
                **metrics
            }
            metric = StrategyMetric(**metric_data)
            session.add(metric)
            session.commit()
            session.refresh(metric)
            logger.debug(f"Strategy metrics saved: {strategy_name}")
            return metric
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error saving strategy metrics: {e}")
            return None
        finally:
            session.close()
    
    def get_strategy_metrics(
        self,
        strategy_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[StrategyMetric]:
        """Get strategy metrics history."""
        session = self.get_session()
        try:
            query = session.query(StrategyMetric).filter(
                StrategyMetric.strategy_name == strategy_name
            )
            
            if start_date:
                query = query.filter(StrategyMetric.timestamp >= start_date)
            if end_date:
                query = query.filter(StrategyMetric.timestamp <= end_date)
            
            query = query.order_by(StrategyMetric.timestamp.desc()).limit(limit)
            return query.all()
        finally:
            session.close()
    
    def get_latest_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetric]:
        """Get latest metrics for a strategy."""
        session = self.get_session()
        try:
            return session.query(StrategyMetric).filter(
                StrategyMetric.strategy_name == strategy_name
            ).order_by(StrategyMetric.timestamp.desc()).first()
        finally:
            session.close()
    
    # ========== Risk Metrics Operations ==========
    
    def save_risk_metrics(self, metrics: Dict[str, Any]) -> Optional[RiskMetric]:
        """Save risk metrics snapshot."""
        session = self.get_session()
        try:
            metric_data = {
                'timestamp': datetime.utcnow(),
                **metrics
            }
            metric = RiskMetric(**metric_data)
            session.add(metric)
            session.commit()
            session.refresh(metric)
            logger.debug("Risk metrics saved")
            return metric
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error saving risk metrics: {e}")
            return None
        finally:
            session.close()
    
    def get_risk_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[RiskMetric]:
        """Get risk metrics history."""
        session = self.get_session()
        try:
            query = session.query(RiskMetric)
            
            if start_date:
                query = query.filter(RiskMetric.timestamp >= start_date)
            if end_date:
                query = query.filter(RiskMetric.timestamp <= end_date)
            
            query = query.order_by(RiskMetric.timestamp.desc()).limit(limit)
            return query.all()
        finally:
            session.close()
    
    def get_latest_risk_metrics(self) -> Optional[RiskMetric]:
        """Get latest risk metrics."""
        session = self.get_session()
        try:
            return session.query(RiskMetric).order_by(
                RiskMetric.timestamp.desc()
            ).first()
        finally:
            session.close()
    
    # ========== Performance Snapshot Operations ==========
    
    def save_performance_snapshot(self, snapshot_data: Dict[str, Any]) -> Optional[PerformanceSnapshot]:
        """Save performance snapshot."""
        session = self.get_session()
        try:
            snapshot = PerformanceSnapshot(**snapshot_data)
            session.add(snapshot)
            session.commit()
            session.refresh(snapshot)
            logger.debug("Performance snapshot saved")
            return snapshot
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error saving performance snapshot: {e}")
            return None
        finally:
            session.close()
    
    def get_performance_snapshots(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[PerformanceSnapshot]:
        """Get performance snapshot history."""
        session = self.get_session()
        try:
            query = session.query(PerformanceSnapshot)
            
            if start_date:
                query = query.filter(PerformanceSnapshot.timestamp >= start_date)
            if end_date:
                query = query.filter(PerformanceSnapshot.timestamp <= end_date)
            
            query = query.order_by(PerformanceSnapshot.timestamp.desc()).limit(limit)
            return query.all()
        finally:
            session.close()
    
    # ========== Statistics & Analytics ==========
    
    def get_trade_statistics(
        self,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get trade statistics."""
        session = self.get_session()
        try:
            query = session.query(Trade)
            
            if exchange:
                query = query.filter(Trade.exchange == exchange)
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            if strategy_name:
                query = query.filter(Trade.strategy_name == strategy_name)
            if start_date:
                query = query.filter(Trade.timestamp >= start_date)
            if end_date:
                query = query.filter(Trade.timestamp <= end_date)
            
            trades = query.all()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'total_volume': 0.0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0
                }
            
            total_trades = len(trades)
            total_volume = sum(t.usd_value for t in trades)
            
            # Calculate PnL from positions
            total_pnl = 0.0
            winning_trades = 0
            
            for trade in trades:
                if trade.position_id:
                    pos = session.query(Position).filter(Position.id == trade.position_id).first()
                    if pos and pos.is_closed:
                        if pos.realized_pnl > 0:
                            winning_trades += 1
                        total_pnl += pos.realized_pnl
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            return {
                'total_trades': total_trades,
                'total_volume': round(total_volume, 2),
                'total_pnl': round(total_pnl, 2),
                'win_rate': round(win_rate, 2),
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades
            }
        finally:
            session.close()
    
    def get_database_path(self) -> Path:
        """Get path to database file (for SQLite)."""
        return self.database_dir

