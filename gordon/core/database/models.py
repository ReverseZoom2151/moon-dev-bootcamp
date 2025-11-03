"""
Database Models
===============
SQLAlchemy ORM models for trades, positions, and performance metrics.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text, JSON, ForeignKey, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Trade(Base):
    """Trade execution record."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), unique=True, nullable=False, index=True)  # Exchange trade ID
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    order_type = Column(String(20))  # 'market', 'limit', etc.
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    fee_currency = Column(String(10), default='USD')
    usd_value = Column(Float, nullable=False)
    
    # Position tracking
    position_id = Column(Integer, ForeignKey('positions.id'), nullable=True, index=True)
    
    # Strategy and signal tracking
    strategy_name = Column(String(100), nullable=True, index=True)
    signal_id = Column(String(100), nullable=True)
    confidence = Column(Float, default=0.0)  # 0-1
    
    # Metadata
    metadata_json = Column(JSON, default={})  # Additional trade data
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    position = relationship("Position", back_populates="trades")
    
    __table_args__ = (
        Index('idx_trades_exchange_symbol', 'exchange', 'symbol'),
        Index('idx_trades_timestamp', 'timestamp'),
        Index('idx_trades_strategy', 'strategy_name'),
    )


class Position(Base):
    """Position tracking record."""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    exchange = Column(String(50), nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    
    # Position details
    side = Column(String(10), nullable=False)  # 'long' or 'short'
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    leverage = Column(Integer, default=1)
    
    # PnL tracking
    unrealized_pnl = Column(Float, default=0.0)
    unrealized_pnl_pct = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    realized_pnl_pct = Column(Float, default=0.0)
    
    # Position value
    usd_value = Column(Float, nullable=False)
    margin_used = Column(Float, default=0.0)
    
    # Status
    is_open = Column(Boolean, default=True, index=True)
    is_closed = Column(Boolean, default=False, index=True)
    
    # Strategy tracking
    strategy_name = Column(String(100), nullable=True, index=True)
    
    # Stop loss / Take profit
    stop_loss_price = Column(Float, nullable=True)
    take_profit_price = Column(Float, nullable=True)
    stop_loss_pct = Column(Float, nullable=True)
    take_profit_pct = Column(Float, nullable=True)
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Metadata
    metadata_json = Column(JSON, default={})
    
    # Relationships
    trades = relationship("Trade", back_populates="position")
    
    __table_args__ = (
        Index('idx_positions_exchange_symbol', 'exchange', 'symbol'),
        Index('idx_positions_open', 'is_open'),
        Index('idx_positions_strategy', 'strategy_name'),
    )


class StrategyMetric(Base):
    """Strategy performance metrics snapshot."""
    __tablename__ = 'strategy_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Trade metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    # PnL metrics
    total_pnl = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    
    # Execution metrics
    total_signals = Column(Integer, default=0)
    signals_executed = Column(Integer, default=0)
    execution_rate = Column(Float, default=0.0)
    
    # Error tracking
    error_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    
    # Status
    status = Column(String(20), default='stopped')  # 'stopped', 'running', 'paused', 'error'
    
    # Metadata
    metadata_json = Column(JSON, default={})
    
    __table_args__ = (
        Index('idx_strategy_metrics_name_timestamp', 'strategy_name', 'timestamp'),
    )


class RiskMetric(Base):
    """Risk metrics snapshot."""
    __tablename__ = 'risk_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Balance tracking
    total_balance = Column(Float, nullable=False)
    available_balance = Column(Float, nullable=False)
    used_margin = Column(Float, default=0.0)
    
    # Position metrics
    total_positions = Column(Integer, default=0)
    open_positions = Column(Integer, default=0)
    total_position_value = Column(Float, default=0.0)
    
    # PnL metrics
    total_pnl = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    daily_pnl = Column(Float, default=0.0)
    daily_pnl_pct = Column(Float, default=0.0)
    
    # Risk metrics
    var_95 = Column(Float, nullable=True)  # Value at Risk (95%)
    sharpe_ratio = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    
    # Drawdown tracking
    current_drawdown = Column(Float, default=0.0)
    current_drawdown_pct = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    peak_balance = Column(Float, nullable=True)
    
    # Limits tracking
    daily_loss_limit = Column(Float, nullable=True)
    daily_loss_used = Column(Float, default=0.0)
    daily_loss_remaining = Column(Float, nullable=True)
    max_drawdown_limit = Column(Float, nullable=True)
    max_drawdown_exceeded = Column(Boolean, default=False)
    
    # Exchange breakdown (JSON)
    exchange_breakdown = Column(JSON, default={})
    
    # Metadata
    metadata_json = Column(JSON, default={})
    
    __table_args__ = (
        Index('idx_risk_metrics_timestamp', 'timestamp'),
    )


class PerformanceSnapshot(Base):
    """Overall performance snapshot."""
    __tablename__ = 'performance_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Portfolio metrics
    total_balance = Column(Float, nullable=False)
    total_pnl = Column(Float, default=0.0)
    total_pnl_pct = Column(Float, default=0.0)
    
    # Strategy performance summary (JSON)
    strategy_summary = Column(JSON, default={})  # {strategy_name: {metrics...}}
    
    # Risk summary
    risk_summary = Column(JSON, default={})
    
    # Position summary
    position_summary = Column(JSON, default={})  # Counts, values, etc.
    
    # Metadata
    metadata_json = Column(JSON, default={})
    
    __table_args__ = (
        Index('idx_performance_snapshots_timestamp', 'timestamp'),
    )

