"""
Database API Routes
===================
API endpoints for querying database data (trades, positions, metrics).
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

from gordon.api import get_database_manager

router = APIRouter(prefix="/api/database", tags=["database"])


class TradeResponse(BaseModel):
    """Trade response model."""
    id: int
    trade_id: str
    exchange: str
    symbol: str
    side: str
    order_type: str
    amount: float
    price: float
    fee: float
    usd_value: float
    strategy_name: Optional[str]
    timestamp: datetime


class PositionResponse(BaseModel):
    """Position response model."""
    id: int
    exchange: str
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    usd_value: float
    is_open: bool
    opened_at: datetime


class StrategyMetricsResponse(BaseModel):
    """Strategy metrics response model."""
    id: int
    strategy_name: str
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float


class RiskMetricsResponse(BaseModel):
    """Risk metrics response model."""
    id: int
    timestamp: datetime
    total_balance: float
    total_positions: int
    open_positions: int
    total_pnl: float
    total_pnl_pct: float
    daily_pnl: float
    current_drawdown_pct: float
    risk_score: Optional[float]


@router.get("/trades")
async def get_trades(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_name: Optional[str] = Query(None, description="Filter by strategy"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    db=Depends(get_database_manager)
):
    """Get trades from database."""
    try:
        trades = db.get_trades(
            exchange=exchange,
            symbol=symbol,
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return {
            "count": len(trades),
            "trades": [
                {
                    "id": t.id,
                    "trade_id": t.trade_id,
                    "exchange": t.exchange,
                    "symbol": t.symbol,
                    "side": t.side,
                    "order_type": t.order_type,
                    "amount": t.amount,
                    "price": t.price,
                    "fee": t.fee,
                    "usd_value": t.usd_value,
                    "strategy_name": t.strategy_name,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in trades
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_name: Optional[str] = Query(None, description="Filter by strategy"),
    open_only: bool = Query(True, description="Only return open positions"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    db=Depends(get_database_manager)
):
    """Get positions from database."""
    try:
        if open_only:
            positions = db.get_open_positions(
                exchange=exchange,
                strategy_name=strategy_name
            )
        else:
            positions = db.get_positions(
                exchange=exchange,
                symbol=symbol,
                strategy_name=strategy_name,
                limit=limit
            )
        
        return {
            "count": len(positions),
            "positions": [
                {
                    "id": p.id,
                    "exchange": p.exchange,
                    "symbol": p.symbol,
                    "side": p.side,
                    "size": p.size,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    "usd_value": p.usd_value,
                    "is_open": p.is_open,
                    "opened_at": p.opened_at.isoformat()
                }
                for p in positions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_name}/metrics")
async def get_strategy_metrics(
    strategy_name: str,
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    db=Depends(get_database_manager)
):
    """Get strategy metrics history."""
    try:
        metrics = db.get_strategy_metrics(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return {
            "strategy_name": strategy_name,
            "count": len(metrics),
            "metrics": [
                {
                    "id": m.id,
                    "timestamp": m.timestamp.isoformat(),
                    "total_trades": m.total_trades,
                    "winning_trades": m.winning_trades,
                    "losing_trades": m.losing_trades,
                    "win_rate": m.win_rate,
                    "total_pnl": m.total_pnl,
                    "total_pnl_pct": m.total_pnl_pct,
                    "sharpe_ratio": m.sharpe_ratio,
                    "max_drawdown": m.max_drawdown
                }
                for m in metrics
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_name}/metrics/latest")
async def get_latest_strategy_metrics(
    strategy_name: str,
    db=Depends(get_database_manager)
):
    """Get latest strategy metrics."""
    try:
        metric = db.get_latest_strategy_metrics(strategy_name)
        
        if not metric:
            raise HTTPException(status_code=404, detail="No metrics found")
        
        return {
            "strategy_name": strategy_name,
            "timestamp": metric.timestamp.isoformat(),
            "total_trades": metric.total_trades,
            "winning_trades": metric.winning_trades,
            "losing_trades": metric.losing_trades,
            "win_rate": metric.win_rate,
            "total_pnl": metric.total_pnl,
            "total_pnl_pct": metric.total_pnl_pct,
            "sharpe_ratio": metric.sharpe_ratio,
            "max_drawdown": metric.max_drawdown,
            "status": metric.status
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/metrics")
async def get_risk_metrics(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    db=Depends(get_database_manager)
):
    """Get risk metrics history."""
    try:
        metrics = db.get_risk_metrics(
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        return {
            "count": len(metrics),
            "metrics": [
                {
                    "id": m.id,
                    "timestamp": m.timestamp.isoformat(),
                    "total_balance": m.total_balance,
                    "total_positions": m.total_positions,
                    "open_positions": m.open_positions,
                    "total_pnl": m.total_pnl,
                    "total_pnl_pct": m.total_pnl_pct,
                    "daily_pnl": m.daily_pnl,
                    "current_drawdown_pct": m.current_drawdown_pct,
                    "risk_score": m.risk_score
                }
                for m in metrics
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/metrics/latest")
async def get_latest_risk_metrics(
    db=Depends(get_database_manager)
):
    """Get latest risk metrics."""
    try:
        metric = db.get_latest_risk_metrics()
        
        if not metric:
            raise HTTPException(status_code=404, detail="No metrics found")
        
        return {
            "timestamp": metric.timestamp.isoformat(),
            "total_balance": metric.total_balance,
            "total_positions": metric.total_positions,
            "open_positions": metric.open_positions,
            "total_pnl": metric.total_pnl,
            "total_pnl_pct": metric.total_pnl_pct,
            "daily_pnl": metric.daily_pnl,
            "current_drawdown_pct": metric.current_drawdown_pct,
            "risk_score": metric.risk_score
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/trades")
async def get_trade_statistics(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    strategy_name: Optional[str] = Query(None, description="Filter by strategy"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    db=Depends(get_database_manager)
):
    """Get trade statistics."""
    try:
        stats = db.get_trade_statistics(
            exchange=exchange,
            symbol=symbol,
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date
        )
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

