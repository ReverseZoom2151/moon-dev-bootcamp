"""
Trading API Routes - Manual trading operations and trade management
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from services.strategy_engine import StrategyEngine
from services.portfolio_manager import PortfolioManager
from services.risk_manager import RiskManager

router = APIRouter()

class ManualTradeRequest(BaseModel):
    symbol: str
    action: str  # "BUY" or "SELL"
    size: float
    price: Optional[float] = None  # Market order if None


class ClosePositionRequest(BaseModel):
    symbol: str


async def get_strategy_engine() -> StrategyEngine:
    """Dependency to get strategy engine"""
    pass


async def get_portfolio_manager() -> PortfolioManager:
    """Dependency to get portfolio manager"""
    pass


async def get_risk_manager() -> RiskManager:
    """Dependency to get risk manager"""
    pass


@router.post("/manual-trade")
async def execute_manual_trade(
    trade_request: ManualTradeRequest,
    portfolio: PortfolioManager = Depends(get_portfolio_manager),
    risk: RiskManager = Depends(get_risk_manager)
):
    """Execute a manual trade"""
    try:
        # Create a mock signal for manual trade
        from strategies.base_strategy import StrategySignal, SignalAction
        
        action = SignalAction.BUY if trade_request.action.upper() == "BUY" else SignalAction.SELL
        
        signal = StrategySignal(
            symbol=trade_request.symbol,
            action=action,
            price=trade_request.price or 0.0,  # Will be filled with market price
            confidence=1.0,  # Manual trades have full confidence
            timestamp=datetime.utcnow(),
            strategy_name="manual",
            metadata={"manual_trade": True}
        )
        
        # Check risk approval
        if not await risk.approve_trade(signal):
            raise HTTPException(status_code=400, detail="Trade rejected by risk manager")
        
        # Execute the trade
        result = await portfolio.execute_trade(signal, trade_request.size)
        
        if not result.success:
            raise HTTPException(status_code=400, detail=f"Trade execution failed: {result.error}")
        
        return {
            "success": True,
            "trade_id": result.trade_id,
            "executed_price": result.executed_price,
            "executed_size": result.executed_size,
            "message": f"Manual trade executed: {trade_request.action} {trade_request.size} {trade_request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-position")
async def close_position(
    close_request: ClosePositionRequest,
    portfolio: PortfolioManager = Depends(get_portfolio_manager)
):
    """Close a specific position"""
    try:
        success = await portfolio.close_position(close_request.symbol)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to close position for {close_request.symbol}")
        
        return {
            "success": True,
            "message": f"Position closed for {close_request.symbol}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-all-positions")
async def close_all_positions(portfolio: PortfolioManager = Depends(get_portfolio_manager)):
    """Close all open positions"""
    try:
        await portfolio.close_all_positions()
        return {
            "success": True,
            "message": "All positions closed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel-all-orders")
async def cancel_all_orders(portfolio: PortfolioManager = Depends(get_portfolio_manager)):
    """Cancel all pending orders"""
    try:
        await portfolio.cancel_all_orders()
        return {
            "success": True,
            "message": "All orders cancelled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions(portfolio: PortfolioManager = Depends(get_portfolio_manager)):
    """Get all open positions"""
    try:
        positions = await portfolio.get_open_positions()
        return {
            "positions": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "pnl": pos.pnl,
                    "pnl_pct": pos.pnl_pct,
                    "timestamp": pos.timestamp.isoformat(),
                    "strategy": pos.strategy
                }
                for pos in positions
            ],
            "total_positions": len(positions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
async def get_trades(
    strategy: Optional[str] = None,
    portfolio: PortfolioManager = Depends(get_portfolio_manager)
):
    """Get trade history"""
    try:
        if strategy:
            trades = await portfolio.get_strategy_trades(strategy)
        else:
            trades = portfolio.trades
        
        return {
            "trades": [
                {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "size": trade.size,
                    "price": trade.price,
                    "timestamp": trade.timestamp.isoformat(),
                    "strategy": trade.strategy,
                    "pnl": trade.pnl,
                    "fees": trade.fees
                }
                for trade in trades
            ],
            "total_trades": len(trades)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_trading_status(
    engine: StrategyEngine = Depends(get_strategy_engine),
    portfolio: PortfolioManager = Depends(get_portfolio_manager),
    risk: RiskManager = Depends(get_risk_manager)
):
    """Get overall trading status"""
    try:
        return {
            "engine_status": engine.status.value,
            "engine_running": engine.is_running,
            "active_strategies": len(engine.active_strategies),
            "portfolio_value": await portfolio.get_total_value(),
            "open_positions": len(await portfolio.get_open_positions()),
            "risk_score": await risk.get_risk_score(),
            "risk_active": risk.is_active,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 