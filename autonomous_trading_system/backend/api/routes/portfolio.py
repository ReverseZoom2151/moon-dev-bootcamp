"""
Portfolio API Routes - Portfolio management and monitoring
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from services.portfolio_manager import PortfolioManager
from services.risk_manager import RiskManager

router = APIRouter()

async def get_portfolio_manager() -> PortfolioManager:
    """Dependency to get portfolio manager"""
    pass


async def get_risk_manager() -> RiskManager:
    """Dependency to get risk manager"""
    pass


@router.get("/summary")
async def get_portfolio_summary(portfolio: PortfolioManager = Depends(get_portfolio_manager)):
    """Get portfolio summary"""
    try:
        return await portfolio.get_portfolio_summary()
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


@router.get("/performance")
async def get_portfolio_performance(
    portfolio: PortfolioManager = Depends(get_portfolio_manager),
    risk: RiskManager = Depends(get_risk_manager)
):
    """Get portfolio performance metrics"""
    try:
        portfolio_summary = await portfolio.get_portfolio_summary()
        risk_metrics = await risk.get_risk_metrics()
        
        return {
            "portfolio": portfolio_summary,
            "risk_metrics": {
                "max_drawdown": risk_metrics.max_drawdown,
                "current_drawdown": risk_metrics.current_drawdown,
                "var_95": risk_metrics.var_95,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "volatility": risk_metrics.volatility,
                "risk_score": risk_metrics.risk_score
            },
            "daily_pnl": await portfolio.get_daily_pnl(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk")
async def get_risk_status(risk: RiskManager = Depends(get_risk_manager)):
    """Get risk management status"""
    try:
        return await risk.get_risk_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/emergency-stop")
async def trigger_emergency_stop(risk: RiskManager = Depends(get_risk_manager)):
    """Trigger emergency stop"""
    try:
        await risk.emergency_stop()
        return {"message": "Emergency stop triggered", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/resume")
async def resume_trading(risk: RiskManager = Depends(get_risk_manager)):
    """Resume trading after emergency stop"""
    try:
        await risk.resume_trading()
        return {"message": "Trading resumed", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 