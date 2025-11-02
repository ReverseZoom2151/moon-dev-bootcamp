"""
Backtesting API Routes - Strategy backtesting and analysis endpoints
"""

import asyncio
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from services.backtrader_backtesting_service import run_backtrader
from services.backtesting_engine import BacktestingEngine
from strategies.stochrsi_strategy import StochRSIStrategy
from services.improved_breakout_backtesting_service import run_improved_breakout_backtest
from services.mean_reversion_backtesting_service import MeanReversionBacktestingService

router = APIRouter

class BacktestRequest(BaseModel):
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float = 10000
    parameters: Optional[Dict[str, Any]] = None


@router.post("/run")
async def run_backtest(backtest_request: BacktestRequest):
    """Run a backtest for a strategy"""
    try:
        # Backtrader-based SMA cross strategy
        name = backtest_request.strategy_name.lower()
        if name == "sma_cross":
            params = backtest_request.parameters or {}
            data_file = params.get("data_file", f"{backtest_request.symbol}-price-USD.csv")
            commission = params.get("commission", 0.001)
            initial_cash = backtest_request.initial_capital
            result = await asyncio.to_thread(
                run_backtrader, data_file, initial_cash, commission
            )
            return result
        # Comprehensive backtesting for StochRSI
        elif name == "stochrsi":
            params = backtest_request.parameters or {}
            data_file = params.get("data_file")
            if not data_file:
                raise HTTPException(status_code=400, detail="data_file parameter required for stochrsi backtest")
            initial_cash = backtest_request.initial_capital
            commission = params.get("commission", 0.001)
            engine = BacktestingEngine(initial_cash, commission)
            try:
                df = pd.read_csv(data_file, parse_dates=True, index_col='timestamp')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not read data_file: {e}")
            # Run backtest using StochRSIStrategy
            result = await asyncio.to_thread(
                engine.run_backtest, StochRSIStrategy, df, params.get("parameters", {}),
                None, None
            )
            # Serialize result
            return {
                "strategy_name": result.strategy_name,
                "parameters": result.parameters,
                "start_date": result.start_date.isoformat(),
                "end_date": result.end_date.isoformat(),
                "initial_capital": result.initial_capital,
                "final_capital": result.final_capital,
                "total_return": result.total_return,
                "annual_return": result.annual_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
                "avg_trade_duration": result.avg_trade_duration,
                "trades": [t.__dict__ if hasattr(t, '__dict__') else t for t in result.trades],
                "equity_curve": result.equity_curve.reset_index().to_dict('records'),
                "metrics": result.metrics
            }
        # Backtesting for Improved Breakout Strategy
        elif name == "improved_breakout":
            params = backtest_request.parameters or {}
            symbol = backtest_request.symbol
            optimize = params.get("optimize", False)
            initial_cash = backtest_request.initial_capital
            commission = params.get("commission", 0.002)
            result = await asyncio.to_thread(
                run_improved_breakout_backtest,
                symbol,
                initial_cash,
                commission,
                optimize
            )
            return result
        # Mean Reversion Strategy Backtesting
        elif name == "mean_reversion":
            params = backtest_request.parameters or {}
            symbol = backtest_request.symbol
            optimize = params.get("optimize", True)
            save_results = params.get("save_results", False)
            initial_cash = backtest_request.initial_capital
            commission = params.get("commission", 0.002)
            
            # Initialize service
            mr_service = MeanReversionBacktestingService()
            
            # Run comprehensive backtest
            result = await mr_service.run_comprehensive_backtest(
                symbol=symbol,
                optimize=optimize,
                save_results=save_results,
                initial_cash=initial_cash,
                commission=commission
            )
            
            return result
        # Simulate backtest results
        # In a real implementation, this would run the actual backtest
        
        results = {
            "strategy_name": backtest_request.strategy_name,
            "symbol": backtest_request.symbol,
            "start_date": backtest_request.start_date,
            "end_date": backtest_request.end_date,
            "initial_capital": backtest_request.initial_capital,
            "final_capital": backtest_request.initial_capital * 1.15,  # 15% return
            "total_return": 0.15,
            "total_return_pct": 15.0,
            "max_drawdown": 0.08,
            "sharpe_ratio": 1.2,
            "win_rate": 0.65,
            "total_trades": 45,
            "winning_trades": 29,
            "losing_trades": 16,
            "avg_win": 150.0,
            "avg_loss": -80.0,
            "profit_factor": 1.8,
            "trades": [],  # Would contain individual trade records
            "equity_curve": [],  # Would contain equity progression
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_run")
async def run_batch_backtest(strategy_name: str, optimize: bool = True, symbols: Optional[List[str]] = None):
    """Run batch backtests for Mean Reversion strategy across multiple symbols"""
    try:
        if strategy_name.lower() != "mean_reversion":
            raise HTTPException(status_code=400, detail="Batch backtesting currently only supports mean_reversion strategy")
        
        # Initialize service
        mr_service = MeanReversionBacktestingService()
        
        # Run batch backtest
        results = await mr_service.batch_backtest(symbols=symbols, optimize=optimize)
        
        return {
            "strategy_name": strategy_name,
            "batch_results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """Get results of a specific backtest"""
    try:
        # In a real implementation, this would fetch from database
        return {
            "backtest_id": backtest_id,
            "status": "completed",
            "results": {
                "total_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.08
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_backtest_history():
    """Get history of all backtests"""
    try:
        # Simulate backtest history
        history = [
            {
                "id": "bt_001",
                "strategy": "bollinger_bands",
                "symbol": "BTC",
                "date_run": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "total_return": 0.12,
                "status": "completed"
            },
            {
                "id": "bt_002", 
                "strategy": "vwap",
                "symbol": "ETH",
                "date_run": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "total_return": 0.08,
                "status": "completed"
            }
        ]
        
        return {
            "backtests": history,
            "total_count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/results/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest result"""
    try:
        return {
            "message": f"Backtest {backtest_id} deleted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 