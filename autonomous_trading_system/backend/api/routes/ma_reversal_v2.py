"""
API Endpoints for the MA Reversal Strategy V2
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from strategies.ma_reversal_strategy_v2 import MAReversalStrategyV2
from services.backtesting_engine import BacktestingEngine
from core.config import get_settings
from data.market_data_manager import MarketDataManager

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

class BacktestRequest(BaseModel):
    symbol: str = settings.MA_REVERSAL_V2_SYMBOL
    cash: int = 1_000_000
    commission: float = 0.002
    parameters: Dict[str, Any] = {
        'ma_fast': settings.MA_REVERSAL_V2_FAST_MA,
        'ma_slow': settings.MA_REVERSAL_V2_SLOW_MA,
        'take_profit': settings.MA_REVERSAL_V2_TAKE_PROFIT,
        'stop_loss': settings.MA_REVERSAL_V2_STOP_LOSS,
    }

class OptimizationRequest(BaseModel):
    symbol: str = settings.MA_REVERSAL_V2_SYMBOL
    cash: int = 1_000_000
    commission: float = 0.002

@router.post("/ma-reversal-v2/backtest", tags=["MA Reversal Strategy V2"])
async def run_ma_reversal_backtest(
    request: BacktestRequest,
    backtesting_engine: BacktestingEngine = Depends(),
    market_data: MarketDataManager = Depends()
):
    """Run a single backtest for the MA Reversal V2 strategy."""
    try:
        data_df = await market_data.get_data(request.symbol, "1d", "3650d")
        if data_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")

        stats = backtesting_engine.run_backtest(
            strategy=MAReversalStrategyV2,
            data=data_df,
            cash=request.cash,
            commission=request.commission,
            parameters=request.parameters
        )
        return stats.to_dict()
    except Exception as e:
        logger.error(f"Error during MA Reversal V2 backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ma-reversal-v2/backtest-optimized", tags=["MA Reversal Strategy V2"])
async def run_ma_reversal_backtest_optimized(
    request: BacktestRequest,
    backtesting_engine: BacktestingEngine = Depends(),
    market_data: MarketDataManager = Depends()
):
    """Run a backtest using the pre-defined optimized parameters."""
    try:
        data_df = await market_data.get_data(request.symbol, "1d", "3650d")
        if data_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        optimized_params = {
            'ma_fast': settings.MA_REVERSAL_V2_OPTIMIZED_FAST_MA,
            'ma_slow': settings.MA_REVERSAL_V2_OPTIMIZED_SLOW_MA,
            'take_profit': settings.MA_REVERSAL_V2_OPTIMIZED_TAKE_PROFIT,
            'stop_loss': settings.MA_REVERSAL_V2_OPTIMIZED_STOP_LOSS,
        }

        stats = backtesting_engine.run_backtest(
            strategy=MAReversalStrategyV2,
            data=data_df,
            cash=request.cash,
            commission=request.commission,
            parameters=optimized_params
        )
        return stats.to_dict()
    except Exception as e:
        logger.error(f"Error during optimized MA Reversal V2 backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ma-reversal-v2/optimize", tags=["MA Reversal Strategy V2"])
async def run_ma_reversal_optimization(
    request: OptimizationRequest,
    backtesting_engine: BacktestingEngine = Depends(),
    market_data: MarketDataManager = Depends()
):
    """Run parameter optimization for the MA Reversal V2 strategy."""
    try:
        data_df = await market_data.get_data(request.symbol, "1d", "3650d")
        if data_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")

        optimization_params = {
            'ma_fast': range(10, 31, 2),
            'ma_slow': range(30, 61, 2),
            'take_profit': [i/100 for i in range(1, 11)],
            'stop_loss': [i/100 for i in range(1, 11)],
            'constraint': lambda p: p.ma_fast < p.ma_slow
        }

        results = backtesting_engine.optimize(
            strategy=MAReversalStrategyV2,
            data=data_df,
            cash=request.cash,
            commission=request.commission,
            maximize='Equity Final [$]',
            **optimization_params
        )
        return results._asdict()
    except Exception as e:
        logger.error(f"Error during MA Reversal V2 optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))
