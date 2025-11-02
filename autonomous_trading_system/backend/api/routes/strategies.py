"""
Strategies API Routes - Manage and monitor trading strategies
"""

from fastapi import APIRouter, HTTPException, Depends
from services.strategy_engine import StrategyEngine

router = APIRouter()

async def get_strategy_engine() -> StrategyEngine:
    """Dependency to get strategy engine"""
    # This will be injected by the main app
    pass


@router.get("/")
async def list_strategies(engine: StrategyEngine = Depends(get_strategy_engine)):
    """Get list of all available strategies"""
    try:
        strategies = await engine.get_all_strategies_status()
        return {
            "strategies": strategies,
            "total_count": len(strategies),
            "active_count": len([s for s in strategies.values() if s.get("enabled", False)])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_name}")
async def get_strategy(strategy_name: str, engine: StrategyEngine = Depends(get_strategy_engine)):
    """Get detailed information about a specific strategy"""
    try:
        strategy_info = await engine.get_strategy_status(strategy_name)
        if not strategy_info:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")
        return strategy_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_name}/enable")
async def enable_strategy(strategy_name: str, engine: StrategyEngine = Depends(get_strategy_engine)):
    """Enable a strategy"""
    try:
        success = await engine.enable_strategy(strategy_name)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to enable strategy '{strategy_name}'")
        return {"message": f"Strategy '{strategy_name}' enabled successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_name}/disable")
async def disable_strategy(strategy_name: str, engine: StrategyEngine = Depends(get_strategy_engine)):
    """Disable a strategy"""
    try:
        success = await engine.disable_strategy(strategy_name)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to disable strategy '{strategy_name}'")
        return {"message": f"Strategy '{strategy_name}' disabled successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_name}/performance")
async def get_strategy_performance(strategy_name: str, engine: StrategyEngine = Depends(get_strategy_engine)):
    """Get performance metrics for a strategy"""
    try:
        strategy_info = await engine.get_strategy_status(strategy_name)
        if not strategy_info:
            raise HTTPException(status_code=404, detail=f"Strategy '{strategy_name}' not found")
        
        return {
            "strategy_name": strategy_name,
            "performance_metrics": strategy_info.get("performance_metrics", {}),
            "last_signal": strategy_info.get("last_signal"),
            "last_execution": strategy_info.get("last_execution")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types/available")
async def get_available_strategy_types():
    """Get list of available strategy types"""
    return {
        "strategy_types": [
            {
                "name": "mean_reversion",
                "display_name": "Mean Reversion",
                "description": "Trades when price deviates significantly from moving average",
                "source": "Day_20_Projects"
            },
            {
                "name": "bollinger_bands",
                "display_name": "Bollinger Bands",
                "description": "Trades on band compression indicating potential breakouts",
                "source": "Day_10_Projects"
            },
            {
                "name": "supply_demand",
                "display_name": "Supply/Demand Zones",
                "description": "Identifies and trades based on supply and demand zones",
                "source": "Day_11_Projects"
            },
            {
                "name": "vwap",
                "display_name": "VWAP Strategy",
                "description": "Trades based on VWAP deviations with probabilistic direction",
                "source": "Day_12_Projects"
            },
            {
                "name": "stochrsi",
                "display_name": "Stochastic RSI",
                "description": "Trades based on StochRSI momentum signals and crossovers",
                "source": "Day_16_Projects"
            },
            {
                "name": "liquidation",
                "display_name": "Liquidation Strategy",
                "description": "Trades based on liquidation events and volume spikes",
                "source": "Day_21_Projects"
            }
        ]
    } 