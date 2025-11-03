"""
API Routes for Strategy Management
==================================
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging

from .. import get_strategy_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


class StrategyConfig(BaseModel):
    """Strategy configuration model."""
    enabled: bool = True
    config: Optional[Dict[str, Any]] = None


@router.get("/")
async def list_strategies(strategy_engine=Depends(get_strategy_engine)):
    """List all available strategies."""
    try:
        strategies = strategy_engine.strategies
        return {
            "strategies": [
                {
                    "name": name,
                    "enabled": name in strategy_engine.active_strategies,
                    "status": "running" if name in strategy_engine.active_strategies else "stopped"
                }
                for name in strategies.keys()
            ],
            "total": len(strategies)
        }
    except Exception as e:
        logger.error(f"Failed to list strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{strategy_name}")
async def get_strategy(
    strategy_name: str,
    strategy_engine=Depends(get_strategy_engine)
):
    """Get strategy details."""
    try:
        if strategy_name not in strategy_engine.strategies:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
        
        strategy = strategy_engine.strategies[strategy_name]
        status = "running" if strategy_name in strategy_engine.active_strategies else "stopped"
        
        return {
            "name": strategy_name,
            "status": status,
            "enabled": strategy_name in strategy_engine.active_strategies,
            "config": strategy.config if hasattr(strategy, 'config') else {}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_name}/start")
async def start_strategy(
    strategy_name: str,
    config: Optional[StrategyConfig] = None,
    strategy_engine=Depends(get_strategy_engine)
):
    """Start a strategy."""
    try:
        await strategy_engine.start_strategy(
            strategy_name,
            config.config if config else None
        )
        return {
            "success": True,
            "message": f"Strategy {strategy_name} started",
            "strategy": strategy_name
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_name}/stop")
async def stop_strategy(
    strategy_name: str,
    strategy_engine=Depends(get_strategy_engine)
):
    """Stop a strategy."""
    try:
        await strategy_engine.stop_strategy(strategy_name)
        return {
            "success": True,
            "message": f"Strategy {strategy_name} stopped",
            "strategy": strategy_name
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

