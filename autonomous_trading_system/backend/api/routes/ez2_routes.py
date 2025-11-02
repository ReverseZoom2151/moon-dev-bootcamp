"""
EZ2 Trading API Routes

API endpoints for executing all trading modes from ez_2.py:
- Mode 0: Close position (in chunks)
- Mode 1: Open buying position (in chunks)
- Mode 2: ETH SMA based strategy
- Mode 4: Close positions based on PnL thresholds
- Mode 5: Simple market making (buy under/sell over)
"""

from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from services.ez2_service import EZ2SolanaTradingService

router = APIRouter()

# Pydantic models for request/response
class EZ2ExecuteRequest(BaseModel):
    mode: int
    symbol: Optional[str] = None
    target_usd_size: Optional[float] = None
    duration_minutes: Optional[int] = None

class EZ2StatusResponse(BaseModel):
    service_name: str
    status: str
    market_maker_status: str
    available_modes: Dict[str, str]
    configuration: Dict[str, Any]

class EZ2TradeResponse(BaseModel):
    success: bool
    mode: int
    symbol: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    final_position_usd: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

# Global service instance
ez2_service: Optional[EZ2SolanaTradingService] = None

async def get_ez2_service() -> EZ2SolanaTradingService:
    """Get or create EZ2 service instance"""
    global ez2_service
    if not ez2_service:
        ez2_service = EZ2SolanaTradingService()
        await ez2_service.start()
    return ez2_service

@router.get("/ez2/status", response_model=EZ2StatusResponse)
async def get_ez2_status():
    """Get EZ2 trading service status"""
    try:
        service = await get_ez2_service()
        status = await service.get_service_status()
        return EZ2StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting EZ2 status: {str(e)}")

@router.post("/ez2/execute", response_model=EZ2TradeResponse)
async def execute_ez2_mode(request: EZ2ExecuteRequest):
    """Execute a specific EZ2 trading mode"""
    try:
        service = await get_ez2_service()
        
        # Validate mode
        if request.mode not in [0, 1, 2, 4, 5]:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}. Valid modes are 0, 1, 2, 4, 5")
        
        # Execute the mode
        kwargs = {}
        if request.target_usd_size:
            kwargs['target_usd_size'] = request.target_usd_size
        if request.duration_minutes:
            kwargs['duration_minutes'] = request.duration_minutes
        
        result = await service.execute_mode(
            mode=request.mode,
            symbol=request.symbol,
            **kwargs
        )
        
        return EZ2TradeResponse(
            success=result.success,
            mode=result.mode,
            symbol=result.symbol,
            message=result.message,
            error=result.error,
            final_position_usd=result.final_position_usd,
            start_time=result.start_time,
            end_time=result.end_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing EZ2 mode {request.mode}: {str(e)}")

@router.post("/ez2/close-position", response_model=EZ2TradeResponse)
async def close_position(
    symbol: str = Body(..., description="Token mint address to close position for")
):
    """Close position for specified symbol (Mode 0)"""
    try:
        service = await get_ez2_service()
        result = await service.execute_mode(mode=0, symbol=symbol)
        
        return EZ2TradeResponse(
            success=result.success,
            mode=result.mode,
            symbol=result.symbol,
            message=result.message,
            error=result.error,
            final_position_usd=result.final_position_usd,
            start_time=result.start_time,
            end_time=result.end_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error closing position: {str(e)}")

@router.post("/ez2/buy-position", response_model=EZ2TradeResponse)
async def buy_position(
    symbol: str = Body(..., description="Token mint address to buy"),
    target_usd_size: Optional[float] = Body(None, description="Target position size in USD")
):
    """Open buying position for specified symbol (Mode 1)"""
    try:
        service = await get_ez2_service()
        result = await service.execute_mode(
            mode=1, 
            symbol=symbol, 
            target_usd_size=target_usd_size
        )
        
        return EZ2TradeResponse(
            success=result.success,
            mode=result.mode,
            symbol=result.symbol,
            message=result.message,
            error=result.error,
            final_position_usd=result.final_position_usd,
            start_time=result.start_time,
            end_time=result.end_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error buying position: {str(e)}")

@router.post("/ez2/pnl-close", response_model=EZ2TradeResponse)
async def pnl_close():
    """Close positions based on PnL thresholds (Mode 4)"""
    try:
        service = await get_ez2_service()
        result = await service.execute_mode(mode=4)
        
        return EZ2TradeResponse(
            success=result.success,
            mode=result.mode,
            symbol=result.symbol,
            message=result.message,
            error=result.error,
            final_position_usd=result.final_position_usd,
            start_time=result.start_time,
            end_time=result.end_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing PnL close: {str(e)}")

@router.post("/ez2/market-maker/start")
async def start_market_maker(
    symbol: str = Body(..., description="Token mint address for market making"),
    duration_minutes: Optional[int] = Body(None, description="Duration in minutes (optional)")
):
    """Start market maker mode for specified symbol (Mode 5)"""
    try:
        service = await get_ez2_service()
        
        if duration_minutes:
            # Run for specific duration
            result = await service.execute_mode(
                mode=5, 
                symbol=symbol, 
                duration_minutes=duration_minutes
            )
            return {
                "success": result.success,
                "message": result.message,
                "error": result.error
            }
        else:
            # Start continuous market maker
            message = await service.start_continuous_market_maker(symbol)
            return {
                "success": True,
                "message": message
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting market maker: {str(e)}")

@router.post("/ez2/market-maker/stop")
async def stop_market_maker():
    """Stop the active market maker"""
    try:
        service = await get_ez2_service()
        message = await service.stop_market_maker()
        return {
            "success": True,
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping market maker: {str(e)}")

@router.get("/ez2/modes")
async def get_available_modes():
    """Get list of available trading modes"""
    return {
        "modes": {
            "0": {
                "name": "Close Position",
                "description": "Close position for specified symbol in chunks",
                "parameters": ["symbol"]
            },
            "1": {
                "name": "Buy Position", 
                "description": "Open buying position up to target size",
                "parameters": ["symbol", "target_usd_size (optional)"]
            },
            "2": {
                "name": "ETH SMA Strategy",
                "description": "Execute trades based on ETH SMA signals",
                "parameters": [],
                "status": "Not implemented - requires ETH data source"
            },
            "4": {
                "name": "PnL Close",
                "description": "Close positions based on portfolio PnL thresholds",
                "parameters": []
            },
            "5": {
                "name": "Market Maker",
                "description": "Simple market making with buy under/sell over logic",
                "parameters": ["symbol", "duration_minutes (optional)"]
            }
        }
    }

@router.get("/ez2/config")
async def get_ez2_config():
    """Get current EZ2 trading configuration"""
    try:
        service = await get_ez2_service()
        status = await service.get_service_status()
        return {
            "configuration": status.get("configuration", {}),
            "primary_symbol": status.get("configuration", {}).get("primary_symbol"),
            "trading_parameters": {
                "usd_size": status.get("configuration", {}).get("usd_size"),
                "max_usd_order_size": status.get("configuration", {}).get("max_usd_order_size"),
                "slippage": status.get("configuration", {}).get("slippage"),
                "orders_per_open": status.get("configuration", {}).get("orders_per_open")
            },
            "thresholds": {
                "lowest_balance": status.get("configuration", {}).get("lowest_balance"),
                "target_balance": status.get("configuration", {}).get("target_balance"),
                "buy_under": status.get("configuration", {}).get("buy_under"),
                "sell_over": status.get("configuration", {}).get("sell_over")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")

@router.post("/ez2/start")
async def start_ez2_service():
    """Start the EZ2 trading service"""
    try:
        service = await get_ez2_service()
        return {
            "success": True,
            "message": "EZ2 trading service started",
            "status": "active" if service.is_running else "inactive"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting EZ2 service: {str(e)}")

@router.post("/ez2/stop")
async def stop_ez2_service():
    """Stop the EZ2 trading service"""
    global ez2_service
    try:
        if ez2_service:
            await ez2_service.stop()
            ez2_service = None
        
        return {
            "success": True,
            "message": "EZ2 trading service stopped"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping EZ2 service: {str(e)}") 