"""
API Endpoints for the EZ Bot Service
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Request, Body
from autonomous_trading_system.backend.services.ez_bot_service import EZBotService
from autonomous_trading_system.backend.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# --- Dependency Injection ---

def get_ez_bot_service(request: Request) -> EZBotService:
    """Dependency to get the EZBotService instance from the app state."""
    service = getattr(request.app.state, 'ez_bot_service', None)
    if service is None:
        raise HTTPException(status_code=503, detail="EZ Bot Service is not available.")
    return service

# --- API Endpoints ---

@router.post("/ez-bot/execute-action", tags=["EZ Bot"])
async def execute_bot_action(
    action_code: int = Body(..., embed=True, description="Action code (0-3). 0: Close, 1: Market Buy, 2: Demand Zone Buy, 3: Supply Zone Close."),
    contract_address: str = Body(None, embed=True, description="The token contract address. Uses default if not provided."),
    target_usd_size: float = Body(None, embed=True, description="Target USD size for market buy. Uses default if not provided."),
    service: EZBotService = Depends(get_ez_bot_service)
):
    """
    Executes a specific action from the EZ Bot.
    - **Action 0**: Close full position.
    - **Action 1**: Market buy to a target USD size.
    - **Action 2**: Check if price is in Demand Zone and buy if it is. This is a one-off check, not a continuous loop.
    - **Action 3**: Check if price is in Supply Zone and close if it is. This is a one-off check, not a continuous loop.
    """
    token = contract_address if contract_address else settings.EZ_BOT_DEFAULT_CONTRACT
    
    try:
        if action_code == 0:
            result = service.close_full_position(token)
        elif action_code == 1:
            result = service.market_buy_to_target(token, target_usd_size)
        elif action_code == 2:
            result = service.demand_zone_buy(token)
        elif action_code == 3:
            result = service.supply_zone_close(token)
        else:
            raise HTTPException(status_code=400, detail="Invalid action code. Please use 0, 1, 2, or 3.")
        
        return result
    except Exception as e:
        logger.error(f"Error executing EZ Bot action {action_code} for {token}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ez-bot/status/{contract_address}", tags=["EZ Bot"])
async def get_bot_status(contract_address: str, service: EZBotService = Depends(get_ez_bot_service)):
    """Get the current position status for a given token."""
    try:
        pos_units, pos_usd = service._get_current_position_usd(contract_address)
        return {"contract_address": contract_address, "position_units": pos_units, "position_usd": pos_usd}
    except Exception as e:
        logger.error(f"Error getting status for {contract_address}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ez-bot/config", tags=["EZ Bot"])
async def get_ez_bot_config():
    """Get the current configuration settings for the EZ Bot."""
    return {
        key: value for key, value in settings.model_dump().items() if key.startswith("EZ_BOT_")
    }
