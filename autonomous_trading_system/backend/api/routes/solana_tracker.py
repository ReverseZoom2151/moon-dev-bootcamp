"""
API Endpoints for the Solana Token Tracker Service
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from services.solana_token_tracker_service import SolanaTokenTrackerService

router = APIRouter()

def get_solana_token_tracker_service(request: Request) -> SolanaTokenTrackerService:
    """Dependency to get the solana token tracker service instance from the app state."""
    # This assumes the service is attached to the app's state in main.py
    # e.g., app.state.solana_token_tracker = ...
    service = getattr(request.app.state, 'solana_token_tracker', None)
    if service is None:
        raise HTTPException(status_code=503, detail="Solana Token Tracker service is not available.")
    return service

@router.get("/solana-tracker/status", tags=["Solana Token Tracker"])
async def get_tracker_status(service: SolanaTokenTrackerService = Depends(get_solana_token_tracker_service)):
    """Get the current status of the Solana Token Tracker service."""
    return {
        "status": service.status,
        "last_run": service.last_run_timestamp,
        "settings": {
            "enabled": service.settings.ENABLE_SOLANA_TOKEN_TRACKER,
            "interval_seconds": service.settings.SOLANA_TOKEN_TRACKER_INTERVAL_SECONDS
        }
    }

@router.get("/solana-tracker/trending", tags=["Solana Token Tracker"])
async def get_trending_tokens(service: SolanaTokenTrackerService = Depends(get_solana_token_tracker_service)):
    """Get the latest trending tokens."""
    if service.latest_trending_df.empty:
        return JSONResponse(content={"message": "No trending token data available yet. Please wait for the next run."}, status_code=204)
    return JSONResponse(content=service.latest_trending_df.to_dict(orient='records'))

@router.get("/solana-tracker/new-listings", tags=["Solana Token Tracker"])
async def get_new_listings(service: SolanaTokenTrackerService = Depends(get_solana_token_tracker_service)):
    """Get the latest new token listings."""
    if service.latest_new_listings_df.empty:
        return JSONResponse(content={"message": "No new listing data available yet. Please wait for the next run."}, status_code=204)
    return JSONResponse(content=service.latest_new_listings_df.to_dict(orient='records'))

@router.get("/solana-tracker/gems", tags=["Solana Token Tracker"])
async def get_possible_gems(service: SolanaTokenTrackerService = Depends(get_solana_token_tracker_service)):
    """Get possible gem tokens from the latest trending list."""
    df = service.latest_trending_df
    if df.empty:
        return JSONResponse(content={"message": "No trending token data available to filter for gems."}, status_code=204)
    
    gems_df = df[df['marketcap'].fillna(float('inf')) <= service.settings.SOLANA_TOKEN_TRACKER_GEMS_MAX_MARKET_CAP].copy()
    
    if gems_df.empty:
        return JSONResponse(content={"message": "No gems found in the latest trending tokens list."}, status_code=204)
        
    return JSONResponse(content=gems_df.to_dict(orient='records'))

@router.get("/solana-tracker/consistent-trending", tags=["Solana Token Tracker"])
async def get_consistent_trending(service: SolanaTokenTrackerService = Depends(get_solana_token_tracker_service)):
    """Get the tokens that have been trending consistently."""
    df = service.latest_history_df
    if df.empty:
        return JSONResponse(content={"message": "No historical data for consistent trending is available yet."}, status_code=204)
    
    filtered_df = df[~df['address'].isin(service.settings.SOLANA_TOKEN_TRACKER_IGNORE_LIST)]
    top_n = service.settings.SOLANA_TOKEN_TRACKER_TOP_CONSISTENT_TOKENS * 2
    top_tokens = filtered_df.nlargest(top_n, 'appearances')

    return JSONResponse(content=top_tokens.to_dict(orient='records'))
