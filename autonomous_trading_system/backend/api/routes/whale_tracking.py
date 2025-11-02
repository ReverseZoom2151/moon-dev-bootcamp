# autonomous_trading_system/backend/api/whale_tracking_api.py

from fastapi import APIRouter, Depends
from ...services.whale_tracking_service import WhaleTrackingService
import pandas as pd
import os

router = APIRouter()

# This is a simplified dependency injection. In a real app, you might use a more robust system.
def get_whale_tracking_service():
    # In a real application, you would pass configuration here,
    # e.g., from a settings file.
    return WhaleTrackingService()

@router.post("/whales/track", tags=["Whale Tracking"])
async def track_whales(service: WhaleTrackingService = Depends(get_whale_tracking_service)):
    """
    Triggers the whale tracking service to fetch and save the latest whale positions.
    This is an asynchronous endpoint that will run the scraping and analysis in the background.
    """
    # In a real application, you would likely run this as a background task
    # using something like Celery or FastAPI's BackgroundTasks.
    # For simplicity here, we run it directly.
    df = service.get_whale_positions(source="hyperdash")
    if not df.empty:
        service.save_positions_to_csv(df)
        return {"message": "Whale tracking complete.", "positions_found": len(df)}
    return {"message": "Whale tracking ran, but no new positions were found."}

@router.get("/whales/positions", tags=["Whale Tracking"])
async def get_whale_positions(service: WhaleTrackingService = Depends(get_whale_tracking_service)):
    """
    Retrieves the latest saved whale positions.
    """
    try:
        longs_df = pd.read_csv(os.path.join(service.data_dir, "top_long_positions.csv"))
        shorts_df = pd.read_csv(os.path.join(service.data_dir, "top_short_positions.csv"))
        return {
            "long_positions": longs_df.to_dict(orient="records"),
            "short_positions": shorts_df.to_dict(orient="records")
        }
    except FileNotFoundError:
        return {"message": "No position data found. Please run the tracking first."}
    except Exception as e:
        return {"message": f"An error occurred: {e}"}
