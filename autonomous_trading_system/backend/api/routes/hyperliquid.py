"""
API Routes for Hyperliquid Whale Tracker Service
"""

import pandas as pd
import os
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any
from services.hyperliquid_whale_service import HyperliquidWhaleTrackerService
from core.config import get_settings

router = APIRouter()
settings = get_settings()

def get_whale_tracker_service() -> HyperliquidWhaleTrackerService:
    """Dependency injector for the whale tracker service."""
    return HyperliquidWhaleTrackerService()

@router.post("/hyperliquid/scan", status_code=202)
async def start_scan(
    background_tasks: BackgroundTasks,
    source: str = "arbiscan",
    dump_raw: bool = False,
    service: HyperliquidWhaleTrackerService = Depends(get_whale_tracker_service)
):
    """
    Starts a new scan for whale positions in the background.
    
    - **source**: 'file', 'arbiscan', or 'hyperdash'.
    - **dump_raw**: Set to true to dump raw API data for one address (for debugging).
    """
    if source not in ["file", "arbiscan", "hyperdash"]:
        raise HTTPException(status_code=400, detail="Invalid source provided. Must be 'file', 'arbiscan', or 'hyperdash'.")
    
    background_tasks.add_task(service.run_scan, source=source, dump_raw=dump_raw)
    
    return {"message": f"Whale position scan started in the background from source: {source}."}

def read_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """Helper function to read and parse CSV data, handling file not found errors."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Data file not found: {os.path.basename(file_path)}. Please run a scan first.")
    try:
        df = pd.read_csv(file_path)
        # Handle NaN values which are not JSON compliant
        df = df.replace({pd.NA: None, np.nan: None})
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading or parsing data file: {str(e)}")

@router.get("/hyperliquid/results/{file_name}")
async def get_scan_results(file_name: str):
    """
    Retrieves the latest scan results from a specified CSV file.
    
    Available `file_name` values:
    - `all_positions.csv`
    - `aggregated_positions.csv`
    - `top_whale_long_positions.csv`
    - `top_whale_short_positions.csv`
    - `liquidation_closest_long_positions.csv`
    - `liquidation_closest_short_positions.csv`
    """
    valid_files = [
        "all_positions.csv", "aggregated_positions.csv", 
        "top_whale_long_positions.csv", "top_whale_short_positions.csv",
        "liquidation_closest_long_positions.csv", "liquidation_closest_short_positions.csv"
    ]
    
    if file_name not in valid_files:
        raise HTTPException(status_code=400, detail=f"Invalid file name. Please choose from: {', '.join(valid_files)}")
        
    data_dir = getattr(settings, 'HYPERLIQUID_DATA_DIR', "data/hyperliquid_positions")
    file_path = os.path.join(data_dir, file_name)
    
    return read_csv_data(file_path)

@router.get("/hyperliquid/results")
async def list_available_results():
    """Lists all available result files from the latest scan."""
    data_dir = getattr(settings, 'HYPERLIQUID_DATA_DIR', "data/hyperliquid_positions")
    if not os.path.isdir(data_dir):
        return {"message": "Data directory not found. Please run a scan first.", "available_files": []}
    
    try:
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        return {"available_files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files in data directory: {str(e)}") 