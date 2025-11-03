"""
FastAPI REST API Layer for Gordon
==================================
Provides REST API endpoints for all Gordon functionality.
Enables web dashboard, mobile apps, and external integrations.
"""

from fastapi import FastAPI, HTTPException, Query, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Gordon Trading API",
    description="REST API for Gordon Financial Research & Trading Agent",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Gordon instance (will be set during startup)
gordon_agent: Optional[Any] = None
strategy_engine: Optional[Any] = None
risk_manager: Optional[Any] = None
portfolio_manager: Optional[Any] = None
websocket_manager: Optional[Any] = None
database_manager: Optional[Any] = None


def get_gordon():
    """Dependency to get Gordon agent instance."""
    if gordon_agent is None:
        raise HTTPException(status_code=503, detail="Gordon agent not initialized")
    return gordon_agent


def get_strategy_engine():
    """Dependency to get strategy engine."""
    if strategy_engine is None:
        raise HTTPException(status_code=503, detail="Strategy engine not initialized")
    return strategy_engine


def get_risk_manager():
    """Dependency to get risk manager."""
    if risk_manager is None:
        raise HTTPException(status_code=503, detail="Risk manager not initialized")
    return risk_manager


def get_portfolio_manager():
    """Dependency to get portfolio manager."""
    if portfolio_manager is None:
        raise HTTPException(status_code=503, detail="Portfolio manager not initialized")
    return portfolio_manager


def get_websocket_manager():
    """Dependency to get WebSocket manager."""
    if websocket_manager is None:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")
    return websocket_manager


def get_database_manager():
    """Dependency to get database manager."""
    if database_manager is None:
        raise HTTPException(status_code=503, detail="Database manager not initialized")
    return database_manager


def set_database_manager(db_manager: Any):
    """Set database manager globally."""
    global database_manager
    database_manager = db_manager
    logger.info("Database manager set globally")


# Import and include routers
from .routes import trading, strategies, risk, portfolio, websocket, research, market, backtesting, ml, whales, dashboard, sentiment, realtime, database

app.include_router(trading.router)
app.include_router(strategies.router)
app.include_router(risk.router)
app.include_router(portfolio.router)
app.include_router(websocket.router)
app.include_router(research.router)
app.include_router(market.router)
app.include_router(backtesting.router)
app.include_router(ml.router)
app.include_router(whales.router)
app.include_router(dashboard.router)
app.include_router(sentiment.router)
app.include_router(realtime.router)
app.include_router(database.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Gordon Trading API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "strategies": "/api/strategies",
            "trading": "/api/trading",
            "risk": "/api/risk",
            "portfolio": "/api/portfolio",
            "websocket": "/api/websocket",
            "research": "/api/research",
            "market": "/api/market",
            "backtesting": "/api/backtesting",
            "ml": "/api/ml",
            "whales": "/api/whales",
            "dashboard": "/api/dashboard",
            "sentiment": "/api/sentiment",
            "database": "/api/database"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "gordon_agent": gordon_agent is not None,
            "strategy_engine": strategy_engine is not None,
            "risk_manager": risk_manager is not None,
            "portfolio_manager": portfolio_manager is not None,
            "websocket_manager": websocket_manager is not None,
            "database_manager": database_manager is not None
        }
    }


def initialize_api(gordon_instance: Any, strategy_engine_instance: Any,
                   risk_manager_instance: Any, portfolio_manager_instance: Any,
                   websocket_manager_instance: Any):
    """Initialize API with Gordon components."""
    global gordon_agent, strategy_engine, risk_manager, portfolio_manager, websocket_manager
    
    gordon_agent = gordon_instance
    strategy_engine = strategy_engine_instance
    risk_manager = risk_manager_instance
    portfolio_manager = portfolio_manager_instance
    websocket_manager = websocket_manager_instance
    
    logger.info("API initialized with Gordon components")


