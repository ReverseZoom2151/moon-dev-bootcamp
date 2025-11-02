"""
API Routes for Interactive Brokers Trading
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from services.interactive_brokers_service import get_ib_service, InteractiveBrokersService

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models for Request Bodies ---
class MarketOrderRequest(BaseModel):
    symbol: str = Field(..., description="The stock or futures symbol (e.g., 'AAPL', 'ES').")
    direction: str = Field(..., description="'BUY' or 'SELL'.")
    quantity: float = Field(..., gt=0, description="The number of shares or contracts.")
    sec_type: str = Field("STK", description="'STK' for stocks or 'FUT' for futures.")

class LimitOrderRequest(BaseModel):
    symbol: str = Field(..., description="The stock symbol (e.g., 'AAPL').")
    direction: str = Field(..., description="'BUY' or 'SELL'.")
    quantity: float = Field(..., gt=0, description="The number of shares.")
    limit_price: float = Field(..., gt=0, description="The limit price for the order.")

class BracketOrderRequest(BaseModel):
    symbol: str = Field(..., description="The stock symbol (e.g., 'AAPL').")
    direction: str = Field(..., description="'BUY' or 'SELL'.")
    quantity: float = Field(..., gt=0, description="The number of shares.")
    entry_price: float = Field(..., gt=0, description="The limit price for the parent entry order.")
    take_profit_price: float = Field(..., gt=0, description="The limit price for the take-profit order.")
    stop_loss_price: float = Field(..., gt=0, description="The stop price for the stop-loss order.")

class StopOrderRequest(BaseModel):
    symbol: str = Field(..., description="The stock symbol (e.g., 'AAPL').")
    direction: str = Field(..., description="'BUY' or 'SELL'.")
    quantity: float = Field(..., gt=0, description="The number of shares.")
    stop_price: float = Field(..., gt=0, description="The stop price for the order.")

class CancelSymbolRequest(BaseModel):
    symbol: str = Field(..., description="The symbol for which to cancel all open orders.")

# --- API Endpoints ---

@router.get("/ib/market/status", summary="Check Market Hours")
async def get_market_status(
    timezone: str = Query("US/Eastern", description="Timezone to check (e.g., 'US/Eastern', 'Europe/London')."),
    ib_service: InteractiveBrokersService = Depends(get_ib_service)
):
    """Checks if the market for the specified timezone is open."""
    return ib_service.is_market_open(timezone)

@router.get("/ib/data/historical/{symbol}", summary="Get Historical OHLCV Data")
async def get_historical_data(
    symbol: str,
    duration: str = Query("1 M", description="Duration string (e.g., '1 D', '5 D', '1 M', '1 Y')."),
    bar_size: str = Query("1 day", description="Bar size string (e.g., '1 min', '5 mins', '1 hour', '1 day')."),
    ib_service: InteractiveBrokersService = Depends(get_ib_service)
):
    """Fetches historical OHLCV data for a given symbol."""
    try:
        df = await ib_service.get_historical_data(symbol, duration, bar_size)
        return {"status": "success", "data": df.to_dict('records')}
    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ib/data/tick/{symbol}", summary="Get Bid/Ask Price")
async def get_tick_data(symbol: str, ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Gets the current bid and ask price for a stock."""
    try:
        result = await ib_service.get_bid_ask(symbol)
        if result is None:
            raise HTTPException(status_code=404, detail="Could not retrieve bid/ask price. Market data may not be available.")
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Failed to get tick data for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ib/options/leaps", summary="Scan for LEAP Options")
async def get_leap_options(
    symbol: str,
    months_out: int = Query(6, ge=1, le=24, description="Target number of months until expiration."),
    num_strikes: int = Query(10, ge=2, le=50, description="Number of strikes to fetch around the current price."),
    ib_service: InteractiveBrokersService = Depends(get_ib_service)
):
    """
    Performs a scan for LEAP (Long-term) options for a given underlying stock.
    It finds a suitable expiration date and fetches prices for strikes around the current stock price.
    """
    try:
        results = await ib_service.get_leap_options(symbol, months_out, num_strikes)
        return {"status": "success", "data": results}
    except Exception as e:
        logger.error(f"Failed to get LEAP options for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ib/status", summary="Get IB Connection Status")
async def get_status(ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Checks the connection status of the Interactive Brokers service."""
    if ib_service.client.is_connected:
        return {"status": "success", "connection": "connected", "next_order_id": ib_service.client.nextValidOrderId}
    else:
        return {"status": "success", "connection": "disconnected"}

@router.post("/ib/market-order", summary="Place a Market Order")
async def post_market_order(request: MarketOrderRequest, ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Places a market order for a stock or a future."""
    try:
        result = await ib_service.place_market_order(request.symbol, request.direction, request.quantity, request.sec_type)
        return {"status": "success", "order_status": result}
    except Exception as e:
        logger.error(f"Failed to place market order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ib/limit-order", summary="Place a Limit Order")
async def post_limit_order(request: LimitOrderRequest, ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Places a limit order for a stock."""
    try:
        result = await ib_service.place_limit_order(request.symbol, request.direction, request.quantity, request.limit_price)
        return {"status": "success", "order_status": result}
    except Exception as e:
        logger.error(f"Failed to place limit order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ib/bracket-order", summary="Place a Bracket Order")
async def post_bracket_order(request: BracketOrderRequest, ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Places a bracket order (entry, take-profit, and stop-loss) for a stock."""
    try:
        results = await ib_service.place_bracket_order(
            request.symbol, request.direction, request.quantity,
            request.entry_price, request.take_profit_price, request.stop_loss_price
        )
        return {"status": "success", "order_statuses": results}
    except Exception as e:
        logger.error(f"Failed to place bracket order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ib/stop-order", summary="Place a Stop Order")
async def post_stop_order(request: StopOrderRequest, ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Places a simple stop order for a stock."""
    try:
        result = await ib_service.place_stop_order(request.symbol, request.direction, request.quantity, request.stop_price)
        return {"status": "success", "order_status": result}
    except Exception as e:
        logger.error(f"Failed to place stop order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ib/positions", summary="Get Account Positions", deprecated=True)
async def get_positions(ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """
    DEPRECATED: Use /ib/account-summary instead.
    Retrieves all current positions from the connected IB account.
    """
    try:
        summary = await ib_service.get_account_summary()
        return {"status": "success", "positions": summary.get("positions", [])}
    except Exception as e:
        logger.error(f"Failed to get positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ib/open-orders", summary="Get Open Orders")
async def get_open_orders(ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Retrieves a list of all pending open orders."""
    try:
        open_orders_df = await ib_service.get_open_orders()
        return {"status": "success", "open_orders": open_orders_df.to_dict('records')}
    except Exception as e:
        logger.error(f"Failed to get open orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ib/account-summary", summary="Get Account Summary and Positions")
async def get_account_summary(ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Subscribes to and retrieves a snapshot of the account summary and all positions."""
    try:
        summary = await ib_service.get_account_summary()
        return {"status": "success", "account_summary": summary}
    except Exception as e:
        logger.error(f"Failed to get account summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ib/cancel-all-orders", summary="Cancel All Open Orders")
async def post_cancel_all(ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """Sends a request to cancel all pending open orders."""
    try:
        await ib_service.cancel_all_orders()
        return {"status": "success", "message": "Request to cancel all orders sent."}
    except Exception as e:
        logger.error(f"Failed to cancel all orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ib/close-all-positions", summary="Close All Positions")
async def post_close_all(ib_service: InteractiveBrokersService = Depends(get_ib_service)):
    """
    The 'kill switch'. Cancels all open orders and then closes all open positions
    by submitting market orders.
    """
    try:
        results = await ib_service.close_all_positions()
        return {"status": "success", "message": "Request to close all positions sent.", "closing_orders": results}
    except Exception as e:
        logger.error(f"Failed to close all positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ib/orders/cancel-by-symbol", summary="Cancel Orders by Symbol")
async def post_cancel_by_symbol(
    request: CancelSymbolRequest,
    ib_service: InteractiveBrokersService = Depends(get_ib_service)
):
    """Cancels all open orders for a specific symbol."""
    try:
        count = await ib_service.cancel_orders_by_symbol(request.symbol)
        return {"status": "success", "message": f"Sent cancellation requests for {count} order(s) for symbol {request.symbol}."}
    except Exception as e:
        logger.error(f"Failed to cancel orders for {request.symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
