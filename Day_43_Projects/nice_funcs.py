import threading
import time
import queue
import pandas as pd
import os
import pytz
import logging
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connection Configuration
PAPER_PORT = 7497        # Paper trading port
LIVE_PORT = 7496         # Live trading port
IS_PAPER_TRADING = True  # Set to False for live trading
DEFAULT_CLIENT_ID = 10   # Default client ID for connections
DATA_PATH = os.path.join(os.getcwd(), "data", "IB_data") # Use relative path

# Timeout Configuration
CONNECT_TIMEOUT = 10     # Timeout for initial connection & nextValidId
DEFAULT_REQUEST_TIMEOUT = 15 # Default timeout for data requests
OHLCV_BASE_TIMEOUT = 20      # Base timeout for historical data
OHLCV_TIMEOUT_PER_DAY = 0.1 # Additional seconds per day (adjust as needed)

class IBApi(EWrapper, EClient):
    """Interactive Brokers API Wrapper Class with enhanced synchronization."""
    def __init__(self):
        EClient.__init__(self, self)
        self._next_req_id = 1 # Internal counter for request IDs
        self.nextValidOrderId = None

        # Synchronization Events
        self.connected_event = threading.Event()
        self._request_events = {} # reqId -> threading.Event()

        # Data Storage
        self._contract_details_map = {} # reqId -> ContractDetails
        self._tick_data_map = {} # reqId -> {'bid': None, 'ask': None, 'last': None}
        self._historical_data_map = {} # reqId -> list of bars
        self._open_orders_list = []
        self._open_orders_received = False
        self.error_messages = queue.Queue() # Use queue for thread safety

        logging.info("🌙 Moon Dev's IB API Handler initialized! 🚀")

    def _get_next_req_id(self):
        """Generates the next unique request ID."""
        req_id = self._next_req_id
        self._next_req_id += 1
        return req_id

    def _create_event(self, req_id):
        """Creates and stores a threading event for a request ID."""
        event = threading.Event()
        self._request_events[req_id] = event
        return event

    def _signal_event(self, req_id):
        """Signals the event associated with a request ID."""
        if req_id in self._request_events:
            self._request_events[req_id].set()
        else:
            # This might happen for unsolicited messages or if event cleanup logic is flawed
            logging.debug(f"Attempted to signal event for unknown ReqId: {req_id}")

    def _cleanup_event(self, req_id):
         """Removes the event associated with a request ID."""
         self._request_events.pop(req_id, None)

    # --- EWrapper Methods ---
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Ignore common informational messages or warnings
        ignore_codes = {2104, 2106, 2108, 2158, 2109, 2150, 321, 399, 202, 10167}
        # 2109: Market Data Farm connection is OK
        # 2150: Invalid positionAttribute
        # 321: Error validating request: 'HMDS query returned no data'
        # 399: Order message
        # 202: Order Canceled
        # 10167: Requested market data is not subscribed

        error_msg = f"Error Code {errorCode} (ReqId: {reqId}): {errorString}"
        self.error_messages.put(error_msg)

        if errorCode not in ignore_codes:
            logging.error(f"❌ {error_msg}")
            if advancedOrderRejectJson:
                 logging.error(f"   Advanced Reject Info: {advancedOrderRejectJson}")

        # Signal completion on certain errors to prevent deadlocks
        # e.g., No security definition found, historical data query cancelled, etc.
        if errorCode in [200, 162, 322]:
             logging.warning(f"Signaling event for ReqId {reqId} due to error {errorCode}.")
             self._signal_event(reqId)

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        logging.info(f"🔑 Moon Dev's Next Valid Order ID: {orderId}")
        # This is the signal that the connection is ready
        self.connected_event.set()

    def connectionClosed(self):
        logging.warning("🔌 Connection closed by IB.")
        self.connected_event.clear() # Reset connected state
        # Potentially signal all pending events to unblock waiting threads
        for req_id in list(self._request_events.keys()):
             logging.warning(f"Signaling event {req_id} due to connection closure.")
             self._signal_event(req_id)

    def contractDetails(self, reqId, contractDetails):
        logging.debug(f"Received contract details for ReqId: {reqId}")
        self._contract_details_map[reqId] = contractDetails

    def contractDetailsEnd(self, reqId):
        logging.info(f"✅ Contract details ended for ReqId: {reqId}")
        self._signal_event(reqId)

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        # Primarily for order cancellation confirmation in this script
        logging.info(f"Order {orderId} Status: {status}, Filled: {filled}")
        if status == "Cancelled":
            logging.info(f"✨ Order {orderId} successfully cancelled by Moon Dev! ✨")
        # We could potentially use an event per orderId if waiting for specific statuses

    def openOrder(self, orderId, contract, order, orderState):
        # Received in response to reqAllOpenOrders
        self._open_orders_list.append({
            "orderId": orderId,
            "symbol": contract.symbol,
            "contract": contract,
            "order": order,
            "status": orderState.status
        })
        logging.debug(f"Received open order: {orderId} - {contract.symbol} - {order.action} {order.totalQuantity} - Status: {orderState.status}")

    def openOrderEnd(self):
        # Signals the end of the open orders stream
        logging.info(f"✅ Received {len(self._open_orders_list)} open orders.")
        self._open_orders_received = True
        # Assume there's a dedicated event for reqAllOpenOrders
        # Since reqAllOpenOrders doesn't take a reqId, we use a known ID or flag
        self._signal_event('reqAllOpenOrders')

    def tickPrice(self, reqId, tickType, price, attrib):
        # Tick types: 1=bid, 2=ask, 4=last, 6=high, 7=low, 9=close
        # Delayed ticks: 66=bid, 67=ask, 68=last
        if reqId not in self._tick_data_map:
            self._tick_data_map[reqId] = {'bid': None, 'ask': None, 'last': None}

        tick_data = self._tick_data_map[reqId]
        updated = False
        if tickType in [1, 66]:
            tick_data['bid'] = price
            updated = True
            logging.debug(f"Tick ReqId {reqId}: Bid = {price}")
        elif tickType in [2, 67]:
            tick_data['ask'] = price
            updated = True
            logging.debug(f"Tick ReqId {reqId}: Ask = {price}")
        elif tickType in [4, 68]:
            tick_data['last'] = price
            updated = True
            logging.debug(f"Tick ReqId {reqId}: Last = {price}")

        # Signal if we have both bid and ask (sufficient for get_bid_ask)
        if updated and tick_data['bid'] is not None and tick_data['ask'] is not None:
            self._signal_event(reqId)

    def historicalData(self, reqId, bar):
        if reqId not in self._historical_data_map:
            self._historical_data_map[reqId] = []
            logging.info(f"🎬 Starting to receive historical data for ReqId: {reqId}...")

        self._historical_data_map[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

        # Log progress periodically
        count = len(self._historical_data_map[reqId])
        if count % 200 == 0:
            logging.info(f"...received {count} historical bars for ReqId {reqId}...")

    def historicalDataEnd(self, reqId, start, end):
        count = len(self._historical_data_map.get(reqId, []))
        logging.info(f"✅ Historical data ended for ReqId: {reqId}. Total bars received: {count}")
        self._signal_event(reqId)

# --- Standalone Functions --- #

def connect_ib(port=None, client_id=DEFAULT_CLIENT_ID):
    """Establishes connection to TWS/Gateway and waits for it to be ready.

    Args:
        port (int, optional): Connection port. Defaults to paper/live based on IS_PAPER_TRADING.
        client_id (int): Client ID for the connection.

    Returns:
        IBApi: Connected and ready IBApi instance, or None if connection fails.
    """
    if port is None:
        port = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
    trading_mode = "Paper" if port == PAPER_PORT else "Live"

    app = IBApi()
    logging.info(f"🚀 Connecting to {trading_mode} Trading on 127.0.0.1:{port} (ClientId: {client_id})...")
    app.connect("127.0.0.1", port, clientId=client_id)

    # Start the message processing thread
    con_thread = threading.Thread(target=app.run, daemon=True)
    con_thread.start()
    logging.info("Connection thread started.")

    # Wait for connection and nextValidId
    if app.connected_event.wait(timeout=CONNECT_TIMEOUT):
        logging.info("✅ Successfully connected and received nextValidId.")
        return app
    else:
        logging.error(f"❌ Connection timeout after {CONNECT_TIMEOUT} seconds.")
        # Attempt to disconnect even if connection failed partially
        try:
            app.disconnect()
        except Exception as e:
            logging.error(f"Error during disconnect after timeout: {e}")
        return None

def disconnect_ib(app):
    """Disconnects the IB API connection safely."""
    if app and app.isConnected():
        logging.info("⏳ Disconnecting from IB...")
        app.disconnect()
        # Give a moment for disconnect process
        time.sleep(0.5)
        logging.info("👋 Disconnected safely.")
    elif app:
         logging.info("Already disconnected.")
    else:
         logging.warning("No valid app instance provided to disconnect.")

def create_contract(symbol, sec_type="STK", exchange="SMART", currency="USD", **kwargs):
    """Creates an IB Contract object.

    Args:
        symbol (str): The asset symbol.
        sec_type (str): Security type (STK, OPT, FUT, IND, FOP, etc.).
        exchange (str): The destination exchange (e.g., SMART, CME, NYMEX).
        currency (str): The currency (e.g., USD, EUR).
        **kwargs: Additional contract attributes (e.g., lastTradeDateOrContractMonth, strike, right).

    Returns:
        Contract: An initialized IB Contract object.
    """
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.exchange = exchange
    contract.currency = currency
    for key, value in kwargs.items():
        setattr(contract, key, value)
    logging.debug(f"Created Contract: {vars(contract)}")
    return contract

def cancel_all_orders(app=None):
    """Cancels all open orders globally.

    Args:
        app (IBApi, optional): Existing connected IBApi instance. If None, will connect.

    Returns:
        bool: True if cancellation request was sent, False otherwise.
    """
    logging.info("🔄 Requesting global cancellation of all open orders...")
    manage_connection = app is None
    if manage_connection:
        app = connect_ib()
        if not app:
            return False

    success = False
    try:
        app.reqGlobalCancel()
        # Note: There's no specific confirmation callback for reqGlobalCancel success.
        # We assume it's sent if no immediate error occurs.
        # Confirmation would come via orderStatus for individual cancelled orders.
        logging.info("✅ Global cancel request sent. Check orderStatus messages for confirmation.")
        success = True
        time.sleep(0.5) # Give a moment for the request to process
    except Exception as e:
        logging.error(f"❌ Error sending global cancel request: {e}", exc_info=True)
    finally:
        if manage_connection:
            disconnect_ib(app)
    return success

def cancel_symbol_orders(symbol, app=None):
    """Cancels all open orders for a specific symbol.

    Args:
        symbol (str): The trading symbol.
        app (IBApi, optional): Existing connected IBApi instance. If None, will connect.

    Returns:
        int: Number of orders successfully requested for cancellation.
    """
    logging.info(f"🔄 Cancelling open orders specifically for {symbol}...")
    manage_connection = app is None
    if manage_connection:
        app = connect_ib()
        if not app:
            return 0

    cancelled_count = 0
    try:
        app._open_orders_list = [] # Clear previous list
        app._open_orders_received = False
        open_order_event = app._create_event('reqAllOpenOrders') # Use a special key

        app.reqAllOpenOrders()
        logging.info("Requested open orders. Waiting for response...")

        if open_order_event.wait(timeout=DEFAULT_REQUEST_TIMEOUT):
            orders_to_cancel = [o for o in app._open_orders_list if o['symbol'] == symbol]
            if not orders_to_cancel:
                logging.info(f"ℹ️ No open orders found for {symbol}.")
            else:
                logging.info(f"Found {len(orders_to_cancel)} open order(s) for {symbol}. Requesting cancellation...")
                for order_data in orders_to_cancel:
                    order_id = order_data['orderId']
                    try:
                        app.cancelOrder(order_id, "") # Manual cancel time optional
                        logging.info(f" -> Sent cancel request for OrderID: {order_id}")
                        cancelled_count += 1
                        time.sleep(0.1) # Small delay between cancel requests
                    except Exception as cancel_e:
                         logging.error(f"Error sending cancel for OrderID {order_id}: {cancel_e}")
                logging.info(f"✅ Finished sending {cancelled_count} cancellation requests for {symbol}.")
        else:
            logging.error("❌ Timeout waiting for open orders list.")

    except Exception as e:
        logging.error(f"❌ Error during cancellation process for {symbol}: {e}", exc_info=True)
    finally:
        app._cleanup_event('reqAllOpenOrders')
        if manage_connection:
            disconnect_ib(app)
    return cancelled_count

def get_bid_ask(symbol, app=None):
    """Gets the current bid and ask prices for a symbol.

    Args:
        symbol (str): The trading symbol.
        app (IBApi, optional): Existing connected IBApi instance. If None, will connect.

    Returns:
        dict: {'bid': float, 'ask': float} or None if data couldn't be retrieved.
    """
    logging.info(f"💰 Fetching Bid/Ask for {symbol}...")
    manage_connection = app is None
    if manage_connection:
        app = connect_ib()
        if not app:
            return None

    req_id = app._get_next_req_id()
    bid_ask_event = app._create_event(req_id)
    app._tick_data_map[req_id] = {'bid': None, 'ask': None, 'last': None}
    result = None

    try:
        contract = create_contract(symbol)
        # Request streaming data. Snapshot=False
        app.reqMktData(req_id, contract, "", False, False, [])
        logging.info(f"Requested market data (ReqId: {req_id}). Waiting for Bid/Ask...")

        if bid_ask_event.wait(timeout=DEFAULT_REQUEST_TIMEOUT):
            tick_data = app._tick_data_map.get(req_id)
            if tick_data and tick_data['bid'] is not None and tick_data['ask'] is not None:
                result = {'bid': tick_data['bid'], 'ask': tick_data['ask']}
                logging.info(f"✅ {symbol} -> Bid: ${result['bid']:.2f}, Ask: ${result['ask']:.2f}")
            else:
                 logging.warning(f"⚠️ Event signaled for ReqId {req_id}, but bid/ask data missing in map.")
        else:
            logging.error(f"❌ Timeout waiting for Bid/Ask for {symbol} (ReqId: {req_id}).")
            # Check if partial data was received
            partial_data = app._tick_data_map.get(req_id)
            if partial_data:
                 logging.warning(f"   Partial data: {partial_data}")
            else:
                 logging.warning(f"   No tick data received for ReqId {req_id}.")

    except Exception as e:
        logging.error(f"❌ Error getting Bid/Ask for {symbol}: {e}", exc_info=True)
    finally:
        # Always cancel market data subscription
        try:
            app.cancelMktData(req_id)
            logging.debug(f"Cancelled market data for ReqId: {req_id}")
        except Exception as cancel_e:
            logging.error(f"Error cancelling market data for ReqId {req_id}: {cancel_e}")
        app._cleanup_event(req_id)
        app._tick_data_map.pop(req_id, None) # Clean up data entry
        if manage_connection:
            disconnect_ib(app)
    return result

def get_ohlcv(symbol, duration="1 D", bar_size="1 min", app=None):
    """Fetches OHLCV data for a symbol and saves it to a CSV file.

    Args:
        symbol (str): Trading symbol.
        duration (str): Data duration (e.g., "1 D", "5 D", "1 M", "1 Y").
        bar_size (str): Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day").
        app (IBApi, optional): Existing connected IBApi instance. If None, will connect.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data, or empty DataFrame on failure.
    """
    logging.info(f"📊 Fetching {duration} of {bar_size} OHLCV for {symbol}...")
    manage_connection = app is None
    if manage_connection:
        app = connect_ib()
        if not app:
            return pd.DataFrame()

    req_id = app._get_next_req_id()
    hist_data_event = app._create_event(req_id)
    app._historical_data_map[req_id] = [] # Initialize empty list
    df = pd.DataFrame()

    try:
        contract = create_contract(symbol)

        # Calculate timeout
        try:
             duration_value = int(''.join(filter(str.isdigit, duration)))
             # Crude estimation of days based on duration unit (needs refinement)
             unit = duration.split()[-1].upper()
             days_estimate = duration_value
             if unit in ['W', 'WEEKS']:
                 days_estimate *= 7
             elif unit in ['M', 'MONTHS']:
                 days_estimate *= 30
             elif unit in ['Y', 'YEARS']:
                 days_estimate *= 365
             timeout = max(OHLCV_BASE_TIMEOUT, days_estimate * OHLCV_TIMEOUT_PER_DAY)
        except:
             logging.warning("Could not parse duration for timeout calculation, using base timeout.")
             timeout = OHLCV_BASE_TIMEOUT
        logging.info(f"⏳ Setting request timeout to {timeout:.1f} seconds.")

        app.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1, # YYYYMMDD HH:MM:SS
            keepUpToDate=False,
            chartOptions=[]
        )
        logging.info(f"Requested historical data (ReqId: {req_id}). Waiting for completion...")

        if hist_data_event.wait(timeout=timeout):
            data = app._historical_data_map.get(req_id, [])
            if data:
                logging.info(f"✅ Received {len(data)} bars for {symbol}.")
                df = pd.DataFrame(data)
                # Process DataFrame (convert dates, sort)
                try:
                    # Attempt parsing, handle potential extra timezone info more robustly
                    df['date'] = df['date'].str.split(' ').str[:2].str.join(' ') # Keep only YYYYMMDD HH:MM:SS
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d %H:%M:%S')
                    df = df.sort_values('date').reset_index(drop=True)
                    logging.info(f"Data range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

                    # Save to CSV
                    os.makedirs(DATA_PATH, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"{symbol}_{duration.replace(' ','')}_{bar_size.replace(' ','')}_{timestamp}.csv"
                    fpath = os.path.join(DATA_PATH, fname)
                    df.to_csv(fpath, index=False)
                    logging.info(f"💾 Data saved to: {fpath}")

                    # Log latest bar
                    latest = df.iloc[-1]
                    logging.info(f"📈 Latest Bar - O:{latest['open']:.2f} H:{latest['high']:.2f} L:{latest['low']:.2f} C:{latest['close']:.2f} V:{latest['volume']}")
                except Exception as parse_e:
                    logging.error(f"Error processing DataFrame: {parse_e}", exc_info=True)
                    df = pd.DataFrame() # Return empty if processing fails
            else:
                logging.warning(f"⚠️ No historical data bars received for {symbol} (ReqId: {req_id}), though request finished.")
        else:
            logging.error(f"❌ Timeout waiting for historical data for {symbol} (ReqId: {req_id}).")
            # Check if any partial data was received
            partial_data = app._historical_data_map.get(req_id, [])
            if partial_data:
                 logging.warning(f"   Received {len(partial_data)} bars before timeout.")
                 # Optionally process partial data here if needed
            else:
                 logging.warning(f"   No data received for ReqId {req_id} before timeout.")

    except Exception as e:
        logging.error(f"❌ Error getting OHLCV for {symbol}: {e}", exc_info=True)
        df = pd.DataFrame() # Ensure empty DF on error
    finally:
        app._cleanup_event(req_id)
        app._historical_data_map.pop(req_id, None) # Clean up data entry
        if manage_connection:
            disconnect_ib(app)
    return df

def is_market_hours(tz_name='US/Eastern'):
    """Checks if the current time is within typical US market hours (9:30 AM - 4:00 PM) in the specified timezone.

    Args:
        tz_name (str): Timezone name (e.g., 'US/Eastern', 'Europe/London').

    Returns:
        bool: True if within market hours (Mon-Fri, 9:30-16:00), False otherwise.
    """
    try:
        tz = pytz.timezone(tz_name)
    except pytz.UnknownTimeZoneError:
        logging.error(f"Unknown timezone: {tz_name}. Defaulting to US/Eastern.")
        tz = pytz.timezone('US/Eastern')

    now = datetime.now(tz)
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0).time()
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0).time()
    current_time = now.time()

    # Check if it's a weekday (Monday=0, Sunday=6)
    is_weekday = now.weekday() < 5
    is_within_hours = market_open_time <= current_time < market_close_time

    if is_weekday and is_within_hours:
        time_left = datetime.combine(now.date(), market_close_time, tzinfo=tz) - now
        hours, remainder = divmod(time_left.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        logging.info(f"🌟 Market ({tz_name}) is OPEN! Closes in {hours}h {minutes}m.")
        return True
    else:
        status = "WEEKEND" if not is_weekday else (
            "PRE-MARKET" if current_time < market_open_time else "POST-MARKET/CLOSED"
        )
        logging.info(f"🌙 Market ({tz_name}) is {status}.")
        # Optional: Calculate time until next open
        return False

# --- Example Usage --- #
if __name__ == "__main__":
    logging.info("--- Running Nice Funcs Examples ---")

    # Example 1: Check Market Hours
    is_open = is_market_hours()
    logging.info(f"Market Open Check: {is_open}")
    time.sleep(1)

    # Example 2: Get Bid/Ask
    # Note: Requires market data subscription or USE_DELAYED_DATA in IB Gateway settings
    # bid_ask_data = get_bid_ask("AAPL")
    # if bid_ask_data:
    #     logging.info(f"AAPL Bid/Ask: {bid_ask_data}")
    # time.sleep(1)

    # Example 3: Get OHLCV Data
    # ohlcv_df = get_ohlcv("MSFT", duration="5 D", bar_size="15 mins")
    # if not ohlcv_df.empty:
    #     logging.info(f"MSFT OHLCV Data (last 5 rows):\n{ohlcv_df.tail()}")
    # time.sleep(1)

    # Example 4: Cancel Orders (Use with caution!)
    # cancelled_count = cancel_symbol_orders("TEST") # Replace TEST with actual symbol if needed
    # logging.info(f"Cancellation requests sent for TEST: {cancelled_count}")
    # time.sleep(1)
    # cancel_all_success = cancel_all_orders()
    # logging.info(f"Global cancellation request sent: {cancel_all_success}")

    logging.info("--- Examples Complete ---")
