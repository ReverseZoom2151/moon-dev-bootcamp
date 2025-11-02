from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Trading Configuration
SYMBOL = "ROKU"           # Symbol to trade - Available symbols:
                         # SNAP, PLTR, RDDT, APP, NVDA, TSLA, COIN, GOOG, SMCI, SHOP, 
                         # HOOD, CRSPR, LLY, TMSC, MELI, CRWD, RBLX, CROX, XYZ, TTD,
                         # NTRA, ROKU, LVMH, ADBE, AMD, DE, TWST, PATH, BEAM, JD, TXG,
                         # GME, RKLB, CHWY, GTLB, DDOG, SNOW, CRM, PYPL, INTU, LMT,
                         # CAT, TEAM, NOW, AUR, PSTG, KMTUY, MSTR, MARA
DIRECTION = "BUY"        # "BUY" or "SELL"
QUANTITY = 100           # Number of shares to trade
LIMIT_PRICE = 11.00      # Limit price for the order (ensure float)
ORDER_WAIT_TIMEOUT = 30  # Seconds to wait for order fill/final status

# Connection Configuration
PAPER_PORT = 7497         # Paper trading port
LIVE_PORT = 7496          # Live trading port
IS_PAPER_TRADING = True   # Set to False for live trading
CONNECT_TIMEOUT = 10      # Seconds to wait for connection and nextValidId

class TradeApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = None

        # Events for synchronization
        self.connected_event = threading.Event()
        self.order_status_events = {} # orderId -> threading.Event()

        # Data storage
        self.order_status_cache = {} # orderId -> status dict

        logging.info("🌙 Moon Dev's Limit Order Trader initialized! 🚀")

    # --- EWrapper Methods ---
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Ignore informational messages
        ignore_codes = {2104, 2106, 2108, 2158, 399, 202} # Info, order warnings, order cancelled
        if errorCode not in ignore_codes:
            logging.error(f"❌ Error {errorCode}: {errorString} (ReqId: {reqId})")
            if advancedOrderRejectJson:
                 logging.error(f"   Advanced Reject Info: {advancedOrderRejectJson}")
        # Signal relevant events on specific errors if needed (e.g., order reject)
        # if errorCode in [relevant_error_codes]:
        #    order_id = find_order_id_from_reqId(reqId) # Need a mapping if reqId != orderId
        #    if order_id in self.order_status_events:
        #        self.order_status_events[order_id].set()

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        logging.info(f"🔑 Moon Dev's Next Valid Order ID: {orderId}")
        if not self.connected_event.is_set():
            self.connected_event.set() # Signal connection established

    def connectionClosed(self):
        logging.warning("🔌 Connection closed.")
        self.connected_event.clear()

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, 
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        status_info = {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avgFillPrice': avgFillPrice,
            'permId': permId
        }
        self.order_status_cache[orderId] = status_info
        logging.info(
            f"🔄 Order {orderId} status: {status}, Filled: {filled}, Remaining: {remaining}, AvgFillPrice: {avgFillPrice}"
        )
        if status == "Submitted":
            logging.info(f"   Order details: Type=LMT, Price=${LIMIT_PRICE:.2f}") # Log limit price on submit

        # Signal the event if the order is considered done (Filled, Cancelled, Inactive, ApiCancelled)
        # Or potentially Rejected status if handled in error callback
        if status in ["Filled", "Cancelled", "Inactive", "ApiCancelled"]:
            if orderId in self.order_status_events:
                self.order_status_events[orderId].set()
            # Clean up event if it exists to prevent reuse
            self.order_status_events.pop(orderId, None)

    # --- Helper/Action Methods ---
    def get_contract(self, symbol):
        """Creates a simple stock contract."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        logging.debug(f"Created STK contract for {symbol}")
        return contract

    def create_limit_order(self, direction, quantity, limit_price):
        """Creates a simple limit order."""
        order = Order()
        order.action = direction
        order.orderType = "LMT"
        order.totalQuantity = abs(quantity)
        order.lmtPrice = float(limit_price) # Ensure price is float
        order.transmit = True # Transmit immediately
        order.tif = "DAY" # Limit orders usually DAY by default, could be GTC
        logging.debug(f"Created LMT order: {direction} {abs(quantity)} @ ${limit_price:.2f}")
        return order

    def place_limit_order_flow(self, symbol, direction, quantity, limit_price):
        """Handles the flow of placing a limit order and waiting for status."""
        if self.nextValidOrderId is None:
            logging.error("❌ Cannot place order: nextValidOrderId is not set.")
            return False

        order_placed_successfully = False
        current_order_id = self.nextValidOrderId
        try:
            # 1. Get Contract
            contract = self.get_contract(symbol)

            # 2. Create Limit Order
            limit_order = self.create_limit_order(direction, quantity, limit_price)

            # 3. Place Order and Prepare for Status Update
            logging.info(f"\n🎮 Placing {direction} limit order for {quantity} {symbol} @ ${limit_price:.2f} (OrderID: {current_order_id})...")
            self.order_status_events[current_order_id] = threading.Event()
            self.order_status_cache.pop(current_order_id, None) # Clear previous status if any

            self.placeOrder(current_order_id, contract, limit_order)
            order_placed_successfully = True # Assume success unless placeOrder raises exception

            # 4. Wait for Order Status Update (Fill or Final State)
            logging.info(f"⏳ Waiting for fill or final status for order {current_order_id} (Timeout: {ORDER_WAIT_TIMEOUT}s)...")
            if self.order_status_events[current_order_id].wait(timeout=ORDER_WAIT_TIMEOUT):
                final_status = self.order_status_cache.get(current_order_id, {}).get('status', 'Unknown')
                if final_status == "Filled":
                    filled_qty = self.order_status_cache.get(current_order_id, {}).get('filled', 0)
                    avg_price = self.order_status_cache.get(current_order_id, {}).get('avgFillPrice', 0)
                    logging.info(f"✅ Successfully filled order {current_order_id}: {direction} {filled_qty} {symbol} at avg price ${avg_price:.2f}")
                    return True # Filled successfully
                else:
                    logging.warning(f"⚠️ Order {current_order_id} ({symbol}) finished with status '{final_status}'.")
                    return False # Order finished but not filled (e.g., Cancelled)
            else:
                # Timeout occurred - check last known status
                last_status = self.order_status_cache.get(current_order_id, {}).get('status', 'Unknown')
                logging.warning(f"⌛ Timeout waiting for final status for order {current_order_id} ({symbol}). Last known status: '{last_status}'.")
                logging.warning(f"   Order may still be working at ${limit_price:.2f}. Check TWS/Client Portal.")
                return False # Timeout, status uncertain

        except Exception as e:
            logging.error(f"❌ Failed to place or monitor limit order for {symbol}: {e}", exc_info=True)
            # Clean up event if order placement failed before wait
            if not order_placed_successfully:
                self.order_status_events.pop(current_order_id, None)
            return False
        finally:
            # Clean up event if it still exists (e.g., timeout occurred)
            self.order_status_events.pop(current_order_id, None)
            # Ensure next order ID increments even if flow failed
            if self.nextValidOrderId is not None:
                 self.nextValidOrderId +=1

def run_connection_loop(app_instance):
    """Runs the TWS/Gateway client's message loop."""
    try:
        app_instance.run()
    except Exception as e:
         logging.error(f"❌ Exception in TWS/Gateway message loop: {e}", exc_info=True)

def main():
    """Main function to connect and place the limit order."""
    app = TradeApp()
    port_to_use = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
    trading_mode = "Paper" if IS_PAPER_TRADING else "Live"
    logging.info(f"🚀 Moon Dev connecting to {trading_mode} trading on port {port_to_use}")

    app.connect("127.0.0.1", port_to_use, clientId=5) # Use a different clientId

    con_thread = threading.Thread(target=run_connection_loop, args=(app,), daemon=True)
    con_thread.start()
    logging.info("⏳ Waiting for server connection and nextValidId...")

    if not app.connected_event.wait(timeout=CONNECT_TIMEOUT):
        logging.error(f"❌ Timeout connecting or getting nextValidId after {CONNECT_TIMEOUT} seconds. Ensure TWS/Gateway is running and API is enabled.")
        # Attempt disconnect even if connect failed
        try:
            app.disconnect()
        except Exception:
            pass
        return

    # --- Place the Order --- 
    order_outcome = "Failed or Timed Out"
    try:
        success = app.place_limit_order_flow(
            SYMBOL,
            DIRECTION,
            QUANTITY,
            LIMIT_PRICE
        )
        if success:
            order_outcome = "Filled Successfully"
        # If success is False, the flow method logs the reason (cancelled, timed out, etc.)

    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during trading logic: {e}", exc_info=True)
        order_outcome = "Error during execution"
    finally:
        logging.info(f"🚦 Order Outcome: {order_outcome}")
        logging.info("⏳ Disconnecting...")
        app.disconnect()
        # Wait slightly for disconnect to complete
        if con_thread.is_alive():
             time.sleep(1)
        logging.info("👋 Moon Dev has disconnected safely.")

if __name__ == "__main__":
    main()