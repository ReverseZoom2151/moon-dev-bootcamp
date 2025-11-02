import logging
import threading
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Trading Configuration
SYMBOL = "ES"            # Futures symbol to trade
DIRECTION = "BUY"        # "BUY" or "SELL"
QUANTITY = 1             # Number of contracts to trade

# Connection Configuration
PAPER_PORT = 7497        # Paper trading port
LIVE_PORT = 7496         # Live trading port
IS_PAPER_TRADING = True  # Set to False for live trading
CONTRACT_WAIT_TIMEOUT = 10 # Seconds to wait for contract details
ORDER_FILL_TIMEOUT = 10    # Seconds to wait for order fill confirmation

class TradeApp(EWrapper, EClient):

    # Static data for futures contracts
    FUTURES_EXCHANGES = {
        "ES": "CME", "MES": "CME", "NQ": "CME", "CL": "NYMEX", "GC": "COMEX",
        "SI": "COMEX", "ZB": "ECBOT", "ZN": "ECBOT", "YM": "ECBOT", "RTY": "CME",
        "6E": "CME", "ZC": "ECBOT"
    }
    FUTURES_MULTIPLIERS = {
        "ES": "50", "MES": "5", "NQ": "20", "CL": "1000", "GC": "100", "SI": "5000",
        "ZB": "1000", "ZN": "1000", "YM": "5", "RTY": "50", "6E": "125000", "ZC": "5000"
    }
    FUTURES_TRADING_CLASSES = {
        "ES": "ES", "MES": "MES", "NQ": "NQ", "CL": "CL", "GC": "GC", "SI": "SI",
        "ZB": "ZB", "ZN": "ZN", "YM": "YM", "RTY": "M2K", # RTY uses M2K
        "6E": "EUR", # 6E uses EUR
        "ZC": "ZC"
    }

    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = None
        self._next_req_id = 1 # Internal request ID counter

        # Events for synchronization
        self.order_status_events = {} # orderId -> event
        self.contract_details_events = {} # reqId -> event

        # Data storage
        self.contract_details_results = {} # reqId -> list of ContractDetails
        self.order_status_cache = {} # orderId -> status dict
        self._active_contract_for_error = None # Store last contract used for context

    def _get_next_req_id(self):
        """Gets the next available request ID."""
        req_id = self._next_req_id
        self._next_req_id += 1
        return req_id

    # --- EWrapper Methods ---
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Ignore informational messages
        ignore_codes = {2104, 2106, 2108, 2158, 399, 202} # 202: Order Canceled
        if errorCode not in ignore_codes:
            logging.error(f"❌ Error {errorCode}: {errorString} (ReqId: {reqId})")
            if advancedOrderRejectJson:
                logging.error(f"   Advanced Reject Info: {advancedOrderRejectJson}")
            # Provide context if it's a contract-related error
            if errorCode == 200 and self._active_contract_for_error: # 200: No security definition found
                c = self._active_contract_for_error
                logging.error(f"   Context: Contract Symbol={c.symbol}, SecType={c.secType}, Exchange={c.exchange}, Currency={c.currency}, TradingClass={getattr(c, 'tradingClass', 'N/A')}, Multiplier={getattr(c, 'multiplier', 'N/A')}")

        # Signal relevant events on error to prevent deadlocks
        if reqId in self.contract_details_events:
            self.contract_details_events[reqId].set() # Signal completion even on error

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        if self.nextValidOrderId is None: # Only log the first time
             logging.info(f"🔑 Moon Dev's Next Valid Order ID: {orderId}")
        self.nextValidOrderId = orderId

    def contractDetails(self, reqId, contractDetails):
        logging.debug(f"Received contract details for ReqId: {reqId} - {contractDetails.contract.localSymbol}")
        if reqId not in self.contract_details_results:
            self.contract_details_results[reqId] = []
        self.contract_details_results[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId):
        logging.info(f"✅ Contract details received fully for ReqId: {reqId}")
        if reqId in self.contract_details_events:
            self.contract_details_events[reqId].set()

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
        # Signal the event if the order is considered done (Filled, Cancelled, Inactive)
        if status in ["Filled", "Cancelled", "Inactive", "ApiCancelled"]:
            if orderId in self.order_status_events:
                self.order_status_events[orderId].set()
            # Clean up event if it exists to prevent reuse
            self.order_status_events.pop(orderId, None)

    # --- Helper/Action Methods ---
    def create_market_order(self, direction, quantity):
        """Creates a simple futures market order."""
        order = Order()
        order.action = direction
        order.orderType = "MKT"
        order.totalQuantity = abs(quantity)
        order.transmit = True # Transmit immediately
        # order.tif = "GTC" # Market orders are usually DAY or IOC, GTC might not be suitable
        logging.debug(f"Created MKT order: {direction} {abs(quantity)}")
        return order

    def get_active_futures_contract(self, symbol):
        """Fetches contract details and selects the most suitable active future."""
        base_contract = Contract()
        base_contract.symbol = symbol
        base_contract.secType = "FUT"
        base_contract.currency = "USD"
        base_contract.exchange = self.FUTURES_EXCHANGES.get(symbol, "CME") # Default to CME

        # Apply specific trading class and multiplier if known
        if symbol in self.FUTURES_TRADING_CLASSES:
            base_contract.tradingClass = self.FUTURES_TRADING_CLASSES[symbol]
        if symbol in self.FUTURES_MULTIPLIERS:
            base_contract.multiplier = self.FUTURES_MULTIPLIERS[symbol]

        # Store for potential error context
        self._active_contract_for_error = base_contract

        logging.info(f"\n🔍 Locating active contract for {symbol}...")
        logging.info(f"   Base Contract: Symbol={base_contract.symbol}, Exchange={base_contract.exchange}, TradingClass={getattr(base_contract, 'tradingClass', 'N/A')}")

        req_id = self._get_next_req_id()
        self.contract_details_results[req_id] = []
        self.contract_details_events[req_id] = threading.Event()

        logging.info(f"⏳ Requesting contract details from IB (ReqId: {req_id})...")
        self.reqContractDetails(req_id, base_contract)

        active_contract = None
        try:
            if self.contract_details_events[req_id].wait(timeout=CONTRACT_WAIT_TIMEOUT):
                details_list = self.contract_details_results.get(req_id, [])
                if not details_list:
                    logging.warning(f"⚠️ No contract details returned for {symbol} (ReqId: {req_id})")
                    raise ConnectionError(f"No contract details found for {symbol}. Check symbol/exchange/trading class.")

                logging.info(f"✅ Found {len(details_list)} contracts for {symbol}. Analyzing expirations...")

                # Sort contracts by expiration date (string comparison YYYYMMDD works)
                sorted_contracts = sorted(details_list,
                                       key=lambda d: d.contract.lastTradeDateOrContractMonth)

                # Select the first contract expiring more than ~5 days out
                current_date = datetime.now()
                selected_detail = None
                for detail in sorted_contracts:
                    expiry_str = detail.contract.lastTradeDateOrContractMonth
                    if expiry_str and len(expiry_str) == 8:
                        try:
                            expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
                            days_to_expiry = (expiry_date - current_date).days
                            logging.debug(f"   - {detail.contract.localSymbol} expires {expiry_date.strftime('%Y-%m-%d')} ({days_to_expiry} days)")
                            if days_to_expiry > 5:
                                selected_detail = detail
                                break # Found the first suitable contract
                        except ValueError:
                            logging.warning(f"   - Could not parse expiry date '{expiry_str}' for {detail.contract.localSymbol}")
                    else:
                         logging.warning(f"   - Invalid expiry date format '{expiry_str}' for {detail.contract.localSymbol}")

                if selected_detail:
                    active_contract = selected_detail.contract
                    logging.info(f"🎯 Selected active contract: {active_contract.localSymbol} (Expires: {active_contract.lastTradeDateOrContractMonth})")
                elif sorted_contracts: # Fallback: Use the furthest dated contract if no suitable one found
                    active_contract = sorted_contracts[-1].contract
                    logging.warning(f"⚠️ No contract expiring >5 days found. Using furthest dated: {active_contract.localSymbol} (Expires: {active_contract.lastTradeDateOrContractMonth})")
                else:
                    # Should not happen if details_list was not empty, but handle defensively
                    raise ConnectionError(f"Contract analysis failed for {symbol}. No contracts remained after sorting/filtering.")
            else:
                logging.error(f"❌ Timeout waiting for contract details for {symbol} (ReqId: {req_id})")
                raise TimeoutError(f"Timeout waiting for contract details for {symbol}")
        finally:
            # Clean up event and results
            self.contract_details_events.pop(req_id, None)
            self.contract_details_results.pop(req_id, None)
            self._active_contract_for_error = None # Clear context contract

        if active_contract is None:
             raise ConnectionError(f"Failed to identify a suitable active contract for {symbol}.")

        return active_contract

    def place_futures_market_order_flow(self, symbol, direction, quantity):
        """Handles the flow of placing a futures market order."""
        if self.nextValidOrderId is None:
            logging.error("❌ Cannot place order: nextValidOrderId is not set.")
            return False

        try:
            # 1. Get Active Contract
            contract = self.get_active_futures_contract(symbol)
            logging.info(f"📝 Using contract: {contract.localSymbol} | Multiplier: {contract.multiplier}")

            # 2. Create Market Order
            market_order = self.create_market_order(direction, quantity)
            current_order_id = self.nextValidOrderId

            # 3. Place Order and Wait for Confirmation
            logging.info(f"\n🌲 Placing market order: {direction} {quantity} {contract.localSymbol} (OrderID: {current_order_id}) ...")
            self.order_status_events[current_order_id] = threading.Event()
            self.order_status_cache.pop(current_order_id, None) # Clear previous status

            self.placeOrder(current_order_id, contract, market_order)

            logging.info(f"⏳ Waiting for fill/final status for order {current_order_id}...")
            if self.order_status_events[current_order_id].wait(timeout=ORDER_FILL_TIMEOUT):
                final_status = self.order_status_cache.get(current_order_id, {}).get('status', 'Unknown')
                if final_status == "Filled":
                    filled_qty = self.order_status_cache.get(current_order_id, {}).get('filled', 0)
                    avg_price = self.order_status_cache.get(current_order_id, {}).get('avgFillPrice', 0)
                    logging.info(f"✅ Successfully filled order {current_order_id}: {direction} {filled_qty} {contract.localSymbol} at avg price ${avg_price:.2f}")
                    return True
                else:
                    logging.warning(f"⚠️ Order {current_order_id} ({contract.localSymbol}) finished with status '{final_status}'. Manual check advised.")
                    return False # Order placed but not confirmed filled
            else:
                # Check cache even on timeout
                last_status = self.order_status_cache.get(current_order_id, {}).get('status', 'Unknown')
                logging.warning(f"⌛ Timeout waiting for fill confirmation for order {current_order_id} ({contract.localSymbol}). Last known status: '{last_status}'. Manual check required.")
                return False # Order placed but status uncertain

        except (ConnectionError, TimeoutError, ValueError, Exception) as e:
            logging.error(f"❌ Failed to place market order for {symbol}: {e}", exc_info=True)
            return False
        finally:
            # Ensure next order ID is ready even if this one failed mid-way
            if self.nextValidOrderId is not None:
                 self.nextValidOrderId +=1 # Increment for the next potential order attempt


def run_connection_loop(app_instance):
    """Runs the TWS/Gateway client's message loop."""
    try:
        app_instance.run()
    except Exception as e:
         logging.error(f"❌ Exception in TWS/Gateway message loop: {e}", exc_info=True)

def main():
    """Main function to connect and place the futures order."""
    app = TradeApp()
    port_to_use = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
    trading_mode = "Paper" if IS_PAPER_TRADING else "Live"
    logging.info(f"🚀 Moon Dev connecting to {trading_mode} trading on port {port_to_use}")

    app.connect("127.0.0.1", port_to_use, clientId=3) # Use a different clientId

    con_thread = threading.Thread(target=run_connection_loop, args=(app,), daemon=True)
    con_thread.start()
    logging.info("⏳ Waiting for server connection and nextValidId...")
    time.sleep(2) # Allow time for connection and initial messages

    # Wait longer for nextValidId if needed
    timeout_seconds = 10
    start_time = time.time()
    while app.nextValidOrderId is None and (time.time() - start_time) < timeout_seconds:
        time.sleep(0.2)

    if app.nextValidOrderId is None:
        logging.error("❌ Timeout waiting for valid order ID. Ensure TWS/Gateway is running and API is enabled.")
        app.disconnect()
        return

    # --- Place the Order --- 
    success = False
    try:
        success = app.place_futures_market_order_flow(
            SYMBOL,
            DIRECTION,
            QUANTITY
        )
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during trading logic: {e}", exc_info=True)
    finally:
        if success:
            logging.info("🎉 Moon Dev futures order process completed successfully!")
        else:
            logging.warning("🚦 Moon Dev futures order process completed with issues or did not confirm fill.")

        logging.info("⏳ Allowing time for final messages before disconnect...")
        time.sleep(3)
        app.disconnect()
        logging.info("👋 Moon Dev has disconnected safely.")

if __name__ == "__main__":
    main()
