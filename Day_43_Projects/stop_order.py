import logging
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from typing import Optional, Any # For type hinting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Trading Configuration
SYMBOL: str = "SNAP"           # Symbol to trade
DIRECTION: str = "SELL"       # "BUY" or "SELL"
QUANTITY: float = 100           # Number of shares to trade (use float for IB API compatibility)
STOP_PRICE: float = 10.84      # Stop price for the order

# Connection Configuration
PAPER_PORT: int = 7497
LIVE_PORT: int = 7496
IS_PAPER_TRADING: bool = True
WAIT_TIME_CONNECT: int = 10
WAIT_TIME_ORDER: int = 60 # Increased wait time for order status
CLIENT_ID_STOP_ORDER: int = 4 # Unique client ID

class StopOrderApp(EWrapper, EClient):
    """Application to place a stop order with Interactive Brokers."""
    def __init__(self):
        EClient.__init__(self, self)
        self.next_valid_order_id: Optional[int] = None
        self.connection_event: threading.Event = threading.Event()
        self.order_status_event: threading.Event = threading.Event()
        self.order_final_status: Optional[str] = None # To store final status like Filled, Cancelled

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        ignore_codes = {2104, 2106, 2108, 2158, 2109, 2150, 321, 399, 202, 10167}
        if errorCode not in ignore_codes:
            logging.error(f"Error {errorCode} (ReqId: {reqId}): {errorString}")
            if advancedOrderRejectJson:
                logging.error(f"   Advanced Reject Info: {advancedOrderRejectJson}")
        # Signal events on critical errors to prevent blocking
        if errorCode in [200, 162, 322, 504, 502]: # 502: Couldn't connect
             logging.warning(f"Signaling events due to error code: {errorCode}")
             self.connection_event.set()
             self.order_status_event.set()

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        logging.info(f"Received next valid order ID: {orderId}")
        self.connection_event.set()

    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                    avgFillPrice: float, permId: int, parentId: int,
                    lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        logging.info(f"Order {orderId} Status: {status}, Filled: {filled}, Remaining: {remaining}, AvgFillPrice: {avgFillPrice}")
        # Check for terminal states
        if status in ["Filled", "Cancelled", "ApiCancelled", "Inactive", "PendingCancel"]:
            self.order_final_status = status
            self.order_status_event.set()
            if status == "Filled":
                logging.info(f"✅ Order {orderId} filled at average price: ${avgFillPrice:.2f}")
            elif status == "Cancelled" or status == "ApiCancelled":
                logging.warning(f"⚠️ Order {orderId} was cancelled.")
            else:
                logging.warning(f"Order {orderId} reached terminal state: {status}")

    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: Any):
        # Log details of the order when it's acknowledged by TWS
        logging.info(f"Order {orderId} acknowledged by TWS: {order.action} {order.totalQuantity} {contract.symbol} {order.orderType} STP @ {order.auxPrice} Status: {orderState.status}")

    def connect_to_ib(self, host: str = "127.0.0.1", port: int = None, client_id: int = CLIENT_ID_STOP_ORDER) -> bool:
        if port is None:
            port = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
        trading_mode = "Paper" if port == PAPER_PORT else "Live"
        logging.info(f"Connecting to {trading_mode} Trading on {host}:{port} (ClientId: {client_id})...")
        self.connect(host, port, clientId=client_id)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logging.info("Connection thread started.")
        if self.connection_event.wait(timeout=WAIT_TIME_CONNECT):
            logging.info("Successfully connected and received nextValidId.")
            return True
        else:
            logging.error(f"Connection timeout after {WAIT_TIME_CONNECT} seconds.")
            self.disconnect_from_ib()
            return False

    def disconnect_from_ib(self):
        if self.isConnected():
            logging.info("Disconnecting from IB...")
            self.disconnect()
            if self.thread.is_alive():
                self.thread.join(timeout=2)
            logging.info("Disconnected safely.")
        else:
            logging.info("Already disconnected or connection failed.")

    @staticmethod
    def create_stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = currency
        contract.exchange = exchange
        return contract

    @staticmethod
    def create_stop_order(action: str, quantity: float, stop_price: float) -> Order:
        order = Order()
        order.action = action
        order.orderType = "STP"
        order.totalQuantity = quantity
        order.auxPrice = stop_price
        # Optional: Add TIF (Time-in-Force), e.g., order.tif = "GTC"
        return order

    def place_stop_order(self, contract: Contract, order: Order) -> bool:
        if not self.isConnected():
            logging.error("Not connected. Cannot place order.")
            return False
        if self.next_valid_order_id is None:
            logging.error("Invalid order ID. Connection might not be fully established.")
            return False

        order_id = self.next_valid_order_id
        self.next_valid_order_id += 1 # Increment for potential future orders in the same session
        self.order_status_event.clear()
        self.order_final_status = None

        logging.info(f"Placing {order.action} STOP order for {order.totalQuantity} {contract.symbol} @ {order.auxPrice} (OrderId: {order_id})...")
        self.placeOrder(order_id, contract, order)

        logging.info(f"Waiting up to {WAIT_TIME_ORDER} seconds for order {order_id} to reach a final state...")
        if self.order_status_event.wait(timeout=WAIT_TIME_ORDER):
            logging.info(f"Order {order_id} processing complete. Final Status: {self.order_final_status}")
            return self.order_final_status == "Filled" # Consider success only if filled
        else:
            logging.warning(f"Timeout waiting for final status for order {order_id}. Check TWS/Client Portal.")
            return False # Indicate timeout/uncertainty

def main():
    logging.info("--- Starting Stop Order Script ---")
    app = StopOrderApp()

    if app.connect_to_ib(client_id=CLIENT_ID_STOP_ORDER):
        order_placed_successfully = False
        try:
            contract = StopOrderApp.create_stock_contract(SYMBOL)
            order = StopOrderApp.create_stop_order(DIRECTION, QUANTITY, STOP_PRICE)
            order_placed_successfully = app.place_stop_order(contract, order)
            
            if order_placed_successfully:
                logging.info(f"Stop order for {SYMBOL} was successfully filled.")
            else:
                logging.warning(f"Stop order for {SYMBOL} did not complete as expected (Timeout or non-Filled status). Please verify manually.")

        except Exception as e:
            logging.error(f"An error occurred during order placement: {e}", exc_info=True)
        finally:
            app.disconnect_from_ib()
    else:
        logging.error("Failed to connect to Interactive Brokers. Exiting.")
    
    logging.info("--- Stop Order Script Finished ---")

if __name__ == "__main__":
    main()
