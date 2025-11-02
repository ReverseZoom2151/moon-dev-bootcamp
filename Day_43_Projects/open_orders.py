# This should go ahead and see our current open orders. 

import logging
import threading
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract # Added for type hinting if contract objects are used directly
from typing import Dict, Any # For type hinting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connection Configuration
PAPER_PORT: int = 7497        # Paper trading port
LIVE_PORT: int = 7496         # Live trading port
IS_PAPER_TRADING: bool = True  # Set to False for live trading
WAIT_TIME: int = 10           # Time to wait for orders data (increased slightly)
CLIENT_ID_OPEN_ORDERS: int = 2 # Unique client ID for this script

class OpenOrdersApp(EWrapper, EClient):
    """
    Application to fetch and display open orders from Interactive Brokers.
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.order_df: pd.DataFrame = pd.DataFrame(columns=['PermId', 'ClientId', 'OrderId',
                                                           'Account', 'Symbol', 'SecType',
                                                           'Exchange', 'Action', 'OrderType',
                                                           'TotalQty', 'CashQty', 'LmtPrice',
                                                           'AuxPrice', 'Status'])
        self.orders_event: threading.Event = threading.Event()
        self.connection_event: threading.Event = threading.Event() # For nextValidId
        self.next_valid_order_id: int = -1

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        # Ignore common informational messages or warnings
        ignore_codes = {2104, 2106, 2108, 2158, 2109, 2150, 321, 399, 202, 10167}
        if errorCode not in ignore_codes:
            logging.error(f"Error {errorCode} (ReqId: {reqId}): {errorString}")
            if advancedOrderRejectJson:
                logging.error(f"   Advanced Reject Info: {advancedOrderRejectJson}")
        # For critical errors that might block, signal relevant events
        if errorCode in [200, 162, 322, 504]: # 504: Not connected
             self.orders_event.set() # Unblock if waiting for orders
             self.connection_event.set() # Unblock if waiting for connection


    def nextValidId(self, orderId: int):
        """Receives the next valid order ID from TWS/Gateway after connection."""
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        logging.info(f"Received next valid order ID: {orderId}")
        self.connection_event.set() # Signal that connection is established and ID received


    def openOrder(self, orderId: int, contract: Contract, order: Any, orderState: Any):
        """Callback receiving open order details."""
        super().openOrder(orderId, contract, order, orderState)
        order_details: Dict[str, Any] = {
            "PermId": order.permId,
            "ClientId": order.clientId,
            "OrderId": orderId,
            "Account": order.account,
            "Symbol": contract.symbol,
            "SecType": contract.secType,
            "Exchange": contract.exchange,
            "Action": order.action,
            "OrderType": order.orderType,
            "TotalQty": order.totalQuantity,
            "CashQty": order.cashQty,
            "LmtPrice": order.lmtPrice,
            "AuxPrice": order.auxPrice,
            "Status": orderState.status
        }
        new_row = pd.DataFrame([order_details])
        self.order_df = pd.concat([self.order_df, new_row], ignore_index=True)
        logging.info(f"Order received: {order.action} {order.totalQuantity} {contract.symbol} @ "
                     f"{order.lmtPrice if order.lmtPrice != 0 and order.lmtPrice != 1.7976931348623157e+308 else order.auxPrice if order.auxPrice != 0 and order.auxPrice != 1.7976931348623157e+308 else 'N/A'}")

    def openOrderEnd(self):
        """Callback indicating all open orders have been sent."""
        super().openOrderEnd()
        logging.info("All open orders have been received.")
        self.orders_event.set() # Signal that order fetching is complete

    def connect_to_ib(self, host: str = "127.0.0.1", port: int = None, client_id: int = CLIENT_ID_OPEN_ORDERS) -> bool:
        """Establishes connection to TWS/Gateway."""
        if port is None:
            port = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
        
        trading_mode = "Paper" if port == PAPER_PORT else "Live"
        logging.info(f"Connecting to {trading_mode} Trading on {host}:{port} (ClientId: {client_id})...")
        
        self.connect(host, port, clientId=client_id)
        
        # Start the message processing thread
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logging.info("Connection thread started.")

        # Wait for nextValidId
        if self.connection_event.wait(timeout=WAIT_TIME):
            logging.info("Successfully connected and received nextValidId.")
            return True
        else:
            logging.error(f"Connection timeout after {WAIT_TIME} seconds.")
            self.disconnect_from_ib() # Attempt to clean up
            return False

    def disconnect_from_ib(self):
        """Disconnects from TWS/Gateway."""
        if self.isConnected():
            logging.info("Disconnecting from IB...")
            self.disconnect()
            # Give a moment for disconnect process
            if self.thread.is_alive():
                 self.thread.join(timeout=2) # Wait for thread to finish
            logging.info("Disconnected safely.")
        else:
            logging.info("Already disconnected or connection failed.")

    def fetch_open_orders(self) -> pd.DataFrame:
        """Requests and retrieves open orders."""
        if not self.isConnected():
            logging.error("Not connected. Cannot fetch open orders.")
            return pd.DataFrame()

        logging.info("Fetching open orders...")
        self.order_df = pd.DataFrame(columns=self.order_df.columns) # Clear previous data
        self.orders_event.clear() # Reset event
        
        self.reqOpenOrders()
        
        if self.orders_event.wait(timeout=WAIT_TIME):
            if not self.order_df.empty:
                logging.info(f"Retrieved {len(self.order_df)} open order(s).")
            else:
                logging.info("No open orders found.")
        else:
            logging.warning("Timeout waiting for open orders.")
            # Potentially return partial data if any received, or empty
            if not self.order_df.empty:
                 logging.warning(f"Returning {len(self.order_df)} partially received orders before timeout.")
            else:
                 logging.info("No orders received before timeout.")
        return self.order_df.copy() # Return a copy

def display_orders(df: pd.DataFrame):
    """Formats and prints the orders DataFrame."""
    if df.empty:
        logging.info("📭 No open orders to display.")
        return

    display_df = df.copy()
    # Format price columns (handle potential string or float max values)
    for col in ['LmtPrice', 'AuxPrice']:
        display_df[col] = display_df[col].apply(
            lambda x: f'${float(x):,.2f}' if pd.notna(x) and x != 0 and x != float('inf') and x != 1.7976931348623157e+308 else '-'
        )
    # Format quantity
    display_df['TotalQty'] = display_df['TotalQty'].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) else '-')
    
    logging.info("📊 Open Orders Summary:")
    header = "=" * 120
    logging.info(header)
    # Select specific columns for a cleaner display, ensure they exist
    cols_to_display = ['Symbol', 'Action', 'OrderType', 'TotalQty', 'LmtPrice', 'AuxPrice', 'Status']
    existing_cols_to_display = [col for col in cols_to_display if col in display_df.columns]
    
    # Convert DataFrame to string for logging
    # Using print directly here for formatted table output as logging might mangle it.
    # Or, convert to a list of strings for logging line by line.
    df_string = display_df[existing_cols_to_display].to_string(index=False)
    print(df_string) # Using print for table formatting. For pure logging, reformat.
    logging.info(header)

def main():
    """Main function to connect, fetch orders, display, and disconnect."""
    logging.info("--- Starting Open Orders Script ---")
    app = OpenOrdersApp()

    if app.connect_to_ib(client_id=CLIENT_ID_OPEN_ORDERS):
        try:
            open_orders_df = app.fetch_open_orders()
            display_orders(open_orders_df)
        except Exception as e:
            logging.error(f"An error occurred during order fetching: {e}", exc_info=True)
        finally:
            app.disconnect_from_ib()
    else:
        logging.error("Failed to connect to Interactive Brokers. Exiting.")
    
    logging.info("--- Open Orders Script Finished ---")

if __name__ == "__main__":
    main()

