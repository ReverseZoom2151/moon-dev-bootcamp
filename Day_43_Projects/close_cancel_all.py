# This file will close all positions and cancel any open orders good for the end of the day. 

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import threading
import time
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connection Configuration
PAPER_PORT = 7497        # Paper trading port
LIVE_PORT = 7496         # Live trading port
IS_PAPER_TRADING = True  # Set to False for live trading
WAIT_TIME_POSITIONS = 5 # Increased timeout for position data
WAIT_TIME_CONTRACTS = 5 # Timeout for contract details
WAIT_TIME_FILL = 10      # Timeout waiting for fill confirmation

class TradeApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self._pos_data = [] # Internal list to store position dictionaries
        self.pos_df = pd.DataFrame(columns=['Account', 'Symbol', 'SecType', 'Position', 'Avg cost', 'LocalSymbol']) # Public DataFrame
        self.nextValidOrderId = None
        self._next_req_id = 1 # Internal request ID counter

        # Events for synchronization
        self.positions_event = threading.Event()
        self.contract_details_events = {} # reqId -> event
        self.order_status_events = {} # orderId -> event

        # Data storage
        self.contract_details_results = {} # reqId -> list of contractDetails
        self.order_status_cache = {} # orderId -> status dict

        # Static data
        self.FUTURES_EXCHANGES = {
            "ES": "CME", "NQ": "CME", "CL": "NYMEX", "GC": "COMEX", "SI": "COMEX",
            "ZB": "CBOT", "ZN": "CBOT", "6E": "CME", "LBR": "CME" # Added more from original
        }

    def _get_next_req_id(self):
        """Gets the next available request ID."""
        req_id = self._next_req_id
        self._next_req_id += 1
        return req_id

    # --- EWrapper Methods ---
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2108, 2158, 399]: # 399 is often order warnings
            logging.error(f"❌ Error {errorCode}: {errorString} (ReqId: {reqId})")
            if advancedOrderRejectJson:
                logging.error(f"Advanced Reject Info: {advancedOrderRejectJson}")
        # Signal event if it's related to a contract details request failure
        if reqId in self.contract_details_events:
            self.contract_details_events[reqId].set() # Signal completion even on error

    def contractDetails(self, reqId, contractDetails):
        logging.debug(f"Received contract details for ReqId: {reqId}")
        if reqId not in self.contract_details_results:
            self.contract_details_results[reqId] = []
        self.contract_details_results[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId):
        logging.info(f"✅ Contract details received fully for ReqId: {reqId}")
        if reqId in self.contract_details_events:
            self.contract_details_events[reqId].set()

    def position(self, account, contract, position, avgCost):
        super().position(account, contract, position, avgCost)
        # Store raw data first
        pos_dict = {
            "Account": account,
            "Symbol": contract.symbol,
            "SecType": contract.secType,
            "Position": position,
            "Avg cost": avgCost,
            "LocalSymbol": getattr(contract, 'localSymbol', contract.symbol), # Use getattr for safety
            "ConId": contract.conId # Store conId for potential use
        }
        self._pos_data.append(pos_dict)
        logging.debug(f"Received position: {pos_dict['Symbol']}, Pos: {pos_dict['Position']}")

    def positionEnd(self):
        logging.info("🎯 Moon Dev position data stream ended.")
        # Process raw data into DataFrame
        if self._pos_data:
            self.pos_df = pd.DataFrame(self._pos_data)
            # Drop duplicates based on key identifiers, keep first entry
            self.pos_df = self.pos_df.drop_duplicates(
                subset=['Account', 'Symbol', 'SecType', 'LocalSymbol', 'ConId'], keep='first'
            )
            logging.info(f"📊 Processed {len(self.pos_df)} unique positions.")
        else:
            self.pos_df = pd.DataFrame(columns=['Account', 'Symbol', 'SecType', 'Position', 'Avg cost', 'LocalSymbol', 'ConId'])
            logging.info("📊 No position data received.")
        self.positions_event.set() # Signal completion

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        logging.info(f"🔑 Moon Dev's Next Valid Order ID: {orderId}")

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
             # Clean up event if it exists
             self.order_status_events.pop(orderId, None)

    # --- Helper/Action Methods ---
    def get_active_futures_contract(self, symbol):
        """Gets the active futures contract details for a given symbol."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "FUT"
        contract.currency = "USD"
        # Basic exchange mapping (can be expanded)
        contract.exchange = self.FUTURES_EXCHANGES.get(symbol, "CME") # Default to CME if not mapped

        req_id = self._get_next_req_id()
        logging.info(f"⏳ Requesting contract details for {symbol} (ReqId: {req_id})...")
        self.contract_details_results[req_id] = []
        self.contract_details_events[req_id] = threading.Event()

        self.reqContractDetails(req_id, contract)

        if self.contract_details_events[req_id].wait(timeout=WAIT_TIME_CONTRACTS):
            details = self.contract_details_results.get(req_id, [])
            if details:
                # Typically, the last one in the list returned by IB is the most active
                # Or sort by expiration if needed: sorted(details, key=lambda x: x.contract.lastTradeDateOrContractMonth)
                active_contract = details[-1].contract # Assuming last is most active
                logging.info(f"✅ Found active contract for {symbol}: {active_contract.localSymbol}")
                del self.contract_details_events[req_id]
                del self.contract_details_results[req_id]
                return active_contract
            else:
                logging.warning(f"⚠️ No contract details returned for {symbol} (ReqId: {req_id})")
        else:
            logging.error(f"❌ Timeout waiting for contract details for {symbol} (ReqId: {req_id})")
        
        # Cleanup event and results in case of timeout or no details
        self.contract_details_events.pop(req_id, None)
        self.contract_details_results.pop(req_id, None)
        raise Exception(f"Could not find active contract for {symbol}")

    def get_contract(self, symbol, sec_type, local_symbol=None, exchange="SMART"):
        """Creates a contract object based on security type."""
        if sec_type == "STK":
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.currency = "USD"
            contract.exchange = exchange
            logging.debug(f"Created STK contract for {symbol}")
            return contract
        elif sec_type == "FUT":
            # FUT requires finding the active contract if local_symbol isn't provided
            if local_symbol:
                contract = Contract()
                contract.symbol = symbol
                contract.secType = sec_type
                contract.currency = "USD"
                contract.localSymbol = local_symbol
                contract.exchange = self.FUTURES_EXCHANGES.get(symbol, "CME") # Use class attribute mapping
                logging.debug(f"Created FUT contract for {local_symbol}")
                return contract
            else:
                logging.warning(f"LocalSymbol not provided for FUT {symbol}, fetching active contract...")
                return self.get_active_futures_contract(symbol)
        else:
            raise ValueError(f"Unsupported security type: {sec_type}")

    def create_market_order(self, direction, quantity):
        """Creates a simple market order."""
        order = Order()
        order.action = direction
        order.orderType = "MKT"
        order.totalQuantity = abs(quantity) # Ensure positive quantity
        order.tif = "GTC" # Good Till Cancelled - appropriate for closing positions
        order.transmit = True
        logging.debug(f"Created MKT order: {direction} {abs(quantity)}")
        return order

    def cancel_and_close_positions(self):
        """Main flow to cancel open orders and close all positions."""
        if self.nextValidOrderId is None:
            logging.error("❌ Cannot proceed: nextValidOrderId is not set.")
            return

        # 1. Cancel all open orders
        logging.info("🔄 Moon Dev requesting cancellation of all open orders...")
        self.reqGlobalCancel()
        time.sleep(1) # Allow time for cancellations to be processed

        # 2. Request and wait for positions
        logging.info("📊 Moon Dev fetching all open positions...")
        self._pos_data = [] # Clear previous position data
        self.positions_event.clear()
        self.reqPositions()

        if not self.positions_event.wait(timeout=WAIT_TIME_POSITIONS):
            logging.error("❌ Timeout waiting for positions data. Cannot close positions.")
            return

        # 3. Process and close positions
        if self.pos_df.empty:
            logging.info("✨ No open positions found - Moon Dev is already in cash!")
            return

        logging.info(f"🔍 Found {len(self.pos_df)} unique position(s) to close:")
        order_id = self.nextValidOrderId

        positions_to_close = self.pos_df[self.pos_df['Position'] != 0].copy()

        for _, row in positions_to_close.iterrows():
            current_order_id = order_id
            try:
                ticker = row["Symbol"]
                sec_type = row["SecType"]
                local_symbol = row.get("LocalSymbol") # Use .get for safety
                position_size = float(row["Position"])
                quantity_to_close = abs(position_size)

                if quantity_to_close == 0:
                    logging.warning(f"⚠️ Skipping {local_symbol or ticker} - zero position size detected after processing.")
                    continue

                direction = "SELL" if position_size > 0 else "BUY"
                logging.info(f"Initiating close for {local_symbol or ticker}: {direction} {quantity_to_close} shares/contracts (OrderID: {current_order_id})...")
                
                contract = self.get_contract(ticker, sec_type, local_symbol)
                market_order = self.create_market_order(direction, quantity_to_close)

                # Prepare to wait for this order
                self.order_status_events[current_order_id] = threading.Event()
                self.order_status_cache.pop(current_order_id, None) # Clear previous status

                self.placeOrder(current_order_id, contract, market_order)
                
                # Wait for fill confirmation using the event
                logging.info(f"⏳ Waiting for fill/final status for order {current_order_id} ({local_symbol or ticker})...")
                if self.order_status_events[current_order_id].wait(timeout=WAIT_TIME_FILL):
                    final_status = self.order_status_cache.get(current_order_id, {}).get('status', 'Unknown')
                    if final_status == "Filled":
                        filled_qty = self.order_status_cache.get(current_order_id, {}).get('filled', 0)
                        avg_price = self.order_status_cache.get(current_order_id, {}).get('avgFillPrice', 0)
                        logging.info(f"✅ Successfully closed {filled_qty} of {local_symbol or ticker} at avg price ${avg_price:.2f} (Order {current_order_id})")
                    else:
                         logging.warning(f"⚠️ Order {current_order_id} ({local_symbol or ticker}) finished with status '{final_status}'. Manual check advised.")
                else:
                    logging.warning(f"⌛ Timeout waiting for fill confirmation for order {current_order_id} ({local_symbol or ticker}). Status: {self.order_status_cache.get(current_order_id, {}).get('status', 'Unknown')}")
                
                order_id += 1 # Increment for next potential order
                # Remove event after handling
                self.order_status_events.pop(current_order_id, None)
                time.sleep(0.2) # Small delay before next closing order

            except Exception as pos_e:
                logging.error(f"❌ Error processing position {row.get('LocalSymbol') or row.get('Symbol')}: {pos_e}", exc_info=True)
                # Ensure order_id is incremented even if one position fails
                order_id += 1
                # Remove event if created
                self.order_status_events.pop(current_order_id, None)
                continue # Attempt to close next position

        logging.info("🎉 Moon Dev has finished processing all positions!")

def run_connection_loop(app_instance):
    """Runs the TWS/Gateway client's message loop."""
    app_instance.run()

def main():
    """Main function to connect and manage closing process."""
    app = TradeApp()
    port_to_use = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
    trading_mode = "Paper" if IS_PAPER_TRADING else "Live"
    logging.info(f"🚀 Moon Dev connecting to {trading_mode} trading on port {port_to_use}")

    app.connect("127.0.0.1", port_to_use, clientId=2) # Use a different clientId if running alongside bracket order script

    con_thread = threading.Thread(target=run_connection_loop, args=(app,), daemon=True)
    con_thread.start()
    logging.info("⏳ Waiting for server connection and nextValidId...")
    time.sleep(2)

    timeout_seconds = 10
    start_time = time.time()
    while app.nextValidOrderId is None and (time.time() - start_time) < timeout_seconds:
        time.sleep(0.2)

    if app.nextValidOrderId is None:
        logging.error("❌ Timeout waiting for valid order ID. Ensure TWS/Gateway is running and API is enabled.")
        app.disconnect()
        return

    try:
        app.cancel_and_close_positions()
    except Exception as e:
        logging.error(f"❌ An error occurred during the close/cancel process: {e}", exc_info=True)
    finally:
        logging.info("⏳ Allowing time for final messages before disconnect...")
        time.sleep(3)
        app.disconnect()
        logging.info("👋 Moon Dev has disconnected safely. Portfolio should be 100% cash soon!")

if __name__ == "__main__":
    main()

