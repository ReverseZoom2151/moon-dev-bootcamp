# This will allow us to put a bracket order where we can have a stop-loss and take profit all at the same time 

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
SYMBOL = "SNAP"           # Symbol to trade - Available symbols:
                         # SNAP, PLTR, RDDT, APP, NVDA, TSLA, COIN, GOOG, SMCI, SHOP, 
                         # HOOD, CRSPR, LLY, TMSC, MELI, CRWD, RBLX, CROX, XYZ, TTD

                        
DIRECTION = "BUY"        # "BUY" or "SELL"
QUANTITY = 100           # Number of shares to trade
ENTRY_PRICE = 10.50      # Limit price for entry
STOP_LOSS_PRICE = 10       # Stop loss price (renamed for clarity)
TAKE_PROFIT_PRICE = 12.00     # Take profit price (renamed for clarity)

# Connection Configuration
PAPER_PORT = 7497        # Paper trading port
LIVE_PORT = 7496         # Live trading port
IS_PAPER_TRADING = True  # Set to False for live trading

class TradeApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = None
        self.order_placed_event = threading.Event() # Renamed for clarity
        self._active_order_ids = set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        # Ignore informational messages
        if errorCode not in [2104, 2106, 2108, 2158]:
            logging.error(f"❌ Error {errorCode}: {errorString} (ReqId: {reqId})")
            if advancedOrderRejectJson:
                logging.error(f"Advanced Reject Info: {advancedOrderRejectJson}")

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        logging.info(f"🎯 Moon Dev's Next Valid Order ID: {orderId}")

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                   permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        logging.info(
            f"🔄 Order {orderId} status: {status}, Filled: {filled}, Remaining: {remaining}, AvgFillPrice: {avgFillPrice}"
        )
        if orderId in self._active_order_ids:
            if status == "Submitted" or status == "Filled" or status == "PreSubmitted":
                # Consider an order "placed" if it's submitted or filled
                if status == "Filled" and parentId == 0 : # Primary order filled
                    logging.info(f"🎉 Parent Order {orderId} has been filled!")
                self.order_placed_event.set()
            elif status == "Cancelled" or status == "Inactive" or status == "ApiCancelled":
                logging.warning(f"Order {orderId} is {status}.")
                # If a crucial part of the bracket fails, we might need to set the event or handle it
                self._active_order_ids.discard(orderId)
                if not self._active_order_ids: # If all orders are resolved (e.g. cancelled)
                    self.order_placed_event.set()

    def get_contract(self, symbol, sectype="STK", currency="USD", exchange="SMART"):
        """Creates and returns a contract object."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sectype
        contract.currency = currency
        contract.exchange = exchange
        logging.info(f"📜 Created contract for {symbol} ({sectype} on {exchange})")
        return contract

    def create_bracket_order(self, parent_order_id, direction, quantity, entry_price, stop_loss_price, take_profit_price, symbol_log_name):
        """Creates a list of orders for a bracket order."""
        orders = []
        self._active_order_ids.clear() # Clear for new bracket

        # Parent Entry Order
        parent = Order()
        parent.orderId = parent_order_id
        parent.action = direction
        parent.orderType = "LMT"
        parent.totalQuantity = quantity
        parent.lmtPrice = entry_price
        parent.transmit = False  # Transmit is False for parent and first child
        logging.info(f"🎮 Parent Order ({parent.orderId}): {direction} {quantity} {symbol_log_name} @ ${entry_price}")
        orders.append(parent)
        self._active_order_ids.add(parent.orderId)

        # Stop Loss Order
        stop_loss = Order()
        stop_loss.orderId = parent_order_id + 1
        stop_loss.action = "SELL" if direction == "BUY" else "BUY"
        stop_loss.orderType = "STP"
        stop_loss.totalQuantity = quantity
        stop_loss.auxPrice = stop_loss_price
        stop_loss.parentId = parent_order_id
        stop_loss.transmit = False # Transmit is False for parent and first child
        logging.info(f"🛑 Stop Loss Order ({stop_loss.orderId}): {stop_loss.action} {quantity} {symbol_log_name} @ ${stop_loss_price}")
        orders.append(stop_loss)
        self._active_order_ids.add(stop_loss.orderId)

        # Take Profit Order
        take_profit = Order()
        take_profit.orderId = parent_order_id + 2
        take_profit.action = "SELL" if direction == "BUY" else "BUY"
        take_profit.orderType = "LMT"
        take_profit.totalQuantity = quantity
        take_profit.lmtPrice = take_profit_price
        take_profit.parentId = parent_order_id
        take_profit.transmit = True  # Last order in the bracket transmits all
        logging.info(f"💰 Take Profit Order ({take_profit.orderId}): {take_profit.action} {quantity} {symbol_log_name} @ ${take_profit_price}")
        orders.append(take_profit)
        self._active_order_ids.add(take_profit.orderId)
        
        return orders

    def place_bracket_order_flow(self, symbol, direction, quantity, entry_price, stop_loss_val, take_profit_val):
        """Handles the flow of placing a bracket order."""
        if self.nextValidOrderId is None:
            logging.error("❌ Cannot place order: nextValidOrderId is not set.")
            return False

        logging.info(f"\\n🎯 Moon Dev creating bracket order for {symbol}:")
        logging.info(f"Entry: ${entry_price} | Stop Loss: ${stop_loss_val} | Take Profit: ${take_profit_val}")

        contract = self.get_contract(symbol)
        bracket_orders = self.create_bracket_order(
            self.nextValidOrderId,
            direction,
            quantity,
            entry_price,
            stop_loss_val,
            take_profit_val,
            symbol # for logging in create_bracket_order
        )

        self.order_placed_event.clear() # Reset event for this placement

        for order in bracket_orders:
            self.placeOrder(order.orderId, contract, order)
            # IB API recommends a small delay, but usually not needed if transmit flags are set correctly.
            # time.sleep(0.1) 
        
        logging.info("⏳ Waiting for order placement confirmation...")
        if self.order_placed_event.wait(timeout=10): # Increased timeout
            logging.info(f"\\n✨ Moon Dev bracket order for {symbol} processed (check orderStatus for final states)!")
            risk = abs(entry_price - stop_loss_val)
            reward = abs(take_profit_val - entry_price)
            logging.info(f"🎯 Entry Price: ${entry_price}")
            if risk > 0:
                logging.info(f"📉 Stop Loss: ${stop_loss_val} (Risk: ${risk:.2f} per share)")
                logging.info(f"📈 Take Profit: ${take_profit_val} (Reward: ${reward:.2f} per share)")
                logging.info(f"📊 Risk/Reward Ratio: 1:{(reward/risk):.2f}")
            else:
                logging.warning("⚠️ Risk is zero, cannot calculate Risk/Reward ratio.")
            return True
        else:
            logging.warning("⚠️ Timeout waiting for order confirmation via orderStatus.")
            return False

def run_connection_loop(app_instance):
    """Runs the TWS/Gateway client's message loop."""
    app_instance.run()

def main():
    """Main function to connect and place the trade."""
    app = TradeApp()
    port_to_use = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
    trading_mode = "Paper" if IS_PAPER_TRADING else "Live"
    logging.info(f"🚀 Moon Dev connecting to {trading_mode} trading on port {port_to_use}")

    app.connect("127.0.0.1", port_to_use, clientId=1)

    # Start connection thread
    con_thread = threading.Thread(target=run_connection_loop, args=(app,), daemon=True)
    con_thread.start()
    
    logging.info("⏳ Waiting for server connection and nextValidId...")
    time.sleep(2)  # Initial wait for connection

    timeout_seconds = 10 # Increased timeout
    start_time = time.time()
    while app.nextValidOrderId is None and (time.time() - start_time) < timeout_seconds:
        time.sleep(0.2)

    if app.nextValidOrderId is None:
        logging.error("❌ Timeout waiting for valid order ID from TWS/Gateway. Ensure TWS is running and API is enabled.")
        app.disconnect()
        return

    try:
        app.place_bracket_order_flow(
            SYMBOL,
            DIRECTION,
            QUANTITY,
            ENTRY_PRICE,
            STOP_LOSS_PRICE, # Use renamed var
            TAKE_PROFIT_PRICE # Use renamed var
        )
    except Exception as e:
        logging.error(f"❌ An error occurred during trading logic: {e}", exc_info=True)
    finally:
        logging.info("⏳ Allowing time for any final messages before disconnect...")
        time.sleep(3) 
        app.disconnect()
        logging.info("👋 Moon Dev has disconnected safely.")

if __name__ == "__main__":
    main() 