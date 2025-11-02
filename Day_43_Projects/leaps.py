'''
LEAPs option strategy 

Hand traders can't really trade options very well because they always turn into short-term options. 
Leaps are a solid way to get a little bit of leverage. It's just hard to do as a human. 
So I'm going to use interactive brokers here in order to build out a leap strategy that can go long or short. 
'''

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime, timedelta
import pandas as pd
import time
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Trading Configuration
SYMBOL = "NVDA"           # Default symbol to trade
MONTHS_OUT = 6           # Target months out for LEAP options
MAX_WAIT_TIME = 10        # Default maximum time to wait for data in seconds
PRICE_WAIT_TIMEOUT = 5   # Timeout specifically for price ticks
OPTION_PRICE_TIMEOUT = 15 # Timeout for fetching multiple option prices
USE_DELAYED_DATA = True   # Use delayed data instead of real-time (requires no market data subscription)
NUM_STRIKES = 10          # Number of strikes to show (N/2 above and N/2 below current price)

# Connection Configuration
PAPER_PORT = 7497         # Paper trading port
LIVE_PORT = 7496          # Live trading port
IS_PAPER_TRADING = True   # Set to False for live trading

class LeapOptionsTrader(EClient, EWrapper):
    # Request ID constants for clarity
    REQ_ID_CONTRACT_DETAILS = 1
    REQ_ID_STOCK_PRICE = 2
    REQ_ID_OPTION_CHAIN = 3
    REQ_ID_OPTION_PRICE_BASE = 1000 # Start option price reqIds from here

    def __init__(self):
        EClient.__init__(self, self)
        self._next_req_id_counter = self.REQ_ID_OPTION_PRICE_BASE # Counter for dynamic reqIds
        self.nextValidOrderId = None

        # Events for synchronization
        self.connected_event = threading.Event()
        self.contract_details_event = threading.Event()
        self.stock_price_event = threading.Event()
        self.option_chain_event = threading.Event()
        self._option_price_events = {} # reqId -> threading.Event()

        # Data storage
        self._stock_contract_details = None
        self._current_stock_price = None
        self._option_chain_params = {} # Store params like expirations, strikes
        self._options_price_data = {} # reqId -> {'contract': Contract, 'bid': None, 'ask': None, 'last': None, 'received_event': Event}

        logging.info("🌙 Moon Dev's LEAP Options Trader initialized! 🚀")

    def _get_next_req_id(self):
        """Generates unique request IDs for option prices."""
        req_id = self._next_req_id_counter
        self._next_req_id_counter += 1
        return req_id

    # --- EWrapper Methods ---
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        ignore_codes = {2104, 2106, 2108, 2158, 10091, 10167, 322, 200} # Info, delayed data, no security def
        if errorCode not in ignore_codes:
            logging.error(f"❌ Error {errorCode}: {errorString} (ReqId: {reqId})")

        # Signal events on critical errors to prevent deadlocks
        if reqId == self.REQ_ID_CONTRACT_DETAILS and not self.contract_details_event.is_set():
            self.contract_details_event.set()
        elif reqId == self.REQ_ID_STOCK_PRICE and not self.stock_price_event.is_set():
            self.stock_price_event.set()
        elif reqId == self.REQ_ID_OPTION_CHAIN and not self.option_chain_event.is_set():
            self.option_chain_event.set()
        elif reqId in self._option_price_events and not self._option_price_events[reqId].is_set():
            logging.warning(f"Signaling option price event for ReqId {reqId} due to error {errorCode}.")
            self._option_price_events[reqId].set() # Signal completion even on error

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        logging.info(f"🔑 Moon Dev's Next Valid Order ID: {orderId}")
        if not self.connected_event.is_set():
            self.connected_event.set() # Signal connection established

    def connectionClosed(self):
        logging.warning("🔌 Connection closed.")
        self.connected_event.clear()

    def contractDetails(self, reqId, contractDetails):
        if reqId == self.REQ_ID_CONTRACT_DETAILS:
            self._stock_contract_details = contractDetails
            logging.info(f"📝 Received contract details for {contractDetails.contract.symbol}")

    def contractDetailsEnd(self, reqId):
        if reqId == self.REQ_ID_CONTRACT_DETAILS:
            logging.info("✅ Contract details request complete.")
            self.contract_details_event.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        # Tick types for delayed data often used: 66(bid), 67(ask), 68(last)
        # Tick types for real-time: 1(bid), 2(ask), 4(last), 9(close)
        is_delayed = tickType >= 66
        price_type = 'delayed' if is_delayed else 'real-time'

        if reqId == self.REQ_ID_STOCK_PRICE:
            # Use Last price preferably, fallback to Ask/Bid if needed
            if tickType in [4, 68]: # Last, Delayed Last
                self._current_stock_price = price
                logging.info(f"📈 Stock Price ({price_type} Last): ${price}")
                self.stock_price_event.set()
            elif self._current_stock_price is None and tickType in [2, 67]: # Ask, Delayed Ask
                 self._current_stock_price = price # Use Ask as fallback
                 logging.debug(f"📈 Stock Price ({price_type} Ask): ${price} (using as fallback)")
                 # Don't set event yet, prefer Last
            elif self._current_stock_price is None and tickType in [1, 66]: # Bid, Delayed Bid
                 self._current_stock_price = price # Use Bid as last resort fallback
                 logging.debug(f"📈 Stock Price ({price_type} Bid): ${price} (using as last resort)")
                 # Don't set event yet

        elif reqId in self._options_price_data:
            option_info = self._options_price_data[reqId]
            updated = False
            if tickType in [1, 66]: # Bid, Delayed Bid
                option_info['bid'] = price
                updated = True
            elif tickType in [2, 67]: # Ask, Delayed Ask
                option_info['ask'] = price
                updated = True
            elif tickType in [4, 68]: # Last, Delayed Last
                option_info['last'] = price
                updated = True

            if updated and not option_info['received_event'].is_set():
                # Signal as soon as we have any price component (bid, ask, or last)
                logging.debug(f"Received {price_type} price for option ReqId {reqId}, signaling event.")
                option_info['received_event'].set()
        else:
            logging.debug(f"Received tick for unknown ReqId: {reqId}, Type: {tickType}, Price: {price}")

    def tickSize(self, reqId, tickType, size):
        # Can be used to check volume if needed, but ignored for now
        pass

    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass,
                                        multiplier, expirations, strikes):
        if reqId == self.REQ_ID_OPTION_CHAIN:
            if exchange == "SMART": # Focus on SMART-routed options
                logging.info(f"Received option parameters for {tradingClass} on {exchange}")
                if not self._option_chain_params: # Store the first SMART result
                    self._option_chain_params = {
                        'exchange': exchange,
                        'tradingClass': tradingClass,
                        'multiplier': multiplier,
                        'expirations': set(expirations), # Use set for faster lookup
                        'strikes': set(strikes)
                    }
                else: # Merge expirations/strikes from potentially fragmented results
                    self._option_chain_params['expirations'].update(expirations)
                    self._option_chain_params['strikes'].update(strikes)
            else:
                logging.debug(f"Ignoring option parameters from exchange: {exchange}")

    def securityDefinitionOptionParameterEnd(self, reqId):
        if reqId == self.REQ_ID_OPTION_CHAIN:
            logging.info("✅ Option chain parameter request complete.")
            if not self._option_chain_params:
                logging.error("❌ No SMART option parameters received!")
            self.option_chain_event.set()

    # --- Helper/Action Methods ---
    def _request_stock_price(self, symbol):
        """Internal method to request stock price."""
        if not self._stock_contract_details:
            logging.error("Cannot request stock price without contract details.")
            return False

        logging.info(f"💰 Requesting {'delayed' if USE_DELAYED_DATA else 'real-time'} price for {symbol}...")
        self.stock_price_event.clear()
        self._current_stock_price = None

        market_data_type = 3 if USE_DELAYED_DATA else 1
        self.reqMarketDataType(market_data_type)
        time.sleep(0.1) # Allow change to propagate

        # Use generic ticks that work for both delayed and real-time
        # 100: Option Volume, 101: Option Open Interest, 104: Historical Volatility, 106: Avg Option Volume
        # 165: Misc Stats, 233: RT Volume timestamp, 258: RT Volume
        # For price: Use standard Bid/Ask/Last (1, 2, 4) and Delayed (66, 67, 68)
        generic_ticks = ""
        snapshot = False # Streaming updates are better
        self.reqMktData(self.REQ_ID_STOCK_PRICE, self._stock_contract_details.contract, generic_ticks, snapshot, False, [])
        return True

    def _request_contract_details(self, symbol):
        """Internal method to request stock contract details."""
        logging.info(f"🔎 Requesting contract details for {symbol}...")
        self.contract_details_event.clear()
        self._stock_contract_details = None

        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        # contract.primaryExch = "NASDAQ" # Optional: Specify primary exchange if needed

        self.reqContractDetails(self.REQ_ID_CONTRACT_DETAILS, contract)
        return True

    def _request_option_chain_params(self, symbol):
        """Internal method to request option chain parameters."""
        if not self._stock_contract_details:
            logging.error("Cannot request option chain without stock contract details.")
            return False

        logging.info(f"⛓️ Requesting option chain parameters for {symbol}...")
        self.option_chain_event.clear()
        self._option_chain_params = {}

        self.reqSecDefOptParams(self.REQ_ID_OPTION_CHAIN, symbol, "", "STK", self._stock_contract_details.contract.conId)
        return True

    def _find_target_leap_expiration(self):
        """Finds the best LEAP expiration date from the received parameters."""
        if not self._option_chain_params or 'expirations' not in self._option_chain_params:
            logging.error("Option chain parameters not available.")
            return None

        target_min_date = datetime.now() + timedelta(days=MONTHS_OUT * 30 - 15) # Allow some flexibility
        target_max_date = datetime.now() + timedelta(days=MONTHS_OUT * 30 + 45) # Look a bit further out

        valid_expirations = []
        for exp_str in self._option_chain_params['expirations']:
            try:
                exp_date = datetime.strptime(exp_str, '%Y%m%d')
                if target_min_date <= exp_date <= target_max_date:
                    valid_expirations.append((exp_date, exp_str))
            except ValueError:
                logging.warning(f"Could not parse expiration date: {exp_str}")
                continue

        if not valid_expirations:
            logging.error(f"No suitable LEAP expiration found around {MONTHS_OUT} months out.")
            # Fallback: Find closest after target_min_date
            min_diff = float('inf')
            fallback_exp = None
            now = datetime.now()
            for exp_str in self._option_chain_params['expirations']:
                 try:
                     exp_date = datetime.strptime(exp_str, '%Y%m%d')
                     if exp_date > now:
                         diff = abs((exp_date - target_min_date).days)
                         if diff < min_diff:
                             min_diff = diff
                             fallback_exp = exp_str
                 except ValueError:
                     continue
            if fallback_exp:
                logging.warning(f"Using fallback expiration: {fallback_exp}")
                return fallback_exp
            else:
                 return None

        # Sort by closest to the ideal target date (MONTHS_OUT * 30 days)
        ideal_target_date = datetime.now() + timedelta(days=MONTHS_OUT * 30)
        valid_expirations.sort(key=lambda x: abs(x[0] - ideal_target_date))

        chosen_exp_str = valid_expirations[0][1]
        chosen_exp_date = valid_expirations[0][0]
        logging.info(f"🎯 Selected LEAP expiration: {chosen_exp_date.strftime('%Y-%m-%d')} ({chosen_exp_str})")
        return chosen_exp_str

    def _get_relevant_strikes(self, current_price):
         """Filters and returns relevant strikes around the current price."""
         if not self._option_chain_params or 'strikes' not in self._option_chain_params:
            logging.error("Option strikes not available.")
            return []
         if current_price is None:
            logging.error("Current stock price is not available.")
            return []

         all_strikes = sorted(list(self._option_chain_params['strikes'])) # Ensure sorted list
         if not all_strikes:
             logging.error("Strike list is empty.")
             return []

         strikes_array = pd.Series(all_strikes)
         # Find index of strike closest to current price
         closest_strike_index = (strikes_array - current_price).abs().idxmin()

         # Calculate start and end indices for NUM_STRIKES around the closest strike
         half_n = NUM_STRIKES // 2
         start_index = max(0, closest_strike_index - half_n)
         end_index = start_index + NUM_STRIKES # Grab exactly N strikes if possible
         end_index = min(end_index, len(all_strikes)) # Adjust if near the end

         # Adjust start index if near the end of the list to ensure N strikes if possible
         if end_index - start_index < NUM_STRIKES:
              start_index = max(0, end_index - NUM_STRIKES)

         relevant_strikes = all_strikes[start_index:end_index]
         logging.info(f"🎯 Identified {len(relevant_strikes)} relevant strikes around ${current_price:.2f}.")
         return relevant_strikes

    def _request_multiple_option_prices(self, symbol, expiration, strikes):
        """Requests market data for multiple options (calls and puts) concurrently."""
        if not self._option_chain_params or 'tradingClass' not in self._option_chain_params:
            logging.error("Option chain parameters (tradingClass) not loaded.")
            return False

        logging.info(f"📊 Requesting prices for {len(strikes)} strikes (Calls & Puts) for {expiration}...")
        self._options_price_data = {} # Clear previous price data
        self._option_price_events = {} # Clear previous events
        request_ids = []

        # Set market data type (important for options)
        market_data_type = 3 if USE_DELAYED_DATA else 1
        self.reqMarketDataType(market_data_type)
        time.sleep(0.1)

        # Create contracts and request data for Calls and Puts
        for strike in strikes:
            for right in ["C", "P"]:
                req_id = self._get_next_req_id()
                request_ids.append(req_id)

                contract = Contract()
                contract.symbol = symbol
                contract.secType = "OPT"
                contract.exchange = "SMART"
                contract.currency = "USD"
                contract.strike = strike
                contract.right = right
                contract.lastTradeDateOrContractMonth = expiration
                contract.multiplier = self._option_chain_params.get('multiplier', "100")
                contract.tradingClass = self._option_chain_params['tradingClass']

                # Prepare storage and event for this request
                price_event = threading.Event()
                self._options_price_data[req_id] = {
                    'contract': contract,
                    'bid': None, 'ask': None, 'last': None,
                    'received_event': price_event
                }
                self._option_price_events[req_id] = price_event

                # Request data (use generic ticks for compatibility)
                generic_ticks = "" # Empty string usually sufficient for Bid/Ask/Last
                snapshot = False
                self.reqMktData(req_id, contract, generic_ticks, snapshot, False, [])
                logging.debug(f"Requested price for ReqId {req_id}: {symbol} {expiration} {strike} {right}")
                # Small delay between requests might help IB backend
                time.sleep(0.05)

        # Wait for all price events to be set
        logging.info(f"⏳ Waiting for {len(request_ids)} option price updates (Timeout: {OPTION_PRICE_TIMEOUT}s)...")
        start_wait_time = time.time()
        all_received = True
        for req_id in request_ids:
            event = self._option_price_events.get(req_id)
            if event:
                # Calculate remaining timeout
                elapsed_time = time.time() - start_wait_time
                remaining_timeout = max(0, OPTION_PRICE_TIMEOUT - elapsed_time)
                if not event.wait(timeout=remaining_timeout):
                    logging.warning(f"⌛ Timeout waiting for price data for ReqId {req_id}")
                    all_received = False
                    # Don't break, try to wait for others
            else:
                 logging.warning(f"Event not found for ReqId {req_id}, skipping wait.")
                 all_received = False

        # Cancel all market data requests after waiting
        logging.info("⏹️ Cancelling option price market data requests...")
        for req_id in request_ids:
             try:
                 self.cancelMktData(req_id)
             except Exception as cancel_e:
                 logging.error(f"Error cancelling mkt data for ReqId {req_id}: {cancel_e}")
        time.sleep(0.5) # Allow cancellations to process

        if all_received:
            logging.info("✅ All option price updates received.")
        else:
            logging.warning("⚠️ Some option price updates may have timed out.")

        return True

    def get_and_display_leap_options(self, symbol):
        """Main orchestration method to fetch and display LEAP options."""
        if not self.connected_event.wait(timeout=MAX_WAIT_TIME):
            logging.error("Connection not established within timeout.")
            return None

        # 1. Get Contract Details
        if not self._request_contract_details(symbol) or \
           not self.contract_details_event.wait(timeout=MAX_WAIT_TIME) or \
           not self._stock_contract_details:
            logging.error(f"Failed to get contract details for {symbol}.")
            return None

        # 2. Get Current Stock Price
        if not self._request_stock_price(symbol) or \
           not self.stock_price_event.wait(timeout=PRICE_WAIT_TIMEOUT) or \
           self._current_stock_price is None:
            logging.warning(f"Failed to get current price for {symbol}. Proceeding without it for strike selection might be inaccurate.")
            # Attempt to continue, but strike selection will be less accurate
            # Use a default price or fetch historical close? For now, proceed cautiously.

        current_price = self._current_stock_price if self._current_stock_price is not None else 0 # Handle None case
        if current_price == 0:
             logging.error("Stock price is 0, cannot reliably select strikes.")
             return None

        # Cancel stock price subscription if we got a price
        self.cancelMktData(self.REQ_ID_STOCK_PRICE)

        # 3. Get Option Chain Parameters
        if not self._request_option_chain_params(symbol) or \
           not self.option_chain_event.wait(timeout=MAX_WAIT_TIME) or \
           not self._option_chain_params:
            logging.error(f"Failed to get option chain parameters for {symbol}.")
            return None

        # 4. Find Target Expiration
        target_expiration = self._find_target_leap_expiration()
        if not target_expiration:
            logging.error(f"Could not determine a suitable LEAP expiration for {symbol}.")
            return None

        # 5. Find Relevant Strikes
        relevant_strikes = self._get_relevant_strikes(current_price)
        if not relevant_strikes:
            logging.error(f"Could not determine relevant strikes for {symbol} around price {current_price:.2f}.")
            return None

        # 6. Request Prices for Relevant Options Concurrently
        if not self._request_multiple_option_prices(symbol, target_expiration, relevant_strikes):
             logging.error("Failed to initiate or complete option price requests.")
             # Continue to process whatever data was received

        # 7. Assemble DataFrame
        logging.info(" assembling results into DataFrame...")
        data = []
        exp_date_str = datetime.strptime(target_expiration, '%Y%m%d').strftime('%Y-%m-%d')
        months_out = (datetime.strptime(target_expiration, '%Y%m%d') - datetime.now()).days / 30.0

        # Iterate through the stored price data instead of request IDs
        for req_id, price_info in self._options_price_data.items():
            contract = price_info['contract']
            if contract.lastTradeDateOrContractMonth == target_expiration and contract.strike in relevant_strikes:
                strike = contract.strike
                distance = round((strike - current_price) / current_price * 100, 1) if current_price else 0
                data.append({
                    'Expiration': exp_date_str,
                    'Strike': strike,
                    'Months Out': round(months_out, 1),
                    'Type': 'CALL' if contract.right == 'C' else 'PUT',
                    'Distance': distance,
                    'Bid': price_info['bid'] if price_info['bid'] is not None else 0.0,
                    'Ask': price_info['ask'] if price_info['ask'] is not None else 0.0,
                    'Last': price_info['last'] if price_info['last'] is not None else 0.0
                    # Could add 'Received': price_info['received_event'].is_set()
                })

        if not data:
            logging.warning("No option price data was successfully retrieved or processed.")
            return None

        df = pd.DataFrame(data)
        df = df.sort_values(['Type', 'Strike']).reset_index(drop=True)

        # Format for display
        call_df = df[df['Type'] == 'CALL'].copy()
        put_df = df[df['Type'] == 'PUT'].copy()

        for temp_df in [call_df, put_df]:
            temp_df['Distance'] = temp_df['Distance'].apply(lambda x: f"{x:+.1f}%")
            temp_df['Bid'] = temp_df['Bid'].apply(lambda x: f"${x:.2f}")
            temp_df['Ask'] = temp_df['Ask'].apply(lambda x: f"${x:.2f}")
            temp_df['Last'] = temp_df['Last'].apply(lambda x: f"${x:.2f}")

        logging.info(f"📊 LEAP Options data for {symbol} ({target_expiration}) assembled.")
        return {'CALLS': call_df, 'PUTS': put_df, 'Underlying Price': current_price}

def run_connection_loop(app_instance):
    """Runs the TWS/Gateway client's message loop."""
    try:
        app_instance.run()
    except Exception as e:
         logging.error(f"❌ Exception in TWS/Gateway message loop: {e}", exc_info=True)

def main():
    """Main function to connect, fetch LEAP data, and display it."""
    app = LeapOptionsTrader()
    port = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
    trading_mode = "Paper" if IS_PAPER_TRADING else "Live"
    logging.info(f"🚀 Moon Dev connecting to {trading_mode} trading on port {port}")

    app.connect("127.0.0.1", port, clientId=4) # Use a different clientId

    con_thread = threading.Thread(target=run_connection_loop, args=(app,), daemon=True)
    con_thread.start()

    results = None
    try:
        results = app.get_and_display_leap_options(SYMBOL)

        if isinstance(results, dict):
            current_price = results.get('Underlying Price', 0)
            logging.info(f"\n--- LEAP Options for {SYMBOL} (Target Expiration) ---")
            logging.info(f"--- Underlying Price: ${current_price:.2f} ---")

            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)

            print("\n🟢 CALL Options:")
            print("=" * 80)
            if not results['CALLS'].empty:
                print(results['CALLS'].to_string(index=False))
            else:
                print("No CALL data retrieved.")

            print("\n🔴 PUT Options:")
            print("=" * 80)
            if not results['PUTS'].empty:
                print(results['PUTS'].to_string(index=False))
            else:
                print("No PUT data retrieved.")
        else:
            logging.error("❌ Failed to retrieve LEAP options data.")

    except Exception as e:
        logging.error(f"❌ An unexpected error occurred in main: {e}", exc_info=True)
        logging.error("💡 Tips: Check TWS connection, market data, symbol validity.")
    finally:
        logging.info("⏳ Disconnecting...")
        app.disconnect()
        # Wait slightly for disconnect to complete
        if con_thread.is_alive():
             time.sleep(1)
        logging.info("👋 Moon Dev has disconnected safely.")

if __name__ == "__main__":
    main()
