"""
ATS-Native Interactive Brokers Service
"""

import asyncio
import pytz
import logging
import threading
import pandas as pd
from typing import Optional, List, Dict, Any
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from core.config import get_settings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class IBClient(EWrapper, EClient):
    """
    The core client for interacting with the IB TWS/Gateway.
    This class handles the low-level communication, message processing, and data synchronization.
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.nextValidOrderId = None
        self._next_req_id = 1
        self.is_connected = False
        
        # Events for synchronization
        self.connection_event = threading.Event()
        self.order_status_events: Dict[int, threading.Event] = {}
        self.position_event = threading.Event()
        self.contract_details_events: Dict[int, threading.Event] = {}
        self.open_order_event = threading.Event()
        self.account_summary_event = threading.Event()
        self.option_chain_event = threading.Event()
        self._option_price_events: Dict[int, threading.Event] = {}
        self.historical_data_event = threading.Event()

        # Data storage
        self._pos_data: List[Dict] = []
        self._open_orders_data: List[Dict] = []
        self.account_summary: Dict[str, Any] = {}
        self._stock_contract_details: Optional[Any] = None
        self._current_stock_price: Optional[float] = None
        self._option_chain_params: Dict[str, Any] = {}
        self._options_price_data: Dict[int, Dict] = {}
        self._historical_data_map: Dict[int, List[Dict]] = {}
        self._tick_data_map: Dict[int, Dict] = {}
        self.order_status_cache: Dict[int, Dict] = {}
        self.contract_details_cache: Dict[int, List[Any]] = {}

    def _get_next_req_id(self):
        req_id = self._next_req_id
        self._next_req_id += 1
        return req_id

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2108, 2158, 399, 202, 10167, 322, 200]:
            logger.error(f"❌ IB Error {errorCode}: {errorString} (ReqId: {reqId})")
        if reqId in self.contract_details_events:
            self.contract_details_events[reqId].set()
        if reqId in self._option_price_events:
            self._option_price_events[reqId].set()
        if self.option_chain_event.is_set() is False and errorCode != 0:
             self.option_chain_event.set()

    def connectionClosed(self):
        self.is_connected = False
        logger.warning("🔌 IB connection closed.")
        self.connection_event.clear()

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId
        self.is_connected = True
        logger.info(f"🔑 IB Next Valid Order ID: {orderId}")
        self.connection_event.set()

    def position(self, account, contract, position, avgCost):
        pos_dict = {
            "account": account,
            "symbol": contract.symbol,
            "secType": contract.secType,
            "position": position,
            "avgCost": avgCost,
            "localSymbol": getattr(contract, 'localSymbol', contract.symbol),
            "conId": contract.conId
        }
        self._pos_data.append(pos_dict)

    def positionEnd(self):
        logger.info("🎯 IB position data stream ended.")
        self.position_event.set()

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        status_info = {'status': status, 'filled': filled, 'remaining': remaining, 'avgFillPrice': avgFillPrice}
        self.order_status_cache[orderId] = status_info
        logger.info(f"🔄 Order {orderId} status: {status}, Filled: {filled}, Remaining: {remaining}")
        if status in ["Filled", "Cancelled", "Inactive", "ApiCancelled"]:
            if orderId in self.order_status_events:
                self.order_status_events[orderId].set()

    def contractDetails(self, reqId, contractDetails):
        if 'stock' in str(reqId): # Custom identifier for stock details
            self._stock_contract_details = contractDetails
        elif reqId in self.contract_details_cache:
            if reqId not in self.contract_details_cache:
                self.contract_details_cache[reqId] = []
            self.contract_details_cache[reqId].append(contractDetails)

    def contractDetailsEnd(self, reqId):
        if 'stock' in str(reqId):
            self.contract_details_events[reqId].set()
        elif reqId in self.contract_details_events:
            self.contract_details_events[reqId].set()

    def tickPrice(self, reqId, tickType, price, attrib):
        # Tick types: 1=bid, 2=ask, 4=last. Delayed: 66=bid, 67=ask, 68=last.
        if 'stock_price' in str(reqId):
            if tickType in [4, 68]: # Prioritize Last price
                self._current_stock_price = price
                self.contract_details_events[reqId].set() # Using the same event dict for simplicity
        elif reqId in self._options_price_data:
            option_info = self._options_price_data[reqId]
            if tickType in [1, 66]: option_info['bid'] = price
            elif tickType in [2, 67]: option_info['ask'] = price
            elif tickType in [4, 68]: option_info['last'] = price
            
            # Signal as soon as any price component is received
            if not self._option_price_events[reqId].is_set():
                self._option_price_events[reqId].set()

    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass, multiplier, expirations, strikes):
        if exchange == "SMART":
            if not self._option_chain_params:
                self._option_chain_params = {
                    'tradingClass': tradingClass, 'multiplier': multiplier,
                    'expirations': set(expirations), 'strikes': set(strikes)
                }
            else:
                self._option_chain_params['expirations'].update(expirations)
                self._option_chain_params['strikes'].update(strikes)

    def securityDefinitionOptionParameterEnd(self, reqId):
        self.option_chain_event.set()

    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState: Any):
        """Callback for open order details."""
        super().openOrder(orderId, contract, order, orderState)
        order_details = {
            "permId": order.permId,
            "orderId": orderId,
            "symbol": contract.symbol,
            "secType": contract.secType,
            "action": order.action,
            "orderType": order.orderType,
            "totalQty": order.totalQuantity,
            "lmtPrice": order.lmtPrice,
            "auxPrice": order.auxPrice,
            "status": orderState.status
        }
        self._open_orders_data.append(order_details)
        contract.exchange = "SMART"
        return contract

    def openOrderEnd(self):
        """Callback indicating all open orders have been sent."""
        super().openOrderEnd()
        logger.info("✅ IB open order data stream ended.")
        self.open_order_event.set()

    def updatePortfolio(self, contract: Contract, position: float,
                        marketPrice: float, marketValue: float,
                        averageCost: float, unrealizedPNL: float,
                        realizedPNL: float, accountName: str):
        """Callback for portfolio updates from subscription."""
        if "positions" not in self.account_summary:
            self.account_summary['positions'] = []
        
        # Remove existing entry for this symbol to avoid duplicates
        self.account_summary['positions'] = [p for p in self.account_summary['positions'] if p['symbol'] != contract.symbol]
        
        if position != 0:
            pos_dict = {
                "account": accountName,
                "symbol": contract.symbol,
                "secType": contract.secType,
                "position": position,
                "marketPrice": marketPrice,
                "marketValue": marketValue,
                "averageCost": averageCost,
                "unrealizedPNL": unrealizedPNL,
                "realizedPNL": realizedPNL,
            }
            self.account_summary['positions'].append(pos_dict)
    
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        """Callback for account value updates from subscription."""
        if currency == "USD":
            if "summary" not in self.account_summary:
                self.account_summary['summary'] = {}
            self.account_summary['summary'][key] = val
            self.account_summary['summary']['accountName'] = accountName

    def accountDownloadEnd(self, accountName: str):
        """Callback indicating account download is complete."""
        super().accountDownloadEnd(accountName)
        logger.info(f"✅ IB account download ended for: {accountName}")
        self.account_summary_event.set()

    def historicalData(self, reqId: int, bar):
        """Callback for historical data bars."""
        if reqId not in self._historical_data_map:
            self._historical_data_map[reqId] = []
        self._historical_data_map[reqId].append({
            'date': bar.date, 'open': bar.open, 'high': bar.high,
            'low': bar.low, 'close': bar.close, 'volume': bar.volume
        })

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Callback for end of historical data stream."""
        logger.info(f"✅ Historical data ended for ReqId: {reqId}")
        self.historical_data_event.set()

class InteractiveBrokersService:
    """
    A service to manage trading operations with Interactive Brokers.
    This service abstracts away the threading and event handling required by the ibapi.
    """
    
    # Static data for futures contracts based on futures_mkt_order.py
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
        self.settings = get_settings()
        self.client = IBClient()
        self.loop_thread: Optional[threading.Thread] = None

    async def start(self):
        """Connects to IB TWS/Gateway and starts the message loop."""
        if self.client.is_connected:
            return
        port = self.settings.IB_GATEWAY_PAPER_PORT if self.settings.IB_IS_PAPER_TRADING else self.settings.IB_GATEWAY_LIVE_PORT
        trading_mode = "Paper" if self.settings.IB_IS_PAPER_TRADING else "Live"
        logger.info(f"🚀 Connecting to IB {trading_mode} on {self.settings.IB_GATEWAY_HOST}:{port}")
        
        self.client.connect(self.settings.IB_GATEWAY_HOST, port, clientId=self.settings.IB_CLIENT_ID)
        self.loop_thread = threading.Thread(target=self.client.run, daemon=True)
        self.loop_thread.start()

        # Wait for connection in an async-friendly way
        await asyncio.to_thread(self.client.connection_event.wait, timeout=10)
        if not self.client.is_connected:
            raise ConnectionError("Failed to connect to Interactive Brokers TWS/Gateway.")
        logger.info("✅ IB Service connected successfully.")

    async def stop(self):
        """Disconnects from IB TWS/Gateway."""
        if self.client.is_connected:
            self.client.disconnect()
            logger.info("👋 IB Service disconnected.")
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=2)

    def _get_next_order_id(self):
        if self.client.nextValidOrderId is None:
            raise ConnectionError("Cannot get next order ID, not connected or not received from TWS.")
        order_id = self.client.nextValidOrderId
        self.client.nextValidOrderId += 3 # Increment by 3 for bracket orders, good enough for single orders too
        return order_id

    # --- Contract Methods ---
    def _create_stock_contract(self, symbol: str) -> Contract:
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        return contract

    async def _get_active_futures_contract(self, symbol: str) -> Contract:
        """
        Fetches contract details and selects the most suitable active future,
        using logic from futures_mkt_order.py.
        """
        base_contract = Contract()
        base_contract.symbol = symbol.upper()
        base_contract.secType = "FUT"
        base_contract.currency = "USD"
        base_contract.exchange = self.FUTURES_EXCHANGES.get(symbol.upper(), "CME")

        if symbol.upper() in self.FUTURES_TRADING_CLASSES:
            base_contract.tradingClass = self.FUTURES_TRADING_CLASSES[symbol.upper()]
        if symbol.upper() in self.FUTURES_MULTIPLIERS:
            base_contract.multiplier = self.FUTURES_MULTIPLIERS[symbol.upper()]

        logger.info(f"🔍 Locating active contract for {symbol} with base: {base_contract.symbol}, {base_contract.exchange}, {getattr(base_contract, 'tradingClass', 'N/A')}")
        
        req_id = self.client._get_next_req_id()
        self.client.contract_details_events[req_id] = threading.Event()
        self.client.contract_details_cache[req_id] = [] # Clear previous results
        self.client.reqContractDetails(req_id, base_contract)
        
        try:
            await asyncio.to_thread(self.client.contract_details_events[req_id].wait, timeout=10)
        except Exception:
            raise TimeoutError(f"Timeout waiting for future contract details for {symbol}")
        
        details_list = self.client.contract_details_cache.get(req_id, [])
        if not details_list:
            raise ValueError(f"Could not find contract details for FUT {symbol}. Check symbol and exchange.")

        # Sort by expiration date and select the most appropriate contract
        sorted_contracts = sorted(details_list, key=lambda d: d.contract.lastTradeDateOrContractMonth)
        
        active_contract = None
        current_date = datetime.now()
        for detail in sorted_contracts:
            expiry_str = detail.contract.lastTradeDateOrContractMonth
            if expiry_str and len(expiry_str) == 8:
                try:
                    expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
                    if (expiry_date - current_date).days > 5:
                        active_contract = detail.contract
                        break # Found the first suitable contract
                except ValueError:
                    continue # Skip if date is malformed
        
        # Fallback to furthest dated contract if no suitable one is found
        if not active_contract and sorted_contracts:
            active_contract = sorted_contracts[-1].contract
            logger.warning(f"⚠️ No contract expiring >5 days found. Using furthest dated: {active_contract.localSymbol}")

        if not active_contract:
            raise ValueError(f"Failed to identify a suitable active contract for {symbol}")

        logger.info(f"🎯 Selected active futures contract: {active_contract.localSymbol} (Expires: {active_contract.lastTradeDateOrContractMonth})")
        return active_contract

    # --- Order Placement Methods ---
    async def place_market_order(self, symbol: str, direction: str, quantity: float, sec_type: str = "STK") -> Dict:
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        order_id = self._get_next_order_id()
        
        if sec_type.upper() == "STK":
            contract = self._create_stock_contract(symbol)
        elif sec_type.upper() == "FUT":
            contract = await self._get_active_futures_contract(symbol)
        else:
            raise ValueError(f"Unsupported security type: {sec_type}")

        order = Order()
        order.action = direction.upper()
        order.orderType = "MKT"
        order.totalQuantity = quantity
        
        self.client.order_status_events[order_id] = threading.Event()
        self.client.placeOrder(order_id, contract, order)
        
        await asyncio.to_thread(self.client.order_status_events[order_id].wait, timeout=10)
        return self.client.order_status_cache.get(order_id, {"status": "Timeout"})

    async def place_limit_order(self, symbol: str, direction: str, quantity: float, limit_price: float) -> Dict:
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        order_id = self._get_next_order_id()
        contract = self._create_stock_contract(symbol)

        order = Order()
        order.action = direction.upper()
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = limit_price
        
        self.client.order_status_events[order_id] = threading.Event()
        self.client.placeOrder(order_id, contract, order)
        
        await asyncio.to_thread(self.client.order_status_events[order_id].wait, timeout=30)
        return self.client.order_status_cache.get(order_id, {"status": "Timeout"})

    async def place_stop_order(self, symbol: str, direction: str, quantity: float, stop_price: float) -> Dict:
        """Places a simple stop order."""
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        order_id = self._get_next_order_id()
        contract = self._create_stock_contract(symbol)

        order = Order()
        order.action = direction.upper()
        order.orderType = "STP"
        order.totalQuantity = quantity
        order.auxPrice = stop_price
        
        self.client.order_status_events[order_id] = threading.Event()
        self.client.placeOrder(order_id, contract, order)
        
        await asyncio.to_thread(self.client.order_status_events[order_id].wait, timeout=20)
        return self.client.order_status_cache.get(order_id, {"status": "Timeout", "orderId": order_id})

    async def place_bracket_order(self, symbol: str, direction: str, quantity: float, entry_price: float, take_profit_price: float, stop_loss_price: float) -> List[Dict]:
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        
        parent_id = self._get_next_order_id()
        
        contract = self._create_stock_contract(symbol)
        
        # Create orders
        parent_order = Order()
        parent_order.orderId = parent_id
        parent_order.action = direction.upper()
        parent_order.orderType = "LMT"
        parent_order.lmtPrice = entry_price
        parent_order.totalQuantity = quantity
        parent_order.transmit = False

        tp_order = Order()
        tp_order.orderId = parent_id + 1
        tp_order.action = "SELL" if direction.upper() == "BUY" else "BUY"
        tp_order.orderType = "LMT"
        tp_order.lmtPrice = take_profit_price
        tp_order.totalQuantity = quantity
        tp_order.parentId = parent_id
        tp_order.transmit = False

        sl_order = Order()
        sl_order.orderId = parent_id + 2
        sl_order.action = "SELL" if direction.upper() == "BUY" else "BUY"
        sl_order.orderType = "STP"
        sl_order.auxPrice = stop_loss_price
        sl_order.totalQuantity = quantity
        sl_order.parentId = parent_id
        sl_order.transmit = True

        orders_to_place = [parent_order, tp_order, sl_order]
        for o in orders_to_place:
            self.client.order_status_events[o.orderId] = threading.Event()
            self.client.placeOrder(o.orderId, contract, o)

        # Wait for all three orders to get a status update
        tasks = [asyncio.to_thread(self.client.order_status_events[o.orderId].wait, timeout=10) for o in orders_to_place]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return [self.client.order_status_cache.get(o.orderId, {"status": "Timeout", "orderId": o.orderId}) for o in orders_to_place]

    # --- Account Management Methods ---
    async def cancel_all_orders(self):
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        self.client.reqGlobalCancel()
        logger.info("Sent request to cancel all open orders.")
        await asyncio.sleep(1) # Allow time for cancellations to propagate

    async def get_positions(self) -> pd.DataFrame:
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        self.client._pos_data.clear()
        self.client.position_event.clear()
        self.client.reqPositions()
        await asyncio.to_thread(self.client.position_event.wait, timeout=5)
        return pd.DataFrame(self.client._pos_data)

    async def close_all_positions(self):
        await self.cancel_all_orders()
        positions_df = await self.get_positions()
        
        if positions_df.empty or positions_df['position'].sum() == 0:
            logger.info("No open positions to close.")
            return []

        logger.info(f"Found {len(positions_df[positions_df['position'] != 0])} positions to close.")
        close_tasks = []
        for _, row in positions_df.iterrows():
            pos = row['position']
            if pos == 0: continue
            
            direction = "SELL" if pos > 0 else "BUY"
            quantity = abs(pos)
            symbol = row['symbol']
            sec_type = row['secType']
            
            logger.info(f"Closing position: {direction} {quantity} {symbol} ({sec_type})")
            close_tasks.append(self.place_market_order(symbol, direction, quantity, sec_type))

        results = await asyncio.gather(*close_tasks, return_exceptions=True)
        return results

    async def get_open_orders(self) -> pd.DataFrame:
        """Retrieves a list of all pending open orders."""
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        self.client._open_orders_data.clear()
        self.client.open_order_event.clear()
        self.client.reqOpenOrders()
        await asyncio.to_thread(self.client.open_order_event.wait, timeout=5)
        return pd.DataFrame(self.client._open_orders_data)
        
    async def get_account_summary(self, account_code: str = "") -> Dict:
        """Subscribes to and retrieves a snapshot of the account summary and positions."""
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        self.client.account_summary.clear()
        self.client.account_summary_event.clear()
        
        logger.info(f"Subscribing to account updates for account: {account_code or 'All'}")
        self.client.reqAccountUpdates(True, account_code)
        
        try:
            # Wait for the account download to end, which signifies data is populated.
            await asyncio.to_thread(self.client.account_summary_event.wait, timeout=10)
        except Exception:
            logger.warning("Timeout waiting for full account summary. Data may be incomplete.")
        finally:
            # Always unsubscribe to avoid continuous data flow.
            logger.info("Unsubscribing from account updates.")
            self.client.reqAccountUpdates(False, account_code)
            
        return self.client.account_summary.copy()

    async def _request_stock_details(self, symbol: str) -> bool:
        """Requests fundamental contract details for a stock."""
        req_id = f"stock_details_{symbol}"
        self.client.contract_details_events[req_id] = threading.Event()
        self.client._stock_contract_details = None
        
        contract = self._create_stock_contract(symbol)
        self.client.reqContractDetails(req_id, contract)
        
        await asyncio.to_thread(self.client.contract_details_events[req_id].wait, timeout=10)
        return self.client._stock_contract_details is not None

    async def _request_stock_price(self, contract: Contract) -> Optional[float]:
        """Requests the current market price for a contract."""
        req_id = f"stock_price_{contract.symbol}"
        self.client.contract_details_events[req_id] = threading.Event()
        self.client._current_stock_price = None
        
        # Use delayed data for now as it requires no subscription
        self.client.reqMarketDataType(3)
        await asyncio.sleep(0.1)
        self.client.reqMktData(req_id, contract, "", True, False, [])
        
        try:
            await asyncio.to_thread(self.client.contract_details_events[req_id].wait, timeout=5)
        except Exception:
            logger.warning(f"Timeout getting stock price for {contract.symbol}")
        finally:
            self.client.cancelMktData(req_id)
        
        return self.client._current_stock_price

    async def _request_option_chain_params(self, symbol: str, conId: int) -> bool:
        """Requests the option chain parameters (expirations, strikes)."""
        self.client.option_chain_event.clear()
        self.client._option_chain_params = {}
        self.client.reqSecDefOptParams(self.client._get_next_req_id(), symbol, "", "STK", conId)
        await asyncio.to_thread(self.client.option_chain_event.wait, timeout=10)
        return bool(self.client._option_chain_params)

    def _find_target_leap_expiration(self, months_out: int) -> Optional[str]:
        """Finds the best LEAP expiration date."""
        params = self.client._option_chain_params
        if not params or 'expirations' not in params: return None

        target_min_date = datetime.now() + timedelta(days=months_out * 30 - 15)
        ideal_target_date = datetime.now() + timedelta(days=months_out * 30)

        valid_expirations = []
        for exp_str in params['expirations']:
            try:
                exp_date = datetime.strptime(exp_str, '%Y%m%d')
                if exp_date >= target_min_date:
                    valid_expirations.append((exp_date, exp_str))
            except ValueError:
                continue
        
        if not valid_expirations: return None
        
        valid_expirations.sort(key=lambda x: abs(x[0] - ideal_target_date))
        return valid_expirations[0][1]

    def _get_relevant_strikes(self, current_price: float, num_strikes: int) -> List[float]:
        """Filters strikes around the current price."""
        params = self.client._option_chain_params
        if not params or 'strikes' not in params or not current_price: return []

        all_strikes = sorted(list(params['strikes']))
        if not all_strikes: return []

        closest_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - current_price))
        
        half_n = num_strikes // 2
        start_idx = max(0, closest_idx - half_n)
        end_idx = min(len(all_strikes), start_idx + num_strikes)
        
        return all_strikes[start_idx:end_idx]

    async def _request_multiple_option_prices(self, symbol: str, expiration: str, strikes: List[float]):
        """Requests market data for multiple options concurrently."""
        params = self.client._option_chain_params
        if not params: return

        self.client._options_price_data.clear()
        self.client._option_price_events.clear()
        
        self.client.reqMarketDataType(3) # Delayed data
        await asyncio.sleep(0.1)

        tasks = []
        for strike in strikes:
            for right in ["C", "P"]:
                req_id = self.client._get_next_req_id()
                
                contract = self._create_stock_contract(symbol)
                contract.secType = "OPT"
                contract.right = right
                contract.strike = strike
                contract.lastTradeDateOrContractMonth = expiration
                contract.tradingClass = params.get('tradingClass', symbol)
                contract.multiplier = params.get('multiplier', "100")

                event = threading.Event()
                self.client._options_price_data[req_id] = {'contract': contract, 'bid': None, 'ask': None, 'last': None}
                self.client._option_price_events[req_id] = event

                self.client.reqMktData(req_id, contract, "", True, False, [])
                tasks.append(asyncio.to_thread(event.wait, timeout=15))
        
        await asyncio.gather(*tasks, return_exceptions=True)

        for req_id in self.client._option_price_events.keys():
            self.client.cancelMktData(req_id)

    async def get_leap_options(self, symbol: str, months_out: int, num_strikes: int) -> Dict[str, Any]:
        """Main orchestration method to fetch and process LEAP options."""
        if not self.client.is_connected: raise ConnectionError("IB not connected.")

        # 1. Get Contract Details for the underlying stock
        if not await self._request_stock_details(symbol):
            raise ValueError(f"Failed to get contract details for {symbol}")
        
        stock_contract = self.client._stock_contract_details.contract
        
        # 2. Get Current Stock Price
        current_price = await self._request_stock_price(stock_contract)
        if not current_price:
            raise ValueError(f"Failed to get current price for {symbol}")

        # 3. Get Option Chain Parameters
        if not await self._request_option_chain_params(symbol, stock_contract.conId):
            raise ValueError(f"Failed to get option chain parameters for {symbol}")

        # 4. Find Target Expiration & Relevant Strikes
        target_expiration = self._find_target_leap_expiration(months_out)
        if not target_expiration:
            raise ValueError(f"No suitable LEAP expiration found for {symbol} {months_out} months out.")
            
        relevant_strikes = self._get_relevant_strikes(current_price, num_strikes)
        if not relevant_strikes:
            raise ValueError(f"Could not determine relevant strikes for {symbol}")

        # 5. Request Prices for Options
        await self._request_multiple_option_prices(symbol, target_expiration, relevant_strikes)

        # 6. Assemble DataFrame
        data = []
        for req_id, price_info in self.client._options_price_data.items():
            c = price_info['contract']
            data.append({
                'strike': c.strike, 'type': c.right,
                'bid': price_info.get('bid'), 'ask': price_info.get('ask'), 'last': price_info.get('last')
            })
        
        return {
            "underlyingPrice": current_price,
            "expiration": target_expiration,
            "options": data
        }

    async def get_historical_data(self, symbol: str, duration: str, bar_size: str) -> pd.DataFrame:
        """Fetches OHLCV data for a given symbol."""
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        
        req_id = self.client._get_next_req_id()
        self.client.historical_data_event.clear()
        self.client._historical_data_map[req_id] = []
        
        contract = self._create_stock_contract(symbol)
        
        # Calculate a reasonable timeout
        timeout = 20 + (int(''.join(filter(str.isdigit, duration))) * 0.1)
        
        self.client.reqHistoricalData(
            reqId=req_id, contract=contract, endDateTime="",
            durationStr=duration, barSizeSetting=bar_size,
            whatToShow="TRADES", useRTH=1, formatDate=1, keepUpToDate=False, chartOptions=[]
        )
        
        await asyncio.to_thread(self.client.historical_data_event.wait, timeout=timeout)
        
        data = self.client._historical_data_map.pop(req_id, [])
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d %H:%M:%S')
        df = df.sort_values('date').reset_index(drop=True)
        return df

    async def get_bid_ask(self, symbol: str) -> Optional[Dict[str, float]]:
        """Gets the current bid and ask prices for a symbol."""
        if not self.client.is_connected: raise ConnectionError("IB not connected.")
        
        req_id = self.client._get_next_req_id()
        event = threading.Event()
        self.client._option_price_events[req_id] = event # Re-using this event dict
        self.client._tick_data_map[req_id] = {}
        
        contract = self._create_stock_contract(symbol)
        self.client.reqMktData(req_id, contract, "", True, False, [])
        
        result = None
        try:
            await asyncio.to_thread(event.wait, timeout=10)
            tick_data = self.client._tick_data_map.get(req_id, {})
            if tick_data.get('bid') is not None and tick_data.get('ask') is not None:
                result = {'bid': tick_data['bid'], 'ask': tick_data['ask']}
        finally:
            self.client.cancelMktData(req_id)
            self.client._option_price_events.pop(req_id, None)
            self.client._tick_data_map.pop(req_id, None)
            
        return result

    async def cancel_orders_by_symbol(self, symbol: str) -> int:
        """Cancels all open orders for a specific symbol."""
        open_orders_df = await self.get_open_orders()
        if open_orders_df.empty:
            return 0
            
        orders_to_cancel = open_orders_df[open_orders_df['symbol'].str.upper() == symbol.upper()]
        if orders_to_cancel.empty:
            logger.info(f"No open orders found for symbol {symbol}.")
            return 0
            
        cancelled_count = 0
        for _, row in orders_to_cancel.iterrows():
            order_id = row['orderId']
            logger.info(f"Cancelling order {order_id} for symbol {symbol}...")
            self.client.cancelOrder(order_id, "")
            await asyncio.sleep(0.1) # Small delay
            cancelled_count += 1
        
        return cancelled_count

    def is_market_open(self, tz_name: str = 'US/Eastern') -> Dict[str, Any]:
        """Checks if the US market is currently open."""
        try:
            tz = pytz.timezone(tz_name)
        except pytz.UnknownTimeZoneError:
            tz = pytz.timezone('US/Eastern')

        now = datetime.now(tz)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_weekday = now.weekday() < 5
        is_open = is_weekday and market_open.time() <= now.time() < market_close.time()
        
        status = "CLOSED"
        if is_open: status = "OPEN"
        elif is_weekday and now.time() < market_open.time(): status = "PRE-MARKET"
        elif is_weekday and now.time() >= market_close.time(): status = "POST-MARKET"
        elif not is_weekday: status = "WEEKEND"
        
        return {
            "is_open": is_open,
            "status": status,
            "time_now": now.strftime('%Y-%m-%d %H:%M:%S %Z'),
            "market_open": market_open.strftime('%H:%M:%S'),
            "market_close": market_close.strftime('%H:%M:%S'),
            "timezone": str(tz)
        }

# Singleton instance
_ib_service_instance: Optional[InteractiveBrokersService] = None

def get_ib_service():
    global _ib_service_instance
    if _ib_service_instance is None:
        _ib_service_instance = InteractiveBrokersService()
    return _ib_service_instance
