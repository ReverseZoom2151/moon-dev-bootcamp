from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connection Configuration
PAPER_PORT: int = 7497         # Paper trading port
LIVE_PORT: int = 7496          # Live trading port
IS_PAPER_TRADING: bool = True   # Set to False for live trading
WAIT_TIME: int = 15 # Increased wait time for account updates
CLIENT_ID_POSITIONS: int = 3 # Unique client ID for this script

class PositionDetailsApp(EWrapper, EClient):
    """
    Application to fetch and display account positions and summary from Interactive Brokers.
    """
    def __init__(self):
        EClient.__init__(self, self)
        self.pos_df: pd.DataFrame = pd.DataFrame({
            'Symbol': pd.Series(dtype='str'),
            'SecType': pd.Series(dtype='str'),
            'Position': pd.Series(dtype='float'),
            'Market Value': pd.Series(dtype='float'),
            'Unrealized PnL': pd.Series(dtype='float'),
            'Avg Cost': pd.Series(dtype='float')
        })
        self.account_summary_event: threading.Event = threading.Event()
        self.connection_event: threading.Event = threading.Event()
        self.next_valid_order_id: int = -1 # Though not placing orders, good practice to track
        self.account_name: str = ""
        self.net_liquidation: float = 0.0
        self.unrealized_pnl_total: float = 0.0

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        ignore_codes = {2104, 2106, 2108, 2158, 2109, 2150, 321, 399, 202, 10167}
        if errorCode not in ignore_codes:
            logging.error(f"Error {errorCode} (ReqId: {reqId}): {errorString}")
            if advancedOrderRejectJson:
                logging.error(f"   Advanced Reject Info: {advancedOrderRejectJson}")
        if errorCode in [200, 162, 322, 504]: # Critical errors
            self.account_summary_event.set()
            self.connection_event.set()

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        logging.info(f"Received next valid order ID: {orderId}")
        self.connection_event.set()

    def updatePortfolio(self, contract: Contract, position: float,
                        marketPrice: float, marketValue: float,
                        averageCost: float, unrealizedPNL: float,
                        realizedPNL: float, accountName: str):
        """Callback for portfolio updates."""
        self.account_name = accountName # Store account name
        # Check if position is zero, if so, consider removing or just logging
        if position == 0:
            logging.debug(f"Zero position for {contract.symbol}, might be closing out.")
            # Optionally remove from df if it exists and position becomes 0
            self.pos_df = self.pos_df[self.pos_df['Symbol'] != contract.symbol]
            return # No further processing for zero positions

        mask = self.pos_df['Symbol'] == contract.symbol
        if self.pos_df[mask].empty:
            new_row = pd.DataFrame([{
                'Symbol': contract.symbol,
                'SecType': contract.secType,
                'Position': float(position),
                'Market Value': float(marketValue),
                'Unrealized PnL': float(unrealizedPNL),
                'Avg Cost': float(averageCost)
            }])
            self.pos_df = pd.concat([self.pos_df, new_row], ignore_index=True)
        else:
            self.pos_df.loc[mask, 'Position'] = float(position)
            self.pos_df.loc[mask, 'Market Value'] = float(marketValue)
            self.pos_df.loc[mask, 'Unrealized PnL'] = float(unrealizedPNL)
            self.pos_df.loc[mask, 'Avg Cost'] = float(averageCost)
        
        logging.info(f"Position Update: {position} {contract.symbol} @ ${marketPrice:.2f} | Val: ${marketValue:.2f} | PnL: ${unrealizedPNL:.2f}")

    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        """Callback for account value updates."""
        self.account_name = accountName
        if currency == "USD": # Process only USD values for simplicity here
            if key == "NetLiquidation":
                self.net_liquidation = float(val)
                logging.debug(f"Account Update ({accountName}): Net Liquidation (USD) = {val}")
            elif key == "UnrealizedPnL":
                self.unrealized_pnl_total = float(val)
                logging.debug(f"Account Update ({accountName}): Unrealized PnL (USD) = {val}")
            # Add other keys if needed, e.g., TotalCashValue, MaintMarginReq

    def accountDownloadEnd(self, accountName: str):
        """Callback indicating account download is complete."""
        super().accountDownloadEnd(accountName)
        logging.info(f"Account download ended for: {accountName}")
        self.account_summary_event.set()

    def connect_to_ib(self, host: str = "127.0.0.1", port: int = None, client_id: int = CLIENT_ID_POSITIONS) -> bool:
        if port is None:
            port = PAPER_PORT if IS_PAPER_TRADING else LIVE_PORT
        trading_mode = "Paper" if port == PAPER_PORT else "Live"
        logging.info(f"Connecting to {trading_mode} Trading on {host}:{port} (ClientId: {client_id})...")
        self.connect(host, port, clientId=client_id)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        logging.info("Connection thread started.")
        if self.connection_event.wait(timeout=WAIT_TIME):
            logging.info("Successfully connected and received nextValidId.")
            return True
        else:
            logging.error(f"Connection timeout after {WAIT_TIME} seconds waiting for nextValidId.")
            self.disconnect_from_ib()
            return False

    def disconnect_from_ib(self):
        if self.isConnected():
            logging.info("Unsubscribing from account updates...")
            self.reqAccountUpdates(False, "") # Important to unsubscribe
            time.sleep(1) # Give it a moment
            logging.info("Disconnecting from IB...")
            self.disconnect()
            if self.thread.is_alive():
                self.thread.join(timeout=2)
            logging.info("Disconnected safely.")
        else:
            logging.info("Already disconnected or connection failed.")

    def fetch_account_summary(self, account_code: str = "") -> pd.DataFrame:
        """Subscribes to account updates and fetches positions.
        Args:
            account_code (str): Specific account code. Empty for all subscribed accounts.
        Returns:
            pd.DataFrame: DataFrame of positions.
        """
        if not self.isConnected():
            logging.error("Not connected. Cannot fetch account summary.")
            return pd.DataFrame()

        logging.info(f"Fetching account summary and positions (Account: {account_code if account_code else 'All'})...")
        self.pos_df = self.pos_df.iloc[0:0] # Clear previous position data
        self.net_liquidation = 0.0
        self.unrealized_pnl_total = 0.0
        self.account_summary_event.clear()
        
        self.reqAccountUpdates(True, account_code)
        
        if self.account_summary_event.wait(timeout=WAIT_TIME):
            if not self.pos_df.empty:
                logging.info(f"Retrieved {len(self.pos_df)} position(s).")
            else:
                logging.info("No positions found in the account(s).")
            logging.info(f"Account Name: {self.account_name}, Net Liquidation: {self.net_liquidation:.2f}, Total Unrealized PnL: {self.unrealized_pnl_total:.2f}")
        else:
            logging.warning("Timeout waiting for account summary and positions.")
            if not self.pos_df.empty:
                logging.warning(f"Returning {len(self.pos_df)} partially received positions before timeout.")
        return self.pos_df.copy()

def display_positions(df: pd.DataFrame, net_liq: float, total_pnl: float, acc_name: str):
    if df.empty and net_liq == 0 and total_pnl == 0:
        logging.info("📭 No position or account summary data to display.")
        return

    logging.info(f"\n📊 Account Summary for: {acc_name if acc_name else 'N/A'}")
    header = "=" * 110
    logging.info(header)
    if not df.empty:
        display_df = df.copy()
        for col in ['Market Value', 'Unrealized PnL', 'Avg Cost']:
            display_df[col] = display_df[col].apply(lambda x: f'${float(x):,.2f}' if pd.notna(x) else '-')
        display_df['Position'] = display_df['Position'].apply(lambda x: f'{float(x):,.0f}' if pd.notna(x) else '-')
        cols_to_display = ['Symbol', 'SecType', 'Position', 'Avg Cost', 'Market Value', 'Unrealized PnL']
        existing_cols = [col for col in cols_to_display if col in display_df.columns]
        # Using print for table
        print(display_df[existing_cols].to_string(index=False))
    else:
        logging.info("No positions held.")
    logging.info(header)
    logging.info(f"💰 Total Net Liquidation Value: ${net_liq:,.2f}")
    logging.info(f"📈 Total Unrealized P&L: ${total_pnl:,.2f}")
    logging.info(header)

def main():
    logging.info("--- Starting Position Details Script ---")
    app = PositionDetailsApp()

    # Specify your account code if you have multiple and want to target one
    # e.g., account_code = "U1234567"
    target_account_code = "" 

    if app.connect_to_ib(client_id=CLIENT_ID_POSITIONS):
        try:
            positions_df = app.fetch_account_summary(account_code=target_account_code)
            display_positions(positions_df, app.net_liquidation, app.unrealized_pnl_total, app.account_name)
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
        finally:
            app.disconnect_from_ib() # This now also unsubscribes
    else:
        logging.error("Failed to connect to Interactive Brokers. Exiting.")
    
    logging.info("--- Position Details Script Finished ---")

if __name__ == "__main__":
    main()