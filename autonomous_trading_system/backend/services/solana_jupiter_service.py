"""
Solana Jupiter Service (Enhanced)
---
This service encapsulates all trading, token, and wallet interaction
functions for the Solana blockchain, primarily using the Jupiter and Birdeye APIs.

This enhanced version can be initialized with a specific private key to manage
multiple trading wallets, integrating functionality from both `nice_funcs.py` (Day 48)
and `nice_funcs2.py` (Day 50).
"""
import logging
import os
import base64
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from autonomous_trading_system.backend.core.config import get_settings
from typing import Optional

logger = logging.getLogger(__name__)

class SolanaJupiterService:
    """A comprehensive service for Solana trading via Jupiter."""

    def __init__(self, private_key: Optional[str] = None):
        """
        Initializes the service.

        Args:
            private_key: The base58 encoded private key for the wallet to use.
                         If None, it defaults to the primary SOL_KEY from settings.
        """
        self.settings = get_settings()
        self.rpc_url = self.settings.RPC_URL
        self.birdeye_api_key = self.settings.BIRDEYE_API_KEY
        
        if private_key:
            self.wallet_keypair = Keypair.from_base58_string(private_key)
        else:
            self.wallet_keypair = Keypair.from_base58_string(self.settings.SOL_KEY)
            
        self.http_client = Client(self.rpc_url)
        logger.info(f"✅ SolanaJupiterService initialized for wallet: {self.wallet_keypair.pubkey()}")

    def get_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int):
        """Gets a swap quote from Jupiter API."""
        url = f"https://quote-api.jup.ag/v6/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount}&slippageBps={slippage_bps}&restrictIntermediateTokens=true"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error getting Jupiter quote: {e}")
            return None

    def _get_swap_transaction(self, quote: dict, priority_fee_lamports: int):
        """Gets the swap transaction from Jupiter API."""
        url = 'https://quote-api.jup.ag/v6/swap'
        headers = {"Content-Type": "application/json"}
        data = {
            "quoteResponse": quote,
            "userPublicKey": str(self.wallet_keypair.pubkey()),
            "prioritizationFeeLamports": priority_fee_lamports
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error getting Jupiter swap transaction: {e}")
            return None

    def _send_transaction(self, swap_transaction: str) -> Optional[str]:
        """Decodes, signs, and sends a transaction."""
        try:
            raw_tx = base64.b64decode(swap_transaction)
            versioned_tx = VersionedTransaction.from_bytes(raw_tx)
            signed_tx = VersionedTransaction(versioned_tx.message, [self.wallet_keypair])
            tx_id = self.http_client.send_raw_transaction(bytes(signed_tx), TxOpts(skip_preflight=True)).value
            logger.info(f"✅ Transaction sent successfully: https://solscan.io/tx/{tx_id}")
            return str(tx_id)
        except Exception as e:
            logger.error(f"❌ Error sending transaction: {e}")
            return None

    def market_swap(self, input_mint: str, output_mint: str, amount_atomic: int, slippage_bps: int, priority_fee: int):
        """Performs a market swap (buy or sell)."""
        logger.info(f"🚀 Initiating swap: {amount_atomic} of {input_mint} -> {output_mint}")
        quote = self.get_quote(input_mint, output_mint, amount_atomic, slippage_bps)
        if not quote:
            return None

        swap_data = self._get_swap_transaction(quote, priority_fee)
        if not swap_data or 'swapTransaction' not in swap_data:
            logger.error("❌ Failed to get swap transaction data.")
            return None

        return self._send_transaction(swap_data['swapTransaction'])

    def get_token_overview(self, token_address: str):
        """Fetches token overview from Birdeye."""
        url = f"https://public-api.birdeye.so/defi/token_overview?address={token_address}"
        headers = {"X-API-KEY": self.birdeye_api_key}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get('data', {})
        except requests.RequestException as e:
            logger.error(f"❌ Error fetching Birdeye token overview for {token_address}: {e}")
            return {}

    def get_decimals(self, token_mint_address: str) -> Optional[int]:
        """Fetches the decimal places for a token."""
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
            "params": [token_mint_address, {"encoding": "jsonParsed"}]
        }
        try:
            response = requests.post(self.rpc_url, json=payload)
            response.raise_for_status()
            data = response.json()
            decimals = data['result']['value']['data']['parsed']['info']['decimals']
            return decimals
        except (requests.RequestException, KeyError, TypeError) as e:
            logger.error(f"❌ Could not get decimals for {token_mint_address}: {e}")
            return None
            
    def get_wallet_token_balance(self, token_mint_address: str) -> dict:
        """
        Gets the balance of a specific token for the service's wallet.
        Returns a dictionary with 'uiAmount' and 'amount' (atomic).
        """
        try:
            pubkey = self.wallet_keypair.pubkey()
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
                "params": [
                    str(pubkey),
                    {"mint": token_mint_address},
                    {"encoding": "jsonParsed"}
                ]
            }
            response = requests.post(self.rpc_url, json=payload)
            response.raise_for_status()
            data = response.json()

            if data['result']['value']:
                account_info = data['result']['value'][0]['account']['data']['parsed']['info']
                return {
                    'uiAmount': account_info.get('tokenAmount', {}).get('uiAmount', 0.0),
                    'amount': int(account_info.get('tokenAmount', {}).get('amount', 0))
                }
        except Exception as e:
            logger.error(f"❌ Error getting token balance for {token_mint_address}: {e}")
        
        return {'uiAmount': 0.0, 'amount': 0}

    def fetch_wallet_holdings(self) -> pd.DataFrame:
        """Fetches all token holdings for the configured wallet from Birdeye."""
        url = f"https://public-api.birdeye.so/v1/wallet/token_list?wallet={self.wallet_keypair.pubkey()}"
        headers = {"x-chain": "solana", "X-API-KEY": self.birdeye_api_key}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            json_response = response.json()
            if 'data' in json_response and 'items' in json_response['data']:
                df = pd.DataFrame(json_response['data']['items'])
                if not df.empty:
                    df = df[['address', 'uiAmount', 'valueUsd']]
                    df = df.rename(columns={'address': 'Mint Address', 'uiAmount': 'Amount', 'valueUsd': 'USD Value'})
                    df = df.dropna(subset=['USD Value'])
                    df = df[df['USD Value'] > 0.05]
                    return df
        except (requests.RequestException, KeyError) as e:
            logger.error(f"❌ Failed to retrieve or process wallet holdings: {e}")

        return pd.DataFrame(columns=['Mint Address', 'Amount', 'USD Value'])

    def kill_switch(self, token_mint_address: str):
        """Sells the entire balance of a given token."""
        logger.warning(f"🚨 KILL SWITCH ACTIVATED for token: {token_mint_address}")

        # Safety check: do not sell critical assets like USDC or protected tokens
        if token_mint_address in self.settings.DO_NOT_TRADE_LIST or token_mint_address == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v":
            logger.info(f"🛡️ Token {token_mint_address} is protected. Kill switch aborted.")
            return

        balance = self.get_wallet_token_balance(token_mint_address)
        atomic_balance = balance['amount']

        if atomic_balance <= 0:
            logger.info(f"🤷 No balance of {token_mint_address} to sell.")
            return

        logger.info(f"💰 Selling {balance['uiAmount']} of {token_mint_address}.")
        
        # Market sell the entire atomic balance to USDC
        tx_id = self.market_swap(
            input_mint=token_mint_address,
            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
            amount_atomic=atomic_balance,
            slippage_bps=self.settings.SDZ_V2_SLIPPAGE_BPS,
            priority_fee=self.settings.SDZ_V2_PRIORITY_FEE
        )

        if tx_id:
            logger.info(f"✅ Kill switch successful for {token_mint_address}. Transaction: {tx_id}")
            # Log to closed positions file
            try:
                with open(self.settings.SDZ_V2_CLOSED_POSITIONS_TXT, 'r+') as f:
                    if token_mint_address not in f.read():
                        f.write(f"{token_mint_address}\n")
            except Exception as e:
                logger.error(f"❌ Failed to write to closed positions log: {e}")
        else:
            logger.error(f"❌ Kill switch failed for {token_mint_address}.")

    def _ensure_data_dir(self):
        """Ensures that the data directory and required files exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.closed_positions_file):
            with open(self.closed_positions_file, 'w') as f:
                pass  # Create empty file

    def _make_birdeye_request(self, endpoint: str, params: dict = None):
        """Helper to make GET requests to the Birdeye API."""
        url = f"https://public-api.birdeye.so/{endpoint}"
        headers = {"X-API-KEY": self.birdeye_api_key}
        if "v1/wallet/token_list" in endpoint:
            headers["x-chain"] = "solana"

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Birdeye API request failed for {url}: {e}")
            return None

    def _read_closed_positions(self):
        """Reads and returns a set of token addresses from the closed positions file."""
        try:
            with open(self.closed_positions_file, 'r') as f:
                return set(line.strip() for line in f)
        except FileNotFoundError:
            return set()

    def _add_to_closed_positions(self, token_mint_address: str):
        """Adds a token address to the closed positions file."""
        closed_positions = self._read_closed_positions()
        if token_mint_address not in closed_positions:
            with open(self.closed_positions_file, 'a') as f:
                f.write(token_mint_address + '\n')
            logger.info(f"Added {token_mint_address} to closed positions.")

    def get_token_price(self, address: str):
        """Fetches the current price of a token."""
        response_data = self._make_birdeye_request(f"defi/price?address={address}")
        if response_data and response_data.get('success'):
            return float(response_data['data']['value'])
        logger.warning(f"Could not get price for {address}")
        return None

    def get_wallet_holdings(self, wallet_address: str, token_filter: str = None):
        """Fetches and filters token holdings for a given wallet address."""
        response_data = self._make_birdeye_request(f"v1/wallet/token_list?wallet={wallet_address}")
        if not response_data or not response_data.get('success'):
            return pd.DataFrame()
        
        items = response_data.get('data', {}).get('items', [])
        if not items:
            return pd.DataFrame()
            
        df = pd.DataFrame(items)
        df = df.rename(columns={'address': 'mint_address', 'uiAmount': 'amount', 'valueUsd': 'usd_value'})
        df = df[['mint_address', 'amount', 'usd_value']]
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['usd_value'] = pd.to_numeric(df['usd_value'], errors='coerce')
        df = df.dropna().query('usd_value > 0.05')

        if token_filter:
            df = df[df['mint_address'] == token_filter]

        return df

    def get_token_balance(self, token_mint_address: str):
        """Convenience method to get the balance of a single token in the service's wallet."""
        df = self.get_wallet_holdings(str(self.wallet_keypair.pubkey()), token_filter=token_mint_address)
        if not df.empty:
            return float(df['amount'].iloc[0])
        return 0.0

    def chunk_kill(self, contract_address: str, max_usd_order_size: float, sleep_between: int):
        """Sells off a position in chunks until it's gone."""
        logger.info(f"Initiating CHUNK KILL for {contract_address}.")
        self._add_to_closed_positions(contract_address)

        while True:
            balance = self.get_token_balance(contract_address)
            price = self.get_token_price(contract_address)

            if not balance or not price or balance <= 0 or price <= 0:
                logger.info(f"Balance or price for {contract_address} is zero/invalid. Exiting chunk kill.")
                break

            usd_value = balance * price
            if usd_value < self.settings.EZ_BOT_MIN_BUY_USD_THRESHOLD:
                logger.info(f"Position value (${usd_value:.2f}) is negligible. Chunk kill complete.")
                break
            
            decimals = self.get_decimals(contract_address)
            if decimals is None:
                break
                
            sell_amount_units = balance if usd_value <= max_usd_order_size else max_usd_order_size / price
            sell_amount_lamports = int(sell_amount_units * (10**decimals))

            if sell_amount_lamports <= 0:
                logger.warning("Calculated sell amount is zero lamports. Breaking.")
                break

            for _ in range(3): # Triple tap
                self.market_swap(contract_address, self.settings.QBS_QUOTE_TOKEN_ADDRESS, sell_amount_lamports, self.settings.EZ_BOT_SLIPPAGE_BPS, self.settings.EZ_BOT_PRIORITY_FEE)
                time.sleep(sleep_between)
            time.sleep(15) # Cooldown
    
    def get_ohlcv_data(self, address: str, days_back: int, timeframe: str):
        """Fetches OHLCV data and calculates basic indicators."""
        now = datetime.now()
        time_from = int((now - timedelta(days=days_back)).timestamp())
        params = {'address': address, 'type': timeframe, 'time_from': time_from, 'time_to': int(now.timestamp())}
        
        json_response = self._make_birdeye_request("defi/ohlcv", params=params)
        if not json_response or not json_response.get('success'):
            return pd.DataFrame()

        items = json_response.get('data', {}).get('items', [])
        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
        df['datetime'] = pd.to_datetime(df['unixTime'], unit='s')
        
        # Add indicators
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=40, append=True)
        df.ta.rsi(length=14, append=True)
        return df

    def get_supply_demand_zones(self, contract_address: str, days_back: int, timeframe: str):
        """Calculates supply and demand zones from OHLCV data."""
        df = self.get_ohlcv_data(contract_address, days_back, timeframe)
        if df.empty or len(df) <= 2:
            logger.warning(f"Not enough data for S/D zones for {contract_address}")
            return None
        
        relevant_df = df.iloc[:-2] # Exclude last 2 bars
        support = relevant_df['Close'].min()
        resistance = relevant_df['Close'].max()
        
        return {
            'dz_low': relevant_df['Low'].min(),
            'dz_high': support,
            'sz_low': resistance,
            'sz_high': relevant_df['High'].max()
        }
