"""
Solana Trading Utilities Service

A comprehensive collection of utility functions for Solana trading operations.
Based on nice_funcs.py with enterprise enhancements and async support.

Features:
- Birdeye API integration (token data, prices, OHLCV)
- Jupiter API integration (swaps, quotes)
- Solana wallet and blockchain operations
- Position management and PnL tracking
- AI-powered trading decisions (OpenAI GPT-4)
- Technical analysis and supply/demand zones
- Liquidation data analysis
- Binance funding rate tracking
"""

import asyncio
import logging
import os
import json
import time
import math
import base64
import re
import requests
import pandas as pd
import pandas_ta as ta
import pytz
import websockets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client
from solana.rpc.types import TxOpts

# Setup logging
logger = logging.getLogger(__name__)

# Constants
BIRDEYE_BASE_URL = "https://public-api.birdeye.so/defi"
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com/"

@dataclass
class TokenOverview:
    """Token overview data structure"""
    buy1h: int
    sell1h: int
    trade1h: int
    buy_percentage: float
    sell_percentage: float
    minimum_trades_met: bool
    price_changes: Dict[str, float]
    rug_pull: bool
    unique_wallet_24h: int
    volume_24h_usd: float
    liquidity: float
    social_links: List[Dict[str, str]]

@dataclass
class SupplyDemandZones:
    """Supply and demand zones data"""
    demand_zone: List[float]
    supply_zone: List[float]
    timeframe: str
    calculated_at: datetime

@dataclass
class LiquidationData:
    """Liquidation analysis results"""
    s_liqs_usd: float
    l_liqs_usd: float
    total_liqs_usd: float
    symbol: str
    lookback_minutes: int
    calculated_at: datetime

class SolanaTradeAction(Enum):
    """Trading action types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SolanaTradingUtilsService:
    """
    Solana Trading Utilities Service
    
    Comprehensive trading utilities for Solana including:
    - Market data from Birdeye
    - Trading via Jupiter
    - Position management
    - AI-powered decisions
    - Technical analysis
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.is_running = False
        
        # API configurations
        self.birdeye_api_key = self.config.get('birdeye_api_key')
        self.openai_api_key = self.config.get('openai_api_key')
        self.solana_rpc_endpoint = self.config.get('solana_rpc_endpoint', SOLANA_RPC_URL)
        self.wallet_secret_key = self.config.get('wallet_secret_key')
        self.wallet_address = self.config.get('wallet_address')
        
        # Trading configurations
        self.minimum_trades_threshold = self.config.get('minimum_trades_in_last_hour', 50)
        self.stop_loss_percentage = self.config.get('stop_loss_percentage', -0.20)
        self.sell_at_multiple = self.config.get('sell_at_multiple', 2.0)
        self.priority_fee = self.config.get('priority_fee', 5000)
        self.do_not_trade_list = self.config.get('do_not_trade_list', [USDC_MINT])
        
        # Initialize clients
        self.rpc_client = None
        self.wallet_keypair = None
        self.openai_client = None
        
        # Cache for frequently accessed data
        self.price_cache = {}
        self.cache_ttl = 30  # seconds
        
        logger.info("🔧 Solana Trading Utilities Service initialized")
    
    async def start(self):
        """Start the service and initialize connections"""
        try:
            self.is_running = True
            
            # Initialize Solana RPC client
            self.rpc_client = Client(self.solana_rpc_endpoint)
            
            # Initialize wallet keypair if provided
            if self.wallet_secret_key:
                try:
                    self.wallet_keypair = Keypair.from_base58_string(self.wallet_secret_key)
                    logger.info(f"✅ Wallet keypair loaded: {str(self.wallet_keypair.pubkey())[:8]}...")
                except Exception as e:
                    logger.error(f"❌ Failed to load wallet keypair: {e}")
            
            # Initialize OpenAI client if API key provided
            if self.openai_api_key:
                try:
                    import openai
                    self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                    logger.info("✅ OpenAI client initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            
            logger.info("🚀 Solana Trading Utilities Service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Solana Trading Utilities Service: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the service"""
        self.is_running = False
        logger.info("🛑 Solana Trading Utilities Service stopped")
    
    # ============== UTILITY FUNCTIONS ==============
    
    def print_pretty_json(self, data):
        """Prints JSON data in a readable format"""
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(data)
    
    def find_urls(self, text: str) -> List[str]:
        """Extracts URLs from a string using regex"""
        return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(text))
    
    def round_down(self, value: float, decimals: int) -> float:
        """Rounds a float down to a specified number of decimal places"""
        try:
            factor = 10 ** decimals
            return math.floor(float(value) * factor) / factor
        except (ValueError, TypeError):
            return 0.0
    
    def get_time_range(self, days_back: int = 10) -> Tuple[int, int]:
        """Calculates the timestamp range for the last N days"""
        now = datetime.now()
        start_date = now - timedelta(days=days_back)
        time_to = int(now.timestamp())
        time_from = int(start_date.timestamp())
        return time_from, time_to
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self.price_cache:
            return False
        
        cached_time = self.price_cache[key].get('timestamp', 0)
        return (time.time() - cached_time) < self.cache_ttl
    
    # ============== BIRDEYE API FUNCTIONS ==============
    
    async def get_token_overview(self, token_mint_address: str) -> Optional[TokenOverview]:
        """Fetches comprehensive token overview from Birdeye API"""
        if not self.birdeye_api_key:
            logger.warning("Birdeye API key not configured")
            return None
        
        logger.info(f"Getting token overview for {token_mint_address[-6:]}...")
        
        url = f"{BIRDEYE_BASE_URL}/token_overview?address={token_mint_address}"
        headers = {"X-API-KEY": self.birdeye_api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json().get('data', {})
            if not data:
                logger.warning(f"Empty data received for {token_mint_address[-6:]}")
                return None
            
            # Calculate metrics
            buy1h = data.get('buy1h', 0)
            sell1h = data.get('sell1h', 0)
            trade1h = buy1h + sell1h
            
            buy_percentage = (buy1h / trade1h * 100) if trade1h else 0
            sell_percentage = (sell1h / trade1h * 100) if trade1h else 0
            
            # Extract price changes
            price_changes = {k: v for k, v in data.items() if 'priceChange' in k}
            
            # Check for potential rug pull
            rug_pull = any(value < -80 for value in price_changes.values() if isinstance(value, (int, float)))
            
            if rug_pull:
                logger.warning(f"⚠️ Potential rug pull detected for {token_mint_address[-6:]}")
            
            # Extract social links
            extensions = data.get('extensions', {})
            description = extensions.get('description', '') if extensions else ''
            urls = self.find_urls(description)
            
            social_links = []
            for url in urls:
                if 't.me' in url:
                    social_links.append({'telegram': url})
                elif 'twitter.com' in url:
                    social_links.append({'twitter': url})
                elif 'discord' not in url and 'youtube' not in url:
                    social_links.append({'website': url})
            
            return TokenOverview(
                buy1h=buy1h,
                sell1h=sell1h,
                trade1h=trade1h,
                buy_percentage=buy_percentage,
                sell_percentage=sell_percentage,
                minimum_trades_met=trade1h >= self.minimum_trades_threshold,
                price_changes=price_changes,
                rug_pull=rug_pull,
                unique_wallet_24h=data.get('uniqueWallet24h', 0),
                volume_24h_usd=data.get('v24hUSD', 0),
                liquidity=data.get('liquidity', 0),
                social_links=social_links
            )
            
        except Exception as e:
            logger.error(f"Error getting token overview for {token_mint_address[-6:]}: {e}")
            return None
    
    async def get_token_price(self, token_mint_address: str, use_cache: bool = True) -> Optional[float]:
        """Fetches current token price from Birdeye API with caching"""
        if not self.birdeye_api_key:
            logger.warning("Birdeye API key not configured")
            return None
        
        # Check cache first
        cache_key = f"price_{token_mint_address}"
        if use_cache and self._is_cache_valid(cache_key):
            return self.price_cache[cache_key]['price']
        
        url = f"{BIRDEYE_BASE_URL}/price?address={token_mint_address}"
        headers = {"X-API-KEY": self.birdeye_api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success') and 'value' in data.get('data', {}):
                price = data['data']['value']
                
                # Cache the result
                self.price_cache[cache_key] = {
                    'price': price,
                    'timestamp': time.time()
                }
                
                return price
            else:
                logger.warning(f"Invalid price response for {token_mint_address[-6:]}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting token price for {token_mint_address[-6:]}: {e}")
            return None
    
    async def get_ohlcv_data(self, token_mint_address: str, days_back: int = 10, timeframe: str = '1m') -> pd.DataFrame:
        """Fetches OHLCV data from Birdeye and adds technical indicators"""
        if not self.birdeye_api_key:
            logger.warning("Birdeye API key not configured")
            return pd.DataFrame()
        
        logger.info(f"Fetching {timeframe} OHLCV data for {token_mint_address[-6:]} ({days_back} days)")
        
        time_from, time_to = self.get_time_range(days_back)
        url = f"{BIRDEYE_BASE_URL}/ohlcv?address={token_mint_address}&type={timeframe}&time_from={time_from}&time_to={time_to}"
        headers = {"X-API-KEY": self.birdeye_api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            json_response = response.json()
            items = json_response.get('data', {}).get('items', [])
            
            if not items:
                logger.warning(f"No OHLCV data found for {token_mint_address[-6:]}")
                return pd.DataFrame()
            
            # Process data
            processed_data = []
            for item in items:
                processed_data.append({
                    'datetime': datetime.utcfromtimestamp(item['unixTime']),
                    'Open': item['o'],
                    'High': item['h'],
                    'Low': item['l'],
                    'Close': item['c'],
                    'Volume': item['v']
                })
            
            df = pd.DataFrame(processed_data)
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
            df.set_index('datetime', inplace=True)
            
            # Pad data if insufficient for TA indicators
            required_rows = 40
            if len(df) < required_rows and not df.empty:
                rows_to_add = required_rows - len(df)
                first_row_replicated = pd.concat([df.iloc[0:1]] * rows_to_add, ignore_index=False)
                df = pd.concat([first_row_replicated, df])
                df.sort_index(inplace=True)
            
            # Add technical indicators
            if not df.empty:
                df['MA20'] = ta.sma(df['Close'], length=20)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['MA40'] = ta.sma(df['Close'], length=40)
                df['Price_above_MA20'] = df['Close'] > df['MA20']
                df['Price_above_MA40'] = df['Close'] > df['MA40']
                df['MA20_above_MA40'] = df['MA20'] > df['MA40']
                df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {token_mint_address[-6:]}: {e}")
            return pd.DataFrame()
    
    # ============== SOLANA BLOCKCHAIN FUNCTIONS ==============
    
    async def get_token_decimals(self, token_mint_address: str) -> Optional[int]:
        """Fetches token decimals from Solana RPC"""
        if not self.rpc_client:
            logger.warning("Solana RPC client not initialized")
            return None
        
        try:
            logger.debug(f"Getting decimals for {token_mint_address[-6:]}...")
            
            headers = {"Content-Type": "application/json"}
            payload = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [token_mint_address, {"encoding": "jsonParsed"}]
            })
            
            response = requests.post(self.solana_rpc_endpoint, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            decimals = data['result']['value']['data']['parsed']['info']['decimals']
            return decimals
            
        except Exception as e:
            logger.error(f"Error getting decimals for {token_mint_address[-6:]}: {e}")
            return None
    
    async def get_wallet_holdings(self, wallet_address: str = None) -> pd.DataFrame:
        """Fetches token holdings for a wallet from Birdeye"""
        if not self.birdeye_api_key:
            logger.warning("Birdeye API key not configured")
            return pd.DataFrame()
        
        wallet_addr = wallet_address or self.wallet_address
        if not wallet_addr:
            logger.warning("No wallet address provided or configured")
            return pd.DataFrame()
        
        url = f"https://public-api.birdeye.so/v1/wallet/token_list?wallet={wallet_addr}"
        headers = {"x-chain": "solana", "X-API-KEY": self.birdeye_api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data and 'items' in data['data']:
                items = data['data']['items']
                if items:
                    df = pd.DataFrame(items)
                    if all(col in df.columns for col in ['address', 'uiAmount', 'valueUsd']):
                        holdings_df = df[['address', 'uiAmount', 'valueUsd']].copy()
                        holdings_df.rename(columns={
                            'address': 'Mint Address',
                            'uiAmount': 'Amount',
                            'valueUsd': 'USD Value'
                        }, inplace=True)
                        holdings_df = holdings_df.dropna(subset=['USD Value'])
                        holdings_df = holdings_df[holdings_df['USD Value'] > 0.05]
                        
                        total_usd = holdings_df['USD Value'].sum()
                        logger.info(f"💰 Wallet {wallet_addr[-6:]} Total: ${total_usd:.2f}")
                        
                        return holdings_df
            
            logger.info(f"No significant holdings found for wallet {wallet_addr[-6:]}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting wallet holdings for {wallet_addr[-6:]}: {e}")
            return pd.DataFrame()
    
    async def get_position(self, token_mint_address: str, wallet_address: str = None) -> float:
        """Gets the balance of a specific token in wallet"""
        wallet_addr = wallet_address or self.wallet_address
        if not wallet_addr:
            logger.warning("No wallet address provided or configured")
            return 0.0
        
        holdings_df = await self.get_wallet_holdings(wallet_addr)
        if holdings_df.empty:
            return 0.0
        
        # Filter for specific token
        token_holdings = holdings_df[holdings_df['Mint Address'] == token_mint_address]
        if not token_holdings.empty:
            try:
                return float(token_holdings['Amount'].iloc[0])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert balance to float for {token_mint_address[-6:]}")
                return 0.0
        
        return 0.0
    
    # ============== JUPITER TRADING FUNCTIONS ==============
    
    async def get_jupiter_quote(self, input_mint: str, output_mint: str, amount_lamports: str, slippage_bps: int = 50) -> Optional[Dict]:
        """Gets trading quote from Jupiter API"""
        quote_url = f'{JUPITER_QUOTE_URL}?inputMint={input_mint}&outputMint={output_mint}&amount={amount_lamports}&slippageBps={slippage_bps}'
        
        try:
            response = requests.get(quote_url, timeout=10)
            response.raise_for_status()
            
            quote_data = response.json()
            if 'outAmount' not in quote_data:
                logger.warning(f"Invalid Jupiter quote response: {quote_data}")
                return None
            
            return quote_data
            
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            return None
    
    async def perform_jupiter_swap(self, quote_response: Dict, priority_fee_lamports: int = None) -> Optional[str]:
        """Performs a swap using Jupiter API"""
        if not self.wallet_keypair:
            logger.error("Wallet keypair not loaded")
            return None
        
        if not quote_response:
            logger.error("No quote response provided")
            return None
        
        priority_fee = priority_fee_lamports or self.priority_fee
        
        payload = json.dumps({
            "quoteResponse": quote_response,
            "userPublicKey": str(self.wallet_keypair.pubkey()),
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": priority_fee
        })
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(JUPITER_SWAP_URL, headers=headers, data=payload, timeout=15)
            response.raise_for_status()
            
            swap_data = response.json()
            if 'swapTransaction' not in swap_data:
                logger.warning(f"Invalid Jupiter swap response: {swap_data}")
                return None
            
            # Sign and send transaction
            swap_tx_b64 = swap_data['swapTransaction']
            raw_tx_bytes = base64.b64decode(swap_tx_b64)
            versioned_tx = VersionedTransaction.from_bytes(raw_tx_bytes)
            signed_tx = VersionedTransaction(versioned_tx.message, [self.wallet_keypair])
            
            if self.rpc_client:
                tx_id = self.rpc_client.send_raw_transaction(bytes(signed_tx), TxOpts(skip_preflight=True)).value
                tx_signature = str(tx_id)
                logger.info(f"🚀 Transaction sent: https://solscan.io/tx/{tx_signature}")
                return tx_signature
            else:
                logger.error("RPC client not available")
                return None
                
        except Exception as e:
            logger.error(f"Error performing Jupiter swap: {e}")
            return None
    
    async def market_buy(self, output_token_mint: str, usdc_amount_lamports: str, slippage_bps: int = 50) -> Optional[str]:
        """Buys a token using USDC via Jupiter"""
        logger.info(f"🛒 Market buy: {usdc_amount_lamports} lamports USDC → {output_token_mint[-6:]}")
        
        quote = await self.get_jupiter_quote(USDC_MINT, output_token_mint, usdc_amount_lamports, slippage_bps)
        if quote:
            return await self.perform_jupiter_swap(quote)
        return None
    
    async def market_sell(self, input_token_mint: str, input_token_amount_lamports: str, slippage_bps: int = 50) -> Optional[str]:
        """Sells a token for USDC via Jupiter"""
        logger.info(f"💰 Market sell: {input_token_amount_lamports} lamports {input_token_mint[-6:]} → USDC")
        
        quote = await self.get_jupiter_quote(input_token_mint, USDC_MINT, input_token_amount_lamports, slippage_bps)
        if quote:
            return await self.perform_jupiter_swap(quote)
        return None
    
    # ============== POSITION MANAGEMENT ==============
    
    async def calculate_pnl(self, token_mint_address: str, initial_investment_usd: float) -> Dict[str, Any]:
        """Calculates PnL for a position"""
        try:
            balance_tokens = await self.get_position(token_mint_address)
            if balance_tokens <= 0:
                return {"status": "no_position", "pnl_usd": 0, "pnl_percentage": 0}
            
            current_price = await self.get_token_price(token_mint_address)
            if not current_price:
                return {"status": "price_error", "pnl_usd": 0, "pnl_percentage": 0}
            
            current_value_usd = balance_tokens * current_price
            pnl_usd = current_value_usd - initial_investment_usd
            pnl_percentage = (pnl_usd / initial_investment_usd) * 100 if initial_investment_usd > 0 else 0
            
            target_tp_usd = initial_investment_usd * self.sell_at_multiple
            target_sl_usd = initial_investment_usd * (1 + self.stop_loss_percentage)
            
            status = "holding"
            if current_value_usd >= target_tp_usd:
                status = "take_profit_hit"
            elif current_value_usd <= target_sl_usd:
                status = "stop_loss_hit"
            
            return {
                "status": status,
                "current_value_usd": current_value_usd,
                "pnl_usd": pnl_usd,
                "pnl_percentage": pnl_percentage,
                "target_tp_usd": target_tp_usd,
                "target_sl_usd": target_sl_usd,
                "balance_tokens": balance_tokens,
                "current_price": current_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating PnL for {token_mint_address[-6:]}: {e}")
            return {"status": "error", "pnl_usd": 0, "pnl_percentage": 0}
    
    async def chunk_sell_position(self, token_mint_address: str, max_usd_sell_size: float, slippage_bps: int = 500) -> bool:
        """Gradually closes a position in chunks"""
        logger.info(f"🔻 Starting chunk sell for {token_mint_address[-6:]}, max chunk ${max_usd_sell_size:.2f}")
        
        max_retries = 3
        retry_delay = 5
        
        while True:
            balance_tokens = await self.get_position(token_mint_address)
            if balance_tokens <= 0.000001:
                logger.info(f"✅ Position for {token_mint_address[-6:]} closed")
                return True
            
            current_price = await self.get_token_price(token_mint_address)
            if not current_price or current_price <= 0:
                logger.warning(f"Invalid price for {token_mint_address[-6:]}, retrying...")
                await asyncio.sleep(retry_delay)
                continue
            
            usd_value = balance_tokens * current_price
            logger.info(f"Current balance: {balance_tokens:.6f} tokens (${usd_value:.2f})")
            
            decimals = await self.get_token_decimals(token_mint_address)
            if decimals is None:
                logger.error(f"Cannot get decimals for {token_mint_address[-6:]}")
                return False
            
            # Calculate sell size
            if usd_value <= max_usd_sell_size:
                sell_size_tokens = balance_tokens
            else:
                sell_size_tokens = max_usd_sell_size / current_price
            
            sell_size_tokens_rounded = self.round_down(sell_size_tokens, decimals)
            if sell_size_tokens_rounded <= 0:
                logger.info("Sell size too small, ending chunk sell")
                return True
            
            sell_size_lamports = str(int(sell_size_tokens_rounded * (10**decimals)))
            logger.info(f"Selling chunk: {sell_size_tokens_rounded:.{decimals}f} tokens")
            
            # Attempt to sell with retries
            for attempt in range(max_retries):
                tx_id = await self.market_sell(token_mint_address, sell_size_lamports, slippage_bps)
                if tx_id:
                    logger.info(f"✅ Sell chunk submitted (attempt {attempt+1})")
                    break
                else:
                    logger.warning(f"❌ Sell chunk failed (attempt {attempt+1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to sell chunk after {max_retries} attempts")
                return False
            
            await asyncio.sleep(10)  # Wait for blockchain update
        
        return True
    
    async def emergency_close_position(self, token_mint_address: str, slippage_bps: int = 1000) -> bool:
        """Emergency close entire position"""
        logger.warning(f"🚨 EMERGENCY CLOSE for {token_mint_address[-6:]}")
        
        balance_tokens = await self.get_position(token_mint_address)
        if balance_tokens <= 0.000001:
            logger.info("No position to close")
            return True
        
        decimals = await self.get_token_decimals(token_mint_address)
        if decimals is None:
            logger.error("Cannot get token decimals")
            return False
        
        sell_size_lamports = str(int(balance_tokens * (10**decimals)))
        logger.info(f"Selling full balance: {balance_tokens:.{decimals}f} tokens")
        
        max_retries = 5
        for attempt in range(max_retries):
            tx_id = await self.market_sell(token_mint_address, sell_size_lamports, slippage_bps)
            if tx_id:
                logger.info(f"✅ Emergency sell submitted (attempt {attempt+1})")
                return True
            else:
                logger.warning(f"❌ Emergency sell failed (attempt {attempt+1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
        
        logger.error(f"🚨 EMERGENCY CLOSE FAILED for {token_mint_address[-6:]}")
        return False
    
    # ============== TECHNICAL ANALYSIS ==============
    
    async def calculate_supply_demand_zones(self, token_address: str, timeframe: str = '1h', periods: int = 50) -> Optional[SupplyDemandZones]:
        """Calculates support/resistance zones"""
        logger.info(f"📈 Calculating S/D zones for {token_address[-6:]} ({timeframe}, {periods} periods)")
        
        df = await self.get_ohlcv_data(token_address, days_back=int(periods/24)+1, timeframe=timeframe)
        if df.empty or len(df) < 3:
            logger.warning("Not enough data for S/D zones")
            return None
        
        # Use last `periods` excluding recent 2 bars
        df_calc = df.iloc[-(periods+2):-2]
        if df_calc.empty:
            logger.warning("Insufficient historical data for S/D calculation")
            return None
        
        support_close = df_calc['Close'].min()
        resistance_close = df_calc['Close'].max()
        support_low = df_calc['Low'].min()
        resistance_high = df_calc['High'].max()
        
        return SupplyDemandZones(
            demand_zone=[support_low, support_close],
            supply_zone=[resistance_high, resistance_close],
            timeframe=timeframe,
            calculated_at=datetime.now()
        )
    
    # ============== AI TRADING DECISIONS ==============
    
    async def get_ai_trade_decision(self, token_address: str, prompt: str = None) -> Optional[SolanaTradeAction]:
        """Gets AI-powered trading decision using OpenAI GPT-4"""
        if not self.openai_client:
            logger.warning("OpenAI client not available")
            return None
        
        try:
            # Get market data
            df = await self.get_ohlcv_data(token_address, days_back=5, timeframe='1h')
            if df.empty:
                logger.warning("No market data for AI analysis")
                return None
            
            # Prepare data for AI
            cols_to_include = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'RSI', 'MA40']
            available_cols = [col for col in cols_to_include if col in df.columns]
            df_recent = df.tail(20)[available_cols]
            df_str = df_recent.reset_index().to_string(index=False)
            
            default_prompt = "Analyze this OHLCV data with technical indicators and recommend a trading action"
            analysis_prompt = prompt or default_prompt
            
            detailed_prompt = f"{analysis_prompt}\n\nMarket Data:\n{df_str}\n\nBased on this data, should I BUY, SELL, or HOLD? Respond with only: BUY, SELL, or HOLD"
            
            logger.info("🤖 Consulting AI for trading decision...")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert cryptocurrency trader. Analyze market data and provide clear trading recommendations."},
                        {"role": "user", "content": detailed_prompt}
                    ],
                    timeout=30
                )
            )
            
            if response.choices:
                decision_text = response.choices[0].message.content.strip().upper()
                logger.info(f"🤖 AI Decision: {decision_text}")
                
                if 'BUY' in decision_text:
                    return SolanaTradeAction.BUY
                elif 'SELL' in decision_text:
                    return SolanaTradeAction.SELL
                elif 'HOLD' in decision_text:
                    return SolanaTradeAction.HOLD
                else:
                    logger.warning(f"Could not parse AI decision: {decision_text}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting AI trade decision: {e}")
            return None
    
    async def get_meme_score(self, token_name: str) -> Optional[int]:
        """Gets meme-worthiness score (1-10) for a token name"""
        if not self.openai_client:
            logger.warning("OpenAI client not available")
            return None
        
        try:
            prompt = f"On a scale of 1-10, rate the meme potential of this token name: {token_name}. Consider cultural relevance, humor, and viral potential. Respond with only a number 1-10."
            
            logger.info(f"🎭 Getting meme score for: {token_name}")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You rate meme potential of cryptocurrency names."},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=15
                )
            )
            
            if response.choices:
                score_text = response.choices[0].message.content.strip()
                match = re.search(r'\d+', score_text)
                if match:
                    score = int(match.group(0))
                    logger.info(f"🎭 Meme score: {score}")
                    return min(max(score, 1), 10)  # Clamp to 1-10
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting meme score: {e}")
            return None
    
    # ============== LIQUIDATION DATA ANALYSIS ==============
    
    async def analyze_liquidations(self, symbol_contains: str, lookback_minutes: int, csv_path: str) -> Optional[LiquidationData]:
        """
        Analyze liquidation data for a specific symbol from CSV file.
        
        Args:
            symbol_contains (str): Symbol pattern to search for
            lookback_minutes (int): Minutes to look back from now
            csv_path (str): Path to liquidation CSV file
            
        Returns:
            LiquidationData object or None if error
        """
        try:
            # Read liquidation data
            df = pd.read_csv(csv_path)
            
            # Convert timestamp and filter by time
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            lookback_time = current_time - timedelta(minutes=lookback_minutes)
            
            df['order_trade_time'] = pd.to_datetime(df['order_trade_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            df = df[df['order_trade_time'] > lookback_time]
            
            # Filter by symbol
            symbol_df = df[df['symbol'].str.contains(symbol_contains, case=False, na=False)]
            
            # Calculate liquidations by side
            s_liqs = symbol_df[symbol_df['side'] == 'BUY']['usd_size'].sum()
            l_liqs = symbol_df[symbol_df['side'] == 'SELL']['usd_size'].sum()
            total_liqs = s_liqs + l_liqs
            
            logger.info(f"📊 Liquidations for {symbol_contains} (last {lookback_minutes}m): "
                       f"Short: ${s_liqs:,.2f}, Long: ${l_liqs:,.2f}, Total: ${total_liqs:,.2f}")
            
            return LiquidationData(
                s_liqs_usd=s_liqs,
                l_liqs_usd=l_liqs,
                total_liqs_usd=total_liqs,
                symbol=symbol_contains,
                lookback_minutes=lookback_minutes,
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing liquidations: {e}")
            return None

    async def calculate_liquidations(self, symbol_contains: str, l_liq_threshold: float, 
                                   s_liq_threshold: float, lookback_minutes: int,
                                   csv_path: str = None) -> Tuple[bool, bool, float, float, float]:
        """
        Calculate liquidations and check if thresholds are hit.
        
        Args:
            symbol_contains (str): Symbol pattern to search for
            l_liq_threshold (float): Long liquidation threshold in USD
            s_liq_threshold (float): Short liquidation threshold in USD
            lookback_minutes (int): Minutes to look back from now
            csv_path (str): Path to liquidation CSV file
            
        Returns:
            Tuple of (short_threshold_hit, long_threshold_hit, short_liqs, long_liqs, total_liqs)
        """
        csv_path = csv_path or self.config.get('liquidation_csv_path', 'liquidation_data.csv')
        
        try:
            # Read and process liquidation data
            df = pd.read_csv(csv_path)
            
            # Set column names if not present
            if len(df.columns) == 12:
                df.columns = [
                    "symbol", "side", "order_type", "time_in_force",
                    "original_quantity", "price", "average_price", "order_status",
                    "order_last_filled_quantity", "order_filled_accumulated_quantity",
                    "order_trade_time", "usd_size"
                ]
            
            # Convert timestamp and filter by time
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            lookback_time = current_time - timedelta(minutes=lookback_minutes)
            
            df['order_trade_time'] = pd.to_datetime(df['order_trade_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            df = df[df['order_trade_time'] > lookback_time]
            
            # Filter by symbol
            symbol_df = df[df['symbol'].str.contains(symbol_contains, case=False, na=False)]
            
            # Calculate liquidations by side
            s_liqs = symbol_df[symbol_df['side'] == 'BUY']['usd_size'].sum()
            l_liqs = symbol_df[symbol_df['side'] == 'SELL']['usd_size'].sum()
            total_liqs = s_liqs + l_liqs
            
            # Check thresholds
            short_threshold_hit = s_liqs > s_liq_threshold
            long_threshold_hit = l_liqs > l_liq_threshold
            
            logger.info(f"🔥 Liquidation analysis for {symbol_contains} (last {lookback_minutes}m):")
            logger.info(f"   Short liqs: ${s_liqs:,.2f} (threshold: ${s_liq_threshold:,.2f}) - {'HIT' if short_threshold_hit else 'OK'}")
            logger.info(f"   Long liqs: ${l_liqs:,.2f} (threshold: ${l_liq_threshold:,.2f}) - {'HIT' if long_threshold_hit else 'OK'}")
            logger.info(f"   Total liqs: ${total_liqs:,.2f}")
            
            return short_threshold_hit, long_threshold_hit, s_liqs, l_liqs, total_liqs
            
        except Exception as e:
            logger.error(f"Error calculating liquidations: {e}")
            return False, False, 0.0, 0.0, 0.0

    async def calculate_btc_liquidations(self, lookback_minutes: int = 30,
                                       csv_path: str = None) -> Tuple[float, float]:
        """
        Calculate BTC liquidations for the specified time period.
        
        Args:
            lookback_minutes (int): Minutes to look back from now
            csv_path (str): Path to liquidation CSV file
            
        Returns:
            Tuple of (short_liquidations_usd, long_liquidations_usd)
        """
        csv_path = csv_path or self.config.get('liquidation_csv_path', 'liquidation_data.csv')
        
        try:
            # Read liquidation data
            df = pd.read_csv(csv_path)
            
            # Convert timestamp and filter by time
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            lookback_time = current_time - timedelta(minutes=lookback_minutes)
            
            df['order_trade_time'] = pd.to_datetime(df['order_trade_time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            df = df[df['order_trade_time'] > lookback_time]
            
            # Filter for BTC symbols
            btc_df = df[df['symbol'].str.contains('BTC', case=False, na=False)]
            
            # Calculate liquidations by side
            s_liqs = btc_df[btc_df['side'] == 'BUY']['usd_size'].sum()
            l_liqs = btc_df[btc_df['side'] == 'SELL']['usd_size'].sum()
            
            logger.info(f"₿ BTC liquidations (last {lookback_minutes}m): "
                       f"Short: ${s_liqs:,.2f}, Long: ${l_liqs:,.2f}")
            
            return s_liqs, l_liqs
            
        except Exception as e:
            logger.error(f"Error calculating BTC liquidations: {e}")
            return 0.0, 0.0
    
    # ============== BINANCE FUNDING RATE ==============
    
    async def get_btc_funding_rate(self) -> Optional[float]:
        """Gets BTC funding rate from Binance WebSocket"""
        try:
            symbol = 'btcusdt'
            websocket_url = f'wss://fstream.binance.com/ws/{symbol}@markPrice'
            
            async with websockets.connect(websocket_url) as websocket:
                message = await websocket.recv()
                data = json.loads(message)
                funding_rate = float(data['r'])
                yearly_funding_rate = (funding_rate * 3 * 365) * 100
                
                logger.info(f"📈 BTC Funding Rate: {yearly_funding_rate:.2f}% annually")
                return yearly_funding_rate
                
        except Exception as e:
            logger.error(f"Error getting BTC funding rate: {e}")
            return None
    
    # ============== SERVICE STATUS ==============
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            "service_name": "Solana Trading Utilities Service",
            "status": "active" if self.is_running else "inactive",
            "rpc_connected": self.rpc_client is not None,
            "wallet_loaded": self.wallet_keypair is not None,
            "openai_available": self.openai_client is not None,
            "birdeye_api_configured": bool(self.birdeye_api_key),
            "cache_entries": len(self.price_cache),
            "configuration": {
                "minimum_trades_threshold": self.minimum_trades_threshold,
                "stop_loss_percentage": self.stop_loss_percentage,
                "sell_at_multiple": self.sell_at_multiple,
                "priority_fee": self.priority_fee,
                "cache_ttl": self.cache_ttl
            }
        }

    # ============== ENHANCED FUNCTIONS FROM NICE_FUNCS.PY ==============
    
    async def get_token_security_info(self, token_mint_address: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive token security information including freeze authority,
        creator balance, top holders, and other security metrics.
        
        Args:
            token_mint_address (str): Token mint address
            
        Returns:
            Dict containing security information or None if failed
        """
        if not self.birdeye_api_key:
            logger.error("Birdeye API key not configured")
            return None
            
        url = f"{BIRDEYE_BASE_URL}/token_security?address={token_mint_address}"
        headers = {"X-API-KEY": self.birdeye_api_key}
        
        try:
            async with asyncio.timeout(10):
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                if data.get('success') and data.get('data'):
                    security_data = data['data']
                    logger.info(f"📊 Token security info retrieved for {token_mint_address[-4:]}")
                    return security_data
                else:
                    logger.warning(f"No security data returned for {token_mint_address[-4:]}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching token security info for {token_mint_address[-4:]}: {e}")
            return None
    
    async def get_token_creation_info(self, token_mint_address: str) -> Optional[Dict[str, Any]]:
        """
        Get token creation information including creation slot, timestamp, and creator.
        
        Args:
            token_mint_address (str): Token mint address
            
        Returns:
            Dict containing creation information or None if failed
        """
        if not self.birdeye_api_key:
            logger.error("Birdeye API key not configured")
            return None
            
        url = f"{BIRDEYE_BASE_URL}/token_creation_info?address={token_mint_address}"
        headers = {"X-API-KEY": self.birdeye_api_key}
        
        try:
            async with asyncio.timeout(10):
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                if data.get('success') and data.get('data'):
                    creation_data = data['data']
                    logger.info(f"🏗️ Token creation info retrieved for {token_mint_address[-4:]}")
                    return creation_data
                else:
                    logger.warning(f"No creation data returned for {token_mint_address[-4:]}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching token creation info for {token_mint_address[-4:]}: {e}")
            return None
    
    async def elegant_entry(self, token_mint_address: str, buy_under_price: float, 
                           target_usd_size: float = None, max_chunk_usd: float = None,
                           orders_per_burst: int = 3, slippage_bps: int = 50,
                           tx_delay_seconds: int = 15) -> float:
        """
        Enter a position gradually in small chunks until target size is reached.
        Only buys when price is at or below the specified threshold.
        
        Args:
            token_mint_address (str): Token to buy
            buy_under_price (float): Maximum price to buy at
            target_usd_size (float): Target position size in USD
            max_chunk_usd (float): Maximum USD per chunk
            orders_per_burst (int): Number of orders per burst
            slippage_bps (int): Slippage tolerance in basis points
            tx_delay_seconds (int): Delay between transaction bursts
            
        Returns:
            float: Final position value in USD
        """
        # Use config defaults if not provided
        target_usd_size = target_usd_size or self.config.get('usd_size', 100)
        max_chunk_usd = max_chunk_usd or self.config.get('max_usd_order_size', 25)
        
        logger.info(f"🎯 Starting elegant entry for {token_mint_address[-4:]} - "
                   f"Target: ${target_usd_size}, Max chunk: ${max_chunk_usd}, "
                   f"Buy under: ${buy_under_price}")
        
        # Get initial position and price
        current_position = await self.get_position(token_mint_address)
        current_price = await self.get_token_price(token_mint_address)
        
        if current_price is None:
            logger.error(f"Could not get price for {token_mint_address[-4:]}")
            return 0.0
            
        position_value = current_position * current_price
        
        logger.info(f"📊 Initial state - Position: {current_position:.2f}, "
                   f"Price: ${current_price:.8f}, Value: ${position_value:.2f}")
        
        # Check if position already filled
        if position_value >= (0.97 * target_usd_size):
            logger.info(f"✅ Position already filled (${position_value:.2f}/${target_usd_size})")
            return position_value
        
        # Main buying loop
        while position_value < (0.97 * target_usd_size) and current_price <= buy_under_price:
            size_needed = target_usd_size - position_value
            chunk_size = min(size_needed, max_chunk_usd)
            chunk_lamports = str(int(chunk_size * 10**6))  # USDC has 6 decimals
            
            logger.info(f"💰 Buying chunk: ${chunk_size:.2f} ({chunk_lamports} lamports) "
                       f"at price ${current_price:.8f}")
            
            try:
                # Execute multiple orders in burst
                for i in range(orders_per_burst):
                    tx_url = await self.market_buy(token_mint_address, chunk_lamports, slippage_bps)
                    if tx_url:
                        logger.info(f"✅ Buy order {i+1}/{orders_per_burst} submitted")
                        await asyncio.sleep(1)  # Small delay between orders
                    else:
                        logger.warning(f"❌ Buy order {i+1}/{orders_per_burst} failed")
                
                # Wait for transactions to process
                await asyncio.sleep(tx_delay_seconds)
                
            except Exception as e:
                logger.error(f"❌ Buy error: {e}. Retrying in 30 seconds...")
                await asyncio.sleep(30)
                
                try:
                    # Retry the burst
                    for i in range(orders_per_burst):
                        tx_url = await self.market_buy(token_mint_address, chunk_lamports, slippage_bps)
                        if tx_url:
                            logger.info(f"🔄 Retry buy order {i+1}/{orders_per_burst} submitted")
                            await asyncio.sleep(1)
                        else:
                            logger.warning(f"❌ Retry buy order {i+1}/{orders_per_burst} failed")
                    
                    await asyncio.sleep(tx_delay_seconds)
                    
                except Exception as retry_error:
                    logger.error(f"💥 Final error in buy process: {retry_error}. Exiting.")
                    break
            
            # Update position status
            current_position = await self.get_position(token_mint_address)
            current_price = await self.get_token_price(token_mint_address)
            
            if current_price is None:
                logger.error("Could not get updated price. Exiting elegant entry.")
                break
                
            position_value = current_position * current_price
            
            logger.info(f"📈 Updated state - Position: {current_position:.2f}, "
                       f"Price: ${current_price:.8f}, Value: ${position_value:.2f}")
        
        # Report final status
        if position_value >= (0.97 * target_usd_size):
            logger.info(f"🎉 Elegant entry complete - Position filled: ${position_value:.2f}")
        elif current_price > buy_under_price:
            logger.info(f"🛑 Elegant entry stopped - Price ${current_price:.8f} > ${buy_under_price}")
        else:
            logger.info(f"⏹️ Elegant entry exited - Final position: ${position_value:.2f}")
        
        return position_value
    
    async def pnl_close(self, token_mint_address: str, initial_investment_usd: float = None) -> bool:
        """
        Check if a position should be closed based on profit/loss thresholds.
        
        Args:
            token_mint_address (str): Token to evaluate
            initial_investment_usd (float): Initial investment amount
            
        Returns:
            bool: True if position was closed, False otherwise
        """
        initial_investment_usd = initial_investment_usd or self.config.get('usd_size', 100)
        
        logger.info(f"🔍 Checking PnL exit conditions for {token_mint_address[-4:]}")
        
        # Get current position details
        current_position = await self.get_position(token_mint_address)
        if current_position <= 0:
            logger.info(f"No position to evaluate for {token_mint_address[-4:]}")
            return False
        
        current_price = await self.get_token_price(token_mint_address)
        if current_price is None:
            logger.error(f"Could not get price for {token_mint_address[-4:]}")
            return False
        
        position_value = current_position * current_price
        
        # Calculate thresholds
        take_profit_threshold = self.sell_at_multiple * initial_investment_usd
        stop_loss_threshold = (1 + self.stop_loss_percentage) * initial_investment_usd
        
        logger.info(f"📊 Position: {current_position:.4f}, Price: ${current_price:.8f}, "
                   f"Value: ${position_value:.2f}")
        logger.info(f"🎯 Take Profit: ${take_profit_threshold:.2f}, "
                   f"Stop Loss: ${stop_loss_threshold:.2f}")
        
        # Take Profit Logic
        if position_value > take_profit_threshold:
            logger.info(f"🎉 Take Profit triggered! (${position_value:.2f} > ${take_profit_threshold:.2f})")
            
            try:
                success = await self.chunk_sell_position(token_mint_address, position_value, 500)
                if success:
                    logger.info(f"✅ Take profit executed for {token_mint_address[-4:]}")
                    return True
                else:
                    logger.error(f"❌ Take profit failed for {token_mint_address[-4:]}")
                    
            except Exception as e:
                logger.error(f"❌ Take profit error: {e}")
        
        # Stop Loss Logic
        elif position_value < stop_loss_threshold and position_value > 0.05:
            logger.info(f"🛑 Stop Loss triggered! (${position_value:.2f} < ${stop_loss_threshold:.2f})")
            
            try:
                success = await self.emergency_close_position(token_mint_address, 1000)
                if success:
                    logger.info(f"✅ Stop loss executed for {token_mint_address[-4:]}")
                    
                    # Add to do-not-trade list
                    try:
                        with open('dont_overtrade.txt', 'a') as file:
                            file.write(f"{token_mint_address}\n")
                        logger.info(f"📝 Added {token_mint_address[-4:]} to do-not-trade list")
                    except Exception as e:
                        logger.error(f"Error writing to dont_overtrade.txt: {e}")
                    
                    return True
                else:
                    logger.error(f"❌ Stop loss failed for {token_mint_address[-4:]}")
                    
            except Exception as e:
                logger.error(f"❌ Stop loss error: {e}")
        
        return False
    
    async def chunk_kill_mm(self, token_mint_address: str, max_usd_sell_size: float,
                           slippage_bps: int, sell_over_price: float, 
                           seconds_to_sleep: int = 15) -> bool:
        """
        Market maker position closing - sell gradually when price is above threshold.
        
        Args:
            token_mint_address (str): Token to sell
            max_usd_sell_size (float): Maximum USD value per chunk
            slippage_bps (int): Slippage tolerance in basis points
            sell_over_price (float): Price threshold above which to sell
            seconds_to_sleep (int): Delay between chunks
            
        Returns:
            bool: True if completely closed, False otherwise
        """
        logger.info(f"🤖 Market maker checking {token_mint_address[-4:]} for selling opportunity")
        
        # Get current position
        current_position = await self.get_position(token_mint_address)
        if current_position <= 0:
            logger.info(f"No position to sell for {token_mint_address[-4:]}")
            return True
        
        current_price = await self.get_token_price(token_mint_address)
        if current_price is None:
            logger.error(f"Could not get price for {token_mint_address[-4:]}")
            return False
        
        position_value = current_position * current_price
        
        # Check if price is below threshold
        if current_price <= sell_over_price:
            logger.info(f"💰 Price ${current_price:.8f} <= threshold ${sell_over_price:.8f}. Not selling.")
            return False
        
        logger.info(f"🎯 Market maker selling {token_mint_address[-4:]} - "
                   f"Price: ${current_price:.8f} > ${sell_over_price:.8f}")
        
        # Sell in chunks while price is above threshold
        while position_value > 0 and current_price > sell_over_price:
            # Calculate chunk size
            if position_value < max_usd_sell_size:
                sell_tokens = current_position
            else:
                sell_tokens = max_usd_sell_size / current_price
            
            # Round down to prevent insufficient funds errors
            sell_tokens = self.round_down(sell_tokens, 2)
            
            # Get token decimals and convert to lamports
            decimals = await self.get_token_decimals(token_mint_address) or 9
            sell_lamports = str(int(sell_tokens * 10**decimals))
            
            logger.info(f"💸 Selling {sell_tokens:.2f} tokens (~${sell_tokens * current_price:.2f})")
            
            try:
                # Execute multiple sell orders to ensure execution
                for i in range(3):
                    tx_url = await self.market_sell(token_mint_address, sell_lamports, slippage_bps)
                    if tx_url:
                        logger.info(f"✅ Market maker sell order {i+1}/3 submitted")
                        await asyncio.sleep(1)
                    else:
                        logger.warning(f"❌ Market maker sell order {i+1}/3 failed")
                
                await asyncio.sleep(seconds_to_sleep)
                
            except Exception as e:
                logger.error(f"❌ Market maker sell error: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
            
            # Update position status
            current_position = await self.get_position(token_mint_address)
            current_price = await self.get_token_price(token_mint_address)
            
            if current_price is None:
                logger.error("Could not get updated price. Exiting market maker.")
                break
                
            position_value = current_position * current_price
        
        # Final status
        if position_value <= 0:
            logger.info(f"✅ Market maker: Position fully closed for {token_mint_address[-4:]}")
            return True
        else:
            logger.info(f"⏸️ Market maker: Price dropped below threshold. "
                       f"Remaining: {current_position:.4f} tokens (${position_value:.2f})")
            return False
    
    async def close_all_positions(self, exclude_tokens: List[str] = None) -> int:
        """
        Close all open positions except those in the exclusion list.
        
        Args:
            exclude_tokens (List[str]): List of token addresses to exclude from closing
            
        Returns:
            int: Number of positions successfully closed
        """
        exclude_tokens = exclude_tokens or self.do_not_trade_list
        
        logger.info("🧹 Closing all open positions...")
        
        # Get all positions
        holdings_df = await self.get_wallet_holdings(self.wallet_address)
        if holdings_df.empty:
            logger.info("No open positions found")
            return 0
        
        # Filter out excluded positions
        positions_to_close = holdings_df[~holdings_df['Mint Address'].isin(exclude_tokens)]
        
        if positions_to_close.empty:
            logger.info("No positions to close (all excluded)")
            return 0
        
        logger.info(f"📋 Found {len(positions_to_close)} positions to close")
        closed_count = 0
        
        # Close each position
        for _, row in positions_to_close.iterrows():
            token_mint = row['Mint Address']
            token_value = row['USD Value']
            
            if token_value < 1:
                logger.info(f"⏭️ Skipping {token_mint[-4:]} - value too small (${token_value:.2f})")
                continue
            
            logger.info(f"🎯 Closing position for {token_mint[-4:]} (${token_value:.2f})")
            
            try:
                max_chunk = self.config.get('max_usd_order_size', 25)
                success = await self.chunk_sell_position(token_mint, max_chunk, 500)
                if success:
                    closed_count += 1
                    logger.info(f"✅ Successfully closed {token_mint[-4:]}")
                else:
                    logger.error(f"❌ Failed to close {token_mint[-4:]}")
                    
            except Exception as e:
                logger.error(f"❌ Error closing {token_mint[-4:]}: {e}")
        
        logger.info(f"🏁 Successfully closed {closed_count}/{len(positions_to_close)} positions")
        return closed_count
    
    async def elegant_time_entry(self, token_mint_address: str, buy_under_price: float,
                                seconds_to_sleep: int, target_usd_size: float = None,
                                max_chunk_usd: float = None, orders_per_burst: int = 3,
                                slippage_bps: int = 50) -> float:
        """
        Similar to elegant_entry but with custom sleep timing between bursts.
        
        Args:
            token_mint_address (str): Token to buy
            buy_under_price (float): Maximum price to buy at
            seconds_to_sleep (int): Custom delay between transaction bursts
            target_usd_size (float): Target position size in USD
            max_chunk_usd (float): Maximum USD per chunk
            orders_per_burst (int): Number of orders per burst
            slippage_bps (int): Slippage tolerance in basis points
            
        Returns:
            float: Final position value in USD
        """
        logger.info(f"⏰ Starting elegant time entry for {token_mint_address[-4:]} "
                   f"with {seconds_to_sleep}s delays")
        
        return await self.elegant_entry(
            token_mint_address=token_mint_address,
            buy_under_price=buy_under_price,
            target_usd_size=target_usd_size,
            max_chunk_usd=max_chunk_usd,
            orders_per_burst=orders_per_burst,
            slippage_bps=slippage_bps,
            tx_delay_seconds=seconds_to_sleep
        )

# Convenience functions for API integration
async def get_token_overview(token_mint: str, service: SolanaTradingUtilsService = None) -> Optional[TokenOverview]:
    """Convenience function for token overview"""
    if not service:
        service = SolanaTradingUtilsService()
    return await service.get_token_overview(token_mint)

async def get_token_price(token_mint: str, service: SolanaTradingUtilsService = None) -> Optional[float]:
    """Convenience function for token price"""
    if not service:
        service = SolanaTradingUtilsService()
    return await service.get_token_price(token_mint)

async def perform_market_buy(token_mint: str, usdc_amount: str, slippage_bps: int = 50, service: SolanaTradingUtilsService = None) -> Optional[str]:
    """Convenience function for market buy"""
    if not service:
        service = SolanaTradingUtilsService()
    return await service.market_buy(token_mint, usdc_amount, slippage_bps)

async def perform_market_sell(token_mint: str, token_amount: str, slippage_bps: int = 50, service: SolanaTradingUtilsService = None) -> Optional[str]:
    """Convenience function for market sell"""
    if not service:
        service = SolanaTradingUtilsService()
    return await service.market_sell(token_mint, token_amount, slippage_bps) 