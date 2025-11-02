import asyncio
import json
import requests
from eth_account import Account
from core.config import get_settings
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange
from datetime import datetime

settings = get_settings()

class HyperliquidBotService:
    """Service to place buy/sell orders on Hyperliquid in a loop"""
    def __init__(self):
        if not settings.HYPERLIQUID_PRIVATE_KEY:
            raise ValueError("HYPERLIQUID_PRIVATE_KEY must be set in .env to enable Hyperliquid bot")
        self.account = Account.from_key(settings.HYPERLIQUID_PRIVATE_KEY)
        self.symbol = settings.HYPERLIQUID_SYMBOL if hasattr(settings, 'HYPERLIQUID_SYMBOL') else 'WIF'
        self.loop_interval = settings.HYPERLIQUID_LOOP_INTERVAL

    def ask_bid(self):
        url = 'https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}
        data = {"type": "l2Book", "coin": self.symbol}
        resp = requests.post(url, headers=headers, data=json.dumps(data))
        resp.raise_for_status()
        result = resp.json()
        if 'levels' in result and result['levels']:
            bid = float(result['levels'][0][0]['px'])
            ask = float(result['levels'][0][1]['px'])
            return ask, bid
        return 0.0, 0.0

    def check_wallet(self):
        url = 'https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}
        data = {"type": "userState", "user": self.account.address}
        resp = requests.post(url, headers=headers, data=json.dumps(data))
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, dict) and 'error' in result and 'User does not exist' in result['error']:
            return False
        return True

    async def start(self):
        """Full Day 4 Hyperliquid bot: diagnostics, loop with stats"""
        print(f"Using Hyperliquid API at {constants.MAINNET_API_URL}")
        # Initial market diagnostics
        try:
            meta_url = 'https://api.hyperliquid.xyz/info'
            headers = {'Content-Type': 'application/json'}
            resp = await asyncio.to_thread(requests.post, meta_url, headers=headers, data=json.dumps({"type": "meta"}))
            resp.raise_for_status()
            meta = resp.json()
            if 'universe' in meta:
                coins = [c['name'] for c in meta['universe']]
                print(f"Available coins: {coins}")
                if self.symbol not in coins:
                    print(f"WARNING: {self.symbol} not found in available markets!")
        except Exception as e:
            print(f"Error fetching Hyperliquid markets: {e}")
        # Trading loop with iteration stats
        successful_runs = 0
        failed_runs = 0
        start_time = datetime.now()
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- HYPERLIQUID ITERATION {iteration} ---")
            try:
                ask, bid = await asyncio.to_thread(self.ask_bid)
                if ask == 0 or bid == 0:
                    raise Exception("Invalid prices")
                if not await asyncio.to_thread(self.check_wallet):
                    raise Exception("Wallet not registered")
                # Place BUY order
                print(f"Placing BUY order for {self.symbol} @ {bid}")
                exchange = Exchange(self.account, constants.MAINNET_API_URL)
                buy_res = await asyncio.to_thread(
                    exchange.order,
                    self.symbol,
                    True,
                    settings.HYPERLIQUID_ORDER_SIZE,
                    bid,
                    {"limit": {"tif": "Gtc"}},
                    False
                )
                print(f"BUY result: {buy_res}")
                await asyncio.sleep(self.loop_interval)
                # Place SELL order
                print(f"Placing SELL order for {self.symbol} @ {ask}")
                sell_res = await asyncio.to_thread(
                    exchange.order,
                    self.symbol,
                    False,
                    settings.HYPERLIQUID_ORDER_SIZE,
                    ask,
                    {"limit": {"tif": "Gtc"}},
                    True
                )
                print(f"SELL result: {sell_res}")
                successful_runs += 1
            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
                failed_runs += 1
            total = successful_runs + failed_runs
            if total % 10 == 0:
                duration = datetime.now() - start_time
                print("\n----- HYPERLIQUID BOT STATUS -----")
                print(f"Total executions: {total}")
                print(f"Successful: {successful_runs}, Failed: {failed_runs}")
                print(f"Running for: {duration}")
            await asyncio.sleep(self.loop_interval) 