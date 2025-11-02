import asyncio
import ccxt
from core.config import get_settings
from datetime import datetime

class PhemexService:
    """Service to place and cancel limit orders on Phemex in a loop"""
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        if not self.settings.PHEMEX_API_KEY or not self.settings.PHEMEX_SECRET_KEY:
            raise ValueError("PHEMEX_API_KEY and PHEMEX_SECRET_KEY must be set in .env to enable Phemex bot")
        # Initialize Phemex exchange via ccxt
        self.exchange = ccxt.phemex({
            'enableRateLimit': True,
            'apiKey': self.settings.PHEMEX_API_KEY,
            'secret': self.settings.PHEMEX_SECRET_KEY
        })
        self.symbol = self.settings.PHEMEX_SYMBOL
        self.size = self.settings.PHEMEX_ORDER_SIZE
        self.bid = self.settings.PHEMEX_BID_PRICE
        self.params = {'timeInForce': 'PostOnly'}
        self.interval = self.settings.PHEMEX_LOOP_INTERVAL

    async def start(self):
        """Full Day 4 Phemex bot: diagnostics, loop with stats"""
        # Initial diagnostics
        print("Initializing connection to Phemex exchange...")
        await asyncio.to_thread(self.exchange.load_markets)
        print(f"Successfully loaded {len(self.exchange.markets)} markets")
        # Fetch and display balances
        try:
            balance = await asyncio.to_thread(self.exchange.fetch_balance)
            btc_balance = balance.get('BTC', {}).get('free', 0)
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            print(f"Available BTC balance: {btc_balance} BTC")
            print(f"Available USDT balance: {usdt_balance} USDT")
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, self.symbol)
            price = ticker.get('last')
            btc_value_usd = float(btc_balance) * price
            print(f"Estimated USD value of BTC: ${btc_value_usd:.2f}")
            if btc_balance <= 0:
                print("⚠️ WARNING: You have no BTC balance. This order will likely fail.")
        except Exception as e:
            print(f"Error fetching balance: {e}")
        # Display market info
        print("\n--- ORDER DETAILS ---")
        print(f"Symbol: {self.symbol}")
        try:
            market = await asyncio.to_thread(self.exchange.market, self.symbol)
            print(f"Market ID: {market.get('id')}")
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 'Unknown')
            price_precision = market.get('precision', {}).get('price', 0)
            print(f"Price precision: {price_precision}")
            print(f"Minimum contract amount: {min_amount}")
        except Exception as e:
            print(f"Error fetching market information: {e}")
        # Display current market price
        try:
            ticker = await asyncio.to_thread(self.exchange.fetch_ticker, self.symbol)
            current_price = ticker.get('last')
            print(f"Current market price: ${current_price}")
        except Exception as e:
            print(f"Error fetching current price: {e}")
        # Begin trading loop with iteration stats
        successful_runs = 0
        failed_runs = 0
        start_time = datetime.now()
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- ITERATION {iteration} ---")
            try:
                order = await asyncio.to_thread(
                    self.exchange.create_limit_buy_order,
                    self.symbol, self.size, self.bid, self.params
                )
                print(f"✅ Order placed successfully: {order.get('id')}")
                # Wait before cancellation
                await asyncio.sleep(self.interval)
                cancel_result = await asyncio.to_thread(
                    self.exchange.cancel_all_orders, self.symbol
                )
                print(f"✅ Orders cancelled: {cancel_result}")
                successful_runs += 1
            except Exception as e:
                print(f"❌ Error during iteration {iteration}: {e}")
                failed_runs += 1
            total = successful_runs + failed_runs
            if total % 10 == 0:
                duration = datetime.now() - start_time
                try:
                    balance = await asyncio.to_thread(self.exchange.fetch_balance)
                    btc_balance = balance.get('BTC', {}).get('free', 0)
                    print("\n----- BOT STATUS UPDATE -----")
                    print(f"Total executions: {total}")
                    print(f"Successful: {successful_runs}, Failed: {failed_runs}")
                    print(f"Running for: {duration}")
                    print(f"Current BTC balance: {btc_balance}")
                except:
                    pass
            # Wait before next cycle
            await asyncio.sleep(self.interval) 