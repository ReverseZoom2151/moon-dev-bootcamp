import ccxt, time, datetime, schedule, os, logging, argparse
import pandas as pd
import pandas_ta as ta

# Setup logging
logging.basicConfig(filename='binance_algo_orders.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argparse for configurable parameters
parser = argparse.ArgumentParser(description='Binance Algo Orders')
parser.add_argument('--symbol', default='BTCUSD_PERP', help='Trading symbol')
parser.add_argument('--size', type=int, default=1, help='Order size')
parser.add_argument('--bid', type=float, default=30000, help='Fixed bid price')
parser.add_argument('--max-iterations', type=int, default=10, help='Max iterations for algo loop (0 for infinite)')
parser.add_argument('--dynamic', action='store_true', help='Use dynamic pricing')
parser.add_argument('--dry-run', action='store_true', help='Simulate orders')
parser.add_argument('--testnet', action='store_true')
parser.add_argument('--strategy', default='none', choices=['none', 'rsi'])
parser.add_argument('--max-orders', type=int, default=50, help='Max total orders')
args = parser.parse_args()
if args.max_iterations == 0: args.max_iterations = None

# Use environment variables for API keys
api_key = os.environ.get('BINANCE_API_KEY')
api_secret = os.environ.get('BINANCE_API_SECRET')
if not api_key or not api_secret:
    logging.error('Missing API keys')
    raise ValueError('API keys not set')

# Initialize Binance
logging.info('Initializing Binance...')
binance = ccxt.binance({
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'defaultType': 'future'}
})

# Load markets and validate symbol
binance.load_markets()
if args.symbol not in binance.markets:
    logging.error(f'Invalid symbol: {args.symbol}')
    raise ValueError('Invalid symbol')
symbol = args.symbol
size = args.size
bid = args.bid
params = {'timeInForce': 'GTX'}

# In exchange init, add testnet
if args.testnet:
    binance.set_sandbox_mode(True)

# Add global order_count and pnl
order_count = 0
pnl = 0.0

# Fetch initial data (balance, market info) with logging
try:
    balance = binance.fetch_balance()
    logging.info(f'Balances: BTC={balance.get("BTC", {}).get("free", 0)}, USDT={balance.get("USDT", {}).get("free", 0)}')
    market = binance.market(symbol)
    logging.info(f'Market info loaded for {symbol}')
except Exception as e:
    logging.error(f'Error fetching initial data: {e}')

# Using inverse perpetual symbol for BTC on Binance
print(f"\n--- ORDER DETAILS ---")
print(f"Symbol: {symbol} (BTC-margined perpetual contract)")

# Get market information for this symbol
try:
    # market = binance.market(symbol) # This line is now redundant as market is fetched above
    print(f"Market ID: {market['id']}")
    min_amount = market.get('limits', {}).get('amount', {}).get('min', 'Unknown')
    price_precision = market.get('precision', {}).get('price', 0.1)  # Adjust based on Binance precision
    print(f"Price precision: {price_precision}")
    print(f"Minimum contract amount: {min_amount}")
except Exception as e:
    print(f"Error fetching market information: {e}")

# Get current market price
try:
    ticker = binance.fetch_ticker(symbol)
    current_price = ticker['last']
    print(f"Current market price: ${current_price}")
except Exception as e:
    print(f"Error fetching current price: {e}")
    current_price = None

# Order parameters
# size = 1  # Number of contracts (adjust based on min_amount) # This line is now redundant as size is from args
# bid = 30000  # Fixed price in USD # This line is now redundant as bid is from args
# params = {'timeInForce': 'GTX'}  # PostOnly equivalent in Binance # This line is now redundant as params is from args

class BinanceAlgoBot():
    def __init__(self, exchange, symbol, size, bid, params, max_iterations, dynamic, dry_run, testnet, strategy, max_orders):
        super().__init__(exchange, symbol, size, bid, params, max_iterations, dynamic, dry_run, testnet, strategy, max_orders)
        self.order_count = 0
        self.pnl = 0.0
        self.market = self.exchange.market(self.symbol)
        self.min_amount = self.market.get('limits', {}).get('amount', {}).get('min', 'Unknown')
        self.price_precision = self.market.get('precision', {}).get('price', 0.1)

    def get_market_info(self):
        print(f"Market ID: {self.market['id']}")
        print(f"Minimum contract amount: {self.min_amount}")
        print(f"Price precision: {self.price_precision}")

    def get_current_price(self):
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            self.current_price = ticker['last']
            print(f"Current market price: ${self.current_price}")
        except Exception as e:
            print(f"Error fetching current price: {e}")
            self.current_price = None

    def get_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            btc_balance = balance.get('BTC', {}).get('free', 0)
            print(f"Current BTC balance: {btc_balance}")
            return btc_balance
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0

    def place_order(self):
        global order_count, pnl
        if self.order_count >= self.max_orders:
            logging.info('Max orders reached')
            return False
        print(f"\n===== BOT EXECUTION at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
        
        # Get current market price
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            self.current_price = ticker['last']
            print(f"Current market price: ${self.current_price}")
            
            print(f"Using bid price: ${self.bid}")
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch current price: {e}")
        
        # Check balance
        try:
            balance = self.exchange.fetch_balance()
            btc_balance = balance.get('BTC', {}).get('free', 0)
            print(f"Current BTC balance: {btc_balance}")
            
            if float(btc_balance) <= 0:
                print("⚠️ Warning: Insufficient BTC balance, order may fail")
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch balance: {e}")
        
        # Place order
        try:
            if self.dry_run:
                logging.info(f'DRY RUN: Would place buy order: {self.size} @ {self.bid}')
            else:
                print(f"Placing limit buy order: {self.size} contracts at ${self.bid}...")
                order = self.exchange.create_limit_buy_order(self.symbol, self.size, self.bid, self.params)
                print(f"✅ Order placed successfully!")
                print(f"Order ID: {order.get('id', 'Unknown')}")
                print(f"Status: {order.get('status', 'Unknown')}")
                
                # Wait before cancelling
                print(f"Waiting 5 seconds before cancellation...")
                time.sleep(5)
                
                # Cancel order
                print(f"Cancelling orders for {self.symbol}...")
                result = self.exchange.cancel_all_orders(self.symbol)
                print(f"✅ Orders cancelled successfully!")
                print(f"Cancel result: {result}")
                
            self.order_count += 1
            return True  # Successful execution
        except ccxt.InsufficientFunds as e:
            print(f"❌ Error: Insufficient funds for order: {e}")
            return False
        except Exception as e:
            print(f"❌ Error during bot execution: {e}")
            return False

    def manual_loop(self):
        print("\n===== STARTING MANUAL TRADING LOOP =====")
        print("This will place and cancel orders in a continuous loop")
        print("Press Ctrl+C to stop execution")

        loop_count = 0
        manual_loop_active = True

        try:
            while manual_loop_active:
                loop_count += 1
                print(f"\n----- MANUAL LOOP ITERATION {loop_count} -----")
                
                # Get current market price for reference
                try:
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    current_price = ticker['last']
                    print(f"Current market price: ${current_price}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not fetch current price: {e}")
                
                try:
                    # Place order
                    if self.dry_run:
                        logging.info(f'DRY RUN: Would place buy order: {self.size} @ {self.bid}')
                    else:
                        print(f"Placing limit buy order: {self.size} contracts at ${self.bid}...")
                    order = self.exchange.create_limit_buy_order(self.symbol, self.size, self.bid, self.params)
                    print(f"✅ Order placed successfully at {datetime.datetime.now().strftime('%H:%M:%S')}")
                    print(f"Order ID: {order.get('id', 'Unknown')}")
                    
                    # Wait before cancelling
                    print(f"Waiting 10 seconds before cancellation...")
                    for i in range(10, 0, -1):
                        print(f"Cancelling in {i} seconds...", end="\r")
                        time.sleep(1)
                    print("\nCancelling order now...")
                    
                    # Cancel order
                    result = self.exchange.cancel_all_orders(self.symbol)
                    print(f"✅ Orders cancelled at {datetime.datetime.now().strftime('%H:%M:%S')}")
                    print(f"Cancel result: {result}")
                    
                except ccxt.InsufficientFunds as e:
                    print(f"❌ Error: Insufficient funds for order: {e}")
                    print("Pausing for 30 seconds...")
                    time.sleep(30)
                except Exception as e:
                    print(f"❌ Error during manual loop: {e}")
                    print("Pausing for 10 seconds...")
                    time.sleep(10)
                
                # Brief pause between iterations
                print("Preparing for next iteration...")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\nManual trading loop stopped by user (Ctrl+C)")
            manual_loop_active = False
        except Exception as e:
            print(f"\n\nManual trading loop stopped due to error: {e}")
            manual_loop_active = False

    def scheduled_bot(self):
        print("\n===== STARTING SCHEDULED TRADING BOT =====")
        print(f"Bot is scheduled to run every 28 seconds")
        print("Press Ctrl+C to stop the bot")

        # Clear any existing jobs
        schedule.clear()

        # Schedule the bot to run every 28 seconds
        schedule.every(28).seconds.do(self.place_order)
        print(f"Bot scheduled at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Waiting for scheduled executions...")

        # Counter for successful and failed executions
        successful_runs = 0
        failed_runs = 0
        start_time = datetime.datetime.now()

        # Run the scheduled bot
        try:
            while True:
                try:
                    # Run any pending scheduled tasks
                    job_result = schedule.run_pending()
                    
                    # If a job ran and returned a result, update counters
                    if job_result is not None:
                        if job_result:
                            successful_runs += 1
                        else:
                            failed_runs += 1
                        
                        # Print status update every 10 executions
                        total_runs = successful_runs + failed_runs
                        if total_runs % 10 == 0:
                            print(f"\n----- BOT STATUS UPDATE -----")
                            print(f"Total executions: {total_runs}")
                            print(f"Successful: {successful_runs}, Failed: {failed_runs}")
                            
                            # Calculate run time
                            current_time = datetime.datetime.now()
                            duration = current_time - start_time
                            print(f"Running for: {duration}")
                            
                            # Get current balance
                            try:
                                balance = self.exchange.fetch_balance()
                                btc_balance = balance.get('BTC', {}).get('free', 0)
                                print(f"Current BTC balance: {btc_balance}")
                            except Exception as e:
                                print(f"Could not fetch balance: {e}")
                        
                    # Brief sleep to prevent high CPU usage
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Scheduler error: {e}")
                    print("Waiting 30 seconds before resuming...")
                    time.sleep(30)
                    
        except KeyboardInterrupt:
            print("\n\nScheduled bot stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\n\nScheduled bot stopped due to unexpected error: {e}")
        finally:
            # Print final summary
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            print("\n===== BOT SUMMARY =====")
            print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total run time: {duration}")
            print(f"Total executions: {successful_runs + failed_runs}")
            print(f"Successful executions: {successful_runs}")
            print(f"Failed executions: {failed_runs}")
            
            # Final balance check
            try:
                balance = self.exchange.fetch_balance()
                btc_balance = balance.get('BTC', {}).get('free', 0)
                print(f"Final BTC balance: {btc_balance}")
            except Exception as e:
                print(f"Could not fetch final balance: {e}")
            
            print("\nBot execution completed.")

    def algorithmic_trading(self):
        print("\n=== STARTING ALGORITHMIC TRADING LOOP ===")
        print("This algorithm will repeatedly place and cancel orders")
        print("Press Ctrl+C to stop the algorithm at any time")

        # Configuration
        max_iterations = 10  # Set to None for infinite loop
        wait_time = 5  # Seconds to wait between order placement and cancellation
        iteration = 1
        total_orders_placed = 0
        total_orders_cancelled = 0
        errors_encountered = 0

        try:
            go = True
            start_time = datetime.datetime.now()
            print(f"Algorithm started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            while go:
                print(f"\n--- ITERATION {iteration} ---")
                
                # Get current market price for reference
                try:
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    current_price = ticker['last']
                    print(f"Current market price: ${current_price}")
                    
                    # Dynamically adjust bid price to be 1% below current market price
                    if self.dynamic:
                        bid = round(current_price * 0.99, int(self.price_precision))  # Adjust rounding
                        print(f"Dynamic bid price: ${bid} (1% below market)")
                    else:
                        print(f"Using bid price: ${self.bid}")
                except Exception as e:
                    print(f"Warning: Could not fetch current price: {e}")
                
                # Check balance before placing order
                try:
                    balance = self.exchange.fetch_balance()
                    btc_balance = balance.get('BTC', {}).get('free', 0)
                    print(f"Current BTC balance: {btc_balance}")
                except Exception as e:
                    print(f"Warning: Could not fetch balance: {e}")
                
                # Place order
                try:
                    if self.dry_run:
                        logging.info(f'DRY RUN: Would place buy order: {self.size} @ {self.bid}')
                    else:
                        print(f"Placing limit buy order: {self.size} contracts at ${self.bid}...")
                    order = self.exchange.create_limit_buy_order(self.symbol, self.size, self.bid, self.params)
                    print(f"✅ Order {iteration} placed successfully!")
                    print(f"Order ID: {order.get('id', 'Unknown')}")
                    total_orders_placed += 1
                    
                    # Sleep for the configured wait time
                    print(f"Waiting {wait_time} seconds before cancellation...")
                    time.sleep(wait_time)
                    
                    # Cancel the order
                    print(f"Cancelling orders for {self.symbol}...")
                    cancel_result = self.exchange.cancel_all_orders(self.symbol)
                    print(f"✅ Cancellation request completed")
                    print(f"Cancel result: {cancel_result}")
                    total_orders_cancelled += 1
                    
                except ccxt.InsufficientFunds as e:
                    print(f"❌ Insufficient funds error: {e}")
                    errors_encountered += 1
                    print("Waiting 30 seconds before retrying...")
                    time.sleep(30)  # Longer wait on balance error
                    
                except Exception as e:
                    print(f"❌ Error during iteration {iteration}: {e}")
                    errors_encountered += 1
                    print("Waiting 10 seconds before retrying...")
                    time.sleep(10)
                
                # Check if we should continue
                if max_iterations is not None and iteration >= max_iterations:
                    print(f"\nReached maximum iterations ({max_iterations})")
                    go = False
                
                # Increment iteration counter
                iteration += 1
                
                # Small delay between iterations
                if go:
                    print("Preparing for next iteration...")
                    time.sleep(2)
                    
        except KeyboardInterrupt:
            print("\n\nAlgorithm stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\n\nAlgorithm stopped due to unexpected error: {e}")
        finally:
            # Final summary
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            print("\n=== ALGORITHM SUMMARY ===")
            print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {duration}")
            print(f"Total iterations: {iteration - 1}")
            print(f"Orders placed: {total_orders_placed}")
            print(f"Orders cancelled: {total_orders_cancelled}")
            print(f"Errors encountered: {errors_encountered}")
            
            # Final account balance
            try:
                balance = self.exchange.fetch_balance()
                btc_balance = balance.get('BTC', {}).get('free', 0)
                print(f"Final BTC balance: {btc_balance}")
            except Exception as e:
                print(f"Could not fetch final balance: {e}")
            
            print("\nAlgorithm execution completed.")

    def get_ohlcv(self, timeframe='1h', limit=100):
        return self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)

    def run_strategy(self):
        if self.strategy == 'rsi':
            ohlcv = self.get_ohlcv()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['rsi'] = ta.rsi(df['close'], length=14)
            rsi = df['rsi'].iloc[-1]
            # Adjust logic
            pass # Placeholder for sell logic

    def update_pnl(self):
        logging.info(f'Current PnL: {self.pnl}')

    def test_note(self):
        print("Test note at bottom") 

# Create an instance of the bot
bot_instance = BinanceAlgoBot(binance, symbol, size, bid, params, args.max_iterations, args.dynamic, args.dry_run, args.testnet, args.strategy, args.max_orders)

# Run the bot
bot_instance.run() 