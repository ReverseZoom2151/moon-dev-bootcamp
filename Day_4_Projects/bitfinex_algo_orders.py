import ccxt, time, datetime, schedule, os, logging, argparse

# Setup logging
logging.basicConfig(filename='bitfinex_algo_orders.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argparse for configurable parameters
parser = argparse.ArgumentParser(description='Bitfinex Algo Orders')
parser.add_argument('--symbol', default='BTCF0:USTF0', help='Trading symbol')
parser.add_argument('--size', type=float, default=0.0001, help='Order size')
parser.add_argument('--bid', type=float, default=30000, help='Fixed bid price')
parser.add_argument('--max-iterations', type=int, default=10, help='Max iterations for algo loop (0 for infinite)')
parser.add_argument('--dynamic', action='store_true', help='Use dynamic pricing')
parser.add_argument('--dry-run', action='store_true', help='Simulate orders')
parser.add_argument('--testnet', action='store_true')
parser.add_argument('--strategy', default='none', choices=['none', 'rsi'])
parser.add_argument('--max-orders', type=int, default=50)
args = parser.parse_args()
if args.max_iterations == 0: args.max_iterations = None

# Use environment variables for API keys
api_key = os.environ.get('BITFINEX_API_KEY')
api_secret = os.environ.get('BITFINEX_API_SECRET')
if not api_key or not api_secret:
    logging.error('Missing API keys')
    raise ValueError('API keys not set')

# Initialize Bitfinex
logging.info('Initializing Bitfinex...')
bitfinex = ccxt.bitfinex({
    'enableRateLimit': True,
    'apiKey': api_key,
    'secret': api_secret,
})

# Load markets and validate symbol
bitfinex.load_markets()
if args.symbol not in bitfinex.markets:
    logging.error(f'Invalid symbol: {args.symbol}')
    raise ValueError('Invalid symbol')
symbol = args.symbol
size = args.size
bid = args.bid
params = {'timeInForce': 'POC'}

# Fetch initial data with logging
try:
    balance = bitfinex.fetch_balance({'type': 'derivatives'})
    logging.info(f'Balances: BTC={balance.get("BTC", {}).get("free", 0)}, USDT={balance.get("USDT", {}).get("free", 0)}')
    market = bitfinex.market(symbol)
    logging.info(f'Market info loaded for {symbol}')
except Exception as e:
    logging.error(f'Error fetching initial data: {e}')

# Using perpetual symbol for BTC on Bitfinex
print(f"\n--- ORDER DETAILS ---")
print(f"Symbol: {symbol} (BTC perpetual contract)")

# Get market information for this symbol
try:
    market = bitfinex.market(symbol)
    print(f"Market ID: {market['id']}")
    min_amount = market.get('limits', {}).get('amount', {}).get('min', 'Unknown')
    price_precision = market.get('precision', {}).get('price', 0.5)
    print(f"Price precision: {price_precision}")
    print(f"Minimum contract amount: {min_amount}")
except Exception as e:
    print(f"Error fetching market information: {e}")

# Get current market price
try:
    ticker = bitfinex.fetch_ticker(symbol)
    current_price = ticker['last']
    print(f"Current market price: ${current_price}")
except Exception as e:
    print(f"Error fetching current price: {e}")
    current_price = None

# Order parameters
size = 0.0001  # Adjust based on Bitfinex min size, e.g., for BTC
bid = 30000  # Fixed price in USD
params = {'timeInForce': 'POC'}  # PostOnly equivalent if available

# Add order_count, pnl
order_count = 0
pnl = 0.0

# get_ohlcv
def get_ohlcv(symbol, timeframe='1h', limit=100):
    return bitfinex.fetch_ohlcv(symbol, timeframe, limit=limit)

# Test note

def bot():
    """
    Automated trading bot function that places a limit buy order and cancels it after 5 seconds.
    This function is designed to be scheduled to run at regular intervals.
    """
    print(f"\n===== BOT EXECUTION at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    
    # Get current market price
    try:
        ticker = bitfinex.fetch_ticker(symbol)
        current_price = ticker['last']
        print(f"Current market price: ${current_price}")
        
        print(f"Using bid price: ${bid}")
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch current price: {e}")
    
    # Check balance
    try:
        balance = bitfinex.fetch_balance({'type': 'derivatives'})
        btc_balance = balance.get('BTC', {}).get('free', 0)
        print(f"Current BTC balance: {btc_balance}")
        
        if float(btc_balance) <= 0:
            print("⚠️ Warning: Insufficient BTC balance, order may fail")
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch balance: {e}")
    
    # Place order
    try:
        if args.dry_run:
            logging.info(f'DRY RUN: Would place buy order: {size} @ {bid}')
        else:
            print(f"Placing limit buy order: {size} contracts at ${bid}...")
            order = bitfinex.create_limit_buy_order(symbol, size, bid, params)
            print(f"✅ Order placed successfully!")
            print(f"Order ID: {order.get('id', 'Unknown')}")
            print(f"Status: {order.get('status', 'Unknown')}")
            
            # Wait before cancelling
            print(f"Waiting 5 seconds before cancellation...")
            time.sleep(5)
            
            # Cancel order
            print(f"Cancelling orders for {symbol}...")
            result = bitfinex.cancel_all_orders(symbol)
            print(f"✅ Orders cancelled successfully!")
            print(f"Cancel result: {result}")
            
        return True  # Successful execution
    except ccxt.InsufficientFunds as e:
        print(f"❌ Error: Insufficient funds for order: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during bot execution: {e}")
        return False

# Manual trading loop
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
            ticker = bitfinex.fetch_ticker(symbol)
            current_price = ticker['last']
            print(f"Current market price: ${current_price}")
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch current price: {e}")
        
        try:
            # Place order
            print(f"Placing limit buy order: {size} contracts at ${bid}...")
            order = bitfinex.create_limit_buy_order(symbol, size, bid, params)
            print(f"✅ Order placed successfully at {datetime.datetime.now().strftime('%H:%M:%S')}")
            print(f"Order ID: {order.get('id', 'Unknown')}")
            
            # Wait before cancelling
            print(f"Waiting 10 seconds before cancellation...")
            for i in range(10, 0, -1):
                print(f"Cancelling in {i} seconds...", end="\r")
                time.sleep(1)
            print("\nCancelling order now...")
            
            # Cancel order
            result = bitfinex.cancel_all_orders(symbol)
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

# Scheduled bot setup
print("\n===== STARTING SCHEDULED TRADING BOT =====")
print(f"Bot is scheduled to run every 28 seconds")
print("Press Ctrl+C to stop the bot")

# Clear any existing jobs
schedule.clear()

# Schedule the bot to run every 28 seconds
schedule.every(28).seconds.do(bot)
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
                        balance = bitfinex.fetch_balance({'type': 'derivatives'})
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
        balance = bitfinex.fetch_balance({'type': 'derivatives'})
        btc_balance = balance.get('BTC', {}).get('free', 0)
        print(f"Final BTC balance: {btc_balance}")
    except Exception as e:
        print(f"Could not fetch final balance: {e}")
        
    print("\nBot execution completed.") 

# Algorithmic trading loop
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
            ticker = bitfinex.fetch_ticker(symbol)
            current_price = ticker['last']
            print(f"Current market price: ${current_price}")
            
            # Dynamically adjust bid price to be 1% below current market price
            if args.dynamic:
                bid = round(current_price * 0.99, int(market['precision']['price']))  # Adjust rounding
                print(f"Dynamic bid price: ${bid} (1% below market)")
            else:
                print(f"Using bid price: ${bid}")
        except Exception as e:
            print(f"Warning: Could not fetch current price: {e}")
        
        # Check balance before placing order
        try:
            balance = bitfinex.fetch_balance({'type': 'derivatives'})
            btc_balance = balance.get('BTC', {}).get('free', 0)
            print(f"Current BTC balance: {btc_balance}")
        except Exception as e:
            print(f"Warning: Could not fetch balance: {e}")
        
        # Place order
        try:
            if args.dry_run:
                logging.info(f'DRY RUN: Would place buy order: {size} @ {bid}')
            else:
                print(f"Placing limit buy order: {size} contracts at ${bid}...")
                order = bitfinex.create_limit_buy_order(symbol, size, bid, params)
                print(f"✅ Order {iteration} placed successfully!")
                print(f"Order ID: {order.get('id', 'Unknown')}")
                total_orders_placed += 1
                
                # Sleep for the configured wait time
                print(f"Waiting {wait_time} seconds before cancellation...")
                time.sleep(wait_time)
                
                # Cancel the order
                print(f"Cancelling orders for {symbol}...")
                cancel_result = bitfinex.cancel_all_orders(symbol)
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
        balance = bitfinex.fetch_balance({'type': 'derivatives'})
        btc_balance = balance.get('BTC', {}).get('free', 0)
        print(f"Final BTC balance: {btc_balance}")
    except Exception as e:
        print(f"Could not fetch final balance: {e}")
    
    print("\nAlgorithm execution completed.") 