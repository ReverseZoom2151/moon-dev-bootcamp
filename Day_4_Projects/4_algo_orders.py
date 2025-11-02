import ccxt, time, datetime, schedule, key_file as k

# Initialize Phemex exchange
print("Initializing connection to Phemex exchange...")
phemex = ccxt.phemex({
    'enableRateLimit': True,
    'apiKey': k.key,
    'secret': k.secret
})

# Load markets to get available symbols
print("Loading markets data from Phemex...")
phemex.load_markets()
print(f"Successfully loaded {len(phemex.markets)} markets")

# Fetch account balance
print("\n--- ACCOUNT INFORMATION ---")
try:
    balance = phemex.fetch_balance()
    btc_balance = balance.get('BTC', {}).get('free', 0)
    usdt_balance = balance.get('USDT', {}).get('free', 0)
    print(f"Available BTC balance: {btc_balance} BTC")
    print(f"Available USDT balance: {usdt_balance} USDT")
    
    # Convert to USD value for reference
    btc_ticker = phemex.fetch_ticker('BTC/USDT')
    btc_price = btc_ticker['last']
    btc_value_usd = float(btc_balance) * btc_price
    print(f"Estimated USD value of BTC: ${btc_value_usd:.2f}")
    
    if float(btc_balance) <= 0:
        print("\n⚠️ WARNING: You have no BTC balance. This order will likely fail.")
        print("   Please deposit BTC to your Phemex account before proceeding.")
except Exception as e:
    print(f"Error fetching balance: {e}")

# Using the correct symbol format for BTC/USD on Phemex
symbol = 'BTC/USD:BTC'  # BTC-denominated BTC contract (inverse perpetual)
print(f"\n--- ORDER DETAILS ---")
print(f"Symbol: {symbol} (BTC-denominated inverse perpetual contract)")

# Get market information for this symbol
try:
    market = phemex.market(symbol)
    print(f"Market ID: {market['id']}")
    min_amount = market.get('limits', {}).get('amount', {}).get('min', 'Unknown')
    price_precision = market.get('precision', {}).get('price', 0.5)
    print(f"Price precision: {price_precision}")
    print(f"Minimum contract amount: {min_amount}")
except Exception as e:
    print(f"Error fetching market information: {e}")

# Get current market price
try:
    ticker = phemex.fetch_ticker(symbol)
    current_price = ticker['last']
    print(f"Current market price: ${current_price}")
except Exception as e:
    print(f"Error fetching current price: {e}")
    current_price = None

# Order parameters
size = 1  # Number of contracts
bid = 30000  # Fixed price in USD
params = {'timeInForce': 'PostOnly'}  # Ensures the order only adds liquidity

# print(f"Order size: {size} contracts")
# print(f"Bid price: ${bid}")
# if current_price:
#     price_diff = ((bid - current_price) / current_price) * 100
#     print(f"This is {abs(price_diff):.2f}% {'below' if price_diff < 0 else 'above'} the current market price")
# print(f"Order type: Limit Buy with PostOnly flag")

# # Create a limit buy order
# print("\n--- PLACING ORDER ---")
# try:
#     print(f"Sending order to Phemex at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
#     order = phemex.create_limit_buy_order(symbol, size, bid, params)
#     print(f"✅ Order placed successfully!")
#     print(f"Order ID: {order.get('id')}")
#     print(f"Status: {order.get('status')}")
    
#     # Store order ID for later reference
#     order_id = order.get('id')
    
#     # Sleep for 10 seconds
#     print(f"\nWaiting for 10 seconds before cancelling the order...")
#     for i in range(10, 0, -1):
#         print(f"Cancelling in {i} seconds...", end="\r")
#         time.sleep(1)
#     print("\nTime's up! Proceeding to cancel the order.")
    
#     # Check order status before cancelling
#     try:
#         updated_order = phemex.fetch_order(order_id, symbol)
#         print(f"\nCurrent order status: {updated_order.get('status')}")
#         if updated_order.get('status') == 'filled':
#             print("⚠️ Order has been filled! No need to cancel.")
#         elif updated_order.get('status') == 'canceled':
#             print("⚠️ Order has already been cancelled.")
#         else:
#             # Cancel the order
#             print("\n--- CANCELLING ORDER ---")
#             result = phemex.cancel_all_orders(symbol)
#             print(f"✅ Cancel request sent successfully!")
#             print(f"Cancel result: {result}")
            
#             # Verify cancellation
#             time.sleep(2)  # Brief pause to let cancellation process
#             final_order = phemex.fetch_order(order_id, symbol)
#             print(f"Final order status: {final_order.get('status')}")
#     except Exception as e:
#         print(f"Error checking/cancelling order: {e}")
#         print("Attempting to cancel all orders anyway...")
#         try:
#             phemex.cancel_all_orders(symbol)
#             print("General cancellation request sent.")
#         except Exception as ce:
#             print(f"Failed to cancel orders: {ce}")
            
# except ccxt.InsufficientFunds as e:
#     print(f"❌ Error: Insufficient funds to place this order.")
#     print(f"   You need BTC in your account to trade this inverse perpetual contract.")
#     print(f"   Please deposit BTC to your Phemex account and try again.")
#     print(f"\nTechnical details: {e}")
# except ccxt.ExchangeError as e:
#     print(f"❌ Exchange error: {e}")
#     print("   This could be due to invalid parameters or exchange restrictions.")
# except Exception as e:
#     print(f"❌ Unexpected error: {e}")
    
# print("\n--- SUMMARY ---")
# print(f"Attempted to place a limit buy order for {size} contracts of {symbol} at ${bid}")
# print(f"Algorithm execution completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# print("Check your Phemex account for the final status of any orders.")

# # Algorithmic trading loop
# print("\n=== STARTING ALGORITHMIC TRADING LOOP ===")
# print("This algorithm will repeatedly place and cancel orders")
# print("Press Ctrl+C to stop the algorithm at any time")

# # Configuration
# max_iterations = 10  # Set to None for infinite loop
# wait_time = 5  # Seconds to wait between order placement and cancellation
# iteration = 1
# total_orders_placed = 0
# total_orders_cancelled = 0
# errors_encountered = 0

# try:
#     go = True
#     start_time = datetime.datetime.now()
#     print(f"Algorithm started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
#     while go:
#         print(f"\n--- ITERATION {iteration} ---")
        
#         # Get current market price for reference
#         try:
#             ticker = phemex.fetch_ticker(symbol)
#             current_price = ticker['last']
#             print(f"Current market price: ${current_price}")
            
#             # Dynamically adjust bid price to be 1% below current market price
#             dynamic_bid = round(current_price * 0.99 / price_precision) * price_precision
#             print(f"Dynamic bid price: ${dynamic_bid} (1% below market)")
            
#             # Use dynamic or static price based on preference
#             # Uncomment the line below to use dynamic pricing
#             # bid = dynamic_bid
#             print(f"Using bid price: ${bid}")
#         except Exception as e:
#             print(f"Warning: Could not fetch current price: {e}")
        
#         # Check balance before placing order
#         try:
#             balance = phemex.fetch_balance()
#             btc_balance = balance.get('BTC', {}).get('free', 0)
#             print(f"Current BTC balance: {btc_balance}")
#         except Exception as e:
#             print(f"Warning: Could not fetch balance: {e}")
        
#         # Place order
#         try:
#             print(f"Placing limit buy order: {size} contracts at ${bid}...")
#             order = phemex.create_limit_buy_order(symbol, size, bid, params)
#             print(f"✅ Order {iteration} placed successfully!")
#             print(f"Order ID: {order.get('id', 'Unknown')}")
#             total_orders_placed += 1
            
#             # Sleep for the configured wait time
#             print(f"Waiting {wait_time} seconds before cancellation...")
#             time.sleep(wait_time)
            
#             # Cancel the order
#             print(f"Cancelling orders for {symbol}...")
#             cancel_result = phemex.cancel_all_orders(symbol)
#             print(f"✅ Cancellation request completed")
#             print(f"Cancel result: {cancel_result}")
#             total_orders_cancelled += 1
            
#         except ccxt.InsufficientFunds as e:
#             print(f"❌ Insufficient funds error: {e}")
#             errors_encountered += 1
#             print("Waiting 30 seconds before retrying...")
#             time.sleep(30)  # Longer wait on balance error
            
#         except Exception as e:
#             print(f"❌ Error during iteration {iteration}: {e}")
#             errors_encountered += 1
#             print("Waiting 10 seconds before retrying...")
#             time.sleep(10)
        
#         # Check if we should continue
#         if max_iterations is not None and iteration >= max_iterations:
#             print(f"\nReached maximum iterations ({max_iterations})")
#             go = False
        
#         # Increment iteration counter
#         iteration += 1
        
#         # Small delay between iterations
#         if go:
#             print("Preparing for next iteration...")
#             time.sleep(2)
            
# except KeyboardInterrupt:
#     print("\n\nAlgorithm stopped by user (Ctrl+C)")
# except Exception as e:
#     print(f"\n\nAlgorithm stopped due to unexpected error: {e}")
# finally:
#     # Final summary
#     end_time = datetime.datetime.now()
#     duration = end_time - start_time
    
#     print("\n=== ALGORITHM SUMMARY ===")
#     print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"Duration: {duration}")
#     print(f"Total iterations: {iteration - 1}")
#     print(f"Orders placed: {total_orders_placed}")
#     print(f"Orders cancelled: {total_orders_cancelled}")
#     print(f"Errors encountered: {errors_encountered}")
    
#     # Final account balance
#     try:
#         balance = phemex.fetch_balance()
#         btc_balance = balance.get('BTC', {}).get('free', 0)
#         print(f"Final BTC balance: {btc_balance}")
#     except Exception as e:
#         print(f"Could not fetch final balance: {e}")
    
#     print("\nAlgorithm execution completed.")


# phemex.create_limit_buy_order(symbol, size, bid, params)

# # Sleep for 10 seconds
# print("I just made the order, sleeping for 10 seconds...")
# time.sleep(10)

# # Cancel the order
# phemex.cancel_all_orders(symbol)

def bot():
    """
    Automated trading bot function that places a limit buy order and cancels it after 5 seconds.
    This function is designed to be scheduled to run at regular intervals.
    """
    print(f"\n===== BOT EXECUTION at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    
    # Get current market price
    try:
        ticker = phemex.fetch_ticker(symbol)
        current_price = ticker['last']
        print(f"Current market price: ${current_price}")
        
        # Optionally use dynamic pricing
        # dynamic_bid = round(current_price * 0.99 / price_precision) * price_precision
        # bid = dynamic_bid
        print(f"Using bid price: ${bid}")
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch current price: {e}")
    
    # Check balance
    try:
        balance = phemex.fetch_balance()
        btc_balance = balance.get('BTC', {}).get('free', 0)
        print(f"Current BTC balance: {btc_balance}")
        
        if float(btc_balance) <= 0:
            print("⚠️ Warning: Insufficient BTC balance, order may fail")
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch balance: {e}")
    
    # Place order
    try:
        print(f"Placing limit buy order: {size} contracts at ${bid}...")
        order = phemex.create_limit_buy_order(symbol, size, bid, params)
        print(f"✅ Order placed successfully!")
        print(f"Order ID: {order.get('id', 'Unknown')}")
        print(f"Status: {order.get('status', 'Unknown')}")
        
        # Wait before cancelling
        print(f"Waiting 5 seconds before cancellation...")
        time.sleep(5)
        
        # Cancel order
        print(f"Cancelling orders for {symbol}...")
        result = phemex.cancel_all_orders(symbol)
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
            ticker = phemex.fetch_ticker(symbol)
            current_price = ticker['last']
            print(f"Current market price: ${current_price}")
        except Exception as e:
            print(f"⚠️ Warning: Could not fetch current price: {e}")
        
        try:
            # Place order
            print(f"Placing limit buy order: {size} contracts at ${bid}...")
            order = phemex.create_limit_buy_order(symbol, size, bid, params)
            print(f"✅ Order placed successfully at {datetime.datetime.now().strftime('%H:%M:%S')}")
            print(f"Order ID: {order.get('id', 'Unknown')}")
            
            # Wait before cancelling
            print(f"Waiting 10 seconds before cancellation...")
            for i in range(10, 0, -1):
                print(f"Cancelling in {i} seconds...", end="\r")
                time.sleep(1)
            print("\nCancelling order now...")
            
            # Cancel order
            result = phemex.cancel_all_orders(symbol)
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
                        balance = phemex.fetch_balance()
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
        balance = phemex.fetch_balance()
        btc_balance = balance.get('BTC', {}).get('free', 0)
        print(f"Final BTC balance: {btc_balance}")
    except Exception as e:
        print(f"Could not fetch final balance: {e}")
        
    print("\nBot execution completed.")
