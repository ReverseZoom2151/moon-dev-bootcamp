"""Monitors target Twitter accounts for Solana contract addresses in new tweets
and executes buy orders via Jupiter aggregator if found.
"""

import pandas as pd
from datetime import datetime
import time
import logging
import sys
import os

# Patch httpx before importing twikit - Required for certain environments/proxies
try:
    import httpx
    original_client = httpx.Client
    def patched_client(*args, **kwargs):
        # Attempt to remove proxy, handle if not present
        kwargs.pop('proxy', None)
        return original_client(*args, **kwargs)
    httpx.Client = patched_client
    logging.debug("httpx.Client patched to ignore proxy settings.")
except ImportError:
    logging.warning("httpx not found, skipping patch. Twikit might have issues with proxies.")
except Exception as e:
    logging.warning(f"Failed to patch httpx: {e}")

# Now import twikit and other dependencies
try:
    from twikit import Client as TwitterClient
    from twikit.errors import TooManyRequests, Forbidden, TwitterException
except ImportError:
    logging.error("twikit library not found. Please install it: pip install twikit")
    sys.exit(1)

try:
    import schedule
except ImportError:
    logging.error("schedule library not found. Please install it: pip install schedule")
    sys.exit(1)

# Local imports
import config
import dontshare as ds
from utils import is_contract_address
from jupiter_utils import market_buy # Assuming market_buy is now in jupiter_utils.py

# --- Logging Setup ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
        # Optional: Add FileHandler here if needed
        # logging.FileHandler("bot.log")
    ]
)
logger = logging.getLogger(__name__)


# --- Twitter Credentials and Targets ---
try:
    # It's crucial these are defined in dontshare.py
    USERNAME = ds.user_name
    EMAIL = ds.email
    PASSWORD = ds.password
    TARGET_ACCOUNTS = ds.target_accounts # Expecting a dict like {"id": "username"}
except AttributeError as e:
    logger.error(f"Missing required credentials or target_accounts in dontshare.py: {e}")
    sys.exit(1)


# --- Global State ---
# Use a dictionary to store the latest processed tweet ID for each target user
latest_processed_tweet_ids = {}
df_tweet_log = pd.DataFrame(columns=['timestamp', 'username', 'tweet_text', 'contract_found'])

def load_tweet_log():
    """Loads the tweet log CSV if it exists."""
    global df_tweet_log
    if os.path.exists(config.CSV_FILENAME):
        try:
            df_tweet_log = pd.read_csv(config.CSV_FILENAME)
            logger.info(f"Loaded existing tweet log from {config.CSV_FILENAME}")
        except Exception as e:
            logger.error(f"Error loading tweet log {config.CSV_FILENAME}: {e}")
            # Continue with an empty DataFrame
            df_tweet_log = pd.DataFrame(columns=['timestamp', 'username', 'tweet_text', 'contract_found'])
    else:
        logger.info(f"Tweet log {config.CSV_FILENAME} not found, starting fresh.")

def save_tweet_log(new_data):
    """Appends new data to the tweet log DataFrame and saves to CSV."""
    global df_tweet_log
    try:
        # Use concat instead of append for future compatibility
        new_df = pd.DataFrame([new_data])
        df_tweet_log = pd.concat([df_tweet_log, new_df], ignore_index=True)
        df_tweet_log.to_csv(config.CSV_FILENAME, index=False)
        logger.debug(f"Saved updated tweet log to {config.CSV_FILENAME}")
    except Exception as e:
        logger.error(f"Error saving tweet log: {e}")


def initialize_twitter_client() -> TwitterClient | None:
    """Initializes and authenticates the Twitter client."""
    client = TwitterClient()
    try:
        # Attempt to load cookies first for faster login
        if os.path.exists(config.COOKIES_FILENAME):
             client.load_cookies(config.COOKIES_FILENAME)
             logger.info(f"Loaded Twitter cookies from {config.COOKIES_FILENAME}")
             # Optional: Verify login, e.g., by getting self info
             # Need to check if client is authenticated after loading cookies
             # my_info = client.me # Example: Accessing a property that requires auth
             # logger.info(f"Cookie login successful for user: {client.me.name}")
             logger.info("Cookie login successful (verification pending API call)")
        else:
             logger.info("No cookies file found. Attempting login with credentials...")
             client.login(auth_info_1=USERNAME, auth_info_2=EMAIL, password=PASSWORD)
             client.save_cookies(config.COOKIES_FILENAME)
             logger.info(f"Login successful, cookies saved to {config.COOKIES_FILENAME}")
        return client
    except Forbidden as e:
        logger.error(f"Twitter login failed (Forbidden): {e}. Check credentials or 2FA.")
        logger.error("Ensure USERNAME, EMAIL, PASSWORD in dontshare.py are correct.")
        if os.path.exists(config.COOKIES_FILENAME):
            logger.warning(f"Deleting potentially invalid cookies file: {config.COOKIES_FILENAME}")
            try:
                os.remove(config.COOKIES_FILENAME)
            except OSError as del_e:
                logger.error(f"Failed to delete cookies file: {del_e}")
    except TwitterException as e:
         logger.error(f"Twitter API error during login: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Twitter client initialization: {e}", exc_info=True)

    return None # Return None if initialization fails


def execute_buy_strategy(contract_address: str):
    """Attempts to buy the token aggressively based on config settings."""
    logger.info(f"🚀 Starting aggressive buy attempts for {contract_address}..." )
    buy_attempts = 0
    successful_buys = 0
    try:
        while True: # Loop indefinitely until manual stop or unrecoverable error
            buy_attempts += 1
            logger.info(f"--- Batch {buy_attempts} --- " )
            batch_success = 0
            for i in range(config.ORDERS_PER_OPEN):
                try:
                    size_usdc_wei = int(config.USDC_SIZE * 10**6) # Calculate units here
                    logger.info(f"💰 Attempt {i + 1}/{config.ORDERS_PER_OPEN}: Buying {config.USDC_SIZE} USDC worth of {contract_address}" )
                    tx_sig = market_buy(contract_address, size_usdc_wei)
                    if tx_sig: # Check if market_buy returned a signature
                        logger.info(f"✅ Buy order {i+1} placed successfully! Tx: {tx_sig}" )
                        successful_buys += 1
                        batch_success += 1
                    else:
                        # market_buy should raise an exception on failure, but handle if it doesn't
                        logger.warning(f"⚠️ Buy order {i+1} did not return a signature." )

                    time.sleep(0.5)  # Small delay between orders in a batch

                except Exception as buy_error:
                    logger.error(f"⚠️ Buy attempt {i + 1} failed: {buy_error}" )
                    # Consider if this error is critical enough to stop all buying
                    # For now, we continue to the next attempt/batch
                    time.sleep(1) # Pause longer after a failure

            logger.info(f"--- Batch {buy_attempts} completed ({batch_success}/{config.ORDERS_PER_OPEN} successful) --- " )
            if batch_success == 0 and buy_attempts > 3: # Stop if multiple batches fail completely
                 logger.warning("Multiple consecutive batches failed to place any orders. Stopping buy strategy." )
                 break
            time.sleep(2)  # Pause between batches

    except KeyboardInterrupt:
        logger.warning("⛔ Manually stopped buying strategy." )
    except Exception as e:
        # Catch unexpected errors in the outer loop
        logger.error(f"❌ Unexpected error during buy strategy execution: {e}", exc_info=True)
    finally:
        logger.info(f"🏁 Buy strategy finished. Total successful buys: {successful_buys}" )


def check_new_tweets(client: TwitterClient):
    """Checks for new tweets from target accounts and triggers actions."""
    global latest_processed_tweet_ids
    current_time = datetime.now()

    if not TARGET_ACCOUNTS:
        logger.warning("No target accounts configured in dontshare.py (target_accounts). Skipping check." )
        return

    for account_id, username in TARGET_ACCOUNTS.items():
        logger.info(f"🔍 Checking tweets for {username} (ID: {account_id})..." )
        try:
            # Fetch only the most recent tweets to minimize API usage
            # Note: 'Tweets' might fetch more than needed. Consider specific count if API allows.
            user = client.get_user_by_id(account_id) # Verify user exists
            if not user:
                logger.warning(f"Could not find user {username} with ID {account_id}. Skipping." )
                continue

            tweets = user.get_tweets('Tweets', count=5) # Fetch last 5, adjust as needed
            if not tweets:
                logger.info(f"No tweets found for {username}." )
                continue

            latest_tweet = tweets[0]
            logger.debug(f"Latest tweet from {username} (ID: {latest_tweet.id}): {latest_tweet.text[:50]}..." )

            # Check if this tweet is newer than the last one processed for this user
            last_id = latest_processed_tweet_ids.get(account_id)
            if last_id and latest_tweet.id == last_id:
                # Use debug level for non-events
                logger.debug(f"No new tweets detected for {username} (Tweet ID {latest_tweet.id} already processed)." )
                continue # Skip if it's the same tweet we processed last time

            logger.info(f"🌟 New tweet detected for {username}! (ID: {latest_tweet.id})" )
            logger.info(f"   Tweet Text: {latest_tweet.text}" )

            # Process the new tweet
            contract_address = is_contract_address(latest_tweet.text)
            log_data = {
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'username': username,
                'tweet_text': latest_tweet.text,
                'contract_found': contract_address if contract_address else ''
            }
            save_tweet_log(log_data) # Log every new tweet checked

            if contract_address:
                logger.info(f"💎 Found potential Solana contract address: {contract_address}" )
                # --- Trigger Buy Strategy --- 
                execute_buy_strategy(contract_address)
                # --- Buy Strategy Finished --- 
            else:
                logger.info("No contract address found in this tweet." )

            # Update the latest processed tweet ID for this user AFTER processing
            latest_processed_tweet_ids[account_id] = latest_tweet.id
            logger.debug(f"Updated latest processed tweet ID for {username} to {latest_tweet.id}" )

        except TooManyRequests:
            logger.warning(f"⏳ Rate limit hit while checking {username}. Sleeping for 60 seconds..." )
            time.sleep(60) # Sleep longer for rate limits
            # Consider breaking the loop or implementing backoff for persistent rate limits
            break # Stop checking other users for this cycle if rate limited
        except Forbidden as e:
             logger.error(f"❌ Twitter API Forbidden error for {username}: {e}. May indicate account issues or permissions problem." )
             # This might be serious, potentially stop or alert
        except TwitterException as e:
            logger.error(f"❌ Twitter API error checking {username}: {e}" )
        except Exception as e:
            logger.error(f"❌ Unexpected error checking {username}: {e}", exc_info=True) # Log traceback

        # Short sleep between checking different users (optional, helps avoid hammering)
        time.sleep(1)


def main_loop():
    """Main operational loop, scheduled to run periodically."""
    logger.info("--- Starting new check cycle --- " )
    client = initialize_twitter_client()
    if client:
        check_new_tweets(client)
        logger.info("--- Check cycle finished --- " )
    else:
        logger.error("Failed to initialize Twitter client. Skipping check cycle." )
        # Consider adding a longer sleep or exit strategy if client fails repeatedly
        time.sleep(60)


if __name__ == "__main__":
    logger.info(f"🚀 MoonDev's Twitter Monitor Initializing..." )
    logger.info(f"   Configuration: Slippage={config.SLIPPAGE}bps, PriorityFee={config.PRIORITY_FEE}lamports, OrderSize={config.USDC_SIZE}USDC" )
    logger.info(f"   Polling Interval: {config.POLLING_INTERVAL} seconds" )
    try:
        # Log usernames only for privacy/clarity
        target_usernames = list(TARGET_ACCOUNTS.values())
        logger.info(f"   Target Accounts: {target_usernames}" )
    except Exception:
         logger.error("Could not log target account usernames.") # Handle if TARGET_ACCOUNTS isn't a dict

    load_tweet_log() # Load previous logs

    # --- Initial Run ---
    logger.info("--- Performing initial check run --- " )
    main_loop() # Run once immediately
    logger.info("--- Initial check run complete --- " )

    # --- Scheduling ---
    logger.info(f"🕒 Scheduling checks every {config.POLLING_INTERVAL} seconds." )
    schedule.every(config.POLLING_INTERVAL).seconds.do(main_loop)

    # --- Keep Running ---
    logger.info("--- Starting schedule loop (Press Ctrl+C to stop) ---")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1) # Sleep briefly to avoid busy-waiting
        except KeyboardInterrupt:
            logger.warning("🛑 KeyboardInterrupt received. Shutting down scheduler..." )
            break
        except Exception as e:
            logger.error(f"❌ Unhandled error in scheduler loop: {e}", exc_info=True)
            logger.error("   Attempting to continue after a delay." )
            time.sleep(60) # Delay after scheduler error

    logger.info("👋 Exiting MoonDev's Twitter Monitor." )