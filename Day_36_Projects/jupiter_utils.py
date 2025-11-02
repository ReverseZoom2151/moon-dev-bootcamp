"""Utilities for interacting with the Jupiter Aggregator API."""

import requests
import json
import base64
import time
import logging

from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.api import Client as SolanaClient # Rename to avoid conflict
from solana.rpc.types import TxOpts

# Import configuration and secrets
import config
import dontshare as ds

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Solana keypair from secrets
try:
    SOL_KEYPAIR = Keypair.from_base58_string(ds.sol_key)
except AttributeError:
    logging.error("Solana key ('sol_key') not found in dontshare.py. Please add it.")
    exit(1) # Exit if key is missing
except Exception as e:
    logging.error(f"Error loading Solana keypair: {e}")
    exit(1)

# Initialize Solana RPC client
try:
    HTTP_CLIENT = SolanaClient(ds.ankr_key) # Ensure 'ankr_key' holds the RPC URL
except AttributeError:
    logging.error("Solana RPC URL ('ankr_key') not found in dontshare.py. Please add it.")
    exit(1)
except Exception as e:
    logging.error(f"Error initializing Solana client: {e}")
    exit(1)

def market_buy(token_address: str, amount_usdc_wei: int):
    """Executes a market buy on Jupiter using USDC.

    Args:
        token_address: The mint address of the token to buy.
        amount_usdc_wei: The amount of USDC to spend, in smallest units (wei, 10^-6).
    """
    quote_url = (
        f'https://quote-api.jup.ag/v6/quote'
        f'?inputMint={config.QUOTE_TOKEN}'
        f'&outputMint={token_address}'
        f'&amount={amount_usdc_wei}'
        f'&slippageBps={config.SLIPPAGE}'
        f'&restrictIntermediateTokens=true' # Optional: Restricts routes
    )
    swap_url = 'https://quote-api.jup.ag/v6/swap'

    max_retries = 5
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}: Getting quote for buying {token_address}...")
            quote_response = requests.get(quote_url)
            quote_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            quote = quote_response.json()
            logging.debug(f"Quote received: {quote}")

            logging.info("Requesting swap transaction...")
            swap_payload = {
                "quoteResponse": quote,
                "userPublicKey": str(SOL_KEYPAIR.pubkey()),
                "prioritizationFeeLamports": config.PRIORITY_FEE,
                # "dynamicComputeUnitLimit": True, # Consider enabling for dynamic CU limits
                # "computeUnitPriceMicroLamports": 1000 # Optional: Set compute unit price
            }
            swap_response = requests.post(swap_url,
                                        headers={"Content-Type": "application/json"},
                                        data=json.dumps(swap_payload))
            swap_response.raise_for_status()
            swap_tx_data = swap_response.json()
            logging.debug(f"Swap response: {swap_tx_data}")

            swap_tx_base64 = swap_tx_data['swapTransaction']
            raw_tx_bytes = base64.b64decode(swap_tx_base64)
            versioned_tx = VersionedTransaction.from_bytes(raw_tx_bytes)

            # Sign the transaction
            signed_tx = VersionedTransaction(versioned_tx.message, [SOL_KEYPAIR])

            logging.info("Sending transaction...")
            # Use TxOpts for preflight checks and commitment level if needed
            # opts = TxOpts(skip_preflight=True, preflight_commitment="confirmed")
            opts = TxOpts(skip_preflight=True)
            tx_signature = HTTP_CLIENT.send_raw_transaction(bytes(signed_tx), opts=opts).value

            logging.info(f"Transaction sent! Signature: {tx_signature}")
            print(f"https://solscan.io/tx/{str(tx_signature)}") # Keep direct print for easy access
            return tx_signature # Return signature on success

        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                logging.error("Max retries reached. Failed to execute market buy.")
                raise # Reraise the last exception
            time.sleep(retry_delay)
        except KeyError as e:
            logging.error(f"Error parsing API response (KeyError: {e}) on attempt {attempt + 1}. Response: {quote_response.text if 'quote_response' in locals() else 'N/A'} / {swap_response.text if 'swap_response' in locals() else 'N/A'}")
            if attempt == max_retries - 1:
                 raise
            time.sleep(retry_delay)
        except Exception as e:
            logging.error(f"An unexpected error occurred on attempt {attempt + 1}: {e}")
            logging.exception("Traceback:") # Log full traceback for unexpected errors
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)

    # Should not be reached if max_retries > 0, but provides a fallback
    raise Exception("Failed to execute market buy after multiple retries.") 