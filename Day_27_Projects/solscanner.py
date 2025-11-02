import asyncio
import json
import csv
from datetime import datetime
import aiohttp
from websockets import connect
import os
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dontshareconfig import HELIUS_WS_API # This now imports just the API key

# Construct the full WebSocket URL
HELIUS_API_KEY = HELIUS_WS_API # Rename for clarity
HELIUS_WSS_URL = f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"

RAYDIUM_PROGRAM_ID = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# Get CSV file path from environment variable or use default
# Default path changed to be a local file in the current directory
DEFAULT_CSV_FILE = "./new_sol_tokens.csv"
CSV_FILE = os.environ.get("SOLSCANNER_CSV_FILE", DEFAULT_CSV_FILE)
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com" # Use Helius RPC or QuickNode for better reliability if needed

# Function to configure event loop for Windows compatibility
def configure_event_loop():
    if platform.system() == 'Windows':
        # Set policy to use SelectorEventLoop on Windows (for aiodns compatibility)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logger.info("Configured Windows to use SelectorEventLoop for aiodns compatibility")

async def main():
    while True:
        try:
            await run_websocket()
        except Exception as e:
            logger.error(f"Error in WebSocket connection handler: {e}", exc_info=True)
            logger.info("Attempting to reconnect in 5 seconds...")
            await asyncio.sleep(5)

async def heartbeat():
    """Periodic heartbeat to confirm the script is still running."""
    counter = 0
    while True:
        await asyncio.sleep(30)
        counter += 1
        logger.info(f"Still listening for new tokens... (heartbeat #{counter})")

async def run_websocket():
    async with connect(HELIUS_WSS_URL) as websocket:
        logger.info("Connected to WebSocket")

        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {
                    "mentions": [RAYDIUM_PROGRAM_ID]
                },
                {
                    "commitment": "finalized"
                }
            ]
        }

        await websocket.send(json.dumps(subscribe_msg))
        logger.info("Subscription message sent.")

        seen_signatures = set()
        ensure_csv_file_exists()
        async with aiohttp.ClientSession() as session: # Session managed with async with
            # Start the heartbeat task
            heartbeat_task = asyncio.create_task(heartbeat())
            
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=60) # Increased timeout
                        await process_message(message, session, seen_signatures)
                    except asyncio.TimeoutError:
                        logger.debug("WebSocket timeout, sending ping.")
                        await websocket.ping()
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                        # Depending on the error, you might want to break or continue
                        await asyncio.sleep(1) # Avoid tight loop on continuous errors

            finally:
                heartbeat_task.cancel()
                logger.info("WebSocket loop terminated or error occurred.") # Session closed automatically by async with

async def process_message(message: str, session: aiohttp.ClientSession, seen_signatures: set):
    """Processes a single message received from the WebSocket."""
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON message: {message[:100]}...") # Log snippet
        return

    if "params" in data and "result" in data["params"]:
        result_value = data["params"]["result"]["value"]
        signature = result_value.get("signature")
        if signature and signature not in seen_signatures:
            seen_signatures.add(signature) # Add signature immediately
            logs = result_value.get("logs", [])
            if any("initialize2" in log for log in logs):
                logger.info(f"Detected 'initialize2' in logs for signature: {signature}")
                new_token = await get_new_token(session, signature)
                if new_token:
                    logger.info(f"New token found: {new_token} (Tx: https://solscan.io/tx/{signature})")
                    save_to_csv(new_token, signature)
            else:
                 logger.debug(f"No 'initialize2' log found for signature: {signature}")
        elif signature in seen_signatures:
             logger.debug(f"Signature already seen: {signature}")


async def get_new_token(session: aiohttp.ClientSession, signature: str) -> str | None:
    """Fetches transaction details and extracts the new token address."""
    request_body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {
                "encoding": "jsonParsed",
                "maxSupportedTransactionVersion": 0
            }
        ]
    }

    try:
        async with session.post(SOLANA_RPC_URL, json=request_body, timeout=10) as response: # Added timeout
            response.raise_for_status() # Raise exception for bad status codes
            data = await response.json()

            if "error" in data:
                logger.error(f"Error in getTransaction response for {signature}: {data['error']}")
                return None

            instructions = data.get("result", {}).get("transaction", {}).get("message", {}).get("instructions", [])

            for instruction in instructions:
                if instruction.get("programId") == RAYDIUM_PROGRAM_ID and instruction.get("data", "").startswith("16a40b14bb677619"): # Check for initialize2 instruction discriminator (safer than just programId)
                    accounts = instruction.get("accounts", [])
                    # Index 8 corresponds to the base token account (pool coin token account) in initialize2
                    if len(accounts) > 8:
                        token_address = accounts[8]
                        logger.debug(f"Extracted token address {token_address} from accounts for {signature}")
                        return token_address
                    else:
                        logger.warning(f"Instruction accounts length <= 8 for {signature}. Accounts: {accounts}")

    except aiohttp.ClientError as e:
        logger.error(f"HTTP Client error fetching transaction {signature}: {e}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout fetching transaction {signature}")
    except Exception as e:
        logger.error(f"Unexpected error fetching transaction {signature}: {e}", exc_info=True)

    return None

def ensure_csv_file_exists():
    """Ensures the CSV file exists and has the header row."""
    try:
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            with open(CSV_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Token Address", "Time Found", "Epoch Time", "Solscan Link", "DexScreener Link", "Birdeye Link"])
            logger.info(f"Created or initialized CSV file: {CSV_FILE}")
    except IOError as e:
        logger.error(f"Error ensuring CSV file exists ({CSV_FILE}): {e}")
        # Depending on severity, might want to raise or exit

def save_to_csv(new_token: str, signature: str):
    """Appends a new token record to the CSV file."""
    try:
        with open(CSV_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            now = datetime.utcnow()
            time_found = now.strftime("%Y-%m-%d %H:%M:%S")
            epoch_time = int(now.timestamp())
            solscan_link = f"https://solscan.io/tx/{signature}"
            dexscreener_link = f"https://dexscreener.com/solana/{new_token}"
            birdeye_link = f"https://birdeye.so/token/{new_token}?chain=solana"

            writer.writerow([new_token, time_found, epoch_time, solscan_link, dexscreener_link, birdeye_link])
            logger.debug(f"Saved token {new_token} to CSV.")
    except IOError as e:
        logger.error(f"Error writing to CSV file ({CSV_FILE}): {e}")

if __name__ == "__main__":
    logger.info(f"Starting SolScanner. Outputting to: {CSV_FILE}")
    logger.info(f"Monitoring Raydium Program ID: {RAYDIUM_PROGRAM_ID}")
    logger.info(f"Using Solana RPC URL: {SOLANA_RPC_URL}")
    if "HELIUS_WS_API" not in locals():
        logger.error("HELIUS_WS_API not found in dontshareconfig.py. Please ensure it's defined.")
        exit(1) # Or handle appropriately
    configure_event_loop()
    asyncio.run(main())