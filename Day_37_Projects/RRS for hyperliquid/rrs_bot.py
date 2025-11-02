'''
this bot will simply grab the 15min rrs and long the top performer and short the bottom performere
this will make it a market neutral strategy
'''
# rrs_bot.py
# Standard library imports
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass  # NEW
# Third-party imports
import pandas as pd
from eth_account.signers.local import LocalAccount
from eth_account import Account

# Hyperliquid API imports
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Local application imports
import nice_funcs as n
from dontshareconfig import secret
from config import SYMBOLS, DATA_DIR, RESULTS_DIR
from data_fetcher import fetch_data
from data_processor import calculate_returns_and_volatility, calculate_volume_metrics
from rrs_calculator import calculate_rrs

# Configure logging
logger = logging.getLogger(__name__)
# Allow log level override through env-var (default INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------------
# Trading / Bot configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradingConfig:
    """All tunable parameters for the RRS bot are kept here in one place."""
    # Order sizing / leverage
    USDC_SIZE: float = 10
    LEVERAGE: int = 3  # Leverage for both long and short positions

    # Execution cadence
    SLEEP_SECONDS: int = 30  # Main loop sleep interval
    RRS_CACHE_TIMEOUT: int = 30  # Minutes until RRS should be recomputed

    # Risk management
    TP: float = 10    # Take-profit at 10% combined PnL
    SL: float = -10   # Stop-loss at ‑10% combined PnL

    # Misc behaviour toggles
    AUTO_ADJUST_LEVERAGE: bool = True  # Dynamically adjust leverage per symbol
    LIMIT_ORDER_BUFFER: float = 0.001  # 0.1% buffer above/below market price


# Single source-of-truth config object
CONFIG = TradingConfig()

# Backwards-compatibility aliases (minimise code churn)
USDC_SIZE = CONFIG.USDC_SIZE
LEVERAGE = CONFIG.LEVERAGE
LIMIT_ORDER_BUFFER = CONFIG.LIMIT_ORDER_BUFFER
SLEEP_SECONDS = CONFIG.SLEEP_SECONDS
RRS_CACHE_TIMEOUT = CONFIG.RRS_CACHE_TIMEOUT
TP = CONFIG.TP
SL = CONFIG.SL
AUTO_ADJUST_LEVERAGE = CONFIG.AUTO_ADJUST_LEVERAGE

# ---------------------------------------------------------------------------
# Network / cache paths
# ---------------------------------------------------------------------------

# Data directories using config
BOT_DATA_DIR = Path(DATA_DIR) / "bot_data"
RRS_CACHE_FILE = BOT_DATA_DIR / "rrs_cache.csv"
BOT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Re-use a single Info client to avoid reconnecting every loop
INFO_CLIENT = Info(constants.MAINNET_API_URL, skip_ws=True)

def check_pnl_and_close(account, top_symbol, bottom_symbol):
    """Check combined PnL and close positions if needed"""
    # Get PnL for both positions
    _, _, _, _, _, top_pnl, _ = n.get_position(top_symbol, account)
    _, _, _, _, _, bottom_pnl, _ = n.get_position(bottom_symbol, account)
    
    if top_pnl is None: top_pnl = 0
    if bottom_pnl is None: bottom_pnl = 0
    
    combined_pnl = top_pnl + bottom_pnl
    logger.info(f"📊 Combined PnL: {combined_pnl:.2f}%")
    
    # Close positions if PnL hits targets
    if combined_pnl >= TP or combined_pnl <= SL:
        logger.info(f"{'🎯 Take Profit' if combined_pnl >= TP else '🛑 Stop Loss'} hit! Closing positions...")
        n.close_all_positions(account)
        return True
    return False

def should_calculate_rrs():
    """Check if we need to recalculate RRS based on cache age"""
    if not os.path.exists(RRS_CACHE_FILE):
        return True
    
    cache_modified_time = datetime.fromtimestamp(os.path.getmtime(RRS_CACHE_FILE))
    time_difference = datetime.now() - cache_modified_time
    return time_difference.total_seconds() / 60 > RRS_CACHE_TIMEOUT

def get_cached_rrs():
    """Get RRS scores from cache"""
    if os.path.exists(RRS_CACHE_FILE):
        return pd.read_csv(RRS_CACHE_FILE)
    return None

def calculate_current_rrs():
    """Calculate current RRS for all symbols against BTC"""
    # Check cache first
    if not should_calculate_rrs():
        logger.info("🎯 Using cached RRS scores (less than 30 minutes old)")
        return get_cached_rrs()
    
    logger.info("🌙 Moon Dev - Calculating fresh RRS scores...")
    
    # Fetch BTC data as benchmark
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    benchmark_df = fetch_data('BTC', '15m', start_time, end_time)
    benchmark_df = calculate_returns_and_volatility(benchmark_df)
    
    results = []
    for symbol in SYMBOLS:
        if symbol == 'BTC':
            continue
            
        logger.info(f"Processing {symbol}...")
        df = fetch_data(symbol, '15m', start_time, end_time)
        if not df.empty:
            df = calculate_returns_and_volatility(df)
            df = calculate_volume_metrics(df)
            df = calculate_rrs(df, benchmark_df)
            
            # Get the latest RRS value
            latest_rrs = df['smoothed_rrs'].iloc[-1]
            results.append({'symbol': symbol, 'rrs': latest_rrs})
    
    # Save to cache
    rrs_df = pd.DataFrame(results)
    rrs_df.to_csv(RRS_CACHE_FILE, index=False)
    return rrs_df

def adjust_positions(account, force_close=False):
    """Adjust positions based on RRS scores"""
    current_minute = datetime.now().minute
    
    # Force close at 58 minutes past the hour to avoid funding
    if current_minute == 58 or force_close:
        logger.info("⏰ Closing positions before funding...")
        n.close_all_positions(account)
        if force_close:
            return
    
    # Only open new positions at the top of the hour
    if current_minute != 0:
        return
        
    logger.info("🔄 Moon Dev - Adjusting positions...")
    
    # Calculate current RRS scores
    rrs_df = calculate_current_rrs()
    rrs_df = rrs_df.sort_values('rrs', ascending=False)
    
    # Get top and bottom performers
    top_symbol = rrs_df.iloc[0]['symbol']
    bottom_symbol = rrs_df.iloc[-1]['symbol']
    
    logger.info(f"🔝 Top performer: {top_symbol}")
    logger.info(f"👇 Bottom performer: {bottom_symbol}")
    
    # Long the top performer with aggressive buy price
    logger.info("📈 Setting up long position for %s", top_symbol)
    if AUTO_ADJUST_LEVERAGE:
        lev, size = n.adjust_leverage_usd_size(top_symbol, USDC_SIZE, LEVERAGE, account)
    else:
        # Fallback: derive size from notional and spot price without altering leverage
        _, px_decimals = n.get_sz_px_decimals(top_symbol)
        mkt_px, _, _ = n.ask_bid(top_symbol)
        size = round(USDC_SIZE / mkt_px, px_decimals)
        lev = LEVERAGE
    _, px_decimals = n.get_sz_px_decimals(top_symbol)
    current_price, _, _ = n.ask_bid(top_symbol)
    entry_price = round(current_price * 1.01, px_decimals)  # Bid 1% above market
    
    n.limit_order(
        coin=top_symbol,
        is_buy=True,
        sz=size,
        limit_px=entry_price,
        reduce_only=False,
        account=account
    )
    
    # Short the bottom performer with aggressive sell price
    logger.info("📉 Setting up short position for %s", bottom_symbol)
    if AUTO_ADJUST_LEVERAGE:
        lev, size = n.adjust_leverage_usd_size(bottom_symbol, USDC_SIZE, LEVERAGE, account)
    else:
        _, px_decimals = n.get_sz_px_decimals(bottom_symbol)
        mkt_px, _, _ = n.ask_bid(bottom_symbol)
        size = round(USDC_SIZE / mkt_px, px_decimals)
        lev = LEVERAGE
    _, px_decimals = n.get_sz_px_decimals(bottom_symbol)
    current_price, _, _ = n.ask_bid(bottom_symbol)
    entry_price = round(current_price * 0.99, px_decimals)  # Offer 1% below market
    
    n.limit_order(
        coin=bottom_symbol,
        is_buy=False,
        sz=size,
        limit_px=entry_price,
        reduce_only=False,
        account=account
    )
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rrs_df.to_csv(f"{RESULTS_DIR}/rrs_scores_{timestamp}.csv", index=False)
    
    return top_symbol, bottom_symbol

def check_and_manage_positions(account):
    """Regular check of positions for PnL management"""
    user_state = INFO_CLIENT.user_state(account.address)
    logger.info(f"Current account value: {user_state['marginSummary']['accountValue']}")
    
    # Get positions directly from user state
    positions = user_state["assetPositions"]
    if len(positions) == 2:  # We have both positions
        # Extract symbols and PnL
        pos1 = positions[0]["position"]
        pos2 = positions[1]["position"]
        
        pnl1 = float(pos1["returnOnEquity"]) * 100
        pnl2 = float(pos2["returnOnEquity"]) * 100
        
        combined_pnl = pnl1 + pnl2
        logger.info(f"📊 Combined PnL: {combined_pnl:.2f}%")
        
        # Close positions if PnL hits targets
        if combined_pnl >= TP or combined_pnl <= SL:
            logger.info(f"{'🎯 Take Profit' if combined_pnl >= TP else '🛑 Stop Loss'} hit! Closing positions...")
            n.close_all_positions(account)
            return True
    return False

def main():
    """Main bot function"""
    logger.info("🚀 Moon Dev's RRS Bot Starting...")
    account: LocalAccount = Account.from_key(secret)
    
    while True:
        try:
            current_minute = datetime.now().minute
            
            # Check PnL every 30 seconds if we have positions
            check_and_manage_positions(account)
            
            # Adjust positions at the top of each hour
            if current_minute in (0, 1):
                adjust_positions(account)
            
            # Force close positions before funding
            elif current_minute in (58, 59):
                adjust_positions(account, force_close=True)
            
            time.sleep(SLEEP_SECONDS)
            
        except Exception:
            logger.exception("❌ Error encountered in main loop")
            time.sleep(15)

if __name__ == "__main__":
    main()