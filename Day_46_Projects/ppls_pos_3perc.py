'''
4/1- I updated this one to show the USDC Balance of the top four positions closest to liquidation. 
I thought this was important because if someone's gonna get liquidated but they have unlimited USDC in their spot then it doesn't really matter. 



🌙 Moon Dev's Hyperliquid Position Tracker (API Version) 🚀
Built with love by Moon Dev 🌙 ✨

This script uses the Moon Dev API to get pre-collected Hyperliquid positions data
and performs analysis to identify whale positions and liquidation risks.

Unlike ppls_positions.py, this version doesn't query Hyperliquid directly,
but instead uses centralized data already collected by the Moon Dev API.

disclaimer: this is not financial advice and there is no guarantee of any kind. use at your own risk.
'''

import time
import pandas as pd
import numpy as np
import colorama
from colorama import Fore, Style, Back
import nice_funcs as n
import argparse
import sys
import traceback
import random
from termcolor import colored
import schedule
import requests
import logging
from pathlib import Path

# Add the project root to the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from api import MoonDevAPI

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Configure pandas to display numbers with commas and no scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ===== CONFIGURATION =====
CONFIG = {
    "DATA_DIR_NAME": "ppls_positions_api_results",
    "MIN_POSITION_VALUE": 50000,
    "TOP_N_POSITIONS": 30,
    "TOKENS_TO_ANALYZE": ['BTC', 'ETH', 'XRP', 'SOL'],
    "HIGHLIGHT_THRESHOLD": 2000000,
    "HYPERLIQUID_API_URL": "https://api.hyperliquid.xyz/info",
    "BINANCE_FAPI_URL": "https://fapi.binance.com/fapi/v1/premiumIndex",
    "CSV_FLOAT_FORMAT": '%.2f',
    "SCHEDULE_INTERVAL_MINUTES": 1,
    "AGG_POSITIONS_API_CSV": "aggregated_positions_from_api.csv",
    "ALL_POSITIONS_CSV": "all_positions_details.csv",
    "AGGREGATED_POSITIONS_SUMMARY_CSV": "aggregated_positions_summary.csv",
    "TOP_WHALE_LONG_CSV": "top_whale_long_positions.csv",
    "TOP_WHALE_SHORT_CSV": "top_whale_short_positions.csv",
    "TOP_WHALE_COMBINED_CSV": "top_whale_positions_combined.csv",
    "LIQ_CLOSEST_LONG_CSV": "liquidation_closest_long.csv",
    "LIQ_CLOSEST_SHORT_CSV": "liquidation_closest_short.csv",
    "LIQ_CLOSEST_COMBINED_CSV": "liquidation_closest_combined.csv",
    "LIQ_THRESHOLDS_TABLE_CSV": "liquidation_thresholds_table.csv"
}

# Construct DATA_PATH using pathlib and the project root or a specific base directory
DATA_PATH_BASE = PROJECT_ROOT / "bots" / "hyperliquid" / "data"
DATA_PATH = DATA_PATH_BASE / CONFIG["DATA_DIR_NAME"]

# Moon Dev ASCII Art Banner
MOON_DEV_BANNER = fr"""{Fore.CYAN}
   __  ___                    ____           
  /  |/  /___  ____  ____    / __ \___  _  __
 / /|_/ / __ \/ __ \/ __ \  / / / / _ \| |/_/
/ /  / / /_/ / /_/ / / / / / /_/ /  __/>  <  
/_/  /_/\____/\____/_/ /_(_)____/\___/_/|_|  
                                             
{Fore.MAGENTA}🚀 Hyperliquid Position Tracker API Edition 🌙{Fore.RESET}
"""

# Fun Moon Dev quotes
MOON_DEV_QUOTES = [
    "Tracking whales like a boss! 🐋",
    "Liquidations incoming... maybe! 💥",
    "Moon Dev API - making data fun again! 🎯",
    "To the moon! 🚀 (not financial advice)",
    "Who needs sleep when you have APIs? 😴",
    "Whales move markets, so track the whales! 🐳",
    "Moon Dev sees all the positions... 👀",
    "Diamond hands or liquidation lands? 💎",
    "Watch the whales, follow the money! 💰",
    "API-powered alpha at your fingertips! ✨"
]

def get_random_quote() -> str:
    """Return a random Moon Dev quote"""
    return random.choice(MOON_DEV_QUOTES)

def ensure_data_dir() -> bool:
    """Ensure the data directory exists"""
    try:
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Data directory ready at {DATA_PATH}")
        return True
    except Exception as e:
        logger.error(f"⚠️ Error creating directory {DATA_PATH}: {str(e)}", exc_info=True)
        return False

def get_spot_usdc_balance(address: str) -> float:
    """Get USDC spot position for a given address from Hyperliquid"""
    balance_data_for_logging = None
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "type": "spotClearinghouseState",
            "user": address
        }
        response = requests.post(CONFIG["HYPERLIQUID_API_URL"], headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        balance_data = response.json()
        balance_data_for_logging = balance_data
        
        usdc_balance = 0.0
        if 'balances' in balance_data and isinstance(balance_data['balances'], list):
            for balance_item in balance_data['balances']:
                if isinstance(balance_item, dict) and balance_item.get('coin') == 'USDC':
                    usdc_balance = float(balance_item.get('total', "0"))
                    break
        else:
            logger.warning(f"No 'balances' list found or unexpected format in spotClearinghouseState for address {address}. Data: {balance_data}")
        
        return usdc_balance
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ Timeout fetching USDC spot balance for {address}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ HTTP Error fetching USDC spot balance for {address}: {str(e)}")
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"❌ Data parsing error for USDC spot balance {address}: {str(e)}. Response: {balance_data_for_logging}")
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching USDC spot balance for {address}: {str(e)}", exc_info=True)
    return 0.0

def get_hyperliquid_spot_price_and_symbol_info(symbol: str) -> dict | None:
    """Get spot price and symbol info from Hyperliquid for a given symbol (e.g., BTC)"""
    data_for_logging = None
    try:
        headers = {"Content-Type": "application/json"}
        body = {"type": "spotMetaAndAssetCtxs"}
        
        response = requests.post(CONFIG["HYPERLIQUID_API_URL"], headers=headers, json=body, timeout=10)
        response.raise_for_status()
        data = response.json()
        data_for_logging = data
        
        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[0], dict) or not isinstance(data[1], list):
            logger.error(f"Unexpected API response structure for spotMetaAndAssetCtxs. Data: {data}")
            return None

        tokens = data[0].get('tokens')
        universe = data[0].get('universe')
        asset_ctxs = data[1]

        if not isinstance(tokens, list) or not isinstance(universe, list) or not isinstance(asset_ctxs, list):
            logger.error(f"Missing or invalid critical data types in spotMetaAndAssetCtxs response. Tokens: {type(tokens)}, Universe: {type(universe)}, AssetCtxs: {type(asset_ctxs)}")
            return None
            
        symbol_info_found = None
        for pair in universe:
            if isinstance(pair, dict) and pair.get('name', '').split('/')[0] == symbol.upper():
                symbol_info_found = pair
                break
        
        if symbol_info_found:
            pair_index = symbol_info_found.get('index')
            if not isinstance(pair_index, int) or pair_index < 0 or pair_index >= len(asset_ctxs):
                logger.error(f"Invalid or out-of-bounds index {pair_index} for symbol {symbol} in asset_ctxs.")
                return None

            ctx = asset_ctxs[pair_index]
            if not isinstance(ctx, dict):
                logger.error(f"Asset context for {symbol} (index {pair_index}) is not a dictionary. Ctx: {ctx}")
                return None

            price_str = ctx.get('markPx')
            price = float(price_str) if price_str is not None else None
            
            if price is None:
                logger.warning(f"Mark price (markPx) not available or null for {symbol} in asset_ctxs. Ctx: {ctx}")

            token_indices = symbol_info_found.get('tokens')
            if not isinstance(token_indices, list) or len(token_indices) < 2 or \
               not all(isinstance(ti, int) and 0 <= ti < len(tokens) for ti in token_indices):
                 logger.error(f"Invalid or out-of-bounds token indices for symbol {symbol}. Token indices: {token_indices}, Tokens length: {len(tokens)}")
                 return None

            return {
                'symbol': symbol,
                'price': price,
                'token_info': tokens[token_indices[0]],
                'quote_info': tokens[token_indices[1]],
                'is_canonical': symbol_info_found.get('isCanonical')
            }
        else:
            logger.info(f"Symbol {symbol} not found in Hyperliquid spot universe.")
            return None

    except requests.exceptions.Timeout:
        logger.error(f"❌ Timeout fetching Hyperliquid spot info for {symbol}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ HTTP Error fetching Hyperliquid spot info for {symbol}: {str(e)}")
    except (ValueError, KeyError, IndexError, TypeError) as e:
        logger.error(f"❌ Data parsing/indexing error for Hyperliquid spot info {symbol}: {str(e)}. Response: {data_for_logging}")
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching Hyperliquid spot info for {symbol}: {str(e)}", exc_info=True)
    return None

def display_top_individual_positions(df: pd.DataFrame, n: int = CONFIG["TOP_N_POSITIONS"]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Display top individual long and short positions
    """
    if df is None or df.empty:
        logger.info("No positions to display for top individual positions!")
        return pd.DataFrame(), pd.DataFrame()
    
    display_df = df.copy()
    
    valid_liq_df = display_df[display_df['liquidation_price'] > 0].copy()
    if not valid_liq_df.empty:
        valid_liq_df['position_type_verified'] = np.where(
            valid_liq_df['is_long'],
            valid_liq_df['liquidation_price'] < valid_liq_df['entry_price'],
            valid_liq_df['liquidation_price'] > valid_liq_df['entry_price']
        )
        inconsistent_positions = valid_liq_df[~valid_liq_df['position_type_verified']]
        if not inconsistent_positions.empty:
            logger.warning(f"Note: {len(inconsistent_positions)} positions have been reclassified based on their liquidation prices.")
            valid_liq_df['is_long_corrected'] = valid_liq_df['liquidation_price'] < valid_liq_df['entry_price']
            display_df.loc[valid_liq_df.index, 'is_long'] = valid_liq_df['is_long_corrected']
    
    longs = display_df[display_df['is_long']].sort_values('position_value', ascending=False)
    shorts = display_df[~display_df['is_long']].sort_values('position_value', ascending=False)
    
    logger.info(f"🚀 TOP {n} INDIVIDUAL LONG POSITIONS 📈")
    print(f"{Fore.GREEN}{Style.BRIGHT}{'-'*80}")
    if not longs.empty:
        for i, (_, row) in enumerate(longs.head(n).iterrows(), 1):
            liq_price_val = row['liquidation_price']
            liq_display = f"${liq_price_val:,.2f}" if isinstance(liq_price_val, (int, float)) and liq_price_val > 0 else "N/A"
            print(f"{Fore.GREEN}#{i} {Fore.YELLOW}{row['coin']} {Fore.GREEN}${row['position_value']:,.2f} " + 
                  f"{Fore.BLUE}| Entry: ${row['entry_price']:,.2f} " +
                  f"{Fore.MAGENTA}| PnL: ${row['unrealized_pnl']:,.2f} " +
                  f"{Fore.CYAN}| Leverage: {row['leverage']}x " +
                  f"{Fore.RED}| Liq: {liq_display}")
            print(f"{Fore.CYAN}   Address: {row['address']}")
    else:
        logger.info("No long positions found for top individual display.")
    
    logger.info(f"💥 TOP {n} INDIVIDUAL SHORT POSITIONS 📉")
    print(f"{Fore.RED}{Style.BRIGHT}{'-'*80}")
    if not shorts.empty:
        for i, (_, row) in enumerate(shorts.head(n).iterrows(), 1):
            liq_price_val = row['liquidation_price']
            liq_display = f"${liq_price_val:,.2f}" if isinstance(liq_price_val, (int, float)) and liq_price_val > 0 else "N/A"
            print(f"{Fore.RED}#{i} {Fore.YELLOW}{row['coin']} {Fore.RED}${row['position_value']:,.2f} " + 
                  f"{Fore.BLUE}| Entry: ${row['entry_price']:,.2f} " +
                  f"{Fore.MAGENTA}| PnL: ${row['unrealized_pnl']:,.2f} " +
                  f"{Fore.CYAN}| Leverage: {row['leverage']}x " +
                  f"{Fore.RED}| Liq: {liq_display}")
            print(f"{Fore.CYAN}   Address: {row['address']}")
    else:
        logger.info("No short positions found for top individual display.")
    
    return longs.head(n) if not longs.empty else pd.DataFrame(), shorts.head(n) if not shorts.empty else pd.DataFrame()

def display_risk_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """
    Display metrics for positions closest to liquidation
    """
    empty_df_tuple = pd.DataFrame(), pd.DataFrame(), None
    if df is None or df.empty:
        logger.info("No data for risk metrics display.")
        return empty_df_tuple
    
    risk_df = df[df['liquidation_price'] > 0].copy()
    if risk_df.empty:
        logger.info("No positions with valid liquidation prices for risk metrics.")
        return empty_df_tuple
    
    risk_df = risk_df[risk_df['coin'].isin(CONFIG["TOKENS_TO_ANALYZE"])].copy()
    if risk_df.empty:
        logger.info(f"No positions found for specified TOKENS_TO_ANALYZE: {CONFIG['TOKENS_TO_ANALYZE']}")
        return empty_df_tuple

    unique_coins = risk_df['coin'].unique()
    current_prices = {coin: n.get_current_price(coin) for coin in unique_coins}

    risk_df['current_price'] = risk_df['coin'].map(current_prices).fillna(0)
    risk_df = risk_df[risk_df['current_price'] > 0].copy()
    if risk_df.empty:
        logger.warning("Could not fetch current prices for any relevant coins in risk_df.")
        return pd.DataFrame(), pd.DataFrame(), current_prices

    risk_df['distance_to_liq_pct'] = np.where(
        risk_df['is_long'],
        abs((risk_df['current_price'] - risk_df['liquidation_price']) / risk_df['current_price'] * 100),
        abs((risk_df['liquidation_price'] - risk_df['current_price']) / risk_df['current_price'] * 100)
    )
    risk_df['is_long_corrected'] = risk_df['liquidation_price'] < risk_df['entry_price']
    risk_df['is_long'] = risk_df['is_long_corrected']
    
    risky_longs = risk_df[risk_df['is_long']].sort_values('distance_to_liq_pct')
    risky_shorts = risk_df[~risk_df['is_long']].sort_values('distance_to_liq_pct')
    
    top_n_positions = CONFIG["TOP_N_POSITIONS"]
    highlight_threshold = CONFIG["HIGHLIGHT_THRESHOLD"]

    logger.info(f"🚀 TOP {top_n_positions} LONG POSITIONS CLOSEST TO LIQUIDATION 📈")
    print(f"{Fore.GREEN}{Style.BRIGHT}{'-'*80}")
    if not risky_longs.empty:
        running_total_value = 0.0
        highest_distance = 0.0
        last_pct_threshold = 0
        for i, (_, row) in enumerate(risky_longs.head(top_n_positions).iterrows(), 1):
            highlight = row['position_value'] > highlight_threshold
            running_total_value += row['position_value']
            highest_distance = max(highest_distance, row['distance_to_liq_pct'])
            usdc_balance = get_spot_usdc_balance(row['address']) if i <= 4 else 0.0
            
            cp = row['current_price']
            lp = row['liquidation_price']
            dist_pct = row['distance_to_liq_pct']

            display_text_parts = [
                f"{Fore.GREEN if not highlight else ''}#{i} {Fore.YELLOW if not highlight else ''}{row['coin']} {Fore.GREEN if not highlight else ''}${row['position_value']:,.2f}",
                f"{Fore.BLUE if not highlight else ''}| Entry: ${row['entry_price']:,.2f}",
                f"{Fore.RED if not highlight else ''}| Liq: ${lp:,.2f}",
                f"{Fore.MAGENTA if not highlight else ''}| Current: ${cp:,.2f}",
                f"{Fore.MAGENTA if not highlight else ''}| Distance: {dist_pct:.2f}%",
                f"{Fore.CYAN if not highlight else ''}| Leverage: {row['leverage']}x"
            ]
            if i <= 4:
                display_text_parts.append(f"{Fore.MAGENTA if not highlight else ''}| 💰 USDC: ${usdc_balance:,.2f}")
            
            display_text = " ".join(display_text_parts)
            if highlight:
                plain_text = f"#{i} {row['coin']} ${row['position_value']:,.2f} | Entry: ${row['entry_price']:,.2f} | Liq: ${lp:,.2f} | Current: ${cp:,.2f} | Distance: {dist_pct:.2f}% | Leverage: {row['leverage']}x"
                if i <= 4:
                    plain_text += f" | 💰 USDC: ${usdc_balance:,.2f}"
                display_text = colored(plain_text, 'black', 'on_yellow')
            
            print(display_text)
            print(f"{Fore.CYAN if not highlight else ''}   Address: {row['address']}")
            
            if i % 10 == 0:
                agg_display = f"📊 AGGREGATE (1-{i}): Total Long Positions: ${running_total_value:,.2f} | All Liquidated Within: {highest_distance:.2f}%"
                print(colored(agg_display, 'black', 'on_cyan'))
                print(f"{Fore.CYAN}{Style.BRIGHT}{'-'*80}")
            
            current_pct_threshold = int(dist_pct / 2) * 2
            if current_pct_threshold > last_pct_threshold:
                pct_agg_display = f"📊 LIQUIDATION THRESHOLD 0-{current_pct_threshold}%: Total Long Value: ${running_total_value:,.2f}"
                print(colored(pct_agg_display, 'white', 'on_blue'))
                last_pct_threshold = current_pct_threshold
    else:
        logger.info("No long positions with liquidation prices found for risk metrics.")

    logger.info(f"💥 TOP {top_n_positions} SHORT POSITIONS CLOSEST TO LIQUIDATION 📉")
    print(f"{Fore.RED}{Style.BRIGHT}{'-'*80}")
    if not risky_shorts.empty:
        running_total_value = 0.0
        highest_distance = 0.0
        last_pct_threshold = 0
        for i, (_, row) in enumerate(risky_shorts.head(top_n_positions).iterrows(), 1):
            highlight = row['position_value'] > highlight_threshold
            running_total_value += row['position_value']
            highest_distance = max(highest_distance, row['distance_to_liq_pct'])
            usdc_balance = get_spot_usdc_balance(row['address']) if i <= 4 else 0.0

            cp = row['current_price']
            lp = row['liquidation_price']
            dist_pct = row['distance_to_liq_pct']

            display_text_parts = [
                f"{Fore.RED if not highlight else ''}#{i} {Fore.YELLOW if not highlight else ''}{row['coin']} {Fore.RED if not highlight else ''}${row['position_value']:,.2f}",
                f"{Fore.BLUE if not highlight else ''}| Entry: ${row['entry_price']:,.2f}",
                f"{Fore.RED if not highlight else ''}| Liq: ${lp:,.2f}",
                f"{Fore.MAGENTA if not highlight else ''}| Current: ${cp:,.2f}",
                f"{Fore.MAGENTA if not highlight else ''}| Distance: {dist_pct:.2f}%",
                f"{Fore.CYAN if not highlight else ''}| Leverage: {row['leverage']}x"
            ]
            if i <= 4:
                display_text_parts.append(f"{Fore.MAGENTA if not highlight else ''}| 💰 USDC: ${usdc_balance:,.2f}")

            display_text = " ".join(display_text_parts)
            if highlight:
                plain_text = f"#{i} {row['coin']} ${row['position_value']:,.2f} | Entry: ${row['entry_price']:,.2f} | Liq: ${lp:,.2f} | Current: ${cp:,.2f} | Distance: {dist_pct:.2f}% | Leverage: {row['leverage']}x"
                if i <= 4:
                    plain_text += f" | 💰 USDC: ${usdc_balance:,.2f}"
                display_text = colored(plain_text, 'black', 'on_yellow')
            
            print(display_text)
            print(f"{Fore.CYAN if not highlight else ''}   Address: {row['address']}")

            if i % 10 == 0:
                agg_display = f"📊 AGGREGATE (1-{i}): Total Short Positions: ${running_total_value:,.2f} | All Liquidated Within: {highest_distance:.2f}%"
                print(colored(agg_display, 'black', 'on_cyan'))
                print(f"{Fore.CYAN}{Style.BRIGHT}{'-'*80}")

            current_pct_threshold = int(dist_pct / 2) * 2
            if current_pct_threshold > last_pct_threshold:
                pct_agg_display = f"📊 LIQUIDATION THRESHOLD 0-{current_pct_threshold}%: Total Short Value: ${running_total_value:,.2f}"
                print(colored(pct_agg_display, 'white', 'on_blue'))
                last_pct_threshold = current_pct_threshold
    else:
        logger.info("No short positions with liquidation prices found for risk metrics.")
    
    return risky_longs.head(top_n_positions) if not risky_longs.empty else pd.DataFrame(), \
           risky_shorts.head(top_n_positions) if not risky_shorts.empty else pd.DataFrame(), \
           current_prices

def save_liquidation_risk_to_csv(risky_longs_df: pd.DataFrame | None, risky_shorts_df: pd.DataFrame | None):
    """
    Save positions closest to liquidation to a CSV file
    """
    if (risky_longs_df is None or risky_longs_df.empty) and \
       (risky_shorts_df is None or risky_shorts_df.empty):
        logger.info("🌙 Moon Dev says: No positions with liquidation data to save! 😢")
        return
    
    csv_format = CONFIG["CSV_FLOAT_FORMAT"]
    long_count = 0
    short_count = 0
    
    df_long_to_save = None
    if risky_longs_df is not None and not risky_longs_df.empty:
        df_long_to_save = risky_longs_df.copy()
        df_long_to_save['direction'] = 'LONG'
        longs_file = DATA_PATH / CONFIG["LIQ_CLOSEST_LONG_CSV"]
        try:
            df_long_to_save.to_csv(longs_file, index=False, float_format=csv_format)
            long_count = len(df_long_to_save)
            logger.info(f"Saved {long_count} long liquidation risk positions to {longs_file}")
        except Exception as e:
            logger.error(f"Error saving long liquidation risk CSV to {longs_file}: {e}", exc_info=True)
    
    df_short_to_save = None
    if risky_shorts_df is not None and not risky_shorts_df.empty:
        df_short_to_save = risky_shorts_df.copy()
        df_short_to_save['direction'] = 'SHORT'
        shorts_file = DATA_PATH / CONFIG["LIQ_CLOSEST_SHORT_CSV"]
        try:
            df_short_to_save.to_csv(shorts_file, index=False, float_format=csv_format)
            short_count = len(df_short_to_save)
            logger.info(f"Saved {short_count} short liquidation risk positions to {shorts_file}")
        except Exception as e:
            logger.error(f"Error saving short liquidation risk CSV to {shorts_file}: {e}", exc_info=True)
    
    combined_dfs_list = []
    if df_long_to_save is not None:
        combined_dfs_list.append(df_long_to_save)
    if df_short_to_save is not None:
        combined_dfs_list.append(df_short_to_save)

    if combined_dfs_list:
        combined_df = pd.concat(combined_dfs_list)
        if not combined_df.empty and 'distance_to_liq_pct' in combined_df.columns:
            combined_df = combined_df.sort_values('distance_to_liq_pct')
        elif not combined_df.empty:
            logger.warning("Combined liquidation risk DataFrame is missing 'distance_to_liq_pct' column for sorting.")

        combined_file = DATA_PATH / CONFIG["LIQ_CLOSEST_COMBINED_CSV"]
        try:
            combined_df.to_csv(combined_file, index=False, float_format=csv_format)
            logger.info(f"🌙 Moon Dev says: Saved {long_count} long and {short_count} short (total {len(combined_df)}) positions closest to liquidation to {combined_file} 🚀")
        except Exception as e:
            logger.error(f"Error saving combined liquidation risk CSV to {combined_file}: {e}", exc_info=True)
    elif long_count > 0 or short_count > 0: # One was saved but not the other, and combined_dfs_list remained empty
        logger.info("Only one type (long/short) of liquidation risk positions was saved. No combined file generated.")

def save_top_whale_positions_to_csv(longs_df: pd.DataFrame | None, shorts_df: pd.DataFrame | None):
    """
    Save top whale positions to a CSV file
    """
    if (longs_df is None or longs_df.empty) and \
       (shorts_df is None or shorts_df.empty):
        logger.info(f"🌙 Moon Dev says: No top whale positions to save! 😢")
        return

    csv_format = CONFIG["CSV_FLOAT_FORMAT"]
    long_count = 0
    short_count = 0

    df_long_to_save = None
    if longs_df is not None and not longs_df.empty:
        df_long_to_save = longs_df.copy()
        df_long_to_save['direction'] = 'LONG'
        longs_file = DATA_PATH / CONFIG["TOP_WHALE_LONG_CSV"]
        try:
            df_long_to_save.to_csv(longs_file, index=False, float_format=csv_format)
            long_count = len(df_long_to_save)
            logger.info(f"Saved {long_count} top whale long positions to {longs_file}")
        except Exception as e:
            logger.error(f"Error saving top whale long positions CSV to {longs_file}: {e}", exc_info=True)
    
    df_short_to_save = None
    if shorts_df is not None and not shorts_df.empty:
        df_short_to_save = shorts_df.copy()
        df_short_to_save['direction'] = 'SHORT'
        shorts_file = DATA_PATH / CONFIG["TOP_WHALE_SHORT_CSV"]
        try:
            df_short_to_save.to_csv(shorts_file, index=False, float_format=csv_format)
            short_count = len(df_short_to_save)
            logger.info(f"Saved {short_count} top whale short positions to {shorts_file}")
        except Exception as e:
            logger.error(f"Error saving top whale short positions CSV to {shorts_file}: {e}", exc_info=True)
    
    combined_dfs_list = []
    if df_long_to_save is not None:
        combined_dfs_list.append(df_long_to_save)
    if df_short_to_save is not None:
        combined_dfs_list.append(df_short_to_save)
    
    if combined_dfs_list:
        combined_df = pd.concat(combined_dfs_list)
        if not combined_df.empty and 'position_value' in combined_df.columns:
             combined_df = combined_df.sort_values('position_value', ascending=False)
        elif not combined_df.empty:
            logger.warning("Combined top whale DataFrame is missing 'position_value' column for sorting.")

        combined_file = DATA_PATH / CONFIG["TOP_WHALE_COMBINED_CSV"]
        try:
            combined_df.to_csv(combined_file, index=False, float_format=csv_format)
            logger.info(f"🌙 Moon Dev says: Saved {len(combined_df)} combined top whale positions to {combined_file} 🚀")
        except Exception as e:
            logger.error(f"Error saving combined top whale positions CSV to {combined_file}: {e}", exc_info=True)
    elif long_count > 0 or short_count > 0:
        logger.info("Only one type (long/short) of top whale positions was saved. No combined file generated.")

def process_positions(df, coin_filter=None):
    """
    Process the position data into a more usable format, filtering positions below min value
    and optionally by coin
    """
    if df is None or df.empty:
        print(f"{Fore.RED}No positions data to process!")
        return pd.DataFrame()
    
    print(f"{Fore.CYAN}🔍 Processing {len(df)} positions...")
    
    # Filter positions below minimum value threshold
    filtered_df = df[df['position_value'] >= CONFIG["MIN_POSITION_VALUE"]].copy()
    
    # Make sure numeric columns are the right type
    numeric_cols = ['entry_price', 'position_value', 'unrealized_pnl', 'liquidation_price', 'leverage']
    for col in numeric_cols:
        if col in filtered_df.columns:
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
    
    # Convert boolean columns if needed
    if 'is_long' in filtered_df.columns and filtered_df['is_long'].dtype != bool:
        filtered_df['is_long'] = filtered_df['is_long'].map({'True': True, 'False': False}) 
        # If it's stored as strings like 'True' and 'False'
    
    # Validate position types for positions with valid liquidation prices
    valid_liq_df = filtered_df[filtered_df['liquidation_price'] > 0].copy()
    if not valid_liq_df.empty:
        # Verify if the position type matches the relationship between entry and liquidation price
        valid_liq_df['position_type_verified'] = np.where(
            valid_liq_df['is_long'],
            valid_liq_df['liquidation_price'] < valid_liq_df['entry_price'],  # Long should have liq price < entry
            valid_liq_df['liquidation_price'] > valid_liq_df['entry_price']   # Short should have liq price > entry
        )
        
        # Count inconsistencies
        inconsistent_positions = valid_liq_df[~valid_liq_df['position_type_verified']]
        
        if len(inconsistent_positions) > 0:
            print(f"{Fore.RED}⚠️ WARNING: Found {len(inconsistent_positions)} positions with inconsistent position types!")
            print(f"{Fore.YELLOW}🔧 Correcting position types based on liquidation vs. entry price relationships...")
            
            # Create a corrected position type column based on liquidation vs entry price
            valid_liq_df['is_long_corrected'] = valid_liq_df['liquidation_price'] < valid_liq_df['entry_price']
            
            # Update the is_long column with corrected values
            valid_liq_df['is_long'] = valid_liq_df['is_long_corrected']
            
            # Update the filtered dataframe with corrected position types
            filtered_df.loc[valid_liq_df.index, 'is_long'] = valid_liq_df['is_long']
            
            print(f"{Fore.GREEN}✅ Position types corrected!")
    
    # Filter by coin if specified
    if coin_filter:
        coin_filter = coin_filter.upper()  # Convert to uppercase for case-insensitive matching
        filtered_df = filtered_df[filtered_df['coin'] == coin_filter]
        print(f"{Fore.MAGENTA}🪙 Filtering for {coin_filter} positions only")
    
    print(f"{Fore.GREEN}✅ Processed {len(filtered_df)} positions after filtering (min value: ${CONFIG['MIN_POSITION_VALUE']:,})")
    return filtered_df

def save_positions_to_csv(df, current_prices=None, quiet=False):
    """
    Save all positions to a CSV file and create aggregated views
    """
    if df is None or df.empty:
        print(f"{Fore.RED}🌙 Moon Dev says: No positions found to save! 😢")
        return None, None
    
    # Format numeric columns
    numeric_cols = ['entry_price', 'position_value', 'unrealized_pnl', 'liquidation_price', 'leverage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Save all positions
    positions_file = DATA_PATH / CONFIG["ALL_POSITIONS_CSV"]
    df.to_csv(positions_file, index=False, float_format=CONFIG["CSV_FLOAT_FORMAT"])
    
    # Create and save aggregated view
    agg_df = df.groupby(['coin', 'is_long']).agg({
        'position_value': 'sum',
        'unrealized_pnl': 'sum',
        'address': 'count',
        'leverage': 'mean',  # Average leverage
        'liquidation_price': lambda x: np.nan if all(pd.isna(x)) else np.nanmean(x)  # Average liquidation price, ignoring NaN values
    }).reset_index()
    
    # Add direction and rename columns
    agg_df['direction'] = agg_df['is_long'].apply(lambda x: 'LONG' if x else 'SHORT')
    agg_df = agg_df.rename(columns={
        'address': 'num_traders',
        'position_value': 'total_value',
        'unrealized_pnl': 'total_pnl',
        'leverage': 'avg_leverage',
        'liquidation_price': 'avg_liquidation_price'
    })
    
    # Calculate average value per trader
    agg_df['avg_value_per_trader'] = agg_df['total_value'] / agg_df['num_traders']
    
    # Sort by total value
    agg_df = agg_df.sort_values('total_value', ascending=False)
    
    # Save aggregated view
    agg_file = DATA_PATH / CONFIG["AGGREGATED_POSITIONS_SUMMARY_CSV"]
    agg_df.to_csv(agg_file, index=False, float_format=CONFIG["CSV_FLOAT_FORMAT"])
    
    # Display summaries (for terminal display only, not affecting CSV output)
    print(f"\n{Fore.CYAN}{'='*30} POSITION SUMMARY {'='*30}")
    display_cols = ['coin', 'direction', 'total_value', 'num_traders', 'avg_value_per_trader', 'avg_leverage']
    
    # Temporarily format numbers with commas for display only
    with pd.option_context('display.float_format', '{:,.2f}'.format):
        print(f"{Fore.WHITE}{agg_df[display_cols]}")
        
        print(f"\n{Fore.GREEN}🔝 TOP LONG POSITIONS (AGGREGATED):")
        print(f"{Fore.GREEN}{agg_df[agg_df['is_long']][display_cols].head()}")
        
        print(f"\n{Fore.RED}🔝 TOP SHORT POSITIONS (AGGREGATED):")
        print(f"{Fore.RED}{agg_df[~agg_df['is_long']][display_cols].head()}")
    
    # Display top individual positions and get the dataframes
    longs_df, shorts_df = display_top_individual_positions(df)
    
    # Save top whale positions to CSV
    save_top_whale_positions_to_csv(longs_df, shorts_df)
    
    # Display risk metrics and get risky positions
    risky_longs_df, risky_shorts_df, fetched_prices = display_risk_metrics(df)
    
    # Save liquidation risk positions to CSV
    save_liquidation_risk_to_csv(risky_longs_df, risky_shorts_df)

    # Check if we have current prices from the risk metrics function
    if current_prices is None:
        current_prices = fetched_prices
    
    # Check if we still need to get prices (should never happen as display_risk_metrics has already fetched them)
    if current_prices is None:
        # Fetch current prices for tokens in TOKENS_TO_ANALYZE only if needed
        unique_coins = df[df['coin'].isin(CONFIG["TOKENS_TO_ANALYZE"])]['coin'].unique()
        current_prices = {coin: n.get_current_price(coin) for coin in unique_coins}

    # Use current prices for liquidation impact analysis
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}{'='*20} 💥 LIQUIDATION IMPACT FOR 3% PRICE MOVE 💥 {'='*20}")
    print(f"{Fore.CYAN}{'='*80}")

    # Initialize dictionaries to track liquidation values
    total_long_liquidations = {}
    total_short_liquidations = {}
    all_long_liquidations = 0
    all_short_liquidations = 0
    
    for coin in CONFIG["TOKENS_TO_ANALYZE"]:
        if coin not in current_prices:
            continue
            
        # Filter positions for the current coin
        coin_positions = df[df['coin'] == coin].copy()
        if coin_positions.empty:
            continue

        # Add current price to the coin positions DataFrame
        current_price = current_prices[coin]
        coin_positions['current_price'] = current_price
        
        # Calculate price levels for 3% moves
        price_3pct_down = current_price * 0.97
        price_3pct_up = current_price * 1.03

        # Calculate potential liquidations for long positions
        long_liquidations = coin_positions[(coin_positions['is_long']) &
                                         (coin_positions['liquidation_price'] >= price_3pct_down) &
                                         (coin_positions['liquidation_price'] <= current_price)]
        
        total_long_liquidation_value = long_liquidations['position_value'].sum()
        
        # Calculate potential liquidations for short positions
        short_liquidations = coin_positions[(~coin_positions['is_long']) &
                                          (coin_positions['liquidation_price'] <= price_3pct_up) &
                                          (coin_positions['liquidation_price'] >= current_price)]
        
        total_short_liquidation_value = short_liquidations['position_value'].sum()

        # Store liquidation values in dictionary
        total_long_liquidations[coin] = total_long_liquidation_value
        total_short_liquidations[coin] = total_short_liquidation_value
        
        # Add to total liquidations
        all_long_liquidations += total_long_liquidation_value
        all_short_liquidations += total_short_liquidation_value

        # Display results (only if not in quiet mode)
        if not quiet:
            print(f"{Fore.GREEN}{coin} Long Liquidations (3% move DOWN to ${price_3pct_down:,.2f}): ${total_long_liquidation_value:,.2f}")
            print(f"{Fore.RED}{coin} Short Liquidations (3% move UP to ${price_3pct_up:,.2f}): ${total_short_liquidation_value:,.2f}")

    # Display summary of total liquidations
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}{'='*25} 💰 TOTAL LIQUIDATION SUMMARY 💰 {'='*25}")
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.GREEN}Total Long Liquidations (3% move DOWN): ${all_long_liquidations:,.2f}")
    print(f"{Fore.RED}Total Short Liquidations (3% move UP): ${all_short_liquidations:,.2f}")
    
    # Generate trading recommendations based on liquidation imbalance
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}{'='*20} 🚀 MARKET DIRECTION (NFA) 🚀 {'='*20}")
    print(f"{Fore.CYAN}{'='*80}")
    
    # Overall market direction
    if all_long_liquidations > all_short_liquidations:
        direction = f"MARKET DIRECTION (NFA): SHORT THE MARKET (${all_long_liquidations:,.2f} long liquidations at risk within a 3% move of current price)"
        print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}{direction}{Style.RESET_ALL}")
    else:
        direction = f"MARKET DIRECTION (NFA): LONG THE MARKET (${all_short_liquidations:,.2f} short liquidations at risk within a 3% move of current price)"
        print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}{direction}{Style.RESET_ALL}")
    
    # Individual coin directions
    print(f"\n{Fore.CYAN}{'='*30} INDIVIDUAL COIN DIRECTION (NFA) {'='*30}")
    
    # Sort coins by liquidation imbalance (largest difference first)
    liquidation_imbalance = {}
    for coin in total_long_liquidations.keys():
        if coin in total_short_liquidations:
            liquidation_imbalance[coin] = abs(total_long_liquidations[coin] - total_short_liquidations[coin])
    
    sorted_coins = sorted(liquidation_imbalance.keys(), key=lambda x: liquidation_imbalance[x], reverse=True)
    
    for coin in sorted_coins:
        long_liq = total_long_liquidations[coin]
        short_liq = total_short_liquidations[coin]
        
        # Only show directions for coins with significant liquidation risk
        if long_liq < 10000 and short_liq < 10000:
            continue
            
        if long_liq > short_liq:
            rec = f"{coin}: SHORT (${long_liq:,.2f} long liquidations vs ${short_liq:,.2f} short within a 3% move)"
            print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}{rec}{Style.RESET_ALL}")
        else:
            rec = f"{coin}: LONG (${short_liq:,.2f} short liquidations vs ${long_liq:,.2f} long within a 3% move)"
            print(f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT}{rec}{Style.RESET_ALL}")
    
    print(f"\n{Fore.MAGENTA}💡 Trading strategy: Target coins with largest liquidation imbalance for potential cascade liquidations")
    print(f"{Fore.YELLOW}⚠️ NFA: This analysis is NOT financial advice. Always do your own research! 📚")
    
    # Create liquidation thresholds table
    create_liquidation_thresholds_table(df, current_prices, quiet)

    # Combine the save notifications and execution time in one summary line
    long_count = len(longs_df) if longs_df is not None else 0
    short_count = len(shorts_df) if shorts_df is not None else 0
    print(f"{Fore.GREEN}🌙 Moon Dev saved {long_count} long and {short_count} short positions to CSV files in {DATA_PATH} 📊")
    
    return df, agg_df

def display_highlighted_positions(df):
    """
    Display a table of highlighted positions (value > $2M) from the top 30 positions closest to liquidation
    """
    if df is None or df.empty:
        return
        
    # Filter for positions with value > $2M and only for tokens we analyze
    highlighted_df = df[
        (df['position_value'] > CONFIG['HIGHLIGHT_THRESHOLD']) & 
        (df['coin'].isin(CONFIG["TOKENS_TO_ANALYZE"]))
    ].copy()
    
    if highlighted_df.empty:
        return
        
    # Get current prices only for tokens we have highlighted positions for
    unique_coins = highlighted_df['coin'].unique()
    current_prices = {coin: n.get_current_price(coin) for coin in unique_coins}
    
    # Add current price to the DataFrame
    highlighted_df['current_price'] = highlighted_df['coin'].map(current_prices)
    
    # Calculate distance to liquidation percentage
    highlighted_df['distance_to_liq_pct'] = np.where(
        highlighted_df['is_long'],
        abs((highlighted_df['current_price'] - highlighted_df['liquidation_price']) / highlighted_df['current_price'] * 100),
        abs((highlighted_df['liquidation_price'] - highlighted_df['current_price']) / highlighted_df['current_price'] * 100)
    )
    
    # Split into longs and shorts
    highlighted_longs = highlighted_df[highlighted_df['is_long']].sort_values('distance_to_liq_pct')
    highlighted_shorts = highlighted_df[~highlighted_df['is_long']].sort_values('distance_to_liq_pct')
    
    # Get top 2 for each
    top_longs = highlighted_longs.head(2)
    top_shorts = highlighted_shorts.head(2)
    
    # Only proceed if we have any highlighted positions
    if top_longs.empty and top_shorts.empty:
        return
        
    print(f"\n{Fore.CYAN}{'='*140}")
    print(f"{Fore.CYAN}{'='*15} 🚨 POSITIONS CLOSEST TO LIQUIDATION (>${CONFIG['HIGHLIGHT_THRESHOLD']:,}) 🚨 {'='*15}")
    print(f"{Fore.CYAN}{'='*140}")
    
    # Create header with fixed widths
    header = (f"{Fore.YELLOW}{'Position':<10} | {'Coin':<4} | {'Value':>17} | "
             f"{'Entry':>12} | {'Liq':>12} | {'Distance':>9} | {'Leverage':>8} | {'Address':>42} | {'USDC':>12}")
    separator = f"{Fore.CYAN}{'-'*10}-+-{'-'*4}-+-{'-'*17}-+-{'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*8}-+-{'-'*42}-+-{'-'*12}"
    
    print(header)
    print(separator)
    
    # Display long positions
    for i, (_, row) in enumerate(top_longs.iterrows(), 1):
        usdc_balance = get_spot_usdc_balance(row['address'])
        print(f"{Fore.GREEN}{'LONG #' + str(i):<10} | "
              f"{Fore.YELLOW}{row['coin']:<4} | "
              f"{Fore.GREEN}${row['position_value']:>15,.2f} | "
              f"{Fore.BLUE}${row['entry_price']:>10,.2f} | "
              f"{Fore.RED}${row['liquidation_price']:>10,.2f} | "
              f"{Fore.MAGENTA}{row['distance_to_liq_pct']:>7.2f}% | "
              f"{Fore.CYAN}{row['leverage']:>3}x".ljust(8) + " | "
              f"{Fore.BLUE}{row['address']} | "
              f"{Fore.MAGENTA}${usdc_balance:>10,.2f}")
    
    # Display short positions
    for i, (_, row) in enumerate(top_shorts.iterrows(), 1):
        usdc_balance = get_spot_usdc_balance(row['address'])
        print(f"{Fore.RED}{'SHORT #' + str(i):<10} | "
              f"{Fore.YELLOW}{row['coin']:<4} | "
              f"{Fore.RED}${row['position_value']:>15,.2f} | "
              f"{Fore.BLUE}${row['entry_price']:>10,.2f} | "
              f"{Fore.RED}${row['liquidation_price']:>10,.2f} | "
              f"{Fore.MAGENTA}{row['distance_to_liq_pct']:>7.2f}% | "
              f"{Fore.CYAN}{row['leverage']:>3}x".ljust(8) + " | "
              f"{Fore.BLUE}{row['address']} | "
              f"{Fore.MAGENTA}${usdc_balance:>10,.2f}")
    
    print(f"{Fore.CYAN}{'='*140}")

def display_market_metrics():
    """
    Display market metrics (funding rates) in a compact format
    """
    try:
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}{'='*15} 📊 MARKET METRICS 📊 {'='*15}")
        print(f"{Fore.CYAN}{'='*80}")

        # Get Binance funding rates directly
        binance_funding_rates = {}
        for token in CONFIG["TOKENS_TO_ANALYZE"]:
            try:
                url = f"{CONFIG['BINANCE_FAPI_URL']}?symbol={token}USDT"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Get the raw funding rate from the API and calculate annualized rate
                raw_funding_rate = float(data['lastFundingRate'])
                funding_rate_pct = raw_funding_rate * 100
                yearly_rate = funding_rate_pct * 3 * 365
                binance_funding_rates[token] = yearly_rate
                
            except requests.exceptions.Timeout:
                logger.error(f"❌ Timeout fetching Binance funding rate for {token}")
                binance_funding_rates[token] = None
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ HTTP Error fetching Binance funding rate for {token}: {str(e)}")
                binance_funding_rates[token] = None
            except (ValueError, KeyError, TypeError) as e:
                logger.error(f"❌ Data parsing error for Binance funding rate {token}: {str(e)}. Response data: {data if 'data' in locals() else 'N/A'}")
                binance_funding_rates[token] = None
            except Exception as e:
                logger.error(f"❌ Unexpected error fetching Binance funding rate for {token}: {str(e)}", exc_info=True)
                binance_funding_rates[token] = None

        # Get Hyperliquid funding data
        try:
            url = CONFIG["HYPERLIQUID_API_URL"]
            headers = {"Content-Type": "application/json"}
            body = {"type": "metaAndAssetCtxs"}
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            hl_data = response.json()
            
            # Create a mapping of coins to their funding rates
            hl_funding_rates = {}
            for i, asset in enumerate(hl_data[0]['universe']):
                if asset['name'] in CONFIG["TOKENS_TO_ANALYZE"]:
                    funding_rate = float(hl_data[1][i]['funding'])
                    # Convert hourly rate to yearly (24 hours * 365 days)
                    yearly_rate = funding_rate * 24 * 365 * 100  # Convert to percentage
                    hl_funding_rates[asset['name']] = yearly_rate
        except Exception as e:
            print(f"{Fore.RED}❌ Error fetching Hyperliquid funding rates: {str(e)}")
            hl_funding_rates = {}
        
        # Create header row
        header = f"{Fore.CYAN}{'Metric':<12} | "
        for token in CONFIG["TOKENS_TO_ANALYZE"]:
            header += f"{Fore.WHITE}{token:<25} | "
        
        # Create separator
        separator = f"{Fore.CYAN}{'-'*12} | " + f"{'-'*25} | " * len(CONFIG["TOKENS_TO_ANALYZE"])
        
        # Create Binance funding rate row
        b_funding_row = f"{Fore.YELLOW}{'BNB Funding':<12} | "
        for token in CONFIG["TOKENS_TO_ANALYZE"]:
            if token in binance_funding_rates and binance_funding_rates[token] is not None:
                funding_value = f"{binance_funding_rates[token]:+.2f}%"
                # Color code funding rates
                funding_color = Fore.GREEN if binance_funding_rates[token] < 0 else Fore.RED
                b_funding_row += f"{funding_color}{funding_value:<25} | "
            else:
                b_funding_row += f"{Fore.RED}{'N/A':<25} | "

        # Create Hyperliquid funding rate row
        hl_funding_row = f"{Fore.YELLOW}{'HL Funding':<12} | "
        for token in CONFIG["TOKENS_TO_ANALYZE"]:
            if token in hl_funding_rates:
                funding_value = f"{hl_funding_rates[token]:+.2f}%"
                # Color code funding rates
                funding_color = Fore.GREEN if hl_funding_rates[token] < 0 else Fore.RED
                hl_funding_row += f"{funding_color}{funding_value:<25} | "
            else:
                hl_funding_row += f"{Fore.RED}{'N/A':<25} | "
        
        # Print the table
        print(f"\n{Fore.CYAN}{'='*120}")
        print(header)
        print(separator)
        print(b_funding_row)
        print(hl_funding_row)
        print(f"{Fore.CYAN}{'='*120}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error displaying market metrics: {str(e)}")
        print(f"{Fore.RED}📋 Stack trace:\n{traceback.format_exc()}")

def create_liquidation_thresholds_table(df, current_prices, quiet=False):
    """
    Create and display a table of liquidation thresholds at different price move percentages
    """
    # Display market metrics first
    display_market_metrics()
    
    # Display highlighted positions table second
    display_highlighted_positions(df)
    
    # Display liquidation thresholds table last
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}{'='*15} 📊 PENDING LIQUIDATIONS BY PERCENTAGE MOVE 📊 {'='*15}")
    print(f"{Fore.CYAN}{'='*80}")
    
    # Define thresholds to analyze - small ranges first, then larger ranges
    small_thresholds = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    large_thresholds = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    all_thresholds = small_thresholds + large_thresholds
    
    # Initialize data structures for the table
    table_data = {
        'Threshold': [f"0-{t}%" for t in all_thresholds],
        'Long Liquidations ($)': [],
        'Short Liquidations ($)': [],
        'Total Liquidations ($)': [],
        'Imbalance (%)': [],
        'Direction': []
    }
    
    # Calculate liquidations for each threshold
    for threshold in all_thresholds:
        total_long_liquidations = 0
        total_short_liquidations = 0
        
        for coin in CONFIG["TOKENS_TO_ANALYZE"]:
            if coin not in current_prices:
                continue
                
            # Filter positions for the current coin
            coin_positions = df[df['coin'] == coin].copy()
            if coin_positions.empty:
                continue
                
            # Get current price and calculate threshold prices
            current_price = current_prices[coin]
            price_down = current_price * (1 - threshold/100)
            price_up = current_price * (1 + threshold/100)
            
            # Calculate liquidations for long positions
            long_liquidations = coin_positions[(coin_positions['is_long']) &
                                             (coin_positions['liquidation_price'] >= price_down) &
                                             (coin_positions['liquidation_price'] <= current_price)]
            
            long_value = long_liquidations['position_value'].sum()
            
            # Calculate liquidations for short positions
            short_liquidations = coin_positions[(~coin_positions['is_long']) &
                                              (coin_positions['liquidation_price'] <= price_up) &
                                              (coin_positions['liquidation_price'] >= current_price)]
            
            short_value = short_liquidations['position_value'].sum()
            
            # Add to totals
            total_long_liquidations += long_value
            total_short_liquidations += short_value
        
        # Add data to table
        total_liquidations = total_long_liquidations + total_short_liquidations
        
        # Calculate imbalance as percentage
        if total_liquidations > 0:
            imbalance_pct = ((total_long_liquidations - total_short_liquidations) / total_liquidations) * 100
        else:
            imbalance_pct = 0
        
        # Determine direction based on imbalance percentage
        if total_liquidations == 0:
            direction = ""  # Empty string for no liquidations
        elif abs(imbalance_pct) < 5:  # Less than 5% imbalance
            direction = "NEUTRAL"
        elif imbalance_pct > 0:
            direction = "SHORT"
        else:
            direction = "LONG"
        
        table_data['Long Liquidations ($)'].append(total_long_liquidations)
        table_data['Short Liquidations ($)'].append(total_short_liquidations)
        table_data['Total Liquidations ($)'].append(total_liquidations)
        table_data['Imbalance (%)'].append(imbalance_pct)
        table_data['Direction'].append(direction)
    
    # Create and display the overall table
    table_df = pd.DataFrame(table_data)
    
    # Format numeric columns with commas and percentages
    for col in ['Long Liquidations ($)', 'Short Liquidations ($)', 'Total Liquidations ($)']:
        table_df[col] = table_df[col].apply(lambda x: f"${x:,.2f}")
    
    # Format imbalance as percentage with sign
    table_df['Imbalance (%)'] = table_df['Imbalance (%)'].apply(lambda x: f"{x:+.2f}%")
    
    # Create a styled string representation of the table
    styled_table = f"\n{Fore.CYAN}{'='*120}\n"
    styled_table += f"{Fore.YELLOW}{'Threshold':<12} | {'Long Liquidations':<25} | {'Short Liquidations':<25} | {'Total Liquidations':<25} | {'Imbalance':<15} | {'Direction':<15}\n"
    styled_table += f"{Fore.CYAN}{'-'*120}\n"
    
    for i, row in table_df.iterrows():
        direction = row['Direction']
        if direction:  # Only set color if there's a direction
            direction_color = Fore.GREEN if direction == "LONG" else (Fore.RED if direction == "SHORT" else Fore.YELLOW)
        else:
            direction_color = Fore.WHITE  # Default color for empty direction
        
        # Determine imbalance color based on value
        imbalance_value = float(row['Imbalance (%)'].strip('%+'))
        imbalance_color = Fore.GREEN if imbalance_value < 0 else (Fore.RED if imbalance_value > 0 else Fore.YELLOW)
        
        styled_table += f"{Fore.WHITE}{row['Threshold']:<12} | "
        styled_table += f"{Fore.GREEN}{row['Long Liquidations ($)']:<25} | "
        styled_table += f"{Fore.RED}{row['Short Liquidations ($)']:<25} | "
        styled_table += f"{Fore.CYAN}{row['Total Liquidations ($)']:<25} | "
        styled_table += f"{imbalance_color}{row['Imbalance (%)']:<15} | "
        styled_table += f"{direction_color}{direction:<15}\n"
        
    styled_table += f"{Fore.CYAN}{'='*120}"
    
    print(styled_table)
    
    # Save the table to CSV
    table_file = DATA_PATH / CONFIG["LIQ_THRESHOLDS_TABLE_CSV"]
    table_df.to_csv(table_file, index=False)

def fetch_positions_from_api():
    """
    Fetch positions from Moon Dev API
    """
    
    try:
        # Initialize the API silently
        api = MoonDevAPI()
        
        # Get positions data from the API
        positions_df = api.get_positions_hlp()
        
        if positions_df is None or positions_df.empty:
            print(f"{Fore.YELLOW}⚠️ No positions data received from API, using mock data for testing!")
            
            # Create mock data for testing
            mock_data = {
                'address': [f'wallet_{i}' for i in range(50)],
                'coin': np.random.choice(['BTC', 'ETH', 'SOL', 'XRP'], 50),
                'is_long': np.random.choice([True, False], 50),
                'entry_price': np.random.uniform(10000, 60000, 50),
                'position_value': np.random.uniform(5000, 5000000, 50),
                'unrealized_pnl': np.random.uniform(-500000, 500000, 50),
                'liquidation_price': np.random.uniform(1000, 70000, 50),
                'leverage': np.random.uniform(1, 20, 50)
            }
            
            positions_df = pd.DataFrame(mock_data)
            
            # Ensure the liquidation prices make sense relative to entry prices
            for i, row in positions_df.iterrows():
                if row['is_long']:
                    positions_df.at[i, 'liquidation_price'] = row['entry_price'] * 0.8  # 20% below entry
                else:
                    positions_df.at[i, 'liquidation_price'] = row['entry_price'] * 1.2  # 20% above entry
            
            print(f"{Fore.GREEN}🌙 Moon Dev says: Created mock data with {len(positions_df)} positions for testing! 🚀")
            return positions_df
            
        return positions_df
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error fetching positions: {str(e)}")
        print(f"{Fore.RED}📋 Stack trace:\n{traceback.format_exc()}")
        
        # Create mock data in case of error
        print(f"{Fore.YELLOW}⚠️ Creating mock data due to error...")
        mock_data = {
            'address': [f'wallet_{i}' for i in range(50)],
            'coin': np.random.choice(['BTC', 'ETH', 'SOL', 'XRP'], 50),
            'is_long': np.random.choice([True, False], 50),
            'entry_price': np.random.uniform(10000, 60000, 50),
            'position_value': np.random.uniform(5000, 5000000, 50),
            'unrealized_pnl': np.random.uniform(-500000, 500000, 50),
            'liquidation_price': np.random.uniform(1000, 70000, 50),
            'leverage': np.random.uniform(1, 20, 50)
        }
        
        positions_df = pd.DataFrame(mock_data)
        
        # Ensure the liquidation prices make sense relative to entry prices
        for i, row in positions_df.iterrows():
            if row['is_long']:
                positions_df.at[i, 'liquidation_price'] = row['entry_price'] * 0.8  # 20% below entry
            else:
                positions_df.at[i, 'liquidation_price'] = row['entry_price'] * 1.2  # 20% above entry
        
        print(f"{Fore.GREEN}🌙 Moon Dev says: Created mock data with {len(positions_df)} positions for testing! 🚀")
        return positions_df

def fetch_aggregated_positions_from_api():
    """
    Fetch aggregated positions from Moon Dev API
    """
    try:
        # Initialize the API silently
        api = MoonDevAPI()
        
        # Get aggregated positions data from the API
        agg_positions_df = api.get_agg_positions_hlp()
        
        if agg_positions_df is None or agg_positions_df.empty:
            print(f"{Fore.YELLOW}⚠️ No aggregated positions data received from API, using mock data for testing!")
            
            # Create mock aggregated data
            mock_data = {
                'coin': ['BTC', 'ETH', 'SOL', 'XRP', 'BTC', 'ETH', 'SOL', 'XRP'],
                'is_long': [True, True, True, True, False, False, False, False],
                'total_value': np.random.uniform(1000000, 300000000, 8),
                'num_traders': np.random.randint(50, 500, 8),
                'avg_liquidation_price': [30000, 2000, 80, 0.5, 50000, 3500, 120, 0.8]
            }
            
            agg_positions_df = pd.DataFrame(mock_data)
            
            print(f"{Fore.GREEN}🌙 Moon Dev says: Created mock aggregated data with {len(agg_positions_df)} rows for testing! 🚀")
            return agg_positions_df
            
        return agg_positions_df
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error fetching aggregated positions: {str(e)}")
        print(f"{Fore.RED}📋 Stack trace:\n{traceback.format_exc()}")
        
        # Create mock aggregated data in case of error
        print(f"{Fore.YELLOW}⚠️ Creating mock aggregated data due to error...")
        mock_data = {
            'coin': ['BTC', 'ETH', 'SOL', 'XRP', 'BTC', 'ETH', 'SOL', 'XRP'],
            'is_long': [True, True, True, True, False, False, False, False],
            'total_value': np.random.uniform(1000000, 300000000, 8),
            'num_traders': np.random.randint(50, 500, 8),
            'avg_liquidation_price': [30000, 2000, 80, 0.5, 50000, 3500, 120, 0.8]
        }
        
        agg_positions_df = pd.DataFrame(mock_data)
        
        print(f"{Fore.GREEN}🌙 Moon Dev says: Created mock aggregated data with {len(agg_positions_df)} rows for testing! 🚀")
        return agg_positions_df

def bot():
    """Main function to run the position tracker (renamed from main to bot)"""
    # Use global configuration variables
    global CONFIG
    
    # Display Moon Dev banner
    print(MOON_DEV_BANNER)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="🌙 Moon Dev's Hyperliquid Whale Position Tracker (API Version)")
    parser.add_argument('--min-value', type=int, default=CONFIG["MIN_POSITION_VALUE"], 
                        help=f'Minimum position value to consider (default: ${CONFIG["MIN_POSITION_VALUE"]:,})')
    parser.add_argument('--top-n', type=int, default=CONFIG["TOP_N_POSITIONS"],
                        help=f'Number of top positions to display (default: {CONFIG["TOP_N_POSITIONS"]})')
    parser.add_argument('--agg-only', action='store_true',
                        help='Only show aggregated data (faster, less detailed)')
    parser.add_argument('--coin', type=str, default=None,
                        help='Filter positions by coin (e.g., BTC, ETH, SOL)')
    parser.add_argument('--verify-positions', action='store_true', default=True,
                        help='Verify and correct position types based on liquidation prices')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Reduce verbosity of output')
    parser.add_argument('--no-symbol-debug', action='store_true', default=True,
                        help='Disable printing of individual symbols during analysis')
    args = parser.parse_args()
    
    # Update configuration based on arguments
    CONFIG["MIN_POSITION_VALUE"] = args.min_value
    CONFIG["TOP_N_POSITIONS"] = args.top_n
    
    start_time = time.time()
    
    # Ensure data directory exists
    ensure_data_dir()
    
    # Fetch aggregated positions data first (this is faster)
    agg_df = fetch_aggregated_positions_from_api()
    
    if agg_df is not None:
        # Save aggregated positions
        agg_file = DATA_PATH / CONFIG["AGG_POSITIONS_API_CSV"]
        agg_df.to_csv(agg_file, index=False, float_format=CONFIG["CSV_FLOAT_FORMAT"])
        
        # Add direction column
        agg_df['direction'] = agg_df['is_long'].apply(lambda x: 'LONG' if x else 'SHORT')
        
        # Filter by coin if specified
        if args.coin:
            coin = args.coin.upper()
            agg_df = agg_df[agg_df['coin'] == coin]
        
        if not args.quiet:
            # Display aggregated summaries
            print(f"\n{Fore.CYAN}{'='*30} AGGREGATED POSITION SUMMARY {'='*30}")
            display_cols = ['coin', 'direction', 'total_value', 'num_traders', 'liquidation_price']
            
            # Temporarily format numbers with commas for display only
            with pd.option_context('display.float_format', '{:,.2f}'.format):
                print(f"{Fore.WHITE}{agg_df[display_cols]}")
                
                print(f"\n{Fore.GREEN}🔝 TOP LONG POSITIONS (AGGREGATED):")
                print(f"{Fore.GREEN}{agg_df[agg_df['is_long']][display_cols].head()}")
                
                print(f"\n{Fore.RED}🔝 TOP SHORT POSITIONS (AGGREGATED):")
                print(f"{Fore.RED}{agg_df[~agg_df['is_long']][display_cols].head()}")
    
    # If not only showing aggregated data, fetch and process detailed positions
    if not args.agg_only:
        # Fetch detailed positions data
        positions_df = fetch_positions_from_api()
        
        if positions_df is not None:
            # Process positions (filter by min value, etc.)
            processed_df = process_positions(positions_df, args.coin)
            
            if not processed_df.empty:
                # Display top individual positions and get the dataframes
                longs_df, shorts_df = display_top_individual_positions(processed_df)
                
                # Save top whale positions to CSV
                save_top_whale_positions_to_csv(longs_df, shorts_df)
                
                # Get risk metrics and current prices in one step
                risky_longs_df, risky_shorts_df, current_prices = display_risk_metrics(processed_df)
                
                # Save liquidation risk positions to CSV
                save_liquidation_risk_to_csv(risky_longs_df, risky_shorts_df)
                
                # Pass the already fetched current prices to save_positions_to_csv
                positions_df, _ = save_positions_to_csv(processed_df, current_prices, quiet=args.quiet)
            else:
                print(f"{Fore.RED}⚠️ No positions found after filtering! Try adjusting your filters.")
    
    # Calculate and display execution time
    execution_time = time.time() - start_time
    print(f"{Fore.CYAN}⏱️ Analysis completed in {execution_time:.2f} seconds")

if __name__ == "__main__":
    # Initial run
    bot()
    
    # Schedule the main function to run every minute
    schedule.every(CONFIG["SCHEDULE_INTERVAL_MINUTES"]).minutes.do(bot)
    
    while True:
        try:
            # Run pending scheduled tasks
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            print(f"{Fore.RED}Encountered an error: {e}")
            print(f"{Fore.RED}📋 Stack trace:\n{traceback.format_exc()}")
            # Wait before retrying to avoid rapid error logging
            time.sleep(10)
