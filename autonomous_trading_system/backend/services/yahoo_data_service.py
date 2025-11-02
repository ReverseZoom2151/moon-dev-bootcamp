import yfinance as yf
import pandas as pd
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from backend.core.config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YahooDataService:
    def __init__(self):
        self.save_dir = config.get('YAHOO_DATA_DIR', './yahoo_data')
        self.rate_limit_delay = config.get('RATE_LIMIT_DELAY', 1)
        os.makedirs(self.save_dir, exist_ok=True)
        self.symbols = config.get('YAHOO_SYMBOLS', [
            ('AAPL', 'stock'), ('GOOGL', 'stock'), ('MSFT', 'stock'), ('AMZN', 'stock'),
            ('EURUSD=X', 'forex'), ('GBPUSD=X', 'forex'), ('USDJPY=X', 'forex'), ('AUDUSD=X', 'forex'),
            ('ES=F', 'future'), ('NQ=F', 'future'), ('YM=F', 'future'), ('RTY=F', 'future'),
            ('GC=F', 'future'), ('SI=F', 'future'), ('CL=F', 'future'), ('NG=F', 'future'),
            ('BTC-USD', 'crypto'), ('ETH-USD', 'crypto'), ('XRP-USD', 'crypto'), ('LTC-USD', 'crypto'),
            ('BCH-USD', 'crypto'), ('ADA-USD', 'crypto'), ('DOT-USD', 'crypto'), ('SOL-USD', 'crypto'),
            ('BNB-USD', 'crypto'), ('DOGE-USD', 'crypto'), ('SHIB-USD', 'crypto'), ('AVAX-USD', 'crypto'),
            ('TRX-USD', 'crypto'), ('LINK-USD', 'crypto'), ('MATIC-USD', 'crypto'), ('UNI-USD', 'crypto'),
            ('ATOM-USD', 'crypto'), ('XLM-USD', 'crypto'), ('ETC-USD', 'crypto'), ('TON-USD', 'crypto'),
            ('ICP-USD', 'crypto'), ('HBAR-USD', 'crypto'), ('APT-USD', 'crypto'), ('ARB-USD', 'crypto'),
            ('NEAR-USD', 'crypto'), ('VET-USD', 'crypto'), ('ALGO-USD', 'crypto'), ('QNT-USD', 'crypto'),
            ('FIL-USD', 'crypto'), ('EOS-USD', 'crypto')
        ])
        self.default_start_date = config.get('DEFAULT_START_DATE', '2000-01-01')
        self.hourly_fetch_days = config.get('HOURLY_FETCH_DAYS', 728)

    def load_existing_data(self, filepath: str, index_col: str = 'Date') -> Optional[pd.DataFrame]:
        """Loads existing data from a CSV file, handling potential errors."""
        try:
            existing_data = pd.read_csv(filepath, index_col=index_col, parse_dates=True)
            if isinstance(existing_data.index, pd.DatetimeIndex):
                if existing_data.index.tz is not None:
                    existing_data.index = existing_data.index.tz_localize(None)
                logger.info(f"Loaded {len(existing_data)} rows from {filepath}. Last date: {existing_data.index[-1]}")
                return existing_data
            else:
                logger.warning(f"Index in {filepath} is not a DatetimeIndex after loading. Ignoring file.")
                return None
        except FileNotFoundError:
            logger.info(f"No existing file found at {filepath}.")
            return None
        except pd.errors.EmptyDataError:
            logger.warning(f"Existing file {filepath} is empty. Starting fresh.")
            os.remove(filepath)
            return None
        except Exception as e:
            logger.error(f"Error loading existing data from {filepath}: {e}. Starting fresh.")
            return None

    def download_new_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """Downloads data from yfinance for a given period and interval, handling MultiIndex columns."""
        logger.info(f"Attempting download for {symbol} from {start_date} to {end_date} at interval {interval}...")
        try:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            yf_end_date = end_date_dt + timedelta(days=1)
            data = yf.download(
                tickers=symbol,
                start=start_date,
                end=yf_end_date,
                interval=interval,
                progress=False,
                auto_adjust=False
            )
            if data is None or data.empty:
                logger.warning(f"No data downloaded for {symbol} in the specified period at interval {interval}.")
                return None
            if isinstance(data.columns, pd.MultiIndex):
                logger.debug(f"MultiIndex columns detected for {symbol}. Flattening...")
                data.columns = data.columns.get_level_values(0)
                data = data.loc[:, ~data.columns.duplicated()]
                logger.debug(f"Flattened columns for {symbol}: {data.columns.tolist()}")
            logger.debug(f"Columns after processing for {symbol}: {data.columns.tolist()}")
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            essential_cols = ['Open', 'High', 'Low', 'Close']
            volume_col = 'Volume'
            missing_essential = [col for col in essential_cols if col not in data.columns]
            if missing_essential:
                logger.warning(f"Essential columns missing for {symbol}: {missing_essential}. Skipping row removal.")
            else:
                initial_rows = len(data)
                data.dropna(subset=['Close'], inplace=True)
                if len(data) < initial_rows:
                    logger.debug(f"Removed {initial_rows - len(data)} rows with NaN 'Close' for {symbol}.")
            if volume_col in data.columns:
                try:
                    data[volume_col] = pd.to_numeric(data[volume_col], errors='coerce').fillna(0).astype(int)
                except Exception as vol_e:
                    logger.warning(f"Could not convert Volume column to int for {symbol}: {vol_e}")
            else:
                logger.warning(f"Volume column missing for {symbol}.")
            if data.empty:
                logger.warning(f"Data for {symbol} became empty after cleaning.")
                return None
            logger.info(f"Successfully downloaded and cleaned {len(data)} new rows for {symbol} at interval {interval}.")
            return data
        except KeyError as ke:
            logger.error(f"KeyError processing data for {symbol}: {ke}. Columns: {data.columns.tolist() if data is not None else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"Error during download/processing for {symbol}: {e}", exc_info=True)
            return None

    def process_and_save(self, symbol: str, asset_type: str, existing_data: Optional[pd.DataFrame], new_data: Optional[pd.DataFrame], interval: str) -> int:
        """Merges new data with existing, removes duplicates, sorts, and saves."""
        filepath = os.path.join(self.save_dir, f"{symbol.replace('=', '_')}_{asset_type}_{interval}.csv")
        final_data = existing_data
        if new_data is not None and not new_data.empty:
            if existing_data is not None and not existing_data.empty:
                logger.info(f"Merging {len(existing_data)} existing rows with {len(new_data)} new rows for {symbol} at interval {interval}.")
                combined = pd.concat([existing_data, new_data])
                final_data = combined[~combined.index.duplicated(keep='last')]
            else:
                logger.info(f"No existing data for {symbol}, using {len(new_data)} downloaded rows at interval {interval}.")
                final_data = new_data
        elif existing_data is not None:
            logger.info(f"No new data downloaded for {symbol}. Keeping existing {len(existing_data)} rows at interval {interval}.")
        else:
            logger.info(f"No existing or new data available for {symbol} at interval {interval}.")
            return 0
        if final_data is not None and not final_data.empty:
            try:
                final_data.sort_index(inplace=True)
                date_format = '%Y-%m-%d %H:%M:%S' if interval == '1h' else '%Y-%m-%d'
                final_data.to_csv(filepath, date_format=date_format)
                logger.info(f"Successfully saved {len(final_data)} total rows to {filepath}")
                return len(final_data)
            except IOError as e:
                logger.error(f"Error saving data for {symbol} to {filepath}: {e}")
                return len(final_data)
            except Exception as e:
                logger.error(f"Unexpected error during saving for {symbol}: {e}")
                return len(final_data)
        else:
            return 0

    def download_and_update_symbol(self, symbol_info: Tuple[str, str], interval: str) -> int:
        """Handles the full download and update process for a single symbol at a specified interval."""
        symbol, asset_type = symbol_info
        logger.info(f"--- Processing symbol: {symbol} ({asset_type}) at interval {interval} ---")
        index_col = 'Datetime' if interval == '1h' else 'Date'
        filepath = os.path.join(self.save_dir, f"{symbol.replace('=', '_')}_{asset_type}_{interval}.csv")
        existing_data = self.load_existing_data(filepath, index_col)
        if interval == '1h':
            max_hist_days = self.hourly_fetch_days
            end_date_dt = datetime.now()
            earliest_start_dt = end_date_dt - timedelta(days=max_hist_days)
            if existing_data is not None and not existing_data.empty:
                last_timestamp = existing_data.index[-1]
                start_date_dt = last_timestamp + timedelta(hours=1)
                start_date_dt = max(start_date_dt, earliest_start_dt)
            else:
                start_date_dt = earliest_start_dt
            start_date_str = start_date_dt.strftime('%Y-%m-%d')
            end_date_str = end_date_dt.strftime('%Y-%m-%d')
        else:
            if existing_data is not None and not existing_data.empty:
                start_date_dt = existing_data.index[-1].date() + timedelta(days=1)
                start_date_str = start_date_dt.strftime('%Y-%m-%d')
            else:
                start_date_str = self.default_start_date
                start_date_dt = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date_dt = datetime.now().date() - timedelta(days=1)
            end_date_str = end_date_dt.strftime('%Y-%m-%d')
        new_data = None
        if (interval == '1h' and start_date_dt.date() <= end_date_dt.date()) or (interval != '1h' and start_date_dt <= end_date_dt):
            time.sleep(self.rate_limit_delay)
            new_data = self.download_new_data(symbol, start_date_str, end_date_str, interval)
        elif existing_data is not None:
            logger.info(f"Data for {symbol} is already up-to-date at interval {interval} (last date: {existing_data.index[-1]}).")
        total_rows = self.process_and_save(symbol, asset_type, existing_data, new_data, interval)
        logger.info(f"--- Finished processing {symbol} at interval {interval}. Total rows: {total_rows} ---")
        return total_rows

    async def download_all_data(self, symbols: List[Tuple[str, str]] = None, intervals: List[str] = ['1d', '1h', '1wk', '1mo']) -> Dict[str, Dict[str, int]]:
        """Iterates through symbols and downloads/updates data for each at specified intervals."""
        if symbols is None:
            symbols = self.symbols
        logger.info(f"=== Starting all data download/update process for intervals {intervals} ===")
        total_rows_map = {}
        for symbol_info in symbols:
            symbol_rows_map = {}
            for interval in intervals:
                if interval == '1h' and symbol_info[1] == 'future':
                    continue  # Skip hourly for futures
                try:
                    rows = self.download_and_update_symbol(symbol_info, interval)
                    symbol_rows_map[interval] = rows
                except Exception as e:
                    logger.error(f"Critical error processing symbol {symbol_info[0]} at interval {interval}: {e}", exc_info=True)
                    symbol_rows_map[interval] = -1
            total_rows_map[symbol_info[0]] = symbol_rows_map
        logger.info(f"=== All data download/update attempts completed for intervals {intervals} ===")
        logger.info(f"Final row counts: {total_rows_map}")
        return total_rows_map
