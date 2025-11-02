import pandas as pd
import datetime
import os
import ccxt.async_support as ccxt
import time
import logging
from math import ceil
from backend.core.config import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalDataService:
    def __init__(self):
        self.cache_dir = config.get('DATA_CACHE_DIR', './data_cache')
        self.fetch_limit = config.get('FETCH_LIMIT', 200)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.exchange = None

    async def initialize(self):
        """Initialize the Coinbase exchange connection."""
        try:
            self.exchange = ccxt.coinbase({
                'apiKey': config['COINBASE_API_KEY'],
                'secret': config['COINBASE_API_SECRET'],
                'enableRateLimit': True,
            })
            markets = await self.exchange.load_markets()
            logger.info("Coinbase exchange initialized successfully.")
            return markets
        except Exception as e:
            logger.error(f"Failed to initialize Coinbase exchange: {e}")
            raise

    def timeframe_to_sec(self, timeframe: str) -> int:
        """Converts CCXT timeframe string to seconds."""
        amount = int("".join(filter(str.isdigit, timeframe)))
        unit = "".join(filter(str.isalpha, timeframe))

        if unit == 'm':
            return amount * 60
        elif unit == 'h':
            return amount * 60 * 60
        elif unit == 'd':
            return amount * 24 * 60 * 60
        elif unit == 'w':
            return amount * 7 * 24 * 60 * 60
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    async def get_historical_data(self, symbol: str, timeframe: str, weeks: int) -> pd.DataFrame:
        """
        Fetches historical OHLCV data for a given symbol, timeframe, and number of weeks.
        Caches results to a CSV file.
        """
        cache_filename = os.path.join(self.cache_dir, f'{symbol.replace("/", "_")}-{timeframe}-{weeks}wks-data.csv')

        if os.path.exists(cache_filename):
            logger.info(f"Loading cached data from: {cache_filename}")
            try:
                return pd.read_csv(cache_filename, index_col='datetime', parse_dates=True)
            except Exception as e:
                logger.error(f"Error loading cache file {cache_filename}: {e}. Fetching new data.")

        logger.info(f"No cache found or cache error. Fetching new data for {symbol} ({timeframe}, {weeks} weeks)...")

        if not self.exchange:
            await self.initialize()

        markets = await self.exchange.load_markets()
        if symbol not in markets:
            raise ValueError(f"Symbol {symbol} not found on Coinbase.")

        granularity_sec = self.timeframe_to_sec(timeframe)
        total_time_sec = weeks * 7 * 24 * 60 * 60
        run_times = ceil(total_time_sec / (granularity_sec * self.fetch_limit))

        logger.info(f"Calculated granularity: {granularity_sec}s")
        logger.info(f"Total time to fetch: {total_time_sec}s")
        logger.info(f"Required API calls (runs): {run_times}")

        all_ohlcv = []
        now = datetime.datetime.now(datetime.timezone.utc)
        since_timestamp_ms = int((now - datetime.timedelta(weeks=weeks)).timestamp() * 1000)

        logger.info(f"Starting fetch from: {datetime.datetime.fromtimestamp(since_timestamp_ms / 1000, tz=datetime.timezone.utc)}")

        current_timestamp_ms = since_timestamp_ms
        for i in range(run_times):
            logger.info(f"Fetching run {i+1}/{run_times} starting from timestamp {current_timestamp_ms}...")
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp_ms, limit=self.fetch_limit)

                if not ohlcv:
                    logger.info(f"No more data returned for {symbol} at timestamp {current_timestamp_ms}. Stopping fetch.")
                    break

                all_ohlcv.extend(ohlcv)
                last_candle_ts = ohlcv[-1][0]
                current_timestamp_ms = last_candle_ts + (granularity_sec * 1000)

            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit exceeded: {e}. Waiting...")
                time.sleep(60)
            except ccxt.NetworkError as e:
                logger.error(f"Network error fetching chunk {i+1}: {e}. Retrying after delay...")
                time.sleep(10)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error fetching chunk {i+1}: {e}. Skipping chunk.")
                current_timestamp_ms += (granularity_sec * 1000 * self.fetch_limit)
            except Exception as e:
                logger.error(f"Unexpected error during fetch run {i+1}: {e}. Stopping fetch.")
                break

        if not all_ohlcv:
            logger.error("Error: No data fetched.")
            return pd.DataFrame()

        logger.info(f"Fetched total {len(all_ohlcv)} candles.")

        dataframe = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        dataframe['datetime'] = pd.to_datetime(dataframe['timestamp'], unit='ms', utc=True)
        dataframe.set_index('datetime', inplace=True)
        dataframe.drop('timestamp', axis=1, inplace=True)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        dataframe.dropna(inplace=True)

        initial_rows = len(dataframe)
        dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
        logger.info(f"Removed {initial_rows - len(dataframe)} duplicate entries based on timestamp.")

        dataframe.sort_index(inplace=True)

        try:
            dataframe.to_csv(cache_filename)
            logger.info(f"Data saved successfully to: {cache_filename}")
        except IOError as e:
            logger.error(f"Error saving data to cache file {cache_filename}: {e}")

        return dataframe

    async def close(self):
        """Close the exchange connection."""
        if self.exchange:
            await self.exchange.close()
            logger.info("Coinbase exchange connection closed.")