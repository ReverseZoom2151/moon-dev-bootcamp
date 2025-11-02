import asyncio
import logging
import pandas as pd
import pandas_ta as pta
from typing import Dict, Tuple
from core.config import get_settings
from data.market_data_manager import MarketDataManager
from ta.momentum import RSIIndicator

class IndicatorService:
    """Service to compute and cache various technical indicators for symbols/timeframes."""
    def __init__(self, market_data_manager: MarketDataManager):
        settings = get_settings()
        self.settings = settings
        self.market_data_manager = market_data_manager
        self.logger = logging.getLogger("indicator_service")
        # configuration
        self.loop_interval = settings.INDICATOR_LOOP_INTERVAL
        self.symbols = settings.INDICATOR_SYMBOLS
        self.timeframes = settings.INDICATOR_TIMEFRAMES
        self.limit = settings.INDICATOR_LIMIT
        # indicator parameters
        self.rsi_period = settings.RSI_PERIOD
        self.sma_periods = settings.IND_SMA_PERIODS
        self.ema_periods = settings.IND_EMA_PERIODS
        self.vwma_periods = settings.VWMA_PERIODS
        self.stoch_k = settings.STOCHRSI_K_PERIOD
        self.stoch_d = settings.STOCHRSI_D_PERIOD
        # cache: (symbol, timeframe) -> DataFrame with indicators
        self._cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    async def start(self):
        self.logger.info(
            f"🚀 Starting Indicator Service (symbols={self.symbols}, timeframes={self.timeframes})"
        )
        while True:
            try:
                await self._refresh_all()
            except Exception as e:
                self.logger.error(f"Error refreshing indicators: {e}")
            await asyncio.sleep(self.loop_interval)

    async def _refresh_all(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                # Fetch OHLCV data
                df = await self.market_data_manager.get_ohlcv(symbol, tf, self.limit)
                if df is None or df.empty:
                    continue
                df_ind = df.copy()
                # Simple Moving Averages
                for per in self.sma_periods:
                    df_ind[f'sma_{per}'] = df_ind['close'].rolling(per).mean()
                # Exponential Moving Averages
                for per in self.ema_periods:
                    df_ind[f'ema_{per}'] = df_ind['close'].ewm(span=per, adjust=False).mean()
                # Relative Strength Index
                rsi = RSIIndicator(df_ind['close'], window=self.rsi_period).rsi()
                df_ind[f'rsi_{self.rsi_period}'] = rsi
                # Volume-Weighted Moving Averages
                for per in self.vwma_periods:
                    vol = df_ind['volume']
                    df_ind[f'vwma_{per}'] = (
                        (vol * df_ind['close']).rolling(min_periods=1, window=per).sum() /
                        vol.rolling(min_periods=1, window=per).sum()
                    )
                # Stochastic RSI using pandas_ta
                stoch = pta.stoch(
                    df_ind['high'], df_ind['low'], df_ind['close'],
                    k=self.stoch_k, d=self.stoch_d
                )
                k_col = f'STOCHk_{self.stoch_k}_{self.stoch_d}_{self.stoch_d}'
                d_col = f'STOCHd_{self.stoch_k}_{self.stoch_d}_{self.stoch_d}'
                if k_col in stoch and d_col in stoch:
                    df_ind['stoch_k'] = stoch[k_col]
                    df_ind['stoch_d'] = stoch[d_col]
                # pandas_ta indicators (mimic talib-prefixed outputs)
                for per in self.sma_periods:
                    df_ind[f'talib_sma_{per}'] = pta.sma(df_ind['close'], length=per)
                for per in self.ema_periods:
                    df_ind[f'talib_ema_{per}'] = pta.ema(df_ind['close'], length=per)
                # Bollinger Bands
                bb = pta.bbands(df_ind['close'], length=self.sma_periods[0], std=2)
                df_ind['bb_upper'] = bb[f'BBU_{self.sma_periods[0]}_2']
                df_ind['bb_mid']   = bb[f'BBM_{self.sma_periods[0]}_2']
                df_ind['bb_lower'] = bb[f'BBL_{self.sma_periods[0]}_2']
                # MACD
                macd = pta.macd(df_ind['close'], fast=12, slow=26, signal=9)
                key = f'MACD_12_26_9'
                df_ind['macd_line']   = macd[key]
                df_ind['macd_hist']   = macd[f'MACDh_12_26_9']
                df_ind['macd_signal'] = macd[f'MACDs_12_26_9']
                # ATR
                df_ind[f'atr_{self.rsi_period}'] = pta.atr(df_ind['high'], df_ind['low'], df_ind['close'], length=self.rsi_period)
                # Stochastic oscillator
                stoch = pta.stoch(df_ind['high'], df_ind['low'], df_ind['close'], k=self.stoch_k, d=self.stoch_d)
                df_ind['stoch_k_talib'] = stoch[f'STOCHk_{self.stoch_k}_{self.stoch_d}_{self.stoch_d}']
                df_ind['stoch_d_talib'] = stoch[f'STOCHd_{self.stoch_k}_{self.stoch_d}_{self.stoch_d}']
                # Commodity Channel Index
                df_ind[f'cci_{self.sma_periods[0]}'] = pta.cci(df_ind['high'], df_ind['low'], df_ind['close'], length=self.sma_periods[0])
                # Parabolic SAR
                df_ind['sar'] = pta.sar(df_ind['high'], df_ind['low'], acceleration=0.02, maximum=0.2)
                # On Balance Volume
                df_ind['obv'] = pta.obv(df_ind['close'], df_ind['volume'])
                # Cache the computed indicators
                self._cache[(symbol, tf)] = df_ind
                self.logger.debug(f"Computed indicators for {symbol} {tf}")

    def get(self, symbol: str, timeframe: str, indicator: str):
        """Retrieve a specific indicator series for a symbol and timeframe."""
        key = (symbol, timeframe)
        df = self._cache.get(key)
        if df is None:
            self.logger.warning(f"No indicator data for {symbol} {timeframe}")
            return None
        if indicator not in df.columns:
            self.logger.warning(f"Indicator {indicator} not found for {symbol} {timeframe}")
            return None
        return df[indicator] 