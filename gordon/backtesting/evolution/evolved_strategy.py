"""
Evolved Strategy for Backtesting
=================================
Day 29: Strategy class that uses evolved GP functions.

This strategy class is dynamically populated with GP-evolved functions
to generate trading signals.
"""

import numpy as np
import pandas as pd
# Import external backtesting package, avoiding conflict with gordon.backtesting
from ..utils.backtesting_import import get_backtesting_strategy

Strategy = get_backtesting_strategy()

if Strategy is None:
    raise ImportError("External 'backtesting' package not installed. Install with: pip install backtesting")

from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using fallback indicators.")


class EvolvedStrategy(Strategy):
    """
    Evolved strategy class that uses GP-evolved functions.
    
    This class is dynamically populated with GP functions during evolution.
    The GP function takes 9 inputs and returns a trading signal.
    """

    gp_function: Optional[Callable] = None
    
    # Strategy parameters
    sma_period = 20
    ema_period = 20
    rsi_period = 14
    
    # Signal thresholds
    buy_threshold = 0.5
    sell_threshold = -0.5
    neutral_threshold = 0.2

    def init(self):
        """Initialize strategy with technical indicators."""
        if self.gp_function is None:
            raise ValueError("GP function not set for EvolvedStrategy")
        
        # Precompute indicators
        if TALIB_AVAILABLE:
            self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
            self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
            self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
            
            # MACD
            macd_result = self.I(talib.MACD, self.data.Close)
            if isinstance(macd_result, tuple):
                self.macd = macd_result[0]  # MACD line
            else:
                self.macd = macd_result
        else:
            # Fallback to pandas calculations
            self.sma = self.I(self._calculate_sma, self.data.Close, self.sma_period)
            self.ema = self.I(self._calculate_ema, self.data.Close, self.ema_period)
            self.rsi = self.I(self._calculate_rsi, self.data.Close, self.rsi_period)
            self.macd = self.I(self._calculate_macd, self.data.Close)

    def _calculate_sma(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate SMA using pandas."""
        return close.rolling(window=period).mean()

    def _calculate_ema(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate EMA using pandas."""
        return close.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using pandas."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, close: pd.Series) -> pd.Series:
        """Calculate MACD using pandas."""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        return ema12 - ema26

    def next(self):
        """Execute strategy logic using evolved GP function."""
        # Ensure enough data points
        min_periods = max(self.sma_period, self.ema_period, self.rsi_period) + 10
        if len(self.data) < min_periods:
            return
        
        try:
            # Get latest values
            close = self.data.Close[-1]
            open_ = self.data.Open[-1]
            high = self.data.High[-1]
            low = self.data.Low[-1]
            volume = self.data.Volume[-1]
            sma = self.sma[-1]
            ema = self.ema[-1]
            rsi = self.rsi[-1]
            macd = self.macd[-1]
            
            # Check for NaN values
            values = [close, open_, high, low, volume, sma, ema, rsi, macd]
            if any(np.isnan(x) or x is None for x in values):
                return
            
            # Execute GP function with 9 arguments
            signal = self.gp_function(close, open_, high, low, volume, sma, ema, rsi, macd)
            
            # Normalize signal
            if np.isnan(signal) or np.isinf(signal):
                signal = 0.0
            else:
                signal = np.clip(signal, -10.0, 10.0)
            
            # Trading logic based on signal
            if signal > self.buy_threshold:
                if not self.position:
                    self.buy()
            elif signal < self.sell_threshold:
                if self.position:
                    self.sell()
            elif abs(signal) < self.neutral_threshold:
                # Close position on neutral signal
                if self.position:
                    self.position.close()
                    
        except (OverflowError, ValueError, TypeError, ZeroDivisionError, ArithmeticError):
            # Handle runtime errors silently
            pass
        except Exception as e:
            logger.debug(f"Error in evolved strategy: {e}")

