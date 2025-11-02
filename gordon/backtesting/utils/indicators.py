"""Common indicator utilities"""

import numpy as np
import pandas as pd


class IndicatorHelper:
    """Helper methods for indicator calculations"""
    
    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().values
    
    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values
    
    @staticmethod
    def atr(high, low, close, period):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().values
    
    @staticmethod
    def rsi(data, period):
        """Relative Strength Index"""
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    @staticmethod
    def bbands(data, period, num_std):
        """Bollinger Bands"""
        sma = pd.Series(data).rolling(window=period).mean()
        std = pd.Series(data).rolling(window=period).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return upper.values, sma.values, lower.values
