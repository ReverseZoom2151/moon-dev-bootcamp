import pandas as pd
import pandas_ta as ta
import ccxt
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects.key_file import bitfinex_key, bitfinex_secret

# Initialize exchange
exchange = ccxt.bitfinex({'apiKey': bitfinex_key, 'secret': bitfinex_secret, 'enableRateLimit': True})

# Fetch OHLCV data
symbol = 'BTC:USTF0'
timeframe = '1h'
limit = 30
data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# SMA
df['sma_10'] = ta.sma(df['close'], length=10)

# EMA
df['ema_10'] = ta.ema(df['close'], length=10)

# RSI
df['rsi_14'] = ta.rsi(df['close'], length=14)

# MACD
macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
df = pd.concat([df, macd], axis=1)

# Stochastic Oscillator
stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
df = pd.concat([df, stoch], axis=1)

# Bollinger Bands
bb = ta.bbands(df['close'], length=20, std=2)
df = pd.concat([df, bb], axis=1)

# ATR
df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

# CCI
df['cci_20'] = ta.cci(df['high'], df['low'], df['close'], length=20)

# Parabolic SAR
psar = ta.psar(df['high'], df['low'], df['close'], af0=0.02, af=0.02, max_af=0.2)
df = pd.concat([df, psar], axis=1)

# OBV
df['obv'] = ta.obv(df['close'], df['volume'])

print(df.tail())

# To get help on all indicators
help(ta) 