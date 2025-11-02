# Breakout Trading Strategy Backtester

This script implements a breakout trading strategy with resistance levels using both daily and hourly data.

## Features

- Automatically downloads financial data from Yahoo Finance
- Tests breakout strategy on multiple timeframes
- Includes volume confirmation and Bollinger Band filters
- Implements proper risk management with ATR-based stop losses
- Allows parameter optimization to find the best settings
- Outputs detailed backtest statistics and visualizations

## Requirements

```
pip install pandas numpy backtesting yfinance
```

## Usage

1. Run the script:
```
python 528_bt_bo_multi.py
```

2. Enter the symbol you want to test (default is BTC-USD)
3. Choose whether to optimize the parameters
4. Review the backtest results

## Available Data Sources

The script uses Yahoo Finance as the data source, which provides free data for:

- Cryptocurrencies: 'BTC-USD', 'ETH-USD', 'SOL-USD', etc.
- Stocks: 'AAPL', 'MSFT', 'TSLA', etc.
- ETFs: 'SPY', 'QQQ', 'GLD', etc.
- Forex pairs: 'EURUSD=X', 'USDJPY=X', etc.

## Strategy Overview

The strategy looks for breakouts above key resistance levels with the following confirmation filters:

1. Price must break above the 20-day high (resistance level)
2. Price must be above the upper Bollinger Band
3. Volume must be higher than the 20-period average
4. Uses ATR-based stop losses to manage risk
5. Takes profit at a predetermined percentage

## Customization

You can modify the following parameters:

- `atr_period`: Period for ATR calculation (default: 14)
- `tp_percent`: Take profit percentage (default: 5%)
- `sl_atr_mult`: Stop loss as multiple of ATR (default: 1.5)
- `volume_factor`: Volume must be higher than average by this factor (default: 1.5)
- `volume_period`: Period for volume average (default: 20)
- `bb_period`: Period for Bollinger Bands (default: 20)
- `bb_std`: Standard deviation for Bollinger Bands (default: 2)

## Example Results

The backtest results include:

- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- And many more statistics

The script also generates an HTML visualization of the trades and equity curve. 