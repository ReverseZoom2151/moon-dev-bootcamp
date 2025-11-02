"""
Live Breakout Trading Bot
==========================
Day 18: Live breakout detection and trading bot (binance_531_breakoutbot.py)

This bot scans multiple symbols for breakout opportunities and places trades
when price breaks above daily resistance levels.
"""

import logging
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveBreakoutBot:
    """
    Live Breakout Trading Bot (Day 18 - binance_531_breakoutbot.py).

    Scans multiple symbols for breakout conditions:
    - Monitors daily resistance levels (20-day high)
    - Detects when price breaks above resistance
    - Automatically places trades with stop loss and take profit
    - Manages positions across multiple symbols
    """

    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        """
        Initialize the breakout bot.

        Args:
            exchange_adapter: Exchange connection adapter
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or self._get_default_config()

        # Trading parameters
        self.order_usd_size = self.config.get('order_usd_size', 10)
        self.lookback_hours = self.config.get('lookback_hours', 1)
        self.leverage = self.config.get('leverage', 3)
        self.lookback_days = self.config.get('lookback_days', 20)
        self.stop_loss_percent = self.config.get('stop_loss_percent', 18)
        self.take_profit_percent = self.config.get('take_profit_percent', 3)

        # Symbol management
        self.symbols = []
        self.resistance_levels = {}
        self.active_positions = {}

        # Demo mode
        self.demo_mode = self.config.get('demo_mode', False)
        if self.demo_mode:
            logger.info("🔧 Running in DEMO MODE - no real trades will be executed")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'order_usd_size': 10,
            'lookback_hours': 1,
            'leverage': 3,
            'lookback_days': 20,
            'stop_loss_percent': 18,
            'take_profit_percent': 3,
            'demo_mode': False,
            'max_positions': 5,
            'scan_interval_minutes': 1,
            'symbols_filter': ['USDT'],  # Only USDT pairs
            'excluded_symbols': []  # Symbols to exclude
        }

    async def get_active_symbols(self) -> List[str]:
        """
        Get list of active trading symbols from exchange.

        Returns:
            List of symbol strings
        """
        if self.demo_mode:
            # Return demo symbols
            return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']

        try:
            if not self.exchange_adapter:
                logger.error("No exchange adapter configured")
                return []

            # Get all markets
            markets = await self.exchange_adapter.load_markets()

            # Filter for active USDT perpetual contracts
            symbols = []
            for symbol, market in markets.items():
                if (market.get('active') and
                    symbol.endswith('USDT') and
                    market.get('type') == 'swap'):
                    # Apply additional filters
                    if self.config.get('excluded_symbols'):
                        if symbol in self.config['excluded_symbols']:
                            continue
                    symbols.append(symbol)

            logger.info(f'Found {len(symbols)} active USDT perpetual symbols')
            return symbols[:self.config.get('max_symbols', 50)]  # Limit number of symbols

        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []

    async def fetch_daily_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a symbol.

        Args:
            symbol: Trading symbol
            days: Number of days to fetch

        Returns:
            DataFrame with OHLCV data
        """
        days = days or self.lookback_days

        if self.demo_mode:
            # Generate demo data
            import numpy as np
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            prices = 100 + np.cumsum(np.random.randn(days) * 2)
            return pd.DataFrame({
                'timestamp': dates,
                'open': prices - np.random.rand(days) * 1,
                'high': prices + np.random.rand(days) * 2,
                'low': prices - np.random.rand(days) * 2,
                'close': prices,
                'volume': np.random.rand(days) * 1000000
            })

        try:
            if not self.exchange_adapter:
                return pd.DataFrame()

            # Calculate since timestamp
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            # Fetch OHLCV data
            ohlcv = await self.exchange_adapter.fetch_ohlcv(
                symbol, timeframe='1d', since=since
            )

            if not ohlcv:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df

        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return pd.DataFrame()

    async def calculate_resistance_levels(self) -> Dict[str, float]:
        """
        Calculate daily resistance levels for all symbols.

        Returns:
            Dictionary of symbol: resistance_level
        """
        resistance_levels = {}

        for symbol in self.symbols:
            try:
                # Fetch daily data
                daily_df = await self.fetch_daily_data(symbol)

                if daily_df.empty:
                    continue

                # Calculate resistance as max high over lookback period
                resistance = daily_df['high'].max()
                resistance_levels[symbol] = resistance

                logger.info(f"Resistance for {symbol}: {resistance:.4f}")

            except Exception as e:
                logger.error(f"Error calculating resistance for {symbol}: {e}")

        return resistance_levels

    async def check_breakout(self, symbol: str) -> Optional[Dict]:
        """
        Check if symbol is breaking out above resistance.

        Args:
            symbol: Trading symbol

        Returns:
            Breakout information dictionary or None
        """
        try:
            # Get current price
            if self.demo_mode:
                import random
                current_price = self.resistance_levels.get(symbol, 100) * random.uniform(0.95, 1.05)
            else:
                if not self.exchange_adapter:
                    return None

                ticker = await self.exchange_adapter.fetch_ticker(symbol)
                current_price = ticker.get('last')

            if not current_price:
                return None

            # Get resistance level
            resistance = self.resistance_levels.get(symbol)
            if not resistance:
                logger.warning(f'No resistance level found for {symbol}')
                return None

            # Check for breakout
            if current_price > resistance:
                # Calculate entry, stop loss, and take profit
                entry_price = current_price
                stop_loss = entry_price * (1 - self.stop_loss_percent / 100)
                take_profit = entry_price * (1 + self.take_profit_percent / 100)

                return {
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'resistance': resistance,
                    'breakout_percentage': ((current_price - resistance) / resistance) * 100
                }

            return None

        except Exception as e:
            logger.error(f"Error checking breakout for {symbol}: {e}")
            return None

    async def calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """
        Calculate position size based on account balance and leverage.

        Args:
            symbol: Trading symbol
            entry_price: Entry price for the trade

        Returns:
            Position size
        """
        if self.demo_mode:
            return self.order_usd_size / entry_price

        try:
            if not self.exchange_adapter:
                return 0

            # Get account balance
            balance = await self.exchange_adapter.fetch_balance()
            available_balance = balance.get('USDT', {}).get('free', 0)

            # Calculate position size
            position_value = min(self.order_usd_size, available_balance * 0.95)
            position_size = (position_value / entry_price) * self.leverage

            # Get symbol precision
            market = await self.exchange_adapter.fetch_market(symbol)
            precision = market.get('precision', {}).get('amount', 8)
            position_size = round(position_size, precision)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    async def place_breakout_trade(self, breakout_info: Dict) -> bool:
        """
        Place a trade for a breakout signal.

        Args:
            breakout_info: Breakout information dictionary

        Returns:
            Success status
        """
        symbol = breakout_info['symbol']

        try:
            # Check if already in position
            if symbol in self.active_positions:
                logger.info(f"Already in position for {symbol}")
                return False

            # Check max positions
            if len(self.active_positions) >= self.config.get('max_positions', 5):
                logger.info(f"Max positions reached ({len(self.active_positions)})")
                return False

            # Calculate position size
            position_size = await self.calculate_position_size(
                symbol, breakout_info['entry_price']
            )

            if position_size <= 0:
                logger.error(f"Invalid position size for {symbol}")
                return False

            if self.demo_mode:
                logger.info(f"[DEMO] Would place breakout trade for {symbol}:")
                logger.info(f"  Entry: {breakout_info['entry_price']:.4f}")
                logger.info(f"  Stop Loss: {breakout_info['stop_loss']:.4f}")
                logger.info(f"  Take Profit: {breakout_info['take_profit']:.4f}")
                logger.info(f"  Size: {position_size}")

                # Track demo position
                self.active_positions[symbol] = {
                    'entry_price': breakout_info['entry_price'],
                    'size': position_size,
                    'stop_loss': breakout_info['stop_loss'],
                    'take_profit': breakout_info['take_profit'],
                    'timestamp': datetime.now()
                }
                return True

            else:
                if not self.exchange_adapter:
                    return False

                # Place limit order at entry price
                order = await self.exchange_adapter.create_limit_buy_order(
                    symbol=symbol,
                    amount=position_size,
                    price=breakout_info['entry_price'],
                    params={'timeInForce': 'GTX'}
                )

                if order:
                    logger.info(f"Placed breakout order for {symbol}")
                    logger.info(f"  Order ID: {order.get('id')}")
                    logger.info(f"  Entry: {breakout_info['entry_price']:.4f}")
                    logger.info(f"  Size: {position_size}")

                    # Track position
                    self.active_positions[symbol] = {
                        'order_id': order.get('id'),
                        'entry_price': breakout_info['entry_price'],
                        'size': position_size,
                        'stop_loss': breakout_info['stop_loss'],
                        'take_profit': breakout_info['take_profit'],
                        'timestamp': datetime.now()
                    }

                    # TODO: Set stop loss and take profit orders
                    # This would require exchange-specific implementation

                    return True

                return False

        except Exception as e:
            logger.error(f"Error placing breakout trade for {symbol}: {e}")
            return False

    async def scan_for_breakouts(self) -> List[Dict]:
        """
        Scan all symbols for breakout opportunities.

        Returns:
            List of breakout information dictionaries
        """
        logger.info("Scanning for breakout opportunities...")

        breakouts = []

        for symbol in self.symbols:
            breakout_info = await self.check_breakout(symbol)

            if breakout_info:
                breakouts.append(breakout_info)
                logger.info(f"🚀 Breakout detected for {symbol}!")
                logger.info(f"   Breakout %: {breakout_info['breakout_percentage']:.2f}%")

        logger.info(f"Found {len(breakouts)} breakout opportunities")
        return breakouts

    async def run_cycle(self):
        """Run one complete bot cycle."""
        try:
            logger.info("=" * 60)
            logger.info("Starting breakout bot cycle...")

            # Update symbol list
            self.symbols = await self.get_active_symbols()
            logger.info(f"Monitoring {len(self.symbols)} symbols")

            # Calculate resistance levels
            self.resistance_levels = await self.calculate_resistance_levels()

            # Scan for breakouts
            breakouts = await self.scan_for_breakouts()

            # Place trades for breakouts
            if breakouts:
                # Save to CSV for analysis
                breakout_df = pd.DataFrame(breakouts)
                breakout_df.to_csv(
                    f'breakouts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    index=False
                )
                logger.info(f"Saved {len(breakouts)} breakouts to CSV")

                # Place trades
                for breakout_info in breakouts:
                    await self.place_breakout_trade(breakout_info)

            logger.info("Breakout bot cycle completed")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error in bot cycle: {e}")

    async def start(self, interval_minutes: int = None):
        """
        Start the breakout bot.

        Args:
            interval_minutes: Minutes between scans
        """
        interval = interval_minutes or self.config.get('scan_interval_minutes', 1)

        logger.info(f"Starting Live Breakout Bot (interval: {interval} minutes)")
        logger.info(f"Configuration: {self.config}")

        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(interval * 60)

            except KeyboardInterrupt:
                logger.info("Breakout bot stopped by user")
                break

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            'demo_mode': self.demo_mode,
            'active_positions': len(self.active_positions),
            'monitored_symbols': len(self.symbols),
            'config': self.config,
            'positions': self.active_positions
        }


def main():
    """Main function to run the breakout bot standalone."""
    # Initialize bot with demo mode
    config = {
        'demo_mode': True,
        'order_usd_size': 100,
        'leverage': 3,
        'max_positions': 5,
        'scan_interval_minutes': 1
    }

    bot = LiveBreakoutBot(config=config)

    # Run async event loop
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()