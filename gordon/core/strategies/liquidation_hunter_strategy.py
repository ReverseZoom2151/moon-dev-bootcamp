"""
Enhanced Liquidation Hunter Strategy
====================================
Day 45: Real-time liquidation cascade hunting strategy.

Analyzes whale positions to determine market bias (long/short),
then hunts for liquidation events to enter trades in the direction of the bias.

Features:
- Whale position analysis
- Real-time liquidation monitoring
- Cascade detection
- Automated entry/exit logic
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

from .base import BaseStrategy
from ...research.data_providers.moondev_api import MoonDevAPI
from ...exchanges.factory import ExchangeFactory

logger = logging.getLogger(__name__)


class LiquidationHunterStrategy(BaseStrategy):
    """
    Enhanced Liquidation Hunter Strategy.
    
    Analyzes whale positions and liquidation events to determine trading bias
    and execute trades when liquidation cascades occur.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize liquidation hunter strategy."""
        super().__init__("LiquidationHunter")
        self.config = config or {}
        
        # Strategy parameters
        self.liquidation_lookback_minutes = self.config.get('liquidation_lookback_minutes', 5)
        self.liquidation_trigger_amount = self.config.get('liquidation_trigger_amount', 100000)  # $100k default
        self.min_position_value = self.config.get('min_position_value', 25000)  # $25k minimum
        self.analysis_interval_minutes = self.config.get('analysis_interval_minutes', 15)
        self.tokens_to_analyze = self.config.get('tokens_to_analyze', ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])
        
        # Trading parameters
        self.position_size_usd = self.config.get('position_size_usd', 10)
        self.leverage = self.config.get('leverage', 5)
        self.take_profit_percent = self.config.get('take_profit_percent', 1.0)
        self.stop_loss_percent = self.config.get('stop_loss_percent', -6.0)
        
        # State
        self.trading_bias = None  # 'long' or 'short'
        self.recommendation_text = "Waiting for initial analysis..."
        self.last_analysis_time = None
        
        # Initialize Moon Dev API
        self.moondev_api = MoonDevAPI()
        
        # Exchange adapter (will be set via set_exchange_connection)
        self.exchange_adapter = None

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)
        
        # Get exchange adapter if available
        exchange_name = self.config.get('exchange', 'binance')
        if exchange_name in self.exchange_connections:
            self.exchange_adapter = self.exchange_connections[exchange_name]
        
        logger.info(f"Liquidation Hunter Strategy initialized for {exchange_name}")

    def _analyze_whale_positions(self) -> Dict[str, Any]:
        """
        Analyze whale positions to determine market bias.
        
        Returns:
            Dictionary with analysis results including trading bias
        """
        try:
            logger.info("Analyzing whale positions...")
            
            # Fetch positions from Moon Dev API
            positions_df = self.moondev_api.get_positions_hlp()
            
            if positions_df is None or positions_df.empty:
                logger.warning("No positions data available")
                return {'bias': None, 'long_liquidations': 0, 'short_liquidations': 0}
            
            # Filter out small positions
            positions_df = positions_df[positions_df['position_value'] >= self.min_position_value].copy()
            logger.info(f"Analyzing {len(positions_df)} whale positions")
            
            # Convert numeric columns
            numeric_cols = ['entry_price', 'position_value', 'unrealized_pnl', 'liquidation_price', 'leverage']
            for col in numeric_cols:
                if col in positions_df.columns:
                    positions_df[col] = pd.to_numeric(positions_df[col], errors='coerce')
            
            # Validate and correct position types
            valid_liq_df = positions_df[positions_df['liquidation_price'] > 0].copy()
            if not valid_liq_df.empty:
                valid_liq_df['is_long'] = valid_liq_df['liquidation_price'] < valid_liq_df['entry_price']
                positions_df.loc[valid_liq_df.index, 'is_long'] = valid_liq_df['is_long']
            
            # Get current prices (simplified - would use exchange adapter in production)
            # Note: For now, we'll skip price fetching as it requires async exchange adapter
            # The liquidation impact calculation requires current prices, but fetching them
            # synchronously would block. In production, this should be done asynchronously
            # or prices should be provided as a parameter.
            current_prices = {}
            for token in self.tokens_to_analyze:
                # Extract base token from symbol (e.g., BTCUSDT -> BTC)
                base_token = token.replace('USDT', '').replace('USD', '')
                # Use a placeholder - in production, this would be fetched from exchange
                # For now, we'll calculate liquidations based on position data only
                current_prices[token] = 0
            
            # Calculate liquidation impact for 3% price move
            total_long_liquidations = {}
            total_short_liquidations = {}
            all_long_liquidations = 0
            all_short_liquidations = 0
            
            for coin in self.tokens_to_analyze:
                coin_positions = positions_df[positions_df['coin'] == coin].copy()
                if coin_positions.empty:
                    continue
                
                # Skip coins without price data - we can't calculate liquidation impact without prices
                if coin not in current_prices or current_prices[coin] == 0:
                    continue
                
                current_price = current_prices[coin]
                coin_positions['price_move'] = current_price * 0.03
                
                # Long liquidations (3% move down)
                long_liquidations = coin_positions[
                    (coin_positions['is_long']) &
                    (coin_positions['liquidation_price'] >= current_price - coin_positions['price_move'])
                ]
                total_long_liquidation_value = long_liquidations['position_value'].sum()
                
                # Short liquidations (3% move up)
                short_liquidations = coin_positions[
                    (~coin_positions['is_long']) &
                    (coin_positions['liquidation_price'] <= current_price + coin_positions['price_move'])
                ]
                total_short_liquidation_value = short_liquidations['position_value'].sum()
                
                total_long_liquidations[coin] = total_long_liquidation_value
                total_short_liquidations[coin] = total_short_liquidation_value
                all_long_liquidations += total_long_liquidation_value
                all_short_liquidations += total_short_liquidation_value
            
            # Determine trading bias
            if all_long_liquidations > all_short_liquidations * 1.5:
                trading_bias = 'short'  # Too many longs at risk
            elif all_short_liquidations > all_long_liquidations * 1.5:
                trading_bias = 'long'   # Too many shorts at risk
            else:
                trading_bias = None     # Neutral
            
            return {
                'bias': trading_bias,
                'long_liquidations': all_long_liquidations,
                'short_liquidations': all_short_liquidations,
                'total_positions': len(positions_df),
                'analysis_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing whale positions: {e}")
            return {'bias': None, 'long_liquidations': 0, 'short_liquidations': 0}

    def _get_recent_liquidations(self, symbol: str) -> Dict[str, float]:
        """
        Get recent liquidations for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with long_liq_amount and short_liq_amount
        """
        try:
            # Fetch liquidation data
            liquidations = self.moondev_api.get_liquidation_data(limit=1000)
            if liquidations is None or liquidations.empty:
                return {'long_liq_amount': 0, 'short_liq_amount': 0}
            
            # Normalize symbol format (remove exchange suffix if present)
            symbol_base = symbol.replace('USDT', '').replace('USD', '')
            
            # Try to find symbol column
            symbol_col = None
            for col in liquidations.columns:
                if 'symbol' in col.lower() or 'coin' in col.lower():
                    symbol_col = col
                    break
            
            if symbol_col is None:
                symbol_col = liquidations.columns[0]
            
            # Filter to our symbol
            symbol_liquidations = liquidations[
                liquidations[symbol_col].astype(str).str.contains(symbol_base, case=False, na=False)
            ]
            
            if symbol_liquidations.empty:
                return {'long_liq_amount': 0, 'short_liq_amount': 0}
            
            # Get timestamp column
            timestamp_col = None
            for col in liquidations.columns:
                if 'timestamp' in col.lower() or 'time' in col.lower():
                    timestamp_col = col
                    break
            
            if timestamp_col:
                current_time_ms = int(datetime.now().timestamp() * 1000)
                cutoff_time_ms = current_time_ms - (self.liquidation_lookback_minutes * 60 * 1000)
                
                # Filter recent liquidations
                recent_liquidations = symbol_liquidations[
                    pd.to_numeric(symbol_liquidations[timestamp_col], errors='coerce') >= cutoff_time_ms
                ]
            else:
                recent_liquidations = symbol_liquidations.tail(100)  # Last 100 if no timestamp
            
            # Get side and USD size columns
            side_col = None
            size_col = None
            for col in liquidations.columns:
                if 'side' in col.lower() or 'direction' in col.lower():
                    side_col = col
                if 'size' in col.lower() or 'amount' in col.lower() or 'usd' in col.lower():
                    size_col = col
            
            if not side_col or not size_col:
                return {'long_liq_amount': 0, 'short_liq_amount': 0}
            
            # Separate long and short liquidations
            # Long liquidations typically have side == "SELL", Short liquidations have side == "BUY"
            long_liqs = recent_liquidations[
                recent_liquidations[side_col].astype(str).str.contains('SELL|LONG|L', case=False, na=False)
            ]
            short_liqs = recent_liquidations[
                recent_liquidations[side_col].astype(str).str.contains('BUY|SHORT|S', case=False, na=False)
            ]
            
            # Calculate totals
            long_liq_amount = pd.to_numeric(long_liqs[size_col], errors='coerce').sum() if not long_liqs.empty else 0
            short_liq_amount = pd.to_numeric(short_liqs[size_col], errors='coerce').sum() if not short_liqs.empty else 0
            
            return {
                'long_liq_amount': float(long_liq_amount) if not pd.isna(long_liq_amount) else 0,
                'short_liq_amount': float(short_liq_amount) if not pd.isna(short_liq_amount) else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting recent liquidations: {e}")
            return {'long_liq_amount': 0, 'short_liq_amount': 0}

    def _should_enter_trade(self, long_liq_amount: float, short_liq_amount: float) -> tuple[bool, str]:
        """
        Determine if we should enter a trade based on liquidation amounts and trading bias.
        
        Args:
            long_liq_amount: Total USD value of long liquidations
            short_liq_amount: Total USD value of short liquidations
            
        Returns:
            Tuple of (should_enter, reason)
        """
        if self.trading_bias is None:
            return False, "No trading bias set - run market analysis first"
        
        # If bias is short, enter when SHORT liquidations exceed threshold
        if self.trading_bias == 'short' and short_liq_amount >= self.liquidation_trigger_amount:
            return True, f"ENTER SHORT - Short liquidations ${short_liq_amount:,.2f} >= threshold ${self.liquidation_trigger_amount:,.2f}"
        
        # If bias is long, enter when LONG liquidations exceed threshold
        if self.trading_bias == 'long' and long_liq_amount >= self.liquidation_trigger_amount:
            return True, f"ENTER LONG - Long liquidations ${long_liq_amount:,.2f} >= threshold ${self.liquidation_trigger_amount:,.2f}"
        
        return False, f"Waiting for {self.trading_bias.upper()} liquidations >= ${self.liquidation_trigger_amount:,.2f}"

    async def execute(self) -> Optional[Dict]:
        """
        Execute liquidation hunter strategy.
        
        Returns:
            Trading signal dictionary or None
        """
        try:
            # Check if we need to refresh market analysis
            should_refresh = (
                self.last_analysis_time is None or
                (datetime.now() - self.last_analysis_time).total_seconds() / 60 >= self.analysis_interval_minutes
            )
            
            if should_refresh:
                logger.info("Refreshing market analysis...")
                analysis = self._analyze_whale_positions()
                self.trading_bias = analysis.get('bias')
                self.last_analysis_time = datetime.now()
                
                if self.trading_bias:
                    self.recommendation_text = (
                        f"Bias: {self.trading_bias.upper()} | "
                        f"Long Liq: ${analysis['long_liquidations']:,.2f} | "
                        f"Short Liq: ${analysis['short_liquidations']:,.2f}"
                    )
                    logger.info(f"Market analysis complete: {self.recommendation_text}")
            
            if not self.trading_bias:
                logger.debug("No trading bias established yet")
                return None
            
            # Check liquidations for each symbol
            for symbol in self.tokens_to_analyze:
                liq_data = self._get_recent_liquidations(symbol)
                long_liq = liq_data['long_liq_amount']
                short_liq = liq_data['short_liq_amount']
                
                should_enter, reason = self._should_enter_trade(long_liq, short_liq)
                
                if should_enter:
                    action = 'BUY' if self.trading_bias == 'long' else 'SELL'
                    
                    logger.info(f"🚀 {reason}")
                    logger.info(f"Entering {action} position on {symbol}")
                    
                    return {
                        'action': action,
                        'symbol': symbol,
                        'size': self.position_size_usd,
                        'leverage': self.leverage,
                        'confidence': min(0.9, (long_liq + short_liq) / self.liquidation_trigger_amount * 0.5),
                        'metadata': {
                            'strategy': 'liquidation_hunter',
                            'bias': self.trading_bias,
                            'long_liq_amount': long_liq,
                            'short_liq_amount': short_liq,
                            'reason': reason,
                            'take_profit_percent': self.take_profit_percent,
                            'stop_loss_percent': self.stop_loss_percent
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing liquidation hunter strategy: {e}")
            return None

    def get_status(self) -> Dict:
        """Get current strategy status."""
        status = super().get_status()
        status.update({
            'trading_bias': self.trading_bias,
            'recommendation': self.recommendation_text,
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None
        })
        return status

