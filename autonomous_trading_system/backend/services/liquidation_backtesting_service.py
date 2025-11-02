"""
Liquidation Backtesting Service
Integrated from Day 21 liq_bt_btc.py and liq_bt_enter_lliqs.py scripts

This service implements backtesting strategies based on liquidation data
using the backtesting library for optimization and analysis.
"""

import pandas as pd
import numpy as np
import logging
import os
import logging
import warnings
from typing import Dict, List, Optional, Any
from datetime import datetime
from backtesting import Backtest, Strategy
from dataclasses import dataclass
from core.config import get_settings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
settings = get_settings()

# Add Kalman Filter imports
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logger.warning("pykalman not available - Kalman Filter strategy will be disabled")

@dataclass
class LiquidationBacktestResult:
    """Result structure for liquidation backtesting"""
    symbol: str
    strategy_name: str
    parameters: Dict[str, Any]
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    final_equity: float
    best_trade: float
    worst_trade: float
    avg_trade: float
    profit_factor: float
    equity_curve: List[Dict]
    trades: List[Dict]
    optimization_results: Optional[Dict] = None

class LiquidationStrategy(Strategy):
    """
    Liquidation-based trading strategy for backtesting (original from liq_bt_btc.py)
    
    Strategy Logic:
    - Monitors S LIQ (short liquidation) volume over a time window
    - Enters long positions when S LIQ volume exceeds threshold
    - Uses configurable take profit and stop loss levels
    """
    
    # Optimization parameters with defaults
    liquidation_thresh = 100000  # Threshold for S LIQ volume
    time_window_mins = 20        # Lookback window in minutes
    take_profit = 0.02           # Take profit percentage
    stop_loss = 0.01             # Stop loss percentage

    def init(self):
        """Initialize strategy with liquidation volume data"""
        # Use the precalculated S LIQ volume column
        self.s_liq_volume = self.data.s_liq_volume
        logger.debug(f"LiquidationStrategy initialized with {len(self.data)} data points")

    def next(self):
        """Execute strategy logic for each bar"""
        try:
            current_time = self.data.index[-1]
            start_time = current_time - pd.Timedelta(minutes=self.time_window_mins)

            # Calculate recent S LIQ volume within time window
            start_idx = np.searchsorted(self.data.index, start_time, side='left')
            recent_s_liquidations = self.s_liq_volume[start_idx:].sum()

            # Entry condition: Buy if recent S LIQ volume exceeds threshold and not already in position
            if recent_s_liquidations >= self.liquidation_thresh and not self.position:
                # Calculate TP/SL based on current closing price
                current_price = self.data.Close[-1]
                sl_price = current_price * (1 - self.stop_loss)
                tp_price = current_price * (1 + self.take_profit)
                
                # Enter long position
                self.buy(sl=sl_price, tp=tp_price)
                
                logger.debug(f"LiquidationStrategy position opened at {current_price:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error in LiquidationStrategy.next(): {e}")

class LongOnLLiqStrategy(Strategy):
    """
    Long on L LIQ Strategy (from liq_bt_enter_lliqs.py)
    
    Strategy Logic:
    - Enters long positions when L LIQ (long liquidation) volume exceeds threshold within entry time window
    - Exits positions when S LIQ (short liquidation) volume exceeds threshold within exit time window
    - Also uses traditional take profit and stop loss levels
    """
    
    # Strategy Parameters (to be optimized)
    l_liq_entry_thresh = 100000    # L LIQ volume threshold to trigger entry
    entry_time_window_mins = 5     # Lookback window for L LIQ entry signal (minutes)
    
    s_liq_closure_thresh = 50000   # S LIQ volume threshold to trigger exit
    exit_time_window_mins = 5      # Lookback window for S LIQ exit signal (minutes)
    
    take_profit = 0.02             # Take profit percentage (e.g., 0.02 = 2%)
    stop_loss = 0.01               # Stop loss percentage (e.g., 0.01 = 1%)

    def init(self):
        """Initialize strategy with liquidation volume data"""
        # Pre-calculate or access the required data columns
        self.l_liq_volume = self.data.l_liq_volume
        self.s_liq_volume = self.data.s_liq_volume
        logger.debug(f"LongOnLLiqStrategy initialized with {len(self.data)} data points")

    def next(self):
        """Execute strategy logic for each bar"""
        try:
            current_time = self.data.index[-1]

            # --- Entry Logic --- 
            if not self.position: # Only check for entry if not already in a position
                entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
                entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
                recent_l_liquidations = self.l_liq_volume[entry_start_idx:].sum()

                # Enter long if L LIQ exceeds threshold
                if recent_l_liquidations >= self.l_liq_entry_thresh:
                    sl_price = self.data.Close[-1] * (1 - self.stop_loss)
                    tp_price = self.data.Close[-1] * (1 + self.take_profit)
                    self.buy(sl=sl_price, tp=tp_price)
                    
                    logger.debug(f"LongOnLLiqStrategy position opened at {self.data.Close[-1]:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")

            # --- Exit Logic (based on S LIQ threshold, in addition to TP/SL) --- 
            elif self.position: # Only check for S LIQ exit if in a position
                exit_start_time = current_time - pd.Timedelta(minutes=self.exit_time_window_mins)
                exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
                recent_s_liquidations = self.s_liq_volume[exit_start_idx:].sum()

                # Close position if recent S LIQ exceeds threshold
                if recent_s_liquidations >= self.s_liq_closure_thresh:
                    self.position.close(reason="S Liq Threshold Hit")
                    logger.debug(f"LongOnLLiqStrategy position closed due to S LIQ threshold")
                    
        except Exception as e:
            logger.error(f"Error in LongOnLLiqStrategy.next(): {e}")

class LongOnSLiqStrategy(Strategy):
    """
    Long on S LIQ Strategy (from liq_bt_eth.py)
    
    Strategy Logic:
    - Enters long positions when S LIQ (short liquidation) volume exceeds threshold within entry time window
    - Uses traditional take profit and stop loss levels
    - Similar to LiquidationStrategy but with specific naming from the original ETH script
    """
    
    # Strategy Parameters (to be optimized)
    s_liq_entry_thresh = 100000    # S LIQ volume threshold to trigger entry
    entry_time_window_mins = 20    # Lookback window for S LIQ entry signal (minutes)
    take_profit = 0.02             # Take profit percentage
    stop_loss = 0.01               # Stop loss percentage

    def init(self):
        """Initialize strategy with S LIQ volume data"""
        self.s_liq_volume = self.data.s_liq_volume
        logger.debug(f"LongOnSLiqStrategy initialized with {len(self.data)} data points")

    def next(self):
        """Execute strategy logic for each bar"""
        try:
            current_time = self.data.index[-1]

            # Entry Logic - only check if not in position
            if not self.position:
                entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
                entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
                recent_s_liquidations = self.s_liq_volume[entry_start_idx:].sum()

                # Enter long if S LIQ exceeds threshold
                if recent_s_liquidations >= self.s_liq_entry_thresh:
                    sl_price = self.data.Close[-1] * (1 - self.stop_loss)
                    tp_price = self.data.Close[-1] * (1 + self.take_profit)
                    self.buy(sl=sl_price, tp=tp_price)
                    
                    logger.debug(f"LongOnSLiqStrategy position opened at {self.data.Close[-1]:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")
                    
        except Exception as e:
            logger.error(f"Error in LongOnSLiqStrategy.next(): {e}")

class WIFLongOnSLiqStrategy(Strategy):
    """
    WIF Long on S LIQ Strategy (from liq_bt_wif.py)
    
    Strategy Logic:
    - Enters long positions when S LIQ (short liquidation) volume exceeds threshold within entry time window
    - Uses traditional take profit and stop loss levels
    - Optimized specifically for WIF with manual grid search
    - Uses 1-minute resampling for more granular backtesting
    """
    
    # Strategy Parameters (to be optimized via manual grid search)
    s_liq_entry_thresh = 100000    # S LIQ volume threshold for entry
    entry_time_window_mins = 20    # Lookback window for S LIQ entry signal (minutes)
    take_profit = 0.02             # Take profit percentage
    stop_loss = 0.01               # Stop loss percentage

    def init(self):
        """Initialize strategy with S LIQ volume data"""
        self.s_liq_volume = self.data.s_liq_volume
        logger.debug(f"WIFLongOnSLiqStrategy initialized with {len(self.data)} data points")

    def next(self):
        """Execute strategy logic for each bar"""
        try:
            current_time = self.data.index[-1]

            # Entry Logic - only check if not in position
            if not self.position:
                entry_start_time = current_time - pd.Timedelta(minutes=self.entry_time_window_mins)
                entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
                recent_s_liquidations = self.s_liq_volume[entry_start_idx:].sum()

                # Enter long if S LIQ exceeds threshold
                if recent_s_liquidations >= self.s_liq_entry_thresh:
                    sl_price = self.data.Close[-1] * (1 - self.stop_loss)
                    tp_price = self.data.Close[-1] * (1 + self.take_profit)
                    self.buy(sl=sl_price, tp=tp_price)
                    
                    logger.debug(f"WIFLongOnSLiqStrategy position opened at {self.data.Close[-1]:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")
                    
            # Note: No additional exit logic based on L LIQ in this strategy, only TP/SL.
                    
        except Exception as e:
            logger.error(f"Error in WIFLongOnSLiqStrategy.next(): {e}")

class ShortLiquidationStrategy(Strategy):
    """
    Short Liquidation Strategy (from liq_bt_short_alphadecay.py)
    
    Strategy Logic:
    - Enters SHORT positions when S LIQ (short liquidation) volume exceeds threshold
    - Exits SHORT positions when L LIQ (long liquidation) volume exceeds threshold
    - Also uses traditional take profit and stop loss levels
    - Unique: This is a SHORT-only strategy (opposite of long strategies)
    """
    
    # Strategy Parameters (defaults from original script)
    short_liquidation_thresh = 100000    # S LIQ volume threshold to trigger SHORT entry
    entry_time_window_mins = 5           # Lookback window for S LIQ entry signal (minutes)
    long_liquidation_closure_thresh = 50000  # L LIQ volume threshold to trigger SHORT exit
    exit_time_window_mins = 5            # Lookback window for L LIQ exit signal (minutes)
    take_profit_pct = 0.02               # Take profit percentage for shorts (2% down)
    stop_loss_pct = 0.01                 # Stop loss percentage for shorts (1% up)

    def init(self):
        """Initialize strategy indicators and state"""
        self.short_liquidations = self.data.s_liq_volume
        self.long_liquidations = self.data.l_liq_volume
        self.trade_entry_bar = -1  # Track the bar index of entry
        logger.debug(f"ShortLiquidationStrategy initialized with {len(self.data)} data points")

    def next(self):
        """Execute strategy logic for each bar"""
        try:
            current_bar_idx = len(self.data.Close) - 1

            # Entry Logic (Short Only)
            if not self.position:  # Only check for entry if not already in a position
                entry_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.entry_time_window_mins)
                entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
                recent_short_liquidations = self.short_liquidations[entry_start_idx:].sum()

                # Enter SHORT if S LIQ volume exceeds threshold
                if recent_short_liquidations >= self.short_liquidation_thresh:
                    sl_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
                    tp_price = self.data.Close[-1] * (1 - self.take_profit_pct)
                    self.sell(sl=sl_price, tp=tp_price)
                    self.trade_entry_bar = current_bar_idx  # Record entry bar index
                    
                    logger.debug(f"ShortLiquidationStrategy SHORT position opened at {self.data.Close[-1]:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")

            # Exit Logic (based on L LIQ threshold, in addition to TP/SL)
            elif self.position.is_short:  # Only check for L LIQ exit if in a short position
                # Use a fixed window for exit signal check, independent of entry bar
                exit_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.exit_time_window_mins)
                exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
                recent_long_liquidations = self.long_liquidations[exit_start_idx:].sum()
                
                # Close position if recent L LIQ exceeds threshold
                if recent_long_liquidations >= self.long_liquidation_closure_thresh:
                    self.position.close()
                    logger.debug(f"ShortLiquidationStrategy position closed due to L LIQ threshold")
                    
        except Exception as e:
            logger.error(f"Error in ShortLiquidationStrategy.next(): {e}")

class DelayedLiquidationStrategy(ShortLiquidationStrategy):
    """
    Delayed Liquidation Strategy for Alpha Decay Testing (from liq_bt_short_alphadecay.py)
    
    Extends ShortLiquidationStrategy to introduce a delay between the signal 
    and the actual trade entry. Used for alpha decay testing.
    
    Strategy Logic:
    - Same as ShortLiquidationStrategy but with configurable entry delays
    - Tests how strategy performance degrades with increasing delays
    - Unique: Alpha decay testing capability
    """
    
    # This class variable is modified by the alpha decay test loop
    delay_minutes = 0

    def next(self):
        """Applies delay logic before executing the base strategy's next()."""
        current_bar_idx = len(self.data.Close) - 1
        
        # If we are already in a position OR have just entered on this bar, 
        # let the base class handle exits/management. Delay only affects entry.
        if self.position or getattr(self, 'trade_entry_bar', -1) == current_bar_idx:
             super().next()
             return
             
        # --- Delay Logic for Entry ---
        # Check if a potential SHORT entry signal occurred within the delay window.
        potential_entry_signal = False
        last_potential_entry_bar = -1
        
        for lookback in range(1, self.delay_minutes + 1):
            if current_bar_idx - lookback < 0: break # Bounds check
            
            prev_time = self.data.index[current_bar_idx - lookback]
            entry_start_time = prev_time - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            
            # Check the short entry condition on the past bar
            past_short_liquidations = self.short_liquidations[entry_start_idx : current_bar_idx - lookback + 1].sum()
            
            if past_short_liquidations >= self.short_liquidation_thresh:
                potential_entry_signal = True
                last_potential_entry_bar = current_bar_idx - lookback
                break # Found the most recent signal within the delay window

        # If a signal occurred within the delay window, but we haven't entered yet
        if potential_entry_signal and getattr(self, 'trade_entry_bar', -1) < last_potential_entry_bar:
            # Calculate bars passed since the signal should have occurred
            bars_since_signal = current_bar_idx - last_potential_entry_bar
            
            # Check if the delay period has passed
            if bars_since_signal >= self.delay_minutes:
                 # Delay is over, execute the SHORT entry logic NOW using current price
                 if not self.position: # Double check we aren't in a position
                    sl_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
                    tp_price = self.data.Close[-1] * (1 - self.take_profit_pct)
                    self.sell(sl=sl_price, tp=tp_price)
                    self.trade_entry_bar = current_bar_idx # Mark entry bar as NOW
                    # We entered due to delay, so return here and don't run super().next() for this bar
                    return 
        
        # If no potential signal was found in the lookback, or the delay hasn't passed, 
        # or we already processed an entry for the signal, 
        # run the base logic normally to check for a *new* signal on the *current* bar.
        super().next()

class ShortOnSLiqStrategy(Strategy):
    """
    Short on S LIQ Strategy (from liq_bt_short_monte.py)
    
    Strategy Logic:
    - Enters SHORT positions when S LIQ (short liquidation) volume exceeds threshold
    - Exits SHORT positions when L LIQ (long liquidation) volume exceeds threshold
    - Also uses traditional take profit and stop loss levels
    - Simplified version of ShortLiquidationStrategy focused on single backtest runs
    """
    
    # Strategy Parameters (defaults from original script)
    short_liquidation_thresh = 100000    # S LIQ volume threshold to trigger SHORT entry
    entry_time_window_mins = 5           # Lookback window for S LIQ entry signal (minutes)
    long_liquidation_closure_thresh = 50000  # L LIQ volume threshold to trigger SHORT exit
    exit_time_window_mins = 5            # Lookback window for L LIQ exit signal (minutes)
    take_profit_pct = 0.02               # Take profit percentage for shorts (2% down)
    stop_loss_pct = 0.01                 # Stop loss percentage for shorts (1% up)

    def init(self):
        """Initialize strategy indicators and state."""
        # Access liquidation data columns
        self.short_liquidations = self.data.s_liq_volume
        self.long_liquidations = self.data.l_liq_volume

    def next(self):
        """Define the logic executed on each bar."""
        # --- Entry Logic (Short Only) --- 
        if not self.position: # Only check for entry if not already in a position
            entry_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.entry_time_window_mins)
            entry_start_idx = np.searchsorted(self.data.index, entry_start_time, side='left')
            recent_short_liquidations = self.short_liquidations[entry_start_idx:].sum()

            # Enter SHORT if S LIQ volume exceeds threshold
            if recent_short_liquidations >= self.short_liquidation_thresh:
                sl_price = self.data.Close[-1] * (1 + self.stop_loss_pct)
                tp_price = self.data.Close[-1] * (1 - self.take_profit_pct)
                self.sell(sl=sl_price, tp=tp_price)

        # --- Exit Logic (based on L LIQ threshold, in addition to TP/SL) ---
        elif self.position.is_short: # Only check for L LIQ exit if in a short position
            exit_start_time = self.data.index[-1] - pd.Timedelta(minutes=self.exit_time_window_mins)
            exit_start_idx = np.searchsorted(self.data.index, exit_start_time, side='left')
            recent_long_liquidations = self.long_liquidations[exit_start_idx:].sum()
            
            # Close position if recent L LIQ exceeds threshold
            if recent_long_liquidations >= self.long_liquidation_closure_thresh:
                self.position.close(reason="L Liq Threshold Hit")

class KalmanBreakoutReversal(Strategy):
    """
    Kalman Filter Breakout Reversal Strategy (from bt_template.py)
    
    Strategy Logic:
    - Uses Kalman Filter to smooth price data and identify mean
    - Enters SHORT when price breaks ABOVE the filtered mean (anticipating reversal)
    - Enters LONG when price breaks BELOW the filtered mean (anticipating reversal)
    - Exits when price crosses back over the filtered mean
    - Contrarian/reversal approach rather than trend following
    """
    
    # Strategy Parameters (to be optimized)
    window = 50              # Window parameter (used for reference, Kalman filter doesn't directly use it)
    take_profit = 0.05       # Take profit percentage (5%)
    stop_loss = 0.03         # Stop loss percentage (3%)
    
    # Kalman Filter parameters
    observation_covariance = 1.0     # Measurement noise
    transition_covariance = 0.01     # Process noise
    
    def init(self):
        """Initialize the Kalman Filter."""
        if not KALMAN_AVAILABLE:
            raise ImportError("pykalman is required for KalmanBreakoutReversal strategy")
            
        # Simple Kalman Filter setup for price smoothing
        self.kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=self.observation_covariance,    # Measurement noise
            transition_covariance=self.transition_covariance       # Process noise
        )

        # Apply the Kalman filter to the closing prices
        try:
            self.filtered_state_means, _ = self.kf.filter(self.data.Close.values)
        except ValueError as e:
            logger.warning(f"Kalman filter initialization failed: {e}")
            # Handle cases with insufficient data for filtering
            self.filtered_state_means = np.full_like(self.data.Close, np.nan)

    def next(self):
        """Define the trading logic for each bar."""
        # Ensure enough data and valid filter output
        if len(self.data.Close) < 2 or np.isnan(self.filtered_state_means[-1]):
            return

        filtered_mean = self.filtered_state_means[-1]
        current_close = self.data.Close[-1]
        
        # --- Short Entry --- 
        # If price breaks significantly above the filtered mean, anticipate reversal (short)
        if not self.position.is_short and current_close > filtered_mean: 
             # Add small threshold to avoid noise
             if current_close > filtered_mean * 1.001 and not self.position:
                 self.sell(sl=current_close * (1 + self.stop_loss), 
                           tp=current_close * (1 - self.take_profit))
                 logger.debug(f"KalmanBreakoutReversal SHORT entry at {current_close:.2f}, filtered mean: {filtered_mean:.2f}")

        # --- Short Exit --- 
        # Close short if price reverts below the filtered mean
        elif self.position.is_short and current_close < filtered_mean:
            self.position.close(reason="Price reverted below KF mean")
            logger.debug(f"KalmanBreakoutReversal SHORT exit at {current_close:.2f}, filtered mean: {filtered_mean:.2f}")

        # --- Long Entry --- 
        # If price breaks significantly below the filtered mean, anticipate reversal (long)
        elif not self.position.is_long and current_close < filtered_mean: 
             # Add small threshold to avoid noise
             if current_close < filtered_mean * 0.999 and not self.position:
                 self.buy(sl=current_close * (1 - self.stop_loss), 
                          tp=current_close * (1 + self.take_profit))
                 logger.debug(f"KalmanBreakoutReversal LONG entry at {current_close:.2f}, filtered mean: {filtered_mean:.2f}")

        # --- Long Exit --- 
        # Close long if price reverts above the filtered mean
        elif self.position.is_long and current_close > filtered_mean:
            self.position.close(reason="Price reverted above KF mean")
            logger.debug(f"KalmanBreakoutReversal LONG exit at {current_close:.2f}, filtered mean: {filtered_mean:.2f}")

class LiquidationBacktestingService:
    """
    Service for running liquidation-based backtesting strategies
    """
    
    def __init__(self):
        self.settings = settings
        self.output_dir = self.settings.LIQUIDATION_DATA_OUTPUT_DIR
        self.symbols = self.settings.LIQUIDATION_DATA_SYMBOLS
        self.results: List[LiquidationBacktestResult] = []
        
        logger.info("Liquidation Backtesting Service initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Supported symbols: {self.symbols}")
    
    async def run_single_backtest(
        self,
        symbol: str,
        parameters: Dict[str, Any] = None,
        initial_cash: float = 100000,
        commission: float = 0.002,
        strategy_type: str = "liquidation"
    ) -> LiquidationBacktestResult:
        """
        Run a single backtest for a given symbol
        
        Args:
            symbol: Trading symbol (BTC, ETH, SOL, WIF, 1000PEPE)
            parameters: Strategy parameters override
            initial_cash: Initial capital
            commission: Trading commission
            strategy_type: Strategy type ("liquidation" or "longonlliq")
            
        Returns:
            LiquidationBacktestResult with backtest results
        """
        try:
            logger.info(f"Running {strategy_type} backtest for {symbol}")
            
            # Load data if not already loaded
            if 'data' not in locals():
                # Load different types of data based on strategy type
                if strategy_type.lower() == "kalman_reversal":
                    # Kalman Filter strategy uses OHLCV data
                    data = await self._load_and_prepare_ohlcv_data(symbol)
                elif strategy_type.lower() == "wif_longonsliq":
                    # WIF strategy uses special 1-minute resampled data
                    data = await self._load_and_prepare_data_for_wif(symbol)
                else:
                    # Default liquidation strategies use liquidation data
                    data = await self._load_and_prepare_data(symbol)
                
                if data is None or data.empty:
                    raise ValueError(f"No data available for {symbol}")

            # Select strategy class based on strategy type
            if strategy_type.lower() == "longonlliq":
                strategy_class = LongOnLLiqStrategy
            elif strategy_type.lower() == "longonsliq":
                strategy_class = LongOnSLiqStrategy
            elif strategy_type.lower() == "wif_longonsliq":
                strategy_class = WIFLongOnSLiqStrategy
            elif strategy_type.lower() == "short_liquidation":
                strategy_class = ShortLiquidationStrategy
            elif strategy_type.lower() == "alpha_decay":
                strategy_class = DelayedLiquidationStrategy
            elif strategy_type.lower() == "short_monte":
                strategy_class = ShortOnSLiqStrategy
            elif strategy_type.lower() == "kalman_reversal":
                if not KALMAN_AVAILABLE:
                    raise ValueError("pykalman is required for Kalman Filter strategy. Please install: pip install pykalman")
                strategy_class = KalmanBreakoutReversal
            else:  # default to liquidation strategy
                strategy_class = LiquidationStrategy
            
            # Create and run backtest
            bt = Backtest(data, strategy_class, cash=initial_cash, commission=commission)
            
            # Apply parameters to strategy
            result = bt.run(**parameters)
            
            # Process results
            backtest_result = await self._process_backtest_result(
                symbol, result, parameters, data, strategy_type
            )
            
            self.results.append(backtest_result)
            
            logger.info(f"Backtest completed for {symbol}")
            logger.info(f"Total Return: {backtest_result.total_return:.2%}")
            logger.info(f"Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {backtest_result.max_drawdown:.2%}")
            logger.info(f"Total Trades: {backtest_result.total_trades}")
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
            raise
    
    async def run_optimization(
        self,
        symbol: str,
        optimization_ranges: Dict[str, Any] = None,
        initial_cash: float = 100000,
        commission: float = 0.002,
        maximize: str = 'Equity Final [$]',
        strategy_type: str = "liquidation"
    ) -> LiquidationBacktestResult:
        """
        Run parameter optimization for liquidation strategy
        
        Args:
            symbol: Trading symbol
            optimization_ranges: Parameter ranges for optimization
            initial_cash: Initial capital
            commission: Trading commission
            maximize: Metric to maximize during optimization
            strategy_type: Strategy type ("liquidation" or "longonlliq")
            
        Returns:
            LiquidationBacktestResult with optimized results
        """
        try:
            logger.info(f"Running {strategy_type} optimization for {symbol}")
            
            # Load and prepare data based on strategy type
            if strategy_type.lower() == "kalman_reversal":
                # Kalman Filter strategy uses OHLCV data
                data = await self._load_and_prepare_ohlcv_data(symbol)
            else:
                # Default liquidation strategies use liquidation data
                data = await self._load_and_prepare_data(symbol)
            
            if data is None or data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Select strategy class and set default optimization ranges based on strategy type
            if strategy_type.lower() == "longonlliq":
                strategy_class = LongOnLLiqStrategy
                if optimization_ranges is None:
                    optimization_ranges = {
                        'l_liq_entry_thresh': range(10000, 500000, 10000),
                        'entry_time_window_mins': range(1, 11, 1),
                        's_liq_closure_thresh': range(10000, 500000, 10000),
                        'exit_time_window_mins': range(1, 11, 1),
                        'take_profit': [i / 1000 for i in range(5, 31, 5)],
                        'stop_loss': [i / 1000 for i in range(5, 31, 5)]
                    }
            elif strategy_type.lower() == "longonsliq":
                strategy_class = LongOnSLiqStrategy
                if optimization_ranges is None:
                    optimization_ranges = {
                        's_liq_entry_thresh': range(10000, 500000, 10000),
                        'entry_time_window_mins': range(1, 11, 1),
                        'take_profit': [i / 100 for i in range(1, 5, 1)],
                        'stop_loss': [i / 100 for i in range(1, 5, 1)]
                    }
            elif strategy_type.lower() == "kalman_reversal":
                if not KALMAN_AVAILABLE:
                    raise ValueError("pykalman is required for Kalman Filter strategy. Please install: pip install pykalman")
                strategy_class = KalmanBreakoutReversal
                if optimization_ranges is None:
                    optimization_ranges = {
                        'window': range(20, 101, 10),  # 20 to 100 by 10
                        'take_profit': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
                        'stop_loss': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
                        'observation_covariance': [0.1, 0.5, 1.0, 2.0, 5.0],
                        'transition_covariance': [0.001, 0.01, 0.1, 1.0]
                    }
            else:  # default to liquidation strategy
                strategy_class = LiquidationStrategy
                if optimization_ranges is None:
                    optimization_ranges = {
                        'liquidation_thresh': range(5000, 500000, 25000),
                        'time_window_mins': range(5, 60, 5),
                        'take_profit': [i / 100 for i in range(1, 5, 1)],
                        'stop_loss': [i / 100 for i in range(1, 5, 1)]
                    }
            
            # Create backtest instance
            bt = Backtest(data, strategy_class, cash=initial_cash, commission=commission)
            
            # Run optimization
            logger.info(f"Starting optimization with ranges: {optimization_ranges}")
            optimization_result = bt.optimize(**optimization_ranges, maximize=maximize)
            
            # Extract best parameters based on strategy type
            best_params = {}
            if hasattr(optimization_result, '_strategy'):
                strategy = optimization_result._strategy
                if strategy_type.lower() == "longonlliq":
                    best_params = {
                        'l_liq_entry_thresh': strategy.l_liq_entry_thresh,
                        'entry_time_window_mins': strategy.entry_time_window_mins,
                        's_liq_closure_thresh': strategy.s_liq_closure_thresh,
                        'exit_time_window_mins': strategy.exit_time_window_mins,
                        'take_profit': strategy.take_profit,
                        'stop_loss': strategy.stop_loss
                    }
                elif strategy_type.lower() == "longonsliq":
                    best_params = {
                        's_liq_entry_thresh': strategy.s_liq_entry_thresh,
                        'entry_time_window_mins': strategy.entry_time_window_mins,
                        'take_profit': strategy.take_profit,
                        'stop_loss': strategy.stop_loss
                    }
                elif strategy_type.lower() == "kalman_reversal":
                    best_params = {
                        'window': strategy.window,
                        'take_profit': strategy.take_profit,
                        'stop_loss': strategy.stop_loss,
                        'observation_covariance': strategy.observation_covariance,
                        'transition_covariance': strategy.transition_covariance
                    }
                else:
                    best_params = {
                        'liquidation_thresh': strategy.liquidation_thresh,
                        'time_window_mins': strategy.time_window_mins,
                        'take_profit': strategy.take_profit,
                        'stop_loss': strategy.stop_loss
                    }
            
            # Process results
            backtest_result = await self._process_backtest_result(
                symbol, optimization_result, best_params, data, strategy_type
            )
            
            # Add optimization details
            backtest_result.optimization_results = {
                'optimization_ranges': optimization_ranges,
                'maximize_metric': maximize,
                'best_parameters': best_params,
                'total_combinations_tested': len(list(self._generate_combinations(optimization_ranges)))
            }
            
            self.results.append(backtest_result)
            
            logger.info(f"Optimization completed for {symbol}")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Optimized return: {backtest_result.total_return:.2%}")
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error running optimization for {symbol}: {e}")
            raise
    
    async def run_multi_symbol_backtest(
        self,
        symbols: List[str] = None,
        parameters: Dict[str, Any] = None,
        run_optimization: bool = False,
        strategy_type: str = "liquidation"
    ) -> List[LiquidationBacktestResult]:
        """
        Run backtesting for multiple symbols
        
        Args:
            symbols: List of symbols to test
            parameters: Strategy parameters
            run_optimization: Whether to run optimization
            strategy_type: Strategy type
            
        Returns:
            List of LiquidationBacktestResult
        """
        try:
            if symbols is None:
                symbols = self.symbols
            
            results = []
            for symbol in symbols:
                logger.info(f"Running backtest for {symbol}")
                
                if run_optimization:
                    result = await self.run_optimization(
                        symbol=symbol,
                        strategy_type=strategy_type
                    )
                else:
                    result = await self.run_single_backtest(
                        symbol=symbol,
                        parameters=parameters,
                        strategy_type=strategy_type
                    )
                
                results.append(result)
                
            logger.info(f"Multi-symbol backtest completed for {len(results)} symbols")
            return results
            
        except Exception as e:
            logger.error(f"Error running multi-symbol backtest: {e}")
            raise

    async def run_wif_manual_optimization(
        self,
        symbol: str = "WIF",
        initial_cash: float = 100000,
        commission: float = 0.002
    ) -> LiquidationBacktestResult:
        """
        Run manual grid search optimization for WIF strategy (from liq_bt_wif.py)
        
        This method implements the exact manual optimization logic from the original script
        with progress tracking and comprehensive parameter testing.
        
        Args:
            symbol: Trading symbol (defaults to WIF)
            initial_cash: Initial capital
            commission: Trading commission
            
        Returns:
            LiquidationBacktestResult with best optimization results
        """
        try:
            logger.info(f"Running WIF manual optimization for {symbol}")
            
            # Load and prepare data with 1-minute resampling (WIF-specific)
            data = await self._load_and_prepare_data_for_wif(symbol)
            if data is None or data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Define parameter ranges for manual search (from original script)
            s_liq_thresholds = range(10000, 200000, 20000)    # 10k to 190k by 20k steps
            time_windows = range(10, 61, 10)                  # 10 to 60 by 10 steps
            take_profits = [0.01, 0.02, 0.03, 0.04, 0.05]     # 1% to 5% by 1% steps
            stop_losses = [0.01, 0.02, 0.03, 0.04, 0.05]      # 1% to 5% by 1% steps
            
            # Initialize tracking variables
            best_equity = 0
            best_params = None
            best_stats = None
            total_combinations = len(s_liq_thresholds) * len(time_windows) * len(take_profits) * len(stop_losses)
            current_combo = 0
            optimization_results = []
            
            logger.info(f"Starting manual grid search with {total_combinations} parameter combinations")
            
            # Manual grid search
            for threshold in s_liq_thresholds:
                for window in time_windows:
                    for tp in take_profits:
                        for sl in stop_losses:
                            # Skip invalid combinations (ensure TP > SL)
                            if tp <= sl:
                                continue
                                
                            current_combo += 1
                            logger.debug(f"Testing combination {current_combo}/{total_combinations}: "
                                      f"threshold={threshold}, window={window}, tp={tp:.2f}, sl={sl:.2f}")
                            
                            # Create dynamic strategy class with current parameters
                            class CurrentWIFStrategy(WIFLongOnSLiqStrategy):
                                s_liq_entry_thresh = threshold
                                entry_time_window_mins = window
                                take_profit = tp
                                stop_loss = sl
                            
                            try:
                                # Run backtest with current parameters
                                bt = Backtest(data, CurrentWIFStrategy, cash=initial_cash, commission=commission)
                                stats = bt.run()
                                
                                # Extract key metrics
                                final_equity = stats['Equity Final [$]']
                                
                                # Store optimization result
                                optimization_results.append({
                                    'parameters': {
                                        's_liq_entry_thresh': threshold,
                                        'entry_time_window_mins': window, 
                                        'take_profit': tp,
                                        'stop_loss': sl
                                    },
                                    'final_equity': final_equity,
                                    'return_pct': stats['Return [%]'],
                                    'max_drawdown': stats['Max. Drawdown [%]'],
                                    'sharpe_ratio': stats['Sharpe Ratio'],
                                    'win_rate': stats['Win Rate [%]'],
                                    'total_trades': stats['# Trades']
                                })
                                
                                # Track the best result
                                if final_equity > best_equity:
                                    best_equity = final_equity
                                    best_params = {
                                        's_liq_entry_thresh': threshold,
                                        'entry_time_window_mins': window, 
                                        'take_profit': tp,
                                        'stop_loss': sl
                                    }
                                    best_stats = stats
                                    logger.info(f"✓ New best parameters found! Final Equity: ${final_equity:.2f}")
                                    
                            except Exception as e:
                                logger.warning(f"Error testing combination {current_combo}: {e}")
                                continue
            
            if best_params is None:
                raise ValueError("No valid parameter combinations found during optimization")
            
            # Process the best result into our standard format
            backtest_result = await self._process_backtest_result(
                symbol=symbol,
                result=best_stats,
                parameters=best_params,
                data=data,
                strategy_type="wif_longonsliq"
            )
            
            # Add optimization results
            backtest_result.optimization_results = {
                'total_combinations_tested': len(optimization_results),
                'best_parameters': best_params,
                'best_final_equity': best_equity,
                'optimization_details': optimization_results[:100]  # Limit to first 100 for storage
            }
            
            self.results.append(backtest_result)
            
            logger.info(f"WIF manual optimization completed for {symbol}")
            logger.info(f"Best Final Equity: ${best_equity:.2f}")
            logger.info(f"Best Parameters: {best_params}")
            logger.info(f"Total combinations tested: {len(optimization_results)}")
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error running WIF manual optimization for {symbol}: {e}")
            raise

    async def run_alpha_decay_test(
        self,
        symbol: str = "SOL",
        delays: List[int] = None,
        initial_cash: float = 100000,
        commission: float = 0.002
    ) -> Dict[str, Any]:
        """
        Run alpha decay test for DelayedLiquidationStrategy (from liq_bt_short_alphadecay.py)
        
        Tests how strategy performance degrades with increasing entry delays.
        This is a unique feature that measures the time-sensitivity of liquidation signals.
        
        Args:
            symbol: Trading symbol (defaults to SOL as in original script)
            delays: List of delay values in minutes to test
            initial_cash: Initial capital
            commission: Trading commission
            
        Returns:
            Dictionary with alpha decay test results for all delays
        """
        try:
            logger.info(f"Running alpha decay test for {symbol}")
            
            # Default delays from original script
            if delays is None:
                delays = [0, 1, 2, 5, 10, 15, 30, 60]
            
            # Load and prepare data
            data = await self._load_and_prepare_data(symbol)
            if data is None or data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            alpha_decay_results = []
            
            logger.info(f"Testing {len(delays)} delay configurations: {delays}")
            
            for delay in delays:
                logger.info(f"Running backtest with {delay}-minute delay")
                
                try:
                    # Create dynamic strategy class with current delay
                    class CurrentDelayedStrategy(DelayedLiquidationStrategy):
                        delay_minutes = delay
                    
                    # Run backtest with current delay
                    bt = Backtest(data, CurrentDelayedStrategy, cash=initial_cash, commission=commission)
                    stats = bt.run()
                    
                    # Extract key metrics
                    delay_result = {
                        'delay_minutes': delay,
                        'final_equity': stats['Equity Final [$]'],
                        'total_return': stats['Return [%]'],
                        'max_drawdown': stats['Max. Drawdown [%]'],
                        'sharpe_ratio': stats['Sharpe Ratio'],
                        'win_rate': stats['Win Rate [%]'],
                        'total_trades': stats['# Trades'],
                        'best_trade': stats['Best Trade [%]'],
                        'worst_trade': stats['Worst Trade [%]'],
                        'avg_trade': stats['Avg. Trade [%]'],
                        'profit_factor': stats['Profit Factor']
                    }
                    
                    alpha_decay_results.append(delay_result)
                    logger.info(f"Delay {delay}min - Final Equity: ${delay_result['final_equity']:.2f}, Return: {delay_result['total_return']:.2f}%")
                    
                except Exception as e:
                    logger.warning(f"Error testing delay {delay}: {e}")
                    # Add error result to maintain consistency
                    alpha_decay_results.append({
                        'delay_minutes': delay,
                        'error': str(e),
                        'final_equity': 0,
                        'total_return': 0,
                        'max_drawdown': 0,
                        'sharpe_ratio': 0,
                        'win_rate': 0,
                        'total_trades': 0
                    })
            
            # Calculate alpha decay metrics
            valid_results = [r for r in alpha_decay_results if 'error' not in r]
            
            if len(valid_results) >= 2:
                # Calculate performance degradation
                baseline_return = valid_results[0]['total_return']  # 0-delay performance
                decay_analysis = {
                    'baseline_return_pct': baseline_return,
                    'performance_degradation': [],
                    'alpha_half_life_estimate': None
                }
                
                for result in valid_results[1:]:  # Skip 0-delay
                    degradation_pct = ((result['total_return'] - baseline_return) / baseline_return * 100) if baseline_return != 0 else 0
                    decay_analysis['performance_degradation'].append({
                        'delay_minutes': result['delay_minutes'],
                        'return_pct': result['total_return'],
                        'degradation_from_baseline_pct': degradation_pct
                    })
                
                # Simple alpha half-life estimation (where performance drops to 50% of baseline)
                for degradation in decay_analysis['performance_degradation']:
                    if degradation['degradation_from_baseline_pct'] <= -50:
                        decay_analysis['alpha_half_life_estimate'] = degradation['delay_minutes']
                        break
            else:
                decay_analysis = {'error': 'Insufficient valid results for alpha decay analysis'}
            
            # Create comprehensive result
            alpha_decay_test_result = {
                'symbol': symbol,
                'test_type': 'alpha_decay',
                'strategy_name': 'DelayedLiquidationStrategy',
                'delays_tested': delays,
                'total_tests_run': len(delays),
                'successful_tests': len(valid_results),
                'detailed_results': alpha_decay_results,
                'alpha_decay_analysis': decay_analysis,
                'test_timestamp': datetime.now().isoformat()
            }
            
            # Store result
            self.results.append(LiquidationBacktestResult(
                symbol=symbol,
                strategy_name="alpha_decay_test",
                parameters={'delays': delays},
                start_date=data.index[0].isoformat() if len(data) > 0 else datetime.now().isoformat(),
                end_date=data.index[-1].isoformat() if len(data) > 0 else datetime.now().isoformat(),
                total_return=valid_results[0]['total_return'] / 100 if valid_results else 0,
                sharpe_ratio=valid_results[0]['sharpe_ratio'] if valid_results else 0,
                max_drawdown=valid_results[0]['max_drawdown'] / 100 if valid_results else 0,
                win_rate=valid_results[0]['win_rate'] / 100 if valid_results else 0,
                total_trades=valid_results[0]['total_trades'] if valid_results else 0,
                final_equity=valid_results[0]['final_equity'] if valid_results else 0,
                best_trade=valid_results[0]['best_trade'] / 100 if valid_results else 0,
                worst_trade=valid_results[0]['worst_trade'] / 100 if valid_results else 0,
                avg_trade=valid_results[0]['avg_trade'] / 100 if valid_results else 0,
                profit_factor=valid_results[0]['profit_factor'] if valid_results else 0,
                equity_curve=[],
                trades=[],
                optimization_results=alpha_decay_test_result
            ))
            
            logger.info(f"Alpha decay test completed for {symbol}")
            logger.info(f"Baseline (0-delay) return: {baseline_return:.2f}%" if valid_results else "No valid results")
            if decay_analysis.get('alpha_half_life_estimate'):
                logger.info(f"Estimated alpha half-life: {decay_analysis['alpha_half_life_estimate']} minutes")
            
            return alpha_decay_test_result
            
        except Exception as e:
            logger.error(f"Error running alpha decay test for {symbol}: {e}")
            raise
    
    async def _load_and_prepare_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load and prepare liquidation data for backtesting
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Prepared DataFrame with OHLC and liquidation data (both L LIQ and S LIQ)
        """
        try:
            # Load liquidation data from processor output
            data_path = os.path.join(self.output_dir, f'{symbol}_liq_data.csv')
            
            if not os.path.exists(data_path):
                logger.warning(f"Liquidation data file not found: {data_path}")
                return None
            
            # Load data
            data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
            logger.info(f"Loaded {len(data)} liquidation records for {symbol}")
            
            # Keep only necessary columns
            if not all(col in data.columns for col in ['LIQ_SIDE', 'price', 'usd_size']):
                logger.error(f"Missing required columns in {data_path}")
                return None
                
            data = data[['LIQ_SIDE', 'price', 'usd_size']]
            
            # Create L LIQ and S LIQ volume columns using vectorized operations
            data['l_liq_volume'] = np.where(data['LIQ_SIDE'] == 'L LIQ', data['usd_size'], 0)
            data['s_liq_volume'] = np.where(data['LIQ_SIDE'] == 'S LIQ', data['usd_size'], 0)
            
            # Resample data to 1-minute frequency for backtesting
            agg_funcs = {
                'price': 'mean',         # Mean price of liquidations in the minute
                'l_liq_volume': 'sum',   # Sum of L LIQ volume in the minute
                's_liq_volume': 'sum'    # Sum of S LIQ volume in the minute
            }
            data_resampled = data.resample('T').agg(agg_funcs)
            
            # Create OHLC columns required by backtesting.py
            data_resampled['Open'] = data_resampled['price']
            data_resampled['High'] = data_resampled['price']
            data_resampled['Low'] = data_resampled['price']
            data_resampled['Close'] = data_resampled['price']
            
            # Handle missing data after resampling
            price_columns = ['price', 'Open', 'High', 'Low', 'Close']
            data_resampled[price_columns] = data_resampled[price_columns].ffill()
            volume_columns = ['l_liq_volume', 's_liq_volume']
            data_resampled[volume_columns] = data_resampled[volume_columns].fillna(0)
            
            # Ensure the DataFrame is sorted by index
            data_resampled.sort_index(inplace=True)
            
            # Remove rows with NaN in critical columns (like Close) before backtesting
            data_resampled.dropna(subset=['Close'], inplace=True)
            
            if data_resampled.empty or data_resampled['Close'].isnull().all():
                logger.warning(f"No valid data available for {symbol} after processing")
                return None
            
            logger.info(f"Prepared {len(data_resampled)} data points for {symbol} backtesting")
            logger.info(f"L LIQ volume sum: {data_resampled['l_liq_volume'].sum():,.0f}")
            logger.info(f"S LIQ volume sum: {data_resampled['s_liq_volume'].sum():,.0f}")
            
            logger.info(f"Prepared data for backtesting: {len(data_resampled)} bars")
            return data_resampled
            
        except Exception as e:
            logger.error(f"Error loading/preparing data for {symbol}: {e}")
            return None
    
    async def _load_and_prepare_data_for_wif(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load and prepare liquidation data for WIF strategy with 1-minute resampling
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame with resampled data or None if error
        """
        try:
            # Try different file name patterns
            potential_files = [
                f"{symbol.lower()}_liquidations.csv",
                f"{symbol.upper()}_liquidations.csv",
                f"{symbol}_liquidations.csv",
                f"liquidations_{symbol.lower()}.csv",
                f"liquidations_{symbol.upper()}.csv",
                f"liquidations_{symbol}.csv"
            ]
            
            data_file = None
            for filename in potential_files:
                file_path = os.path.join(self.output_dir, filename)
                if os.path.exists(file_path):
                    data_file = file_path
                    break
            
            if not data_file:
                logger.warning(f"No liquidation data file found for {symbol}")
                return None
            
            logger.info(f"Loading WIF liquidation data from {data_file}")
            
            # Load CSV data
            df = pd.read_csv(data_file)
            
            # Standard column mapping
            expected_columns = ['datetime', 'symbol', 'side', 'size']
            if not all(col in df.columns for col in expected_columns):
                logger.error(f"Missing required columns in {data_file}. Expected: {expected_columns}")
                return None
            
            # Process liquidation data
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            df = df.sort_index()
            
            # Filter for the specific symbol
            df = df[df['symbol'] == symbol]
            
            if df.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return None
            
            # Separate L LIQ and S LIQ  
            l_liq = df[df['side'] == 'buy'].copy()
            s_liq = df[df['side'] == 'sell'].copy()
            
            # For WIF strategy, we only need S LIQ volume (as per original script)
            # Create time series of S LIQ volumes only
            s_liq_volume = s_liq.groupby(s_liq.index.floor('1min'))['size'].sum().fillna(0)
            
            # Generate 1-minute price data (dummy OHLC data for WIF since it focuses on liquidations only)
            # This is a simplified approach - in production, you'd want real OHLC data
            date_range = pd.date_range(start=s_liq_volume.index.min(), 
                                     end=s_liq_volume.index.max(), 
                                     freq='1min')
            
            # Create OHLCV DataFrame with S LIQ volume data
            ohlcv_data = pd.DataFrame(index=date_range)
            ohlcv_data['Open'] = 100.0    # Dummy price data
            ohlcv_data['High'] = 100.5
            ohlcv_data['Low'] = 99.5  
            ohlcv_data['Close'] = 100.0
            ohlcv_data['Volume'] = 1000   # Dummy volume
            
            # Add liquidation volumes (resampled to 1-minute)
            ohlcv_data['s_liq_volume'] = s_liq_volume.reindex(date_range, fill_value=0)
            ohlcv_data['l_liq_volume'] = 0  # WIF strategy doesn't use L LIQ
            
            # Forward fill any missing data
            ohlcv_data = ohlcv_data.fillna(method='ffill')
            ohlcv_data = ohlcv_data.dropna()
            
            logger.info(f"WIF data prepared: {len(ohlcv_data)} rows, "
                       f"Date range: {ohlcv_data.index.min()} to {ohlcv_data.index.max()}")
            
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Error loading WIF liquidation data for {symbol}: {e}")
            return None
    
    async def _load_and_prepare_ohlcv_data(self, symbol: str = None, test_data_folder: str = "test_data") -> Optional[pd.DataFrame]:
        """
        Load and prepare OHLCV data from CSV files for Kalman Filter strategy (from bt_template.py)
        
        This method loads standard OHLCV data from CSV files, as used in the original bt_template.py script.
        It supports loading from a test_data folder with multiple CSV files.
        
        Args:
            symbol: Trading symbol (optional, can load from filename)
            test_data_folder: Folder name containing CSV files (relative to script location)
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            # Try to locate test_data folder relative to current directory or data directory
            potential_folders = [
                os.path.join(os.getcwd(), test_data_folder),
                os.path.join(self.output_dir, test_data_folder),
                os.path.join(os.path.dirname(__file__), "..", "..", test_data_folder),
                test_data_folder  # Absolute path
            ]
            
            data_folder = None
            for folder_path in potential_folders:
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    data_folder = folder_path
                    break
            
            if not data_folder:
                logger.warning(f"Test data folder '{test_data_folder}' not found in any of: {potential_folders}")
                return None
            
            # Find CSV files in the folder
            csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
            if not csv_files:
                logger.warning(f"No CSV files found in {data_folder}")
                return None
            
            # If symbol is provided, try to find a matching file
            target_file = None
            if symbol:
                for csv_file in csv_files:
                    if symbol.lower() in csv_file.lower():
                        target_file = csv_file
                        break
            
            # If no specific file found or no symbol provided, use the first CSV file
            if not target_file:
                target_file = csv_files[0]
                logger.info(f"Using first available CSV file: {target_file}")
            
            file_path = os.path.join(data_folder, target_file)
            logger.info(f"Loading OHLCV data from {file_path}")
            
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Handle different CSV formats
            if df.shape[1] == 6 and 'datetime' in df.columns:
                # Format: datetime, Open, High, Low, Close, Volume
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                # Ensure we have the standard OHLCV columns
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in expected_cols):
                    # Try to map columns positionally if headers don't match
                    if df.shape[1] >= 5:
                        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(df.columns[5:])
            elif df.shape[1] == 5:
                # Format: Open, High, Low, Close, Volume (no datetime column)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                # Create datetime index
                df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='1h')
            else:
                logger.error(f"Unsupported CSV format in {file_path}. Expected OHLCV data.")
                return None
            
            # Data validation
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required OHLCV columns in {file_path}. Found: {df.columns.tolist()}")
                return None
            
            # Basic data cleaning
            if df.isnull().values.any():
                logger.warning(f"NaN values found in {target_file}, performing forward fill")
                df = df.fillna(method='ffill')
                df = df.fillna(method='bfill')  # Backfill any remaining NaNs at start
            
            # Remove any remaining NaN rows
            df = df.dropna()
            
            if df.empty:
                logger.error(f"No valid data remaining after cleaning in {file_path}")
                return None
            
            # Basic OHLC validation
            invalid_rows = (df['High'] < df['Low']) | (df['High'] < df['Open']) | (df['High'] < df['Close']) | \
                          (df['Low'] > df['Open']) | (df['Low'] > df['Close'])
            if invalid_rows.any():
                logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC relationships, removing them")
                df = df[~invalid_rows]
            
            # Sort by datetime
            df = df.sort_index()
            
            logger.info(f"OHLCV data loaded successfully: {len(df)} rows, "
                       f"Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"Sample data:\n{df.head()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading OHLCV data: {e}")
            return None
    
    async def _process_backtest_result(
        self,
        symbol: str,
        result,
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        strategy_type: str
    ) -> LiquidationBacktestResult:
        """
        Process backtesting result into standardized format
        """
        try:
            # Extract key metrics
            stats = result._results if hasattr(result, '_results') else {}
            
            # Calculate metrics safely
            def safe_get_stat(key, default=0.0):
                try:
                    value = getattr(result, key, stats.get(key, default))
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    return default
            
            backtest_result = LiquidationBacktestResult(
                symbol=symbol,
                strategy_name=strategy_type,
                parameters=parameters,
                start_date=data.index[0].isoformat() if len(data) > 0 else datetime.now().isoformat(),
                end_date=data.index[-1].isoformat() if len(data) > 0 else datetime.now().isoformat(),
                total_return=safe_get_stat('Return [%]') / 100,
                sharpe_ratio=safe_get_stat('Sharpe Ratio'),
                max_drawdown=safe_get_stat('Max. Drawdown [%]') / 100,
                win_rate=safe_get_stat('Win Rate [%]') / 100,
                total_trades=int(safe_get_stat('# Trades')),
                final_equity=safe_get_stat('Equity Final [$]'),
                best_trade=safe_get_stat('Best Trade [%]') / 100,
                worst_trade=safe_get_stat('Worst Trade [%]') / 100,
                avg_trade=safe_get_stat('Avg. Trade [%]') / 100,
                profit_factor=safe_get_stat('Profit Factor'),
                equity_curve=[],
                trades=[]
            )
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error processing backtest result: {e}")
            # Return minimal result structure
            return LiquidationBacktestResult(
                symbol=symbol,
                strategy_name=strategy_type,
                parameters=parameters,
                start_date=datetime.now().isoformat(),
                end_date=datetime.now().isoformat(),
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                final_equity=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                avg_trade=0.0,
                profit_factor=0.0,
                equity_curve=[],
                trades=[]
            )
    
    def _generate_combinations(self, ranges: Dict[str, Any]) -> List[Dict]:
        """Generate all parameter combinations for optimization"""
        keys = list(ranges.keys())
        values = list(ranges.values())
        
        import itertools
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    async def get_backtest_status(self) -> Dict[str, Any]:
        """Get current backtesting service status"""
        return {
            "service_enabled": True,
            "supported_symbols": self.symbols,
            "output_directory": self.output_dir,
            "completed_backtests": len(self.results),
            "available_data_files": await self._check_available_data_files()
        }
    
    async def _check_available_data_files(self) -> Dict[str, bool]:
        """Check which liquidation data files are available"""
        available_files = {}
        
        for symbol in self.symbols:
            data_path = os.path.join(self.output_dir, f'{symbol}_liq_data.csv')
            available_files[symbol] = os.path.exists(data_path)
        
        return available_files
    
    def get_recent_results(self, limit: int = 10) -> List[Dict]:
        """Get recent backtest results"""
        recent_results = self.results[-limit:] if self.results else []
        return [
            {
                "symbol": result.symbol,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades,
                "parameters": result.parameters,
                "start_date": result.start_date,
                "end_date": result.end_date
            }
            for result in recent_results
        ]


# Global service instance
liquidation_backtesting_service = LiquidationBacktestingService()

# Utility functions for external access
async def run_liquidation_backtest(
    symbol: str,
    parameters: Dict[str, Any] = None,
    run_optimization: bool = False
) -> LiquidationBacktestResult:
    """Run a liquidation backtest for a specific symbol"""
    if run_optimization:
        return await liquidation_backtesting_service.run_optimization(symbol)
    else:
        return await liquidation_backtesting_service.run_single_backtest(symbol, parameters)

async def get_liquidation_backtest_status() -> Dict[str, Any]:
    """Get liquidation backtesting service status"""
    return await liquidation_backtesting_service.get_backtest_status() 