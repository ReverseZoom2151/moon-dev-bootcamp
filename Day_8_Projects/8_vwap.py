############# Coding VWAP indicator 2024

import logging, sys, os, ccxt, time, pandas as pd 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ta.momentum import *
from datetime import datetime
from Day_4_Projects.key_file import key as xP_KEY, secret as xP_SECRET

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"vwap_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger('vwap_trader')

class VWAPTrader:
    """
    A trading class that uses Volume Weighted Average Price (VWAP) indicator
    to make trading decisions on the Phemex exchange.
    """
    
    def __init__(self, symbol='BTC/USD:BTC', timeframe='15m', limit=100, sma_period=20,
                 target_profit=9, max_loss=-8, max_risk=1000):
        """
        Initialize the VWAP trading system
        
        Args:
            symbol (str): Trading symbol (default: 'BTC/USD:BTC')
            timeframe (str): Chart timeframe (default: '15m')
            limit (int): Number of candles to fetch (default: 100)
            sma_period (int): SMA period (default: 20)
            target_profit (float): Target profit percentage (default: 9)
            max_loss (float): Maximum loss percentage (default: -8)
            max_risk (float): Maximum risk in USD (default: 1000)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.sma_period = sma_period
        self.target_profit = target_profit
        self.max_loss = max_loss
        self.max_risk = max_risk
        self.order_params = {'timeInForce': 'PostOnly'}
        
        # Connect to exchange
        try:
            self.exchange = ccxt.phemex({
                'enableRateLimit': True, 
                'apiKey': xP_KEY,
                'secret': xP_SECRET
            })
            logger.info(f"Connected to Phemex exchange successfully")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise
        
        # Store position indices for quick lookup
        self.position_indices = {
            'BTC/USD:BTC': 4,
            'APEUSD': 2,
            'ETHUSD': 3,
            'DOGEUSD': 1,
            'u100000SHIBUSD': 0
        }
        
    def get_market_data(self):
        """
        Get current market prices (ask and bid)
        
        Returns:
            tuple: (ask_price, bid_price)
        """
        try:
            ob = self.exchange.fetch_order_book(self.symbol)
            
            if not ob['bids'] or not ob['asks']:
                logger.warning(f"Empty order book for {self.symbol}")
                return 0, 0
                
            bid = ob['bids'][0][0]
            ask = ob['asks'][0][0]
            
            logger.info(f"Current prices for {self.symbol} - Ask: {ask}, Bid: {bid}")
            return ask, bid
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return 0, 0
    
    def get_position_info(self):
        """
        Get current position information
        
        Returns:
            tuple: (open_positions, has_position, position_size, is_long, index_pos)
        """
        try:
            # Get position index for symbol
            index_pos = self.position_indices.get(self.symbol, None)
            if index_pos is None:
                logger.warning(f"Unknown symbol: {self.symbol}")
                return None, False, 0, None, None
            
            # Fetch positions from exchange
            params = {'type': 'swap', 'code': 'USD'}
            balance = self.exchange.fetch_balance(params=params)
            positions = balance['info']['data']['positions']
            
            # Check if positions data is valid
            if not positions or index_pos >= len(positions):
                logger.warning(f"No position data available for {self.symbol}")
                return positions, False, 0, None, index_pos
            
            # Extract position details
            position = positions[index_pos]
            side = position['side']
            size = position['size']
            
            # Determine position status
            if side == 'Buy':
                has_position = True
                is_long = True
            elif side == 'Sell':
                has_position = True
                is_long = False
            else:
                has_position = False
                is_long = None
                
            logger.info(f"Position status: has_position={has_position}, size={size}, is_long={is_long}")
            return positions, has_position, size, is_long, index_pos
            
        except Exception as e:
            logger.error(f"Error retrieving position information: {e}")
            return None, False, 0, None, None
    
    def close_position(self):
        """
        Close the current position for the symbol
        
        Returns:
            bool: True if position closed successfully, False otherwise
        """
        logger.info(f"Starting to close position for {self.symbol}")
        
        try:
            # Get current position details
            _, has_position, position_size, is_long, _ = self.get_position_info()
            
            if not has_position:
                logger.info("No position to close")
                return True
                
            logger.info(f"Attempting to close position: size={position_size}, long={is_long}")
            
            # Loop until position is closed or max attempts reached
            max_attempts = 5
            attempts = 0
            
            while has_position and attempts < max_attempts:
                attempts += 1
                logger.info(f"Close attempt {attempts}/{max_attempts}")
                
                # Cancel existing orders
                self.exchange.cancel_all_orders(self.symbol)
                
                # Get updated position data
                _, has_position, position_size, is_long, _ = self.get_position_info()
                
                if not position_size or float(position_size) == 0:
                    logger.info("Position already closed")
                    return True
                    
                position_size = int(position_size)
                
                # Get current market prices
                ask, bid = self.get_market_data()
                
                if ask == 0 or bid == 0:
                    logger.warning("Invalid market prices, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Place appropriate order to close position
                try:
                    if is_long is False:  # Short position: buy to close
                        self.exchange.create_limit_buy_order(
                            self.symbol, position_size, bid, self.order_params
                        )
                        logger.info(f"Created BUY order to close short position: {position_size} @ ${bid}")
                    elif is_long is True:  # Long position: sell to close
                        self.exchange.create_limit_sell_order(
                            self.symbol, position_size, ask, self.order_params
                        )
                        logger.info(f"Created SELL order to close long position: {position_size} @ ${ask}")
                    else:
                        logger.error("Unknown position direction")
                        return False
                except Exception as e:
                    logger.error(f"Error creating order: {e}")
                    time.sleep(5)
                    continue
                
                logger.info("Waiting 30 seconds for order to fill...")
                time.sleep(30)
                
                # Check if position is closed
                _, has_position, _, _, _ = self.get_position_info()
            
            # Check final result
            if has_position and attempts >= max_attempts:
                logger.warning(f"Failed to close position after {max_attempts} attempts")
                return False
                
            logger.info("Position closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in close_position: {e}")
            return False
    
    def check_pnl(self):
        """
        Check current profit/loss and take action if target or stop-loss hit
        
        Returns:
            tuple: (take_profit_triggered, in_position, position_size, is_long)
        """
        logger.info(f"Checking P&L for {self.symbol}")
        
        try:
            # Get position data
            positions_data = self.exchange.fetch_positions(params={'type': 'swap', 'code': 'USD'})
            
            # Get position index
            _, _, _, _, index_pos = self.get_position_info()
            if index_pos is None:
                logger.warning(f"Cannot find position index for {self.symbol}")
                return False, False, 0, None
                
            # Check if we have valid position data
            if not positions_data or index_pos >= len(positions_data):
                logger.warning("No position data available")
                return False, False, 0, None
                
            # Get details for the current position
            position = positions_data[index_pos]
            side = position['side']
            size = position['contracts']
            
            # If no position, return early
            if not size or float(size) == 0:
                logger.info("No active position")
                return False, False, 0, None
                
            # Get entry price and current price
            entry_price = float(position['entryPrice'])
            leverage = float(position['leverage'])
            current_price = self.get_market_data()[1]  # Use bid price
            
            logger.info(f"Position details: Side={side}, Entry=${entry_price}, Leverage={leverage}x, Current=${current_price}")
            
            # Calculate PnL
            if side == 'long':
                is_long = True
                diff = current_price - entry_price
            else:
                is_long = False
                diff = entry_price - current_price
            
            # Calculate percentage PnL
            try:
                perc = round(((diff / entry_price) * leverage), 10) * 100
            except ZeroDivisionError:
                logger.error("Entry price is zero, cannot calculate percentage")
                perc = 0
                
            logger.info(f"Current P&L: {perc}%")
            
            # Initialize result values
            take_profit_hit = False
            in_position = True
            
            # Check if we hit target or stop-loss
            if perc > 0:
                logger.info(f"In profit: {perc}%")
                if perc > self.target_profit:
                    logger.info(f"Take profit hit! {perc}% > {self.target_profit}%")
                    take_profit_hit = True
                    self.close_position()
            elif perc < 0:
                logger.info(f"In loss: {perc}%")
                if perc <= self.max_loss:
                    logger.info(f"Stop loss hit! {perc}% <= {self.max_loss}%")
                    self.close_position()
            
            return take_profit_hit, in_position, size, is_long
            
        except Exception as e:
            logger.error(f"Error in check_pnl: {e}")
            return False, False, 0, None
    
    def check_max_risk(self):
        """
        Check if current position exceeds maximum risk
        
        Returns:
            bool: True if risk is acceptable, False if too risky
        """
        logger.info("Checking position size against maximum risk")
        
        try:
            # Fetch balance and positions
            params = {'type': 'swap', 'code': 'USD'}
            balance = self.exchange.fetch_balance(params=params)
            positions = balance['info']['data']['positions']
            
            # Get position cost
            try:
                position = positions[0]  # First position
                pos_cost = float(position['posCost'])
                openpos_side = position['side']
                openpos_size = position['size']
            except (IndexError, KeyError, ValueError):
                logger.info("No active positions or unable to calculate position cost")
                return True
                
            logger.info(f"Position cost: {pos_cost}, Side: {openpos_side}, Size: {openpos_size}")
            
            # Check against max risk
            if pos_cost > self.max_risk:
                logger.warning(f"EMERGENCY KILL SWITCH: Position cost {pos_cost} exceeds max risk {self.max_risk}")
                self.close_position()
                return False
            else:
                logger.info(f"Position risk acceptable: {pos_cost} < {self.max_risk}")
                return True
                
        except Exception as e:
            logger.error(f"Error in check_max_risk: {e}")
            return True  # Default to safe in case of errors
    
    def calculate_sma(self):
        """
        Calculate Simple Moving Average and generate signals
        
        Returns:
            DataFrame: Price data with SMA indicator
        """
        logger.info(f"Calculating SMA({self.sma_period}) for {self.symbol}")
        
        try:
            # Fetch OHLCV data
            bars = self.exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, limit=self.limit
            )
            
            if not bars:
                logger.warning("No price data received")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate SMA
            df[f'sma{self.sma_period}_{self.timeframe}'] = df['close'].rolling(self.sma_period).mean()
            
            # Get current bid price
            bid = self.get_market_data()[1]
            
            # Generate signals
            df.loc[df[f'sma{self.sma_period}_{self.timeframe}'] > bid, 'signal'] = 'SELL'
            df.loc[df[f'sma{self.sma_period}_{self.timeframe}'] < bid, 'signal'] = 'BUY'
            
            # Calculate support and resistance
            df['support'] = df['close'].rolling(self.limit).min()
            df['resistance'] = df['close'].rolling(self.limit).max()
            
            logger.info(f"SMA calculation completed with {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self):
        """
        Calculate Relative Strength Index
        
        Returns:
            DataFrame: Price data with RSI indicator
        """
        logger.info(f"Calculating RSI for {self.symbol}")
        
        try:
            # Fetch OHLCV data
            bars = self.exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, limit=self.limit
            )
            
            if not bars:
                logger.warning("No price data received")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate RSI
            rsi = RSIIndicator(df['close'])
            df['rsi'] = rsi.rsi()
            
            logger.info(f"RSI calculation completed")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.DataFrame()
    
    def calculate_vwap(self):
        """
        Calculate Volume Weighted Average Price
        
        Returns:
            DataFrame: Price data with VWAP indicator and signals
        """
        logger.info(f"Calculating VWAP for {self.symbol}")
        
        try:
            # Fetch OHLCV data
            bars = self.exchange.fetch_ohlcv(
                self.symbol, timeframe=self.timeframe, limit=self.limit
            )
            
            if not bars:
                logger.warning("No price data received")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate typical price
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP components
            df['volume_x_typical'] = df['volume'] * df['typical_price']
            df['cum_volume'] = df['volume'].cumsum()
            df['cum_vol_x_price'] = df['volume_x_typical'].cumsum()
            
            # Calculate VWAP
            df['vwap'] = df['cum_vol_x_price'] / df['cum_volume']
            
            # Generate VWAP signals (price above/below VWAP)
            current_price = self.get_market_data()[1]  # Use bid price
            last_vwap = df['vwap'].iloc[-1]
            
            df['vwap_signal'] = None
            if current_price > last_vwap:
                df.loc[df.index[-1], 'vwap_signal'] = 'BUY'
                logger.info(f"VWAP signal: BUY (Price ${current_price} > VWAP ${last_vwap})")
            else:
                df.loc[df.index[-1], 'vwap_signal'] = 'SELL'
                logger.info(f"VWAP signal: SELL (Price ${current_price} < VWAP ${last_vwap})")
            
            logger.info(f"VWAP calculation completed")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return pd.DataFrame()
    
    def get_combined_signal(self):
        """
        Combine signals from multiple indicators for a trading decision
        
        Returns:
            str: Combined signal ('BUY', 'SELL', or 'NEUTRAL')
        """
        logger.info("Generating combined trading signal")
        
        try:
            # Calculate indicators
            sma_df = self.calculate_sma()
            rsi_df = self.calculate_rsi()
            vwap_df = self.calculate_vwap()
            
            if sma_df.empty or rsi_df.empty or vwap_df.empty:
                logger.warning("Missing indicator data, cannot generate combined signal")
                return 'NEUTRAL'
            
            # Get latest signals
            sma_signal = sma_df['signal'].iloc[-1] if 'signal' in sma_df.columns else 'NEUTRAL'
            vwap_signal = vwap_df['vwap_signal'].iloc[-1] if 'vwap_signal' in vwap_df.columns else 'NEUTRAL'
            
            # Get latest RSI value
            rsi_value = rsi_df['rsi'].iloc[-1]
            rsi_signal = 'BUY' if rsi_value < 30 else 'SELL' if rsi_value > 70 else 'NEUTRAL'
            
            logger.info(f"Individual signals - SMA: {sma_signal}, VWAP: {vwap_signal}, RSI: {rsi_signal} ({rsi_value})")
            
            # Count buy/sell signals
            buy_count = sum(1 for signal in [sma_signal, vwap_signal, rsi_signal] if signal == 'BUY')
            sell_count = sum(1 for signal in [sma_signal, vwap_signal, rsi_signal] if signal == 'SELL')
            
            # Generate combined signal
            if buy_count >= 2:
                combined_signal = 'BUY'
            elif sell_count >= 2:
                combined_signal = 'SELL'
            else:
                combined_signal = 'NEUTRAL'
                
            logger.info(f"Combined signal: {combined_signal}")
            return combined_signal
            
        except Exception as e:
            logger.error(f"Error generating combined signal: {e}")
            return 'NEUTRAL'
    
    def execute_trade(self, signal, size=1):
        """
        Execute a trade based on the provided signal
        
        Args:
            signal (str): Trading signal ('BUY' or 'SELL')
            size (int): Position size
            
        Returns:
            bool: True if order placed successfully, False otherwise
        """
        logger.info(f"Executing trade: {signal} {size} {self.symbol}")
        
        try:
            # Check if we're already in a position
            _, has_position, _, is_long, _ = self.get_position_info()
            
            # Get current market prices
            ask, bid = self.get_market_data()
            
            if ask == 0 or bid == 0:
                logger.warning("Invalid market prices, cannot execute trade")
                return False
            
            # If we have a position, check if we need to close it first
            if has_position:
                if (signal == 'BUY' and not is_long) or (signal == 'SELL' and is_long):
                    logger.info("Closing existing opposite position first")
                    self.close_position()
                else:
                    logger.info("Already in position with same direction as signal, no action needed")
                    return True
            
            # Place order based on signal
            try:
                if signal == 'BUY':
                    self.exchange.create_limit_buy_order(
                        self.symbol, size, bid, self.order_params
                    )
                    logger.info(f"Created BUY order: {size} {self.symbol} @ ${bid}")
                elif signal == 'SELL':
                    self.exchange.create_limit_sell_order(
                        self.symbol, size, ask, self.order_params
                    )
                    logger.info(f"Created SELL order: {size} {self.symbol} @ ${ask}")
                else:
                    logger.info("Neutral signal, no order placed")
                    return False
                    
                return True
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in execute_trade: {e}")
            return False
            
    def run_trading_cycle(self, auto_trade=False, position_size=1):
        """
        Run a complete trading cycle: check positions, calculate indicators, get signals
        
        Args:
            auto_trade (bool): Whether to automatically execute trades
            position_size (int): Size for new positions
            
        Returns:
            dict: Trading cycle results
        """
        logger.info(f"Starting trading cycle for {self.symbol}")
        
        try:
            # Check max risk first (emergency kill switch)
            self.check_max_risk()
            
            # Check existing positions
            position_info = self.get_position_info()
            has_position = position_info[1]
            
            # If in position, check P&L
            if has_position:
                logger.info("In position, checking P&L")
                self.check_pnl()
            
            # Calculate indicators and get combined signal
            combined_signal = self.get_combined_signal()
            
            # Execute trade if auto trading enabled
            if auto_trade and combined_signal in ['BUY', 'SELL']:
                logger.info(f"Auto-trading enabled, executing {combined_signal} signal")
                self.execute_trade(combined_signal, position_size)
            
            # Return trading cycle results
            results = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'has_position': has_position,
                'combined_signal': combined_signal,
                'indicators': {
                    'sma': self.calculate_sma().tail(1).to_dict('records')[0] if not self.calculate_sma().empty else {},
                    'rsi': self.calculate_rsi().tail(1).to_dict('records')[0] if not self.calculate_rsi().empty else {},
                    'vwap': self.calculate_vwap().tail(1).to_dict('records')[0] if not self.calculate_vwap().empty else {}
                },
                'market_data': {
                    'ask': self.get_market_data()[0],
                    'bid': self.get_market_data()[1]
                }
            }
            
            logger.info(f"Trading cycle completed with signal: {combined_signal}")
            return results
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            return {'error': str(e)}

def main():
    """Main function to run the VWAP trader"""
    
    # Default trading parameters
    symbol = 'BTC/USD:BTC'
    timeframe = '15m'
    check_interval = 60  # seconds
    target_profit = 9
    max_loss = -8
    position_size = 1
    auto_trade = False  # Set to True to enable automated trading
    
    try:
        # Create VWAP trader instance
        trader = VWAPTrader(
            symbol=symbol,
            timeframe=timeframe,
            target_profit=target_profit,
            max_loss=max_loss
        )
        
        logger.info(f"VWAP Trader initialized for {symbol}")
        logger.info(f"Target profit: {target_profit}%, Max loss: {max_loss}%")
        logger.info(f"Auto-trading: {'Enabled' if auto_trade else 'Disabled'}")
        
        # Interactive menu
        print("\n====== VWAP Trading System ======")
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Target profit: {target_profit}%")
        print(f"Max loss: {max_loss}%")
        print(f"Auto-trading: {'Enabled' if auto_trade else 'Disabled'}")
        
        while True:
            print("\nOptions:")
            print("1. Check current position")
            print("2. Check indicators")
            print("3. Calculate VWAP")
            print("4. Run trading cycle")
            print("5. Close position")
            print("6. Toggle auto-trading")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ")
            
            if choice == '1':
                # Check current position
                positions, has_position, size, is_long, _ = trader.get_position_info()
                
                if has_position:
                    print(f"\nCurrent position: {'LONG' if is_long else 'SHORT'} {size} {symbol}")
                    trader.check_pnl()
                else:
                    print(f"\nNo open position for {symbol}")
                    
            elif choice == '2':
                # Check indicators
                print("\nCalculating indicators...")
                
                sma_df = trader.calculate_sma()
                rsi_df = trader.calculate_rsi()
                
                if not sma_df.empty and not rsi_df.empty:
                    print(f"\nLatest SMA: {sma_df[f'sma{trader.sma_period}_{timeframe}'].iloc[-1]:.2f}")
                    print(f"Latest RSI: {rsi_df['rsi'].iloc[-1]:.2f}")
                    print(f"SMA Signal: {sma_df['signal'].iloc[-1] if 'signal' in sma_df.columns else 'N/A'}")
                
            elif choice == '3':
                # Calculate VWAP
                print("\nCalculating VWAP...")
                
                vwap_df = trader.calculate_vwap()
                
                if not vwap_df.empty:
                    print(f"\nLatest VWAP: {vwap_df['vwap'].iloc[-1]:.2f}")
                    print(f"VWAP Signal: {vwap_df['vwap_signal'].iloc[-1] if 'vwap_signal' in vwap_df.columns else 'N/A'}")
                
            elif choice == '4':
                # Run trading cycle
                print("\nRunning trading cycle...")
                
                results = trader.run_trading_cycle(auto_trade=auto_trade, position_size=position_size)
                
                if 'error' not in results:
                    print(f"\nCombined signal: {results['combined_signal']}")
                    
                    if auto_trade and results['combined_signal'] in ['BUY', 'SELL']:
                        print(f"Auto-executed {results['combined_signal']} trade")
                
            elif choice == '5':
                # Close position
                print("\nClosing position...")
                
                success = trader.close_position()
                
                if success:
                    print("Position closed successfully")
                else:
                    print("Failed to close position")
                
            elif choice == '6':
                # Toggle auto-trading
                auto_trade = not auto_trade
                print(f"\nAuto-trading: {'Enabled' if auto_trade else 'Disabled'}")
                
            elif choice == '7':
                # Exit
                print("\nExiting VWAP Trading System...")
                break
                
            else:
                print("\nInvalid choice, please try again")
            
            # Pause before showing menu again
            input("\nPress Enter to continue...")
            
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("VWAP Trading System terminated")

if __name__ == "__main__":
    main()
