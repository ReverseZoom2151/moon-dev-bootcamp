"""
🚀 Binance Liquidation Hunter Bot - Moon Dev Style
🎯 Trading strategy: Target coins with largest liquidation imbalance for potential cascade liquidations
🔍 Analyzes whale positions and market data to determine trading bias
💥 Hunts for liquidation events to enter trades in the direction of the bias

Built with love by Moon Dev 🌙 ✨
Disclaimer: This is not financial advice. Use at your own risk.
"""

import sys
import os
import time
import schedule
import pandas as pd
import traceback
import colorama
import logging
from colorama import Fore
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dontshareconfig import binance_api_key, binance_api_secret
except ImportError:
    binance_api_key = os.getenv('BINANCE_API_KEY')
    binance_api_secret = os.getenv('BINANCE_SECRET_KEY')

# Import local modules
import binance_nice_funcs as n
from binance_api import BinanceAPI

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Moon Dev ASCII Art Banner
MOON_DEV_BANNER = rf"""{Fore.CYAN}
   __  ___                    ____           
  /  |/  /___  ____  ____    / __ \___  _  __
 / /|_/ / __ \/ __ \/ __ \  / / / / _ \| |/_/
/ /  / / /_/ / /_/ / / / / / /_/ /  __/>  <  
/_/  /_/\____/\____/_/ /_(_)____/\___/_/|_|  
                                             
{Fore.MAGENTA}🚀 Binance Liquidation Hunter Bot 🎯{Fore.RESET}
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceTradingBot:
    def __init__(self):
        # ===== CONFIGURATION =====
        self.symbol = 'BTCUSDT'  # Default symbol, can be changed as needed
        self.leverage = 5         # Leverage to use for trading
        self.position_size_usd = 10  # Position size in USD
        
        # Constants
        self.liquidation_lookback_minutes = 5  # Time period to look back for liquidations
        self.liquidation_trigger_amount = 100000  # $100k trigger amount for liquidations
        
        # Take profit and stop loss settings
        self.take_profit_percent = 1  # Take profit percentage
        self.stop_loss_percent = -6.0  # Stop loss percentage
        
        # Analysis constants
        self.min_position_value = 25000  # Only track positions with value >= $25,000
        self.top_n_positions = 25  # Number of top positions to display
        self.highlight_threshold = 2000000  # $2 million
        self.tokens_to_analyze = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
        self.analysis_interval_minutes = 15  # How often to refresh the bias analysis
        
        # API keys validation
        if not binance_api_key or not binance_api_secret:
            logger.error("Binance API keys not found in configuration.")
            sys.exit("Binance API keys not set.")
        
        # Initialize Binance API
        try:
            self.api = BinanceAPI(api_key=binance_api_key, api_secret=binance_api_secret)
        except Exception as e:
            logger.error(f"Failed to initialize Binance API: {e}")
            sys.exit("Invalid Binance API credentials.")
        
        # Trading state
        self.trading_bias = None  # 'long' or 'short'
        self.recommendation_text = "Waiting for initial analysis..."
        self.last_analysis_time = None
    
    def print_banner(self):
        """Print Moon Dev banner"""
        print(MOON_DEV_BANNER)
        print(f"{Fore.YELLOW}🎯 Hunting for liquidation cascades on Binance...{Fore.RESET}")
        print(f"{Fore.GREEN}💰 Target Position Size: ${self.position_size_usd}{Fore.RESET}")
        print(f"{Fore.BLUE}⚡ Leverage: {self.leverage}x{Fore.RESET}")
        print("=" * 80)
    
    def analyze_market(self):
        """
        Analyze market conditions and set trading bias based on liquidation risk analysis
        """
        try:
            print(f"\n{Fore.CYAN}🔍 Analyzing market conditions...{Fore.RESET}")
            
            # Get funding rates for bias analysis
            funding_data = self.api.get_funding_data()
            if not funding_data or funding_data.empty:
                print(f"{Fore.YELLOW}⚠️ Could not get funding data{Fore.RESET}")
                return
            
            # Analyze major symbols
            analysis_results = []
            
            for symbol in self.tokens_to_analyze:
                try:
                    # Get long/short ratios
                    ratio_data = self.api.get_agg_positions(symbol)
                    
                    # Get recent large transactions
                    large_txns = self.api.get_recent_transactions(symbol)
                    
                    # Get current price and order book
                    current_price = n.get_current_price(symbol)
                    ask, bid, l2_data = n.ask_bid(symbol)
                    
                    if current_price and ratio_data is not None and not ratio_data.empty:
                        latest_ratio = ratio_data.iloc[-1]
                        long_percentage = latest_ratio.get('long_percentage', 50)
                        
                        # Calculate liquidation risk
                        liquidation_risk = abs(long_percentage - 50)  # Distance from neutral
                        
                        # Get whale orders from order book
                        whale_analysis = self.analyze_whale_orders(l2_data, current_price)
                        
                        analysis_results.append({
                            'symbol': symbol,
                            'price': current_price,
                            'long_percentage': long_percentage,
                            'liquidation_risk': liquidation_risk,
                            'whale_bias': whale_analysis.get('bias', 'neutral'),
                            'whale_strength': whale_analysis.get('strength', 0),
                            'funding_pressure': self.get_funding_pressure(symbol, funding_data)
                        })
                        
                        print(f"📊 {symbol}: ${current_price:.2f} | Long: {long_percentage:.1f}% | Risk: {liquidation_risk:.1f}")
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {e}")
                    continue
            
            if not analysis_results:
                print(f"{Fore.RED}❌ No analysis results available{Fore.RESET}")
                return
            
            # Determine overall market bias
            df_analysis = pd.DataFrame(analysis_results)
            
            # Calculate aggregate bias
            total_liquidation_risk = df_analysis['liquidation_risk'].sum()
            weighted_long_bias = (df_analysis['long_percentage'] * df_analysis['liquidation_risk']).sum() / total_liquidation_risk
            
            # Determine trading bias
            if weighted_long_bias > 55:
                self.trading_bias = 'short'  # Too many longs, expect liquidations
                bias_color = Fore.RED
                bias_emoji = "📉"
            elif weighted_long_bias < 45:
                self.trading_bias = 'long'   # Too many shorts, expect liquidations
                bias_color = Fore.GREEN
                bias_emoji = "📈"
            else:
                self.trading_bias = None     # Neutral market
                bias_color = Fore.YELLOW
                bias_emoji = "⚖️"
            
            # Generate recommendation
            if self.trading_bias:
                self.recommendation_text = f"Market Bias: {self.trading_bias.upper()} | Weighted Long%: {weighted_long_bias:.1f}%"
                print(f"\n{bias_color}{bias_emoji} TRADING BIAS: {self.trading_bias.upper()}{Fore.RESET}")
                print(f"{Fore.CYAN}📈 Recommendation: {self.recommendation_text}{Fore.RESET}")
            else:
                self.recommendation_text = "Market neutral - no clear liquidation bias detected"
                print(f"\n{bias_color}{bias_emoji} MARKET NEUTRAL{Fore.RESET}")
            
            # Display top liquidation risks
            top_risks = df_analysis.nlargest(3, 'liquidation_risk')
            print(f"\n{Fore.MAGENTA}🎯 Top Liquidation Risks:{Fore.RESET}")
            for _, row in top_risks.iterrows():
                print(f"   {row['symbol']}: {row['liquidation_risk']:.1f} risk | {row['long_percentage']:.1f}% long")
            
            self.last_analysis_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            traceback.print_exc()
    
    def analyze_whale_orders(self, l2_data, current_price):
        """Analyze whale orders in the order book"""
        try:
            if not l2_data or 'bids' not in l2_data or 'asks' not in l2_data:
                return {'bias': 'neutral', 'strength': 0}
            
            whale_threshold = 50000  # $50k+ orders
            
            # Analyze bids (buy orders)
            whale_bids_value = 0
            for bid in l2_data['bids']:
                price, quantity = float(bid[0]), float(bid[1])
                value = price * quantity
                if value >= whale_threshold:
                    whale_bids_value += value
            
            # Analyze asks (sell orders)
            whale_asks_value = 0
            for ask in l2_data['asks']:
                price, quantity = float(ask[0]), float(ask[1])
                value = price * quantity
                if value >= whale_threshold:
                    whale_asks_value += value
            
            # Determine bias
            total_whale_value = whale_bids_value + whale_asks_value
            if total_whale_value > 0:
                bid_percentage = whale_bids_value / total_whale_value
                
                if bid_percentage > 0.6:
                    return {'bias': 'bullish', 'strength': bid_percentage}
                elif bid_percentage < 0.4:
                    return {'bias': 'bearish', 'strength': 1 - bid_percentage}
            
            return {'bias': 'neutral', 'strength': 0.5}
            
        except Exception as e:
            logger.warning(f"Error analyzing whale orders: {e}")
            return {'bias': 'neutral', 'strength': 0}
    
    def get_funding_pressure(self, symbol, funding_data):
        """Get funding rate pressure for a symbol"""
        try:
            if funding_data.empty:
                return 0
            
            symbol_funding = funding_data[funding_data['symbol'] == symbol]
            if symbol_funding.empty:
                return 0
            
            funding_rate = symbol_funding.iloc[0]['funding_rate_pct']
            return funding_rate
            
        except Exception as e:
            return 0
    
    def get_recent_liquidations(self):
        """
        Get recent liquidations from large trades analysis
        """
        try:
            liquidation_data = []
            
            for symbol in self.tokens_to_analyze[:3]:  # Check top 3 symbols
                large_trades = self.api.get_recent_transactions(symbol)
                
                if large_trades is not None and not large_trades.empty:
                    # Filter for very large trades (potential liquidations)
                    liquidation_threshold = large_trades['value_usd'].quantile(0.98)
                    potential_liquidations = large_trades[large_trades['value_usd'] >= liquidation_threshold]
                    
                    for _, trade in potential_liquidations.iterrows():
                        liquidation_data.append({
                            'symbol': symbol,
                            'side': trade['side'],
                            'amount_usd': trade['value_usd'],
                            'price': trade['price'],
                            'timestamp': trade['timestamp']
                        })
            
            return liquidation_data
            
        except Exception as e:
            logger.error(f"Error getting recent liquidations: {e}")
            return []
    
    def should_enter_trade(self, long_liq_amount, short_liq_amount):
        """
        Determine if we should enter a trade based on liquidation amounts and trading bias
        """
        try:
            if not self.trading_bias:
                return False, "No trading bias established"
            
            total_liquidations = long_liq_amount + short_liq_amount
            
            if total_liquidations < self.liquidation_trigger_amount:
                return False, f"Liquidation amount ${total_liquidations:,.0f} below trigger ${self.liquidation_trigger_amount:,.0f}"
            
            # Check if liquidations align with our bias
            if self.trading_bias == 'long' and short_liq_amount > long_liq_amount * 1.5:
                return True, f"SHORT liquidations dominating: ${short_liq_amount:,.0f} vs ${long_liq_amount:,.0f}"
            elif self.trading_bias == 'short' and long_liq_amount > short_liq_amount * 1.5:
                return True, f"LONG liquidations dominating: ${long_liq_amount:,.0f} vs ${short_liq_amount:,.0f}"
            
            return False, "Liquidations don't align with trading bias"
            
        except Exception as e:
            logger.error(f"Error in trade decision logic: {e}")
            return False, "Error in analysis"
    
    def execute_trade(self, symbol, side):
        """Execute a trade based on the signal"""
        try:
            print(f"\n{Fore.MAGENTA}🎯 EXECUTING TRADE{Fore.RESET}")
            print(f"Symbol: {symbol}")
            print(f"Side: {side}")
            print(f"Position Size: ${self.position_size_usd}")
            print(f"Leverage: {self.leverage}x")
            
            # Set leverage
            n.adjust_leverage(symbol, self.leverage)
            
            # Calculate position size
            position_size = n.adjust_leverage_usd_size(symbol, self.position_size_usd, self.leverage)
            
            if position_size <= 0:
                print(f"{Fore.RED}❌ Invalid position size calculated{Fore.RESET}")
                return False
            
            # Get current price and calculate entry
            current_price = n.get_current_price(symbol)
            if not current_price:
                print(f"{Fore.RED}❌ Could not get current price for {symbol}{Fore.RESET}")
                return False
            
            # Calculate limit price (slightly more aggressive)
            if side == 'long':
                limit_price = current_price * 1.001  # 0.1% above market
                is_buy = True
            else:
                limit_price = current_price * 0.999  # 0.1% below market
                is_buy = False
            
            # Place the order
            order_result = n.limit_order(symbol, is_buy, position_size, limit_price)
            
            if order_result:
                print(f"{Fore.GREEN}✅ Order placed successfully!{Fore.RESET}")
                print(f"Order ID: {order_result.get('orderId', 'N/A')}")
                
                # Set stop loss and take profit levels
                sl_price = current_price * (1 + self.stop_loss_percent/100) if side == 'long' else current_price * (1 - self.stop_loss_percent/100)
                tp_price = current_price * (1 + self.take_profit_percent/100) if side == 'long' else current_price * (1 - self.take_profit_percent/100)
                
                print(f"Entry: ${current_price:.6f}")
                print(f"Stop Loss: ${sl_price:.6f}")
                print(f"Take Profit: ${tp_price:.6f}")
                
                return True
            else:
                print(f"{Fore.RED}❌ Failed to place order{Fore.RESET}")
                return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            traceback.print_exc()
            return False
    
    def monitor_positions(self):
        """Monitor open positions and manage risk"""
        try:
            for symbol in self.tokens_to_analyze:
                position = n.get_position(symbol)
                
                if position:
                    pnl_percentage = position.get('percentage', 0)
                    
                    print(f"\n📊 Position: {symbol}")
                    print(f"   Side: {position['side']}")
                    print(f"   Size: {position['size']}")
                    print(f"   PnL: {pnl_percentage:.2f}%")
                    
                    # Check for profit taking or stop loss
                    closed = n.pnl_close(symbol, self.take_profit_percent, self.stop_loss_percent)
                    
                    if closed:
                        print(f"{Fore.YELLOW}📤 Position closed for {symbol}{Fore.RESET}")
                    
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")
    
    def run_bot_cycle(self):
        """
        Main bot function that runs on each cycle
        """
        try:
            print(f"\n{Fore.CYAN}{'='*50}{Fore.RESET}")
            print(f"{Fore.CYAN}🤖 Bot Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Fore.RESET}")
            print(f"{Fore.CYAN}{'='*50}{Fore.RESET}")
            
            # Analyze market conditions (every 15 minutes or if not done yet)
            if (not self.last_analysis_time or 
                (datetime.now() - self.last_analysis_time).total_seconds() > self.analysis_interval_minutes * 60):
                self.analyze_market()
            
            # Monitor existing positions
            self.monitor_positions()
            
            # Get recent liquidations
            liquidations = self.get_recent_liquidations()
            
            if liquidations:
                print(f"\n{Fore.YELLOW}⚡ Recent liquidations detected:{Fore.RESET}")
                
                # Aggregate liquidations by side
                long_liq_total = sum([liq['amount_usd'] for liq in liquidations if liq['side'] == 'SELL'])
                short_liq_total = sum([liq['amount_usd'] for liq in liquidations if liq['side'] == 'BUY'])
                
                print(f"Long Liquidations: ${long_liq_total:,.0f}")
                print(f"Short Liquidations: ${short_liq_total:,.0f}")
                
                # Determine if we should trade
                should_trade, reason = self.should_enter_trade(long_liq_total, short_liq_total)
                
                print(f"Trade Decision: {reason}")
                
                if should_trade and self.trading_bias:
                    # Check if we already have a position in the primary symbol
                    existing_position = n.get_position(self.symbol)
                    
                    if not existing_position:
                        success = self.execute_trade(self.symbol, self.trading_bias)
                        if success:
                            print(f"{Fore.GREEN}🚀 Trade executed based on liquidation cascade!{Fore.RESET}")
                    else:
                        print(f"{Fore.YELLOW}⏸️ Already have position in {self.symbol}{Fore.RESET}")
            else:
                print(f"\n{Fore.BLUE}💤 No significant liquidations detected{Fore.RESET}")
            
            # Display current status
            print(f"\n{Fore.MAGENTA}📊 Current Status:{Fore.RESET}")
            print(f"   Trading Bias: {self.trading_bias or 'None'}")
            print(f"   Last Analysis: {self.last_analysis_time.strftime('%H:%M:%S') if self.last_analysis_time else 'Never'}")
            print(f"   Recommendation: {self.recommendation_text}")
            
            # Display account balance
            balance = n.get_balance()
            print(f"   Account Balance: ${balance:.2f}")
            
        except Exception as e:
            logger.error(f"Error in bot cycle: {e}")
            traceback.print_exc()

def run_scheduled_bot():
    """Wrapper function for scheduled execution"""
    try:
        bot_instance.run_bot_cycle()
    except Exception as e:
        logger.error(f"Error in scheduled bot execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    bot_instance = BinanceTradingBot()
    
    # Display banner
    bot_instance.print_banner()
    
    print(f"{Fore.CYAN}🚀 Starting Binance Liquidation Hunter Bot...{Fore.RESET}")
    print(f"{Fore.YELLOW}📅 Bot will run every 2 minutes{Fore.RESET}")
    print(f"{Fore.YELLOW}📊 Market analysis every 15 minutes{Fore.RESET}")
    print(f"{Fore.RED}⚠️  Press Ctrl+C to stop the bot{Fore.RESET}")
    print("=" * 80)
    
    # Schedule the bot to run every 2 minutes
    schedule.every(2).minutes.do(run_scheduled_bot)
    
    # Run initial cycle
    print(f"{Fore.GREEN}🎬 Running initial analysis...{Fore.RESET}")
    bot_instance.run_bot_cycle()
    
    # Main loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}🛑 Bot stopped by user{Fore.RESET}")
        print(f"{Fore.YELLOW}👋 Thanks for using Moon Dev's Binance Liquidation Hunter!{Fore.RESET}")
        
        # Cancel all orders before exit
        try:
            n.cancel_all_orders()
            print(f"{Fore.GREEN}✅ All orders cancelled safely{Fore.RESET}")
        except:
            pass
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        traceback.print_exc()
