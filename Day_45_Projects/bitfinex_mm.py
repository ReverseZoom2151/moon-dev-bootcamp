"""
🏛️ Bitfinex Professional Liquidation Hunter Bot - Moon Dev Style
🎯 Institutional trading strategy: Target symbols with largest liquidation imbalance
🔍 Analyzes professional order flow and institutional positions for trading bias
💥 Hunts for liquidation events to enter trades with institutional-grade risk management

Built with love by Moon Dev 🌙 ✨
Disclaimer: This is not financial advice. Use at your own risk.
"""

import sys
import os
import time
import schedule
import pandas as pd
import numpy as np
import traceback
import colorama
import logging
from colorama import Fore
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dontshareconfig import bitfinex_api_key, bitfinex_api_secret
except ImportError:
    bitfinex_api_key = os.getenv('BITFINEX_API_KEY')
    bitfinex_api_secret = os.getenv('BITFINEX_SECRET_KEY')

# Import local modules
import bitfinex_nice_funcs as n
from bitfinex_api import BitfinexAPI

# Initialize colorama for terminal colors
colorama.init(autoreset=True)

# Moon Dev Professional ASCII Art Banner
MOON_DEV_BANNER = rf"""{Fore.CYAN}
   __  ___                    ____           
  /  |/  /___  ____  ____    / __ \___  _  __
 / /|_/ / __ \/ __ \/ __ \  / / / / _ \| |/_/
/ /  / / /_/ / /_/ / / / / / /_/ /  __/>  <  
/_/  /_/\____/\____/_/ /_(_)____/\___/_/|_|  
                                             
{Fore.MAGENTA}🏛️ Bitfinex Professional Liquidation Hunter 🎯{Fore.RESET}
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BitfinexProfessionalTradingBot:
    def __init__(self):
        # ===== INSTITUTIONAL CONFIGURATION =====
        self.symbol = 'tBTCUSD'  # Default symbol, can be changed as needed
        self.position_size_usd = 25  # Professional position size in USD
        
        # Professional constants
        self.liquidation_lookback_minutes = 10  # Longer lookback for institutional analysis
        self.liquidation_trigger_amount = 250000  # $250k trigger for institutional liquidations
        
        # Institutional risk management
        self.take_profit_percent = 2   # Conservative take profit
        self.stop_loss_percent = -4.0  # Tight stop loss
        
        # Professional analysis constants
        self.min_position_value = 50000  # Higher threshold for institutional tracking
        self.top_n_positions = 15  # Focus on top institutional positions
        self.highlight_threshold = 5000000  # $5 million institutional threshold
        self.tokens_to_analyze = ['tBTCUSD', 'tETHUSD', 'tLTCUSD', 'tXRPUSD', 'tADAUSD', 'tEOSUSD']
        self.analysis_interval_minutes = 20  # Longer interval for deeper analysis
        
        # API keys validation
        if not bitfinex_api_key or not bitfinex_api_secret:
            logger.error("Bitfinex API keys not found in configuration.")
            sys.exit("Bitfinex API keys not set.")
        
        # Initialize Bitfinex API
        try:
            self.api = BitfinexAPI(api_key=bitfinex_api_key, api_secret=bitfinex_api_secret)
        except Exception as e:
            logger.error(f"Failed to initialize Bitfinex API: {e}")
            sys.exit("Invalid Bitfinex API credentials.")
        
        # Professional trading state
        self.trading_bias = None  # 'long' or 'short'
        self.institutional_sentiment = "Analyzing institutional flow..."
        self.last_analysis_time = None
        self.confidence_score = 0  # Confidence in current bias (0-1)
    
    def print_banner(self):
        """Print professional Moon Dev banner"""
        print(MOON_DEV_BANNER)
        print(f"{Fore.YELLOW}🎯 Professional liquidation hunting on Bitfinex...{Fore.RESET}")
        print(f"{Fore.GREEN}💰 Institutional Position Size: ${self.position_size_usd}{Fore.RESET}")
        print(f"{Fore.BLUE}⚡ Professional Risk Management: {self.take_profit_percent}%/{self.stop_loss_percent}%{Fore.RESET}")
        print(f"{Fore.MAGENTA}🏛️ Minimum Position Value: ${self.min_position_value:,}{Fore.RESET}")
        print("=" * 80)
    
    def analyze_institutional_market(self):
        """
        Comprehensive institutional market analysis for liquidation risk assessment
        """
        try:
            print(f"\n{Fore.CYAN}🏛️ Conducting institutional market analysis...{Fore.RESET}")
            
            # Get professional funding data
            funding_data = self.api.get_funding_data()
            if not funding_data or funding_data.empty:
                print(f"{Fore.YELLOW}⚠️ Could not get institutional funding data{Fore.RESET}")
                return
            
            # Analyze major institutional symbols
            institutional_analysis = []
            
            for symbol in self.tokens_to_analyze:
                try:
                    # Get institutional position data
                    position_data = self.api.get_agg_positions(symbol)
                    
                    # Get recent institutional transactions
                    institutional_txns = self.api.get_recent_transactions(symbol)
                    
                    # Get current price and professional order book
                    current_price = n.get_current_price(symbol)
                    ask, bid, l2_data = n.ask_bid(symbol)
                    
                    if current_price and position_data is not None and not position_data.empty:
                        latest_position = position_data.iloc[-1]
                        long_percentage = latest_position.get('long_percentage', 50)
                        
                        # Calculate institutional liquidation risk
                        institutional_risk = self.calculate_institutional_risk(
                            long_percentage, current_price, symbol
                        )
                        
                        # Analyze institutional order flow
                        flow_analysis = n.institutional_order_flow_analysis(symbol)
                        
                        # Get professional whale analysis
                        whale_analysis = self.analyze_institutional_orders(l2_data, current_price)
                        
                        institutional_analysis.append({
                            'symbol': symbol,
                            'price': current_price,
                            'long_percentage': long_percentage,
                            'institutional_risk': institutional_risk,
                            'flow_bias': flow_analysis.get('institutional_bias', 'neutral') if flow_analysis else 'neutral',
                            'flow_strength': flow_analysis.get('net_flow', 0) if flow_analysis else 0,
                            'whale_tier': whale_analysis.get('dominant_tier', 'unknown'),
                            'whale_bias': whale_analysis.get('bias', 'neutral'),
                            'funding_pressure': self.get_institutional_funding_pressure(symbol, funding_data),
                            'liquidity_risk': whale_analysis.get('liquidity_risk', 0)
                        })
                        
                        print(f"🏛️ {symbol}: ${current_price:.2f} | Long: {long_percentage:.1f}% | Risk: {institutional_risk:.1f}")
                    
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {e}")
                    continue
            
            if not institutional_analysis:
                print(f"{Fore.RED}❌ No institutional analysis results available{Fore.RESET}")
                return
            
            # Determine institutional market bias
            df_analysis = pd.DataFrame(institutional_analysis)
            
            # Calculate institutional aggregate bias with confidence scoring
            total_risk = df_analysis['institutional_risk'].sum()
            if total_risk > 0:
                weighted_long_bias = (df_analysis['long_percentage'] * df_analysis['institutional_risk']).sum() / total_risk
                
                # Calculate confidence based on flow consistency
                flow_consistency = len(df_analysis[df_analysis['flow_bias'] == df_analysis['flow_bias'].mode()[0]]) / len(df_analysis)
                self.confidence_score = flow_consistency * 0.7 + (abs(weighted_long_bias - 50) / 50) * 0.3
            else:
                weighted_long_bias = 50
                self.confidence_score = 0
            
            # Determine professional trading bias
            if weighted_long_bias > 60 and self.confidence_score > 0.6:
                self.trading_bias = 'short'  # Institutional overextension - expect liquidations
                bias_color = Fore.RED
                bias_emoji = "📉"
            elif weighted_long_bias < 40 and self.confidence_score > 0.6:
                self.trading_bias = 'long'   # Institutional oversold - expect recovery
                bias_color = Fore.GREEN
                bias_emoji = "📈"
            else:
                self.trading_bias = None     # Insufficient conviction
                bias_color = Fore.YELLOW
                bias_emoji = "⚖️"
            
            # Generate institutional recommendation
            if self.trading_bias:
                self.institutional_sentiment = (f"Institutional Bias: {self.trading_bias.upper()} | "
                                              f"Long%: {weighted_long_bias:.1f}% | "
                                              f"Confidence: {self.confidence_score:.1f}")
                print(f"\n{bias_color}{bias_emoji} INSTITUTIONAL BIAS: {self.trading_bias.upper()}{Fore.RESET}")
                print(f"{Fore.CYAN}🏛️ Professional Assessment: {self.institutional_sentiment}{Fore.RESET}")
            else:
                self.institutional_sentiment = f"Institutional neutral - confidence too low ({self.confidence_score:.2f})"
                print(f"\n{bias_color}{bias_emoji} INSTITUTIONAL NEUTRAL{Fore.RESET}")
            
            # Display top institutional risks
            top_risks = df_analysis.nlargest(3, 'institutional_risk')
            print(f"\n{Fore.MAGENTA}🎯 Top Institutional Liquidation Risks:{Fore.RESET}")
            for _, row in top_risks.iterrows():
                print(f"   {row['symbol']}: {row['institutional_risk']:.1f} risk | {row['whale_tier']} dominant | {row['flow_bias']} flow")
            
            # Display funding pressure analysis
            high_pressure = df_analysis[abs(df_analysis['funding_pressure']) > 0.1]
            if not high_pressure.empty:
                print(f"\n{Fore.RED}💰 High Funding Pressure:{Fore.RESET}")
                for _, row in high_pressure.iterrows():
                    print(f"   {row['symbol']}: {row['funding_pressure']:.3f}% funding rate")
            
            self.last_analysis_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in institutional market analysis: {e}")
            traceback.print_exc()
    
    def calculate_institutional_risk(self, long_percentage, price, symbol):
        """Calculate sophisticated institutional liquidation risk score"""
        try:
            # Base risk from position imbalance
            base_risk = abs(long_percentage - 50) * 2
            
            # Get historical volatility
            df = n.get_ohclv(symbol, '1h', 168)  # 1 week of hourly data
            if not df.empty:
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(24)  # Daily volatility
                
                # Higher volatility increases liquidation risk
                volatility_multiplier = min(1 + volatility * 10, 2.0)
                base_risk *= volatility_multiplier
            
            # Professional position concentration risk
            if abs(long_percentage - 50) > 25:  # Extreme positioning
                base_risk *= 1.5
            
            return min(base_risk, 100)  # Cap at 100
            
        except Exception as e:
            logger.warning(f"Error calculating institutional risk: {e}")
            return abs(long_percentage - 50)
    
    def analyze_institutional_orders(self, l2_data, current_price):
        """Analyze institutional orders in the professional order book"""
        try:
            if not l2_data or 'bids' not in l2_data or 'asks' not in l2_data:
                return {'bias': 'neutral', 'dominant_tier': 'unknown', 'liquidity_risk': 0}
            
            # Institutional thresholds
            professional_threshold = 100000   # $100k+ professional orders
            institutional_threshold = 500000  # $500k+ institutional orders
            sovereign_threshold = 2000000     # $2M+ sovereign orders
            
            # Analyze institutional bids
            tier_analysis = {'Professional': 0, 'Institutional': 0, 'Sovereign': 0}
            total_bid_value = 0
            total_ask_value = 0
            
            for bid in l2_data['bids']:
                if len(bid) >= 3:
                    price, count, amount = float(bid[0]), int(bid[1]), abs(float(bid[2]))
                    value = price * amount
                    total_bid_value += value
                    
                    if value >= sovereign_threshold:
                        tier_analysis['Sovereign'] += value
                    elif value >= institutional_threshold:
                        tier_analysis['Institutional'] += value
                    elif value >= professional_threshold:
                        tier_analysis['Professional'] += value
            
            for ask in l2_data['asks']:
                if len(ask) >= 3:
                    price, count, amount = float(ask[0]), int(ask[1]), abs(float(ask[2]))
                    value = price * amount
                    total_ask_value += value
            
            # Determine dominant institutional tier
            dominant_tier = max(tier_analysis, key=tier_analysis.get)
            
            # Calculate bias and liquidity risk
            total_value = total_bid_value + total_ask_value
            bid_percentage = total_bid_value / total_value if total_value > 0 else 0.5
            
            bias = 'bullish' if bid_percentage > 0.55 else 'bearish' if bid_percentage < 0.45 else 'neutral'
            liquidity_risk = (max(tier_analysis.values()) / total_value) if total_value > 0 else 0
            
            return {
                'bias': bias,
                'dominant_tier': dominant_tier,
                'liquidity_risk': liquidity_risk,
                'institutional_dominance': sum(tier_analysis.values()) / total_value if total_value > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing institutional orders: {e}")
            return {'bias': 'neutral', 'dominant_tier': 'unknown', 'liquidity_risk': 0}
    
    def get_institutional_funding_pressure(self, symbol, funding_data):
        """Get institutional funding rate pressure analysis"""
        try:
            if funding_data.empty:
                return 0
            
            # Map trading symbols to funding symbols
            funding_map = {
                'tBTCUSD': 'fUSD', 'tETHUSD': 'fUSD', 'tLTCUSD': 'fUSD'
            }
            
            funding_symbol = funding_map.get(symbol, 'fUSD')
            symbol_funding = funding_data[funding_data['symbol'] == funding_symbol]
            
            if symbol_funding.empty:
                return 0
            
            # Get annualized funding rate
            daily_rate = symbol_funding.iloc[0].get('funding_rate_daily', 0)
            return daily_rate
            
        except Exception as e:
            return 0
    
    def get_institutional_liquidations(self):
        """
        Get recent institutional liquidations from large trade analysis
        """
        try:
            institutional_liquidation_data = []
            
            for symbol in self.tokens_to_analyze[:4]:  # Check top 4 symbols
                institutional_trades = self.api.get_recent_transactions(symbol)
                
                if institutional_trades is not None and not institutional_trades.empty:
                    # Filter for institutional-level trades (top 1% by value)
                    liquidation_threshold = institutional_trades['value_usd'].quantile(0.99)
                    potential_liquidations = institutional_trades[institutional_trades['value_usd'] >= liquidation_threshold]
                    
                    for _, trade in potential_liquidations.iterrows():
                        institutional_liquidation_data.append({
                            'symbol': symbol,
                            'side': trade['side'],
                            'amount_usd': trade['value_usd'],
                            'price': trade['price'],
                            'timestamp': trade['timestamp'],
                            'institutional_tier': self.classify_trade_tier(trade['value_usd'])
                        })
            
            return institutional_liquidation_data
            
        except Exception as e:
            logger.error(f"Error getting institutional liquidations: {e}")
            return []
    
    def classify_trade_tier(self, value_usd):
        """Classify trade into institutional tiers"""
        if value_usd >= 5000000:  # $5M+
            return 'Sovereign'
        elif value_usd >= 1000000:  # $1M+
            return 'Institutional'
        elif value_usd >= 250000:  # $250k+
            return 'Professional'
        else:
            return 'Large Retail'
    
    def should_enter_institutional_trade(self, long_liq_amount, short_liq_amount):
        """
        Professional trade decision logic based on institutional liquidation flow
        """
        try:
            if not self.trading_bias or self.confidence_score < 0.6:
                return False, f"Insufficient institutional conviction (confidence: {self.confidence_score:.2f})"
            
            total_liquidations = long_liq_amount + short_liq_amount
            
            if total_liquidations < self.liquidation_trigger_amount:
                return False, f"Institutional liquidation amount ${total_liquidations:,.0f} below trigger ${self.liquidation_trigger_amount:,.0f}"
            
            # Professional liquidation flow analysis
            liquidation_ratio = max(long_liq_amount, short_liq_amount) / min(long_liq_amount, short_liq_amount) if min(long_liq_amount, short_liq_amount) > 0 else float('inf')
            
            if self.trading_bias == 'long' and short_liq_amount > long_liq_amount and liquidation_ratio > 2:
                return True, f"Institutional SHORT liquidation cascade: ${short_liq_amount:,.0f} vs ${long_liq_amount:,.0f} (ratio: {liquidation_ratio:.1f})"
            elif self.trading_bias == 'short' and long_liq_amount > short_liq_amount and liquidation_ratio > 2:
                return True, f"Institutional LONG liquidation cascade: ${long_liq_amount:,.0f} vs ${short_liq_amount:,.0f} (ratio: {liquidation_ratio:.1f})"
            
            return False, "Liquidation flow doesn't meet institutional criteria"
            
        except Exception as e:
            logger.error(f"Error in institutional trade decision: {e}")
            return False, "Error in institutional analysis"
    
    def execute_institutional_trade(self, symbol, side):
        """Execute an institutional-grade trade with professional risk management"""
        try:
            print(f"\n{Fore.MAGENTA}🏛️ EXECUTING INSTITUTIONAL TRADE{Fore.RESET}")
            print(f"Symbol: {symbol}")
            print(f"Side: {side.upper()}")
            print(f"Position Size: ${self.position_size_usd}")
            print(f"Confidence Score: {self.confidence_score:.2f}")
            
            # Calculate professional position size
            position_size = n.adjust_leverage_usd_size(symbol, self.position_size_usd, 1)  # No leverage for professional trading
            
            if position_size <= 0:
                print(f"{Fore.RED}❌ Invalid institutional position size calculated{Fore.RESET}")
                return False
            
            # Get current price for institutional entry
            current_price = n.get_current_price(symbol)
            if not current_price:
                print(f"{Fore.RED}❌ Could not get current price for {symbol}{Fore.RESET}")
                return False
            
            # Calculate institutional limit price (more conservative)
            if side == 'long':
                limit_price = current_price * 1.0005  # 0.05% above market
                is_buy = True
            else:
                limit_price = current_price * 0.9995  # 0.05% below market
                is_buy = False
            
            # Place the institutional order
            order_result = n.spot_limit_order(symbol, is_buy, position_size, limit_price)
            
            if order_result:
                print(f"{Fore.GREEN}✅ Institutional order placed successfully!{Fore.RESET}")
                print(f"Order details: {order_result}")
                
                # Calculate professional stop loss and take profit levels
                if side == 'long':
                    sl_price = current_price * (1 + self.stop_loss_percent/100)
                    tp_price = current_price * (1 + self.take_profit_percent/100)
                else:
                    sl_price = current_price * (1 - self.stop_loss_percent/100)
                    tp_price = current_price * (1 - self.take_profit_percent/100)
                
                print(f"Entry: ${current_price:.6f}")
                print(f"Professional Stop Loss: ${sl_price:.6f}")
                print(f"Professional Take Profit: ${tp_price:.6f}")
                
                return True
            else:
                print(f"{Fore.RED}❌ Failed to place institutional order{Fore.RESET}")
                return False
            
        except Exception as e:
            logger.error(f"Error executing institutional trade: {e}")
            traceback.print_exc()
            return False
    
    def monitor_institutional_positions(self):
        """Monitor positions with institutional-grade risk management"""
        try:
            for symbol in self.tokens_to_analyze:
                position = n.get_position(symbol)
                
                if position:
                    pnl_percentage = position.get('percentage', 0)
                    
                    print(f"\n🏛️ Institutional Position: {symbol}")
                    print(f"   Side: {position['side'].upper()}")
                    print(f"   Size: {position['size']}")
                    print(f"   Professional PnL: {pnl_percentage:.2f}%")
                    print(f"   Tier: {self.classify_trade_tier(position['size'] * position.get('mark_price', position.get('entry_price', 0)))}")
                    
                    # Professional risk management
                    closed = n.pnl_close(symbol, self.take_profit_percent, self.stop_loss_percent)
                    
                    if closed:
                        print(f"{Fore.YELLOW}🏛️ Institutional position closed for {symbol}{Fore.RESET}")
                    
        except Exception as e:
            logger.error(f"Error monitoring institutional positions: {e}")
    
    def run_institutional_bot_cycle(self):
        """
        Main institutional bot function with professional-grade analysis
        """
        try:
            print(f"\n{Fore.CYAN}{'='*60}{Fore.RESET}")
            print(f"{Fore.CYAN}🏛️ Institutional Bot Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Fore.RESET}")
            print(f"{Fore.CYAN}{'='*60}{Fore.RESET}")
            
            # Institutional market analysis (every 20 minutes or initially)
            if (not self.last_analysis_time or 
                (datetime.now() - self.last_analysis_time).total_seconds() > self.analysis_interval_minutes * 60):
                self.analyze_institutional_market()
            
            # Monitor institutional positions
            self.monitor_institutional_positions()
            
            # Get institutional liquidations
            institutional_liquidations = self.get_institutional_liquidations()
            
            if institutional_liquidations:
                print(f"\n{Fore.YELLOW}⚡ Institutional liquidations detected:{Fore.RESET}")
                
                # Aggregate by tier and side
                long_liq_total = sum([liq['amount_usd'] for liq in institutional_liquidations if liq['side'] == 'SELL'])
                short_liq_total = sum([liq['amount_usd'] for liq in institutional_liquidations if liq['side'] == 'BUY'])
                
                # Analyze by institutional tier
                tier_breakdown = {}
                for liq in institutional_liquidations:
                    tier = liq['institutional_tier']
                    if tier not in tier_breakdown:
                        tier_breakdown[tier] = {'long': 0, 'short': 0}
                    if liq['side'] == 'SELL':
                        tier_breakdown[tier]['long'] += liq['amount_usd']
                    else:
                        tier_breakdown[tier]['short'] += liq['amount_usd']
                
                print(f"Institutional Long Liquidations: ${long_liq_total:,.0f}")
                print(f"Institutional Short Liquidations: ${short_liq_total:,.0f}")
                
                print(f"\n🏛️ Liquidation Breakdown by Tier:")
                for tier, amounts in tier_breakdown.items():
                    print(f"   {tier}: Long ${amounts['long']:,.0f} | Short ${amounts['short']:,.0f}")
                
                # Professional trade decision
                should_trade, reason = self.should_enter_institutional_trade(long_liq_total, short_liq_total)
                
                print(f"Institutional Decision: {reason}")
                
                if should_trade and self.trading_bias:
                    # Check existing institutional positions
                    existing_position = n.get_position(self.symbol)
                    
                    if not existing_position:
                        success = self.execute_institutional_trade(self.symbol, self.trading_bias)
                        if success:
                            print(f"{Fore.GREEN}🚀 Institutional trade executed on liquidation cascade!{Fore.RESET}")
                    else:
                        print(f"{Fore.YELLOW}⏸️ Already holding institutional position in {self.symbol}{Fore.RESET}")
            else:
                print(f"\n{Fore.BLUE}💤 No significant institutional liquidations detected{Fore.RESET}")
            
            # Display institutional status
            print(f"\n{Fore.MAGENTA}🏛️ Institutional Status:{Fore.RESET}")
            print(f"   Trading Bias: {self.trading_bias or 'Neutral'}")
            print(f"   Confidence: {self.confidence_score:.2f}")
            print(f"   Last Analysis: {self.last_analysis_time.strftime('%H:%M:%S') if self.last_analysis_time else 'Never'}")
            print(f"   Assessment: {self.institutional_sentiment}")
            
            # Display professional account info
            balance = n.get_balance()
            account_info = n.get_account_value()
            print(f"   Professional Balance: ${balance:.2f}")
            if account_info:
                print(f"   Total Wallet Value: ${account_info.get('total_wallet_balance', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error in institutional bot cycle: {e}")
            traceback.print_exc()

def run_scheduled_institutional_bot():
    """Wrapper function for scheduled institutional execution"""
    try:
        bot_instance.run_institutional_bot_cycle()
    except Exception as e:
        logger.error(f"Error in scheduled institutional bot execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    bot_instance = BitfinexProfessionalTradingBot()
    
    # Display professional banner
    bot_instance.print_banner()
    
    print(f"{Fore.CYAN}🏛️ Starting Bitfinex Professional Liquidation Hunter...{Fore.RESET}")
    print(f"{Fore.YELLOW}📅 Institutional bot runs every 5 minutes{Fore.RESET}")
    print(f"{Fore.YELLOW}📊 Deep market analysis every 20 minutes{Fore.RESET}")
    print(f"{Fore.RED}⚠️  Press Ctrl+C to stop the professional bot{Fore.RESET}")
    print("=" * 80)
    
    # Schedule the institutional bot to run every 5 minutes
    schedule.every(5).minutes.do(run_scheduled_institutional_bot)
    
    # Run initial institutional cycle
    print(f"{Fore.GREEN}🎬 Running initial institutional analysis...{Fore.RESET}")
    bot_instance.run_institutional_bot_cycle()
    
    # Professional main loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute (more conservative)
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}🛑 Professional bot stopped by user{Fore.RESET}")
        print(f"{Fore.YELLOW}👋 Thanks for using Moon Dev's Bitfinex Professional Hunter!{Fore.RESET}")
        
        # Cancel all orders professionally before exit
        try:
            n.cancel_all_orders()
            print(f"{Fore.GREEN}✅ All institutional orders cancelled safely{Fore.RESET}")
        except:
            pass
    except Exception as e:
        logger.error(f"Unexpected error in institutional main loop: {e}")
        traceback.print_exc()
