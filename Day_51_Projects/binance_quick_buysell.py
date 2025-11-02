'''
Binance Quick Buy/Sell Trading Bot

This bot provides rapid buy and sell execution for Binance trading pairs.
It monitors a text file for token symbols and executes trades instantly.

Usage:
- Add token symbol to token_addresses.txt file to buy
- Add token symbol followed by 'x' or 'c' to sell (e.g., "BTC x")

Features:
- Instant market order execution
- Real-time file monitoring
- Position tracking and verification
- Professional error handling and logging
- Risk management and limits
'''

import time
import os
import sys
import hmac
import hashlib
import requests
import colorama
import pandas as pd
from datetime import datetime
from urllib.parse import urlencode
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from colorama import Fore, Style
from typing import Dict, Optional
from binance_CONFIG import get_app_config, validate_binance_qbs_config

# Initialize colorama for colored output
colorama.init()

def get_signature(query_string: str, api_secret: str) -> str:
    """Generate HMAC SHA256 signature for Binance API"""
    return hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def make_binance_request(config: Dict, endpoint: str, params: Dict = None, signed: bool = False, method: str = 'GET') -> Optional[Dict]:
    """Make authenticated or public request to Binance API"""
    if params is None:
        params = {}
    
    base_url = config.get('BINANCE_TESTNET_URL' if config.get('USE_TESTNET') else 'BINANCE_BASE_URL', 'https://api.binance.com')
    url = f"{base_url}{endpoint}"
    
    headers = {}
    
    if signed:
        api_key = config.get('BINANCE_API_KEY')
        api_secret = config.get('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print(f"{Fore.RED}❌ Error: Binance API credentials not configured{Style.RESET_ALL}")
            return None
        
        params['timestamp'] = int(time.time() * 1000)
        query_string = urlencode(params)
        signature = get_signature(query_string, api_secret)
        params['signature'] = signature
        
        headers['X-MBX-APIKEY'] = api_key
    
    try:
        if method == 'GET':
            response = requests.get(url, params=params, headers=headers, timeout=10)
        elif method == 'POST':
            response = requests.post(url, params=params, headers=headers, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}❌ Binance API request failed: {e}{Style.RESET_ALL}")
        return None

def get_symbol_info(config: Dict, symbol: str) -> Optional[Dict]:
    """Get trading symbol information from Binance"""
    endpoint = "/api/v3/exchangeInfo"
    data = make_binance_request(config, endpoint)
    
    if not data or 'symbols' not in data:
        return None
    
    for symbol_info in data['symbols']:
        if symbol_info['symbol'] == symbol:
            return symbol_info
    
    return None

def normalize_symbol(config: Dict, token_input: str) -> Optional[str]:
    """Convert token input to valid Binance symbol"""
    token_input = token_input.upper().strip()
    
    # Check if it's already a valid pair
    if token_input.endswith('USDT') or token_input.endswith('BUSD') or token_input.endswith('USDC'):
        return token_input
    
    # Check symbol mappings
    symbol_mappings = config.get('SYMBOL_MAPPINGS', {})
    if token_input in symbol_mappings:
        return symbol_mappings[token_input]
    
    # Try adding preferred quote asset
    preferred_quote = config.get('PREFERRED_QUOTE', 'USDT')
    candidate_symbol = f"{token_input}{preferred_quote}"
    
    # Validate symbol exists on exchange
    symbol_info = get_symbol_info(config, candidate_symbol)
    if symbol_info and symbol_info.get('status') == 'TRADING':
        return candidate_symbol
    
    # Try other quote assets
    for quote in config.get('QUOTE_ASSETS', ['USDT', 'BUSD', 'USDC']):
        if quote != preferred_quote:
            candidate_symbol = f"{token_input}{quote}"
            symbol_info = get_symbol_info(config, candidate_symbol)
            if symbol_info and symbol_info.get('status') == 'TRADING':
                return candidate_symbol
    
    return None

def get_account_balance(config: Dict, asset: str = 'USDT') -> float:
    """Get account balance for specific asset"""
    endpoint = "/api/v3/account"
    data = make_binance_request(config, endpoint, signed=True)
    
    if not data or 'balances' not in data:
        return 0.0
    
    for balance in data['balances']:
        if balance['asset'] == asset:
            return float(balance['free'])
    
    return 0.0

def get_position(config: Dict, symbol: str) -> float:
    """Get current position value in quote asset (USDT)"""
    try:
        # Extract base asset from symbol
        base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        
        # Get balance
        balance = get_account_balance(config, base_asset)
        
        if balance <= 0:
            return 0.0
        
        # Get current price
        price_data = make_binance_request(config, "/api/v3/ticker/price", {"symbol": symbol})
        if not price_data:
            return 0.0
        
        current_price = float(price_data['price'])
        return balance * current_price
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error getting position for {symbol}: {e}{Style.RESET_ALL}")
        return 0.0

def market_buy(config: Dict, symbol: str, amount_usdt: float) -> Optional[Dict]:
    """Execute market buy order"""
    try:
        print(f"{Fore.CYAN}🟢 Executing market BUY: {symbol} for ${amount_usdt:.2f}{Style.RESET_ALL}")
        
        # Validate symbol
        symbol_info = get_symbol_info(config, symbol)
        if not symbol_info:
            print(f"{Fore.RED}❌ Invalid symbol: {symbol}{Style.RESET_ALL}")
            return None
        
        if symbol_info.get('status') != 'TRADING':
            print(f"{Fore.RED}❌ Symbol not trading: {symbol}{Style.RESET_ALL}")
            return None
        
        # Check do not trade list
        if symbol in config.get('DO_NOT_TRADE_LIST', []):
            print(f"{Fore.YELLOW}⚠️  Symbol on do not trade list: {symbol}{Style.RESET_ALL}")
            return None
        
        # Check available balance
        quote_asset = 'USDT'  # Default
        if symbol.endswith('BUSD'):
            quote_asset = 'BUSD'
        elif symbol.endswith('USDC'):
            quote_asset = 'USDC'
        
        available_balance = get_account_balance(config, quote_asset)
        
        if available_balance < amount_usdt:
            print(f"{Fore.RED}❌ Insufficient {quote_asset} balance: ${available_balance:.2f} < ${amount_usdt:.2f}{Style.RESET_ALL}")
            return None
        
        # Place market buy order
        endpoint = "/api/v3/order"
        params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET',
            'quoteOrderQty': f"{amount_usdt:.2f}"
        }
        
        result = make_binance_request(config, endpoint, params, signed=True, method='POST')
        
        if result:
            print(f"{Fore.GREEN}✅ Market BUY successful: {symbol}{Style.RESET_ALL}")
            log_trade(config, symbol, 'BUY', amount_usdt, result)
            return result
        else:
            print(f"{Fore.RED}❌ Market BUY failed: {symbol}{Style.RESET_ALL}")
            return None
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error in market_buy: {e}{Style.RESET_ALL}")
        return None

def market_sell(config: Dict, symbol: str, sell_percentage: float = None) -> Optional[Dict]:
    """Execute market sell order"""
    try:
        if sell_percentage is None:
            sell_percentage = config.get('SELL_AMOUNT_PERC', 0.8)
        
        print(f"{Fore.CYAN}🔴 Executing market SELL: {symbol} ({sell_percentage:.1%}){Style.RESET_ALL}")
        
        # Get current position
        base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        total_balance = get_account_balance(config, base_asset)
        
        if total_balance <= 0:
            print(f"{Fore.YELLOW}⚠️  No {base_asset} balance to sell{Style.RESET_ALL}")
            return None
        
        # Calculate sell quantity
        sell_quantity = total_balance * sell_percentage
        
        # Get symbol info for precision
        symbol_info = get_symbol_info(config, symbol)
        if not symbol_info:
            print(f"{Fore.RED}❌ Invalid symbol: {symbol}{Style.RESET_ALL}")
            return None
        
        # Round quantity to appropriate precision
        base_precision = symbol_info['baseAssetPrecision']
        sell_quantity = round(sell_quantity, base_precision)
        
        if sell_quantity <= 0:
            print(f"{Fore.YELLOW}⚠️  Sell quantity too small: {sell_quantity}{Style.RESET_ALL}")
            return None
        
        # Place market sell order
        endpoint = "/api/v3/order"
        params = {
            'symbol': symbol,
            'side': 'SELL',
            'type': 'MARKET',
            'quantity': f"{sell_quantity:.{base_precision}f}"
        }
        
        result = make_binance_request(config, endpoint, params, signed=True, method='POST')
        
        if result:
            print(f"{Fore.GREEN}✅ Market SELL successful: {sell_quantity:.{base_precision}f} {base_asset}{Style.RESET_ALL}")
            log_trade(config, symbol, 'SELL', sell_quantity, result)
            return result
        else:
            print(f"{Fore.RED}❌ Market SELL failed: {symbol}{Style.RESET_ALL}")
            return None
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error in market_sell: {e}{Style.RESET_ALL}")
        return None

def log_trade(config: Dict, symbol: str, side: str, amount: float, order_result: Dict):
    """Log trade information"""
    trade_data = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'side': side,
        'amount': amount,
        'order_id': order_result.get('orderId'),
        'status': order_result.get('status'),
        'exchange': 'Binance'
    }
    
    # Save to CSV
    trade_file = config.get('TRADE_HISTORY_CSV', 'csvs/binance/qbs_trades.csv')
    os.makedirs(os.path.dirname(trade_file), exist_ok=True)
    
    try:
        if os.path.exists(trade_file):
            df = pd.read_csv(trade_file)
            new_row = pd.DataFrame([trade_data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([trade_data])
        
        df.to_csv(trade_file, index=False)
        print(f"{Fore.CYAN}📊 Trade logged: {side} {symbol}{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error logging trade: {e}{Style.RESET_ALL}")

class TokenFileHandler(FileSystemEventHandler):
    """File system event handler for monitoring token address file"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.file_path = os.path.join(os.path.dirname(__file__), self.config['QBS_TOKEN_ADDRESSES_FILE'])
        self.last_processed = set()
        
        # Ensure file exists
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                f.write("# Add token symbols here\n")
                f.write("# Format: SYMBOL (to buy) or SYMBOL x (to sell)\n")
            print(f"{Fore.YELLOW}📄 Created token file: {self.file_path}{Style.RESET_ALL}")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path.endswith(self.config['QBS_TOKEN_ADDRESSES_FILE']):
            # Add small delay to ensure file write is complete
            time.sleep(0.1)
            self.process_new_tokens()
    
    def quick_buy(self, token_input: str):
        """Execute quick buy with minimal checks"""
        try:
            # Normalize symbol
            symbol = normalize_symbol(self.config, token_input)
            if not symbol:
                print(f"{Fore.RED}❌ Could not resolve symbol: {token_input}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.GREEN}🚀 QUICK BUY initiated for {symbol}{Style.RESET_ALL}")
            
            # Execute buy orders
            buy_amount = self.config.get('USDC_SIZE', 10)
            buys_per_batch = self.config.get('BUYS_PER_BATCH', 1)
            
            for i in range(buys_per_batch):
                print(f"{Fore.CYAN}📦 Buy order {i+1}/{buys_per_batch}{Style.RESET_ALL}")
                result = market_buy(self.config, symbol, buy_amount)
                
                if not result:
                    print(f"{Fore.RED}❌ Buy order {i+1} failed{Style.RESET_ALL}")
                    break
                
                if i < buys_per_batch - 1:
                    time.sleep(0.5)  # Brief pause between orders
            
            # Wait and check position
            time.sleep(self.config.get('CHECK_DELAY', 3))
            
            position_value = get_position(self.config, symbol)
            expected_min = buy_amount * buys_per_batch * 0.8  # 80% threshold
            
            if position_value >= expected_min:
                print(f"{Fore.GREEN}✅ Position verified: ${position_value:.2f}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️  Position may be incomplete: ${position_value:.2f}{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error in quick_buy: {e}{Style.RESET_ALL}")
    
    def quick_sell(self, token_input: str):
        """Execute quick sell with minimal checks"""
        try:
            # Normalize symbol
            symbol = normalize_symbol(self.config, token_input)
            if not symbol:
                print(f"{Fore.RED}❌ Could not resolve symbol: {token_input}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.RED}🚀 QUICK SELL initiated for {symbol}{Style.RESET_ALL}")
            
            # Check if we have a position
            position_value = get_position(self.config, symbol)
            min_position = self.config.get('MIN_POSITION_VALUE', 5)
            
            if position_value < min_position:
                print(f"{Fore.YELLOW}⚠️  Position too small to sell: ${position_value:.2f}{Style.RESET_ALL}")
                return
            
            # Execute sell orders
            sells_per_batch = self.config.get('SELLS_PER_BATCH', 2)
            sell_percentage_per_order = self.config.get('SELL_AMOUNT_PERC', 0.8) / sells_per_batch
            
            for i in range(sells_per_batch):
                print(f"{Fore.CYAN}📦 Sell order {i+1}/{sells_per_batch}{Style.RESET_ALL}")
                result = market_sell(self.config, symbol, sell_percentage_per_order)
                
                if not result:
                    print(f"{Fore.RED}❌ Sell order {i+1} failed{Style.RESET_ALL}")
                    break
                
                if i < sells_per_batch - 1:
                    time.sleep(0.5)  # Brief pause between orders
            
            # Wait and check remaining position
            time.sleep(self.config.get('CHECK_DELAY', 3))
            
            remaining_value = get_position(self.config, symbol)
            print(f"{Fore.CYAN}💰 Remaining position: ${remaining_value:.2f}{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error in quick_sell: {e}{Style.RESET_ALL}")
    
    def process_new_tokens(self):
        """Process new tokens from file"""
        try:
            if not os.path.exists(self.file_path):
                return
            
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
            
            current_token_addresses_for_set = set()
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                current_token_addresses_for_set.add(line)
                
                # Process only new entries
                if line in self.last_processed:
                    continue
                
                # Parse command
                parts = line.split()
                if not parts:
                    continue
                
                token_input = parts[0]
                command = None
                
                if len(parts) == 2:
                    command_char = parts[1].lower()
                    if command_char in ['x', 'c']:
                        command = 'sell'
                
                # Execute command
                if command == 'sell':
                    print(f"{Fore.MAGENTA}🔴 Sell command: {token_input}{Style.RESET_ALL}")
                    self.quick_sell(token_input)
                else:
                    print(f"{Fore.MAGENTA}🟢 Buy command: {token_input}{Style.RESET_ALL}")
                    self.quick_buy(token_input)
            
            self.last_processed = current_token_addresses_for_set
        
        except FileNotFoundError:
            print(f"{Fore.RED}❌ Token file not found: {self.file_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing token file: {e}{Style.RESET_ALL}")

def print_startup_info(config):
    """Print startup information"""
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🚀 BINANCE QUICK BUY/SELL BOT{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Lightning-fast trading execution for Binance{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print()
    
    # Configuration display
    print(f"{Fore.YELLOW}⚙️  Configuration:{Style.RESET_ALL}")
    print(f"   💰 Trade Size: ${config.get('USDC_SIZE', 'N/A')}")
    print(f"   📊 Slippage: {config.get('SLIPPAGE_BPS', 'N/A')} bps")
    print(f"   📁 Token File: {config.get('QBS_TOKEN_ADDRESSES_FILE', 'N/A')}")
    print(f"   🔄 Max Positions: {config.get('MAX_POSITIONS', 'N/A')}")
    print(f"   💸 Sell %: {config.get('SELL_AMOUNT_PERC', 0.8):.1%}")
    print()
    
    # Trading instructions
    print(f"{Fore.GREEN}📖 Instructions:{Style.RESET_ALL}")
    print(f"   🟢 Add token symbol to buy (e.g., 'BTC')")
    print(f"   🔴 Add 'x' or 'c' after symbol to sell (e.g., 'BTC x')")
    print(f"   📝 Edit {config.get('QBS_TOKEN_ADDRESSES_FILE')} to add commands")
    print()

if __name__ == "__main__":
    print(f"{Fore.CYAN}🌙 Binance Quick Buy/Sell Monitor Starting...{Style.RESET_ALL}")
    
    # Load and validate configuration
    config = get_app_config()
    validation_errors = validate_binance_qbs_config()
    
    if validation_errors:
        print(f"{Fore.RED}❌ Configuration Errors:{Style.RESET_ALL}")
        for error in validation_errors:
            print(f"   - {error}")
        print(f"{Fore.YELLOW}Please fix configuration errors before running.{Style.RESET_ALL}")
        sys.exit(1)
    
    print_startup_info(config)
    
    # Set up file monitoring
    token_file_to_watch = os.path.join(os.path.dirname(__file__), config['QBS_TOKEN_ADDRESSES_FILE'])
    print(f"{Fore.YELLOW}👀 Watching: {token_file_to_watch}{Style.RESET_ALL}")
    
    event_handler = TokenFileHandler(config)
    
    # Initial processing
    print(f"{Fore.MAGENTA}🔍 Initial file check...{Style.RESET_ALL}")
    event_handler.process_new_tokens()
    
    # Start file observer
    observer = Observer()
    watch_path = os.path.dirname(token_file_to_watch)
    observer.schedule(event_handler, path=watch_path, recursive=False)
    observer.start()
    
    print(f"{Fore.GREEN}👁️  File monitoring active. Ready for trading commands!{Style.RESET_ALL}")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}🛑 Shutting down gracefully...{Style.RESET_ALL}")
    finally:
        observer.stop()
        observer.join()
        print(f"{Fore.GREEN}✅ Bot stopped successfully{Style.RESET_ALL}")
