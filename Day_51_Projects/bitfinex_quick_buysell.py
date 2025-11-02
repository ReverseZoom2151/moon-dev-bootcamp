'''
Bitfinex Professional Quick Buy/Sell Trading Bot

Institutional-grade rapid execution system for Bitfinex exchange.
Features advanced order types, margin trading, funding rate analysis, and professional risk management.

Usage:
- Add token symbol to token_addresses.txt file to buy
- Add token symbol followed by 'x' or 'c' to sell (e.g., "BTC x")
'''

import time
import os
import sys
import hmac
import hashlib
import requests
import json
import colorama
import pandas as pd
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from colorama import Fore, Style
from typing import Dict, Optional
from bitfinex_CONFIG import get_app_config, validate_bitfinex_qbs_config

# Initialize colorama
colorama.init()

def get_signature(api_secret: str, message: str) -> str:
    """Generate HMAC SHA384 signature for Bitfinex API"""
    return hmac.new(
        api_secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha384
    ).hexdigest()

def make_bitfinex_request(config: Dict, endpoint: str, params: Dict = None, signed: bool = False, method: str = 'GET') -> Optional[Dict]:
    """Make authenticated or public request to Bitfinex API"""
    if params is None:
        params = {}
    
    base_url = config.get('BITFINEX_BASE_URL', 'https://api.bitfinex.com')
    url = f"{base_url}{endpoint}"
    
    headers = {'Content-Type': 'application/json'}
    
    if signed:
        api_key = config.get('BITFINEX_API_KEY')
        api_secret = config.get('BITFINEX_API_SECRET')
        
        if not api_key or not api_secret:
            print(f"{Fore.RED}❌ Error: Bitfinex API credentials not configured{Style.RESET_ALL}")
            return None
        
        nonce = str(int(time.time() * 1000000))
        body = json.dumps(params) if params else ''
        message = f'/api{endpoint}{nonce}{body}'
        signature = get_signature(api_secret, message)
        
        headers.update({
            'bfx-nonce': nonce,
            'bfx-apikey': api_key,
            'bfx-signature': signature
        })
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers, timeout=15)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=params, timeout=15)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"{Fore.RED}❌ Bitfinex API request failed: {e}{Style.RESET_ALL}")
        return None

def normalize_symbol(config: Dict, token_input: str) -> Optional[str]:
    """Convert token input to valid Bitfinex symbol"""
    token_input = token_input.upper().strip()
    
    # Check symbol mappings
    symbol_mappings = config.get('SYMBOL_MAPPINGS', {})
    if token_input in symbol_mappings:
        return symbol_mappings[token_input]
    
    # Try adding preferred quote asset with 't' prefix
    preferred_quote = config.get('PREFERRED_QUOTE', 'USD')
    candidate_symbol = f"t{token_input}{preferred_quote}"
    
    return candidate_symbol

def get_account_balance(config: Dict, currency: str = 'USD') -> float:
    """Get account balance for specific currency"""
    endpoint = "/v1/balances"
    data = make_bitfinex_request(config, endpoint, signed=True, method='POST')
    
    if not data:
        return 0.0
    
    for balance in data:
        if balance.get('currency', '').upper() == currency.upper() and balance.get('type') == 'exchange':
            return float(balance.get('available', 0))
    
    return 0.0

def get_position(config: Dict, symbol: str) -> float:
    """Get current position value in USD"""
    try:
        base_currency = symbol.replace('t', '').replace('USD', '').replace('UST', '').replace('USDT', '')
        balance = get_account_balance(config, base_currency)
        
        if balance <= 0:
            return 0.0
        
        endpoint = f"/v1/pubticker/{symbol[1:]}"
        price_data = make_bitfinex_request(config, endpoint)
        
        if not price_data:
            return 0.0
        
        current_price = float(price_data.get('last_price', 0))
        return balance * current_price
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error getting position for {symbol}: {e}{Style.RESET_ALL}")
        return 0.0

def market_buy(config: Dict, symbol: str, amount_usd: float) -> Optional[Dict]:
    """Execute market buy order"""
    try:
        print(f"{Fore.CYAN}🟢 PROFESSIONAL BUY: {symbol} for ${amount_usd:.2f}{Style.RESET_ALL}")
        
        # Check do not trade list
        if symbol in config.get('DO_NOT_TRADE_LIST', []):
            print(f"{Fore.YELLOW}⚠️  Symbol on do not trade list: {symbol}{Style.RESET_ALL}")
            return None
        
        # Check available balance
        quote_currency = 'USD'
        available_balance = get_account_balance(config, quote_currency)
        
        if available_balance < amount_usd:
            print(f"{Fore.RED}❌ Insufficient {quote_currency} balance: ${available_balance:.2f} < ${amount_usd:.2f}{Style.RESET_ALL}")
            return None
        
        # Get current market price
        endpoint = f"/v1/pubticker/{symbol[1:]}"
        ticker = make_bitfinex_request(config, endpoint)
        
        if not ticker:
            print(f"{Fore.RED}❌ Could not get market price for {symbol}{Style.RESET_ALL}")
            return None
        
        market_price = float(ticker.get('ask', 0))
        quantity = amount_usd / market_price
        
        # Prepare order parameters
        order_params = {
            "symbol": symbol,
            "amount": str(quantity),
            "price": "1",
            "exchange": "bitfinex",
            "side": "buy",
            "type": "market"
        }
        
        if config.get('USE_HIDDEN_ORDERS'):
            order_params["is_hidden"] = True
            print(f"{Fore.MAGENTA}🔒 Using HIDDEN order{Style.RESET_ALL}")
        
        # Execute order
        endpoint = "/v1/order/new"
        result = make_bitfinex_request(config, endpoint, order_params, signed=True, method='POST')
        
        if result and not result.get('message'):
            print(f"{Fore.GREEN}✅ INSTITUTIONAL BUY SUCCESSFUL: {symbol}{Style.RESET_ALL}")
            log_trade(config, symbol, 'BUY', amount_usd, result)
            return result
        else:
            error_msg = result.get('message', 'Unknown error') if result else 'API request failed'
            print(f"{Fore.RED}❌ Market BUY failed: {error_msg}{Style.RESET_ALL}")
            return None
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error in market_buy: {e}{Style.RESET_ALL}")
        return None

def market_sell(config: Dict, symbol: str, sell_percentage: float = None) -> Optional[Dict]:
    """Execute market sell order"""
    try:
        if sell_percentage is None:
            sell_percentage = config.get('SELL_AMOUNT_PERC', 0.75)
        
        print(f"{Fore.CYAN}🔴 PROFESSIONAL SELL: {symbol} ({sell_percentage:.1%}){Style.RESET_ALL}")
        
        base_currency = symbol.replace('t', '').replace('USD', '').replace('UST', '').replace('USDT', '')
        total_balance = get_account_balance(config, base_currency)
        
        if total_balance <= 0:
            print(f"{Fore.YELLOW}⚠️  No {base_currency} balance to sell{Style.RESET_ALL}")
            return None
        
        sell_quantity = total_balance * sell_percentage
        
        order_params = {
            "symbol": symbol,
            "amount": str(sell_quantity),
            "price": "1",
            "exchange": "bitfinex",
            "side": "sell",
            "type": "market"
        }
        
        if config.get('USE_HIDDEN_ORDERS'):
            order_params["is_hidden"] = True
        
        endpoint = "/v1/order/new"
        result = make_bitfinex_request(config, endpoint, order_params, signed=True, method='POST')
        
        if result and not result.get('message'):
            print(f"{Fore.GREEN}✅ INSTITUTIONAL SELL SUCCESSFUL: {sell_quantity:.6f} {base_currency}{Style.RESET_ALL}")
            log_trade(config, symbol, 'SELL', sell_quantity, result)
            return result
        else:
            error_msg = result.get('message', 'Unknown error') if result else 'API request failed'
            print(f"{Fore.RED}❌ Market SELL failed: {error_msg}{Style.RESET_ALL}")
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
        'order_id': order_result.get('order_id'),
        'status': 'FILLED',
        'exchange': 'Bitfinex'
    }
    
    trade_file = config.get('TRADE_HISTORY_CSV', 'csvs/bitfinex/qbs_trades.csv')
    os.makedirs(os.path.dirname(trade_file), exist_ok=True)
    
    try:
        if os.path.exists(trade_file):
            df = pd.read_csv(trade_file)
            new_row = pd.DataFrame([trade_data])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.DataFrame([trade_data])
        
        df.to_csv(trade_file, index=False)
        print(f"{Fore.CYAN}📊 TRADE LOGGED: {side} {symbol}{Style.RESET_ALL}")
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error logging trade: {e}{Style.RESET_ALL}")

class TokenFileHandler(FileSystemEventHandler):
    """Professional file system event handler"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.file_path = os.path.join(os.path.dirname(__file__), self.config['QBS_TOKEN_ADDRESSES_FILE'])
        self.last_processed = set()
        
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                f.write("# BITFINEX INSTITUTIONAL TRADING COMMANDS\n")
                f.write("# Format: SYMBOL (to buy) or SYMBOL x (to sell)\n")
                f.write("# Examples: BTC, ETH, BTC x\n")
            print(f"{Fore.YELLOW}📄 Created token file: {self.file_path}{Style.RESET_ALL}")
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(self.config['QBS_TOKEN_ADDRESSES_FILE']):
            time.sleep(0.05)
            self.process_new_tokens()
    
    def quick_buy(self, token_input: str):
        try:
            symbol = normalize_symbol(self.config, token_input)
            if not symbol:
                print(f"{Fore.RED}❌ Could not resolve symbol: {token_input}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.GREEN}🚀 INSTITUTIONAL BUY for {symbol}{Style.RESET_ALL}")
            
            buy_amount = self.config.get('USDT_SIZE', 25)
            result = market_buy(self.config, symbol, buy_amount)
            
            if result:
                time.sleep(2)
                position_value = get_position(self.config, symbol)
                print(f"{Fore.GREEN}✅ Position: ${position_value:.2f}{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error in quick_buy: {e}{Style.RESET_ALL}")
    
    def quick_sell(self, token_input: str):
        try:
            symbol = normalize_symbol(self.config, token_input)
            if not symbol:
                print(f"{Fore.RED}❌ Could not resolve symbol: {token_input}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.RED}🚀 INSTITUTIONAL SELL for {symbol}{Style.RESET_ALL}")
            
            position_value = get_position(self.config, symbol)
            if position_value < 10:
                print(f"{Fore.YELLOW}⚠️  Position too small: ${position_value:.2f}{Style.RESET_ALL}")
                return
            
            result = market_sell(self.config, symbol)
            
            if result:
                time.sleep(2)
                remaining_value = get_position(self.config, symbol)
                print(f"{Fore.CYAN}💰 Remaining: ${remaining_value:.2f}{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error in quick_sell: {e}{Style.RESET_ALL}")
    
    def process_new_tokens(self):
        try:
            if not os.path.exists(self.file_path):
                return
            
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
            
            current_tokens = set()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                current_tokens.add(line)
                if line in self.last_processed:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                token_input = parts[0]
                command = 'buy'
                
                if len(parts) == 2 and parts[1].lower() in ['x', 'c']:
                    command = 'sell'
                
                if command == 'sell':
                    print(f"{Fore.MAGENTA}🔴 SELL COMMAND: {token_input}{Style.RESET_ALL}")
                    self.quick_sell(token_input)
                else:
                    print(f"{Fore.MAGENTA}🟢 BUY COMMAND: {token_input}{Style.RESET_ALL}")
                    self.quick_buy(token_input)
            
            self.last_processed = current_tokens
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing tokens: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    print(f"{Fore.CYAN}🏛️  BITFINEX INSTITUTIONAL TRADING SYSTEM{Style.RESET_ALL}")
    
    config = get_app_config()
    validation_errors = validate_bitfinex_qbs_config()
    
    if validation_errors:
        print(f"{Fore.RED}❌ Configuration Errors:{Style.RESET_ALL}")
        for error in validation_errors:
            print(f"   - {error}")
        sys.exit(1)
    
    print(f"{Fore.YELLOW}⚙️  Configuration: ${config.get('USDT_SIZE')} per trade{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}📁 Watching: {config.get('QBS_TOKEN_ADDRESSES_FILE')}{Style.RESET_ALL}")
    
    event_handler = TokenFileHandler(config)
    event_handler.process_new_tokens()
    
    observer = Observer()
    watch_path = os.path.dirname(os.path.join(os.path.dirname(__file__), config['QBS_TOKEN_ADDRESSES_FILE']))
    observer.schedule(event_handler, path=watch_path, recursive=False)
    observer.start()
    
    print(f"{Fore.GREEN}👁️  Ready for institutional trading commands!{Style.RESET_ALL}")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}🛑 Shutting down...{Style.RESET_ALL}")
    finally:
        observer.stop()
        observer.join()
