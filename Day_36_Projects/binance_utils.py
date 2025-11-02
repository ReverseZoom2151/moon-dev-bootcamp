"""Utilities for interacting with the Binance Spot Trading API."""

import requests
import dontshare as ds
import time
import logging
import hmac
import hashlib
from urllib.parse import urlencode
from typing import Dict, Optional, List
from decimal import Decimal, ROUND_DOWN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Binance API endpoints
BINANCE_BASE_URL = "https://api.binance.com"
SPOT_BASE_URL = f"{BINANCE_BASE_URL}/api/v3"

# Load Binance API credentials from secrets
try:
    BINANCE_API_KEY = ds.binance_api_key
    BINANCE_SECRET_KEY = ds.binance_secret_key
except AttributeError as e:
    logging.error(f"Binance credentials not found in dontshare.py: {e}")
    exit(1)

def create_signature(query_string: str) -> str:
    """Create HMAC SHA256 signature for Binance API."""
    return hmac.new(
        BINANCE_SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def get_headers() -> Dict[str, str]:
    """Get standard headers for Binance API requests."""
    return {
        'X-MBX-APIKEY': BINANCE_API_KEY,
        'Content-Type': 'application/json'
    }

def get_server_time() -> int:
    """Get Binance server time for timestamp synchronization."""
    try:
        response = requests.get(f"{SPOT_BASE_URL}/time")
        response.raise_for_status()
        return response.json()['serverTime']
    except Exception as e:
        logging.warning(f"Failed to get server time, using local time: {e}")
        return int(time.time() * 1000)

def get_symbol_info(symbol: str) -> Optional[Dict]:
    """Get trading rules and information for a symbol."""
    try:
        response = requests.get(f"{SPOT_BASE_URL}/exchangeInfo")
        response.raise_for_status()
        data = response.json()
        
        for symbol_info in data['symbols']:
            if symbol_info['symbol'] == symbol.upper():
                return symbol_info
        return None
    except Exception as e:
        logging.error(f"Failed to get symbol info for {symbol}: {e}")
        return None

def format_quantity(quantity: float, symbol_info: Dict) -> str:
    """Format quantity according to symbol's step size."""
    try:
        # Find LOT_SIZE filter
        for filter_info in symbol_info['filters']:
            if filter_info['filterType'] == 'LOT_SIZE':
                step_size = Decimal(filter_info['stepSize'])
                break
        else:
            step_size = Decimal('0.00001')  # Default
        
        # Format to step size
        quantity_decimal = Decimal(str(quantity))
        formatted = quantity_decimal.quantize(step_size, rounding=ROUND_DOWN)
        return str(formatted)
    except Exception as e:
        logging.warning(f"Failed to format quantity: {e}")
        return f"{quantity:.6f}".rstrip('0').rstrip('.')

def get_account_balance(asset: str = "USDT") -> float:
    """Get account balance for specified asset."""
    try:
        timestamp = get_server_time()
        query_string = f"timestamp={timestamp}"
        signature = create_signature(query_string)
        
        url = f"{SPOT_BASE_URL}/account?{query_string}&signature={signature}"
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        
        account_data = response.json()
        for balance in account_data['balances']:
            if balance['asset'] == asset.upper():
                return float(balance['free'])
        return 0.0
    except Exception as e:
        logging.error(f"Failed to get account balance: {e}")
        return 0.0

def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a trading pair."""
    try:
        response = requests.get(f"{SPOT_BASE_URL}/ticker/price", 
                              params={'symbol': symbol.upper()})
        response.raise_for_status()
        return float(response.json()['price'])
    except Exception as e:
        logging.error(f"Failed to get price for {symbol}: {e}")
        return None

def market_buy(symbol: str, usdt_amount: float, max_slippage: float = 0.05) -> Optional[Dict]:
    """Execute a market buy order on Binance.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        usdt_amount: Amount of USDT to spend
        max_slippage: Maximum acceptable slippage (default 5%)
    
    Returns:
        Order response dict if successful, None otherwise
    """
    try:
        symbol = symbol.upper()
        logging.info(f"🎯 Executing market buy: {usdt_amount} USDT worth of {symbol}")
        
        # Get symbol information
        symbol_info = get_symbol_info(symbol)
        if not symbol_info:
            logging.error(f"Symbol {symbol} not found or not tradeable")
            return None
        
        # Check if symbol is active
        if symbol_info['status'] != 'TRADING':
            logging.error(f"Symbol {symbol} is not in TRADING status: {symbol_info['status']}")
            return None
        
        # Get current price for reference
        current_price = get_current_price(symbol)
        if not current_price:
            logging.error(f"Could not get current price for {symbol}")
            return None
        
        # Check account balance
        available_balance = get_account_balance("USDT")
        if available_balance < usdt_amount:
            logging.error(f"Insufficient USDT balance: {available_balance} < {usdt_amount}")
            return None
        
        # Calculate approximate quantity
        estimated_quantity = usdt_amount / current_price
        formatted_quantity = format_quantity(estimated_quantity, symbol_info)
        
        logging.info(f"📊 Current price: {current_price} USDT")
        logging.info(f"📊 Estimated quantity: {formatted_quantity} {symbol}")
        
        # Prepare order parameters
        timestamp = get_server_time()
        order_params = {
            'symbol': symbol,
            'side': 'BUY',
            'type': 'MARKET',
            'quoteOrderQty': str(usdt_amount),  # Use quote order qty for precise USDT amount
            'timestamp': timestamp
        }
        
        # Create signature
        query_string = urlencode(order_params)
        signature = create_signature(query_string)
        order_params['signature'] = signature
        
        # Execute order
        logging.info("🚀 Sending market buy order...")
        response = requests.post(
            f"{SPOT_BASE_URL}/order",
            headers=get_headers(),
            data=order_params
        )
        
        if response.status_code == 200:
            order_result = response.json()
            logging.info(f"✅ Order successful!")
            logging.info(f"📋 Order ID: {order_result['orderId']}")
            logging.info(f"📋 Status: {order_result['status']}")
            logging.info(f"📋 Executed Qty: {order_result.get('executedQty', 'N/A')}")
            logging.info(f"📋 Cumulative Quote Qty: {order_result.get('cummulativeQuoteQty', 'N/A')}")
            
            return order_result
        else:
            error_response = response.json()
            logging.error(f"❌ Order failed: {error_response}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Market buy failed: {e}")
        logging.exception("Full traceback:")
        return None

def get_order_status(symbol: str, order_id: int) -> Optional[Dict]:
    """Get status of a specific order."""
    try:
        timestamp = get_server_time()
        query_params = {
            'symbol': symbol.upper(),
            'orderId': order_id,
            'timestamp': timestamp
        }
        
        query_string = urlencode(query_params)
        signature = create_signature(query_string)
        
        url = f"{SPOT_BASE_URL}/order?{query_string}&signature={signature}"
        response = requests.get(url, headers=get_headers())
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get order status: {e}")
        return None

def get_24hr_stats(symbol: str) -> Optional[Dict]:
    """Get 24hr ticker statistics for a symbol."""
    try:
        response = requests.get(f"{SPOT_BASE_URL}/ticker/24hr",
                              params={'symbol': symbol.upper()})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get 24hr stats for {symbol}: {e}")
        return None

def validate_symbol_format(symbol: str) -> bool:
    """Validate if symbol follows Binance format (e.g., BTCUSDT)."""
    symbol = symbol.upper()
    # Basic validation - should end with USDT, BTC, ETH, or BNB
    base_currencies = ['USDT', 'BUSD', 'BTC', 'ETH', 'BNB']
    return any(symbol.endswith(base) for base in base_currencies)

def search_new_listings() -> List[Dict]:
    """Get information about recently listed tokens."""
    try:
        response = requests.get(f"{SPOT_BASE_URL}/exchangeInfo")
        response.raise_for_status()
        data = response.json()
        
        # Filter for USDT pairs that are actively trading
        usdt_pairs = []
        for symbol_info in data['symbols']:
            if (symbol_info['symbol'].endswith('USDT') and 
                symbol_info['status'] == 'TRADING' and
                'SPOT' in symbol_info['permissions']):
                usdt_pairs.append({
                    'symbol': symbol_info['symbol'],
                    'baseAsset': symbol_info['baseAsset'],
                    'quoteAsset': symbol_info['quoteAsset'],
                    'status': symbol_info['status']
                })
        
        return usdt_pairs
    except Exception as e:
        logging.error(f"Failed to get new listings: {e}")
        return []

if __name__ == "__main__":
    # Test the utilities
    logging.info("🧪 Testing Binance utilities...")
    
    # Test server time
    server_time = get_server_time()
    logging.info(f"Server time: {server_time}")
    
    # Test symbol info
    btc_info = get_symbol_info("BTCUSDT")
    if btc_info:
        logging.info(f"BTC symbol status: {btc_info['status']}")
    
    # Test price
    btc_price = get_current_price("BTCUSDT")
    if btc_price:
        logging.info(f"BTC price: {btc_price}")
    
    # Test balance (will show 0 if no API keys)
    balance = get_account_balance("USDT")
    logging.info(f"USDT balance: {balance}")
    
    logging.info("✅ Binance utilities test complete")
