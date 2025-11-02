"""Utilities for interacting with the Bitfinex Professional Trading API."""

import requests
import json
import time
import logging
import hmac
import hashlib
import base64
import dontshare as ds
from typing import Dict, Optional, Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Bitfinex API endpoints
BITFINEX_BASE_URL = "https://api.bitfinex.com"
REST_BASE_URL = f"{BITFINEX_BASE_URL}/v2"
AUTH_BASE_URL = f"{BITFINEX_BASE_URL}/v1"

# Load Bitfinex API credentials from secrets
try:
    BITFINEX_API_KEY = ds.bitfinex_api_key
    BITFINEX_SECRET_KEY = ds.bitfinex_secret_key
except AttributeError as e:
    logging.error(f"Bitfinex credentials not found in dontshare.py: {e}")
    exit(1)

def create_auth_headers(path: str, nonce: str, body: str = "") -> Dict[str, str]:
    """Create authenticated headers for Bitfinex API v1."""
    message = f"/api/{path}{nonce}{body}"
    signature = hmac.new(
        BITFINEX_SECRET_KEY.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha384
    ).hexdigest()
    
    return {
        'X-BFX-APIKEY': BITFINEX_API_KEY,
        'X-BFX-PAYLOAD': base64.b64encode(body.encode('utf-8')).decode('utf-8'),
        'X-BFX-SIGNATURE': signature,
        'Content-Type': 'application/json'
    }

def create_auth_headers_v2(path: str, nonce: str, body: str = "") -> Dict[str, str]:
    """Create authenticated headers for Bitfinex API v2."""
    message = f"/api/v2{path}{nonce}{body}"
    signature = hmac.new(
        BITFINEX_SECRET_KEY.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha384
    ).hexdigest()
    
    return {
        'bfx-nonce': nonce,
        'bfx-apikey': BITFINEX_API_KEY,
        'bfx-signature': signature,
        'Content-Type': 'application/json'
    }

def get_nonce() -> str:
    """Generate nonce for API authentication."""
    return str(int(time.time() * 1000000))

def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to Bitfinex format (e.g., BTCUSD -> btcusd)."""
    # Bitfinex uses lowercase symbols
    symbol = symbol.lower()
    # Handle common conversions
    if symbol.endswith('usdt'):
        symbol = symbol.replace('usdt', 'usd')
    return symbol

def get_symbol_details(symbol: str) -> Optional[Dict]:
    """Get trading rules and information for a symbol."""
    try:
        symbol = normalize_symbol(symbol)
        response = requests.get(f"{REST_BASE_URL}/conf/pub:info:pair")
        response.raise_for_status()
        
        # Bitfinex returns array format
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            symbols_info = data[0]  # Get the actual data array
            
            # Look for our symbol in the trading pairs
            for pair_info in symbols_info:
                if isinstance(pair_info, list) and len(pair_info) > 0:
                    pair_symbol = pair_info[0]
                    if pair_symbol == symbol.upper():
                        return {
                            'symbol': pair_symbol,
                            'min_order_size': pair_info[3] if len(pair_info) > 3 else 0.001,
                            'max_order_size': pair_info[4] if len(pair_info) > 4 else 1000000,
                            'price_precision': pair_info[5] if len(pair_info) > 5 else 5
                        }
        return None
    except Exception as e:
        logging.error(f"Failed to get symbol details for {symbol}: {e}")
        return None

def get_ticker(symbol: str) -> Optional[Dict]:
    """Get current ticker information for a symbol."""
    try:
        symbol = normalize_symbol(symbol)
        response = requests.get(f"{REST_BASE_URL}/ticker/t{symbol.upper()}")
        response.raise_for_status()
        
        ticker_data = response.json()
        if isinstance(ticker_data, list) and len(ticker_data) >= 7:
            return {
                'symbol': symbol,
                'bid': float(ticker_data[0]),
                'bid_size': float(ticker_data[1]),
                'ask': float(ticker_data[2]),
                'ask_size': float(ticker_data[3]),
                'daily_change': float(ticker_data[4]),
                'daily_change_perc': float(ticker_data[5]),
                'last_price': float(ticker_data[6]),
                'volume': float(ticker_data[7]) if len(ticker_data) > 7 else 0,
                'high': float(ticker_data[8]) if len(ticker_data) > 8 else 0,
                'low': float(ticker_data[9]) if len(ticker_data) > 9 else 0
            }
        return None
    except Exception as e:
        logging.error(f"Failed to get ticker for {symbol}: {e}")
        return None

def get_account_balance() -> List[Dict]:
    """Get account balances for all currencies."""
    try:
        nonce = get_nonce()
        path = "v1/balances"
        body = json.dumps({
            "request": f"/v1/balances",
            "nonce": nonce
        })
        
        headers = create_auth_headers(path, nonce, body)
        response = requests.post(f"{AUTH_BASE_URL}/balances", 
                               headers=headers, data=body)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get account balances: {e}")
        return []

def get_available_balance(currency: str = "USD") -> float:
    """Get available balance for specific currency."""
    try:
        balances = get_account_balance()
        currency = currency.lower()
        
        for balance in balances:
            if (balance.get('currency', '').lower() == currency and 
                balance.get('type') == 'exchange'):
                return float(balance.get('available', 0))
        return 0.0
    except Exception as e:
        logging.error(f"Failed to get available balance for {currency}: {e}")
        return 0.0

def calculate_order_amount(symbol: str, usd_amount: float) -> Tuple[float, float]:
    """Calculate order amount and price for market buy."""
    try:
        ticker = get_ticker(symbol)
        if not ticker:
            return 0.0, 0.0
        
        # Use ask price for buying (slightly higher for market execution)
        price = ticker['ask']
        amount = usd_amount / price
        
        return amount, price
    except Exception as e:
        logging.error(f"Failed to calculate order amount: {e}")
        return 0.0, 0.0

def market_buy(symbol: str, usd_amount: float, order_type: str = "exchange market") -> Optional[Dict]:
    """Execute a market buy order on Bitfinex.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSD')
        usd_amount: Amount of USD to spend  
        order_type: Order type - 'exchange market' for spot, 'market' for margin
    
    Returns:
        Order response dict if successful, None otherwise
    """
    try:
        original_symbol = symbol
        symbol = normalize_symbol(symbol)
        logging.info(f"🎯 Executing Bitfinex market buy: {usd_amount} USD worth of {original_symbol}")
        
        # Get current market data
        ticker = get_ticker(symbol)
        if not ticker:
            logging.error(f"Could not get ticker data for {symbol}")
            return None
        
        # Check available balance
        available_balance = get_available_balance("USD")
        if available_balance < usd_amount:
            logging.error(f"Insufficient USD balance: {available_balance} < {usd_amount}")
            return None
        
        # Calculate order parameters
        amount, estimated_price = calculate_order_amount(symbol, usd_amount)
        if amount <= 0:
            logging.error("Could not calculate valid order amount")
            return None
        
        # Format amount to appropriate precision (typically 8 decimal places)
        formatted_amount = f"{amount:.8f}".rstrip('0').rstrip('.')
        
        logging.info(f"📊 Current ask price: {ticker['ask']} USD")
        logging.info(f"📊 Order amount: {formatted_amount} {symbol.upper()}")
        logging.info(f"📊 Estimated cost: {amount * estimated_price:.2f} USD")
        
        # Prepare order payload
        nonce = get_nonce()
        order_payload = {
            "request": "/v1/order/new",
            "nonce": nonce,
            "symbol": symbol,
            "amount": formatted_amount,
            "price": "0",  # Market order - price is ignored
            "side": "buy",
            "type": order_type,
            "exchange": "bitfinex"
        }
        
        body = json.dumps(order_payload)
        headers = create_auth_headers("v1/order/new", nonce, body)
        
        # Execute order
        logging.info("🚀 Sending Bitfinex market buy order...")
        response = requests.post(f"{AUTH_BASE_URL}/order/new", 
                               headers=headers, data=body)
        
        if response.status_code == 200:
            order_result = response.json()
            logging.info(f"✅ Order successful!")
            logging.info(f"📋 Order ID: {order_result.get('id')}")
            logging.info(f"📋 Symbol: {order_result.get('symbol')}")
            logging.info(f"📋 Amount: {order_result.get('original_amount')}")
            logging.info(f"📋 Side: {order_result.get('side')}")
            logging.info(f"📋 Type: {order_result.get('type')}")
            logging.info(f"📋 Status: {order_result.get('is_live')}")
            
            return order_result
        else:
            error_response = response.json()
            logging.error(f"❌ Order failed: {error_response}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Market buy failed: {e}")
        logging.exception("Full traceback:")
        return None

def get_order_status(order_id: int) -> Optional[Dict]:
    """Get status of a specific order."""
    try:
        nonce = get_nonce()
        payload = {
            "request": "/v1/order/status",
            "nonce": nonce,
            "order_id": order_id
        }
        
        body = json.dumps(payload)
        headers = create_auth_headers("v1/order/status", nonce, body)
        
        response = requests.post(f"{AUTH_BASE_URL}/order/status",
                               headers=headers, data=body)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get order status: {e}")
        return None

def get_active_orders() -> List[Dict]:
    """Get all active orders."""
    try:
        nonce = get_nonce()
        payload = {
            "request": "/v1/orders",
            "nonce": nonce
        }
        
        body = json.dumps(payload)
        headers = create_auth_headers("v1/orders", nonce, body)
        
        response = requests.post(f"{AUTH_BASE_URL}/orders",
                               headers=headers, data=body)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logging.error(f"Failed to get active orders: {e}")
        return []

def cancel_order(order_id: int) -> Optional[Dict]:
    """Cancel a specific order."""
    try:
        nonce = get_nonce()
        payload = {
            "request": "/v1/order/cancel",
            "nonce": nonce,
            "order_id": order_id
        }
        
        body = json.dumps(payload)
        headers = create_auth_headers("v1/order/cancel", nonce, body)
        
        response = requests.post(f"{AUTH_BASE_URL}/order/cancel",
                               headers=headers, data=body)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logging.error(f"Failed to cancel order {order_id}: {e}")
        return None

def get_funding_book(symbol: str, limit_bids: int = 50, limit_asks: int = 50) -> Optional[Dict]:
    """Get funding book for margin trading."""
    try:
        symbol = normalize_symbol(symbol)
        currency = symbol.replace('usd', '')  # Extract base currency
        
        response = requests.get(f"{REST_BASE_URL}/book/f{currency.upper()}/P0",
                              params={'len': str(limit_bids + limit_asks)})
        response.raise_for_status()
        
        book_data = response.json()
        return {
            'symbol': currency,
            'funding_book': book_data
        }
    except Exception as e:
        logging.error(f"Failed to get funding book for {symbol}: {e}")
        return None

def validate_symbol_format(symbol: str) -> bool:
    """Validate if symbol follows Bitfinex format."""
    symbol = symbol.lower()
    # Basic validation - should be valid crypto pairs
    valid_quote_currencies = ['usd', 'usdt', 'eur', 'btc', 'eth']
    return any(symbol.endswith(quote) for quote in valid_quote_currencies)

def get_trading_pairs() -> List[str]:
    """Get list of all available trading pairs."""
    try:
        response = requests.get(f"{REST_BASE_URL}/conf/pub:list:pair:exchange")
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]  # Return the list of trading pairs
        return []
    except Exception as e:
        logging.error(f"Failed to get trading pairs: {e}")
        return []

def get_recent_trades(symbol: str, limit: int = 50) -> List[Dict]:
    """Get recent trades for a symbol."""
    try:
        symbol = normalize_symbol(symbol)
        response = requests.get(f"{REST_BASE_URL}/trades/t{symbol.upper()}/hist",
                              params={'limit': limit})
        response.raise_for_status()
        
        trades_data = response.json()
        formatted_trades = []
        
        for trade in trades_data:
            if len(trade) >= 4:
                formatted_trades.append({
                    'id': trade[0],
                    'timestamp': trade[1],
                    'amount': float(trade[2]),
                    'price': float(trade[3])
                })
        
        return formatted_trades
    except Exception as e:
        logging.error(f"Failed to get recent trades for {symbol}: {e}")
        return []

if __name__ == "__main__":
    # Test the utilities
    logging.info("🧪 Testing Bitfinex utilities...")
    
    # Test ticker
    btc_ticker = get_ticker("BTCUSD")
    if btc_ticker:
        logging.info(f"BTC ticker: {btc_ticker}")
    
    # Test symbol details
    btc_details = get_symbol_details("BTCUSD")
    if btc_details:
        logging.info(f"BTC details: {btc_details}")
    
    # Test balance (will show error if no API keys configured)
    try:
        balance = get_available_balance("USD")
        logging.info(f"USD balance: {balance}")
    except:
        logging.info("Balance check requires API credentials")
    
    # Test trading pairs
    pairs = get_trading_pairs()
    logging.info(f"Available pairs: {len(pairs)} pairs")
    
    logging.info("✅ Bitfinex utilities test complete")
