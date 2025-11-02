import eth_account, json, time, requests, dontshare as d
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange

symbol = 'WIF'
timeframe = '4h'

def check_wallet_registered(account_address):
    """Check if the wallet is registered on Hyperliquid"""
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": "userState",
        "user": account_address
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        result = response.json()
        if result and isinstance(result, dict):
            if 'error' in result and 'User does not exist' in result['error']:
                return False
            return True
        return False
    except Exception as e:
        print(f"Error checking wallet registration: {e}")
        return False

def ask_bid(symbol):
    '''this gets the ask and bid for any symbol passed in'''
    # Direct API call since the Info client methods don't work as expected
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {
        "type": "l2Book",  # Correct case: l2Book not l2book
        "coin": symbol
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")
        
        result = response.json()
        
        if 'levels' in result and len(result['levels']) > 0 and len(result['levels'][0]) >= 2:
            bid = float(result['levels'][0][0]['px'])
            ask = float(result['levels'][0][1]['px'])
            return ask, bid, result
        else:
            print(f"No valid order data found for {symbol}")
            return 0.0, 0.0, {}
    except Exception as e:
        print(f"Error in ask_bid API call: {e}")
        return 0.0, 0.0, {}

def get_sz_px_decimals(coin):
    '''this returns the size and price decimal places for a coin'''
    # Direct API call for meta info
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {"type": "meta"}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            data = response.json()
            if 'universe' in data:
                symbol_info = next((s for s in data['universe'] if s['name'] == coin), None)
                if symbol_info and 'szDecimals' in symbol_info:
                    sz_decimals = symbol_info['szDecimals']
                    print(f"Found {coin} in universe with size decimals: {sz_decimals}")
                else:
                    print(f"{coin} not found in API response")
                    sz_decimals = 2  # Default value
            else:
                print("Universe not found in API response")
                sz_decimals = 2  # Default value
        else:
            print(f"Error in meta API call: {response.status_code}")
            sz_decimals = 2  # Default value
    except Exception as e:
        print(f"Error getting size decimals: {e}")
        sz_decimals = 2  # Default value

    try:
        # Get ask price to determine price decimals
        ask, _, _ = ask_bid(coin)
        if ask > 0:
            ask_str = str(ask)
            if '.' in ask_str:
                px_decimals = len(ask_str.split('.')[1])
            else:
                px_decimals = 0
        else:
            px_decimals = 5  # Default price decimals for WIF
    except Exception as e:
        print(f"Error getting ask price: {e}")
        px_decimals = 5  # Default value
    
    print(f'{coin} size decimals: {sz_decimals}, price decimals: {px_decimals}')
    return sz_decimals, px_decimals

def limit_order(coin, is_buy, limit_px, sz, reduce_only, account):
    try:
        exchange = Exchange(account, constants.MAINNET_API_URL)
        sz_decimals, px_decimals = get_sz_px_decimals(coin)
        
        # Round the size and price to the appropriate decimals
        sz = round(sz, sz_decimals)
        
        # Ensure minimum size of 1 for coins with 0 decimals
        if sz_decimals == 0 and sz < 1:
            print(f"Adjusting order size from {sz} to minimum size of 1 for {coin}")
            sz = 1
            
        limit_px = round(limit_px, px_decimals)
        
        print(f'coin: {coin}, type: {type(coin)}')
        print(f'is_buy: {is_buy}, type: {type(is_buy)}')
        print(f'sz: {sz}, type: {type(sz)}')
        print(f'limit_px: {limit_px}, type: {type(limit_px)}')
        print(f'reduce_only: {reduce_only}, type: {type(reduce_only)}')

        print(f'placing limit order for {coin} {sz} @ {limit_px}')
        
        # Place the order
        order_result = exchange.order(
            coin, 
            is_buy, 
            sz, 
            limit_px, 
            {"limit": {"tif": 'Gtc'}}, 
            reduce_only=reduce_only
        )
        
        # Print the full order result for debugging
        print(f"Order result: {json.dumps(order_result, indent=2)}")
        
        # Handle the order result properly
        if isinstance(order_result, dict) and 'response' in order_result:
            if isinstance(order_result['response'], dict) and 'data' in order_result['response']:
                if 'statuses' in order_result['response']['data']:
                    status = order_result['response']['data']['statuses']
                    if is_buy:
                        print(f"limit BUY order placed, status: {status}")
                    else:
                        print(f"limit SELL order placed, status: {status}")
                else:
                    print(f"Order placed but no status available")
            else:
                print(f"Order placed but unexpected response format")
        else:
            print(f"Order placed but unexpected result format")
            
        return order_result
    except Exception as e:
        print(f"Error placing order: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main execution
try:
    coin = symbol
    is_buy = True
    sz = 1  # Default size of 1 for WIF since it has 0 decimals
    
    print(f"Using Hyperliquid API at {constants.MAINNET_API_URL}")
    
    # Get available markets using direct API call
    try:
        url = 'https://api.hyperliquid.xyz/info'
        headers = {'Content-Type': 'application/json'}
        data = {"type": "meta"}
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            meta_data = response.json()
            if 'universe' in meta_data:
                available_coins = [coin['name'] for coin in meta_data['universe']]
                print(f"Available coins: {available_coins}")
                
                if coin not in available_coins:
                    print(f"WARNING: {coin} not found in available markets!")
            else:
                print("Universe data not found in API response")
        else:
            print(f"Failed to get available markets: {response.status_code}")
    except Exception as e:
        print(f"Failed to get available markets: {e}")
    
    # Get current prices
    ask, bid, l2 = ask_bid(coin)
    
    if ask == 0.0 or bid == 0.0:
        print("Failed to get valid prices. Exiting.")
    else:
        print(f"Got prices for {coin}: Ask = {ask}, Bid = {bid}")
        reduce_only = False
        secret_key = d.private_key
        account = eth_account.Account.from_key(secret_key)
        
        # Check if wallet is registered
        wallet_address = account.address
        print(f"Using wallet address: {wallet_address}")
        
        if not check_wallet_registered(wallet_address):
            print("\n⚠️ ERROR: WALLET NOT REGISTERED")
            print(f"The wallet address {wallet_address} is not registered on Hyperliquid.")
            print("You need to register this wallet on Hyperliquid before placing orders.")
            print("1. Go to https://hyperliquid.xyz")
            print("2. Connect your wallet")
            print("3. Complete the registration process\n")
            raise Exception("Wallet not registered on Hyperliquid")
        
        print("✅ Wallet is registered on Hyperliquid")
        
        # Place buy order
        print(f"Placing BUY order for {coin}")
        limit_order(coin, is_buy, bid, sz, reduce_only, account)
        
        time.sleep(5)
        
        # Place sell order
        is_buy = False 
        reduce_only = True
        print(f"Placing SELL order for {coin}")
        limit_order(coin, is_buy, ask, sz, reduce_only, account)
except Exception as e:
    print(f"Error in main execution: {e}")