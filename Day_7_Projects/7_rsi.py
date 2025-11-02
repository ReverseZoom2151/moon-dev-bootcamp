############ Coding RSI indicator 2024

import sys, os, ccxt, time, pandas as pd 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ta.momentum import RSIIndicator
from Day_4_Projects.key_file import key as xP_KEY, secret as xP_SECRET

# Initialize exchange connection
phemex = ccxt.phemex({
    'enableRateLimit': True, 
    'apiKey': xP_KEY,
    'secret': xP_SECRET
})

# Default values
symbol = 'BTC/USD:BTC'
size = 1 
default_params = {'timeInForce': 'PostOnly'}
timeframe = '15m'
limit = 100
sma = 20 
target = 9 
max_loss = -8

# Symbol to index mapping
SYMBOL_INDEX_MAP = {
    'BTC/USD:BTC': 4,
    'APE/USD:BTC': 2,
    'ETH/USD:BTC': 3,
    'DOGE/USD:BTC': 1,
    'u100000SHIB/USD:BTC': 0
}

def open_positions(symbol=symbol):
    """
    Get information about open positions for a given symbol
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        tuple: (positions_data, is_position_open, position_size, is_long, index_position)
    """
    # Get index position for the symbol
    index_pos = SYMBOL_INDEX_MAP.get(symbol)
    
    # Handle unknown symbols
    if index_pos is None:
        print(f"Warning: Unknown symbol '{symbol}'. Position data may be inaccurate.")
    
    try:
        params = {'type': 'swap', 'code': 'USD'}
        phe_bal = phemex.fetch_balance(params=params)
        open_positions = phe_bal['info']['data']['positions']
        
        # If index_pos is None or out of range, return empty values
        if index_pos is None or index_pos >= len(open_positions):
            print(f"No position data available for symbol: {symbol}")
            return open_positions, False, 0, None, index_pos
        
        openpos_side = open_positions[index_pos]['side']
        openpos_size = open_positions[index_pos]['size']
        
        if openpos_side == 'Buy':
            openpos_bool = True 
            long = True 
        elif openpos_side == 'Sell':
            openpos_bool = True
            long = False
        else:
            openpos_bool = False
            long = None 
            
        print(f'Position status: | Open: {openpos_bool} | Size: {openpos_size} | Long: {long} | Index: {index_pos}')
        
        return open_positions, openpos_bool, openpos_size, long, index_pos
    except Exception as e:
        print(f"Error fetching position data: {e}")
        return [], False, 0, None, index_pos
   
def ask_bid(symbol=symbol):
    """
    Get current ask and bid prices for a symbol
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        tuple: (ask_price, bid_price)
    """
    try:
        ob = phemex.fetch_order_book(symbol)
        
        if not ob['bids'] or not ob['asks']:
            print(f"Warning: Empty order book for {symbol}")
            return 0, 0
            
        bid = ob['bids'][0][0]
        ask = ob['asks'][0][0]
        
        print(f"Current prices for {symbol} - Ask: {ask}, Bid: {bid}")
        
        return ask, bid
    except Exception as e:
        print(f"Error fetching order book: {e}")
        return 0, 0

def kill_switch(symbol=symbol):
    """
    Close all open positions for a symbol
    
    Args:
        symbol (str): Trading symbol
    """
    print(f'Starting the kill switch for {symbol}')
    
    try:
        position_data = open_positions(symbol)
        openposi = position_data[1]  # position open boolean
        long = position_data[3]      # long/short boolean
        kill_size = position_data[2] # position size
        
        print(f'Position status: Open: {openposi}, Long: {long}, Size: {kill_size}')
        
        attempts = 0
        max_attempts = 5
        
        while openposi and attempts < max_attempts:
            attempts += 1
            print(f'Kill switch attempt {attempts}/{max_attempts}')
            
            try:
                # Cancel existing orders
                phemex.cancel_all_orders(symbol)
                
                # Get updated position data
                position_data = open_positions(symbol)
                openposi = position_data[1]
                long = position_data[3]
                kill_size = position_data[2]
                
                if not kill_size or float(kill_size) == 0:
                    print("Position already closed")
                    break
                    
                kill_size = int(kill_size)
                
                # Get market prices
                ask, bid = ask_bid(symbol)
                
                if ask == 0 or bid == 0:
                    print("Could not retrieve valid prices, retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Place closing order based on position direction
                if long == False:
                    phemex.create_limit_buy_order(symbol, kill_size, bid, default_params)
                    print(f'Created BUY to CLOSE order: {kill_size} {symbol} at ${bid}')
                elif long == True:
                    phemex.create_limit_sell_order(symbol, kill_size, ask, default_params)
                    print(f'Created SELL to CLOSE order: {kill_size} {symbol} at ${ask}')
                else:
                    print('Error: Unknown position direction')
                    break
                
                print('Waiting 30 seconds for order to fill...')
                time.sleep(30)
                
                # Check if position still open
                openposi = open_positions(symbol)[1]
                
            except Exception as e:
                print(f"Error during kill switch execution: {e}")
                time.sleep(5)
        
        if openposi and attempts >= max_attempts:
            print(f"WARNING: Failed to close position after {max_attempts} attempts!")
        
    except Exception as e:
        print(f"Error in kill switch: {e}")

def pnl_close(symbol=symbol, target=target, max_loss=max_loss):
    """
    Check PNL and close position if target or stop loss is hit
    
    Args:
        symbol (str): Trading symbol
        target (float): Target profit percentage 
        max_loss (float): Maximum loss percentage (negative number)
        
    Returns:
        tuple: (pnl_close_triggered, in_position, position_size, is_long)
    """
    print(f'Checking PNL exit conditions for {symbol}')
    
    try:
        # Get position data
        params = {'type': 'swap', 'code': 'USD'}
        pos_dict = phemex.fetch_positions(params=params)
        
        index_pos = open_positions(symbol)[4]
        
        if index_pos is None or index_pos >= len(pos_dict):
            print(f"No position data available for {symbol}")
            return False, False, 0, None
            
        position = pos_dict[index_pos]
        side = position['side']
        size = position['contracts']
        
        try:
            entry_price = float(position['entryPrice'])
            leverage = float(position['leverage'])
        except (ValueError, KeyError):
            print("Error parsing position data")
            return False, False, 0, None
            
        current_price = ask_bid(symbol)[1]
        
        print(f'Position: {side} | Entry: {entry_price} | Leverage: {leverage}')
        
        # Determine if position is long or short
        if side == 'long':
            diff = current_price - entry_price
            long = True
        else: 
            diff = entry_price - current_price
            long = False

        # Calculate PNL percentage
        try: 
            perc = round(((diff/entry_price) * leverage), 10) * 100
        except ZeroDivisionError:
            perc = 0
            
        print(f'PNL for {symbol}: {perc}%')

        pnlclose = False 
        in_pos = True
        
        # Check if we should close based on profit target or stop loss
        if perc > 0:
            print(f'Position for {symbol} is in profit')
            if perc > target:
                print(f'Profit target of {target}% reached at {perc}%, closing position')
                pnlclose = True
                kill_switch(symbol)
            else:
                print(f'Current profit {perc}% has not reached target {target}%')
        elif perc < 0: 
            print(f'Position for {symbol} is in loss: {perc}%')
            if perc <= max_loss: 
                print(f'Stop loss of {max_loss}% triggered at {perc}%, closing position')
                kill_switch(symbol)
            else:
                print(f'Current loss {perc}% has not reached max loss {max_loss}%')
        else:
            print('No open position or zero PNL')
            in_pos = False

        print(f'PNL check completed for {symbol}')
        return pnlclose, in_pos, size, long
        
    except Exception as e:
        print(f"Error in PNL close function: {e}")
        return False, False, 0, None

def size_kill(max_risk=1000):
    """
    Check if position size exceeds maximum risk and close if needed
    
    Args:
        max_risk (float): Maximum position cost allowed
    """
    print(f'Checking position sizes against max risk of {max_risk}')
    
    try:
        params = {'type': 'swap', 'code': 'USD'}
        all_phe_balance = phemex.fetch_balance(params=params)
        open_positions = all_phe_balance['info']['data']['positions']
        
        if not open_positions:
            print("No open positions found")
            return
        
        # Check all positions, not just the first one
        for index, position in enumerate(open_positions):
            try:
                pos_cost = float(position.get('posCost', 0))
                openpos_side = position.get('side', 'None')
                openpos_size = position.get('size', 0)
                symbol_info = position.get('symbol', 'Unknown')
                
                print(f'Position {index}: {symbol_info} | Cost: {pos_cost} | Side: {openpos_side} | Size: {openpos_size}')
                
                if pos_cost > max_risk:
                    print(f'EMERGENCY: Position size {pos_cost} exceeds max risk {max_risk}')
                    kill_switch(symbol_info)
                    print('Emergency exit completed')
            except Exception as e:
                print(f"Error processing position {index}: {e}")
                
    except Exception as e:
        print(f"Error in size kill function: {e}")

def df_sma(symbol=symbol, timeframe=timeframe, limit=limit, sma=sma):
    """
    Calculate SMA indicator and determine buy/sell signals
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Chart timeframe
        limit (int): Number of candles to fetch
        sma (int): SMA period
        
    Returns:
        DataFrame: Price data with SMA indicator
    """
    print(f'Calculating SMA({sma}) indicator for {symbol} on {timeframe} timeframe')
    
    try:
        bars = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        if not bars:
            print("No price data received")
            return pd.DataFrame()
            
        df_sma = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_sma['timestamp'] = pd.to_datetime(df_sma['timestamp'], unit='ms')

        # Calculate SMA
        df_sma[f'sma{sma}_{timeframe}'] = df_sma.close.rolling(sma).mean()

        # Get current bid price
        bid = ask_bid(symbol)[1]
        
        # Generate signals
        df_sma.loc[df_sma[f'sma{sma}_{timeframe}']>bid, 'signal'] = 'SELL'
        df_sma.loc[df_sma[f'sma{sma}_{timeframe}']<bid, 'signal'] = 'BUY'

        # Calculate support and resistance
        df_sma['support'] = df_sma['close'].rolling(limit).min()
        df_sma['resistance'] = df_sma['close'].rolling(limit).max()

        print(f'SMA calculation completed with {len(df_sma)} candles')
        return df_sma
        
    except Exception as e:
        print(f"Error in SMA calculation: {e}")
        return pd.DataFrame()

def df_rsi(symbol=symbol, timeframe=timeframe, limit=limit, rsi_period=14):
    """
    Calculate RSI indicator
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Chart timeframe
        limit (int): Number of candles to fetch
        rsi_period (int): RSI period
        
    Returns:
        DataFrame: Price data with RSI indicator
    """
    print(f'Calculating RSI({rsi_period}) for {symbol} on {timeframe} timeframe')
    
    try:
        bars = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        if not bars:
            print("No price data received")
            return pd.DataFrame()
            
        df_rsi = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_rsi['timestamp'] = pd.to_datetime(df_rsi['timestamp'], unit='ms')

        # Calculate RSI
        rsi = RSIIndicator(df_rsi['close'], window=rsi_period)
        df_rsi['rsi'] = rsi.rsi()
        
        # Generate signals based on RSI
        df_rsi.loc[df_rsi['rsi'] > 70, 'signal'] = 'SELL'
        df_rsi.loc[df_rsi['rsi'] < 30, 'signal'] = 'BUY'
        
        print(f'RSI calculation completed with {len(df_rsi)} candles')
        return df_rsi
        
    except Exception as e:
        print(f"Error in RSI calculation: {e}")
        return pd.DataFrame()

# Main execution block
if __name__ == "__main__":
    print("Starting RSI trading bot...")
    
    # Check for open positions
    positions, is_position_open, position_size, is_long, index_position = open_positions(symbol)
    print(f"Current position status: {'Open' if is_position_open else 'Closed'}")
    
    # Get RSI data
    rsi_data = df_rsi(symbol)
    
    if not rsi_data.empty:
        # Get the latest RSI value
        latest_rsi = rsi_data['rsi'].iloc[-1]
        signal = rsi_data['signal'].iloc[-1] if 'signal' in rsi_data.iloc[-1] else "NEUTRAL"
        
        print(f"Latest RSI for {symbol}: {latest_rsi:.2f}")
        print(f"Signal: {signal}")
        
        # Implement your trading logic here based on RSI values and signals
        # For example:
        # if not is_position_open and signal == 'BUY':
        #     # Place buy order logic
        # elif is_position_open:
        #     # Check if we should close based on PNL
        #     pnl_close(symbol)
    
    # Safety check
    size_kill()
    
    print("RSI trading bot execution completed")