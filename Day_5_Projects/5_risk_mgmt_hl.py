import sys, os, eth_account, nice_funcs as n
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Day_4_Projects import dontshare as d

# Configuration constants
SYMBOL = 'WIF'
MAX_LOSS = -5
TARGET = 4
ACCT_MIN = 7
TIMEFRAME = '4h'
SIZE = 10

# Initialize account from private key
secret_key = d.private_key
account = eth_account.Account.from_key(secret_key)

def bot():
    """Main bot function to monitor position and manage risk"""
    print('Starting risk management bot...')
    print('Controlling risk with PNL monitoring')
    
    # Check PNL against targets
    n.pnl_close(SYMBOL, TARGET, MAX_LOSS, account)
    
    # Check account value against minimum threshold
    acct_val = float(n.acct_bal(account))
    if acct_val < ACCT_MIN:
        print(f'Account value ${acct_val} is below minimum threshold ${ACCT_MIN}')
        print('Closing all positions')
        n.kill_switch(SYMBOL, account)

if __name__ == "__main__":
    bot()

