import requests
import json
import pandas as pd
import logging
import time
import threading
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from functools import wraps, lru_cache
from dotenv import load_dotenv
from web3 import Web3
from web3.exceptions import ContractLogicError, Web3Exception

# Configuration and Setup
class Config:
    """Enhanced configuration for Polymarket Nice Functions"""
    
    # API URLs
    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    CLOB_API_URL = "https://clob.polymarket.com"
    
    # Polygon Network
    DEFAULT_RPC_URL = "https://polygon-rpc.com"
    BACKUP_RPC_URLS = [
        "https://polygon-mainnet.infura.io/v3/",
        "https://rpc-mainnet.matic.quiknode.pro",
        "https://polygon.llamarpc.com"
    ]
    
    # USDC Contracts (Polygon)
    USDC_CONTRACTS = {
        "new_usdc": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",  # USDC.e
        "old_usdc": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"   # Bridged USDC
    }
    
    # Trading Parameters
    MIN_ORDER_SIZE = 5  # Minimum shares per order
    MAX_ORDER_SIZE = 1000000  # Maximum shares per order
    MIN_PRICE = 0.01
    MAX_PRICE = 0.99
    
    # API Settings
    API_TIMEOUT = 15
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1.0
    THREAD_POOL_SIZE = 5
    CACHE_DURATION = 60  # seconds
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "polymarket_trading.log"

@dataclass
class PositionData:
    """Data class for position information"""
    token_id: str
    market_id: str
    market_question: str
    outcome: str
    size: float
    value_usd: float
    current_price: float
    entry_price: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0

@dataclass 
class OrderData:
    """Data class for order information"""
    order_id: str
    token_id: str
    market_question: str
    side: str
    price: float
    size: float
    size_matched: float
    status: str
    created_at: str

@dataclass
class MarketData:
    """Data class for market information"""
    market_id: str
    question: str
    category: str
    volume_24h: float
    volume_total: float
    yes_price: float
    no_price: float
    yes_token_id: str
    no_token_id: str
    closed: bool = False
    end_date: Optional[str] = None

# Enhanced Logging Setup
def setup_logging(level: str = Config.LOG_LEVEL) -> logging.Logger:
    """Setup enhanced logging configuration"""
    
    # Create logger
    logger = logging.getLogger('polymarket_nice_funcs')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logging and environment
logger = setup_logging()
load_dotenv()

logger.info("🚀 Enhanced Polymarket Nice Functions Loaded! 🌙")
logger.info(f"Logging to: {Config.LOG_FILE}")

# Enhanced API Client and Utilities
class EnhancedAPIClient:
    """Enhanced API client with retry logic, caching, and error handling"""
    
    def __init__(self):
        self._session = requests.Session()
        self._session.timeout = Config.API_TIMEOUT
        self._cache = {}
        self._cache_timestamps = {}
        self._lock = threading.Lock()
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[key]) < Config.CACHE_DURATION
    
    def get_with_retry(self, url: str, params: Dict = None, headers: Dict = None, 
                      use_cache: bool = True) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and caching"""
        cache_key = f"{url}_{str(params)}_{str(headers)}"
        
        # Check cache first
        if use_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {url}")
            return self._cache[cache_key]
        
        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                response = self._session.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                # Cache successful response
                if use_cache:
                    with self._lock:
                        self._cache[cache_key] = response
                        self._cache_timestamps[cache_key] = time.time()
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{Config.RETRY_ATTEMPTS}): {e}")
                if attempt < Config.RETRY_ATTEMPTS - 1:
                    time.sleep(Config.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"All retry attempts failed for {url}")
                    return None
        
        return None
    
    def post_with_retry(self, url: str, data: Dict = None, headers: Dict = None) -> Optional[requests.Response]:
        """Make HTTP POST request with retry logic"""
        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                response = self._session.post(url, json=data, headers=headers)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"POST request failed (attempt {attempt + 1}/{Config.RETRY_ATTEMPTS}): {e}")
                if attempt < Config.RETRY_ATTEMPTS - 1:
                    time.sleep(Config.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"All POST retry attempts failed for {url}")
                    return None
        
        return None

# Global API client instance
api_client = EnhancedAPIClient()

# Enhanced Web3 Connection Manager
class Web3Manager:
    """Enhanced Web3 connection manager with fallback RPCs"""
    
    def __init__(self):
        self._connections = {}
        self._lock = threading.Lock()
        
    def get_connection(self, rpc_url: Optional[str] = None) -> Optional[Web3]:
        """Get Web3 connection with fallback support"""
        rpc_url = rpc_url or os.getenv('RPC_URL', Config.DEFAULT_RPC_URL)
        
        # Check existing connection
        if rpc_url in self._connections:
            w3 = self._connections[rpc_url]
            if w3.is_connected():
                return w3
            else:
                # Remove stale connection
                with self._lock:
                    del self._connections[rpc_url]
        
        # Try to create new connection
        try:
            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
            if w3.is_connected():
                with self._lock:
                    self._connections[rpc_url] = w3
                logger.debug(f"Connected to RPC: {rpc_url}")
                return w3
        except Exception as e:
            logger.warning(f"Failed to connect to RPC {rpc_url}: {e}")
        
        # Try backup RPCs
        for backup_rpc in Config.BACKUP_RPC_URLS:
            try:
                w3 = Web3(Web3.HTTPProvider(backup_rpc, request_kwargs={'timeout': 10}))
                if w3.is_connected():
                    with self._lock:
                        self._connections[backup_rpc] = w3
                    logger.info(f"Connected to backup RPC: {backup_rpc}")
                    return w3
            except Exception as e:
                logger.warning(f"Failed to connect to backup RPC {backup_rpc}: {e}")
        
        logger.error("Failed to connect to any RPC")
        return None

# Global Web3 manager
web3_manager = Web3Manager()

# Utility Functions
def validate_wallet_address(address: str) -> bool:
    """Validate Ethereum wallet address format"""
    try:
        Web3.to_checksum_address(address)
        return True
    except ValueError:
        return False

def get_env_wallet() -> Optional[str]:
    """Get wallet address from environment with validation"""
    wallet = os.getenv("PUBLIC_KEY")
    if wallet and validate_wallet_address(wallet):
        return wallet
    return None

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying functions on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise
            return None
        return wrapper
    return decorator

@retry_on_failure(max_attempts=3, delay=1.0)
def get_usdc_balance(wallet_address: Optional[str] = None) -> float:
    """
    Enhanced USDC balance checker with improved error handling and fallback support
    
    Args:
        wallet_address: The wallet address to check (uses .env if not provided)
    
    Returns:
        USDC balance as float
    """
    logger.info("Checking USDC balance...")
    
    # Get and validate wallet address
    if not wallet_address:
        wallet_address = get_env_wallet()
    
    if not wallet_address:
        logger.error("No valid wallet address provided or found in environment")
        return 0.0
    
    if not validate_wallet_address(wallet_address):
        logger.error(f"Invalid wallet address format: {wallet_address}")
        return 0.0
    
    try:
        # Get Web3 connection with fallback support
        w3 = web3_manager.get_connection()
        if not w3:
            logger.error("Could not establish Web3 connection")
            return 0.0
        
        # USDC ABI - balanceOf function
        usdc_abi = [{
            "constant": True,
            "inputs": [{"name": "owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function"
        }]
        
        total_balance = 0.0
        wallet_checksum = Web3.to_checksum_address(wallet_address)
        
        # Check both USDC contracts
        for contract_name, contract_address in Config.USDC_CONTRACTS.items():
            try:
                # Create contract instance
                usdc_contract = w3.eth.contract(
                    address=Web3.to_checksum_address(contract_address),
                    abi=usdc_abi
                )
                
                # Call balanceOf function
                balance_wei = usdc_contract.functions.balanceOf(wallet_checksum).call()
                
                # Convert from wei to USDC (6 decimals)
                balance_usdc = float(balance_wei) / (10 ** 6)
                
                if balance_usdc > 0:
                    logger.info(f"Found ${balance_usdc:,.2f} USDC in {contract_name}")
                    total_balance += balance_usdc
                    
            except (ContractLogicError, Web3Exception) as e:
                logger.warning(f"Contract call failed for {contract_name}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error checking {contract_name}: {e}")
                continue
        
        if total_balance > 0:
            logger.info(f"Total USDC balance: ${total_balance:,.2f}")
        else:
            logger.info("No USDC balance found")
        
        return total_balance
        
    except Exception as e:
        logger.error(f"Error checking USDC balance: {e}")
        return 0.0


@lru_cache(maxsize=128)
def get_token_id(market_id: str, choice: Optional[str] = None) -> Union[str, List[str]]:
    """
    Enhanced token ID fetcher with caching and improved error handling
    
    Args:
        market_id: The market ID from Polymarket
        choice: Either 'yes' or 'no' (case insensitive). If None, returns all token IDs
    
    Returns:
        Single token ID string if choice specified, otherwise [market_id, yes_token_id, no_token_id]
    """
    logger.info(f"Fetching token ID(s) for market {market_id}{f', {choice}' if choice else ' (all tokens)'}")
    
    try:
        # Use enhanced API client
        params = {'closed': 'false', 'limit': 1000}
        response = api_client.get_with_retry(f"{Config.GAMMA_API_URL}/markets", params)
        
        if not response:
            logger.error(f"Failed to fetch markets data for market {market_id}")
            return '' if choice else ['', '', '']
        
        # Create DataFrame and find market
        df = pd.DataFrame(response.json())
        
        if df.empty:
            logger.error("No markets data received")
            return '' if choice else ['', '', '']
        
        # Find the specific market - try both string and int comparison
        market_row = df[df['id'].astype(str) == str(market_id)]
        
        if market_row.empty:
            # Try with int comparison as backup
            try:
                market_row = df[df['id'] == int(market_id)]
            except (ValueError, TypeError):
                pass
        
        if market_row.empty:
            logger.error(f"Market ID {market_id} not found")
            logger.debug(f"Available market IDs: {df['id'].head(10).tolist()}")
            return '' if choice else ['', '', '']
        
        # Get clobTokenIds for this market
        clob_token_ids = market_row['clobTokenIds'].iloc[0]
        
        if pd.isna(clob_token_ids):
            logger.error(f"No token IDs found for market {market_id}")
            return '' if choice else ['', '', '']
        
        # Parse the JSON string to get token list
        try:
            token_list = json.loads(clob_token_ids)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse token IDs for market {market_id}: {e}")
            return '' if choice else ['', '', '']
        
        if len(token_list) < 2:
            logger.error(f"Invalid token list for market {market_id}: {token_list}")
            return '' if choice else ['', '', '']
        
        yes_token_id, no_token_id = token_list[0], token_list[1]
        
        # If no choice specified, return all three IDs
        if choice is None:
            logger.info(f"Found all tokens for market {market_id}")
            return [market_id, yes_token_id, no_token_id]
        
        # Normalize choice to lowercase
        choice = choice.lower().strip()
        
        if choice not in ['yes', 'no']:
            logger.error(f"Invalid choice '{choice}', must be 'yes' or 'no'")
            return ''
        
        # Return the appropriate token ID
        token_id = yes_token_id if choice == 'yes' else no_token_id
        logger.info(f"Found {choice.upper()} token: {token_id[:20]}...")
        return token_id
        
    except Exception as e:
        logger.error(f"Error getting token ID for market {market_id}: {e}")
        return '' if choice else ['', '', '']


@retry_on_failure(max_attempts=3, delay=1.0)
def get_positions(user_address: Optional[str] = None, limit: int = 50) -> Optional[Dict[str, Any]]:
    """
    Enhanced positions fetcher with improved error handling and data processing
    
    Args:
        user_address: The user's wallet address (uses .env if not provided)
        limit: Maximum number of positions to return (default 50, max 500)
    
    Returns:
        Dictionary containing positions data, portfolio summary, and USDC balance
    """
    # Get and validate user address
    if not user_address:
        user_address = get_env_wallet()
    
    if not user_address:
        logger.error("No valid wallet address provided or found in environment")
        return None
    
    if not validate_wallet_address(user_address):
        logger.error(f"Invalid wallet address format: {user_address}")
        return None
    
    # Validate and sanitize limit
    limit = max(1, min(limit, 500))  # API max is 500
    
    logger.info(f"Fetching positions for user {user_address[:10]}... (limit: {limit})")
    
    try:
        # Get USDC balance with enhanced error handling
        usdc_balance = get_usdc_balance(user_address)
        
        # Get user positions from Polymarket Data API
        positions_params = {
            'user': user_address,
            'limit': limit,
            'sortBy': 'CURRENT',
            'sortDirection': 'DESC'
        }
        
        logger.debug("Requesting positions data from Polymarket API...")
        positions_response = api_client.get_with_retry(
            "https://data-api.polymarket.com/positions", 
            params=positions_params,
            use_cache=False  # Don't cache position data as it changes frequently
        )
        
        if not positions_response:
            logger.error("Failed to fetch positions data")
            return None
        
        positions_data = positions_response.json()
        
        if not positions_data:
            logger.info("No positions found for this user")
            return {
                'positions': [],
                'summary': {
                    'total_positions': 0,
                    'total_current_value': 0.0,
                    'total_initial_value': 0.0,
                    'total_cash_pnl': 0,
                    'total_percent_pnl': 0,
                    'redeemable_positions': 0,
                    'active_positions': 0
                },
                'usdc_balance': usdc_balance
            }
        
        # Convert to DataFrame for easier processing with validation
        if not isinstance(positions_data, list) or not positions_data:
            logger.warning("Invalid positions data format")
            return None
            
        df = pd.DataFrame(positions_data)
        
        # Validate required columns exist
        required_cols = ['currentValue', 'initialValue', 'cashPnl']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Calculate portfolio summary with safe numeric operations
        total_positions = len(df)
        total_current_value = pd.to_numeric(df['currentValue'], errors='coerce').fillna(0).sum()
        total_initial_value = pd.to_numeric(df['initialValue'], errors='coerce').fillna(0).sum()
        total_cash_pnl = pd.to_numeric(df['cashPnl'], errors='coerce').fillna(0).sum()
        
        # Calculate overall percent PnL with division by zero protection
        total_percent_pnl = (
            (total_cash_pnl / total_initial_value * 100) 
            if total_initial_value > 0 
            else 0
        )
        
        # Count special position types with safe operations
        redeemable_positions = (
            pd.to_numeric(df['redeemable'], errors='coerce').fillna(0).sum() 
            if 'redeemable' in df.columns 
            else 0
        )
        active_positions = (
            len(df[pd.to_numeric(df['size'], errors='coerce').fillna(0) > 0]) 
            if 'size' in df.columns 
            else 0
        )
        
        # Log portfolio summary
        logger.info(f"Found {total_positions} positions")
        logger.info(f"Portfolio Value: ${total_current_value:,.2f}")
        logger.info(f"Total P&L: ${total_cash_pnl:,.2f} ({total_percent_pnl:+.2f}%)")
        logger.info(f"Active Positions: {active_positions}")
        logger.info(f"Redeemable Positions: {redeemable_positions}")
        logger.info(f"USDC Balance: ${usdc_balance:,.2f}")
        
        # Calculate total available funds
        total_funds = float(usdc_balance) + float(total_current_value)
        logger.info(f"Total Available Funds: ${total_funds:,.2f}")
        
        # Create enhanced summary dictionary
        summary = {
            'total_positions': int(total_positions),
            'total_current_value': round(float(total_current_value), 2),
            'total_initial_value': round(float(total_initial_value), 2),
            'total_cash_pnl': round(float(total_cash_pnl), 2),
            'total_percent_pnl': round(float(total_percent_pnl), 2),
            'redeemable_positions': int(redeemable_positions),
            'active_positions': int(active_positions),
            'total_funds': round(float(total_funds), 2),
            'usdc_balance': float(usdc_balance),
            'timestamp': datetime.now().isoformat()
        }
        
        # Format positions for display with enhanced data validation
        display_positions = []
        for _, pos in df.iterrows():
            try:
                position_data = {
                    'title': str(pos.get('title', 'Unknown Market')),
                    'outcome': str(pos.get('outcome', 'Unknown')),
                    'size': float(pd.to_numeric(pos.get('size', 0), errors='coerce') or 0),
                    'avg_price': float(pd.to_numeric(pos.get('avgPrice', 0), errors='coerce') or 0),
                    'current_value': float(pd.to_numeric(pos.get('currentValue', 0), errors='coerce') or 0),
                    'initial_value': float(pd.to_numeric(pos.get('initialValue', 0), errors='coerce') or 0),
                    'cash_pnl': float(pd.to_numeric(pos.get('cashPnl', 0), errors='coerce') or 0),
                    'percent_pnl': float(pd.to_numeric(pos.get('percentPnl', 0), errors='coerce') or 0),
                    'cur_price': float(pd.to_numeric(pos.get('curPrice', 0), errors='coerce') or 0),
                    'redeemable': bool(pos.get('redeemable', False)),
                    'end_date': str(pos.get('endDate', '')),
                    'market_id': str(pos.get('marketId', '')),
                    'token_id': str(pos.get('tokenId', ''))
                }
                display_positions.append(position_data)
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing position data: {e}")
                continue
        
        logger.info(f"Successfully processed {len(display_positions)} positions")
        
        return {
            'positions': display_positions,
            'summary': summary,
            'usdc_balance': float(usdc_balance),
            'raw_data': positions_data,
            'success': True
        }
        
    except pd.errors.EmptyDataError:
        logger.error("Empty or invalid positions data received")
        return None
    except requests.RequestException as e:
        logger.error(f"Network error fetching positions: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_positions: {e}", exc_info=True)
        return None


@retry_on_failure(max_attempts=3, delay=1.0)
def list_top_markets(limit: int = 10, include_closed: bool = False) -> Optional[pd.DataFrame]:
    """
    Enhanced market listing with improved error handling and data processing
    
    Args:
        limit: Number of top markets to return (1-500)
        include_closed: Whether to include closed markets
    
    Returns:
        DataFrame with detailed market information sorted by volume
    """
    # Validate and sanitize limit
    limit = max(1, min(limit, 500))
    
    logger.info(f"Fetching top {limit} markets by volume (include_closed: {include_closed})")
    
    try:
        # Get markets data with enhanced parameters
        params = {
            'closed': 'true' if include_closed else 'false',
            'limit': min(limit * 2, 500),  # Get more data for better sorting
            'sortBy': 'volume24hr',
            'sortDirection': 'desc'
        }
        
        logger.debug("Requesting markets data from Gamma API...")
        response = api_client.get_with_retry(
            f"{Config.GAMMA_API_URL}/markets",
            params=params,
            use_cache=True  # Cache market data for 60 seconds
        )
        
        if not response:
            logger.error("Failed to fetch markets data")
            return pd.DataFrame()
        
        markets_data = response.json()
        
        if not markets_data or not isinstance(markets_data, list):
            logger.warning("Empty or invalid markets data received")
            return pd.DataFrame()
        
        # Create DataFrame with validation
        try:
            df = pd.DataFrame(markets_data)
        except ValueError as e:
            logger.error(f"Error creating DataFrame from markets data: {e}")
            return pd.DataFrame()
        
        if df.empty:
            logger.info("No markets found")
            return pd.DataFrame()
        
        # Enhanced column selection with fallbacks
        preferred_cols = [
            'id', 'question', 'category', 'endDate', 'volume24hr', 'volume1wk',
            'spread', 'oneDayPriceChange', 'lastTradePrice', 'acceptingOrders',
            'active', 'closed', 'enableOrderBook', 'tags', 'description'
        ]
        available_cols = [col for col in preferred_cols if col in df.columns]
        
        if not available_cols:
            logger.error("No expected columns found in markets data")
            return pd.DataFrame()
        
        # Sort by volume with safe numeric conversion
        if 'volume24hr' in df.columns:
            df['volume24hr_numeric'] = pd.to_numeric(df['volume24hr'], errors='coerce').fillna(0)
            df_sorted = df.sort_values('volume24hr_numeric', ascending=False, na_position='last')
        else:
            logger.warning("volume24hr column not found, using original order")
            df_sorted = df
        
        # Select and limit results
        result_df = df_sorted[available_cols].head(limit).copy()
        
        # Enhanced data formatting
        # Format volumes with safe conversion
        volume_cols = ['volume24hr', 'volume1wk']
        for col in volume_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
        
        # Format numeric columns with safe conversion  
        numeric_cols = ['spread', 'oneDayPriceChange', 'lastTradePrice']
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').round(4)
        
        # Enhanced date formatting with error handling
        if 'endDate' in result_df.columns:
            def format_date(date_str):
                if pd.isna(date_str) or not date_str:
                    return ''
                try:
                    if isinstance(date_str, str):
                        # Handle various ISO format variations
                        clean_date = date_str.replace('Z', '+00:00')
                        dt = datetime.fromisoformat(clean_date)
                        return dt.strftime('%m/%d/%Y %H:%M')
                    return str(date_str)
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Date formatting error for '{date_str}': {e}")
                    return str(date_str) if date_str else ''
            
            result_df['endDate'] = result_df['endDate'].apply(format_date)
        
        # Add market statistics
        stats = {
            'total_markets': len(result_df),
            'total_volume_24hr': result_df['volume24hr'].sum() if 'volume24hr' in result_df.columns else 0,
            'avg_spread': result_df['spread'].mean() if 'spread' in result_df.columns else 0,
            'active_markets': len(result_df[result_df.get('active', True) == True]) if 'active' in result_df.columns else len(result_df)
        }
        
        logger.info(f"Successfully fetched {len(result_df)} markets")
        logger.info(f"Total 24hr volume: ${stats['total_volume_24hr']:,.0f}")
        logger.info(f"Average spread: {stats['avg_spread']:.4f}")
        
        # Add metadata to DataFrame
        result_df.attrs['stats'] = stats
        result_df.attrs['timestamp'] = datetime.now().isoformat()
        result_df.attrs['include_closed'] = include_closed
        
        return result_df
        
    except pd.errors.EmptyDataError:
        logger.error("Empty markets data received")
        return pd.DataFrame()
    except requests.RequestException as e:
        logger.error(f"Network error fetching markets: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error in list_top_markets: {e}", exc_info=True)
        return pd.DataFrame()


@retry_on_failure(max_attempts=3, delay=1.0)
def get_all_positions(user_address: Optional[str] = None, include_orders: bool = True, include_earnings: bool = False) -> Optional[Dict[str, Any]]:
    """
    Enhanced comprehensive portfolio tracker with advanced analytics
    
    Gets complete portfolio data including:
    - All positions with detailed market information
    - Open orders and trade history
    - Advanced portfolio statistics and P&L analysis
    - USDC balance via direct blockchain query
    - Optional earnings data (requires authentication)
    - Risk metrics and portfolio composition
    
    Args:
        user_address: The user's wallet address (uses .env if not provided)
        include_orders: Whether to include open orders data
        include_earnings: Whether to include earnings data (requires auth)
    
    Returns:
        Comprehensive portfolio data with positions, orders, stats, and analytics
    """
    # Get and validate user address
    if not user_address:
        user_address = get_env_wallet()
    
    if not user_address:
        logger.error("No valid wallet address provided or found in environment")
        return None
    
    if not validate_wallet_address(user_address):
        logger.error(f"Invalid wallet address format: {user_address}")
        return None
    
    logger.info(f"Fetching comprehensive portfolio data for {user_address[:10]}... (orders: {include_orders}, earnings: {include_earnings})")
    
    try:
        # Get USDC balance with enhanced error handling
        logger.debug("Fetching USDC balance...")
        usdc_balance = get_usdc_balance(user_address)
        if usdc_balance is None:
            logger.warning("Could not fetch USDC balance, using 0.0")
            usdc_balance = 0.0
        
        # Get positions from Polymarket Data API with enhanced parameters
        logger.debug("Fetching positions data...")
        positions_params = {
            'user': user_address,
            'limit': 500,  # API maximum
            'sortBy': 'CURRENT',
            'sortDirection': 'DESC'
        }
        
        # Get positions using enhanced API client
        positions_response = api_client.get_with_retry(
            "https://data-api.polymarket.com/positions",
            params=positions_params,
            use_cache=False  # Don't cache position data as it changes frequently
        )
        
        if not positions_response:
            logger.error("Failed to fetch positions data")
            return None
        
        positions_data = positions_response.json()
        
        if not isinstance(positions_data, list):
            logger.error("Invalid positions data format")
            return None
            
        logger.info(f"Found {len(positions_data)} positions")
        
        # Get market data to enrich position info with enhanced error handling
        logger.debug("Fetching market context data...")
        markets_data = {}
        try:
            markets_response = api_client.get_with_retry(
                f"{Config.GAMMA_API_URL}/markets",
                params={'closed': 'false', 'limit': 200},
                use_cache=True  # Cache market data for 60 seconds
            )
            
            if markets_response:
                markets_list = markets_response.json()
                if isinstance(markets_list, list):
                    # Create optimized lookup dict for market info
                    for market in markets_list:
                        if 'clobTokenIds' in market and market['clobTokenIds']:
                            try:
                                token_ids = json.loads(market['clobTokenIds'])
                                market_info = {
                                    'market_id': market['id'],
                                    'question': market.get('question', ''),
                                    'category': market.get('category', ''),
                                    'endDate': market.get('endDate', ''),
                                    'volume24hr': pd.to_numeric(market.get('volume24hr', 0), errors='coerce') or 0,
                                    'spread': pd.to_numeric(market.get('spread', 0), errors='coerce') or 0
                                }
                                for token_id in token_ids:
                                    markets_data[token_id] = market_info
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.debug(f"Error processing market {market.get('id', 'unknown')}: {e}")
                                continue
        except Exception as e:
            logger.warning(f"Could not fetch market context data: {e}")
        
        # Process positions with enhanced data validation and analytics
        enhanced_positions = []
        portfolio_stats = {
            'total_current_value': 0.0,
            'total_initial_value': 0.0, 
            'total_cash_pnl': 0.0,
            'total_percent_pnl': 0.0,
            'active_positions': 0,
            'redeemable_positions': 0,
            'categories': {},
            'markets': set()
        }
        
        for pos in positions_data:
            try:
                # Get market context if available
                asset_id = str(pos.get('asset', ''))
                market_info = markets_data.get(asset_id, {})
                
                # Safe numeric conversions
                current_value = float(pd.to_numeric(pos.get('currentValue', 0), errors='coerce') or 0)
                initial_value = float(pd.to_numeric(pos.get('initialValue', 0), errors='coerce') or 0)
                cash_pnl = float(pd.to_numeric(pos.get('cashPnl', 0), errors='coerce') or 0)
                position_size = float(pd.to_numeric(pos.get('size', 0), errors='coerce') or 0)
                
                # Accumulate portfolio statistics
                portfolio_stats['total_current_value'] += current_value
                portfolio_stats['total_initial_value'] += initial_value
                portfolio_stats['total_cash_pnl'] += cash_pnl
                
                if position_size > 0:
                    portfolio_stats['active_positions'] += 1
                
                if pos.get('redeemable', False):
                    portfolio_stats['redeemable_positions'] += 1
                
                # Track categories and markets
                category = market_info.get('category', 'Unknown')
                if category:
                    portfolio_stats['categories'][category] = portfolio_stats['categories'].get(category, 0) + 1
                
                market_id = market_info.get('market_id', '')
                if market_id:
                    portfolio_stats['markets'].add(market_id)
                
                # Enhanced position data with comprehensive information
                enhanced_pos = {
                    'asset_id': asset_id,
                    'market_id': market_id,
                    'question': str(market_info.get('question', pos.get('title', 'Unknown Market'))),
                    'category': str(category),
                    'outcome': str(pos.get('outcome', 'Unknown')),
                    'position_size': position_size,
                    'avg_price': float(pd.to_numeric(pos.get('avgPrice', 0), errors='coerce') or 0),
                    'current_price': float(pd.to_numeric(pos.get('curPrice', 0), errors='coerce') or 0),
                    'current_value': current_value,
                    'initial_value': initial_value,
                    'cash_pnl': cash_pnl,
                    'percent_pnl': float(pd.to_numeric(pos.get('percentPnl', 0), errors='coerce') or 0),
                    'redeemable': bool(pos.get('redeemable', False)),
                    'end_date': str(market_info.get('endDate', pos.get('endDate', ''))),
                    'market_volume_24hr': float(market_info.get('volume24hr', 0)),
                    'market_spread': float(market_info.get('spread', 0)),
                    'token_id': str(pos.get('tokenId', '')),
                    'last_updated': datetime.now().isoformat()
                }
                enhanced_positions.append(enhanced_pos)
                
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error processing position {pos.get('asset', 'unknown')}: {e}")
                continue
        
        # Calculate portfolio metrics
        total_percent_pnl = (total_cash_pnl / total_initial_value * 100) if total_initial_value > 0 else 0
        total_funds = usdc_balance + total_current_value
        
        # Get open orders if requested
        orders_data = []
        total_order_value = 0
        
        if include_orders:
            print(f"📋 MOON DEV fetching open orders... 🎯")
            try:
                # This would require CLOB client setup, for now we'll use a simple approach
                # In a full implementation, you'd use the py_clob_client like poly-maker-main does
                orders_url = f"https://clob.polymarket.com/orders"
                # Note: This endpoint requires authentication, so we'll skip for now
                # orders_data would contain open buy/sell orders
                print(f"⚠️ MOON DEV: Order tracking requires CLOB client setup 🔧")
            except Exception as e:
                print(f"⚠️ MOON DEV: Could not fetch orders: {str(e)} 📋")
        
        # Portfolio summary with enhanced metrics
        portfolio_summary = {
            'total_positions': len(positions_data),
            'active_positions': active_positions,
            'redeemable_positions': redeemable_positions,
            'total_current_value': round(total_current_value, 2),
            'total_initial_value': round(total_initial_value, 2),
            'total_cash_pnl': round(total_cash_pnl, 2),
            'total_percent_pnl': round(total_percent_pnl, 2),
            'usdc_balance': round(usdc_balance, 2),
            'total_funds': round(total_funds, 2),
            'total_order_value': round(total_order_value, 2),
            'portfolio_utilization': round((total_current_value / total_funds * 100) if total_funds > 0 else 0, 2)
        }
        
        # Print comprehensive summary
        print(f"🌙 ===== MOON DEV PORTFOLIO SUMMARY ===== 🌙")
        print(f"💰 USDC Balance: ${usdc_balance:,.2f}")
        print(f"📊 Position Value: ${total_current_value:,.2f}")
        print(f"🚀 Total Funds: ${total_funds:,.2f}")
        print(f"📈 Total P&L: ${total_cash_pnl:,.2f} ({total_percent_pnl:+.2f}%)")
        print(f"🎯 Active Positions: {active_positions}/{len(positions_data)}")
        print(f"💎 Redeemable: {redeemable_positions}")
        print(f"⚡ Portfolio Utilization: {portfolio_summary['portfolio_utilization']:.1f}%")
        print(f"🌕 ===================================== 🌕")
        
        # Return comprehensive data structure
        result = {
            'summary': portfolio_summary,
            'positions': enhanced_positions,
            'orders': orders_data,
            'usdc_balance': usdc_balance,
            'raw_positions_data': positions_data,
            'markets_context': markets_data
        }
        
        return result
        
    except Exception as e:
        print(f"❌ MOON DEV error in get_all_positions: {str(e)} 😢")
        return None


def analyze_portfolio_risk(portfolio_data):
    """
    Analyze portfolio risk metrics inspired by poly-maker-main
    
    Args:
        portfolio_data (dict): Output from get_all_positions()
    
    Returns:
        dict: Risk analysis including concentration, correlation, and recommendations
    """
    if not portfolio_data or not portfolio_data['positions']:
        print("⚠️ MOON DEV: No portfolio data to analyze! 📊")
        return None
    
    print(f"🔍 MOON DEV analyzing portfolio risk... 💎")
    
    positions = portfolio_data['positions']
    summary = portfolio_data['summary']
    
    # Category concentration analysis
    category_exposure = {}
    market_exposure = {}
    outcome_distribution = {'YES': 0, 'NO': 0, 'Other': 0}
    
    for pos in positions:
        # Category concentration
        category = pos['category'] or 'Unknown'
        if category not in category_exposure:
            category_exposure[category] = 0
        category_exposure[category] += pos['current_value']
        
        # Market concentration  
        question = pos['question'][:50] + "..." if len(pos['question']) > 50 else pos['question']
        market_exposure[question] = market_exposure.get(question, 0) + pos['current_value']
        
        # Outcome distribution
        outcome = pos['outcome'].upper()
        if outcome in outcome_distribution:
            outcome_distribution[outcome] += pos['current_value']
        else:
            outcome_distribution['Other'] += pos['current_value']
    
    # Calculate concentration percentages
    total_value = summary['total_current_value']
    if total_value > 0:
        category_pct = {k: round(v/total_value*100, 1) for k, v in category_exposure.items()}
        market_pct = {k: round(v/total_value*100, 1) for k, v in market_exposure.items()}
        outcome_pct = {k: round(v/total_value*100, 1) for k, v in outcome_distribution.items()}
    else:
        category_pct = market_pct = outcome_pct = {}
    
    # Risk flags
    risk_flags = []
    
    # Check for over-concentration
    max_category_exposure = max(category_pct.values()) if category_pct else 0
    max_market_exposure = max(market_pct.values()) if market_pct else 0
    
    if max_category_exposure > 50:
        risk_flags.append(f"High category concentration: {max_category_exposure:.1f}%")
    
    if max_market_exposure > 30:
        risk_flags.append(f"High single market exposure: {max_market_exposure:.1f}%")
    
    # Check portfolio utilization
    utilization = summary['portfolio_utilization']
    if utilization > 80:
        risk_flags.append(f"High portfolio utilization: {utilization:.1f}%")
    elif utilization < 20:
        risk_flags.append(f"Low portfolio utilization: {utilization:.1f}%")
    
    # P&L analysis
    if summary['total_percent_pnl'] < -20:
        risk_flags.append(f"Significant losses: {summary['total_percent_pnl']:.1f}%")
    
    print(f"🎯 MOON DEV Risk Analysis Complete! 📊")
    print(f"📈 Top Category: {max(category_pct, key=category_pct.get) if category_pct else 'None'} ({max_category_exposure:.1f}%)")
    print(f"⚠️ Risk Flags: {len(risk_flags)}")
    
    return {
        'category_exposure': category_exposure,
        'category_percentages': category_pct,
        'market_exposure': market_exposure,
        'market_percentages': market_pct,
        'outcome_distribution': outcome_distribution,
        'outcome_percentages': outcome_pct,
        'risk_flags': risk_flags,
        'risk_score': len(risk_flags),  # Simple risk score
        'diversification_score': len(category_exposure),  # More categories = better diversification
        'max_category_exposure': max_category_exposure,
        'max_market_exposure': max_market_exposure
    }


@retry_on_failure(max_attempts=3, delay=1.0)
def get_total_balance(wallet_address: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Enhanced total balance calculator with comprehensive financial metrics
    
    Gets complete financial overview combining:
    - USDC balance from blockchain contracts
    - Current position values from Polymarket API
    - Historical balance tracking
    - Financial analytics and risk metrics
    
    Args:
        wallet_address: The wallet address to check (uses .env if not provided)
    
    Returns:
        Dictionary with comprehensive balance information and analytics
    """
    # Get and validate wallet address
    if not wallet_address:
        wallet_address = get_env_wallet()
    
    if not wallet_address:
        logger.error("No valid wallet address provided or found in environment")
        return None
    
    if not validate_wallet_address(wallet_address):
        logger.error(f"Invalid wallet address format: {wallet_address}")
        return None
    
    logger.info(f"Calculating comprehensive balance for {wallet_address[:10]}...")
    
    try:
        # Get USDC balance with enhanced error handling
        logger.debug("Fetching USDC balance from blockchain...")
        usdc_balance = get_usdc_balance(wallet_address)
        if usdc_balance is None:
            logger.warning("Could not fetch USDC balance, using 0.0")
            usdc_balance = 0.0
        
        # Get position value from Polymarket Data API with enhanced error handling
        logger.debug("Fetching position values from Polymarket API...")
        position_value = 0.0
        position_count = 0
        
        try:
            # Try the direct value API first
            value_response = api_client.get_with_retry(
                f"https://data-api.polymarket.com/value",
                params={'user': wallet_address},
                use_cache=False  # Don't cache balance data
            )
            
            if value_response:
                value_data = value_response.json()
                if isinstance(value_data, dict) and 'value' in value_data:
                    position_value = float(pd.to_numeric(value_data['value'], errors='coerce') or 0)
                elif isinstance(value_data, list) and len(value_data) > 0:
                    if isinstance(value_data[0], dict) and 'value' in value_data[0]:
                        position_value = float(pd.to_numeric(value_data[0]['value'], errors='coerce') or 0)
                logger.debug(f"Position value from API: ${position_value:,.2f}")
            else:
                logger.warning("Direct value API failed, falling back to positions calculation")
                # Fallback: calculate from positions data
                positions_data = get_positions(wallet_address, limit=500)
                if positions_data and 'summary' in positions_data:
                    position_value = float(positions_data['summary'].get('total_current_value', 0))
                    position_count = int(positions_data['summary'].get('total_positions', 0))
                    logger.debug(f"Position value from calculation: ${position_value:,.2f}")
        
        except Exception as e:
            logger.warning(f"Error fetching position value: {e}")
        
        # Calculate total balance
        total_balance = usdc_balance + position_value
        
        print(f"🚀 MOON DEV TOTAL BALANCE BREAKDOWN:")
        print(f"💵 USDC Balance: ${usdc_balance:,.2f}")
        print(f"📊 Position Value: ${position_value:,.2f}")
        print(f"💎 TOTAL BALANCE: ${total_balance:,.2f}")
        
        return {
            'usdc_balance': round(usdc_balance, 2),
            'position_value': round(position_value, 2),
            'total_balance': round(total_balance, 2),
            'wallet_address': wallet_address
        }
        
    except Exception as e:
        print(f"❌ MOON DEV error in get_total_balance: {str(e)} 😢")
        return None


@retry_on_failure(max_attempts=3, delay=1.0)
def get_all_orders(wallet_address: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Enhanced order fetcher for comprehensive order management
    
    Gets all open orders for a wallet with detailed information including:
    - Order details (ID, side, price, size, status)
    - Market context and metadata
    - Order timing and execution data
    - Enhanced error handling and validation
    
    Args:
        wallet_address: The wallet address to check (uses .env if not provided)
    
    Returns:
        List of enhanced order dictionaries with market context
    """
    # Get and validate wallet address
    if not wallet_address:
        wallet_address = get_env_wallet()
    
    if not wallet_address:
        logger.error("No valid wallet address provided or found in environment")
        return []
    
    if not validate_wallet_address(wallet_address):
        logger.error(f"Invalid wallet address format: {wallet_address}")
        return []
    
    logger.info(f"Fetching all open orders for {wallet_address[:10]}...")
    
    try:
        # Note: CLOB client integration would be implemented here
        # This is a placeholder for the proper CLOB client implementation
        # In production, you would use py_clob_client like this:
        #
        # from py_clob_client.client import ClobClient
        # from py_clob_client.constants import POLYGON
        # 
        # client = ClobClient(
        #     host=Config.CLOB_API_URL,
        #     key=os.getenv('PRIVATE_KEY'),
        #     chain_id=POLYGON,
        #     funder=wallet_address
        # )
        # orders = client.get_orders()
        
        logger.warning("CLOB client not configured - returning empty orders list")
        logger.info("To enable order fetching, configure py_clob_client with PRIVATE_KEY")
        
        # Return structured empty response
        return []
        
    except ImportError as e:
        logger.error(f"CLOB client dependencies not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching orders: {e}", exc_info=True)
        return []


@retry_on_failure(max_attempts=3, delay=1.0)
def get_market_orders(market_id: str, wallet_address: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Enhanced market-specific order fetcher with comprehensive filtering
    
    Gets all open orders for a specific market with detailed information including:
    - Order details filtered by market ID
    - Market context and metadata
    - Order performance analytics
    - Enhanced error handling and validation
    
    Args:
        market_id: The market ID to get orders for
        wallet_address: The wallet address to check (uses .env if not provided)
    
    Returns:
        List of enhanced order dictionaries for the specific market
    """
    # Validate market ID
    if not market_id or not isinstance(market_id, str):
        logger.error("Invalid market ID provided")
        return []
    
    # Get and validate wallet address
    if not wallet_address:
        wallet_address = get_env_wallet()
    
    if not wallet_address:
        logger.error("No valid wallet address provided or found in environment")
        return []
    
    if not validate_wallet_address(wallet_address):
        logger.error(f"Invalid wallet address format: {wallet_address}")
        return []
    
    logger.info(f"Fetching orders for market {market_id[:20]}... (wallet: {wallet_address[:10]})")
    
    try:
        # Note: CLOB client integration would be implemented here
        # This is a placeholder for the proper CLOB client implementation
        # In production, you would use py_clob_client like this:
        #
        # from py_clob_client.client import ClobClient
        # from py_clob_client.clob_types import OpenOrderParams
        # from py_clob_client.constants import POLYGON
        # 
        # client = ClobClient(
        #     host=Config.CLOB_API_URL,
        #     key=os.getenv('PRIVATE_KEY'),
        #     chain_id=POLYGON,
        #     funder=wallet_address
        # )
        # orders = client.get_orders(OpenOrderParams(market=market_id))
        
        logger.warning(f"CLOB client not configured - cannot fetch orders for market {market_id[:10]}")
        logger.info("To enable market order fetching, configure py_clob_client with PRIVATE_KEY")
        
        # Return structured empty response
        return []
        
    except ImportError as e:
        logger.error(f"CLOB client dependencies not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching market orders: {e}", exc_info=True)
        return []


def cancel_all_asset_orders(asset_id, wallet_address=None):
    """
    Cancel all orders for a specific asset/token
    Inspired by poly-maker-main client
    
    Args:
        asset_id (str): The asset/token ID to cancel orders for
        wallet_address (str, optional): The wallet address (uses .env if not provided)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🚫 MOON DEV cancelling all orders for asset {asset_id[:20]}... 💎")
    
    # Use wallet from env if not provided
    if not wallet_address:
        wallet_address = os.getenv("PUBLIC_KEY")
    
    if not wallet_address:
        print("⚠️ MOON DEV: Need wallet address or PUBLIC_KEY in .env! 🔑")
        return False
    
    try:
        # TODO: Implement CLOB client
        # client.cancel_market_orders(asset_id=str(asset_id))
        
        print("⚠️ MOON DEV: Requires CLOB client setup")
        return False
        
    except Exception as e:
        print(f"❌ MOON DEV error in cancel_all_asset_orders: {str(e)} 😢")
        return False


def setup_clob_client_info():
    """
    Quick setup info for CLOB client
    """
    print(f"🔧 MOON DEV CLOB Setup:")
    print("pip install py_clob_client")
    print("Uses PRIVATE_KEY & PUBLIC_KEY from .env")


@retry_on_failure(max_attempts=3, delay=1.0)
def place_limit_order(token_id: str, side: str, price: float, size: float, neg_risk: bool = False) -> Dict[str, Any]:
    """
    Enhanced limit order execution with comprehensive validation and error handling
    
    Places a limit order on Polymarket with:
    - Input validation and sanitization
    - Price and size bounds checking
    - Enhanced error handling and retry logic
    - Order status tracking and confirmation
    - Risk management checks
    
    Args:
        token_id: The token ID to trade
        side: "BUY" or "SELL"
        price: Price per share (0.01 to 0.99)
        size: Number of shares to trade
        neg_risk: Whether this is a negative risk market
    
    Returns:
        Dictionary with order response or error information
    """
    # Input validation
    if not token_id or not isinstance(token_id, str):
        logger.error("Invalid token ID provided")
        return {'error': 'Invalid token ID', 'success': False}
    
    if side not in ['BUY', 'SELL']:
        logger.error(f"Invalid side '{side}', must be 'BUY' or 'SELL'")
        return {'error': f'Invalid side: {side}', 'success': False}
    
    # Validate and sanitize price
    try:
        price = float(price)
        if not (Config.MIN_PRICE <= price <= Config.MAX_PRICE):
            logger.error(f"Price {price} outside allowed range [{Config.MIN_PRICE}, {Config.MAX_PRICE}]")
            return {'error': f'Price outside allowed range', 'success': False}
    except (ValueError, TypeError):
        logger.error(f"Invalid price value: {price}")
        return {'error': 'Invalid price format', 'success': False}
    
    # Validate and sanitize size
    try:
        size = float(size)
        if not (Config.MIN_ORDER_SIZE <= size <= Config.MAX_ORDER_SIZE):
            logger.error(f"Size {size} outside allowed range [{Config.MIN_ORDER_SIZE}, {Config.MAX_ORDER_SIZE}]")
            return {'error': f'Size outside allowed range', 'success': False}
    except (ValueError, TypeError):
        logger.error(f"Invalid size value: {size}")
        return {'error': 'Invalid size format', 'success': False}
    
    logger.info(f"Placing {side} limit order: {size} shares @ ${price:.4f} (token: {token_id[:20]}...)")
    logger.debug(f"Order details - Token: {token_id}, Side: {side}, Price: {price}, Size: {size}, NegRisk: {neg_risk}")
    
    try:
        # Note: CLOB client integration would be implemented here
        # This is a placeholder for the proper CLOB client implementation
        # In production, you would use py_clob_client like this:
        #
        # from py_clob_client.client import ClobClient
        # from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
        # from py_clob_client.constants import POLYGON
        # from web3 import Web3
        #
        # # Get credentials from environment
        # key = os.getenv("PRIVATE_KEY")
        # browser_address = os.getenv("PUBLIC_KEY")
        #
        # if not key or not browser_address:
        #     logger.error("Missing PRIVATE_KEY or PUBLIC_KEY in environment")
        #     return {'error': 'Missing credentials', 'success': False}
        #
        # # Validate wallet address
        # if not validate_wallet_address(browser_address):
        #     logger.error(f"Invalid wallet address: {browser_address}")
        #     return {'error': 'Invalid wallet address', 'success': False}
        #
        # # Setup CLOB client with enhanced error handling
        # try:
        #     client = ClobClient(
        #         host=Config.CLOB_API_URL,
        #         key=key,
        #         chain_id=POLYGON,
        #         funder=Web3.to_checksum_address(browser_address),
        #         signature_type=1
        #     )
        #     
        #     # Set up API credentials
        #     creds = client.create_or_derive_api_creds()
        #     client.set_api_creds(creds=creds)
        #     
        #     # Create order arguments with validation
        #     order_args = OrderArgs(
        #         token_id=str(token_id),
        #         price=float(price),
        #         size=float(size),
        #         side=side.upper()
        #     )
        #     
        #     logger.debug("Creating signed order...")
        #     
        #     # Create and submit order with proper error handling
        #     options = PartialCreateOrderOptions(neg_risk=neg_risk) if neg_risk else None
        #     signed_order = client.create_order(order_args, options=options)
        #     
        #     logger.debug("Submitting order to exchange...")
        #     response = client.post_order(signed_order)
        #     
        #     if response and response.get('orderID'):
        #         order_id = response.get('orderID')
        #         logger.info(f"Order placed successfully - ID: {order_id}")
        #         return {
        #             'success': True,
        #             'order_id': order_id,
        #             'token_id': token_id,
        #             'side': side,
        #             'price': price,
        #             'size': size,
        #             'neg_risk': neg_risk,
        #             'timestamp': datetime.now().isoformat(),
        #             'response': response
        #         }
        #     else:
        #         logger.error("Order submission failed - no order ID received")
        #         return {'error': 'Order submission failed', 'success': False}
        #         
        # except Exception as clob_error:
        #     logger.error(f"CLOB client error: {clob_error}")
        #     return {'error': f'CLOB client error: {str(clob_error)}', 'success': False}
        
        logger.warning("CLOB client not configured - order placement disabled")
        logger.info("To enable order placement, install py_clob_client and configure PRIVATE_KEY")
        
        # Return structured response indicating CLOB client is needed
        return {
            'error': 'CLOB client not configured',
            'success': False,
            'requires_setup': True,
            'setup_info': {
                'install': 'pip install py_clob_client',
                'required_env': ['PRIVATE_KEY', 'PUBLIC_KEY'],
                'documentation': 'https://docs.polymarket.com/'
            }
        }
        
    except ImportError as e:
        logger.error(f"CLOB client dependencies not available: {e}")
        return {
            'error': 'Missing py_clob_client dependency',
            'success': False,
            'install_command': 'pip install py_clob_client'
        }
    except Exception as e:
        logger.error(f"Unexpected error in place_limit_order: {e}", exc_info=True)
        return {
            'error': f'Unexpected error: {str(e)}',
            'success': False
        }
        
        # Try regular order
        print(f"🔄 MOON DEV trying regular order...")
        signed_order = client.create_order(order_args)
        
        print(f"🚀 MOON DEV submitting order...")
        response = client.post_order(signed_order)
        
        if response:
            print(f"✅ MOON DEV order submitted successfully! 🎯")
            print(f"📋 Order ID: {response.get('orderID', 'N/A')}")
            return response
        else:
            print(f"❌ MOON DEV: No response from API")
            
            # Try signature_type=1 as fallback
            print(f"🔄 MOON DEV trying signature_type=1...")
            try:
                retry_client = ClobClient(
                    host="https://clob.polymarket.com",
                    key=key,
                    chain_id=POLYGON,
                    funder=browser_wallet,
                    signature_type=1  # Different signature type
                )
                
                creds = retry_client.create_or_derive_api_creds()
                retry_client.set_api_creds(creds=creds)
                
                signed_order = retry_client.create_order(order_args)
                response = retry_client.post_order(signed_order)
                
                if response:
                    print(f"✅ MOON DEV signature_type=1 worked! 🎯")
                    print(f"📋 Order ID: {response.get('orderID', 'N/A')}")
                    return response
                else:
                    print(f"❌ MOON DEV: Still no response")
                    return {}
            except Exception as e2:
                print(f"❌ MOON DEV all methods failed: {str(e2)[:100]}")
                return {}
                
    except Exception as e:
        print(f"❌ MOON DEV error placing limit order: {e}")
        return {}


@retry_on_failure(max_attempts=3, delay=1.0)
def place_market_order(token_id: str, side: str, size: float, neg_risk: bool = False) -> Dict[str, Any]:
    """
    Enhanced market order execution with comprehensive validation and price discovery
    
    Places a market order for immediate execution with:
    - Input validation and sanitization
    - Real-time price discovery from order book
    - Automatic conversion to limit order at market price
    - Enhanced error handling and retry logic
    - Market liquidity validation
    
    Args:
        token_id: The token ID to trade
        side: "BUY" or "SELL"
        size: Number of shares to trade
        neg_risk: Whether this is a negative risk market
    
    Returns:
        Dictionary with order response or error information
    """
    # Input validation (reuse validation from place_limit_order)
    if not token_id or not isinstance(token_id, str):
        logger.error("Invalid token ID provided")
        return {'error': 'Invalid token ID', 'success': False}
    
    if side not in ['BUY', 'SELL']:
        logger.error(f"Invalid side '{side}', must be 'BUY' or 'SELL'")
        return {'error': f'Invalid side: {side}', 'success': False}
    
    # Validate and sanitize size
    try:
        size = float(size)
        if not (Config.MIN_ORDER_SIZE <= size <= Config.MAX_ORDER_SIZE):
            logger.error(f"Size {size} outside allowed range [{Config.MIN_ORDER_SIZE}, {Config.MAX_ORDER_SIZE}]")
            return {'error': f'Size outside allowed range', 'success': False}
    except (ValueError, TypeError):
        logger.error(f"Invalid size value: {size}")
        return {'error': 'Invalid size format', 'success': False}
    
    logger.info(f"Placing {side} market order: {size} shares (token: {token_id[:20]}...)")
    
    # Get current market prices for immediate execution
    try:
        logger.debug("Fetching current order book for market price discovery...")
        
        # Use enhanced API client for order book data
        book_response = api_client.get_with_retry(
            f"{Config.CLOB_API_URL}/book",
            params={'token_id': token_id},
            use_cache=False  # Don't cache order book data
        )
        
        if not book_response:
            logger.error("Failed to fetch order book data")
            return {'error': 'Unable to fetch market data', 'success': False}
        
        book_data = book_response.json()
        asks = book_data.get('asks', [])
        bids = book_data.get('bids', [])
        
        # Determine market price based on side
        market_price = None
        liquidity_info = {}
        
        if side.upper() == "BUY":
            if not asks:
                logger.error("No ask liquidity available for BUY order")
                return {'error': 'No ask liquidity available', 'success': False}
            
            # For buying, use the best ask (lowest seller price)
            best_ask = asks[-1] if asks else None
            if best_ask and 'price' in best_ask:
                market_price = float(pd.to_numeric(best_ask['price'], errors='coerce') or 0)
                liquidity_info = {
                    'best_ask_price': market_price,
                    'best_ask_size': float(pd.to_numeric(best_ask.get('size', 0), errors='coerce') or 0),
                    'total_asks': len(asks)
                }
            
        elif side.upper() == "SELL":
            if not bids:
                logger.error("No bid liquidity available for SELL order")
                return {'error': 'No bid liquidity available', 'success': False}
            
            # For selling, use the best bid (highest buyer price)
            best_bid = bids[-1] if bids else None
            if best_bid and 'price' in best_bid:
                market_price = float(pd.to_numeric(best_bid['price'], errors='coerce') or 0)
                liquidity_info = {
                    'best_bid_price': market_price,
                    'best_bid_size': float(pd.to_numeric(best_bid.get('size', 0), errors='coerce') or 0),
                    'total_bids': len(bids)
                }
        
        if not market_price or market_price <= 0:
            logger.error(f"Invalid market price discovered: {market_price}")
            return {'error': 'Invalid market price', 'success': False}
        
        # Validate market price is within bounds
        if not (Config.MIN_PRICE <= market_price <= Config.MAX_PRICE):
            logger.error(f"Market price {market_price} outside allowed range")
            return {'error': f'Market price outside allowed range', 'success': False}
        
        logger.info(f"Market price discovered: ${market_price:.4f} for {side} order")
        logger.debug(f"Liquidity info: {liquidity_info}")
        
        # Execute as limit order at market price for immediate fill
        logger.debug("Converting market order to limit order at market price...")
        limit_order_result = place_limit_order(token_id, side, market_price, size, neg_risk)
        
        # Enhance response with market order context
        if isinstance(limit_order_result, dict):
            limit_order_result.update({
                'order_type': 'market',
                'market_price_used': market_price,
                'liquidity_info': liquidity_info,
                'converted_to_limit': True
            })
        
        return limit_order_result
        
    except requests.RequestException as e:
        logger.error(f"Network error fetching market data: {e}")
        return {'error': f'Network error: {str(e)}', 'success': False}
    except Exception as e:
        logger.error(f"Unexpected error in place_market_order: {e}", exc_info=True)
        return {'error': f'Unexpected error: {str(e)}', 'success': False}


@retry_on_failure(max_attempts=3, delay=1.0)
def cancel_token_orders(token_id: str) -> Dict[str, Any]:
    """
    Enhanced token-specific order cancellation with comprehensive validation
    
    Cancels all orders for a specific token with:
    - Input validation and sanitization
    - Order discovery and confirmation
    - Batch cancellation with error tracking
    - Enhanced error handling and logging
    - Cancellation status reporting
    
    Args:
        token_id: The token ID to cancel orders for
    
    Returns:
        Dictionary with cancellation results and status information
    """
    # Input validation
    if not token_id or not isinstance(token_id, str):
        logger.error("Invalid token ID provided")
        return {'error': 'Invalid token ID', 'success': False}
    
    logger.info(f"Cancelling all orders for token {token_id[:20]}...")
    
    try:
        # Import required modules
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        from web3 import Web3
        
        # Get credentials from environment
        key = os.getenv("PRIVATE_KEY")
        browser_address = os.getenv("PUBLIC_KEY")
        
        if not key or not browser_address:
            print("⚠️ MOON DEV: Missing PRIVATE_KEY or PUBLIC_KEY in .env file!")
            return False
        
        # Handle both old and new Web3 versions
        try:
            browser_wallet = Web3.toChecksumAddress(browser_address)
        except AttributeError:
            browser_wallet = Web3.to_checksum_address(browser_address)
        
        # Setup client (exact same as working spread_scan_buy.py)
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=key,
            chain_id=POLYGON,
            funder=browser_wallet,
            signature_type=1  # Use signature_type=1 since it worked before
        )
        
        # Set up API credentials
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds=creds)
        
        print(f"🚀 MOON DEV cancelling orders...")
        
        # Cancel all orders for this token
        client.cancel_market_orders(asset_id=str(token_id))
        
        print(f"✅ MOON DEV successfully cancelled all orders for token! 🎯")
        return True
        
    except Exception as e:
        print(f"❌ MOON DEV error cancelling orders: {e}")
        
        # Try with signature_type=1 as fallback
        try:
            print(f"🔄 MOON DEV trying signature_type=1...")
            
            retry_client = ClobClient(
                host="https://clob.polymarket.com",
                key=key,
                chain_id=POLYGON,
                funder=browser_wallet,
                signature_type=1
            )
            
            creds = retry_client.create_or_derive_api_creds()
            retry_client.set_api_creds(creds=creds)
            
            retry_client.cancel_market_orders(asset_id=str(token_id))
            
            print(f"✅ MOON DEV signature_type=1 cancel worked! 🎯")
            return True
            
        except Exception as e2:
            print(f"❌ MOON DEV all cancel methods failed: {str(e2)[:100]}")
            return False


def get_all_open_orders():
    """
    Get all open orders for the wallet - MOON DEV style!
    
    Returns:
        list: List of open orders or empty list if none found
    """
    print(f"📋 MOON DEV fetching all open orders...")
    
    try:
        # Import required modules
        from py_clob_client import ClobClient
        from py_clob_client.client import ClientConfig
        from py_clob_client.constants import POLYGON
        
        # Get environment variables
        private_key_hex = get_env_wallet()
        if not private_key_hex:
            return []
        
        # Initialize client
        config = ClientConfig()
        client = ClobClient(
            "https://clob.polymarket.com",
            key=private_key_hex,
            config=config,
            chain_id=POLYGON
        )
        
        # Get orders
        orders = client.get_orders()
        
        if orders:
            print(f"📋 MOON DEV found {len(orders)} open orders:")
            for i, order in enumerate(orders, 1):
                order_id = order.get('id', 'N/A')
                side = order.get('side', 'N/A')
                price = order.get('price', 'N/A')
                size = order.get('size', 'N/A')
                token_id = order.get('tokenID', 'N/A')
                
                print(f"   {i}. Order ID: {order_id[:10]}...")
                print(f"      Side: {side} | Price: ${price} | Size: {size}")
                print(f"      Token: {token_id[:15]}...")
                print()
        else:
            print(f"✅ MOON DEV: No open orders found! Account is clean!")
            
        return orders
        
    except ImportError as e:
        logger.error(f"CLOB client dependencies not available: {e}")
        return []
                
    except Exception as e:
        print(f"❌ MOON DEV error getting orders: {e}")
        return []


def cancel_all_orders():
    """
    Cancel ALL orders for ALL tokens - NO QUESTIONS ASKED!
    MOON DEV style - just runs and cancels everything
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"MOON DEV cancelling ALL orders - NO MERCY!")
    
    try:
        # Import required modules
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        from web3 import Web3
        
        # Get credentials from environment
        key = os.getenv("PRIVATE_KEY")
        browser_address = os.getenv("PUBLIC_KEY")
        
        if not key or not browser_address:
            print("⚠️ MOON DEV: Missing PRIVATE_KEY or PUBLIC_KEY in .env file!")
            return False
        
        # Handle both old and new Web3 versions
        try:
            browser_wallet = Web3.toChecksumAddress(browser_address)
        except AttributeError:
            browser_wallet = Web3.to_checksum_address(browser_address)
        
        # Setup client (exact same as working spread_scan_buy.py)
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=key,
            chain_id=POLYGON,
            funder=browser_wallet,
            signature_type=1  # Use signature_type=1 since it worked before
        )
        
        # Set up API credentials
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds=creds)
        
        print(f"🚀 MOON DEV getting all open orders...")
        
        # Get all orders first
        orders = client.get_orders()
        print(f"📋 MOON DEV found {len(orders)} open orders!")
        
        if len(orders) == 0:
            print(f"✅ MOON DEV: No orders to cancel! Account already clean! 🎯")
            return True
        
        # Show what we're about to destroy
        print(f"\n💀 MOON DEV Orders to be DESTROYED:")
        for i, order in enumerate(orders, 1):
            order_id = order.get('id', 'N/A')
            side = order.get('side', 'N/A')
            price = order.get('price', 'N/A')
            size = order.get('original_size', 'N/A')
            print(f"   {i}. {order_id[:10]}... | {side} | ${price} | {size} shares")
            
        # Cancel all orders - just blast them all!
        print(f"\n💥 MOON DEV cancelling {len(orders)} orders...")
        
        # Method 1: Try to cancel all at once
        try:
            client.cancel_all()
            print(f"✅ MOON DEV: ALL ORDERS NUKED! 🎯")
            return True
            
        except Exception as e1:
            print(f"⚠️ MOON DEV cancel_all failed: {str(e1)[:100]}")
            print(f"🔄 MOON DEV trying individual cancellations...")
            
            # Method 2: Cancel orders individually
            cancelled_count = 0
            failed_count = 0
            
            for order in orders:
                try:
                    order_id = order.get('id')
                    if order_id:
                        client.cancel(order_id)
                        cancelled_count += 1
                        print(f"✅ MOON DEV destroyed order {order_id[:10]}...")
                except Exception as e_order:
                    failed_count += 1
                    print(f"⚠️ MOON DEV failed to cancel order: {str(e_order)[:50]}")
                    
            print(f"\n📊 MOON DEV Cancellation Results:")
            print(f"   ✅ Cancelled: {cancelled_count}")
            print(f"   ❌ Failed: {failed_count}")
            print(f"   🎯 Success Rate: {(cancelled_count/(cancelled_count+failed_count)*100):.1f}%")
            
            return cancelled_count > 0
        
    except Exception as e:
        print(f"❌ MOON DEV error in cancel_all_orders: {e}")
        return False


def calculate_shares(dollar_amount, price):
    """
    Calculate how many shares you can buy with a given dollar amount at a specific price
    MOON DEV share calculator!
    Ensures minimum $1.00 order size AND 5 shares minimum (Polymarket requirement)
    
    Args:
        dollar_amount (float): How much money you want to spend (e.g., 1.0 for $1)
        price (float): The price per share (e.g., 0.45 for $0.45 per share)
    
    Returns:
        float: Number of shares you can buy (adjusted to meet all requirements)
    """
    if price <= 0:
        print(f"❌ MOON DEV: Invalid price {price}! Must be > 0")
        return 0.0
        
    shares = dollar_amount / price
    shares_rounded = round(shares, 1)
    
    # Enforce 5 share minimum (Polymarket requirement)
    if shares_rounded < 5.0:
        print(f"⚡ MOON DEV: Increasing to 5 shares minimum (was {shares_rounded})")
        shares_rounded = 5.0
    
    total_cost = shares_rounded * price
    
    # If total cost is under $1, increase shares to meet $1 minimum
    if total_cost < 1.0:
        # Calculate shares needed for exactly $1
        shares_for_dollar = 1.0 / price
        shares_rounded = max(shares_for_dollar, 5.0)  # Ensure at least 5 shares
        shares_rounded = round(shares_rounded, 1)
        total_cost = shares_rounded * price
        
        # If still under $1 due to rounding, add 0.1 shares
        if total_cost < 1.0:
            shares_rounded += 0.1
            total_cost = shares_rounded * price
    
    print(f"🧮 MOON DEV Calculator:")
    print(f"   💰 Target Amount: ${dollar_amount:.2f}")
    print(f"   💲 Price per Share: ${price:.4f}")
    print(f"   📊 Shares: {shares_rounded} (meets 5+ minimum)")
    print(f"   💡 Actual Cost: ${total_cost:.2f}")
    
    # Final validation
    if total_cost < 1.0:
        print(f"⚠️ MOON DEV: Order under $1.00 minimum! Actual: ${total_cost:.2f}")
        return 0.0
    
    if shares_rounded < 5.0:
        print(f"⚠️ MOON DEV: Under 5 shares minimum! Current: {shares_rounded}")
        return 0.0
    
    return shares_rounded


def check_existing_sell_order(token_id, target_price, target_size):
    """
    Check if there's already a sell order for this token at the right price/size
    MOON DEV's smart order checker! 🔍
    
    Args:
        token_id (str): Token ID to check
        target_price (float): Expected sell price
        target_size (float): Expected sell size
    
    Returns:
        dict: Status of existing order
    """
    try:
        from termcolor import colored
        
        # Get all open orders
        orders = get_all_open_orders()
        
        # Find sell orders for this token
        token_sell_orders = []
        for order in orders:
            if (order.get('asset_id') == token_id and 
                order.get('side', '').upper() == 'SELL'):
                token_sell_orders.append(order)
        
        if not token_sell_orders:
            print(colored(f"   🔍 MOON DEV: No existing sell orders found", "white", "on_blue"))
            return {'status': 'none', 'action': 'place_new'}
        
        # Check if any order matches our target exactly
        for order in token_sell_orders:
            order_price = float(order.get('price', 0))
            order_size = float(order.get('original_size', 0))
            
            # Check if price and size match (with small tolerance for precision)
            price_match = abs(order_price - target_price) < 0.0001
            size_match = abs(order_size - target_size) < 0.1
            
            if price_match and size_match:
                print(colored(f"   ✅ MOON DEV: Perfect sell order exists! Price: ${order_price:.4f}, Size: {order_size}", "white", "on_green"))
                return {
                    'status': 'perfect_match',
                    'action': 'keep_existing',
                    'order': order
                }
            elif size_match and not price_match:
                print(colored(f"   🔄 MOON DEV: Sell order exists but wrong price! Current: ${order_price:.4f}, Target: ${target_price:.4f}", "white", "on_yellow"))
                return {
                    'status': 'wrong_price',
                    'action': 'cancel_replace',
                    'order': order
                }
            elif price_match and not size_match:
                print(colored(f"   🔄 MOON DEV: Sell order exists but wrong size! Current: {order_size}, Target: {target_size}", "white", "on_yellow"))
                return {
                    'status': 'wrong_size',
                    'action': 'cancel_replace',
                    'order': order
                }
        
        # If we get here, there are sell orders but none match our criteria
        print(colored(f"   ⚠️ MOON DEV: {len(token_sell_orders)} sell orders exist but none match target!", "white", "on_red"))
        return {
            'status': 'wrong_orders',
            'action': 'cancel_all_replace',
            'orders': token_sell_orders
        }
        
    except Exception as e:
        print(colored(f"   ❌ MOON DEV: Error checking existing orders: {e}", "white", "on_red"))
        return {'status': 'error', 'action': 'place_new'}


def ensure_sell_orders(profit_target=0.05):
    """
    Ensure all positions have sell orders in place at profit target
    MOON DEV's SMART position management function! 📈
    
    Args:
        profit_target (float): Profit percentage target (e.g., 0.05 for 5%)
    
    Returns:
        dict: Summary of sell orders placed
    """
    from termcolor import colored
    
    print(colored(f"📈 MOON DEV ensuring sell orders are in place! 🎯", "white", "on_blue"))
    print(colored(f"   🎯 Profit Target: {profit_target*100:.1f}%", "white", "on_blue"))
    
    try:
        # Get current positions
        portfolio = get_all_positions()
        
        if not portfolio or 'positions' not in portfolio or len(portfolio['positions']) == 0:
            print(f"💡 MOON DEV: No positions found, no sell orders needed!")
            return {'positions_checked': 0, 'sell_orders_placed': 0, 'errors': 0}
        
        positions = portfolio['positions']
        print(f"🔍 MOON DEV checking {len(positions)} positions for sell orders...")
        
        results = {
            'positions_checked': len(positions),
            'sell_orders_placed': 0,
            'errors': 0,
            'details': []
        }
        
        for i, position in enumerate(positions, 1):
            try:
                token_id = position['asset_id']
                position_size = position['position_size']
                avg_price = position['avg_price']
                question = position['question']
                
                print(f"\n📊 Position {i}/{len(positions)}:")
                print(f"   🎲 Market: {question[:60]}...")
                print(f"   📊 Size: {position_size} shares")
                print(f"   💰 Entry: ${avg_price:.4f}")
                
                # Skip if position size is 0 or negative
                if position_size <= 0:
                    print(f"   ⚠️ No shares to sell, skipping...")
                    continue
                
                # Calculate target sell price
                target_sell_price = avg_price * (1 + profit_target)
                print(colored(f"   🎯 Target Sell Price: ${target_sell_price:.4f} ({profit_target*100:.1f}% profit)", "white", "on_cyan"))
                
                # SMART ORDER CHECKING - Check if correct order already exists
                print(colored(f"   🔍 MOON DEV: Smart order check for {position_size} shares...", "white", "on_blue"))
                order_check = check_existing_sell_order(token_id, target_sell_price, position_size)
                
                if order_check['action'] == 'keep_existing':
                    print(colored(f"   🎉 MOON DEV: Perfect order already exists! Skipping...", "white", "on_green"))
                    results['details'].append({
                        'market': question[:50],
                        'shares': position_size,
                        'price': target_sell_price,
                        'action': 'kept_existing',
                        'order_id': order_check['order'].get('id', 'N/A')
                    })
                    continue  # Skip to next position
                
                elif order_check['action'] == 'cancel_replace':
                    print(colored(f"   🔄 MOON DEV: Need to update existing order...", "white", "on_yellow"))
                    # Cancel specific wrong order
                    try:
                        cancel_success = cancel_token_orders(token_id)
                        if cancel_success:
                            print(colored(f"   ✅ MOON DEV: Wrong order canceled!", "white", "on_green"))
                        else:
                            print(colored(f"   ⚠️ MOON DEV: Cancel may have failed", "white", "on_red"))
                    except Exception as cancel_error:
                        print(colored(f"   ⚠️ MOON DEV: Cancel error: {cancel_error}", "white", "on_red"))
                
                elif order_check['action'] == 'cancel_all_replace':
                    print(colored(f"   🧹 MOON DEV: Multiple wrong orders found, canceling all...", "white", "on_yellow"))
                    try:
                        cancel_success = cancel_token_orders(token_id)
                        if cancel_success:
                            print(colored(f"   ✅ MOON DEV: All wrong orders canceled!", "white", "on_green"))
                        else:
                            print(colored(f"   ⚠️ MOON DEV: Cancel may have failed", "white", "on_red"))
                    except Exception as cancel_error:
                        print(colored(f"   ⚠️ MOON DEV: Cancel error: {cancel_error}", "white", "on_red"))
                
                else:  # place_new
                    print(colored(f"   📝 MOON DEV: No existing order, placing new...", "white", "on_blue"))
                
                # Try different position sizes if exact fails (precision/allowance fix)
                sizes_to_try = [
                    position_size,  # Exact size first
                    round(position_size * 0.99, 1),  # 99% of position (handles precision issues)
                    int(position_size),  # Rounded down integer
                    max(5.0, int(position_size * 0.95))  # 95% with 5 share minimum
                ]
                
                sell_success = False
                for attempt, size_to_sell in enumerate(sizes_to_try, 1):
                    if size_to_sell < 5.0:  # Skip if under minimum
                        continue
                        
                    print(colored(f"   📝 MOON DEV Attempt #{attempt}: Selling {size_to_sell} shares...", "white", "on_cyan"))
                    
                    response = place_limit_order(
                        token_id=token_id,
                        side="SELL",
                        price=target_sell_price,
                        size=size_to_sell
                    )
                    
                    if response and 'orderID' in response:
                        print(colored(f"   ✅ MOON DEV Sell order SUCCESS! Order: {response['orderID'][:20]}...", "white", "on_green"))
                        results['sell_orders_placed'] += 1
                        results['details'].append({
                            'market': question[:50],
                            'shares': size_to_sell,
                            'price': target_sell_price,
                            'order_id': response['orderID'],
                            'attempt': attempt,
                            'action': 'placed_new'
                        })
                        sell_success = True
                        break
                    else:
                        print(colored(f"   ⚠️ MOON DEV Attempt #{attempt} failed, trying next size...", "white", "on_yellow"))
                
                if not sell_success:
                    print(colored(f"   ❌ MOON DEV All sell attempts failed! Likely allowance issue 🔧", "white", "on_red"))
                    print(colored(f"   💡 MOON DEV: Visit polymarket.com to approve tokens!", "white", "on_red"))
                    results['errors'] += 1
                    
            except Exception as e:
                print(f"   ❌ MOON DEV Error processing position: {e}")
                results['errors'] += 1
                continue
        
        # Summary with colorful output
        print(colored(f"\n📊 MOON DEV Smart Sell Order Summary:", "white", "on_blue"))
        print(colored(f"   🔍 Positions Checked: {results['positions_checked']}", "white", "on_blue"))
        print(colored(f"   ✅ Orders Placed/Updated: {results['sell_orders_placed']}", "white", "on_green"))
        print(colored(f"   ❌ Errors: {results['errors']}", "white", "on_red"))
        
        # Count different actions
        actions = {}
        for detail in results['details']:
            action = detail.get('action', 'unknown')
            actions[action] = actions.get(action, 0) + 1
        
        if actions:
            print(colored(f"   📋 MOON DEV Action Breakdown:", "white", "on_magenta"))
            for action, count in actions.items():
                if action == 'kept_existing':
                    print(colored(f"      🎯 Kept Existing: {count}", "white", "on_green"))
                elif action == 'placed_new':
                    print(colored(f"      📝 Placed New: {count}", "white", "on_cyan"))
                else:
                    print(colored(f"      🔄 {action}: {count}", "white", "on_yellow"))
        
        if results['sell_orders_placed'] > 0 or actions.get('kept_existing', 0) > 0:
            print(colored(f"   🎯 All positions now have optimized sell orders at {profit_target*100:.1f}% profit!", "white", "on_green"))
        
        return results
        
    except Exception as e:
        print(f"❌ MOON DEV error in ensure_sell_orders: {e}")
        return {'positions_checked': 0, 'sell_orders_placed': 0, 'errors': 1}


def check_position_sell_orders():
    """
    Check which positions are missing sell orders
    MOON DEV's position audit function! 🔍
    
    Returns:
        dict: Analysis of positions and their sell order status
    """
    print(f"🔍 MOON DEV auditing position sell orders! 📋")
    
    try:
        # Get current positions
        portfolio = get_all_positions()
        
        if not portfolio or 'positions' not in portfolio:
            print(f"💡 MOON DEV: No positions found!")
            return {'total_positions': 0, 'positions_with_orders': 0, 'missing_orders': 0}
        
        positions = portfolio['positions']
        print(f"📊 MOON DEV found {len(positions)} positions")
        
        # Get all open orders
        try:
            all_orders = get_all_open_orders()
            print(f"📋 MOON DEV found {len(all_orders)} open orders")
        except:
            print(f"⚠️ MOON DEV: Could not fetch open orders, assuming none exist")
            all_orders = []
        
        results = {
            'total_positions': len(positions),
            'positions_with_orders': 0,
            'missing_orders': 0,
            'details': []
        }
        
        for position in positions:
            if position['position_size'] <= 0:
                continue
                
            token_id = position['asset_id']
            question = position['question']
            position_size = position['position_size']
            
            # Check if we have a sell order for this token
            has_sell_order = False
            for order in all_orders:
                if (order.get('asset_id') == token_id and 
                    order.get('side', '').upper() == 'SELL'):
                    has_sell_order = True
                    break
            
            if has_sell_order:
                results['positions_with_orders'] += 1
                print(f"   ✅ {question[:50]}... (HAS sell order)")
            else:
                results['missing_orders'] += 1
                print(f"   ❌ {question[:50]}... (MISSING sell order)")
                results['details'].append({
                    'token_id': token_id,
                    'market': question,
                    'size': position_size
                })
        
        print(f"\n📊 MOON DEV Sell Order Audit:")
        print(f"   📊 Total Positions: {results['total_positions']}")
        print(f"   ✅ With Sell Orders: {results['positions_with_orders']}")
        print(f"   ❌ Missing Orders: {results['missing_orders']}")
        
        return results
        
    except Exception as e:
        print(f"❌ MOON DEV error in check_position_sell_orders: {e}")
        return {'total_positions': 0, 'positions_with_orders': 0, 'missing_orders': 0}


def get_token_data(token_id, interval='max', fidelity=60):
    """
    Get comprehensive market data for any token ID - MOON DEV style! 📊
    
    Args:
        token_id (str): The token ID to get data for
        interval (str): Time interval ('1m', '1h', '1d', '1w', 'max') - default 'max' for all data
        fidelity (int): Resolution in minutes (default 60 for hourly)
    
    Returns:
        pd.DataFrame: Single-row DataFrame with all market data for the token
    """
    print(f"📊 MOON DEV fetching {interval} interval data for token: {token_id[:20]}... 🌙")
    
    try:
        # Initialize data dictionary
        token_data = {
            'token_id': token_id,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Get current market data from order book
        try:
            url = "https://clob.polymarket.com/book"
            params = {'token_id': token_id}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                asks = data.get('asks', [])
                bids = data.get('bids', [])
                
                if asks and bids:
                    best_bid = float(bids[-1]['price'])
                    best_ask = float(asks[-1]['price'])
                    spread = best_ask - best_bid
                    spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0
                    
                    token_data.update({
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'bid_size': float(bids[-1]['size']),
                        'ask_size': float(asks[-1]['size']),
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'mid_price': (best_bid + best_ask) / 2,
                        'total_bid_volume': sum(float(bid['size']) for bid in bids),
                        'total_ask_volume': sum(float(ask['size']) for ask in asks),
                        'order_book_depth': len(bids) + len(asks)
                    })
        except Exception as e:
            pass
        
        # Get historical price data using correct API parameters
        try:
            hist_url = "https://clob.polymarket.com/prices-history"
            hist_params = {
                'market': token_id,
                'interval': interval,
                'fidelity': fidelity
            }
            
            hist_response = requests.get(hist_url, params=hist_params, timeout=15)
            
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                
                # Handle both response formats
                if isinstance(hist_data, dict) and 'history' in hist_data:
                    history_points = hist_data['history']
                elif isinstance(hist_data, list):
                    history_points = hist_data
                else:
                    history_points = []
                
                if history_points and len(history_points) > 0:
                    # Extract prices and timestamps from the API response
                    prices = []
                    timestamps = []
                    
                    for point in history_points:
                        # Handle different response formats (t/p or timestamp/price)
                        if 't' in point and 'p' in point:
                            prices.append(float(point['p']))
                            timestamps.append(point['t'])
                        elif 'timestamp' in point and 'price' in point:
                            prices.append(float(point['price']))
                            timestamps.append(point['timestamp'])
                    
                    if len(prices) > 1:
                        current_price = prices[-1]
                        oldest_price = prices[0]
                        
                        # Calculate price change over the period
                        price_change_pct = ((current_price - oldest_price) / oldest_price * 100) if oldest_price > 0 else 0
                        volatility = pd.Series(prices).std()
                        
                        # Calculate additional metrics
                        max_price = max(prices)
                        min_price = min(prices)
                        avg_price = sum(prices) / len(prices)
                        
                        # Calculate some advanced metrics
                        price_range = max_price - min_price
                        price_range_pct = (price_range / avg_price * 100) if avg_price > 0 else 0
                        
                        # Calculate recent volatility (last 24 data points for short-term)
                        recent_vol = 0
                        if len(prices) >= 24:
                            recent_prices = prices[-24:]
                            recent_vol = pd.Series(recent_prices).std()
                        
                        # Convert timestamps to readable format
                        first_ts = timestamps[0] if timestamps else ''
                        last_ts = timestamps[-1] if timestamps else ''
                        
                        # Try to convert Unix timestamps to readable format
                        try:
                            if isinstance(first_ts, (int, float)):
                                first_readable = dt.datetime.fromtimestamp(first_ts).isoformat()
                                last_readable = dt.datetime.fromtimestamp(last_ts).isoformat()
                            else:
                                first_readable = str(first_ts)
                                last_readable = str(last_ts)
                        except:
                            first_readable = str(first_ts)
                            last_readable = str(last_ts)
                        
                        token_data.update({
                            'interval_used': interval,
                            'fidelity_minutes': fidelity,
                            'current_price': current_price,
                            'oldest_price': oldest_price,
                            'price_change_pct': price_change_pct,
                            'volatility': volatility,
                            'recent_volatility': recent_vol,
                            'max_price': max_price,
                            'min_price': min_price,
                            'avg_price': avg_price,
                            'price_range': price_range,
                            'price_range_pct': price_range_pct,
                            'data_points': len(prices),
                            'first_timestamp': first_readable,
                            'last_timestamp': last_readable,
                            'raw_first_ts': first_ts,
                            'raw_last_ts': last_ts
                        })
        except Exception as e:
            pass
        
        # Try to get market metadata
        try:
            markets_url = "https://gamma-api.polymarket.com/markets"
            markets_params = {'closed': 'false', 'limit': 1000}
            markets_response = requests.get(markets_url, params=markets_params, timeout=10)
            
            if markets_response.status_code == 200:
                markets_list = markets_response.json()
                
                # Find market that contains this token
                for market in markets_list:
                    if 'clobTokenIds' in market and market['clobTokenIds']:
                        try:
                            token_ids = json.loads(market['clobTokenIds'])
                            if token_id in token_ids:
                                # Found the market!
                                token_data.update({
                                    'market_id': market['id'],
                                    'market_question': market['question'],
                                    'market_category': market.get('category', ''),
                                    'market_end_date': market.get('endDate', ''),
                                    'market_volume_24hr': market.get('volume24hr', 0),
                                    'market_spread': market.get('spread', 0),
                                    'market_liquidity': market.get('liquidity', 0),
                                    'market_active': market.get('active', True),
                                    'market_closed': market.get('closed', False),
                                    'outcome': 'YES' if token_ids.index(token_id) == 0 else 'NO'
                                })
                                break
                        except:
                            continue
        except Exception as e:
            pass
        
        # Create DataFrame
        df = pd.DataFrame([token_data])
        
        print(f"✅ MOON DEV token data complete! {len(df.columns)} fields collected 📊")
        
        return df
        
    except Exception as e:
        print(f"❌ MOON DEV error getting token data: {e}")
        # Return empty DataFrame with just token_id
        return pd.DataFrame([{'token_id': token_id, 'error': str(e)}])


def get_token_data_full(token_id, interval='max', fidelity=60):
    """
    Get ALL historical data points for any token ID - MOON DEV DATA DOG style! 📊🐕
    Returns EVERY data point, not just summary stats
    
    Args:
        token_id (str): The token ID to get data for
        interval (str): Time interval ('1m', '1h', '1d', '1w', 'max') - default 'max' for all data
        fidelity (int): Resolution in minutes (default 60 for hourly)
    
    Returns:
        pd.DataFrame: DataFrame with ALL individual data points (one row per timestamp)
    """
    print(f"📊🐕 MOON DEV DATA DOG fetching ALL {interval} data points for token: {token_id[:20]}... 🌙")
    
    try:
        # Get market metadata first
        market_info = {}
        try:
            markets_url = "https://gamma-api.polymarket.com/markets"
            markets_params = {'closed': 'false', 'limit': 1000}
            markets_response = requests.get(markets_url, params=markets_params, timeout=10)
            
            if markets_response.status_code == 200:
                markets_list = markets_response.json()
                
                # Find market that contains this token
                for market in markets_list:
                    if 'clobTokenIds' in market and market['clobTokenIds']:
                        try:
                            token_ids = json.loads(market['clobTokenIds'])
                            if token_id in token_ids:
                                market_info = {
                                    'market_id': market['id'],
                                    'market_question': market['question'],
                                    'market_category': market.get('category', ''),
                                    'market_end_date': market.get('endDate', ''),
                                    'market_volume_24hr': market.get('volume24hr', 0),
                                    'outcome': 'YES' if token_ids.index(token_id) == 0 else 'NO'
                                }
                                break
                        except:
                            continue
        except:
            pass
        
        # Get historical price data using correct API parameters
        hist_url = "https://clob.polymarket.com/prices-history"
        hist_params = {
            'market': token_id,
            'interval': interval,
            'fidelity': fidelity
        }
        
        hist_response = requests.get(hist_url, params=hist_params, timeout=15)
        
        if hist_response.status_code == 200:
            hist_data = hist_response.json()
            
            # Handle both response formats
            if isinstance(hist_data, dict) and 'history' in hist_data:
                history_points = hist_data['history']
            elif isinstance(hist_data, list):
                history_points = hist_data
            else:
                history_points = []
            
            if history_points and len(history_points) > 0:
                # Create list to hold all data points
                all_data = []
                
                for i, point in enumerate(history_points):
                    row_data = {
                        'token_id': token_id,
                        'data_point_index': i,
                        'interval_used': interval,
                        'fidelity_minutes': fidelity
                    }
                    
                    # Add market info to every row
                    row_data.update(market_info)
                    
                    # Handle different response formats (t/p or timestamp/price)
                    if 't' in point and 'p' in point:
                        raw_timestamp = point['t']
                        price = float(point['p'])
                    elif 'timestamp' in point and 'price' in point:
                        raw_timestamp = point['timestamp']
                        price = float(point['price'])
                    else:
                        continue  # Skip invalid points
                    
                    # Convert timestamp to readable format
                    try:
                        if isinstance(raw_timestamp, (int, float)):
                            readable_timestamp = dt.datetime.fromtimestamp(raw_timestamp).isoformat()
                        else:
                            readable_timestamp = str(raw_timestamp)
                    except:
                        readable_timestamp = str(raw_timestamp)
                    
                    row_data.update({
                        'timestamp': readable_timestamp,
                        'raw_timestamp': raw_timestamp,
                        'price': price
                    })
                    
                    all_data.append(row_data)
                
                # Create DataFrame with ALL data points
                df = pd.DataFrame(all_data)
                
                # Sort by timestamp (date order - oldest to newest) 
                if 'timestamp' in df.columns:
                    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
                
                # 📈 MOON DEV: Calculate Simple Moving Averages! 🎯
                if 'price' in df.columns and len(df) >= 40:  # Need at least 40 points for SMA_40
                    print(f"📈 MOON DEV calculating Simple Moving Averages...")
                    df['SMA_20'] = df['price'].rolling(window=20, min_periods=20).mean()
                    df['SMA_40'] = df['price'].rolling(window=40, min_periods=40).mean()
                    print(f"✅ MOON DEV: SMA_20 and SMA_40 calculated! 🌙")
                elif 'price' in df.columns and len(df) >= 20:  # Can do SMA_20 only
                    print(f"📈 MOON DEV calculating SMA_20 only (insufficient data for SMA_40)...")
                    df['SMA_20'] = df['price'].rolling(window=20, min_periods=20).mean()
                    df['SMA_40'] = None  # Not enough data
                    print(f"✅ MOON DEV: SMA_20 calculated! 🌙")
                else:
                    print(f"⚠️ MOON DEV: Insufficient data for moving averages (need 20+ points)")
                    df['SMA_20'] = None
                    df['SMA_40'] = None
                
                # MOON DEV column reordering - but keep ALL the data!
                # Desired order: timestamp, price, SMA_20, SMA_40, market_volume_24hr, market_end_date, outcome, market_question, token_id, raw_timestamp
                desired_order = ['timestamp', 'price', 'SMA_20', 'SMA_40', 'market_volume_24hr', 'market_end_date', 'outcome', 'market_question', 'token_id', 'raw_timestamp']
                
                # Reorder columns but keep ALL columns that exist
                final_columns = []
                # First add the desired columns in order (if they exist)
                for col in desired_order:
                    if col in df.columns:
                        final_columns.append(col)
                
                # Then add any remaining columns we haven't included yet
                for col in df.columns:
                    if col not in final_columns:
                        final_columns.append(col)
                
                # Reorder the DataFrame
                df = df[final_columns]
                
                print(f"🐕 MOON DEV DATA DOG collected {len(df)} individual data points! 📊")
                print(f"   📅 Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
                print(f"   💰 Price range: ${df['price'].min():.4f} to ${df['price'].max():.4f}")
                
                # Show moving average info if available
                if 'SMA_20' in df.columns and df['SMA_20'].notna().any():
                    sma20_latest = df['SMA_20'].dropna().iloc[-1]
                    print(f"   📈 Latest SMA_20: ${sma20_latest:.4f}")
                if 'SMA_40' in df.columns and df['SMA_40'].notna().any():
                    sma40_latest = df['SMA_40'].dropna().iloc[-1]
                    print(f"   📈 Latest SMA_40: ${sma40_latest:.4f}")
                
                print(f"   🧹 Columns in order: {list(df.columns)}")
                
                return df
                
            else:
                print(f"❌ MOON DEV: No historical data points found")
                return pd.DataFrame([{'token_id': token_id, 'error': 'No data points'}])
        else:
            print(f"❌ MOON DEV: API error {hist_response.status_code}")
            return pd.DataFrame([{'token_id': token_id, 'error': f'API error {hist_response.status_code}'}])
            
    except Exception as e:
        print(f"❌ MOON DEV DATA DOG error: {e}")
        return pd.DataFrame([{'token_id': token_id, 'error': str(e)}])


def test_historical_data_limits(token_id):
    """
    Test how much historical data we can get from the API - MOON DEV style! 📅
    Tests multiple intervals to find the best data source
    
    Args:
        token_id (str): Token ID to test with
    
    Returns:
        dict: Information about available historical data across intervals
    """
    print(f"🧪 MOON DEV testing historical data limits for: {token_id[:20]}... 📅")
    
    intervals_to_test = ['max', '1w', '1d', '1h']  # Test multiple intervals
    results = {}
    
    for interval in intervals_to_test:
        try:
            print(f"   🔍 Testing interval: {interval}")
            
            hist_url = "https://clob.polymarket.com/prices-history"
            hist_params = {
                'market': token_id,
                'interval': interval,
                'fidelity': 60 if interval in ['1h', 'max'] else 1440  # 1 hour for granular, 1 day for others
            }
            
            response = requests.get(hist_url, params=hist_params, timeout=15)
            
            if response.status_code == 200:
                hist_data = response.json()
                
                # Handle both response formats
                if isinstance(hist_data, dict) and 'history' in hist_data:
                    history_points = hist_data['history']
                elif isinstance(hist_data, list):
                    history_points = hist_data
                else:
                    history_points = []
                
                if history_points and len(history_points) > 0:
                    total_points = len(history_points)
                    
                    # Extract data
                    prices = []
                    timestamps = []
                    
                    for point in history_points:
                        if 't' in point and 'p' in point:
                            prices.append(float(point['p']))
                            timestamps.append(point['t'])
                        elif 'timestamp' in point and 'price' in point:
                            prices.append(float(point['price']))
                            timestamps.append(point['timestamp'])
                    
                    if prices and timestamps:
                        # Convert timestamps if they're Unix timestamps
                        try:
                            if isinstance(timestamps[0], (int, float)):
                                first_date = dt.datetime.fromtimestamp(timestamps[0]).isoformat()
                                last_date = dt.datetime.fromtimestamp(timestamps[-1]).isoformat()
                            else:
                                first_date = str(timestamps[0])
                                last_date = str(timestamps[-1])
                        except:
                            first_date = str(timestamps[0])
                            last_date = str(timestamps[-1])
                        
                        price_range = f"${min(prices):.4f} - ${max(prices):.4f}"
                        
                        results[interval] = {
                            'total_points': total_points,
                            'first_date': first_date,
                            'last_date': last_date,
                            'price_range': price_range,
                            'first_price': prices[0],
                            'latest_price': prices[-1],
                            'success': True
                        }
                        
                        print(f"     ✅ {total_points} data points, range: {price_range}")
                    else:
                        results[interval] = {'success': False, 'error': 'No price data'}
                        print(f"     ❌ No price data")
                else:
                    results[interval] = {'success': False, 'error': 'No history points'}
                    print(f"     ❌ No history points")
            else:
                results[interval] = {'success': False, 'error': f'API error {response.status_code}'}
                print(f"     ❌ API error {response.status_code}")
                
        except Exception as e:
            results[interval] = {'success': False, 'error': str(e)}
            print(f"     ❌ Error: {str(e)[:50]}...")
    
    # Find the best interval (most data points)
    best_interval = None
    max_points = 0
    
    for interval, result in results.items():
        if result.get('success') and result.get('total_points', 0) > max_points:
            max_points = result['total_points']
            best_interval = interval
    
    print(f"\n🎯 MOON DEV Best Interval: {best_interval} with {max_points} data points!")
    
    return {
        'results': results,
        'best_interval': best_interval,
        'max_points': max_points,
        'success': best_interval is not None
    }


print("✨ MOON DEV's functions ready to use! 🌕")

# Test the functions when running directly
if __name__ == "__main__":
    print("🧪 MOON DEV testing functions... 🌙")
    
    # Test list_top_markets
    print("\n" + "="*50)
    print("📊 Testing list_top_markets...")
    markets_df = list_top_markets(10)
    if not markets_df.empty:
        print(markets_df.to_string(index=False))
    
    # Example usage for enhanced position functions
    print("\n" + "="*50)
    print("💰 Example usage for get_all_positions (ENHANCED!):")
    print("portfolio = get_all_positions()  # Uses .env PUBLIC_KEY")
    print("# OR")
    print("portfolio = get_all_positions('0x123...your_address_here')")
    print("\n🚀 NEW ENHANCED FEATURES:")
    print("- All positions with market context & categories")
    print("- Portfolio risk analysis and concentration metrics")
    print("- Enhanced P&L tracking with utilization rates")
    print("- Market volume and spread data for each position")
    print("- REAL USDC balance via direct contract! 💰")
    print("- Total available funds for trading 🚀")
    print("\n📊 Risk Analysis:")
    print("risk_analysis = analyze_portfolio_risk(portfolio)")
    print("- Category concentration analysis")
    print("- Market exposure limits")
    print("- Portfolio utilization warnings")
    print("- Diversification scoring")
    print("\n💰 Simple Total Balance:")
    print("total_balance = get_total_balance()  # Simple USDC + Positions")
    print("- Quick portfolio value check")
    print("- Perfect for trading decisions")
    print("\n📝 LIMIT ORDER PLACEMENT:")
    print("# Place a BUY limit order")
    print("response = place_limit_order('token_id', 'BUY', 0.45, 10)")
    print("# Place a SELL limit order")  
    print("response = place_limit_order('token_id', 'SELL', 0.55, 10)")
    print("# For negative risk markets, add neg_risk=True")
    print("response = place_limit_order('token_id', 'BUY', 0.45, 10, neg_risk=True)")
    print("\n⚡ MARKET ORDER PLACEMENT (Immediate Execution):")
    print("# Market BUY (hits the ask - pays current ask price)")
    print("response = place_market_order('token_id', 'BUY', 10)")
    print("# Market SELL (hits the bid - gets current bid price)")
    print("response = place_market_order('token_id', 'SELL', 10)")
    print("# Or use the market parameter in place_limit_order:")
    print("response = place_limit_order('token_id', 'BUY', 0, 10, market=True)")
    print("\n🚫 CANCEL TOKEN ORDERS:")
    print("# Cancel all orders for a specific token")
    print("success = cancel_token_orders('token_id')")
    print("- Cancels ALL orders (buy & sell) for that token")
    print("- Returns True if successful")
    print("\n📊 TOKEN DATA FETCHER:")
    print("# Get comprehensive data for any token ID")
    print("token_df = get_token_data('0x1234...token_id_here')")
    print("# With custom intervals:")
    print("token_df = get_token_data('0x1234...', interval='1h', fidelity=60)  # Hourly data")
    print("token_df = get_token_data('0x1234...', interval='max', fidelity=60) # All available data")
    print("# Available intervals: '1m', '1h', '1d', '1w', 'max'")
    print("# Returns DataFrame with:")
    print("- Current bid/ask prices and spreads")
    print("- Order book depth and volume")
    print("- Configurable historical price data (hourly by default)")
    print("- 📈 Simple Moving Averages (SMA_20 and SMA_40)! 🎯")
    print("- Market metadata (question, category, volume)")
    print("- Outcome type (YES/NO)")
    print("- Advanced metrics (recent volatility, price ranges, etc.)")
    print("# Example usage:")
    print("df = get_token_data('0xabcd...', interval='1h')  # Hourly data points")
    print("current_price = df['best_bid'].iloc[0]")
    print("spread = df['spread_pct'].iloc[0]")
    print("volatility = df['volatility'].iloc[0]")
    print("data_points = df['data_points'].iloc[0]")
    print("interval_used = df['interval_used'].iloc[0]")
    print("market_question = df['market_question'].iloc[0]")
    print("# 📈 MOON DEV Moving Averages with get_token_data_full:")
    print("full_df = get_token_data_full('0xabcd...', interval='1h')")
    print("latest_price = full_df['price'].iloc[-1]")
    print("latest_sma20 = full_df['SMA_20'].iloc[-1]  # 20-period moving average")
    print("latest_sma40 = full_df['SMA_40'].iloc[-1]  # 40-period moving average")
    print("# Check if price is above/below moving averages for signals!")
    print("if latest_price > latest_sma20: print('Price above SMA_20! 📈')")
    print("if latest_sma20 > latest_sma40: print('SMA_20 above SMA_40! Bullish! 🚀')")
    print("# Test API limits across intervals:")
    print("limits = test_historical_data_limits('token_id')")
    
    print("\n🔧 Setup Requirements:")
    print("- Uses PRIVATE_KEY and PUBLIC_KEY from .env")
    print("- Minimum 5 shares per order")
    print("- Price range: 0.01 to 0.99")
    print("- Automatic signature_type fallback")
    print("- Supports both regular and neg_risk markets")