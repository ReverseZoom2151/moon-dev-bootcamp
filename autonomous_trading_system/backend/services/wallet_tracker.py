"""
Wallet Tracker Service
Based on Day_46_Projects wallet following and copy trading functionality
"""

import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WalletTransaction:
    """Represents a wallet transaction"""
    wallet_address: str
    token_address: str
    transaction_type: str  # 'buy' or 'sell'
    amount_usd: float
    token_amount: float
    price: float
    timestamp: datetime
    transaction_hash: str
    profit_loss: Optional[float] = None


@dataclass
class WalletPerformance:
    """Represents wallet performance metrics"""
    wallet_address: str
    total_trades: int
    win_rate: float
    total_pnl: float
    avg_trade_size: float
    avg_hold_time: float
    sharpe_ratio: float
    max_drawdown: float
    last_active: datetime
    tracked_since: datetime


class WalletTracker:
    """
    Tracks successful traders and analyzes their trading patterns
    """
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.get('BIRDEYE_KEY')
        self.base_url = "https://public-api.birdeye.so/defi"
        
        # Tracking configuration
        self.min_trade_size_usd = config.get('MIN_TRADE_SIZE_USD', 100)
        self.min_win_rate = config.get('MIN_WIN_RATE', 0.6)
        self.min_total_trades = config.get('MIN_TOTAL_TRADES', 10)
        self.max_tracked_wallets = config.get('MAX_TRACKED_WALLETS', 100)
        
        # Data storage
        self.tracked_wallets: Set[str] = set()
        self.wallet_transactions: Dict[str, List[WalletTransaction]] = {}
        self.wallet_performance: Dict[str, WalletPerformance] = {}
        self.copy_trading_enabled = config.get('COPY_TRADING_ENABLED', False)
        
        # Copy trading settings
        self.copy_trade_percentage = config.get('COPY_TRADE_PERCENTAGE', 0.1)  # 10% of their trade size
        self.copy_trade_delay = config.get('COPY_TRADE_DELAY_SECONDS', 30)  # Delay to avoid front-running
        
        # Cache
        self.transaction_cache = {}
        self.cache_duration = timedelta(minutes=5)
    
    async def add_wallet_to_track(self, wallet_address: str, reason: str = "Manual"):
        """Add a wallet to the tracking list"""
        try:
            if wallet_address in self.tracked_wallets:
                logger.info(f"📊 Wallet {wallet_address[:8]}... already being tracked")
                return
            
            if len(self.tracked_wallets) >= self.max_tracked_wallets:
                # Remove least performing wallet
                await self._remove_worst_performing_wallet()
            
            self.tracked_wallets.add(wallet_address)
            self.wallet_transactions[wallet_address] = []
            
            # Initialize performance tracking
            self.wallet_performance[wallet_address] = WalletPerformance(
                wallet_address=wallet_address,
                total_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_trade_size=0.0,
                avg_hold_time=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                last_active=datetime.now(),
                tracked_since=datetime.now()
            )
            
            logger.info(f"🎯 Added wallet {wallet_address[:8]}... to tracking list (Reason: {reason})")
            
            # Fetch historical transactions
            await self._fetch_wallet_history(wallet_address)
            
        except Exception as e:
            logger.error(f"❌ Error adding wallet to track: {e}")
    
    async def remove_wallet_from_tracking(self, wallet_address: str):
        """Remove a wallet from tracking"""
        try:
            if wallet_address in self.tracked_wallets:
                self.tracked_wallets.remove(wallet_address)
                self.wallet_transactions.pop(wallet_address, None)
                self.wallet_performance.pop(wallet_address, None)
                logger.info(f"🗑️ Removed wallet {wallet_address[:8]}... from tracking")
        except Exception as e:
            logger.error(f"❌ Error removing wallet from tracking: {e}")
    
    async def discover_successful_wallets(self, token_address: str = None) -> List[str]:
        """Discover successful wallets from recent profitable trades"""
        try:
            discovered_wallets = []
            
            if token_address:
                # Find wallets that made profitable trades on specific token
                profitable_wallets = await self._find_profitable_wallets_for_token(token_address)
                discovered_wallets.extend(profitable_wallets)
            else:
                # Find generally successful wallets from trending tokens
                trending_tokens = await self._get_trending_tokens()
                for token in trending_tokens[:10]:  # Check top 10 trending
                    profitable_wallets = await self._find_profitable_wallets_for_token(token)
                    discovered_wallets.extend(profitable_wallets)
            
            # Filter and validate discovered wallets
            validated_wallets = []
            for wallet in set(discovered_wallets):
                if await self._validate_wallet_performance(wallet):
                    validated_wallets.append(wallet)
                    await self.add_wallet_to_track(wallet, "Auto-discovered")
            
            logger.info(f"🔍 Discovered {len(validated_wallets)} new successful wallets")
            return validated_wallets
            
        except Exception as e:
            logger.error(f"❌ Error discovering successful wallets: {e}")
            return []
    
    async def monitor_tracked_wallets(self):
        """Monitor all tracked wallets for new transactions"""
        try:
            logger.info(f"👀 Monitoring {len(self.tracked_wallets)} tracked wallets...")
            
            for wallet_address in list(self.tracked_wallets):
                try:
                    new_transactions = await self._fetch_recent_transactions(wallet_address)
                    
                    for transaction in new_transactions:
                        await self._process_new_transaction(transaction)
                        
                        # Copy trade if enabled
                        if self.copy_trading_enabled:
                            await self._consider_copy_trade(transaction)
                    
                    # Update performance metrics
                    await self._update_wallet_performance(wallet_address)
                    
                except Exception as e:
                    logger.error(f"❌ Error monitoring wallet {wallet_address[:8]}...: {e}")
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"❌ Error in wallet monitoring: {e}")
    
    async def _fetch_wallet_history(self, wallet_address: str, days_back: int = 30):
        """Fetch historical transactions for a wallet"""
        try:
            # This would integrate with Birdeye or other APIs to get transaction history
            # For now, implementing a placeholder structure
            
            url = f"{self.base_url}/wallet/token_list?wallet={wallet_address}"
            headers = {"X-API-KEY": self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                # Process historical data and create WalletTransaction objects
                # This is a simplified implementation
                logger.info(f"📈 Fetched history for wallet {wallet_address[:8]}...")
            
        except Exception as e:
            logger.error(f"❌ Error fetching wallet history: {e}")
    
    async def _fetch_recent_transactions(self, wallet_address: str) -> List[WalletTransaction]:
        """Fetch recent transactions for a wallet"""
        try:
            # Check cache first
            cache_key = f"recent_tx_{wallet_address}"
            if cache_key in self.transaction_cache:
                cached_data, timestamp = self.transaction_cache[cache_key]
                if datetime.now() - timestamp < self.cache_duration:
                    return cached_data
            
            # Fetch from API (placeholder implementation)
            transactions = []
            
            # Cache the result
            self.transaction_cache[cache_key] = (transactions, datetime.now())
            
            return transactions
            
        except Exception as e:
            logger.error(f"❌ Error fetching recent transactions: {e}")
            return []
    
    async def _find_profitable_wallets_for_token(self, token_address: str) -> List[str]:
        """Find wallets that made profitable trades on a specific token"""
        try:
            # This would analyze recent transactions for the token
            # and identify wallets with profitable trades
            profitable_wallets = []
            
            # Placeholder implementation
            # In reality, this would:
            # 1. Get recent transactions for the token
            # 2. Analyze buy/sell patterns
            # 3. Calculate profits for each wallet
            # 4. Return wallets with significant profits
            
            return profitable_wallets
            
        except Exception as e:
            logger.error(f"❌ Error finding profitable wallets: {e}")
            return []
    
    async def _get_trending_tokens(self) -> List[str]:
        """Get list of trending token addresses"""
        try:
            url = f"{self.base_url}/defi/trending_tokens"
            headers = {"X-API-KEY": self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                tokens = data.get('tokens', [])
                return [token.get('address') for token in tokens if token.get('address')]
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting trending tokens: {e}")
            return []
    
    async def _validate_wallet_performance(self, wallet_address: str) -> bool:
        """Validate if a wallet meets performance criteria"""
        try:
            # Analyze wallet's trading history to determine if it's worth tracking
            # This would check:
            # - Win rate
            # - Total number of trades
            # - Average trade size
            # - Consistency of profits
            
            # Placeholder validation
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating wallet performance: {e}")
            return False
    
    async def _process_new_transaction(self, transaction: WalletTransaction):
        """Process a new transaction from a tracked wallet"""
        try:
            wallet_address = transaction.wallet_address
            
            # Add to transaction history
            if wallet_address not in self.wallet_transactions:
                self.wallet_transactions[wallet_address] = []
            
            self.wallet_transactions[wallet_address].append(transaction)
            
            # Log the transaction
            action = "🟢 BUY" if transaction.transaction_type == 'buy' else "🔴 SELL"
            logger.info(
                f"{action} {wallet_address[:8]}... | "
                f"{transaction.token_address[:8]}... | "
                f"${transaction.amount_usd:,.2f}"
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing new transaction: {e}")
    
    async def _consider_copy_trade(self, transaction: WalletTransaction):
        """Consider copying a trade from a tracked wallet"""
        try:
            if not self.copy_trading_enabled:
                return
            
            wallet_performance = self.wallet_performance.get(transaction.wallet_address)
            if not wallet_performance:
                return
            
            # Only copy trades from high-performing wallets
            if (wallet_performance.win_rate < self.min_win_rate or 
                wallet_performance.total_trades < self.min_total_trades):
                return
            
            # Calculate copy trade size
            copy_amount = transaction.amount_usd * self.copy_trade_percentage
            
            # Add delay to avoid front-running
            await asyncio.sleep(self.copy_trade_delay)
            
            logger.info(
                f"🔄 COPY TRADE: {transaction.transaction_type.upper()} "
                f"{transaction.token_address[:8]}... | ${copy_amount:,.2f}"
            )
            
            # Here you would integrate with your portfolio manager to execute the trade
            # await self.portfolio_manager.copy_trade(transaction, copy_amount)
            
        except Exception as e:
            logger.error(f"❌ Error considering copy trade: {e}")
    
    async def _update_wallet_performance(self, wallet_address: str):
        """Update performance metrics for a wallet"""
        try:
            transactions = self.wallet_transactions.get(wallet_address, [])
            if not transactions:
                return
            
            # Calculate performance metrics
            total_trades = len(transactions)
            profitable_trades = sum(1 for tx in transactions if tx.profit_loss and tx.profit_loss > 0)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(tx.profit_loss for tx in transactions if tx.profit_loss)
            avg_trade_size = sum(tx.amount_usd for tx in transactions) / total_trades if total_trades > 0 else 0
            
            # Update performance object
            if wallet_address in self.wallet_performance:
                performance = self.wallet_performance[wallet_address]
                performance.total_trades = total_trades
                performance.win_rate = win_rate
                performance.total_pnl = total_pnl
                performance.avg_trade_size = avg_trade_size
                performance.last_active = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Error updating wallet performance: {e}")
    
    async def _remove_worst_performing_wallet(self):
        """Remove the worst performing wallet to make room for new ones"""
        try:
            if not self.wallet_performance:
                return
            
            # Find wallet with lowest performance score
            worst_wallet = min(
                self.wallet_performance.keys(),
                key=lambda w: self.wallet_performance[w].win_rate * self.wallet_performance[w].total_pnl
            )
            
            await self.remove_wallet_from_tracking(worst_wallet)
            logger.info(f"🗑️ Removed worst performing wallet: {worst_wallet[:8]}...")
            
        except Exception as e:
            logger.error(f"❌ Error removing worst performing wallet: {e}")
    
    async def get_top_performers(self, limit: int = 10) -> List[WalletPerformance]:
        """Get top performing tracked wallets"""
        try:
            performances = list(self.wallet_performance.values())
            
            # Sort by a combination of win rate and total PnL
            performances.sort(
                key=lambda p: p.win_rate * max(p.total_pnl, 0),
                reverse=True
            )
            
            return performances[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error getting top performers: {e}")
            return []
    
    async def get_tracking_stats(self) -> Dict:
        """Get wallet tracking statistics"""
        try:
            total_tracked = len(self.tracked_wallets)
            total_transactions = sum(len(txs) for txs in self.wallet_transactions.values())
            
            avg_win_rate = 0
            avg_pnl = 0
            if self.wallet_performance:
                avg_win_rate = sum(p.win_rate for p in self.wallet_performance.values()) / len(self.wallet_performance)
                avg_pnl = sum(p.total_pnl for p in self.wallet_performance.values()) / len(self.wallet_performance)
            
            return {
                'total_tracked_wallets': total_tracked,
                'total_transactions': total_transactions,
                'average_win_rate': avg_win_rate,
                'average_pnl': avg_pnl,
                'copy_trading_enabled': self.copy_trading_enabled,
                'cache_size': len(self.transaction_cache)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting tracking stats: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the transaction cache"""
        self.transaction_cache.clear()
        logger.info("🧹 Wallet tracker cache cleared") 