"""
Token Security Analyzer Service
Based on Day_50_Projects token security analysis functionality
"""

import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class TokenSecurityAnalyzer:
    """
    Analyzes token security metrics to detect potential rug pulls and unsafe tokens
    """
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.get('BIRDEYE_KEY')
        self.base_url = "https://public-api.birdeye.so/defi"
        
        # Security thresholds
        self.rug_pull_threshold = -80  # Price drop percentage indicating rug pull
        self.min_liquidity = config.get('MIN_LIQUIDITY_USD', 1000)
        self.max_top_holders_pct = config.get('MAX_TOP_10_HOLDERS_PCT', 0.7)
        self.min_unique_wallets = config.get('MIN_UNIQUE_WALLETS_24H', 30)
        
        # Do not trade list
        self.do_not_trade_list = config.get('DO_NOT_TRADE_LIST', [])
        
        # Cache for security data
        self.security_cache = {}
        self.cache_duration = timedelta(minutes=30)
    
    async def analyze_token_security(self, token_address: str) -> Dict:
        """
        Comprehensive token security analysis
        
        Returns:
            Dict with security metrics and risk assessment
        """
        try:
            # Check cache first
            cache_key = f"security_{token_address}"
            if cache_key in self.security_cache:
                cached_data, timestamp = self.security_cache[cache_key]
                if datetime.now() - timestamp < self.cache_duration:
                    return cached_data
            
            # Gather all security data
            overview_data = await self.get_token_overview(token_address)
            security_data = await self.get_token_security_info(token_address)
            creation_data = await self.get_token_creation_info(token_address)
            
            # Analyze security metrics
            security_analysis = {
                'token_address': token_address,
                'timestamp': datetime.now().isoformat(),
                'is_safe': True,
                'risk_score': 0.0,  # 0-100, higher is riskier
                'warnings': [],
                'metrics': {},
                'recommendation': 'SAFE'
            }
            
            # Analyze overview data
            if overview_data:
                security_analysis['metrics'].update(overview_data)
                await self._analyze_overview_metrics(security_analysis, overview_data)
            
            # Analyze security info
            if security_data:
                security_analysis['metrics'].update(security_data)
                await self._analyze_security_metrics(security_analysis, security_data)
            
            # Analyze creation info
            if creation_data:
                security_analysis['metrics'].update(creation_data)
                await self._analyze_creation_metrics(security_analysis, creation_data)
            
            # Check do not trade list
            if token_address in self.do_not_trade_list:
                security_analysis['is_safe'] = False
                security_analysis['risk_score'] = 100
                security_analysis['warnings'].append("Token is in DO_NOT_TRADE_LIST")
                security_analysis['recommendation'] = 'BLACKLISTED'
            
            # Final risk assessment
            await self._calculate_final_risk_score(security_analysis)
            
            # Cache the result
            self.security_cache[cache_key] = (security_analysis, datetime.now())
            
            return security_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing token security for {token_address}: {e}")
            return {
                'token_address': token_address,
                'is_safe': False,
                'risk_score': 100,
                'warnings': [f"Analysis failed: {str(e)}"],
                'recommendation': 'UNKNOWN'
            }
    
    async def get_token_overview(self, token_address: str) -> Optional[Dict]:
        """Get token overview data from Birdeye API"""
        try:
            url = f"{self.base_url}/token_overview?address={token_address}"
            headers = {"X-API-KEY": self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                
                # Calculate additional metrics
                buy1h = data.get('buy1h', 0)
                sell1h = data.get('sell1h', 0)
                trade1h = buy1h + sell1h
                
                processed_data = {
                    'buy1h': buy1h,
                    'sell1h': sell1h,
                    'trade1h': trade1h,
                    'buy_percentage': (buy1h / trade1h * 100) if trade1h else 0,
                    'sell_percentage': (sell1h / trade1h * 100) if trade1h else 0,
                    'uniqueWallet24h': data.get('uniqueWallet24h', 0),
                    'v24hUSD': data.get('v24hUSD', 0),
                    'liquidity': data.get('liquidity', 0),
                    'watch': data.get('watch', 0),
                    'view24h': data.get('view24h', 0)
                }
                
                # Extract price changes
                price_changes = {k: v for k, v in data.items() if 'priceChange' in k}
                processed_data['price_changes'] = price_changes
                
                return processed_data
                
            else:
                logger.warning(f"⚠️ Failed to get token overview: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting token overview: {e}")
            return None
    
    async def get_token_security_info(self, token_address: str) -> Optional[Dict]:
        """Get token security information from Birdeye API"""
        try:
            url = f"{self.base_url}/token_security?address={token_address}"
            headers = {"X-API-KEY": self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                return {
                    'freeze_authority': data.get('freezeAuthority'),
                    'mint_authority': data.get('mintAuthority'),
                    'creator_balance': data.get('creatorBalance', 0),
                    'creator_percentage': data.get('creatorPercentage', 0),
                    'top10_holder_balance': data.get('top10HolderBalance', 0),
                    'top10_holder_percent': data.get('top10HolderPercent', 0),
                    'total_supply': data.get('totalSupply', 0),
                    'is_token_2022': data.get('isToken2022', False),
                    'is_true_token': data.get('isTrueToken'),
                    'mutable_metadata': data.get('mutableMetadata', True)
                }
            else:
                logger.warning(f"⚠️ Failed to get token security info: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting token security info: {e}")
            return None
    
    async def get_token_creation_info(self, token_address: str) -> Optional[Dict]:
        """Get token creation information"""
        try:
            url = f"{self.base_url}/token_creation_info?address={token_address}"
            headers = {"X-API-KEY": self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', {})
                return {
                    'creation_time': data.get('creationTime'),
                    'creation_tx': data.get('creationTx'),
                    'creator_address': data.get('creatorAddress'),
                    'mint_time': data.get('mintTime'),
                    'mint_tx': data.get('mintTx')
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting token creation info: {e}")
            return None
    
    async def _analyze_overview_metrics(self, analysis: Dict, overview_data: Dict):
        """Analyze overview metrics for security risks"""
        risk_score = 0
        warnings = []
        
        # Check for rug pull indicators
        price_changes = overview_data.get('price_changes', {})
        for timeframe, change in price_changes.items():
            if change and change < self.rug_pull_threshold:
                risk_score += 50
                warnings.append(f"Potential rug pull: {change:.1f}% drop in {timeframe}")
                analysis['is_safe'] = False
        
        # Check liquidity
        liquidity = overview_data.get('liquidity', 0)
        if liquidity < self.min_liquidity:
            risk_score += 30
            warnings.append(f"Low liquidity: ${liquidity:,.2f} (min: ${self.min_liquidity:,.2f})")
        
        # Check unique wallets
        unique_wallets = overview_data.get('uniqueWallet24h', 0)
        if unique_wallets < self.min_unique_wallets:
            risk_score += 20
            warnings.append(f"Low unique wallets: {unique_wallets} (min: {self.min_unique_wallets})")
        
        # Check trading activity
        trade1h = overview_data.get('trade1h', 0)
        if trade1h == 0:
            risk_score += 25
            warnings.append("No trading activity in last hour")
        
        # Check buy/sell ratio
        sell_percentage = overview_data.get('sell_percentage', 0)
        if sell_percentage > 80:
            risk_score += 35
            warnings.append(f"High sell pressure: {sell_percentage:.1f}% sells")
        
        analysis['risk_score'] += risk_score
        analysis['warnings'].extend(warnings)
    
    async def _analyze_security_metrics(self, analysis: Dict, security_data: Dict):
        """Analyze security-specific metrics"""
        risk_score = 0
        warnings = []
        
        # Check freeze authority
        freeze_authority = security_data.get('freeze_authority')
        if freeze_authority is not None:
            risk_score += 40
            warnings.append("Token has freeze authority - can be frozen")
        
        # Check mint authority
        mint_authority = security_data.get('mint_authority')
        if mint_authority is not None:
            risk_score += 30
            warnings.append("Token has mint authority - supply can be inflated")
        
        # Check creator holdings
        creator_percentage = security_data.get('creator_percentage', 0)
        if creator_percentage > 50:
            risk_score += 35
            warnings.append(f"Creator holds {creator_percentage:.1f}% of supply")
        
        # Check top holder concentration
        top10_percent = security_data.get('top10_holder_percent', 0)
        if top10_percent > self.max_top_holders_pct:
            risk_score += 25
            warnings.append(f"Top 10 holders own {top10_percent:.1%} of supply")
        
        # Check if metadata is mutable
        mutable_metadata = security_data.get('mutable_metadata', True)
        if mutable_metadata:
            risk_score += 10
            warnings.append("Token metadata is mutable")
        
        analysis['risk_score'] += risk_score
        analysis['warnings'].extend(warnings)
    
    async def _analyze_creation_metrics(self, analysis: Dict, creation_data: Dict):
        """Analyze token creation metrics"""
        risk_score = 0
        warnings = []
        
        # Check token age
        creation_time = creation_data.get('creation_time')
        if creation_time:
            creation_date = datetime.fromtimestamp(creation_time)
            age_hours = (datetime.now() - creation_date).total_seconds() / 3600
            
            if age_hours < 24:
                risk_score += 20
                warnings.append(f"Very new token: {age_hours:.1f} hours old")
            elif age_hours < 168:  # 1 week
                risk_score += 10
                warnings.append(f"New token: {age_hours/24:.1f} days old")
        
        analysis['risk_score'] += risk_score
        analysis['warnings'].extend(warnings)
    
    async def _calculate_final_risk_score(self, analysis: Dict):
        """Calculate final risk score and recommendation"""
        risk_score = min(analysis['risk_score'], 100)  # Cap at 100
        analysis['risk_score'] = risk_score
        
        if risk_score >= 80:
            analysis['is_safe'] = False
            analysis['recommendation'] = 'AVOID'
        elif risk_score >= 60:
            analysis['is_safe'] = False
            analysis['recommendation'] = 'HIGH_RISK'
        elif risk_score >= 40:
            analysis['recommendation'] = 'MEDIUM_RISK'
        elif risk_score >= 20:
            analysis['recommendation'] = 'LOW_RISK'
        else:
            analysis['recommendation'] = 'SAFE'
    
    async def is_token_safe(self, token_address: str) -> bool:
        """Quick safety check for a token"""
        try:
            analysis = await self.analyze_token_security(token_address)
            return analysis.get('is_safe', False)
        except Exception as e:
            logger.error(f"❌ Error checking token safety: {e}")
            return False
    
    async def get_risk_score(self, token_address: str) -> float:
        """Get risk score for a token (0-100)"""
        try:
            analysis = await self.analyze_token_security(token_address)
            return analysis.get('risk_score', 100)
        except Exception as e:
            logger.error(f"❌ Error getting risk score: {e}")
            return 100
    
    async def batch_analyze_tokens(self, token_addresses: List[str]) -> Dict[str, Dict]:
        """Analyze multiple tokens in batch"""
        results = {}
        
        # Process in batches to avoid rate limiting
        batch_size = 5
        for i in range(0, len(token_addresses), batch_size):
            batch = token_addresses[i:i + batch_size]
            
            # Analyze batch concurrently
            tasks = [self.analyze_token_security(addr) for addr in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for addr, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error analyzing {addr}: {result}")
                    results[addr] = {
                        'is_safe': False,
                        'risk_score': 100,
                        'warnings': [f"Analysis failed: {str(result)}"]
                    }
                else:
                    results[addr] = result
            
            # Rate limiting delay
            await asyncio.sleep(1)
        
        return results
    
    def clear_cache(self):
        """Clear the security analysis cache"""
        self.security_cache.clear()
        logger.info("🧹 Token security cache cleared")
    
    async def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_tokens': len(self.security_cache),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60
        } 