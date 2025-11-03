"""
Natural Language Command Parser
================================
Converts natural language queries to CLI commands for better compatibility.
"""

import re
from typing import Optional, Dict, Tuple


class NLCommandParser:
    """
    Parses natural language queries and converts them to CLI commands.
    
    This allows users to use natural language for all commands instead of
    requiring exact syntax.
    """
    
    # Command mappings: natural language patterns -> command prefixes
    COMMAND_PATTERNS = {
        # Day 48: Position Management
        'pnl_close': [
            r'close.*position.*pnl',
            r'pnl.*close',
            r'close.*profit.*loss',
            r'close.*based.*pnl',
            r'close.*take.*profit',
            r'close.*stop.*loss'
        ],
        'chunk_close': [
            r'close.*position.*chunk',
            r'chunk.*close',
            r'close.*in.*chunks',
            r'sell.*in.*chunks',
            r'close.*gradually'
        ],
        'easy_entry': [
            r'easy.*entry',
            r'average.*in',
            r'enter.*position',
            r'build.*position',
            r'accumulate.*position'
        ],
        'track_position': [
            r'track.*position',
            r'start.*tracking',
            r'watch.*position',
            r'monitor.*position'
        ],
        'check_pnl': [
            r'check.*pnl',
            r'pnl.*status',
            r'profit.*loss.*status',
            r'how.*much.*profit',
            r'current.*pnl'
        ],
        
        # Day 49: MA Reversal
        'ma_reversal': [
            r'ma.*reversal',
            r'2x.*ma',
            r'dual.*moving.*average',
            r'ma.*crossover.*reversal',
            r'reversal.*strategy',
            r'run.*ma.*reversal'
        ],
        
        # Day 50: Enhanced Supply/Demand
        'enhanced_sd': [
            r'enhanced.*supply.*demand',
            r'enhanced.*sd',
            r'supply.*demand.*zone',
            r'sd.*zone.*strategy',
            r'zone.*strategy'
        ],
        
        # ML Indicators
        'ml_evaluate': [
            r'evaluate.*indicator',
            r'ml.*evaluate',
            r'test.*indicator',
            r'analyze.*indicator'
        ],
        'ml_loop': [
            r'loop.*indicator',
            r'ml.*loop',
            r'generation.*indicator',
            r'evolve.*indicator'
        ],
        'ml_top': [
            r'top.*indicator',
            r'best.*indicator',
            r'rank.*indicator',
            r'ml.*top'
        ],
        'ml_discover': [
            r'discover.*indicator',
            r'list.*indicator',
            r'available.*indicator',
            r'ml.*discover'
        ],
        
        # RRS
        'rrs_analyze': [
            r'rrs.*analyze',
            r'relative.*rotation.*strength',
            r'rrs.*analysis',
            r'analyze.*rrs'
        ],
        'rrs_rankings': [
            r'rrs.*ranking',
            r'rank.*by.*rrs',
            r'rrs.*compare',
            r'rrs.*list'
        ],
        'rrs_signals': [
            r'rrs.*signal',
            r'rrs.*trading.*signal',
            r'rrs.*buy.*signal',
            r'rrs.*sell.*signal'
        ],
        
        # Trader Intelligence
        'trader_analyze': [
            r'analyze.*trader',
            r'early.*buyer',
            r'trader.*intelligence',
            r'smart.*money',
            r'find.*trader'
        ],
        'find_accounts': [
            r'find.*account',
            r'accounts.*to.*follow',
            r'who.*to.*follow',
            r'follow.*account'
        ],
        'institutional_traders': [
            r'institutional.*trader',
            r'large.*trader',
            r'pro.*trader',
            r'institution'
        ],
        
        # Whale Tracking
        'whale_track': [
            r'whale.*track',
            r'track.*whale',
            r'large.*position',
            r'whale.*position'
        ],
        'multi_whale_track': [
            r'multi.*whale',
            r'multiple.*whale',
            r'multi.*address.*track',
            r'track.*multiple.*whale'
        ],
        'liquidation_risk': [
            r'liquidation.*risk',
            r'risk.*liquidation',
            r'distance.*liquidation',
            r'closest.*liquidation'
        ],
        'aggregate_positions': [
            r'aggregate.*position',
            r'aggregated.*position',
            r'summary.*position',
            r'combined.*position'
        ],
        
        # Market Dashboard
        'market_dashboard': [
            r'market.*dashboard',
            r'market.*tracker',
            r'full.*market.*analysis',
            r'market.*overview'
        ],
        'trending_tokens': [
            r'trending.*token',
            r'trending.*coin',
            r'hot.*token',
            r'rising.*token'
        ],
        'new_listings': [
            r'new.*listing',
            r'new.*token',
            r'recent.*listing',
            r'newly.*listed'
        ],
        'volume_leaders': [
            r'volume.*leader',
            r'highest.*volume',
            r'top.*volume',
            r'volume.*ranking'
        ],
        'funding_rates': [
            r'funding.*rate',
            r'funding.*cost',
            r'funding.*fee'
        ],
        
        # Liquidation Hunter
        'liq_hunter': [
            r'liquidation.*hunter',
            r'hunt.*liquidation',
            r'liquidation.*analysis',
            r'liquidation.*cascade'
        ],
        'moondev_data': [
            r'moondev.*data',
            r'moon.*dev',
            r'liquidation.*data',
            r'funding.*data',
            r'open.*interest.*data'
        ],
        'orderbook_analyze': [
            r'order.*book.*analyze',
            r'analyze.*order.*book',
            r'orderbook.*depth',
            r'market.*depth'
        ],
        
        # Position Sizing
        'position_size': [
            r'position.*size',
            r'calculate.*size',
            r'how.*much.*buy',
            r'position.*amount',
            r'size.*position'
        ],
    }
    
    @staticmethod
    def parse(query: str) -> Optional[Tuple[str, Dict]]:
        """
        Parse natural language query and convert to command.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (command_prefix, parameters_dict) or None if no match
        """
        query_lower = query.lower()
        
        # Check each command pattern
        for command_prefix, patterns in NLCommandParser.COMMAND_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Extract parameters
                    params = NLCommandParser._extract_parameters(query, command_prefix)
                    return (command_prefix, params)
        
        return None
    
    @staticmethod
    def _extract_parameters(query: str, command: str) -> Dict:
        """Extract parameters from natural language query."""
        params = {}
        
        # Extract symbol (common pattern)
        symbol_match = re.search(r'\b([A-Z]{2,10}USDT?|[A-Z]{2,10}USD|[A-Z]{2,10}BTC)\b', query, re.IGNORECASE)
        if symbol_match:
            params['symbol'] = symbol_match.group(1).upper()
        
        # Extract numbers
        numbers = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', query)
        if numbers:
            params['numbers'] = [float(n.replace(',', '')) for n in numbers]
        
        # Extract percentages
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', query)
        if pct_match:
            params['percentage'] = float(pct_match.group(1))
        
        # Extract exchange names
        if 'binance' in query.lower():
            params['exchange'] = 'binance'
        elif 'bitfinex' in query.lower():
            params['exchange'] = 'bitfinex'
        
        # Extract action words
        if 'check' in query.lower() or 'status' in query.lower():
            params['action'] = 'check'
        elif 'execute' in query.lower() or 'run' in query.lower():
            params['action'] = 'execute'
        
        # Extract entry types for easy-entry
        if command == 'easy_entry':
            if 'market' in query.lower():
                params['entry_type'] = 'market'
            elif 'sd' in query.lower() or 'supply' in query.lower() or 'demand' in query.lower():
                params['entry_type'] = 'sd-zone'
            elif 'trend' in query.lower() or 'sma' in query.lower():
                params['entry_type'] = 'sma-trend'
        
        # Extract data types for moondev
        if command == 'moondev_data':
            if 'liquidation' in query.lower():
                params['data_type'] = 'liquidations'
            elif 'funding' in query.lower():
                params['data_type'] = 'funding'
            elif 'open.*interest' in query.lower() or 'oi' in query.lower():
                params['data_type'] = 'oi'
            elif 'position' in query.lower():
                params['data_type'] = 'positions'
            elif 'whale' in query.lower():
                params['data_type'] = 'whales'
        
        return params
    
    @staticmethod
    def convert_to_command(command_prefix: str, params: Dict) -> str:
        """
        Convert parsed command and parameters to CLI command string.
        
        Args:
            command_prefix: Command prefix (e.g., 'ma_reversal')
            params: Extracted parameters
            
        Returns:
            CLI command string
        """
        # Map command prefixes to CLI commands
        command_map = {
            'pnl_close': 'pnl-close',
            'chunk_close': 'chunk-close',
            'easy_entry': 'easy-entry',
            'track_position': 'track-position',
            'check_pnl': 'check-pnl',
            'ma_reversal': 'ma-reversal',
            'enhanced_sd': 'enhanced-sd',
            'ml_evaluate': 'ml-evaluate-indicators',
            'ml_loop': 'ml-loop-indicators',
            'ml_top': 'ml-top-indicators',
            'ml_discover': 'ml-discover-indicators',
            'rrs_analyze': 'rrs-analyze',
            'rrs_rankings': 'rrs-rankings',
            'rrs_signals': 'rrs-signals',
            'trader_analyze': 'trader-analyze',
            'find_accounts': 'find-accounts',
            'institutional_traders': 'institutional-traders',
            'whale_track': 'whale-track',
            'multi_whale_track': 'multi-whale-track',
            'liquidation_risk': 'liquidation-risk',
            'aggregate_positions': 'aggregate-positions',
            'market_dashboard': 'market-dashboard',
            'trending_tokens': 'trending-tokens',
            'new_listings': 'new-listings',
            'volume_leaders': 'volume-leaders',
            'funding_rates': 'funding-rates',
            'liq_hunter': 'liq-hunter',
            'moondev_data': 'moondev-data',
            'orderbook_analyze': 'orderbook-analyze',
            'position_size': 'position-size',
        }
        
        cli_command = command_map.get(command_prefix, command_prefix.replace('_', '-'))
        
        # Build command string
        parts = [cli_command]
        
        # Add symbol if present
        if 'symbol' in params:
            parts.append(params['symbol'])
        
        # Add action if present
        if 'action' in params and command_prefix == 'ma_reversal':
            parts.append(params['action'])
        
        # Add entry type for easy-entry
        if 'entry_type' in params:
            parts.append(params['entry_type'])
        
        # Add data type for moondev-data
        if 'data_type' in params:
            parts.append(params['data_type'])
        
        # Add exchange
        if 'exchange' in params:
            parts.append(params['exchange'])
        
        # Add numbers/percentages
        if 'numbers' in params:
            parts.extend([str(n) for n in params['numbers'][:3]])  # Limit to 3 numbers
        
        if 'percentage' in params:
            parts.append(str(params['percentage']))
        
        return ' '.join(parts)

