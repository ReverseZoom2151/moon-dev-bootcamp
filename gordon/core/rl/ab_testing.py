"""
RL A/B Testing Framework
========================
Compares RL-enhanced trading decisions vs non-RL baseline decisions.
"""

import logging
import random
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result of a single A/B test comparison."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    trade_id: str = ""
    rl_decision: Dict[str, Any] = field(default_factory=dict)
    baseline_decision: Dict[str, Any] = field(default_factory=dict)
    actual_outcome: Optional[Dict[str, Any]] = None  # Filled after trade completes
    rl_used: bool = False


class RLABTesting:
    """A/B testing framework for RL vs baseline decisions."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize A/B testing framework.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get('rl', {}).get('ab_testing', {}).get('enabled', False)
        self.split_ratio = self.config.get('rl', {}).get('ab_testing', {}).get('split_ratio', 0.5)  # 50/50 split
        self.track_results = self.config.get('rl', {}).get('ab_testing', {}).get('track_results', True)
        
        self.results: List[ABTestResult] = []
        self.rl_group_stats = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'avg_confidence': 0.0,
            'total_confidence': 0.0
        })
        self.baseline_group_stats = defaultdict(lambda: {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0
        })
        
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self.logger.info(f"A/B testing enabled with {self.split_ratio:.1%} RL / {1-self.split_ratio:.1%} baseline split")
    
    def should_use_rl(self, trade_id: str) -> bool:
        """
        Determine if RL should be used for this trade (A/B testing).
        
        Args:
            trade_id: Unique trade identifier
            
        Returns:
            True if RL should be used, False for baseline
        """
        if not self.enabled:
            return True  # Default to RL if A/B testing disabled
        
        # Use deterministic hash for consistent assignment
        hash_value = hash(trade_id) % 100
        use_rl = hash_value < (self.split_ratio * 100)
        
        return use_rl
    
    def record_decision(
        self,
        trade_id: str,
        symbol: str,
        rl_decision: Dict[str, Any],
        baseline_decision: Dict[str, Any],
        rl_used: bool
    ):
        """
        Record a decision for A/B testing.
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            rl_decision: Decision made with RL
            baseline_decision: Decision made without RL
            rl_used: Whether RL was actually used
        """
        if not self.track_results:
            return
        
        result = ABTestResult(
            trade_id=trade_id,
            symbol=symbol,
            rl_decision=rl_decision.copy(),
            baseline_decision=baseline_decision.copy(),
            rl_used=rl_used
        )
        self.results.append(result)
    
    def record_outcome(
        self,
        trade_id: str,
        pnl: float,
        pnl_pct: float,
        success: bool
    ):
        """
        Record the outcome of a trade.
        
        Args:
            trade_id: Unique trade identifier
            pnl: Profit/loss amount
            pnl_pct: Profit/loss percentage
            success: Whether trade was successful
        """
        if not self.track_results:
            return
        
        # Find the result for this trade
        for result in self.results:
            if result.trade_id == trade_id:
                result.actual_outcome = {
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'success': success
                }
                
                # Update statistics
                if result.rl_used:
                    stats = self.rl_group_stats[result.symbol]
                    stats['total_trades'] += 1
                    stats['total_pnl'] += pnl
                    stats['total_pnl_pct'] += pnl_pct
                    if success:
                        stats['winning_trades'] += 1
                    if 'confidence' in result.rl_decision:
                        stats['total_confidence'] += result.rl_decision['confidence']
                        stats['avg_confidence'] = stats['total_confidence'] / stats['total_trades']
                else:
                    stats = self.baseline_group_stats[result.symbol]
                    stats['total_trades'] += 1
                    stats['total_pnl'] += pnl
                    stats['total_pnl_pct'] += pnl_pct
                    if success:
                        stats['winning_trades'] += 1
                
                break
    
    def get_comparison_report(self, symbol: Optional[str] = None) -> str:
        """
        Get comparison report between RL and baseline.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Formatted comparison report
        """
        if not self.results:
            return "No A/B test results available."
        
        # Filter by symbol if provided
        rl_stats = {}
        baseline_stats = {}
        
        if symbol:
            rl_stats = {symbol: self.rl_group_stats.get(symbol, {})}
            baseline_stats = {symbol: self.baseline_group_stats.get(symbol, {})}
        else:
            rl_stats = dict(self.rl_group_stats)
            baseline_stats = dict(self.baseline_group_stats)
        
        lines = ["A/B Testing Comparison Report:"]
        lines.append("=" * 60)
        
        for sym in set(list(rl_stats.keys()) + list(baseline_stats.keys())):
            lines.append(f"\nSymbol: {sym}")
            lines.append("-" * 60)
            
            rl = rl_stats.get(sym, {})
            baseline = baseline_stats.get(sym, {})
            
            rl_trades = rl.get('total_trades', 0)
            baseline_trades = baseline.get('total_trades', 0)
            
            if rl_trades > 0:
                rl_win_rate = rl.get('winning_trades', 0) / rl_trades
                rl_avg_pnl = rl.get('total_pnl_pct', 0) / rl_trades
                lines.append(f"RL Group: {rl_trades} trades, {rl_win_rate:.1%} win rate, {rl_avg_pnl:.2%} avg PnL")
            
            if baseline_trades > 0:
                baseline_win_rate = baseline.get('winning_trades', 0) / baseline_trades
                baseline_avg_pnl = baseline.get('total_pnl_pct', 0) / baseline_trades
                lines.append(f"Baseline Group: {baseline_trades} trades, {baseline_win_rate:.1%} win rate, {baseline_avg_pnl:.2%} avg PnL")
            
            if rl_trades > 0 and baseline_trades > 0:
                improvement = rl_avg_pnl - baseline_avg_pnl
                improvement_pct = (improvement / abs(baseline_avg_pnl) * 100) if baseline_avg_pnl != 0 else 0
                lines.append(f"RL Improvement: {improvement:.2%} ({improvement_pct:+.1f}%)")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return {
            'enabled': self.enabled,
            'split_ratio': self.split_ratio,
            'total_results': len(self.results),
            'rl_group_stats': dict(self.rl_group_stats),
            'baseline_group_stats': dict(self.baseline_group_stats)
        }

