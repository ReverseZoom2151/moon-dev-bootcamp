"""
RL Performance Monitor
======================
Tracks RL component performance metrics including latency, success rates, and error rates.
"""

import time
import logging
from typing import Dict, Optional, Any, List
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RLMetrics:
    """Metrics for a single RL component call."""
    component_name: str
    operation: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RLPerformanceMonitor:
    """Monitors RL component performance."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.component_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_latency_ms': 0.0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'min_latency_ms': float('inf'),
            'recent_latencies': deque(maxlen=100)  # Last 100 latencies
        })
        self.logger = logging.getLogger(__name__)
        
    def record_call(
        self,
        component_name: str,
        operation: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Record a single RL component call.
        
        Args:
            component_name: Name of RL component (e.g., 'regime_detector')
            operation: Operation performed (e.g., 'detect_regime')
            latency_ms: Latency in milliseconds
            success: Whether call was successful
            error: Error message if failed
        """
        metric = RLMetrics(
            component_name=component_name,
            operation=operation,
            latency_ms=latency_ms,
            success=success,
            error=error
        )
        self.metrics_history.append(metric)
        
        # Update component statistics
        stats = self.component_stats[component_name]
        stats['total_calls'] += 1
        stats['total_latency_ms'] += latency_ms
        stats['recent_latencies'].append(latency_ms)
        
        if success:
            stats['successful_calls'] += 1
        else:
            stats['failed_calls'] += 1
            if error:
                self.logger.warning(f"RL {component_name}.{operation} failed: {error}")
        
        # Update min/max latency
        if latency_ms > stats['max_latency_ms']:
            stats['max_latency_ms'] = latency_ms
        if latency_ms < stats['min_latency_ms']:
            stats['min_latency_ms'] = latency_ms
        
        # Update average latency
        if stats['total_calls'] > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total_calls']
        
        # Log warning if latency is high
        if latency_ms > 100:  # 100ms threshold
            self.logger.warning(
                f"RL {component_name}.{operation} took {latency_ms:.2f}ms "
                f"(above 100ms threshold)"
            )
    
    def get_component_stats(self, component_name: str) -> Dict[str, Any]:
        """Get statistics for a specific component."""
        return self.component_stats.get(component_name, {}).copy()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all components."""
        return {k: v.copy() for k, v in self.component_stats.items()}
    
    def get_recent_avg_latency(self, component_name: str, window: int = 10) -> float:
        """
        Get average latency for recent calls.
        
        Args:
            component_name: Component name
            window: Number of recent calls to consider
            
        Returns:
            Average latency in milliseconds
        """
        stats = self.component_stats.get(component_name, {})
        latencies = list(stats.get('recent_latencies', []))
        if not latencies:
            return 0.0
        
        recent = latencies[-window:] if len(latencies) > window else latencies
        return sum(recent) / len(recent) if recent else 0.0
    
    def get_success_rate(self, component_name: str) -> float:
        """Get success rate for a component."""
        stats = self.component_stats.get(component_name, {})
        total = stats.get('total_calls', 0)
        if total == 0:
            return 0.0
        return stats.get('successful_calls', 0) / total
    
    def should_warn(self, component_name: str, latency_threshold_ms: float = 100) -> bool:
        """Check if component should be warned about."""
        stats = self.component_stats.get(component_name, {})
        return stats.get('avg_latency_ms', 0) > latency_threshold_ms
    
    def get_summary(self) -> str:
        """Get a summary of all RL component performance."""
        if not self.component_stats:
            return "No RL component calls recorded."
        
        lines = ["RL Performance Summary:"]
        for component, stats in self.component_stats.items():
            success_rate = self.get_success_rate(component)
            lines.append(
                f"  {component}: "
                f"{stats['total_calls']} calls, "
                f"{success_rate:.1%} success rate, "
                f"{stats['avg_latency_ms']:.2f}ms avg latency"
            )
        
        return "\n".join(lines)

