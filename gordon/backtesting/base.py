"""
Base Classes for Backtesting Module

Provides framework-agnostic interfaces and data structures for all backtesting
components, ensuring consistency across different frameworks and strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtests"""
    initial_cash: float = 10000
    commission: float = 0.001
    slippage: float = 0.0
    data_provider: str = 'mock'
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbol: str = 'BTCUSDT'
    timeframe: str = '1h'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        if data['start_date']:
            data['start_date'] = data['start_date'].isoformat()
        if data['end_date']:
            data['end_date'] = data['end_date'].isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class BacktestMetrics:
    """Key metrics from backtest results"""
    initial_value: float
    final_value: float
    total_return: float  # percentage
    sharpe_ratio: float
    max_drawdown: float  # percentage
    num_trades: int
    win_rate: float  # percentage
    avg_trade: Optional[float] = None
    profit_factor: Optional[float] = None
    max_consecutive_losses: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio"""
        if self.max_drawdown == 0:
            return float('inf')
        return abs(self.total_return / self.max_drawdown)
    
    @property
    def profit_per_trade(self) -> float:
        """Calculate average profit per trade"""
        if self.num_trades == 0:
            return 0
        return self.total_return / self.num_trades


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float  # profit/loss
    pnl_pct: float  # profit/loss percentage
    commission: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'side': self.side,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission
        }


@dataclass
class BacktestResult:
    """Unified result format for all backtests"""
    strategy_name: str
    framework: str  # 'backtrader' or 'backtesting_lib'
    metrics: BacktestMetrics
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    optimal_params: Optional[Dict] = None
    execution_time: float = 0.0  # seconds
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'framework': self.framework,
            'metrics': self.metrics.to_dict(),
            'config': self.config.to_dict(),
            'trades': [t.to_dict() for t in self.trades],
            'optimal_params': self.optimal_params,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Convert to JSON string or save to file"""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Results saved to {filepath}")
        return json_str
    
    @staticmethod
    def from_dict(data: Dict) -> 'BacktestResult':
        """Create from dictionary"""
        metrics = BacktestMetrics(**data['metrics'])
        config = BacktestConfig(**data['config'])
        trades = [Trade(**t) for t in data['trades']]
        return BacktestResult(
            strategy_name=data['strategy_name'],
            framework=data['framework'],
            metrics=metrics,
            config=config,
            trades=trades,
            optimal_params=data.get('optimal_params'),
            execution_time=data.get('execution_time', 0.0)
        )


class BaseStrategy(ABC):
    """Base class for all strategies (framework-agnostic interface)"""
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        pass
    
    @abstractmethod
    def init(self):
        """Initialize strategy indicators and state"""
        pass
    
    @abstractmethod
    def next(self):
        """Execute strategy logic for current bar"""
        pass


class DataProvider(ABC):
    """Abstract data provider interface"""
    
    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, 
              start: datetime, end: datetime) -> 'pd.DataFrame':
        """Fetch OHLCV data"""
        pass
    
    @abstractmethod
    def validate_data(self, df: 'pd.DataFrame') -> bool:
        """Validate data integrity"""
        pass
    
    @property
    @abstractmethod
    def column_mapping(self) -> Dict[str, str]:
        """Get column name mapping"""
        pass


class BacktestRunner(ABC):
    """Base class for backtest runners"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = {}
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def run(self, strategy_class: type, **kwargs) -> BacktestResult:
        """Run backtest with given strategy"""
        pass
    
    @abstractmethod
    def optimize(self, strategy_class: type, param_ranges: Dict, 
                 **kwargs) -> BacktestResult:
        """Optimize strategy parameters"""
        pass
