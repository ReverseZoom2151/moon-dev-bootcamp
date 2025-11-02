import pandas as pd
import numpy as np
import logging
import asyncio
import itertools
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BacktestMode(Enum):
    SINGLE = "single"
    OPTIMIZATION = "optimization"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"

@dataclass
class BacktestResult:
    """Backtest result data structure"""
    strategy_name: str
    parameters: Dict[str, Any]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    trades: List[Dict]
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]

@dataclass
class Trade:
    """Trade data structure"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    duration: timedelta
    commission: float

class BacktestingEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.results: List[BacktestResult] = []
        
    async def run_backtest(
        self,
        strategy_class,
        data: pd.DataFrame,
        parameters: Dict[str, Any] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """Run a single backtest"""
        
        if parameters is None:
            parameters = {}
            
        # Filter data by date range
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)
            
        # Initialize strategy
        strategy = strategy_class(**parameters)
        
        # Run simulation
        trades, equity_curve = await self._simulate_strategy(strategy, data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)
        
        # Create result
        result = BacktestResult(
            strategy_name=strategy_class.__name__,
            parameters=parameters,
            start_date=data.index[0] if len(data) > 0 else datetime.now(),
            end_date=data.index[-1] if len(data) > 0 else datetime.now(),
            initial_capital=self.initial_capital,
            final_capital=equity_curve['equity'].iloc[-1] if len(equity_curve) > 0 else self.initial_capital,
            total_return=metrics.get('total_return', 0.0),
            annual_return=metrics.get('annual_return', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            total_trades=len(trades),
            avg_trade_duration=metrics.get('avg_trade_duration', 0.0),
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics
        )
        
        self.results.append(result)
        logger.info(f"Backtest completed: {result.strategy_name} - Return: {result.total_return:.2%}")
        
        return result
    
    async def optimize_parameters(
        self,
        strategy_class,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, List],
        optimization_metric: str = 'sharpe_ratio',
        max_workers: int = None
    ) -> List[BacktestResult]:
        """Optimize strategy parameters"""
        
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)
            
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        logger.info(f"Starting parameter optimization with {len(param_combinations)} combinations")
        
        # Run backtests in parallel
        tasks = []
        for params in param_combinations:
            task = self.run_backtest(strategy_class, data, params)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_results = [r for r in results if isinstance(r, BacktestResult)]
        
        # Sort by optimization metric
        valid_results.sort(key=lambda x: getattr(x, optimization_metric), reverse=True)
        
        logger.info(f"Optimization completed. Best {optimization_metric}: {getattr(valid_results[0], optimization_metric):.4f}")
        
        return valid_results
    
    async def walk_forward_analysis(
        self,
        strategy_class,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, List],
        train_period_days: int = 252,
        test_period_days: int = 63,
        step_days: int = 21
    ) -> List[BacktestResult]:
        """Perform walk-forward analysis"""
        
        results = []
        data_start = data.index[0]
        data_end = data.index[-1]
        
        current_date = data_start
        
        while current_date + timedelta(days=train_period_days + test_period_days) <= data_end:
            # Define train and test periods
            train_start = current_date
            train_end = current_date + timedelta(days=train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_period_days)
            
            # Split data
            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]
            
            logger.info(f"Walk-forward: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")
            
            # Optimize on training data
            optimization_results = await self.optimize_parameters(
                strategy_class, train_data, parameter_ranges
            )
            
            if optimization_results:
                # Use best parameters on test data
                best_params = optimization_results[0].parameters
                test_result = await self.run_backtest(
                    strategy_class, test_data, best_params
                )
                test_result.strategy_name += "_WalkForward"
                results.append(test_result)
            
            # Move to next period
            current_date += timedelta(days=step_days)
            
        logger.info(f"Walk-forward analysis completed with {len(results)} periods")
        return results
    
    async def monte_carlo_simulation(
        self,
        strategy_class,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        num_simulations: int = 1000,
        bootstrap_length: int = None
    ) -> List[BacktestResult]:
        """Perform Monte Carlo simulation"""
        
        if bootstrap_length is None:
            bootstrap_length = len(data)
            
        results = []
        
        for i in range(num_simulations):
            # Bootstrap sample the data
            sampled_data = data.sample(n=bootstrap_length, replace=True).sort_index()
            
            # Run backtest
            result = await self.run_backtest(strategy_class, sampled_data, parameters)
            result.strategy_name += f"_MC_{i+1}"
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Monte Carlo simulation: {i+1}/{num_simulations} completed")
                
        logger.info(f"Monte Carlo simulation completed with {num_simulations} runs")
        return results
    
    def _filter_data_by_date(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Filter data by date range"""
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        return data
    
    async def _simulate_strategy(self, strategy, data: pd.DataFrame) -> Tuple[List[Dict], pd.DataFrame]:
        """Simulate strategy execution"""
        trades = []
        equity_curve = []
        
        cash = self.initial_capital
        position = 0
        position_value = 0
        entry_price = 0
        entry_time = None
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Update strategy with current data
            signal = await self._get_strategy_signal(strategy, data.iloc[:i+1])
            
            current_price = row['close']
            equity = cash + position * current_price
            
            # Process signals
            if signal == 'BUY' and position <= 0:
                # Close short position if any
                if position < 0:
                    trade_pnl = position * (entry_price - current_price)
                    commission_cost = abs(position) * current_price * self.commission
                    net_pnl = trade_pnl - commission_cost
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'side': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'quantity': abs(position),
                        'pnl': net_pnl,
                        'pnl_percent': net_pnl / (abs(position) * entry_price),
                        'duration': (timestamp - entry_time).total_seconds() / 3600,  # hours
                        'commission': commission_cost
                    })
                    
                    cash += abs(position) * entry_price + net_pnl
                    position = 0
                
                # Open long position
                if cash > 0:
                    commission_cost = cash * self.commission
                    position = (cash - commission_cost) / current_price
                    cash = 0
                    entry_price = current_price
                    entry_time = timestamp
                    
            elif signal == 'SELL' and position >= 0:
                # Close long position if any
                if position > 0:
                    trade_pnl = position * (current_price - entry_price)
                    commission_cost = position * current_price * self.commission
                    net_pnl = trade_pnl - commission_cost
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'side': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'quantity': position,
                        'pnl': net_pnl,
                        'pnl_percent': net_pnl / (position * entry_price),
                        'duration': (timestamp - entry_time).total_seconds() / 3600,  # hours
                        'commission': commission_cost
                    })
                    
                    cash = position * current_price - commission_cost + net_pnl
                    position = 0
                
                # Open short position (if allowed)
                # This is simplified - in reality, shorting requires margin
                
            # Record equity
            equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': cash,
                'position': position,
                'position_value': position * current_price
            })
        
        return trades, pd.DataFrame(equity_curve)
    
    async def _get_strategy_signal(self, strategy, data: pd.DataFrame) -> str:
        """Get signal from strategy"""
        try:
            if hasattr(strategy, 'generate_signal'):
                return await strategy.generate_signal(data)
            elif hasattr(strategy, 'get_signal'):
                return strategy.get_signal(data)
            else:
                return 'HOLD'
        except Exception as e:
            logger.error(f"Error getting strategy signal: {e}")
            return 'HOLD'
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(equity_curve) == 0:
            return {}
            
        equity_series = equity_curve['equity']
        
        # Basic returns
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        
        # Annual return
        days = (equity_curve['timestamp'].iloc[-1] - equity_curve['timestamp'].iloc[0]).days
        annual_return = (1 + total_return) ** (365.25 / max(days, 1)) - 1
        
        # Daily returns for Sharpe ratio
        daily_returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades)
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            avg_trade_duration = np.mean([t['duration'] for t in trades])
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_duration = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'volatility': daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
            
        return combinations
    
    def get_summary_statistics(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Get summary statistics across multiple results"""
        if not results:
            return {}
            
        returns = [r.total_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'std_max_drawdown': np.std(max_drawdowns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'win_rate': len([r for r in returns if r > 0]) / len(returns)
        }
    
    def export_results(self, results: List[BacktestResult], filename: str):
        """Export results to CSV"""
        summary_data = []
        
        for result in results:
            summary_data.append({
                'strategy': result.strategy_name,
                'parameters': str(result.parameters),
                'start_date': result.start_date,
                'end_date': result.end_date,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")

class StrategyTester:
    """Helper class for testing individual strategies"""
    
    def __init__(self, backtesting_engine: BacktestingEngine):
        self.engine = backtesting_engine
        
    async def test_strategy_robustness(
        self,
        strategy_class,
        data: pd.DataFrame,
        base_parameters: Dict[str, Any],
        sensitivity_ranges: Dict[str, List],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, List[float]]:
        """Test strategy robustness by varying parameters"""
        
        results = {}
        
        for param_name, param_values in sensitivity_ranges.items():
            param_results = []
            
            for value in param_values:
                test_params = base_parameters.copy()
                test_params[param_name] = value
                
                result = await self.engine.run_backtest(strategy_class, data, test_params)
                param_results.append(getattr(result, metric))
                
            results[param_name] = param_results
            
        return results
    
    async def compare_strategies(
        self,
        strategy_classes: List,
        data: pd.DataFrame,
        parameters_list: List[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        if parameters_list is None:
            parameters_list = [{}] * len(strategy_classes)
            
        results = []
        
        for strategy_class, parameters in zip(strategy_classes, parameters_list):
            result = await self.engine.run_backtest(strategy_class, data, parameters)
            results.append(result)
            
        # Create comparison DataFrame
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Total Return': f"{result.total_return:.2%}",
                'Annual Return': f"{result.annual_return:.2%}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.3f}",
                'Max Drawdown': f"{result.max_drawdown:.2%}",
                'Win Rate': f"{result.win_rate:.2%}",
                'Total Trades': result.total_trades
            })
            
        return pd.DataFrame(comparison_data) 