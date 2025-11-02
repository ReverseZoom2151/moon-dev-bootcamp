"""
Genetic Algorithm Strategy Optimizer
Automatically optimizes trading strategy parameters using evolutionary algorithms
Based on Day 29 genetic programming implementation
"""

import logging
import numpy as np
import pandas as pd
import random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from deap import base, creator, tools
from strategies.base_strategy import BaseStrategy, StrategySignal

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Results from genetic optimization"""
    best_parameters: Dict[str, Any]
    best_fitness: float
    generation: int
    population_size: int
    convergence_history: List[float]
    optimization_time: float


class ParameterSpace:
    """Defines the parameter space for optimization"""
    
    def __init__(self):
        self.parameters = {}
    
    def add_parameter(self, name: str, param_type: str, min_val: float, max_val: float, step: float = None):
        """Add a parameter to optimize"""
        self.parameters[name] = {
            'type': param_type,
            'min': min_val,
            'max': max_val,
            'step': step
        }
    
    def generate_random_individual(self) -> Dict[str, Any]:
        """Generate a random parameter set"""
        individual = {}
        for name, param in self.parameters.items():
            if param['type'] == 'float':
                individual[name] = random.uniform(param['min'], param['max'])
            elif param['type'] == 'int':
                individual[name] = random.randint(int(param['min']), int(param['max']))
            elif param['type'] == 'discrete':
                # For discrete parameters, step defines the possible values
                values = np.arange(param['min'], param['max'] + param['step'], param['step'])
                individual[name] = random.choice(values)
        
        return individual
    
    def mutate_individual(self, individual: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutate an individual's parameters"""
        mutated = individual.copy()
        
        for name, param in self.parameters.items():
            if random.random() < mutation_rate:
                if param['type'] == 'float':
                    # Gaussian mutation
                    mutation_strength = (param['max'] - param['min']) * 0.1
                    mutated[name] = np.clip(
                        individual[name] + random.gauss(0, mutation_strength),
                        param['min'], param['max']
                    )
                elif param['type'] == 'int':
                    # Random integer mutation
                    mutated[name] = random.randint(int(param['min']), int(param['max']))
                elif param['type'] == 'discrete':
                    values = np.arange(param['min'], param['max'] + param['step'], param['step'])
                    mutated[name] = random.choice(values)
        
        return mutated
    
    def crossover_individuals(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover two individuals"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for name in self.parameters.keys():
            if random.random() < 0.5:
                child1[name], child2[name] = child2[name], child1[name]
        
        return child1, child2


class StrategyBacktester:
    """Backtests strategies with given parameters"""
    
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
    
    async def evaluate_parameters(self, strategy_class, parameters: Dict[str, Any]) -> float:
        """Evaluate strategy parameters and return fitness score"""
        try:
            # Create strategy instance with parameters
            config = {**parameters, 'symbols': ['BTC']}  # Use BTC for optimization
            
            # Simulate backtesting (in production, use actual backtesting)
            # This would run the strategy on historical data and calculate performance
            
            # For now, simulate realistic performance based on parameters
            fitness = self._simulate_strategy_performance(parameters)
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return -float('inf')  # Worst possible fitness
    
    def _simulate_strategy_performance(self, parameters: Dict[str, Any]) -> float:
        """Simulate strategy performance for given parameters"""
        # This is a simplified simulation
        # In production, this would run actual backtesting
        
        # Generate realistic performance based on parameter values
        base_return = 0.05  # 5% base return
        
        # Add parameter-based adjustments
        if 'sma_period' in parameters:
            # Optimal SMA period around 20
            sma_penalty = abs(parameters['sma_period'] - 20) * 0.001
            base_return -= sma_penalty
        
        if 'rsi_period' in parameters:
            # Optimal RSI period around 14
            rsi_penalty = abs(parameters['rsi_period'] - 14) * 0.001
            base_return -= rsi_penalty
        
        if 'buy_threshold' in parameters and 'sell_threshold' in parameters:
            # Penalize if thresholds are too close
            threshold_diff = abs(parameters['sell_threshold'] - parameters['buy_threshold'])
            if threshold_diff < 5:
                base_return -= 0.02
        
        # Add some randomness to simulate market conditions
        noise = random.gauss(0, 0.02)
        
        return base_return + noise


class GeneticOptimizer:
    """Genetic algorithm optimizer for trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.population_size = config.get('population_size', 50)
        self.generations = config.get('generations', 20)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.tournament_size = config.get('tournament_size', 3)
        
        # Setup DEAP
        self._setup_deap()
        
        logger.info(f"🧬 Genetic Optimizer initialized:")
        logger.info(f"   Population Size: {self.population_size}")
        logger.info(f"   Generations: {self.generations}")
        logger.info(f"   Mutation Rate: {self.mutation_rate}")
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", dict, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
    
    async def optimize_strategy(
        self, 
        strategy_class, 
        parameter_space: ParameterSpace,
        historical_data: pd.DataFrame
    ) -> OptimizationResult:
        """Optimize strategy parameters using genetic algorithm"""
        start_time = datetime.utcnow()
        
        logger.info(f"🧬 Starting genetic optimization for {strategy_class.__name__}")
        
        # Initialize backtester
        backtester = StrategyBacktester(historical_data)
        
        # Setup genetic algorithm
        self.toolbox.register("individual", self._create_individual, parameter_space)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual, strategy_class, backtester)
        self.toolbox.register("mate", self._crossover, parameter_space)
        self.toolbox.register("mutate", self._mutate, parameter_space)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Track convergence
        convergence_history = []
        hall_of_fame = tools.HallOfFame(1)
        
        # Evolution loop
        for generation in range(self.generations):
            logger.info(f"🧬 Generation {generation + 1}/{self.generations}")
            
            # Evaluate population
            fitnesses = []
            for individual in population:
                fitness = await backtester.evaluate_parameters(strategy_class, individual)
                individual.fitness.values = (fitness,)
                fitnesses.append(fitness)
            
            # Update hall of fame
            hall_of_fame.update(population)
            
            # Track best fitness
            best_fitness = max(fitnesses)
            convergence_history.append(best_fitness)
            
            logger.info(f"   Best fitness: {best_fitness:.4f}")
            logger.info(f"   Avg fitness: {np.mean(fitnesses):.4f}")
            
            # Selection and reproduction
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Replace population
            population[:] = offspring
        
        # Get best result
        best_individual = hall_of_fame[0]
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = OptimizationResult(
            best_parameters=dict(best_individual),
            best_fitness=best_individual.fitness.values[0],
            generation=self.generations,
            population_size=self.population_size,
            convergence_history=convergence_history,
            optimization_time=optimization_time
        )
        
        logger.info(f"🧬 Optimization complete!")
        logger.info(f"   Best parameters: {result.best_parameters}")
        logger.info(f"   Best fitness: {result.best_fitness:.4f}")
        logger.info(f"   Time taken: {optimization_time:.1f}s")
        
        return result
    
    def _create_individual(self, parameter_space: ParameterSpace):
        """Create a random individual"""
        individual = creator.Individual(parameter_space.generate_random_individual())
        return individual
    
    async def _evaluate_individual(self, strategy_class, backtester: StrategyBacktester, individual):
        """Evaluate an individual's fitness"""
        return await backtester.evaluate_parameters(strategy_class, individual)
    
    def _crossover(self, parameter_space: ParameterSpace, ind1, ind2):
        """Crossover two individuals"""
        child1, child2 = parameter_space.crossover_individuals(ind1, ind2)
        ind1.clear()
        ind1.update(child1)
        ind2.clear()
        ind2.update(child2)
        return ind1, ind2
    
    def _mutate(self, parameter_space: ParameterSpace, individual):
        """Mutate an individual"""
        mutated = parameter_space.mutate_individual(individual, self.mutation_rate)
        individual.clear()
        individual.update(mutated)
        return individual,


class GeneticOptimizerStrategy(BaseStrategy):
    """
    Strategy that uses genetic algorithms to optimize other strategies
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager, name: str = "GeneticOptimizer"):
        super().__init__(config, market_data_manager, name)
        
        # Optimization configuration
        self.target_strategy = config.get('target_strategy', 'mean_reversion')
        self.optimization_interval = config.get('optimization_interval', 24)  # hours
        self.lookback_days = config.get('lookback_days', 30)
        
        # Genetic algorithm settings
        self.ga_config = {
            'population_size': config.get('population_size', 50),
            'generations': config.get('generations', 20),
            'mutation_rate': config.get('mutation_rate', 0.1),
            'crossover_rate': config.get('crossover_rate', 0.7)
        }
        
        # Optimizer
        self.optimizer = GeneticOptimizer(self.ga_config)
        
        # Optimization tracking
        self.last_optimization = None
        self.current_best_parameters = None
        self.optimization_history = []
        
        logger.info(f"🧬 Genetic Optimizer Strategy initialized:")
        logger.info(f"   Target Strategy: {self.target_strategy}")
        logger.info(f"   Optimization Interval: {self.optimization_interval}h")
    
    async def _initialize_strategy(self):
        """Initialize genetic optimizer strategy"""
        try:
            logger.info("✅ Genetic optimizer strategy validation complete")
        except Exception as e:
            logger.error(f"❌ Failed to initialize genetic optimizer strategy: {e}")
            raise
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate signal by optimizing target strategy"""
        try:
            # Check if optimization is needed
            if await self._should_optimize():
                await self._run_optimization()
            
            # For now, return None as this is primarily an optimization strategy
            # In production, this would use the optimized parameters to generate signals
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in genetic optimizer: {e}", exc_info=True)
            return None
    
    async def _should_optimize(self) -> bool:
        """Check if optimization should be run"""
        if self.last_optimization is None:
            return True
        
        hours_since_optimization = (datetime.utcnow() - self.last_optimization).total_seconds() / 3600
        return hours_since_optimization >= self.optimization_interval
    
    async def _run_optimization(self):
        """Run genetic algorithm optimization"""
        try:
            logger.info(f"🧬 Running optimization for {self.target_strategy}")
            
            # Get historical data
            historical_data = await self._get_historical_data()
            if historical_data is None:
                return
            
            # Define parameter space based on target strategy
            parameter_space = self._get_parameter_space()
            
            # Run optimization
            from strategies.mean_reversion import MeanReversionStrategy  # Example
            result = await self.optimizer.optimize_strategy(
                MeanReversionStrategy,
                parameter_space,
                historical_data
            )
            
            # Store results
            self.current_best_parameters = result.best_parameters
            self.optimization_history.append(result)
            self.last_optimization = datetime.utcnow()
            
            logger.info(f"🧬 Optimization completed with fitness: {result.best_fitness:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error running optimization: {e}")
    
    async def _get_historical_data(self) -> Optional[pd.DataFrame]:
        """Get historical data for optimization"""
        try:
            # Get data for the first symbol
            symbol = self.symbols[0] if self.symbols else 'BTC'
            data = await self._get_market_data(symbol, limit=self.lookback_days * 24)  # Hourly data
            
            if data is None:
                return None
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data: {e}")
            return None
    
    def _get_parameter_space(self) -> ParameterSpace:
        """Define parameter space for target strategy"""
        space = ParameterSpace()
        
        if self.target_strategy == 'mean_reversion':
            space.add_parameter('sma_period', 'int', 10, 50)
            space.add_parameter('buy_threshold', 'float', 5.0, 20.0)
            space.add_parameter('sell_threshold', 'float', 10.0, 30.0)
        elif self.target_strategy == 'bollinger_bands':
            space.add_parameter('period', 'int', 15, 30)
            space.add_parameter('std_dev', 'float', 1.5, 3.0)
        elif self.target_strategy == 'rsi':
            space.add_parameter('rsi_period', 'int', 10, 20)
            space.add_parameter('overbought', 'float', 70.0, 85.0)
            space.add_parameter('oversold', 'float', 15.0, 30.0)
        
        return space
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and optimization results"""
        return {
            "name": self.name,
            "type": "genetic_optimizer",
            "target_strategy": self.target_strategy,
            "optimization_interval": self.optimization_interval,
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "current_best_parameters": self.current_best_parameters,
            "optimization_count": len(self.optimization_history),
            "best_fitness_history": [r.best_fitness for r in self.optimization_history[-10:]],
            "status": self.status.value,
            "enabled": self.enabled
        } 