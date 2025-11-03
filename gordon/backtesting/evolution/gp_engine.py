"""
Genetic Programming Strategy Evolution Engine
==============================================
Day 29: Genetic Programming for automated trading strategy discovery.

Uses DEAP (Distributed Evolutionary Algorithms in Python) to evolve
mathematical expressions that combine technical indicators into trading strategies.

Features:
- Evolves strategy logic (not just parameters)
- Multi-objective fitness (Return, Sharpe, Drawdown)
- Multiprocessing support for parallel evaluation
- Hall of Fame for best strategies
- Strategy persistence and reloading
"""

import numpy as np
import pandas as pd
import random
import operator
import warnings
import os
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass

# Optional DEAP import - gracefully handle if not installed
try:
    from deap import algorithms, base, creator, tools, gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    algorithms = base = tools = None
    # Create dummy classes for type hints
    class _DummyCreator:
        Individual = None
        FitnessMax = None
    creator = _DummyCreator()  # type: ignore
    gp = None  # type: ignore

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import talib  # type: ignore[import-untyped]
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None  # type: ignore[assignment]
    logger.warning("TA-Lib not available. Some indicators may not work.")

# Initialize DEAP creator classes at module level (if DEAP is available)
# This must be done before type annotations that reference creator.Individual
if DEAP_AVAILABLE and creator:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


@dataclass
class GPConfig:
    """Configuration for Genetic Programming evolution."""
    # Data
    data_path: Optional[str] = None
    
    # Technical Indicators
    sma_period: int = 20
    ema_period: int = 20
    rsi_period: int = 14
    
    # Backtesting
    initial_cash: float = 1_000_000
    commission_pct: float = 0.002
    
    # Genetic Programming
    population_size: int = 200
    generations: int = 30
    cxpb: float = 0.5  # Crossover probability
    mutpb: float = 0.2  # Mutation probability
    tournament_size: int = 3
    hall_of_fame_size: int = 5
    gp_min_depth: int = 1
    gp_max_depth: int = 4
    gp_mut_min_depth: int = 0
    gp_mut_max_depth: int = 2
    
    # Fitness weights
    return_weight: float = 1.0
    sharpe_weight: float = 0.3
    drawdown_weight: float = -0.1
    complexity_penalty: float = 0.005


class GPEvolutionEngine:
    """
    Genetic Programming evolution engine for trading strategies.
    
    Evolves mathematical expressions that combine technical indicators
    to generate trading signals.
    """

    def __init__(self, config: Optional[GPConfig] = None):
        """
        Initialize GP evolution engine.
        
        Args:
            config: GP configuration
        """
        self.config = config or GPConfig()
        self.pset = None
        self.toolbox = None
        self.hall_of_fame = None
        self.logbook = None
        
        # Setup DEAP
        self._setup_deap()
        
    def _setup_deap(self):
        """Setup DEAP creator classes (already initialized at module level)."""
        # Classes are already created at module level, but we verify they exist
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        logger.info("DEAP creator classes verified")

    def _protected_div(self, left: float, right: float) -> float:
        """Protected division (returns 1 for division by zero)."""
        return left / right if right != 0 else 1.0

    def _protected_log(self, x: float) -> float:
        """Protected logarithm (handles non-positive inputs)."""
        return np.log(x) if x > 0 else 0.0

    def _protected_sqrt(self, x: float) -> float:
        """Protected square root."""
        return np.sqrt(abs(x))

    def _generate_random_uniform(self) -> float:
        """Generate random float between -1.0 and 1.0."""
        return random.uniform(-1.0, 1.0)

    def create_primitive_set(self) -> Any:  # Returns gp.PrimitiveSet when DEAP available
        """
        Create DEAP primitive set with functions and terminals.
        
        Returns:
            PrimitiveSet with operations and inputs
        """
        # 9 input arguments: close, open, high, low, volume, sma, ema, rsi, macd
        pset = gp.PrimitiveSet("MAIN", 9, prefix="ARG")
        
        # Arithmetic operators
        pset.addPrimitive(operator.add, 2, name="add")
        pset.addPrimitive(operator.sub, 2, name="sub")
        pset.addPrimitive(operator.mul, 2, name="mul")
        pset.addPrimitive(self._protected_div, 2, name="pdiv")
        pset.addPrimitive(operator.neg, 1, name="neg")
        
        # Math functions
        pset.addPrimitive(np.sin, 1, name="sin")
        pset.addPrimitive(np.cos, 1, name="cos")
        pset.addPrimitive(self._protected_log, 1, name="plog")
        pset.addPrimitive(self._protected_sqrt, 1, name="psqrt")
        
        # Constants
        pset.addTerminal(1.0, name="const_1")
        pset.addTerminal(0.5, name="const_05")
        pset.addEphemeralConstant("rand101", self._generate_random_uniform)
        
        self.pset = pset
        logger.info("Primitive set created with 9 arguments and operators")
        return pset

    def compile_individual(
        self,
        individual: Any,  # creator.Individual when DEAP available
        pset: Any  # gp.PrimitiveSet when DEAP available
    ) -> Optional[Callable]:
        """
        Compile GP individual tree into callable function.
        
        Args:
            individual: GP individual (tree)
            pset: Primitive set
            
        Returns:
            Compiled function or None if compilation fails
        """
        try:
            return gp.compile(individual, pset)
        except Exception as e:
            logger.debug(f"Error compiling individual: {e}")
            return None

    def evaluate_strategy(
        self,
        individual: Any,  # creator.Individual when DEAP available
        pset: Any,  # gp.PrimitiveSet when DEAP available
        data: pd.DataFrame,
        strategy_class: Any,
        config: GPConfig
    ) -> Tuple[float]:
        """
        Evaluate a GP individual using backtesting.
        
        Args:
            individual: GP individual to evaluate
            pset: Primitive set
            data: Market data DataFrame
            strategy_class: Strategy class to use
            config: GP configuration
            
        Returns:
            Fitness tuple (fitness value,)
        """
        from backtesting import Backtest
        
        # Compile individual to function
        gp_func = self.compile_individual(individual, pset)
        if gp_func is None:
            return (config.return_weight * -1000.0,)  # Very bad fitness
        
        # Set GP function in strategy class
        strategy_class.gp_function = gp_func
        
        try:
            # Run backtest
            bt = Backtest(
                data,
                strategy_class,
                cash=config.initial_cash,
                commission=config.commission_pct
            )
            stats = bt.run()
            
            # Extract metrics
            return_pct = stats.get('Return [%]', 0.0)
            sharpe_ratio = stats.get('Sharpe Ratio', 0.0)
            max_drawdown = stats.get('Max. Drawdown [%]', 0.0)
            
            # Handle NaN/inf values
            if np.isnan(return_pct) or np.isinf(return_pct):
                return_pct = -1000.0
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                sharpe_ratio = 0.0
            if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                max_drawdown = 100.0
            
            # Calculate multi-objective fitness
            fitness = (
                config.return_weight * return_pct +
                config.sharpe_weight * sharpe_ratio * 10.0 +  # Scale Sharpe
                config.drawdown_weight * abs(max_drawdown) -
                config.complexity_penalty * len(individual)  # Penalize complexity
            )
            
            return (fitness,)
            
        except Exception as e:
            logger.debug(f"Error evaluating strategy: {e}")
            return (config.return_weight * -1000.0,)  # Very bad fitness

    def setup_toolbox(
        self,
        pset: Any,  # gp.PrimitiveSet when DEAP available
        data: pd.DataFrame,
        strategy_class: Any,
        config: GPConfig,
        use_multiprocessing: bool = True
    ) -> Any:  # base.Toolbox when DEAP available
        """
        Setup DEAP toolbox with registered functions.
        
        Args:
            pset: Primitive set
            data: Market data
            strategy_class: Strategy class
            config: GP configuration
            use_multiprocessing: Whether to use multiprocessing
            
        Returns:
            Configured toolbox
        """
        toolbox = base.Toolbox() if DEAP_AVAILABLE and base else None
        
        # Expression generator
        toolbox.register(
            "expr",
            gp.genHalfAndHalf,
            pset=pset,
            min_=config.gp_min_depth,
            max_=config.gp_max_depth
        )
        
        # Individual and population
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual if DEAP_AVAILABLE and creator else None,
            toolbox.expr
        )
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual
        )
        
        # Evaluation function
        toolbox.register(
            "evaluate",
            self.evaluate_strategy,
            pset=pset,
            data=data,
            strategy_class=strategy_class,
            config=config
        )
        
        # Genetic operators
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=config.tournament_size
        )
        toolbox.register("mate", gp.cxOnePoint)  # Crossover
        toolbox.register(
            "expr_mut",
            gp.genFull,
            min_=config.gp_mut_min_depth,
            max_=config.gp_mut_max_depth
        )
        toolbox.register(
            "mutate",
            gp.mutUniform,
            expr=toolbox.expr_mut,
            pset=pset
        )
        
        # Constraints (limit tree depth)
        toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        
        # Multiprocessing support
        if use_multiprocessing:
            try:
                import multiprocessing
                start_method = 'fork' if hasattr(os, 'fork') else None
                mp_context = multiprocessing.get_context(start_method)
                num_workers = min(60, os.cpu_count() if os.cpu_count() else 4)
                pool = mp_context.Pool(processes=num_workers)
                toolbox.register("map", pool.map)
                logger.info(f"Multiprocessing enabled with {num_workers} workers")
            except Exception as e:
                logger.warning(f"Multiprocessing setup failed: {e}")
        
        self.toolbox = toolbox
        return toolbox

    def run_evolution(
        self,
        toolbox: Any,  # base.Toolbox when DEAP available
        config: GPConfig
    ) -> Tuple[List, Any, Any]:  # (final_population, logbook, hall_of_fame)
        """
        Run genetic programming evolution.
        
        Args:
            toolbox: Configured DEAP toolbox
            config: GP configuration
            
        Returns:
            Tuple of (final_population, logbook, hall_of_fame)
        """
        # Initialize population
        pop = toolbox.population(n=config.population_size)
        hof = tools.HallOfFame(config.hall_of_fame_size)
        
        # Statistics
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else np.nan)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.nanmean)
        mstats.register("std", np.nanstd)
        mstats.register("min", np.nanmin)
        mstats.register("max", np.nanmax)
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + mstats.fields
        
        logger.info(
            f"Starting evolution: {config.population_size} individuals, "
            f"{config.generations} generations"
        )
        
        try:
            # Run evolution
            pop, logbook = algorithms.eaSimple(
                pop,
                toolbox,
                cxpb=config.cxpb,
                mutpb=config.mutpb,
                ngen=config.generations,
                stats=mstats,
                halloffame=hof,
                verbose=True
            )
            
            logger.info("Evolution completed successfully")
            
        except Exception as e:
            logger.error(f"Error during evolution: {e}")
        finally:
            # Cleanup multiprocessing
            if hasattr(toolbox, 'map') and hasattr(toolbox.map, '__self__'):
                try:
                    pool = toolbox.map.__self__
                    pool.close()
                    pool.join()
                    toolbox.unregister("map")
                    logger.info("Multiprocessing pool closed")
                except:
                    pass
        
        self.hall_of_fame = hof
        self.logbook = logbook
        
        return pop, logbook, hof

    def save_results(
        self,
        hof: Any,  # tools.HallOfFame when DEAP available
        filepath: str,
        symbol: str = "UNKNOWN"
    ):
        """
        Save Hall of Fame results to CSV.
        
        Args:
            hof: Hall of Fame
            filepath: Output file path
            symbol: Trading symbol
        """
        if not hof:
            logger.warning("Hall of Fame is empty. No results to save.")
            return
        
        results = []
        for i, ind in enumerate(hof):
            try:
                fitness_val = ind.fitness.values[0] if ind.fitness.valid else 'Invalid'
                results.append({
                    'Rank': i + 1,
                    'Symbol': symbol,
                    'Fitness': fitness_val,
                    'Size': len(ind),
                    'Height': ind.height if hasattr(ind, 'height') else 'N/A',
                    'Strategy_Tree': str(ind)
                })
            except Exception as e:
                logger.error(f"Error processing individual {i}: {e}")
        
        if not results:
            logger.warning("No valid individuals in Hall of Fame to save.")
            return
        
        try:
            results_df = pd.DataFrame(results)
            results_df.to_csv(filepath, index=False)
            logger.info(f"Hall of Fame results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

