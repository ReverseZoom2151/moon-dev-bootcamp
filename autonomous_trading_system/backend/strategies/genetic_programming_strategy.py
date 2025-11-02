"""
Genetic Programming Trading Strategy
Evolves trading strategies using DEAP Genetic Programming framework
Based on Day_29_Projects/ge_p1.py implementation
"""

import logging
import numpy as np
import pandas as pd
import operator
import random
import multiprocessing
import warnings
import os
import pickle
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass
from deap import algorithms, base, creator, tools, gp
from strategies.base_strategy import BaseStrategy, StrategySignal, SignalAction, TechnicalIndicatorMixin

# Try to import talib, but make it optional
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not installed. Using fallback technical indicators. Some functionality may be limited.")

logger = logging.getLogger(__name__)

# Suppress runtime warnings for cleaner evolution output
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class GPStrategyConfig:
    """Configuration for Genetic Programming Strategy"""
    # Data parameters
    initial_cash: float = 1_000_000
    commission_pct: float = 0.002
    
    # Technical indicators
    sma_period: int = 20
    ema_period: int = 20
    rsi_period: int = 14
    
    # Genetic Programming parameters
    population_size: int = 300
    generations: int = 40
    crossover_prob: float = 0.5
    mutation_prob: float = 0.2
    tournament_size: int = 3
    hall_of_fame_size: int = 10
    
    # GP tree constraints
    gp_min_depth: int = 1
    gp_max_depth: int = 4
    gp_mut_min_depth: int = 0
    gp_mut_max_depth: int = 2
    gp_max_height: int = 17
    
    # Evolution settings
    worst_fitness: Tuple[float] = (-float('inf'),)
    buy_threshold: float = 0.5
    sell_threshold: float = -0.5
    
    # Multiprocessing
    max_workers: int = None  # Auto-detect
    
    # Persistence
    save_results: bool = True
    save_models: bool = True
    results_dir: str = "./data/gp_evolution"

class GPHelper:
    """Helper functions for Genetic Programming"""
    
    @staticmethod
    def protected_div(left: float, right: float) -> float:
        """Safely divide, returning 1 for division by zero."""
        return left / right if abs(right) > 1e-10 else 1.0
    
    @staticmethod
    def protected_log(x: float) -> float:
        """Safely compute log, handling non-positive inputs."""
        return np.log(abs(x)) if abs(x) > 1e-10 else 0.0
    
    @staticmethod
    def protected_sqrt(x: float) -> float:
        """Safely compute square root."""
        return np.sqrt(abs(x))
    
    @staticmethod
    def protected_pow(base: float, exp: float) -> float:
        """Safely compute power, with constraints."""
        try:
            if abs(exp) > 10:  # Limit exponent
                exp = 10 if exp > 0 else -10
            if abs(base) > 1000:  # Limit base
                base = 1000 if base > 0 else -1000
            result = np.power(abs(base), exp)
            return result if abs(result) < 1e10 else 1.0
        except (OverflowError, ValueError):
            return 1.0
    
    @staticmethod
    def generate_random_uniform() -> float:
        """Generates a random float between -1.0 and 1.0 for GP ephemeral constants."""
        return random.uniform(-1.0, 1.0)
    
    @staticmethod
    def if_then_else(condition: float, val_if_true: float, val_if_false: float) -> float:
        """Conditional function for GP."""
        return val_if_true if condition > 0 else val_if_false

class EvolvedTradingFunction:
    """Wrapper for evolved trading functions"""
    
    def __init__(self, gp_function: Callable, individual_str: str, fitness: float):
        self.gp_function = gp_function
        self.individual_str = individual_str
        self.fitness = fitness
        self.created_at = datetime.now()
    
    def __call__(self, *args):
        """Execute the evolved function with error handling"""
        try:
            return self.gp_function(*args)
        except (OverflowError, ValueError, TypeError, ZeroDivisionError):
            return 0.0  # Neutral signal on error
        except Exception as e:
            logger.warning(f"Unexpected error in evolved function: {e}")
            return 0.0

class TechnicalIndicators:
    """Fallback technical indicators when TA-Lib is not available"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26) -> pd.Series:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        return ema_fast - ema_slow

class GeneticProgrammingStrategy(BaseStrategy, TechnicalIndicatorMixin):
    """
    Genetic Programming Strategy that evolves trading algorithms
    
    Features:
    - Evolves complete trading strategies using GP
    - Uses multiple technical indicators as inputs
    - Supports multi-generation evolution
    - Saves and loads evolved strategies
    - Real-time strategy execution
    """
    
    def __init__(self, config: Dict[str, Any], market_data_manager=None, name: Optional[str] = None):
        # Initialize base strategy
        super().__init__(config, market_data_manager, name)
        
        # GP-specific configuration
        self.gp_config = GPStrategyConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.gp_config, key):
                    setattr(self.gp_config, key, value)
        
        # GP components
        self.pset: Optional[gp.PrimitiveSet] = None
        self.toolbox: Optional[base.Toolbox] = None
        self.evolved_function: Optional[EvolvedTradingFunction] = None
        self.hall_of_fame: Optional[tools.HallOfFame] = None
        
        # Evolution tracking
        self.evolution_stats = {}
        self.generation_history = []
        self.is_evolved = False
        
        # Ensure results directory exists
        os.makedirs(self.gp_config.results_dir, exist_ok=True)
        
        logger.info("🧬 Genetic Programming Strategy initialized")
    
    async def _initialize_strategy(self):
        """Strategy-specific initialization logic"""
        logger.info("🧬 Initializing GP strategy components...")
        
        # Create GP primitive set
        self.pset = self.create_primitive_set()
        self.setup_deap()
        
        # Try to load existing evolved strategy
        try:
            # Look for existing models in results directory
            import glob
            model_files = glob.glob(os.path.join(self.gp_config.results_dir, "gp_model_*.pkl"))
            if model_files:
                # Load the most recent model
                latest_model = max(model_files, key=os.path.getctime)
                if self.load_evolved_strategy(latest_model):
                    logger.info(f"✅ Loaded existing evolved strategy from {latest_model}")
                else:
                    logger.info("🧬 No existing evolved strategy found, will need to evolve new one")
        except Exception as e:
            logger.warning(f"⚠️ Could not load existing evolved strategy: {e}")
    
    async def generate_signal(self) -> Optional[StrategySignal]:
        """Generate trading signal using the evolved strategy."""
        if not self.is_evolved or not self.evolved_function:
            return None
        
        try:
            # Get the first symbol (GP strategy works with single symbol)
            symbol = self.symbols[0] if self.symbols else "BTC"
            
            # Get latest market data
            data = await self._get_market_data(symbol, limit=50)
            if data is None or len(data) < 20:
                return None
            
            # Calculate indicators using the data
            indicators = await self._calculate_indicators(symbol, data)
            
            # Get latest values
            latest = data.iloc[-1]
            
            # Get signal from evolved function
            signal_value = self.evolved_function(
                latest['close'], latest['open'], latest['high'], 
                latest['low'], latest['volume'],
                indicators['sma'], indicators['ema'], 
                indicators['rsi'], indicators['macd']
            )
            
            # Interpret signal
            if signal_value > self.gp_config.buy_threshold:
                action = SignalAction.BUY
                confidence = min(abs(signal_value) / 2.0, 1.0)
            elif signal_value < self.gp_config.sell_threshold:
                action = SignalAction.SELL
                confidence = min(abs(signal_value) / 2.0, 1.0)
            else:
                action = SignalAction.HOLD
                confidence = 0.1
            
            # Create signal
            signal = self._create_signal(
                symbol=symbol,
                action=action,
                price=latest['close'],
                confidence=confidence,
                metadata={
                    'gp_signal': signal_value,
                    'indicators': indicators,
                    'evolved_fitness': self.evolved_function.fitness,
                    'strategy_expression': self.evolved_function.individual_str
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating GP signal: {e}")
            return None
    
    async def _calculate_indicators(self, symbol: str, data) -> Dict[str, Any]:
        """Calculate technical indicators for the strategy"""
        try:
            close_prices = data['close']
            
            # Use TA-Lib if available, otherwise use fallback
            if TALIB_AVAILABLE:
                sma = talib.SMA(close_prices, timeperiod=self.gp_config.sma_period)[-1]
                ema = talib.EMA(close_prices, timeperiod=self.gp_config.ema_period)[-1]
                rsi = talib.RSI(close_prices, timeperiod=self.gp_config.rsi_period)[-1]
                macd, _, _ = talib.MACD(close_prices)
                macd_val = macd[-1]
            else:
                # Use fallback indicators
                sma = TechnicalIndicators.sma(close_prices, self.gp_config.sma_period).iloc[-1]
                ema = TechnicalIndicators.ema(close_prices, self.gp_config.ema_period).iloc[-1]
                rsi = TechnicalIndicators.rsi(close_prices, self.gp_config.rsi_period).iloc[-1]
                macd_val = TechnicalIndicators.macd(close_prices).iloc[-1]
            
            return {
                'sma': sma,
                'ema': ema,
                'rsi': rsi,
                'macd': macd_val
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return {
                'sma': 0.0,
                'ema': 0.0,
                'rsi': 50.0,
                'macd': 0.0
            }
    
    def create_primitive_set(self) -> gp.PrimitiveSet:
        """Creates the DEAP GP Primitive Set with functions and terminals."""
        # 9 input arguments: close, open, high, low, volume, sma, ema, rsi, macd
        pset = gp.PrimitiveSet("MAIN", 9, prefix="ARG")
        
        # Arithmetic Operators
        pset.addPrimitive(operator.add, 2, name="add")
        pset.addPrimitive(operator.sub, 2, name="sub")
        pset.addPrimitive(operator.mul, 2, name="mul")
        pset.addPrimitive(GPHelper.protected_div, 2, name="pdiv")
        pset.addPrimitive(operator.neg, 1, name="neg")
        pset.addPrimitive(abs, 1, name="abs")
        
        # Mathematical Functions
        pset.addPrimitive(np.sin, 1, name="sin")
        pset.addPrimitive(np.cos, 1, name="cos")
        pset.addPrimitive(np.tanh, 1, name="tanh")
        pset.addPrimitive(GPHelper.protected_log, 1, name="plog")
        pset.addPrimitive(GPHelper.protected_sqrt, 1, name="psqrt")
        pset.addPrimitive(GPHelper.protected_pow, 2, name="ppow")
        
        # Conditional and comparison functions
        pset.addPrimitive(GPHelper.if_then_else, 3, name="if_then_else")
        pset.addPrimitive(max, 2, name="max")
        pset.addPrimitive(min, 2, name="min")
        
        # Constants
        pset.addTerminal(0.0, name="zero")
        pset.addTerminal(1.0, name="one")
        pset.addTerminal(-1.0, name="neg_one")
        pset.addTerminal(0.5, name="half")
        
        # Ephemeral constants
        pset.addEphemeralConstant("rand_float", GPHelper.generate_random_uniform)
        
        # Rename arguments for clarity
        pset.renameArguments(
            ARG0='close',
            ARG1='open',
            ARG2='high', 
            ARG3='low',
            ARG4='volume',
            ARG5='sma',
            ARG6='ema',
            ARG7='rsi',
            ARG8='macd'
        )
        
        logger.info("🔧 GP Primitive Set created with enhanced operators")
        return pset
    
    def setup_deap(self) -> None:
        """Sets up DEAP creator classes for Fitness and Individual."""
        # Clear existing classes to avoid conflicts
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize Return
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        logger.info("🏗️ DEAP creator classes configured")
    
    def compile_individual(self, individual) -> Optional[Callable]:
        """Safely compiles a DEAP individual tree into a callable function."""
        try:
            compiled_func = gp.compile(individual, self.pset)
            return compiled_func
        except (MemoryError, RecursionError, Exception) as e:
            logger.debug(f"Error compiling individual: {e}")
            return None
    
    def evaluate_individual(self, individual, data: pd.DataFrame) -> Tuple[float]:
        """Evaluates a single GP individual using backtesting simulation."""
        gp_func = self.compile_individual(individual)
        if gp_func is None:
            return self.gp_config.worst_fitness
        
        try:
            # Simulate trading using the evolved function
            returns = self.simulate_trading(gp_func, data)
            
            if returns is None or np.isnan(returns):
                return self.gp_config.worst_fitness
            
            return (returns,)
            
        except Exception as e:
            logger.debug(f"Error evaluating individual: {e}")
            return self.gp_config.worst_fitness
    
    def simulate_trading(self, gp_function: Callable, data: pd.DataFrame) -> Optional[float]:
        """Simulates trading using the evolved function."""
        try:
            # Prepare indicators
            data = data.copy()
            
            if TALIB_AVAILABLE:
                data['SMA'] = talib.SMA(data['Close'], timeperiod=self.gp_config.sma_period)
                data['EMA'] = talib.EMA(data['Close'], timeperiod=self.gp_config.ema_period)
                data['RSI'] = talib.RSI(data['Close'], timeperiod=self.gp_config.rsi_period)
                macd, _, _ = talib.MACD(data['Close'])
                data['MACD'] = macd
            else:
                # Use fallback technical indicators
                data['SMA'] = TechnicalIndicators.sma(data['Close'], self.gp_config.sma_period)
                data['EMA'] = TechnicalIndicators.ema(data['Close'], self.gp_config.ema_period)
                data['RSI'] = TechnicalIndicators.rsi(data['Close'], self.gp_config.rsi_period)
                data['MACD'] = TechnicalIndicators.macd(data['Close'])
            
            # Remove initial NaN values
            data = data.dropna()
            
            if len(data) < 50:  # Minimum data requirement
                return None
            
            # Initialize trading simulation
            cash = self.gp_config.initial_cash
            position = 0
            position_value = 0
            
            for i in range(len(data)):
                row = data.iloc[i]
                
                # Get current values
                close = row['Close']
                open_price = row['Open']
                high = row['High']
                low = row['Low']
                volume = row['Volume']
                sma = row['SMA']
                ema = row['EMA']
                rsi = row['RSI']
                macd_val = row['MACD']
                
                # Skip if any indicator is NaN
                if any(np.isnan(x) for x in [sma, ema, rsi, macd_val]):
                    continue
                
                try:
                    # Get signal from evolved function
                    signal = gp_function(close, open_price, high, low, volume, 
                                       sma, ema, rsi, macd_val)
                    
                    # Execute trades based on signal
                    if signal > self.gp_config.buy_threshold and position == 0:
                        # Buy signal
                        shares_to_buy = int(cash / (close * (1 + self.gp_config.commission_pct)))
                        if shares_to_buy > 0:
                            position = shares_to_buy
                            position_value = shares_to_buy * close
                            cash -= position_value * (1 + self.gp_config.commission_pct)
                    
                    elif signal < self.gp_config.sell_threshold and position > 0:
                        # Sell signal
                        sell_value = position * close
                        cash += sell_value * (1 - self.gp_config.commission_pct)
                        position = 0
                        position_value = 0
                
                except Exception:
                    continue
            
            # Calculate final portfolio value
            final_value = cash
            if position > 0:
                final_value += position * data.iloc[-1]['Close']
            
            # Calculate returns
            total_return = ((final_value - self.gp_config.initial_cash) / self.gp_config.initial_cash) * 100
            return total_return
            
        except Exception as e:
            logger.debug(f"Error in trading simulation: {e}")
            return None
    
    def setup_toolbox(self, data: pd.DataFrame) -> base.Toolbox:
        """Sets up the DEAP toolbox with registered functions."""
        toolbox = base.Toolbox()
        
        # Attribute generator: expression (GP tree)
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset,
                        min_=self.gp_config.gp_min_depth, max_=self.gp_config.gp_max_depth)
        
        # Structure initializers
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Evaluation function
        toolbox.register("evaluate", self.evaluate_individual, data=data)
        
        # Genetic operators
        toolbox.register("select", tools.selTournament, tournsize=self.gp_config.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, 
                        min_=self.gp_config.gp_mut_min_depth, 
                        max_=self.gp_config.gp_mut_max_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        
        # Decorators for constraints
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), 
                                              max_value=self.gp_config.gp_max_height))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), 
                                                max_value=self.gp_config.gp_max_height))
        
        logger.info("🔧 DEAP toolbox configured")
        return toolbox
    
    def run_evolution(self, data: pd.DataFrame) -> Tuple[List, tools.Logbook, tools.HallOfFame]:
        """Runs the genetic programming evolutionary algorithm."""
        logger.info(f"🚀 Starting evolution: {self.gp_config.population_size} individuals, {self.gp_config.generations} generations")
        
        # Create initial population
        pop = self.toolbox.population(n=self.gp_config.population_size)
        hof = tools.HallOfFame(self.gp_config.hall_of_fame_size)
        
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
        
        # Setup multiprocessing if enabled
        if self.gp_config.max_workers != 1:
            try:
                num_workers = self.gp_config.max_workers or min(multiprocessing.cpu_count(), 8)
                logger.info(f"🔄 Using {num_workers} parallel workers")
                
                pool = multiprocessing.Pool(processes=num_workers)
                self.toolbox.register("map", pool.map)
                
                # Run evolution
                pop, logbook = algorithms.eaSimple(
                    pop, self.toolbox, 
                    cxpb=self.gp_config.crossover_prob,
                    mutpb=self.gp_config.mutation_prob,
                    ngen=self.gp_config.generations,
                    stats=mstats, 
                    halloffame=hof, 
                    verbose=True
                )
                
                pool.close()
                pool.join()
                self.toolbox.unregister("map")
                
            except Exception as e:
                logger.warning(f"Multiprocessing failed, falling back to single-threaded: {e}")
                # Fallback to single-threaded
                pop, logbook = algorithms.eaSimple(
                    pop, self.toolbox,
                    cxpb=self.gp_config.crossover_prob,
                    mutpb=self.gp_config.mutation_prob,
                    ngen=self.gp_config.generations,
                    stats=mstats,
                    halloffame=hof,
                    verbose=True
                )
        else:
            # Single-threaded execution
            pop, logbook = algorithms.eaSimple(
                pop, self.toolbox,
                cxpb=self.gp_config.crossover_prob,
                mutpb=self.gp_config.mutation_prob,
                ngen=self.gp_config.generations,
                stats=mstats,
                halloffame=hof,
                verbose=True
            )
        
        logger.info("✅ Evolution completed")
        return pop, logbook, hof
    
    def evolve_strategy(self, data: pd.DataFrame) -> bool:
        """Main method to evolve a trading strategy."""
        try:
            logger.info("🧬 Starting Genetic Programming evolution...")
            
            # Setup GP environment
            self.pset = self.create_primitive_set()
            self.setup_deap()
            self.toolbox = self.setup_toolbox(data)
            
            # Run evolution
            final_pop, logbook, hall_of_fame = self.run_evolution(data)
            
            # Store results
            self.hall_of_fame = hall_of_fame
            self.evolution_stats = {
                'final_population_size': len(final_pop),
                'generations': self.gp_config.generations,
                'best_fitness': hall_of_fame[0].fitness.values[0] if hall_of_fame else None,
                'evolution_time': datetime.now()
            }
            
            # Set best evolved function
            if hall_of_fame:
                best_individual = hall_of_fame[0]
                best_function = self.compile_individual(best_individual)
                
                if best_function:
                    self.evolved_function = EvolvedTradingFunction(
                        gp_function=best_function,
                        individual_str=str(best_individual),
                        fitness=best_individual.fitness.values[0]
                    )
                    self.is_evolved = True
                    
                    logger.info(f"🏆 Best strategy evolved with fitness: {best_individual.fitness.values[0]:.4f}%")
                    logger.info(f"📝 Strategy expression: {str(best_individual)}")
                    
                    # Save results
                    if self.gp_config.save_results:
                        self.save_evolution_results(logbook, hall_of_fame)
                    
                    return True
            
            logger.warning("❌ Evolution failed to produce viable strategies")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error during evolution: {e}")
            return False
    
    def save_evolution_results(self, logbook: tools.Logbook, hall_of_fame: tools.HallOfFame) -> None:
        """Saves evolution results to files."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save Hall of Fame
            if hall_of_fame:
                results = []
                for i, ind in enumerate(hall_of_fame):
                    results.append({
                        'Rank': i + 1,
                        'Fitness_Return_Pct': ind.fitness.values[0] if ind.fitness.valid else 'Invalid',
                        'Size': len(ind),
                        'Height': ind.height,
                        'Strategy_Expression': str(ind),
                        'Timestamp': timestamp
                    })
                
                results_df = pd.DataFrame(results)
                results_path = os.path.join(self.gp_config.results_dir, f"gp_hall_of_fame_{timestamp}.csv")
                results_df.to_csv(results_path, index=False)
                logger.info(f"💾 Hall of Fame saved to {results_path}")
            
            # Save evolution statistics
            if logbook:
                logbook_df = pd.DataFrame(logbook)
                logbook_path = os.path.join(self.gp_config.results_dir, f"gp_evolution_log_{timestamp}.csv")
                logbook_df.to_csv(logbook_path, index=False)
                logger.info(f"📊 Evolution log saved to {logbook_path}")
            
            # Save evolved model
            if self.gp_config.save_models and self.evolved_function:
                model_data = {
                    'evolved_function': self.evolved_function,
                    'config': self.gp_config,
                    'evolution_stats': self.evolution_stats,
                    'pset': self.pset
                }
                model_path = os.path.join(self.gp_config.results_dir, f"gp_model_{timestamp}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info(f"🤖 Evolved model saved to {model_path}")
                
        except Exception as e:
            logger.error(f"❌ Error saving evolution results: {e}")
    
    def load_evolved_strategy(self, model_path: str) -> bool:
        """Loads a previously evolved strategy."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.evolved_function = model_data['evolved_function']
            self.gp_config = model_data['config']
            self.evolution_stats = model_data['evolution_stats']
            self.pset = model_data['pset']
            self.is_evolved = True
            
            logger.info(f"✅ Loaded evolved strategy from {model_path}")
            logger.info(f"🏆 Strategy fitness: {self.evolved_function.fitness:.4f}%")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading evolved strategy: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Returns information about the strategy."""
        info = {
            'name': 'Genetic Programming Strategy',
            'description': 'Evolves trading strategies using genetic programming',
            'is_evolved': self.is_evolved,
            'config': {
                'population_size': self.gp_config.population_size,
                'generations': self.gp_config.generations,
                'crossover_prob': self.gp_config.crossover_prob,
                'mutation_prob': self.gp_config.mutation_prob
            }
        }
        
        if self.is_evolved and self.evolved_function:
            info.update({
                'evolved_fitness': self.evolved_function.fitness,
                'evolved_at': self.evolved_function.created_at.isoformat(),
                'strategy_expression': self.evolved_function.individual_str
            })
        
        if self.evolution_stats:
            info['evolution_stats'] = self.evolution_stats
        
        return info
    
    def get_name(self) -> str:
        return "GeneticProgrammingStrategy" 