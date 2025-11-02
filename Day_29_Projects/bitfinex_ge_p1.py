from __future__ import annotations

#!/usr/bin/env python3
"""
Bitfinex Genetic Programming Trading Strategy Evolution

Uses Genetic Programming (GP) with DEAP to evolve trading strategies
for Bitfinex trading pairs using backtesting.py and exchange-specific data.
Adapted for Bitfinex API integration and data formats.
"""

import numpy as np
import pandas as pd
import random
import operator
import talib
import warnings
import os
import requests
from typing import Tuple, List, Dict, Callable
from deap import algorithms, base, creator, tools, gp
from backtesting import Backtest, Strategy
from datetime import datetime

# Import Bitfinex configuration
try:
    from Day_26_Projects.bitfinex_config import (
        API_KEY, API_SECRET, PRIMARY_SYMBOL
    )
except ImportError:
    print("Warning: bitfinex_config not found, using default values")
    API_KEY = ""
    API_SECRET = ""
    PRIMARY_SYMBOL = "btcusd"

# --- Bitfinex Configuration ---
CONFIG = {
    # Data
    "DATA_PATH": f'bitfinex_{PRIMARY_SYMBOL}_1h_data.csv',
    "RESULTS_PATH": f'bitfinex_{PRIMARY_SYMBOL}_ge_results.csv',
    "SYMBOL": PRIMARY_SYMBOL,
    # Bitfinex API
    "API_BASE": "https://api.bitfinex.com",
    "DATA_LIMIT": 2000,  # Max candles per request
    # Technical Indicators
    "SMA_PERIOD": 20,
    "EMA_PERIOD": 20,
    "RSI_PERIOD": 14,
    # Backtesting
    "INITIAL_CASH": 1_000_000,
    "COMMISSION_PCT": 0.0025,  # Bitfinex trading fee ~0.25%
    # Genetic Programming
    "POPULATION_SIZE": 200,  # Smaller for faster iteration
    "GENERATIONS": 30,
    "CXPB": 0.5,  # Crossover probability
    "MUTPB": 0.2, # Mutation probability
    "TOURNAMENT_SIZE": 3,
    "HALL_OF_FAME_SIZE": 5,
    "GP_MIN_DEPTH": 1,
    "GP_MAX_DEPTH": 4,
    "GP_MUT_MIN_DEPTH": 0,
    "GP_MUT_MAX_DEPTH": 2,
    # Evaluation
    "WORST_FITNESS": (-float('inf'),) # Fitness is Return [%]
}

# Ignore runtime warnings (use with caution)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Bitfinex Data Fetcher ---

def fetch_bitfinex_candles(symbol: str, timeframe: str = "1h", limit: int = 2000) -> pd.DataFrame:
    """Fetch historical candle data from Bitfinex API."""
    try:
        # Bitfinex candles endpoint
        url = f"{CONFIG['API_BASE']}/v2/candles/trade:{timeframe}:t{symbol.upper()}/hist"
        
        params = {
            'limit': limit,
            'sort': 1  # Sort ascending (oldest first)
        }
        
        print(f"Fetching {limit} {timeframe} candles for {symbol}...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            raise ValueError(f"No data returned for {symbol}")
        
        # Bitfinex candle format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
        df = pd.DataFrame(data, columns=['timestamp', 'Open', 'Close', 'High', 'Low', 'Volume'])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        # Ensure proper order (oldest to newest)
        df = df.sort_index()
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df.dropna(inplace=True)
        
        print(f"Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        print(f"Error fetching Bitfinex data: {e}")
        raise

def save_bitfinex_data(symbol: str, filepath: str, timeframe: str = "1h"):
    """Fetch and save Bitfinex data to CSV."""
    try:
        df = fetch_bitfinex_candles(symbol, timeframe, CONFIG["DATA_LIMIT"])
        df.to_csv(filepath)
        print(f"Saved {len(df)} candles to {filepath}")
        return df
    except Exception as e:
        print(f"Error saving Bitfinex data: {e}")
        raise

# --- Helper Functions ---

def protected_div(left: float, right: float) -> float:
    """Safely divide, returning 1 for division by zero."""
    return left / right if right != 0 else 1.0

def protected_log(x: float) -> float:
    """Safely compute log, handling non-positive inputs."""
    return np.log(x) if x > 0 else 0.0

def generate_random_uniform() -> float:
    """Generates a random float between -1.0 and 1.0 for GP ephemeral constants."""
    return random.uniform(-1.0, 1.0)

# --- Core Functions ---

def load_and_prepare_data(filepath: str, fetch_fresh: bool = False) -> pd.DataFrame:
    """Loads data from CSV or fetches fresh from Bitfinex API."""
    
    # Option to fetch fresh data
    if fetch_fresh or not os.path.exists(filepath):
        print("Fetching fresh data from Bitfinex...")
        return save_bitfinex_data(CONFIG["SYMBOL"], filepath)
    
    # Load existing CSV data
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        # Load data
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded data shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")

        # Clean column names
        data.columns = data.columns.str.strip().str.capitalize()
        
        # Ensure required columns exist
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(set(data.columns)):
            missing = required_cols - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}. Available: {data.columns.tolist()}")

        # Convert to numeric
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Remove rows with NaN values
        initial_rows = len(data)
        data.dropna(inplace=True)
        print(f"Removed {initial_rows - len(data)} rows with NaNs.")
        
        if data.empty:
            raise ValueError("Data is empty after cleaning.")

        print("Data loaded and prepared successfully.")
        print(data.head())
        return data

    except Exception as e:
        print(f"Error loading or preparing data: {e}")
        raise

def create_primitive_set() -> gp.PrimitiveSet:
    """Creates the DEAP GP Primitive Set with functions and terminals."""
    pset = gp.PrimitiveSet("MAIN", 9, prefix="ARG") # 9 input arguments
    
    # Arithmetic Operators
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="pdiv")
    pset.addPrimitive(operator.neg, 1, name="neg")
    
    # Math Functions
    pset.addPrimitive(np.sin, 1, name="sin")
    pset.addPrimitive(np.cos, 1, name="cos")
    pset.addPrimitive(protected_log, 1, name="plog")
    pset.addPrimitive(np.sqrt, 1, name="sqrt")
    
    # Comparison and Logic
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")
    
    # Constants
    pset.addTerminal(1.0, name="const_1")
    pset.addTerminal(0.0, name="const_0")
    pset.addEphemeralConstant("rand101", generate_random_uniform)
    
    print("Bitfinex GP primitive set created.")
    return pset

def setup_deap() -> None:
    """Sets up DEAP creator classes for Fitness and Individual."""
    # Clear existing creator classes if they exist
    if hasattr(creator, "FitnessMax"):
        delattr(creator, "FitnessMax")
    if hasattr(creator, "Individual"):
        delattr(creator, "Individual")
        
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize Return [%]
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    print("DEAP creator classes (FitnessMax, Individual) set up for Bitfinex.")

# --- Strategy and Evaluation ---

class BitfinexEvolvedStrategy(Strategy):
    """Bitfinex-specific strategy class incorporating the evolved GP function."""
    gp_function: Callable = None
    
    # Strategy parameters
    sma_period = CONFIG["SMA_PERIOD"]
    ema_period = CONFIG["EMA_PERIOD"]
    rsi_period = CONFIG["RSI_PERIOD"]

    def init(self):
        if self.gp_function is None:
            raise ValueError("GP function not set for BitfinexEvolvedStrategy")
            
        # Precompute indicators
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # MACD
        macd_result = self.I(talib.MACD, self.data.Close)
        self.macd = macd_result[0] if isinstance(macd_result, tuple) else macd_result

    def next(self):
        # Ensure enough data points are available
        min_periods = max(self.sma_period, self.ema_period, self.rsi_period) + 5
        if len(self.data) < min_periods:
            return

        # Get latest values
        try:
            close = self.data.Close[-1]
            open_ = self.data.Open[-1]
            high = self.data.High[-1]
            low = self.data.Low[-1]
            volume = self.data.Volume[-1]
            sma = self.sma[-1]
            ema = self.ema[-1]
            rsi = self.rsi[-1]
            macd = self.macd[-1]

            # Check for NaN values
            values = [close, open_, high, low, volume, sma, ema, rsi, macd]
            if any(np.isnan(x) or x is None for x in values):
                return

            # Execute GP function
            signal = self.gp_function(close, open_, high, low, volume, sma, ema, rsi, macd)

            # Trading logic based on signal
            if signal > 0.6:  # Strong buy signal
                if not self.position:
                    self.buy()
            elif signal < -0.6:  # Strong sell signal
                if self.position:
                    self.sell()
            elif abs(signal) < 0.1:  # Neutral signal - close positions
                if self.position:
                    self.sell()

        except (OverflowError, ValueError, TypeError, ZeroDivisionError, ArithmeticError):
            # Handle runtime errors silently
            pass
        except Exception as e:
            print(f"Unexpected error in Bitfinex strategy: {e}")
            pass

def compile_individual(individual: creator.Individual, pset: gp.PrimitiveSet) -> Callable | None:
    """Safely compiles a DEAP individual tree into a callable function."""
    try:
        return gp.compile(individual, pset)
    except Exception as e:
        return None

def evaluate_strategy(individual: creator.Individual, pset: gp.PrimitiveSet, data: pd.DataFrame, config: Dict) -> Tuple[float]:
    """Evaluates a single GP individual using backtesting on Bitfinex data."""
    gp_func = compile_individual(individual, pset)
    if gp_func is None:
        return config["WORST_FITNESS"]

    # Set the compiled function in the strategy class
    BitfinexEvolvedStrategy.gp_function = gp_func

    try:
        # Run backtest
        bt = Backtest(
            data, 
            BitfinexEvolvedStrategy,
            cash=config["INITIAL_CASH"],
            commission=config["COMMISSION_PCT"],
            exclusive_orders=True
        )
        
        stats = bt.run()
        fitness = stats.get('Return [%]', None)

        # Validate fitness
        if fitness is None or np.isnan(fitness) or np.isinf(fitness):
            return config["WORST_FITNESS"]

        # Penalize overly complex strategies
        complexity_penalty = len(individual) * 0.01
        adjusted_fitness = fitness - complexity_penalty

        return (adjusted_fitness,)

    except Exception as e:
        return config["WORST_FITNESS"]

def setup_toolbox(pset: gp.PrimitiveSet, data: pd.DataFrame, config: Dict) -> base.Toolbox:
    """Sets up the DEAP toolbox with registered functions."""
    toolbox = base.Toolbox()

    # Expression generator
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=config["GP_MIN_DEPTH"], max_=config["GP_MAX_DEPTH"])

    # Individual and population
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation
    toolbox.register("evaluate", evaluate_strategy, pset=pset, data=data, config=config)

    # Genetic operators
    toolbox.register("select", tools.selTournament, tournsize=config["TOURNAMENT_SIZE"])
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=config["GP_MUT_MIN_DEPTH"], max_=config["GP_MUT_MAX_DEPTH"])
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Constraints
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=15))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=15))

    print("Bitfinex DEAP toolbox configured.")
    return toolbox

# --- Evolution ---

def run_evolution(toolbox: base.Toolbox, config: Dict) -> Tuple[List, tools.Logbook, tools.HallOfFame]:
    """Runs the genetic programming evolutionary algorithm for Bitfinex strategies."""
    pop_size = config["POPULATION_SIZE"]
    ngen = config["GENERATIONS"]
    cxpb = config["CXPB"]
    mutpb = config["MUTPB"]
    hof_size = config["HALL_OF_FAME_SIZE"]

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(hof_size)

    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else float('-inf'))
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + mstats.fields

    print(f"Starting Bitfinex evolution: {pop_size} individuals, {ngen} generations...")
    
    try:
        # Use simple algorithm without multiprocessing for better Windows compatibility
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
            stats=mstats, halloffame=hof, verbose=True
        )
        print("Bitfinex evolution completed successfully.")
        
    except Exception as e:
        print(f"Error during Bitfinex evolution: {e}")
        return pop, logbook, hof

    return pop, logbook, hof

def save_results(hof: tools.HallOfFame, filepath: str, symbol: str) -> None:
    """Saves the Hall of Fame individuals to a CSV file."""
    if not hof:
        print("Hall of Fame is empty. No results to save.")
        return

    results = []
    for i, ind in enumerate(hof):
        try:
            fitness_val = ind.fitness.values[0] if ind.fitness.valid else 'Invalid'
            results.append({
                'Exchange': 'Bitfinex',
                'Symbol': symbol.upper(),
                'Rank': i + 1,
                'Fitness (Return %)': fitness_val,
                'Strategy Size': len(ind),
                'Timestamp': datetime.now().isoformat(),
                'Strategy Expression': str(ind)[:500]  # Truncate very long expressions
            })
        except Exception as e:
            print(f"Error processing individual {i} for saving: {e}")

    if not results:
        print("No valid individuals in Hall of Fame to save.")
        return

    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(filepath, index=False)
        print(f"Bitfinex Hall of Fame results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    print("=" * 60)
    print(f"BITFINEX GENETIC PROGRAMMING STRATEGY EVOLUTION")
    print(f"Symbol: {CONFIG['SYMBOL'].upper()}")
    print(f"Population: {CONFIG['POPULATION_SIZE']} | Generations: {CONFIG['GENERATIONS']}")
    print("=" * 60)
    
    try:
        # 1. Load or fetch data
        fetch_fresh = input("Fetch fresh data from Bitfinex? (y/n): ").lower().startswith('y')
        data = load_and_prepare_data(CONFIG["DATA_PATH"], fetch_fresh=fetch_fresh)
        
        if len(data) < 1000:
            print(f"Warning: Only {len(data)} data points available. Consider fetching more data.")
        
        # 2. Setup DEAP GP Environment
        pset = create_primitive_set()
        setup_deap()
        toolbox = setup_toolbox(pset, data, CONFIG)

        # 3. Run Evolution
        final_pop, logbook, hall_of_fame = run_evolution(toolbox, CONFIG)

        # 4. Process and Save Results
        print("\n" + "="*50)
        print("EVOLUTION SUMMARY")
        print("="*50)
        print(logbook)

        print("\n" + "="*50)
        print("HALL OF FAME - BEST STRATEGIES")
        print("="*50)
        if hall_of_fame:
            for i, individual in enumerate(hall_of_fame):
                fitness = individual.fitness.values[0] if individual.fitness.valid else "Invalid"
                print(f"\nRank {i+1}: {CONFIG['SYMBOL'].upper()}")
                print(f"  Return: {fitness:.4f}%")
                print(f"  Complexity: {len(individual)} nodes")
                print(f"  Expression: {str(individual)[:200]}...")
                print("-" * 40)
                
            save_results(hall_of_fame, CONFIG["RESULTS_PATH"], CONFIG["SYMBOL"])
        else:
            print("No individuals in Hall of Fame.")

        print(f"\n🎯 Bitfinex GP Evolution Complete!")
        print(f"📊 Best Return: {hall_of_fame[0].fitness.values[0]:.4f}%" if hall_of_fame else "N/A")

    except KeyboardInterrupt:
        print("\n⚠️  Evolution interrupted by user.")
    except FileNotFoundError as e:
        print(f"❌ Critical Error: {e}")
    except ValueError as e:
        print(f"❌ Critical Data Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()
