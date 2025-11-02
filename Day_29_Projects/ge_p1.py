from __future__ import annotations

#!/usr/bin/env python3
"""
Uses Genetic Programming (GP) with DEAP to evolve trading strategies
for BTC-USD 1-hour data using backtesting.py.
Reference: https://www.youtube.com/watch?v=1uz-2lvFoe4&t=17142s
"""

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools, gp
from backtesting import Backtest, Strategy
import random
import operator
import talib
import multiprocessing
import warnings
import os
from typing import Tuple, List, Dict, Callable

# --- Configuration ---
CONFIG = {
    # Data
    "DATA_PATH": 'BTC-1h-100wks-data.csv',
    "RESULTS_PATH": 'grammatical_evolution_results.csv',
    # Technical Indicators
    "SMA_PERIOD": 20,
    "EMA_PERIOD": 20,
    "RSI_PERIOD": 14,
    # Backtesting
    "INITIAL_CASH": 1_000_000,
    "COMMISSION_PCT": 0.002,
    # Genetic Programming
    "POPULATION_SIZE": 300,
    "GENERATIONS": 40,
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

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Loads data, cleans column names, converts types, handles date column, and drops NaNs."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        # Load data without initial date parsing/indexing
        data = pd.read_csv(filepath)
        print(f"Loaded initial data shape: {data.shape}")
        print(f"Initial columns: {data.columns.tolist()}")

        # --- Find and process the date column ---
        date_col_found = None
        possible_date_cols = ['datetime', 'date', 'timestamp', 'time']
        # Store original column names before potential renaming
        original_columns = {col.lower(): col for col in data.columns}

        for potential_name in possible_date_cols:
            if potential_name in original_columns:
                date_col_found = original_columns[potential_name]
                print(f"Identified date column as: '{date_col_found}'")
                break

        if not date_col_found:
            raise ValueError(f"Could not find a suitable date column {possible_date_cols}. Found: {data.columns.tolist()}")

        # Parse dates and set index
        print(f"Parsing dates in column: '{date_col_found}'")
        data[date_col_found] = pd.to_datetime(data[date_col_found], errors='coerce')
        # Drop rows where date parsing failed *before* setting index
        data.dropna(subset=[date_col_found], inplace=True)
        if data.empty:
             raise ValueError(f"Data empty after dropping rows with invalid dates in '{date_col_found}'.")
        
        data.set_index(date_col_found, inplace=True)
        print(f"Set '{date_col_found}' as index.")
        # --- End Date Column Processing ---

        # Clean other column names (make sure not to re-capitalize the index)
        data.columns = data.columns.str.strip().str.capitalize()
        print(f"Cleaned column names: {data.columns.tolist()}")

        # Ensure required columns exist (case-insensitive check on cleaned names)
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        current_cols_lower = {col.lower() for col in data.columns}
        required_cols_lower = {col.lower() for col in required_cols}
        
        if not required_cols_lower.issubset(current_cols_lower):
            missing = required_cols_lower - current_cols_lower
            # Map back to original-case required names for clarity
            missing_original_case = {req for req in required_cols if req.lower() in missing}
            raise ValueError(f"Missing required columns after cleaning: {missing_original_case}. Available: {data.columns.tolist()}")

        # Convert price and volume columns to numeric
        # Find actual column names corresponding to required cols (case-insensitive)
        col_mapping = {req.lower(): actual for req in required_cols for actual in data.columns if actual.lower() == req.lower()}
        
        for req_lower in required_cols_lower:
            actual_col_name = col_mapping[req_lower]
            data[actual_col_name] = pd.to_numeric(data[actual_col_name], errors='coerce')
            print(f"Converted column '{actual_col_name}' to numeric.")

        # Remove any rows with NaN values in numeric columns
        numeric_cols_to_check = [col_mapping[req_lower] for req_lower in required_cols_lower]
        initial_rows = len(data)
        data.dropna(subset=numeric_cols_to_check, inplace=True)
        print(f"Removed {initial_rows - len(data)} rows with NaNs in numeric columns.")
        print(f"Final data shape: {data.shape}")

        if data.empty:
            raise ValueError("Data is empty after cleaning numeric columns.")

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
    # Math Functions (consider adding more if needed)
    pset.addPrimitive(np.sin, 1, name="sin")
    pset.addPrimitive(np.cos, 1, name="cos")
    pset.addPrimitive(protected_log, 1, name="plog")
    # Terminals (Inputs to the strategy)
    # Arguments are automatically named ARG0, ARG1, ...
    # pset.renameArguments(ARG0='close') # Renaming done later for clarity if needed
    # Add a constant terminal
    pset.addTerminal(1.0, name="const_1") # Example constant
    # Add an ephemeral constant using the globally defined function
    pset.addEphemeralConstant("rand101", generate_random_uniform)
    print("Primitive set created.")
    return pset

def setup_deap() -> None:
    """Sets up DEAP creator classes for Fitness and Individual."""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize Return [%]
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    print("DEAP creator classes (FitnessMax, Individual) set up.")

# --- Strategy and Evaluation ---
# Define strategy class outside evaluation function
class EvolvedStrategy(Strategy):
    """ Strategy class incorporating the evolved GP function and TIs."""
    gp_function: Callable = None # To be set dynamically

    # Strategy parameters (can be optimized too, but fixed for now)
    sma_period = CONFIG["SMA_PERIOD"]
    ema_period = CONFIG["EMA_PERIOD"]
    rsi_period = CONFIG["RSI_PERIOD"]

    def init(self):
        if self.gp_function is None:
            raise ValueError("GP function not set for EvolvedStrategy")
        # Precompute indicators
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        macd_full = self.I(talib.MACD, self.data.Close) # MACD returns macd, signal, hist
        self.macd = macd_full[0] # Use the MACD line

    def next(self):
        # Ensure enough data points are available for indicators
        if len(self.data) < max(self.sma_period, self.ema_period, self.rsi_period) + 2:
             return

        # Get latest values (handle potential index issues if indicators lag)
        # Use .iloc[-1] for safety if direct indexing might fail near start
        close = self.data.Close[-1]
        open_ = self.data.Open[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]
        sma = self.sma[-1]
        ema = self.ema[-1]
        rsi = self.rsi[-1]
        macd = self.macd[-1]

        # Check for NaN in indicators (can happen at the beginning)
        if any(np.isnan(x) for x in [sma, ema, rsi, macd]):
            return

        try:
            # Execute the evolved GP function
            signal = self.gp_function(close, open_, high, low, volume, sma, ema, rsi, macd)

            # Simple Signal Interpretation
            if signal > 0.5: # Threshold for buying
                if not self.position:
                    self.buy()
            elif signal < -0.5: # Threshold for selling
                 if self.position:
                    self.sell()
            # Consider adding logic for closing positions if signal is neutral (e.g., between -0.5 and 0.5)

        except (OverflowError, ValueError, TypeError, ZeroDivisionError) as e:
            # Log error but continue simulation (assigns worst fitness later)
            # print(f"Runtime error in strategy evaluation: {e}")
            pass # Error handled by returning worst fitness in evaluate_strategy
        except Exception as e:
            # Catch unexpected errors
            print(f"Unexpected runtime error in strategy: {e}")
            pass # Assign worst fitness

def compile_individual(individual: creator.Individual, pset: gp.PrimitiveSet) -> Callable | None:
    """Safely compiles a DEAP individual tree into a callable function."""
    try:
        return gp.compile(individual, pset)
    except (MemoryError, RecursionError, Exception) as e:
        print(f"Error compiling individual {individual}: {e}")
        return None # Indicate compilation failure

def evaluate_strategy(individual: creator.Individual, pset: gp.PrimitiveSet, data: pd.DataFrame, config: Dict) -> Tuple[float]:
    """Evaluates a single GP individual using backtesting."""
    gp_func = compile_individual(individual, pset)
    if gp_func is None:
        return config["WORST_FITNESS"] # Return worst fitness if compilation fails

    # Dynamically set the compiled function in the strategy class
    current_strategy = EvolvedStrategy
    current_strategy.gp_function = gp_func

    try:
        # Run backtest
        bt = Backtest(data, current_strategy,
                      cash=config["INITIAL_CASH"],
                      commission=config["COMMISSION_PCT"])
        stats = bt.run()

        # Extract fitness (Return [%])
        fitness = stats.get('Return [%]', None)

        # Handle cases where backtest might fail or return no stats
        if fitness is None or np.isnan(fitness):
             # print(f"Warning: Backtest failed or returned NaN fitness for individual. Assigning worst fitness.")
             return config["WORST_FITNESS"]

        return (fitness,) # Return as a tuple

    except Exception as e:
        # Catch errors during Backtest instantiation or run
        print(f"Error during backtesting individual: {e}")
        return config["WORST_FITNESS"]

def setup_toolbox(pset: gp.PrimitiveSet, data: pd.DataFrame, config: Dict) -> base.Toolbox:
    """Sets up the DEAP toolbox with registered functions."""
    toolbox = base.Toolbox()

    # Attribute generator: expression (GP tree)
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                     min_=config["GP_MIN_DEPTH"], max_=config["GP_MAX_DEPTH"])

    # Structure initializers: individual and population
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function (use functools.partial to pass fixed args)
    toolbox.register("evaluate", evaluate_strategy, pset=pset, data=data, config=config)

    # Genetic operators
    toolbox.register("select", tools.selTournament, tournsize=config["TOURNAMENT_SIZE"])
    toolbox.register("mate", gp.cxOnePoint) # Crossover
    toolbox.register("expr_mut", gp.genFull, min_=config["GP_MUT_MIN_DEPTH"], max_=config["GP_MUT_MAX_DEPTH"])
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # Mutation

    # Decorators for constraints (optional but recommended)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    print("DEAP toolbox configured.")
    return toolbox

# --- Evolution ---

def run_evolution(toolbox: base.Toolbox, config: Dict) -> Tuple[List, tools.Logbook, tools.HallOfFame]:
    """Runs the genetic programming evolutionary algorithm."""
    pop_size = config["POPULATION_SIZE"]
    ngen = config["GENERATIONS"]
    cxpb = config["CXPB"]
    mutpb = config["MUTPB"]
    hof_size = config["HALL_OF_FAME_SIZE"]

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(hof_size)

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

    # Setup multiprocessing pool
    start_method = 'fork' if hasattr(os, 'fork') else None
    mp_context = multiprocessing.get_context(start_method)
    print(f"Using multiprocessing context: {mp_context.get_start_method()}")

    # Explicitly limit pool size for Windows compatibility
    num_workers = min(60, os.cpu_count() if os.cpu_count() else 4) # Limit to 60 or CPU cores
    print(f"Limiting multiprocessing pool to {num_workers} workers.")
    pool = mp_context.Pool(processes=num_workers) # Set the number of processes
    toolbox.register("map", pool.map)

    try:
        # Run the algorithm
        print(f"Starting evolution: {pop_size} individuals, {ngen} generations...")
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                           stats=mstats, halloffame=hof, verbose=True)
        print("Evolution finished.")

    except Exception as e:
        print(f"Error during evolution: {e}")
    finally:
        # Ensure pool is closed
        pool.close()
        pool.join()
        print("Multiprocessing pool closed.")
        # Unregister map to avoid issues if toolbox is reused
        toolbox.unregister("map")


    return pop, logbook, hof

def save_results(hof: tools.HallOfFame, filepath: str) -> None:
    """Saves the Hall of Fame individuals to a CSV file."""
    if not hof:
        print("Hall of Fame is empty. No results to save.")
        return

    results = []
    for i, ind in enumerate(hof):
        try:
            fitness_val = ind.fitness.values[0] if ind.fitness.valid else 'Invalid'
            results.append({
                'Rank': i + 1,
                'Fitness (Return %)': fitness_val,
                'Size': len(ind),
                'Strategy Tree': str(ind)
            })
        except Exception as e:
             print(f"Error processing individual {i} for saving: {e}")

    if not results:
        print("No valid individuals in Hall of Fame to save.")
        return

    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(filepath, index=False)
        print(f"Hall of Fame results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Grammatical Evolution Trading Strategy ---")
    try:
        # 1. Load Data
        data = load_and_prepare_data(CONFIG["DATA_PATH"])

        # 2. Setup DEAP GP Environment
        pset = create_primitive_set()
        setup_deap() # Creates FitnessMax, Individual
        toolbox = setup_toolbox(pset, data, CONFIG)

        # 3. Run Evolution
        final_pop, logbook, hall_of_fame = run_evolution(toolbox, CONFIG)

        # 4. Process and Save Results
        print("\n--- Evolution Summary ---")
        print(logbook)

        print("\n--- Hall of Fame ---")
        if hall_of_fame:
            for i, individual in enumerate(hall_of_fame):
                print(f"Rank {i+1}:")
                print(f"  Fitness: {individual.fitness.values[0]:.4f}%")
                print(f"  Size: {len(individual)}")
                print(f"  Expression: {individual}")
                print("-" * 20)
            save_results(hall_of_fame, CONFIG["RESULTS_PATH"])
        else:
            print("No individuals in Hall of Fame.")

        print("\n--- Script Finished ---")

    except FileNotFoundError as e:
        print(f"Critical Error: {e}")
    except ValueError as e:
        print(f"Critical Data Error: {e}")
    except Exception as e:
        print(f"An unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()