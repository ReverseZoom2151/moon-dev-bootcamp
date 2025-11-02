from __future__ import annotations

#!/usr/bin/env python3
"""
Binance Genetic Programming Trading Strategy Evolution

Uses Genetic Programming (GP) with DEAP to evolve trading strategies
for Binance trading pairs using backtesting.py and exchange-specific data.
Adapted for Binance API integration and data formats.
"""

import numpy as np
import pandas as pd
import random
import operator
import talib
import warnings
import os
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
from typing import Tuple, List, Dict, Callable
from datetime import datetime
from deap import algorithms, base, creator, tools, gp
from backtesting import Backtest, Strategy

# Import Binance configuration
try:
    from Day_26_Projects.binance_config import (
        API_KEY, API_SECRET, PRIMARY_SYMBOL
    )
except ImportError:
    print("Warning: binance_config not found, using default values")
    API_KEY = ""
    API_SECRET = ""
    PRIMARY_SYMBOL = "BTCUSDT"

# --- Binance Configuration ---
CONFIG = {
    # Data
    "DATA_PATH": f'binance_{PRIMARY_SYMBOL}_1h_data.csv',
    "RESULTS_PATH": f'binance_{PRIMARY_SYMBOL}_ge_results.csv',
    "SYMBOL": PRIMARY_SYMBOL,
    # Binance API
    "API_BASE": "https://api.binance.com/api/v3",
    "DATA_LIMIT": 1000,  # Max 1000 klines per request
    # Technical Indicators
    "SMA_PERIOD": 20,
    "EMA_PERIOD": 20,
    "RSI_PERIOD": 14,
    # Backtesting
    "INITIAL_CASH": 1_000_000,
    "COMMISSION_PCT": 0.001,  # Binance trading fee ~0.1%
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

# --- Binance Data Fetcher ---

class BinanceAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = CONFIG["API_BASE"].replace("/api/v3", "")
        self.headers = {"X-MBX-APIKEY": self.api_key}
    
    def _get_timestamp(self):
        return int(time.time() * 1000)
    
    def _sign(self, params):
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Fetch historical kline data from Binance API."""
    try:
        url = f"{CONFIG['API_BASE'].replace('/api/v3', '')}/api/v3/klines"
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        print(f"Fetching {limit} {interval} klines for {symbol}...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            raise ValueError(f"No data returned for {symbol}")
        
        # Binance kline format: [
        #   0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume,
        #   6: Close time, 7: Quote asset volume, 8: Number of trades,
        #   9: Taker buy base asset volume, 10: Taker buy quote asset volume, 11: Ignore
        # ]
        df = pd.DataFrame(data, columns=[
            'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Keep only OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df.dropna(inplace=True)
        
        print(f"Fetched {len(df)} klines from {df.index[0]} to {df.index[-1]}")
        return df
        
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        raise

def save_binance_data(symbol: str, filepath: str, interval: str = "1h"):
    """Fetch and save Binance data to CSV."""
    try:
        df = fetch_binance_klines(symbol, interval, CONFIG["DATA_LIMIT"])
        df.to_csv(filepath)
        print(f"Saved {len(df)} klines to {filepath}")
        return df
    except Exception as e:
        print(f"Error saving Binance data: {e}")
        raise

# --- Helper Functions ---

def protected_div(left: float, right: float) -> float:
    """Safely divide, returning 1 for division by zero."""
    return left / right if right != 0 else 1.0

def protected_log(x: float) -> float:
    """Safely compute log, handling non-positive inputs."""
    return np.log(x) if x > 0 else 0.0

def protected_sqrt(x: float) -> float:
    """Safely compute square root, handling negative inputs."""
    return np.sqrt(abs(x))

def generate_random_uniform() -> float:
    """Generates a random float between -1.0 and 1.0 for GP ephemeral constants."""
    return random.uniform(-1.0, 1.0)

# --- Core Functions ---

def load_and_prepare_data(filepath: str, fetch_fresh: bool = False) -> pd.DataFrame:
    """Loads data from CSV or fetches fresh from Binance API."""
    
    # Option to fetch fresh data
    if fetch_fresh or not os.path.exists(filepath):
        print("Fetching fresh data from Binance...")
        return save_binance_data(CONFIG["SYMBOL"], filepath)
    
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
    pset.addPrimitive(operator.abs, 1, name="abs")
    
    # Math Functions
    pset.addPrimitive(np.sin, 1, name="sin")
    pset.addPrimitive(np.cos, 1, name="cos")
    pset.addPrimitive(np.tanh, 1, name="tanh")
    pset.addPrimitive(protected_log, 1, name="plog")
    pset.addPrimitive(protected_sqrt, 1, name="psqrt")
    
    # Comparison and Logic
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")
    
    # Constants
    pset.addTerminal(1.0, name="const_1")
    pset.addTerminal(0.0, name="const_0")
    pset.addTerminal(-1.0, name="const_neg1")
    pset.addEphemeralConstant("rand101", generate_random_uniform)
    
    print("Binance GP primitive set created.")
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
    print("DEAP creator classes (FitnessMax, Individual) set up for Binance.")

# --- Strategy and Evaluation ---

class BinanceEvolvedStrategy(Strategy):
    """Binance-specific strategy class incorporating the evolved GP function."""
    gp_function: Callable = None
    
    # Strategy parameters
    sma_period = CONFIG["SMA_PERIOD"]
    ema_period = CONFIG["EMA_PERIOD"]
    rsi_period = CONFIG["RSI_PERIOD"]

    def init(self):
        if self.gp_function is None:
            raise ValueError("GP function not set for BinanceEvolvedStrategy")
            
        # Precompute indicators
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # MACD
        macd_result = self.I(talib.MACD, self.data.Close)
        self.macd = macd_result[0] if isinstance(macd_result, tuple) else macd_result
        
        # Bollinger Bands
        bb_result = self.I(talib.BBANDS, self.data.Close, timeperiod=20)
        self.bb_upper = bb_result[0] if isinstance(bb_result, tuple) else bb_result
        self.bb_middle = bb_result[1] if isinstance(bb_result, tuple) and len(bb_result) > 1 else self.sma
        self.bb_lower = bb_result[2] if isinstance(bb_result, tuple) and len(bb_result) > 2 else self.sma

    def next(self):
        # Ensure enough data points are available
        min_periods = max(self.sma_period, self.ema_period, self.rsi_period) + 10
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

            # Normalize signal to prevent extreme values
            if np.isnan(signal) or np.isinf(signal):
                signal = 0
            else:
                signal = np.clip(signal, -10, 10)  # Clip to reasonable range

            # Trading logic based on signal
            if signal > 0.7:  # Strong buy signal
                if not self.position:
                    self.buy()
            elif signal < -0.7:  # Strong sell signal
                if self.position:
                    self.sell()
            elif abs(signal) < 0.2:  # Neutral signal - close positions
                if self.position:
                    self.sell()

        except (OverflowError, ValueError, TypeError, ZeroDivisionError, ArithmeticError):
            # Handle runtime errors silently
            pass
        except Exception as e:
            # Log unexpected errors but continue
            pass

def compile_individual(individual: creator.Individual, pset: gp.PrimitiveSet) -> Callable | None:
    """Safely compiles a DEAP individual tree into a callable function."""
    try:
        return gp.compile(individual, pset)
    except Exception:
        return None

def evaluate_strategy(individual: creator.Individual, pset: gp.PrimitiveSet, data: pd.DataFrame, config: Dict) -> Tuple[float]:
    """Evaluates a single GP individual using backtesting on Binance data."""
    gp_func = compile_individual(individual, pset)
    if gp_func is None:
        return config["WORST_FITNESS"]

    # Set the compiled function in the strategy class
    BinanceEvolvedStrategy.gp_function = gp_func

    try:
        # Run backtest with Binance-specific settings
        bt = Backtest(
            data, 
            BinanceEvolvedStrategy,
            cash=config["INITIAL_CASH"],
            commission=config["COMMISSION_PCT"],
            exclusive_orders=True
        )
        
        stats = bt.run()
        
        # Get primary fitness metric
        fitness = stats.get('Return [%]', None)

        # Validate fitness
        if fitness is None or np.isnan(fitness) or np.isinf(fitness):
            return config["WORST_FITNESS"]

        # Multi-objective fitness with additional metrics
        sharpe_ratio = stats.get('Sharpe Ratio', 0)
        max_drawdown = stats.get('Max. Drawdown [%]', 0)
        
        # Penalize strategies with high drawdown
        drawdown_penalty = abs(max_drawdown) * 0.1
        
        # Bonus for good Sharpe ratio
        sharpe_bonus = max(0, sharpe_ratio) * 2
        
        # Complexity penalty
        complexity_penalty = len(individual) * 0.005
        
        # Combined fitness
        adjusted_fitness = fitness + sharpe_bonus - drawdown_penalty - complexity_penalty

        return (adjusted_fitness,)

    except Exception:
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

    # Constraints to prevent bloat
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=12))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=12))

    print("Binance DEAP toolbox configured.")
    return toolbox

# --- Evolution ---

def run_evolution(toolbox: base.Toolbox, config: Dict) -> Tuple[List, tools.Logbook, tools.HallOfFame]:
    """Runs the genetic programming evolutionary algorithm for Binance strategies."""
    pop_size = config["POPULATION_SIZE"]
    ngen = config["GENERATIONS"]
    cxpb = config["CXPB"]
    mutpb = config["MUTPB"]
    hof_size = config["HALL_OF_FAME_SIZE"]

    # Initialize population and hall of fame
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(hof_size)

    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else float('-inf'))
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.nanmean)
    mstats.register("std", np.nanstd)
    mstats.register("min", np.nanmin)
    mstats.register("max", np.nanmax)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + mstats.fields

    print(f"Starting Binance evolution: {pop_size} individuals, {ngen} generations...")
    print(f"Target: Maximize Return [%] with risk-adjusted metrics")
    
    try:
        # Use simple evolutionary algorithm for reliability
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
            stats=mstats, halloffame=hof, verbose=True
        )
        print("Binance evolution completed successfully.")
        
    except Exception as e:
        print(f"Error during Binance evolution: {e}")
        import traceback
        traceback.print_exc()
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
                'Exchange': 'Binance',
                'Symbol': symbol.upper(),
                'Rank': i + 1,
                'Fitness (Return %)': fitness_val,
                'Strategy Size': len(ind),
                'Complexity Score': len(ind) * 0.005,
                'Timestamp': datetime.now().isoformat(),
                'Strategy Expression': str(ind)[:1000]  # Truncate very long expressions
            })
        except Exception as e:
            print(f"Error processing individual {i} for saving: {e}")

    if not results:
        print("No valid individuals in Hall of Fame to save.")
        return

    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(filepath, index=False)
        print(f"Binance Hall of Fame results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

def test_best_strategy(hof: tools.HallOfFame, data: pd.DataFrame) -> None:
    """Test the best strategy and show detailed results."""
    if not hof:
        print("No strategies to test.")
        return
    
    try:
        best_individual = hof[0]
        print(f"\n🧪 Testing Best Strategy...")
        print(f"Expression: {str(best_individual)[:200]}...")
        
        # Compile and set the strategy
        pset = create_primitive_set()
        gp_func = compile_individual(best_individual, pset)
        if gp_func is None:
            print("❌ Could not compile best strategy.")
            return
            
        BinanceEvolvedStrategy.gp_function = gp_func
        
        # Run detailed backtest
        bt = Backtest(data, BinanceEvolvedStrategy,
                      cash=CONFIG["INITIAL_CASH"],
                      commission=CONFIG["COMMISSION_PCT"])
        
        stats = bt.run()
        
        print(f"\n📊 DETAILED BACKTEST RESULTS")
        print(f"{'='*40}")
        print(f"Return [%]:           {stats.get('Return [%]', 'N/A'):.2f}%")
        print(f"Sharpe Ratio:         {stats.get('Sharpe Ratio', 'N/A'):.3f}")
        print(f"Max Drawdown [%]:     {stats.get('Max. Drawdown [%]', 'N/A'):.2f}%")
        print(f"Win Rate [%]:         {stats.get('Win Rate [%]', 'N/A'):.1f}%")
        print(f"# Trades:             {stats.get('# Trades', 'N/A')}")
        print(f"Avg Trade [%]:        {stats.get('Avg. Trade [%]', 'N/A'):.3f}%")
        
    except Exception as e:
        print(f"❌ Error testing best strategy: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    print("=" * 65)
    print(f"🚀 BINANCE GENETIC PROGRAMMING STRATEGY EVOLUTION 🚀")
    print(f"Symbol: {CONFIG['SYMBOL']} | Population: {CONFIG['POPULATION_SIZE']} | Generations: {CONFIG['GENERATIONS']}")
    print("=" * 65)
    
    try:
        # 1. Load or fetch data
        print("\n📊 DATA PREPARATION")
        print("-" * 30)
        fetch_fresh = input("Fetch fresh data from Binance API? (y/n): ").lower().startswith('y')
        data = load_and_prepare_data(CONFIG["DATA_PATH"], fetch_fresh=fetch_fresh)
        
        if len(data) < 500:
            print(f"⚠️  Warning: Only {len(data)} data points available. Consider fetching more data.")
            proceed = input("Continue anyway? (y/n): ").lower().startswith('y')
            if not proceed:
                exit(0)
        
        # 2. Setup DEAP GP Environment
        print("\n🧬 GENETIC PROGRAMMING SETUP")
        print("-" * 35)
        pset = create_primitive_set()
        setup_deap()
        toolbox = setup_toolbox(pset, data, CONFIG)

        # 3. Run Evolution
        print(f"\n🔬 EVOLUTION IN PROGRESS")
        print("-" * 30)
        final_pop, logbook, hall_of_fame = run_evolution(toolbox, CONFIG)

        # 4. Process and Save Results
        print("\n" + "="*60)
        print("📈 EVOLUTION COMPLETED - RESULTS SUMMARY")
        print("="*60)
        
        if hall_of_fame:
            print(f"\n🏆 TOP PERFORMING STRATEGIES:")
            print("-" * 45)
            
            for i, individual in enumerate(hall_of_fame):
                fitness = individual.fitness.values[0] if individual.fitness.valid else "Invalid"
                print(f"#{i+1:2d} | Return: {fitness:8.3f}% | Size: {len(individual):3d} nodes")
                
            # Save results
            save_results(hall_of_fame, CONFIG["RESULTS_PATH"], CONFIG["SYMBOL"])
            
            # Test best strategy
            test_best_strategy(hall_of_fame, data)
            
            print(f"\n🎯 EVOLUTION COMPLETE!")
            print(f"🥇 Best Strategy Return: {hall_of_fame[0].fitness.values[0]:.3f}%")
            print(f"📁 Results saved to: {CONFIG['RESULTS_PATH']}")
            
        else:
            print("❌ No valid strategies found in Hall of Fame.")

    except KeyboardInterrupt:
        print("\n⚠️  Evolution interrupted by user.")
    except FileNotFoundError as e:
        print(f"❌ Critical Error: {e}")
        print("💡 Make sure the data file exists or enable fresh data fetching.")
    except ValueError as e:
        print(f"❌ Critical Data Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected critical error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("🏁 BINANCE GP STRATEGY EVOLUTION SESSION ENDED")
        print("="*60)
