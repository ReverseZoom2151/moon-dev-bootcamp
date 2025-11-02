#!/usr/bin/env python3
"""
Binance Top Indicators Analysis

Analyzes feature importance results from Binance ML evaluation pipeline,
filters potentially overfit features, and saves the top N features based on
different trading metrics (Importance, MSE, R2, Direction Accuracy) for each model.
"""

import pandas as pd
import os
import logging
import json
from typing import Dict, Optional
from datetime import datetime

# Import Binance configuration
try:
    from Day_26_Projects.binance_config import PRIMARY_SYMBOL
except ImportError:
    PRIMARY_SYMBOL = "BTCUSDT"

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_results_dir = os.path.join(script_dir, "binance_ml", "results")

CONFIG = {
    # --- Paths ---
    "RESULTS_DIR_PATH": ml_results_dir,
    "IMPORTANCE_FILENAME": "binance_feature_importance.csv",
    "PERFORMANCE_FILENAME": "binance_model_performance.csv",
    "ANALYSIS_SUMMARY_FILENAME": "binance_top_indicators_analysis.json",
    
    # --- Exchange Specific ---
    "EXCHANGE": "Binance",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "TRADING_FOCUS": "spot_trading",
    
    # --- Filtering ---
    "R2_FILTER_THRESHOLD": 0.95,  # Filter potentially overfit features
    "DIRECTION_ACCURACY_MIN": 0.52,  # Minimum direction accuracy for trading
    "MIN_GENERATIONS": 3,  # Minimum appearances across generations
    
    # --- Top N Features ---
    "TOP_N_FEATURES": 30,  # Focused set for Binance spot trading
    "METRICS_TO_SORT": {
        "Importance": (False, "top_30_importance"),
        "R2": (False, "top_30_r2"), 
        "Direction_Accuracy": (False, "top_30_direction_accuracy")
    },
    
    # --- Trading Analysis ---
    "ANALYZE_BY_MODEL": True,
    "ANALYZE_BY_TIMEFRAME": True,
    "CREATE_TRADING_COMBINATIONS": True,
}

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [Binance Analysis] %(message)s'
)
logger = logging.getLogger(__name__)

def load_importance_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load Binance feature importance data with validation."""
    if not os.path.exists(filepath):
        logger.error(f"❌ Binance importance file not found: {filepath}")
        logger.info("💡 Please run binance_loop_all_indis.py first to generate results")
        return None
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded {len(df)} importance records from Binance results")
        
        # Validate required columns
        required_cols = {'Model', 'Feature', 'Importance', 'R2'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Add exchange info if not present
        if 'Exchange' not in df.columns:
            df['Exchange'] = CONFIG['EXCHANGE']
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading importance data: {e}")
        return None

def load_performance_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load Binance model performance data."""
    if not os.path.exists(filepath):
        logger.warning(f"⚠️  Performance file not found: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded {len(df)} performance records")
        return df
    except Exception as e:
        logger.warning(f"⚠️  Could not load performance data: {e}")
        return None

def filter_features_binance(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Apply Binance-specific feature filtering."""
    initial_count = len(df)
    filtered_df = df.copy()
    
    # Filter 1: R2 threshold to remove overfit features
    r2_threshold = config["R2_FILTER_THRESHOLD"]
    filtered_df = filtered_df[filtered_df['R2'] <= r2_threshold]
    logger.info(f"📊 R2 filter: kept {len(filtered_df)} features (removed {initial_count - len(filtered_df)} overfit)")
    
    # Filter 2: Direction accuracy for trading relevance
    if 'Direction_Accuracy' in filtered_df.columns:
        min_acc = config.get("DIRECTION_ACCURACY_MIN", 0.5)
        pre_filter_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Direction_Accuracy'] >= min_acc]
        logger.info(f"🎯 Direction accuracy filter: kept {len(filtered_df)} features (removed {pre_filter_count - len(filtered_df)} poor predictors)")
    
    # Filter 3: Minimum generation appearances (stability)
    if 'Generation' in filtered_df.columns:
        min_gens = config.get("MIN_GENERATIONS", 3)
        feature_counts = filtered_df.groupby(['Model', 'Feature']).size()
        stable_features = feature_counts[feature_counts >= min_gens].index
        
        if len(stable_features) > 0:
            pre_stability_count = len(filtered_df)
            mask = filtered_df.set_index(['Model', 'Feature']).index.isin(stable_features)
            filtered_df = filtered_df[mask]
            logger.info(f"🔒 Stability filter: kept {len(filtered_df)} features (removed {pre_stability_count - len(filtered_df)} unstable)")
    
    final_removed = initial_count - len(filtered_df)
    if final_removed > 0:
        logger.info(f"📉 Total filtering: removed {final_removed} features ({final_removed/initial_count:.1%})")
    
    return filtered_df

def analyze_feature_patterns(df: pd.DataFrame) -> Dict:
    """Analyze Binance-specific feature patterns."""
    analysis = {
        "total_features": len(df['Feature'].unique()),
        "total_models": len(df['Model'].unique()),
        "top_indicators": {}
    }
    
    # Most important indicators by frequency
    feature_freq = df.groupby('Feature').agg({
        'Importance': ['mean', 'std', 'count'],
        'R2': 'mean'
    }).round(4)
    
    feature_freq.columns = ['avg_importance', 'importance_std', 'appearances', 'avg_r2']
    feature_freq = feature_freq.sort_values('avg_importance', ascending=False)
    
    # Categorize by indicator type
    categories = {
        'trend': ['sma', 'ema', 'macd', 'adx', 'aroon'],
        'momentum': ['rsi', 'mom', 'roc', 'stoch', 'willr'],
        'volatility': ['atr', 'bbands', 'natr'],
        'volume': ['obv', 'ad', 'mfi', 'vwap'],
        'pattern': ['cdl', 'doji', 'hammer']
    }
    
    for category, keywords in categories.items():
        category_features = []
        for feature in feature_freq.index:
            if any(keyword in feature.lower() for keyword in keywords):
                category_features.append({
                    'feature': feature,
                    'importance': feature_freq.loc[feature, 'avg_importance'],
                    'appearances': int(feature_freq.loc[feature, 'appearances'])
                })
        
        if category_features:
            analysis["top_indicators"][category] = sorted(category_features, 
                                                        key=lambda x: x['importance'], 
                                                        reverse=True)[:5]
    
    return analysis

def create_trading_combinations(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """Create trading-ready indicator combinations for Binance."""
    combinations = {}
    
    # Get top features by model
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        top_features = model_data.nlargest(top_n, 'Importance')['Feature'].tolist()
        
        # Create trading strategies
        strategies = {
            'scalping_1m': [f for f in top_features if any(keyword in f.lower() 
                           for keyword in ['ema', 'rsi', 'stoch', 'vwap'])],
            'day_trading_15m': [f for f in top_features if any(keyword in f.lower() 
                               for keyword in ['sma', 'macd', 'bbands', 'adx'])],
            'swing_trading_1h': [f for f in top_features if any(keyword in f.lower() 
                                for keyword in ['sma', 'macd', 'rsi', 'atr'])]
        }
        
        # Filter and limit combinations
        for strategy, features in strategies.items():
            if features:
                combinations[f"{model}_{strategy}"] = features[:5]  # Top 5 for each strategy
    
    return combinations

def save_dataframe(df: pd.DataFrame, filepath: str, description: str = "") -> None:
    """Save DataFrame with enhanced logging."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Saved {description}: {len(df)} records to {filepath}")
    except Exception as e:
        logger.error(f"❌ Error saving {description}: {e}")

def save_json_analysis(data: Dict, filepath: str) -> None:
    """Save analysis results to JSON."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"💾 Saved analysis summary to {filepath}")
    except Exception as e:
        logger.error(f"❌ Error saving analysis: {e}")

def process_model_features(model_name: str, model_data: pd.DataFrame, config: Dict) -> None:
    """Process and save top features for a Binance model."""
    logger.info(f"🤖 Processing Binance model: {model_name} ({len(model_data)} features)")
    
    # Apply Binance-specific filtering
    filtered_data = filter_features_binance(model_data, config)
    
    if filtered_data.empty:
        logger.warning(f"⚠️  No features remaining for {model_name} after filtering")
        return
    
    # Sort and save by different metrics
    results_dir = config["RESULTS_DIR_PATH"]
    top_n = config["TOP_N_FEATURES"]
    
    for metric, (ascending, suffix) in config["METRICS_TO_SORT"].items():
        if metric not in filtered_data.columns:
            logger.warning(f"⚠️  Metric '{metric}' not found for {model_name}")
            continue
        
        logger.info(f"📊 Sorting {model_name} by {metric} (top {top_n})")
        
        # Sort and select top N
        sorted_df = filtered_data.sort_values(metric, ascending=ascending).head(top_n)
        
        # Add ranking
        sorted_df = sorted_df.copy()
        sorted_df['Rank'] = range(1, len(sorted_df) + 1)
        sorted_df['Metric_Used'] = metric
        sorted_df['Exchange'] = config['EXCHANGE']
        sorted_df['Symbol'] = config['PRIMARY_SYMBOL']
        
        # Save
        output_filename = f"binance_{model_name}_{suffix}.csv"
        output_filepath = os.path.join(results_dir, output_filename)
        save_dataframe(sorted_df, output_filepath, f"{model_name} top features by {metric}")

def analyze_binance_features(config: Dict) -> None:
    """Main analysis function for Binance feature importance."""
    
    logger.info("🟠" + "="*60 + "🟠")
    logger.info("🚀  BINANCE TOP INDICATORS ANALYSIS  🚀")
    logger.info("🟠" + "="*60 + "🟠")
    
    results_dir = config["RESULTS_DIR_PATH"]
    
    # Load data
    importance_file = os.path.join(results_dir, config["IMPORTANCE_FILENAME"])
    performance_file = os.path.join(results_dir, config["PERFORMANCE_FILENAME"])
    
    df = load_importance_data(importance_file)
    if df is None:
        logger.error("❌ Cannot proceed without importance data")
        return
    
    perf_df = load_performance_data(performance_file)
    
    # Merge with performance data if available
    if perf_df is not None:
        # Add direction accuracy to importance data
        if 'Direction_Accuracy' in perf_df.columns:
            merge_cols = ['Generation', 'Model']
            if all(col in df.columns and col in perf_df.columns for col in merge_cols):
                df = df.merge(perf_df[merge_cols + ['Direction_Accuracy']], 
                            on=merge_cols, how='left')
    
    logger.info(f"📊 Analyzing {len(df)} importance records for {len(df['Model'].unique())} models")
    
    # Process each model
    os.makedirs(results_dir, exist_ok=True)
    models = df['Model'].unique()
    
    for model in models:
        model_df = df[df['Model'] == model].copy()
        process_model_features(model, model_df, config)
    
    # Overall analysis
    logger.info("📈 Generating Binance trading analysis...")
    
    # Feature pattern analysis
    feature_analysis = analyze_feature_patterns(df)
    
    # Trading combinations
    trading_combinations = create_trading_combinations(df, config["TOP_N_FEATURES"])
    
    # Comprehensive summary
    summary = {
        "binance_analysis": {
            "exchange": config["EXCHANGE"],
            "symbol": config["PRIMARY_SYMBOL"],
            "analysis_time": datetime.now().isoformat(),
            "total_generations_analyzed": df['Generation'].nunique() if 'Generation' in df.columns else 0,
            "models_analyzed": df['Model'].unique().tolist(),
            "filtering_applied": {
                "r2_threshold": config["R2_FILTER_THRESHOLD"],
                "direction_accuracy_min": config.get("DIRECTION_ACCURACY_MIN"),
                "min_generations": config.get("MIN_GENERATIONS")
            }
        },
        "feature_patterns": feature_analysis,
        "trading_combinations": trading_combinations,
        "top_features_overall": df.groupby('Feature')['Importance'].mean().nlargest(15).to_dict()
    }
    
    # Save comprehensive analysis
    summary_file = os.path.join(results_dir, config["ANALYSIS_SUMMARY_FILENAME"])
    save_json_analysis(summary, summary_file)
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("📊 BINANCE ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"🔍 Models analyzed: {len(models)}")
    logger.info(f"🎯 Unique features: {len(df['Feature'].unique())}")
    logger.info(f"📈 Top indicator types: {list(feature_analysis['top_indicators'].keys())}")
    logger.info(f"🤖 Trading combinations: {len(trading_combinations)}")
    logger.info(f"💰 Primary symbol: {config['PRIMARY_SYMBOL']}")
    
    logger.info("\n✅ Binance top indicators analysis completed!")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CONFIG["RESULTS_DIR_PATH"] = os.path.join(script_dir, "binance_ml", "results")
        analyze_binance_features(CONFIG)
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
