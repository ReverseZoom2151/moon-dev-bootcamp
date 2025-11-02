#!/usr/bin/env python3
"""
Bitfinex Professional Top Indicators Analysis

Analyzes feature importance results from Bitfinex professional ML evaluation pipeline,
filters potentially overfit features with professional-grade criteria, and saves
the top N features optimized for margin trading, derivatives, and institutional strategies.
"""

import pandas as pd
import os
import logging
import json
from typing import Dict, Optional
from datetime import datetime

try:
    from Day_26_Projects.bitfinex_config import PRIMARY_SYMBOL
except ImportError:
    PRIMARY_SYMBOL = "btcusd"

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_results_dir = os.path.join(script_dir, "bitfinex_ml", "results")

CONFIG = {
    # --- Paths ---
    "RESULTS_DIR_PATH": ml_results_dir,
    "IMPORTANCE_FILENAME": "bitfinex_feature_importance.csv",
    "PERFORMANCE_FILENAME": "bitfinex_model_performance.csv",
    "ANALYSIS_SUMMARY_FILENAME": "bitfinex_professional_analysis.json",
    
    # --- Exchange Specific ---
    "EXCHANGE": "Bitfinex",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "TRADING_TIER": "professional",
    
    # --- Professional Filtering ---
    "R2_FILTER_THRESHOLD": 0.93,  # Stricter for professional use
    "DIRECTION_ACCURACY_MIN": 0.55,  # Higher threshold for professional trading
    "MIN_GENERATIONS": 5,  # More appearances required for stability
    "IMPORTANCE_THRESHOLD": 0.01,  # Minimum feature importance
    
    # --- Top N Features ---
    "TOP_N_FEATURES": 40,  # More features for professional analysis
    "METRICS_TO_SORT": {
        "Importance": (False, "top_40_importance"),
        "R2": (False, "top_40_r2"),
        "Direction_Accuracy": (False, "top_40_direction_accuracy"),
        "Test_MAE": (True, "top_40_mae_best")  # Lower MAE is better
    },
    
    # --- Professional Analysis ---
    "ANALYZE_BY_MODEL": True,
    "ANALYZE_BY_TIMEFRAME": True,
    "ANALYZE_BY_ASSET_CLASS": True,
    "CREATE_PROFESSIONAL_STRATEGIES": True,
    "MARGIN_TRADING_FOCUS": True,
    "DERIVATIVES_ANALYSIS": True,
}

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [Bitfinex Pro Analysis] %(message)s'
)
logger = logging.getLogger(__name__)

def load_importance_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load Bitfinex professional importance data."""
    if not os.path.exists(filepath):
        logger.error(f"❌ Bitfinex professional importance file not found: {filepath}")
        logger.info("💡 Please run bitfinex_loop_all_indis.py first to generate professional results")
        return None
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded {len(df)} professional importance records")
        
        # Validate professional columns
        required_cols = {'Model', 'Feature', 'Importance', 'R2'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.error(f"❌ Missing professional columns: {missing_cols}")
            return None
        
        # Add exchange metadata
        if 'Exchange' not in df.columns:
            df['Exchange'] = CONFIG['EXCHANGE']
        if 'TrainingTier' not in df.columns:
            df['TrainingTier'] = CONFIG['TRADING_TIER']
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading professional data: {e}")
        return None

def load_performance_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load Bitfinex professional performance data."""
    if not os.path.exists(filepath):
        logger.warning(f"⚠️  Professional performance file not found: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded {len(df)} professional performance records")
        return df
    except Exception as e:
        logger.warning(f"⚠️  Could not load professional performance data: {e}")
        return None

def filter_features_professional(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Apply professional-grade feature filtering for Bitfinex."""
    initial_count = len(df)
    filtered_df = df.copy()
    
    logger.info(f"🔍 Applying professional filtering to {initial_count} features...")
    
    # Filter 1: Professional R2 threshold
    r2_threshold = config["R2_FILTER_THRESHOLD"]
    filtered_df = filtered_df[filtered_df['R2'] <= r2_threshold]
    logger.info(f"📊 Professional R2 filter: kept {len(filtered_df)} features")
    
    # Filter 2: Direction accuracy for trading edge
    if 'Direction_Accuracy' in filtered_df.columns:
        min_acc = config.get("DIRECTION_ACCURACY_MIN", 0.55)
        pre_filter_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Direction_Accuracy'] >= min_acc]
        logger.info(f"🎯 Professional direction filter: kept {len(filtered_df)} features")
    
    # Filter 3: Minimum importance threshold
    imp_threshold = config.get("IMPORTANCE_THRESHOLD", 0.01)
    pre_imp_count = len(filtered_df)
    filtered_df = filtered_df[filtered_df['Importance'] >= imp_threshold]
    logger.info(f"⚡ Importance threshold filter: kept {len(filtered_df)} features")
    
    # Filter 4: Professional stability requirement
    if 'Generation' in filtered_df.columns:
        min_gens = config.get("MIN_GENERATIONS", 5)
        feature_counts = filtered_df.groupby(['Model', 'Feature']).size()
        stable_features = feature_counts[feature_counts >= min_gens].index
        
        if len(stable_features) > 0:
            pre_stability_count = len(filtered_df)
            mask = filtered_df.set_index(['Model', 'Feature']).index.isin(stable_features)
            filtered_df = filtered_df[mask]
            logger.info(f"🔒 Professional stability filter: kept {len(filtered_df)} features")
    
    # Filter 5: Remove low-variance features
    if len(filtered_df) > 0:
        feature_variance = filtered_df.groupby('Feature')['Importance'].var()
        high_variance_features = feature_variance[feature_variance > 0.0001].index
        pre_var_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df['Feature'].isin(high_variance_features)]
        logger.info(f"📈 Variance filter: kept {len(filtered_df)} features")
    
    final_removed = initial_count - len(filtered_df)
    logger.info(f"🏛️  Professional filtering: removed {final_removed} features ({final_removed/initial_count:.1%})")
    
    return filtered_df

def analyze_professional_patterns(df: pd.DataFrame) -> Dict:
    """Analyze professional trading patterns in features."""
    analysis = {
        "total_features": len(df['Feature'].unique()),
        "total_models": len(df['Model'].unique()),
        "professional_indicators": {},
        "risk_metrics": {}
    }
    
    # Professional feature frequency analysis
    feature_stats = df.groupby('Feature').agg({
        'Importance': ['mean', 'std', 'count', 'max'],
        'R2': ['mean', 'max'],
        'Direction_Accuracy': 'mean' if 'Direction_Accuracy' in df.columns else lambda x: 0
    }).round(4)
    
    feature_stats.columns = ['avg_importance', 'importance_std', 'appearances', 
                           'max_importance', 'avg_r2', 'max_r2', 'avg_direction_acc']
    feature_stats = feature_stats.sort_values('avg_importance', ascending=False)
    
    # Professional indicator categories
    professional_categories = {
        'trend_professional': ['sma', 'ema', 'macd', 'adx', 'aroon', 'trix', 'dema', 'tema'],
        'momentum_professional': ['rsi', 'mom', 'roc', 'stoch', 'stochrsi', 'willr', 'cci', 'tsi'],
        'volatility_professional': ['atr', 'natr', 'bbands', 'stddev', 'var'],
        'volume_professional': ['obv', 'ad', 'adosc', 'mfi', 'vwap', 'cmf'],
        'pattern_professional': ['cdl', 'doji', 'hammer', 'engulfing', 'harami'],
        'margin_indicators': ['funding', 'margin', 'leverage', 'position'],
        'derivatives_indicators': ['basis', 'premium', 'futures', 'perpetual'],
        'institutional_indicators': ['whale', 'flow', 'institutional', 'large']
    }
    
    for category, keywords in professional_categories.items():
        category_features = []
        for feature in feature_stats.index:
            if any(keyword in feature.lower() for keyword in keywords):
                category_features.append({
                    'feature': feature,
                    'importance': feature_stats.loc[feature, 'avg_importance'],
                    'direction_accuracy': feature_stats.loc[feature, 'avg_direction_acc'],
                    'stability': int(feature_stats.loc[feature, 'appearances']),
                    'max_r2': feature_stats.loc[feature, 'max_r2']
                })
        
        if category_features:
            # Sort by combination of importance and direction accuracy
            category_features.sort(key=lambda x: x['importance'] * x['direction_accuracy'], reverse=True)
            analysis["professional_indicators"][category] = category_features[:8]  # Top 8 per category
    
    # Risk assessment metrics
    if len(feature_stats) > 0:
        analysis["risk_metrics"] = {
            "high_stability_features": len(feature_stats[feature_stats['appearances'] >= 8]),
            "high_direction_accuracy": len(feature_stats[feature_stats['avg_direction_acc'] >= 0.6]),
            "balanced_features": len(feature_stats[
                (feature_stats['avg_direction_acc'] >= 0.55) & 
                (feature_stats['avg_r2'] <= 0.9) & 
                (feature_stats['appearances'] >= 5)
            ]),
            "overfit_risk": len(feature_stats[feature_stats['max_r2'] >= 0.95])
        }
    
    return analysis

def create_professional_strategies(df: pd.DataFrame, top_n: int = 15) -> Dict:
    """Create professional trading strategies for Bitfinex."""
    strategies = {}
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        
        # Get top features with professional criteria
        if 'Direction_Accuracy' in model_data.columns:
            # Weight by both importance and direction accuracy
            model_data = model_data.copy()
            model_data['Professional_Score'] = (
                model_data['Importance'] * model_data['Direction_Accuracy'] * 
                (1 - model_data['R2'].clip(0, 1))  # Penalty for high R2
            )
            top_features = model_data.nlargest(top_n, 'Professional_Score')['Feature'].tolist()
        else:
            top_features = model_data.nlargest(top_n, 'Importance')['Feature'].tolist()
        
        # Professional trading strategies
        professional_strategies = {
            'scalping_professional': [f for f in top_features if any(keyword in f.lower() 
                                    for keyword in ['ema', 'rsi', 'stoch', 'atr', 'vwap'])],
            'margin_trading': [f for f in top_features if any(keyword in f.lower() 
                             for keyword in ['sma', 'ema', 'macd', 'rsi', 'funding', 'margin'])],
            'derivatives_trading': [f for f in top_features if any(keyword in f.lower() 
                                  for keyword in ['basis', 'premium', 'volatility', 'futures'])],
            'institutional_flow': [f for f in top_features if any(keyword in f.lower() 
                                 for keyword in ['volume', 'flow', 'whale', 'institutional'])],
            'cross_timeframe': [f for f in top_features if any(keyword in f.lower() 
                              for keyword in ['sma_50', 'sma_200', 'ema_50', 'ema_200'])],
            'high_frequency': [f for f in top_features if any(keyword in f.lower() 
                             for keyword in ['ema_12', 'rsi_14', 'stoch', 'momentum'])]
        }
        
        # Filter and optimize combinations
        for strategy, features in professional_strategies.items():
            if features:
                # Limit to top performers and ensure diversity
                unique_indicators = set()
                final_features = []
                for feature in features:
                    indicator_base = feature.split('_')[0] if '_' in feature else feature
                    if indicator_base not in unique_indicators and len(final_features) < 6:
                        unique_indicators.add(indicator_base)
                        final_features.append(feature)
                
                if final_features:
                    strategies[f"{model}_{strategy}"] = final_features
    
    return strategies

def save_dataframe(df: pd.DataFrame, filepath: str, description: str = "") -> None:
    """Save DataFrame with professional logging."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Saved {description}: {len(df)} professional records to {filepath}")
    except Exception as e:
        logger.error(f"❌ Error saving {description}: {e}")

def save_json_analysis(data: Dict, filepath: str) -> None:
    """Save professional analysis to JSON."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"💾 Saved professional analysis to {filepath}")
    except Exception as e:
        logger.error(f"❌ Error saving professional analysis: {e}")

def process_professional_model(model_name: str, model_data: pd.DataFrame, config: Dict) -> None:
    """Process professional model features with institutional-grade analysis."""
    logger.info(f"🏛️  Processing professional model: {model_name} ({len(model_data)} features)")
    
    # Apply professional filtering
    filtered_data = filter_features_professional(model_data, config)
    
    if filtered_data.empty:
        logger.warning(f"⚠️  No professional features remaining for {model_name}")
        return
    
    # Professional ranking and analysis
    results_dir = config["RESULTS_DIR_PATH"]
    top_n = config["TOP_N_FEATURES"]
    
    for metric, (ascending, suffix) in config["METRICS_TO_SORT"].items():
        if metric not in filtered_data.columns:
            logger.warning(f"⚠️  Professional metric '{metric}' not found for {model_name}")
            continue
        
        logger.info(f"📊 Professional ranking {model_name} by {metric}")
        
        # Sort and select top N with professional metadata
        sorted_df = filtered_data.sort_values(metric, ascending=ascending).head(top_n)
        
        # Add professional metadata
        sorted_df = sorted_df.copy()
        sorted_df['Professional_Rank'] = range(1, len(sorted_df) + 1)
        sorted_df['Ranking_Metric'] = metric
        sorted_df['Exchange'] = config['EXCHANGE']
        sorted_df['Symbol'] = config['PRIMARY_SYMBOL']
        sorted_df['Trading_Tier'] = config['TRADING_TIER']
        sorted_df['Analysis_Date'] = datetime.now().date()
        
        # Professional scoring
        if 'Direction_Accuracy' in sorted_df.columns and 'Importance' in sorted_df.columns:
            sorted_df['Professional_Score'] = (
                sorted_df['Importance'] * sorted_df['Direction_Accuracy']
            ).round(4)
        
        # Save with professional naming
        output_filename = f"bitfinex_professional_{model_name}_{suffix}.csv"
        output_filepath = os.path.join(results_dir, output_filename)
        save_dataframe(sorted_df, output_filepath, f"{model_name} professional features by {metric}")

def analyze_bitfinex_professional_features(config: Dict) -> None:
    """Main professional analysis for Bitfinex feature importance."""
    
    logger.info("🔵" + "="*70 + "🔵")
    logger.info("🏛️   BITFINEX PROFESSIONAL INDICATORS ANALYSIS   🏛️")
    logger.info("🔵" + "="*70 + "🔵")
    
    results_dir = config["RESULTS_DIR_PATH"]
    
    # Load professional data
    importance_file = os.path.join(results_dir, config["IMPORTANCE_FILENAME"])
    performance_file = os.path.join(results_dir, config["PERFORMANCE_FILENAME"])
    
    df = load_importance_data(importance_file)
    if df is None:
        logger.error("❌ Cannot proceed without professional importance data")
        return
    
    perf_df = load_performance_data(performance_file)
    
    # Merge with performance data for professional analysis
    if perf_df is not None:
        merge_cols = ['Generation', 'Model']
        professional_metrics = ['Direction_Accuracy', 'Test_MAE', 'Test_RMSE']
        
        available_metrics = [col for col in professional_metrics if col in perf_df.columns]
        if available_metrics:
            merge_df = perf_df[merge_cols + available_metrics].drop_duplicates()
            df = df.merge(merge_df, on=merge_cols, how='left')
            logger.info(f"✅ Merged professional performance metrics: {available_metrics}")
    
    logger.info(f"📊 Professional analysis: {len(df)} records, {len(df['Model'].unique())} models")
    
    # Process each model professionally
    os.makedirs(results_dir, exist_ok=True)
    models = df['Model'].unique()
    
    for model in models:
        model_df = df[df['Model'] == model].copy()
        process_professional_model(model, model_df, config)
    
    # Professional comprehensive analysis
    logger.info("🎯 Generating professional trading intelligence...")
    
    # Professional pattern analysis
    professional_patterns = analyze_professional_patterns(df)
    
    # Professional strategy creation
    professional_strategies = create_professional_strategies(df, config["TOP_N_FEATURES"])
    
    # Institutional-grade summary
    professional_summary = {
        "bitfinex_professional_analysis": {
            "exchange": config["EXCHANGE"],
            "symbol": config["PRIMARY_SYMBOL"],
            "trading_tier": config["TRADING_TIER"],
            "analysis_timestamp": datetime.now().isoformat(),
            "professional_standards": {
                "r2_threshold": config["R2_FILTER_THRESHOLD"],
                "direction_accuracy_min": config["DIRECTION_ACCURACY_MIN"],
                "min_generations_stability": config["MIN_GENERATIONS"],
                "importance_threshold": config.get("IMPORTANCE_THRESHOLD")
            },
            "analysis_scope": {
                "total_generations": df['Generation'].nunique() if 'Generation' in df.columns else 0,
                "models_analyzed": df['Model'].unique().tolist(),
                "feature_universe": len(df['Feature'].unique())
            }
        },
        "professional_patterns": professional_patterns,
        "institutional_strategies": professional_strategies,
        "top_professional_features": df.groupby('Feature')['Importance'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False).head(20).to_dict(),
        "risk_assessment": {
            "overfit_features_detected": len(df[df['R2'] > 0.95]['Feature'].unique()) if 'R2' in df.columns else 0,
            "stable_features_count": len(df.groupby('Feature').size()[df.groupby('Feature').size() >= 5]),
            "professional_grade_features": len(df[(df['Importance'] >= 0.01) & (df['R2'] <= 0.93)]['Feature'].unique()) if all(col in df.columns for col in ['Importance', 'R2']) else 0
        }
    }
    
    # Save professional analysis
    summary_file = os.path.join(results_dir, config["ANALYSIS_SUMMARY_FILENAME"])
    save_json_analysis(professional_summary, summary_file)
    
    # Professional summary report
    logger.info("\n" + "="*70)
    logger.info("📊 BITFINEX PROFESSIONAL ANALYSIS SUMMARY")
    logger.info("="*70)
    logger.info(f"🏛️  Trading Tier: {config['TRADING_TIER']}")
    logger.info(f"🔍 Models analyzed: {len(models)}")
    logger.info(f"🎯 Professional features: {len(df['Feature'].unique())}")
    logger.info(f"📈 Indicator categories: {len(professional_patterns.get('professional_indicators', {}))}")
    logger.info(f"🤖 Professional strategies: {len(professional_strategies)}")
    logger.info(f"💰 Primary symbol: {config['PRIMARY_SYMBOL']}")
    logger.info(f"⚡ Risk controls: Applied institutional-grade filtering")
    
    logger.info("\n✅ Bitfinex professional indicators analysis completed!")

if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CONFIG["RESULTS_DIR_PATH"] = os.path.join(script_dir, "bitfinex_ml", "results")
        analyze_bitfinex_professional_features(CONFIG)
    except Exception as e:
        logger.error(f"❌ Professional analysis failed: {e}")
