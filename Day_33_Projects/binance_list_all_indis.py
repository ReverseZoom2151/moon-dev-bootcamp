#!/usr/bin/env python3
"""
Binance Technical Indicators Discovery Tool

Lists available technical indicators from pandas_ta and talib libraries
with Binance-specific configurations and trading-focused categorization.
Saves comprehensive indicator metadata to JSON files for Binance trading strategies.
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import Binance configuration
try:
    from Day_26_Projects.binance_config import PRIMARY_SYMBOL
except ImportError:
    print("Warning: binance_config not found, using default values")
    PRIMARY_SYMBOL = "BTCUSDT"

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_results_dir = os.path.join(script_dir, "binance_ml", "results")

CONFIG = {
    # --- Paths ---
    "RESULTS_DIR_PATH": ml_results_dir,
    # Filenames within RESULTS_DIR_PATH
    "PANDAS_TA_FILENAME": "binance_pandas_ta_indicators.json",
    "EXTENDED_FILENAME": "binance_extended_indicators.json",
    "COMBINED_FILENAME": "binance_all_indicators.json",
    "TRADING_FOCUSED_FILENAME": "binance_trading_indicators.json",
    
    # --- Exchange Specific ---
    "EXCHANGE": "Binance",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "SUPPORTED_TIMEFRAMES": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
    
    # --- Indicator Categories for Trading ---
    "TRADING_CATEGORIES": {
        "trend": ["sma", "ema", "wma", "tema", "dema", "kama", "mama", "trima", "t3", "macd", "macdext", "macdfix", "ppo", "apo", "adx", "adxr", "aroon", "aroonosc", "cci", "dx", "minus_di", "minus_dm", "plus_di", "plus_dm", "trix"],
        "momentum": ["rsi", "roc", "rocp", "rocr", "rocr100", "mom", "bop", "cmo", "ppo", "rsi", "stoch", "stochf", "stochrsi", "tsi", "uo", "willr"],
        "volatility": ["atr", "natr", "trange", "bbands", "stddev", "var"],
        "volume": ["ad", "adosc", "obv", "mfi", "vwap", "vwma", "pvt", "nvi", "pvi"],
        "support_resistance": ["sar", "sarext", "pivot", "pivot_points", "support", "resistance"],
        "pattern": ["cdl2crows", "cdl3blackcrows", "cdl3inside", "cdl3linestrike", "cdl3outside", "cdl3starsinsouth", "cdl3whitesoldiers", "cdlabandonedbaby", "cdladvanceblock", "cdlbelthold", "cdlbreakaway", "cdlclosingmarubozu", "cdlconcealbabyswall", "cdlcounterattack", "cdldarkcloudcover", "cdldoji", "cdldojistar", "cdldragonflydoji", "cdlengulfing", "cdleveningdojistar", "cdleveningstar", "cdlgapsidesidewhite", "cdlgravestonedoji", "cdlhammer", "cdlhangingman", "cdlharami", "cdlharamicross", "cdlhighwave", "cdlhikkake", "cdlhikkakemod", "cdlhomingpigeon", "cdlidentical3crows", "cdlinneck", "cdlinvertedhammer", "cdlkicking", "cdlkickingbylength", "cdlladderbottom", "cdllongleggeddoji", "cdllongline", "cdlmarubozu", "cdlmatchinglow", "cdlmathold", "cdlmorningdojistar", "cdlmorningstar", "cdlonneck", "cdlpiercing", "cdlrickshawman", "cdlrisefall3methods", "cdlseparatinglines", "cdlshootingstar", "cdlshortline", "cdlspinningtop", "cdlstalledpattern", "cdlsticksandwich", "cdltakuri", "cdltasukigap", "cdlthrusting", "cdltristar", "cdlunique3river", "cdlupsidegap2crows", "cdlxsidegap3methods"],
        "binance_specific": ["futures_basis", "spot_premium", "volume_profile", "order_flow", "market_depth"]
    }
}

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Enhanced Functions ---

def list_pandas_ta_indicators() -> Dict[str, Any]:
    """
    Retrieves all available indicator names from pandas_ta with metadata.
    """
    logger.info("🟠 Discovering pandas_ta indicators for Binance trading...")
    
    try:
        import pandas_ta as ta
        
        # Method 1: Try to get indicators from Categories
        indicators_data = {"indicators": [], "categories": {}, "metadata": {}}
        
        try:
            # Get categories if available
            if hasattr(ta, 'Category'):
                categories = ta.Category
                for category_name, category_indicators in categories.items():
                    indicators_data["categories"][category_name] = category_indicators
                    indicators_data["indicators"].extend(category_indicators)
            
            # Get all callable functions that look like indicators
            all_functions = []
            for item_name in dir(ta):
                item = getattr(ta, item_name)
                if callable(item) and not item_name.startswith('_'):
                    # Filter for likely indicators
                    if (item_name.islower() or 
                        any(keyword in item_name.lower() for keyword in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'stoch', 'adx', 'cci', 'mfi', 'obv', 'vwap'])):
                        all_functions.append(item_name)
            
            # Combine and deduplicate
            all_indicators = list(set(indicators_data["indicators"] + all_functions))
            indicators_data["indicators"] = sorted(all_indicators)
            
        except Exception as e:
            logger.warning(f"Category method failed: {e}. Using alternative discovery.")
            # Alternative method: inspect module
            indicators_data["indicators"] = [
                item for item in dir(ta) 
                if callable(getattr(ta, item)) and not item.startswith('_') and item.islower()
            ]
        
        # Add metadata
        indicators_data["metadata"] = {
            "exchange": "Binance",
            "library": "pandas_ta",
            "discovery_time": datetime.now().isoformat(),
            "total_count": len(indicators_data["indicators"]),
            "supported_timeframes": CONFIG["SUPPORTED_TIMEFRAMES"],
            "primary_symbol": CONFIG["PRIMARY_SYMBOL"]
        }
        
        # Categorize for trading
        indicators_data["trading_categories"] = categorize_indicators(indicators_data["indicators"])
        
        logger.info(f"✅ Found {len(indicators_data['indicators'])} pandas_ta indicators for Binance")
        return indicators_data
        
    except ImportError:
        logger.error("❌ pandas_ta library not found. Please install: pip install pandas_ta")
        return {"indicators": [], "categories": {}, "metadata": {"error": "pandas_ta not installed"}}
    except Exception as e:
        logger.error(f"❌ Error discovering pandas_ta indicators: {e}", exc_info=True)
        return {"indicators": [], "categories": {}, "metadata": {"error": str(e)}}

def list_pandas_ta_extended_indicators() -> Dict[str, Any]:
    """
    Extended pandas_ta indicator discovery with traditional TA-Lib style indicators.
    """
    
    try:
        # Extended indicator set using pandas_ta capabilities
        extended_indicators = []
        
        # Core momentum indicators
        momentum_indicators = ['rsi', 'stoch', 'stochf', 'willr', 'cci', 'roc', 'mom', 'macd']
        
        # Trend indicators 
        trend_indicators = ['sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'adx', 'aroon', 'ppo']
        
        # Volume indicators
        volume_indicators = ['obv', 'ad', 'adosc', 'mfi']
        
        # Volatility indicators
        volatility_indicators = ['bbands', 'atr', 'natr', 'trange']
        
        # Price transform indicators
        price_indicators = ['avgprice', 'medprice', 'typprice', 'wclprice']
        
        # Additional indicators
        additional_indicators = ['stochrsi', 'trix', 'ultosc', 'bop', 'cmo']
        
        all_extended = momentum_indicators + trend_indicators + volume_indicators + volatility_indicators + price_indicators + additional_indicators
        
        # Process each indicator with Binance focus
        for indicator in all_extended:
            category = categorize_binance_indicator(indicator, "pandas_ta_extended")
            
            indicator_info = {
                "name": indicator.upper(),
                "category": category,
                "library": "pandas_ta_extended",
                "exchange_focus": "binance_spot_trading",
                "binance_strategies": get_binance_trading_strategies(indicator.upper()),
                "timeframes": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
                "scalping_suitable": indicator.upper() in ['RSI', 'EMA', 'MACD', 'STOCH', 'ATR'],
                "day_trading_suitable": True,
                "swing_trading_suitable": True,
                "pandas_ta_equivalent": indicator
            }
            
            extended_indicators.append(indicator_info)
        
        logger.info(f"🟠 Found {len(extended_indicators)} pandas_ta extended indicators for Binance")
        
        return {
            "indicators": [info["name"] for info in extended_indicators],
            "groups": {
                "momentum": [ind.upper() for ind in momentum_indicators],
                "trend": [ind.upper() for ind in trend_indicators], 
                "volume": [ind.upper() for ind in volume_indicators],
                "volatility": [ind.upper() for ind in volatility_indicators],
                "price": [ind.upper() for ind in price_indicators],
                "additional": [ind.upper() for ind in additional_indicators]
            },
            "metadata": extended_indicators,
            "binance_focus": "spot_trading_optimized_pandas_ta"
        }
    except Exception as e:
        logger.error(f"Error listing pandas_ta extended indicators: {e}")
        return {"indicators": [], "groups": {}, "metadata": {"error": str(e)}}

def categorize_indicators(indicators: List[str]) -> Dict[str, List[str]]:
    """Categorize indicators by their trading purpose."""
    categorized = {category: [] for category in CONFIG["TRADING_CATEGORIES"]}
    
    for indicator in indicators:
        indicator_lower = indicator.lower()
        
        # Check each category
        for category, keywords in CONFIG["TRADING_CATEGORIES"].items():
            if any(keyword in indicator_lower for keyword in keywords):
                categorized[category].append(indicator)
                break  # Only assign to first matching category
        else:
            # If no category matches, add to 'other'
            if 'other' not in categorized:
                categorized['other'] = []
            categorized['other'].append(indicator)
    
    return categorized

def get_binance_recommended_indicators() -> Dict[str, List[str]]:
    """Get indicators commonly used for Binance trading."""
    return {
        "scalping_1m_5m": ["EMA", "RSI", "STOCH", "BBANDS", "VWAP", "ATR"],
        "day_trading_15m_1h": ["SMA", "EMA", "MACD", "RSI", "ADX", "BBANDS", "OBV", "MFI"],
        "swing_trading_4h_1d": ["SMA", "EMA", "MACD", "RSI", "STOCH", "ADX", "BBANDS", "SAR", "CCI"],
        "position_trading_1d_1w": ["SMA", "EMA", "MACD", "RSI", "ADX", "ROC", "AROON"],
        "volume_analysis": ["OBV", "AD", "ADOSC", "MFI", "VWAP"],
        "volatility_analysis": ["ATR", "NATR", "BBANDS", "TRANGE"],
        "trend_analysis": ["SMA", "EMA", "ADX", "ADXR", "AROON", "MACD", "TRIX"],
        "momentum_analysis": ["RSI", "STOCH", "STOCHRSI", "MOM", "ROC", "CCI", "WILLR"],
        "pattern_recognition": ["DOJI", "HAMMER", "ENGULFING", "HARAMI", "MARUBOZU"]
    }

def create_binance_trading_export(pandas_data: Dict, extended_data: Dict) -> Dict[str, Any]:
    """Create a trading-focused export combining both libraries."""
    
    trading_export = {
        "binance_trading_indicators": {
            "exchange": "Binance",
            "creation_time": datetime.now().isoformat(),
            "primary_symbol": CONFIG["PRIMARY_SYMBOL"],
            "supported_timeframes": CONFIG["SUPPORTED_TIMEFRAMES"]
        }
    }
    
    # Combine recommendations
    binance_recs = get_binance_recommended_indicators()
    
    # Create unified indicator sets
    all_pandas = set(pandas_data.get("indicators", []))
    all_extended = set(extended_data.get("indicators", []))
    
    trading_export["indicator_availability"] = {
        "pandas_ta_only": list(all_pandas - all_extended),
        "extended_only": list(all_extended - all_pandas),
        "both_libraries": list(all_pandas & all_extended),
        "total_unique": list(all_pandas | all_extended),
    }
    
    # Add trading strategies
    trading_export["trading_strategies"] = binance_recs
    
    # Add timeframe recommendations
    trading_export["timeframe_strategies"] = {
        "scalping": {"timeframes": ["1m", "3m", "5m"], "indicators": ["EMA", "RSI", "STOCH", "VWAP"]},
        "day_trading": {"timeframes": ["15m", "30m", "1h"], "indicators": ["SMA", "EMA", "MACD", "RSI", "BBANDS"]},
        "swing_trading": {"timeframes": ["4h", "6h", "12h", "1d"], "indicators": ["SMA", "EMA", "MACD", "ADX", "SAR"]},
        "position_trading": {"timeframes": ["1d", "3d", "1w"], "indicators": ["SMA", "EMA", "MACD", "ADX", "ROC"]}
    }
    
    return trading_export

def save_to_json(data: Dict[str, Any], filepath: str, description: str = "") -> None:
    """Save data to JSON file with enhanced error handling."""
    if not data:
        logger.warning(f"⚠️  No data provided for {description}. Skipping.")
        return
        
    try:
        output_dir = os.path.dirname(filepath)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            
        logger.info(f"💾 Saved {description} to {filepath}")
        
    except IOError as e:
        logger.error(f"❌ I/O error saving {description}: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error saving {description}: {e}", exc_info=True)

# --- Main Orchestration ---

def main(config: Dict[str, Any]) -> None:
    """Main function to discover and export Binance trading indicators."""
    
    logger.info("🟠" + "="*70 + "🟠")
    logger.info("🚀    BINANCE TECHNICAL INDICATORS DISCOVERY TOOL    🚀")
    logger.info("🟠" + "="*70 + "🟠")
    
    results_dir = config["RESULTS_DIR_PATH"]
    
    # File paths
    pandas_ta_filepath = os.path.join(results_dir, config["PANDAS_TA_FILENAME"])
    extended_filepath = os.path.join(results_dir, config["EXTENDED_FILENAME"])
    combined_filepath = os.path.join(results_dir, config["COMBINED_FILENAME"])
    trading_filepath = os.path.join(results_dir, config["TRADING_FOCUSED_FILENAME"])
    
    logger.info(f"📊 Primary Symbol: {config['PRIMARY_SYMBOL']}")
    logger.info(f"📁 Output Directory: {results_dir}")
    
    # 1. Discover pandas_ta indicators
    logger.info("\n🔍 Discovering pandas_ta indicators...")
    pandas_ta_data = list_pandas_ta_indicators()
    save_to_json(pandas_ta_data, pandas_ta_filepath, "pandas_ta indicators")
    
    # 2. Discover TA-Lib indicators
    logger.info("\n🔍 Discovering TA-Lib indicators...")
    extended_data = list_pandas_ta_extended_indicators()
    save_to_json(extended_data, extended_filepath, "pandas_ta extended indicators")
    
    # 3. Create combined export
    logger.info("\n🔧 Creating combined indicator database...")
    combined_data = {
        "binance_indicators_database": {
            "exchange": "Binance",
            "creation_time": datetime.now().isoformat(),
            "primary_symbol": config["PRIMARY_SYMBOL"],
            "libraries": ["pandas_ta", "talib"],
            "pandas_ta": pandas_ta_data,
            "extended": extended_data
        }
    }
    save_to_json(combined_data, combined_filepath, "combined indicators database")
    
    # 4. Create trading-focused export
    logger.info("\n📊 Creating Binance trading-focused export...")
    trading_data = create_binance_trading_export(pandas_ta_data, extended_data)
    save_to_json(trading_data, trading_filepath, "trading-focused indicators")
    
    # Summary
    pandas_count = len(pandas_ta_data.get("indicators", []))
    extended_count = len(extended_data.get("indicators", []))
    
    logger.info("\n" + "="*70)
    logger.info("📊 DISCOVERY SUMMARY")
    logger.info("="*70)
    logger.info(f"🔧 pandas_ta indicators: {pandas_count}")
    logger.info(f"📈 TA-Lib indicators: {talib_count}")
    logger.info(f"📁 Files created: 4")
    logger.info(f"🎯 Exchange: Binance")
    logger.info(f"💰 Primary Symbol: {config['PRIMARY_SYMBOL']}")
    
    logger.info("\n✅ Binance indicator discovery completed successfully!")

if __name__ == "__main__":
    main(CONFIG)
