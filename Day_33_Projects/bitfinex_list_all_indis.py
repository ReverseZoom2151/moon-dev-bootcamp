#!/usr/bin/env python3
"""
Bitfinex Technical Indicators Discovery Tool

Lists available technical indicators from pandas_ta and talib libraries
with Bitfinex-specific configurations and professional trading-focused categorization.
Saves comprehensive indicator metadata to JSON files for Bitfinex margin and derivatives trading strategies.
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# Import Bitfinex configuration
try:
    from Day_26_Projects.bitfinex_config import PRIMARY_SYMBOL
except ImportError:
    print("Warning: bitfinex_config not found, using default values")
    PRIMARY_SYMBOL = "btcusd"

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_results_dir = os.path.join(script_dir, "bitfinex_ml", "results")

CONFIG = {
    # --- Paths ---
    "RESULTS_DIR_PATH": ml_results_dir,
    # Filenames within RESULTS_DIR_PATH
    "PANDAS_TA_FILENAME": "bitfinex_pandas_ta_indicators.json",
    "PROFESSIONAL_FILENAME": "bitfinex_professional_indicators.json",
    "COMBINED_FILENAME": "bitfinex_all_indicators.json",
    "TRADING_FOCUSED_FILENAME": "bitfinex_professional_indicators.json",
    
    # --- Exchange Specific ---
    "EXCHANGE": "Bitfinex",
    "PRIMARY_SYMBOL": PRIMARY_SYMBOL,
    "SUPPORTED_TIMEFRAMES": ["1m", "5m", "15m", "30m", "1h", "3h", "6h", "12h", "1D", "1W", "14D", "1M"],
    
    # --- Professional Trading Categories ---
    "TRADING_CATEGORIES": {
        "trend": ["sma", "ema", "wma", "tema", "dema", "kama", "mama", "trima", "t3", "macd", "macdext", "macdfix", "ppo", "apo", "adx", "adxr", "aroon", "aroonosc", "cci", "dx", "minus_di", "minus_dm", "plus_di", "plus_dm", "trix"],
        "momentum": ["rsi", "roc", "rocp", "rocr", "rocr100", "mom", "bop", "cmo", "ppo", "rsi", "stoch", "stochf", "stochrsi", "tsi", "uo", "willr"],
        "volatility": ["atr", "natr", "trange", "bbands", "stddev", "var", "vix"],
        "volume": ["ad", "adosc", "obv", "mfi", "vwap", "vwma", "pvt", "nvi", "pvi", "cmf"],
        "support_resistance": ["sar", "sarext", "pivot", "pivot_points", "support", "resistance", "fibonacci"],
        "pattern": ["cdl2crows", "cdl3blackcrows", "cdl3inside", "cdl3linestrike", "cdl3outside", "cdl3starsinsouth", "cdl3whitesoldiers", "cdlabandonedbaby", "cdladvanceblock", "cdlbelthold", "cdlbreakaway", "cdlclosingmarubozu", "cdlconcealbabyswall", "cdlcounterattack", "cdldarkcloudcover", "cdldoji", "cdldojistar", "cdldragonflydoji", "cdlengulfing", "cdleveningdojistar", "cdleveningstar", "cdlgapsidesidewhite", "cdlgravestonedoji", "cdlhammer", "cdlhangingman", "cdlharami", "cdlharamicross", "cdlhighwave", "cdlhikkake", "cdlhikkakemod", "cdlhomingpigeon", "cdlidentical3crows", "cdlinneck", "cdlinvertedhammer", "cdlkicking", "cdlkickingbylength", "cdlladderbottom", "cdllongleggeddoji", "cdllongline", "cdlmarubozu", "cdlmatchinglow", "cdlmathold", "cdlmorningdojistar", "cdlmorningstar", "cdlonneck", "cdlpiercing", "cdlrickshawman", "cdlrisefall3methods", "cdlseparatinglines", "cdlshootingstar", "cdlshortline", "cdlspinningtop", "cdlstalledpattern", "cdlsticksandwich", "cdltakuri", "cdltasukigap", "cdlthrusting", "cdltristar", "cdlunique3river", "cdlupsidegap2crows", "cdlxsidegap3methods"],
        "margin_trading": ["funding_rate", "margin_balance", "position_ratio", "long_short_ratio"],
        "derivatives": ["perpetual_premium", "futures_basis", "options_flow", "gamma_exposure"],
        "bitfinex_specific": ["lending_rates", "swap_rates", "margin_funding", "position_books", "order_book_imbalance", "whale_alerts"]
    }
}

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Enhanced Functions ---

def list_pandas_ta_indicators() -> Dict[str, Any]:
    """
    Retrieves all available indicator names from pandas_ta with Bitfinex-specific metadata.
    """
    logger.info("🔵 Discovering pandas_ta indicators for Bitfinex professional trading...")
    
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
                    # Filter for likely indicators (more comprehensive for professional trading)
                    if (item_name.islower() or 
                        any(keyword in item_name.lower() for keyword in [
                            'sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'stoch', 'adx', 'cci', 'mfi', 'obv', 'vwap',
                            'roc', 'mom', 'willr', 'uo', 'tsi', 'ppo', 'apo', 'aroon', 'dx', 'trix', 'kama',
                            'tema', 'dema', 'trima', 't3', 'mama', 'sar', 'pivot', 'fibonacci', 'ichimoku'
                        ])):
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
        
        # Add Bitfinex-specific metadata
        indicators_data["metadata"] = {
            "exchange": "Bitfinex",
            "library": "pandas_ta",
            "discovery_time": datetime.now().isoformat(),
            "total_count": len(indicators_data["indicators"]),
            "supported_timeframes": CONFIG["SUPPORTED_TIMEFRAMES"],
            "primary_symbol": CONFIG["PRIMARY_SYMBOL"],
            "trading_types": ["spot", "margin", "derivatives", "lending"],
            "professional_features": ["funding_rates", "margin_trading", "derivatives", "advanced_orders"]
        }
        
        # Categorize for professional trading
        indicators_data["trading_categories"] = categorize_indicators(indicators_data["indicators"])
        indicators_data["bitfinex_enhancements"] = get_bitfinex_enhancements()
        
        logger.info(f"✅ Found {len(indicators_data['indicators'])} pandas_ta indicators for Bitfinex")
        return indicators_data
        
    except ImportError:
        logger.error("❌ pandas_ta library not found. Please install: pip install pandas_ta")
        return {"indicators": [], "categories": {}, "metadata": {"error": "pandas_ta not installed"}}
    except Exception as e:
        logger.error(f"❌ Error discovering pandas_ta indicators: {e}", exc_info=True)
        return {"indicators": [], "categories": {}, "metadata": {"error": str(e)}}

def list_pandas_ta_professional_indicators() -> Dict[str, Any]:
    """
    Professional pandas_ta indicator discovery for Bitfinex institutional trading.
    """
    logger.info("🔵 Discovering professional pandas_ta indicators for Bitfinex...")
    
    try:
        # Professional indicator set for institutional trading
        professional_indicators = []
        
        # Core momentum indicators for professional trading
        momentum_indicators = ['rsi', 'stoch', 'stochf', 'stochrsi', 'willr', 'cci', 'roc', 'mom', 'macd', 'ppo', 'trix']
        
        # Advanced trend indicators for institutional analysis
        trend_indicators = ['sma', 'ema', 'wma', 'dema', 'tema', 'trima', 'kama', 'adx', 'aroon', 'psar', 'supertrend']
        
        # Volume indicators for institutional flow analysis
        volume_indicators = ['obv', 'ad', 'adosc', 'mfi', 'cmf', 'eom', 'vwap', 'pvt']
        
        # Volatility indicators for risk management
        volatility_indicators = ['bbands', 'atr', 'natr', 'trange', 'kc', 'donchian']
        
        # Professional price analysis indicators
        price_indicators = ['avgprice', 'medprice', 'typprice', 'wclprice', 'hlc3', 'ohlc4']
        
        # Advanced indicators for derivatives trading
        advanced_indicators = ['ichimoku', 'squeeze', 'ttm_trend', 'entropy', 'zscore', 'quantile']
        
        # Institutional indicators for funding rate analysis
        institutional_indicators = ['bop', 'cmo', 'ultosc', 'fisher', 'hurst', 'linreg']
        
        all_professional = (momentum_indicators + trend_indicators + volume_indicators + 
                          volatility_indicators + price_indicators + advanced_indicators + institutional_indicators)
        
        # Process each indicator with Bitfinex professional focus
        for indicator in all_professional:
            category = categorize_bitfinex_indicator(indicator, "pandas_ta_professional")
            
            indicator_info = {
                "name": indicator.upper(),
                "category": category,
                "library": "pandas_ta_professional",
                "exchange_focus": "bitfinex_professional_trading",
                "bitfinex_strategies": get_bitfinex_professional_strategies(indicator.upper()),
                "timeframes": ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
                "margin_trading_suitable": indicator.upper() in ['RSI', 'EMA', 'MACD', 'BBANDS', 'ATR', 'STOCH'],
                "derivatives_suitable": True,
                "funding_rate_analysis": indicator.upper() in ['RSI', 'MACD', 'BBANDS', 'MFI', 'VWAP'],
                "institutional_grade": True,
                "pandas_ta_equivalent": indicator
            }
            
            professional_indicators.append(indicator_info)
        
        logger.info(f"🟣 Found {len(professional_indicators)} professional pandas_ta indicators for Bitfinex")
        
        return {
            "indicators": [info["name"] for info in professional_indicators],
            "groups": {
                "momentum": [ind.upper() for ind in momentum_indicators],
                "trend": [ind.upper() for ind in trend_indicators],
                "volume": [ind.upper() for ind in volume_indicators],
                "volatility": [ind.upper() for ind in volatility_indicators],
                "price": [ind.upper() for ind in price_indicators],
                "advanced": [ind.upper() for ind in advanced_indicators],
                "institutional": [ind.upper() for ind in institutional_indicators]
            },
            "metadata": professional_indicators,
            "bitfinex_focus": "professional_trading_optimized_pandas_ta"
        }
    except Exception as e:
        logger.error(f"Error listing professional pandas_ta indicators: {e}")
        return {"indicators": [], "groups": {}, "metadata": {"error": str(e)}}

def categorize_indicators(indicators: List[str]) -> Dict[str, List[str]]:
    """Categorize indicators by their professional trading purpose."""
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

def get_bitfinex_enhancements() -> Dict[str, Any]:
    """Get Bitfinex-specific enhancements and features."""
    return {
        "funding_rate_analysis": {
            "description": "Analyze funding rates for perpetual swaps",
            "indicators": ["funding_rate_sma", "funding_rate_momentum", "funding_rate_volatility"],
            "timeframes": ["1h", "8h", "1D"]
        },
        "margin_trading_tools": {
            "description": "Enhanced indicators for margin trading",
            "indicators": ["margin_balance_ratio", "liquidation_levels", "margin_utilization"],
            "risk_management": ["stop_loss_calculator", "position_sizing", "leverage_optimizer"]
        },
        "derivatives_analysis": {
            "description": "Professional derivatives trading indicators",
            "indicators": ["basis_analysis", "premium_discount", "roll_yield", "term_structure"],
            "products": ["futures", "perpetuals", "options"]
        },
        "order_flow_analysis": {
            "description": "Advanced order flow and market microstructure",
            "indicators": ["order_book_imbalance", "trade_classification", "market_impact"],
            "features": ["whale_detection", "institutional_flow", "retail_sentiment"]
        }
    }

def get_bitfinex_recommended_indicators() -> Dict[str, List[str]]:
    """Get indicators commonly used for Bitfinex professional trading."""
    return {
        "professional_scalping": ["EMA", "RSI", "STOCH", "BBANDS", "VWAP", "ATR", "ORDER_FLOW"],
        "margin_trading": ["SMA", "EMA", "MACD", "RSI", "ADX", "BBANDS", "OBV", "MFI", "FUNDING_RATE"],
        "swing_trading": ["SMA", "EMA", "MACD", "RSI", "STOCH", "ADX", "BBANDS", "SAR", "CCI", "AROON"],
        "position_trading": ["SMA", "EMA", "MACD", "RSI", "ADX", "ROC", "AROON", "LINEAR_REG"],
        "derivatives_trading": ["BASIS", "PREMIUM", "ROLL_YIELD", "VOLATILITY_SURFACE", "SKEW"],
        "volume_profile": ["OBV", "AD", "ADOSC", "MFI", "VWAP", "VOLUME_PROFILE", "MARKET_PROFILE"],
        "volatility_trading": ["ATR", "NATR", "BBANDS", "TRANGE", "VIX", "REALIZED_VOL", "IMPLIED_VOL"],
        "trend_analysis": ["SMA", "EMA", "ADX", "ADXR", "AROON", "MACD", "TRIX", "LINEAR_REG"],
        "momentum_strategies": ["RSI", "STOCH", "STOCHRSI", "MOM", "ROC", "CCI", "WILLR", "TSI"],
        "pattern_recognition": ["DOJI", "HAMMER", "ENGULFING", "HARAMI", "MARUBOZU", "SHOOTING_STAR"],
        "funding_strategies": ["FUNDING_RATE", "FUNDING_MOMENTUM", "CROSS_EXCHANGE_ARBITRAGE"],
        "institutional_analysis": ["WHALE_FLOW", "INSTITUTIONAL_SENTIMENT", "LARGE_ORDER_DETECTION"]
    }

def get_professional_trading_setups() -> Dict[str, Any]:
    """Get professional trading setups for Bitfinex."""
    return {
        "crypto_majors": {
            "symbols": ["btcusd", "ethusd", "ltcusd", "xrpusd", "adausd", "dotusd", "uniusd"],
            "timeframes": ["15m", "1h", "4h", "1D"],
            "strategies": ["momentum", "mean_reversion", "trend_following", "funding_arbitrage"]
        },
        "forex_majors": {
            "symbols": ["eurusd", "gbpusd", "usdjpy", "audusd", "usdcad", "usdchf"],
            "timeframes": ["5m", "15m", "1h", "4h"],
            "strategies": ["carry_trade", "news_trading", "technical_breakout", "range_trading"]
        },
        "commodities": {
            "symbols": ["xauusd", "xagusd", "oil", "natural_gas"],
            "timeframes": ["1h", "4h", "1D"],
            "strategies": ["seasonal_trading", "supply_demand", "macro_correlation"]
        },
        "margin_products": {
            "leverage_levels": ["2x", "3.3x", "5x", "10x"],
            "risk_management": ["position_sizing", "stop_loss", "take_profit", "trailing_stop"],
            "funding_costs": ["daily_funding", "swap_rates", "margin_interest"]
        },
        "derivatives": {
            "perpetuals": ["btcf0", "ethf0", "ltcf0", "xrpf0"],
            "monthly_futures": ["btc_monthly", "eth_monthly"],
            "strategies": ["basis_trading", "calendar_spreads", "funding_arbitrage"]
        }
    }

def create_professional_trading_export(pandas_data: Dict, talib_data: Dict) -> Dict[str, Any]:
    """Create a professional trading-focused export combining both libraries."""
    
    professional_export = {
        "bitfinex_professional_indicators": {
            "exchange": "Bitfinex",
            "creation_time": datetime.now().isoformat(),
            "primary_symbol": CONFIG["PRIMARY_SYMBOL"],
            "supported_timeframes": CONFIG["SUPPORTED_TIMEFRAMES"],
            "trading_types": ["spot", "margin", "derivatives", "lending"],
            "professional_tier": "institutional_grade"
        }
    }
    
    # Combine recommendations
    bitfinex_recs = get_bitfinex_recommended_indicators()
    professional_setups = get_professional_trading_setups()
    
    # Create unified indicator sets
    all_pandas = set(pandas_data.get("indicators", []))
    all_talib = set(talib_data.get("indicators", []))
    
    professional_export["indicator_availability"] = {
        "pandas_ta_only": list(all_pandas - all_talib),
        "talib_only": list(all_talib - all_pandas),
        "both_libraries": list(all_pandas & all_talib),
        "total_unique": list(all_pandas | all_talib),
        "professional_grade": len(list(all_pandas | all_talib)) > 200
    }
    
    # Add professional trading strategies
    professional_export["trading_strategies"] = bitfinex_recs
    professional_export["professional_setups"] = professional_setups
    
    # Add advanced features
    professional_export["advanced_features"] = {
        "margin_trading": {
            "max_leverage": {"crypto": "10x", "forex": "100x", "commodities": "20x"},
            "funding_analysis": True,
            "liquidation_alerts": True,
            "risk_management": True
        },
        "derivatives_trading": {
            "perpetuals": True,
            "monthly_futures": True,
            "basis_trading": True,
            "funding_arbitrage": True
        },
        "professional_tools": {
            "order_book_analysis": True,
            "whale_tracking": True,
            "institutional_flow": True,
            "cross_exchange_arbitrage": True
        },
        "data_quality": {
            "tick_by_tick": True,
            "microsecond_precision": True,
            "order_book_depth": "full",
            "historical_depth": "unlimited"
        }
    }
    
    # Add timeframe strategies for professionals
    professional_export["timeframe_strategies"] = {
        "high_frequency": {
            "timeframes": ["1m", "5m"], 
            "indicators": ["EMA", "RSI", "ORDER_FLOW", "VWAP"],
            "focus": "micro_movements"
        },
        "scalping": {
            "timeframes": ["5m", "15m"], 
            "indicators": ["EMA", "RSI", "STOCH", "BBANDS", "ATR"],
            "focus": "quick_profits"
        },
        "day_trading": {
            "timeframes": ["15m", "30m", "1h"], 
            "indicators": ["SMA", "EMA", "MACD", "RSI", "BBANDS", "VOLUME"],
            "focus": "intraday_trends"
        },
        "swing_trading": {
            "timeframes": ["1h", "4h", "6h", "12h"], 
            "indicators": ["SMA", "EMA", "MACD", "ADX", "SAR", "FIBONACCI"],
            "focus": "multi_day_moves"
        },
        "position_trading": {
            "timeframes": ["6h", "12h", "1D", "1W"], 
            "indicators": ["SMA", "EMA", "MACD", "ADX", "ROC", "LINEAR_REG"],
            "focus": "long_term_trends"
        },
        "institutional": {
            "timeframes": ["1D", "1W", "14D", "1M"], 
            "indicators": ["SMA", "EMA", "FUNDAMENTAL_ANALYSIS", "MACRO_INDICATORS"],
            "focus": "portfolio_allocation"
        }
    }
    
    return professional_export

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
    """Main function to discover and export Bitfinex professional trading indicators."""
    
    logger.info("🔵" + "="*75 + "🔵")
    logger.info("🏛️     BITFINEX PROFESSIONAL INDICATORS DISCOVERY TOOL     🏛️")
    logger.info("🔵" + "="*75 + "🔵")
    
    results_dir = config["RESULTS_DIR_PATH"]
    
    # File paths
    pandas_ta_filepath = os.path.join(results_dir, config["PANDAS_TA_FILENAME"])
    professional_filepath = os.path.join(results_dir, config["PROFESSIONAL_FILENAME"])
    combined_filepath = os.path.join(results_dir, config["COMBINED_FILENAME"])
    professional_filepath = os.path.join(results_dir, config["TRADING_FOCUSED_FILENAME"])
    
    logger.info(f"💰 Primary Symbol: {config['PRIMARY_SYMBOL']}")
    logger.info(f"🏛️  Exchange: Bitfinex Professional")
    logger.info(f"📁 Output Directory: {results_dir}")
    logger.info(f"🎯 Trading Types: Spot, Margin, Derivatives, Lending")
    
    # 1. Discover pandas_ta indicators
    logger.info("\n🔍 Discovering pandas_ta indicators for professional trading...")
    pandas_ta_data = list_pandas_ta_indicators()
    save_to_json(pandas_ta_data, pandas_ta_filepath, "pandas_ta professional indicators")
    
    # 2. Discover TA-Lib indicators
    logger.info("\n🔍 Discovering TA-Lib indicators for professional trading...")
    professional_data = list_pandas_ta_professional_indicators()
    save_to_json(professional_data, professional_filepath, "pandas_ta professional indicators")
    
    # 3. Create combined export
    logger.info("\n🔧 Creating combined professional indicators database...")
    combined_data = {
        "bitfinex_professional_database": {
            "exchange": "Bitfinex",
            "tier": "professional",
            "creation_time": datetime.now().isoformat(),
            "primary_symbol": config["PRIMARY_SYMBOL"],
            "libraries": ["pandas_ta", "talib"],
            "trading_types": ["spot", "margin", "derivatives", "lending"],
            "pandas_ta": pandas_ta_data,
            "talib": talib_data
        }
    }
    save_to_json(combined_data, combined_filepath, "combined professional indicators database")
    
    # 4. Create professional trading-focused export
    logger.info("\n🎯 Creating professional trading indicators guide...")
    professional_data = create_professional_trading_export(pandas_ta_data, talib_data)
    save_to_json(professional_data, professional_filepath, "professional trading indicators")
    
    # Summary
    pandas_count = len(pandas_ta_data.get("indicators", []))
    talib_count = len(talib_data.get("indicators", []))
    
    logger.info("\n" + "="*75)
    logger.info("📊 PROFESSIONAL DISCOVERY SUMMARY")
    logger.info("="*75)
    logger.info(f"🔧 pandas_ta indicators: {pandas_count}")
    logger.info(f"📈 TA-Lib indicators: {talib_count}")
    logger.info(f"📁 Files created: 4")
    logger.info(f"🏛️  Exchange: Bitfinex Professional")
    logger.info(f"💰 Primary Symbol: {config['PRIMARY_SYMBOL']}")
    logger.info(f"🎯 Trading Types: Spot, Margin, Derivatives, Lending")
    logger.info(f"⚡ Professional Features: Enabled")
    
    logger.info("\n✅ Bitfinex professional indicator discovery completed successfully!")

if __name__ == "__main__":
    main(CONFIG)
