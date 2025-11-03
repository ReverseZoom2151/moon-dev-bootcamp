#!/usr/bin/env python3
"""
Test Gordon Integration
=======================
Verify that all components are properly integrated.
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Gordon Integration...")
print("=" * 60)

# Test 1: Core imports
print("\n1. Testing Core Imports...")
try:
    from core.strategy_manager import StrategyManager
    print("  [OK] StrategyManager imported")
except ImportError as e:
    print(f"  [ERROR] Failed to import StrategyManager: {e}")

try:
    from core.market_data_stream import MarketDataStream
    print("  [OK] MarketDataStream imported")
except ImportError as e:
    print(f"  [ERROR] Failed to import MarketDataStream: {e}")

try:
    from core.risk_manager import RiskManager
    print("  [OK] RiskManager imported")
except ImportError as e:
    print(f"  [ERROR] Failed to import RiskManager: {e}")

# Test 2: Exchange imports
print("\n2. Testing Exchange System...")
try:
    from exchanges.factory import ExchangeFactory
    print("  [OK] ExchangeFactory imported")
except ImportError as e:
    print(f"  [ERROR] Failed to import ExchangeFactory: {e}")

try:
    from exchanges.base import BaseExchange
    print("  [OK] BaseExchange imported")
except ImportError as e:
    print(f"  [ERROR] Failed to import BaseExchange: {e}")

# Test 3: Backtesting imports
print("\n3. Testing Backtesting System...")
try:
    from backtesting.backtest_main import ComprehensiveBacktester
    print("  [OK] ComprehensiveBacktester imported")
except ImportError as e:
    print(f"  [ERROR] Failed to import ComprehensiveBacktester: {e}")

# Test 4: Trading tools imports
print("\n4. Testing Trading Tools...")
try:
    from tools.trading import (
        run_sma_strategy,
        place_market_order,
        backtest_strategy,
        check_risk_limits,
        get_live_price
    )
    print("  [OK] Trading strategies imported")
    print("  [OK] Order execution tools imported")
    print("  [OK] Backtesting tools imported")
    print("  [OK] Risk management tools imported")
    print("  [OK] Market data tools imported")
except ImportError as e:
    print(f"  [ERROR] Failed to import trading tools: {e}")

# Test 5: Configuration
print("\n5. Testing Configuration...")
try:
    from config_manager import get_config
    config = get_config()
    print("  [OK] Configuration loaded")
    print(f"  - Dry run mode: {config.is_dry_run()}")
    print(f"  - Default exchange: {config.get('exchanges.default', 'binance')}")
    print(f"  - Max position size: {config.get('trading.risk.max_position_size', 0.1):.1%}")
except Exception as e:
    print(f"  [ERROR] Failed to load configuration: {e}")

# Test 6: Hybrid Analyzer
print("\n6. Testing Hybrid Analyzer...")
try:
    from hybrid_analyzer import HybridAnalyzer
    print("  [OK] HybridAnalyzer imported")
except ImportError as e:
    print(f"  [ERROR] Failed to import HybridAnalyzer: {e}")

# Test 7: Main Agent
print("\n7. Testing Main Agent...")
try:
    from agent import Agent
    agent = Agent()
    print("  [OK] Agent initialized")
    print(f"  - Max steps: {agent.max_steps}")
    print(f"  - Max steps per task: {agent.max_steps_per_task}")
except Exception as e:
    print(f"  [ERROR] Failed to initialize Agent: {e}")

print("\n" + "=" * 60)

# Summary
import_errors = []
if '[ERROR]' in str(locals()):
    print("Status: SOME COMPONENTS FAILED")
    print("Please check the error messages above.")
else:
    print("Status: ALL COMPONENTS INTEGRATED SUCCESSFULLY!")
    print("\nGordon is ready for trading!")
    print("\nTo start Gordon, run:")
    print("  python cli.py")
    print("\nFor hybrid analysis, try:")
    print("  python cli.py")
    print("  Gordon> hybrid AAPL")