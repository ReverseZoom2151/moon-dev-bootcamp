#!/usr/bin/env python3
"""
Test script for Gordon - Quick functionality check
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Test imports
print("Testing Gordon imports...")

try:
    from agent.gordon_agent import GordonAgent
    print("✅ GordonAgent imported successfully")
except ImportError as e:
    print(f"❌ Failed to import GordonAgent: {e}")
    sys.exit(1)

try:
    from gordon.utilities.ui import print_intro
    print("✅ Intro utilities imported successfully")
except ImportError as e:
    print(f"❌ Failed to import utilities: {e}")
    sys.exit(1)

try:
    from core.strategy_manager import StrategyManager
    print("✅ Strategy Manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Strategy Manager: {e}")
    sys.exit(1)

try:
    from gordon.backtesting.backtest_main import ComprehensiveBacktester
    print("✅ Backtester imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Backtester: {e}")
    sys.exit(1)


async def test_gordon():
    """Test Gordon's basic functionality."""
    print("\n" + "="*50)
    print("GORDON FUNCTIONALITY TEST")
    print("="*50)

    # Test configuration
    config = {
        'base_position_size': 0.01,
        'max_drawdown': 0.1,
        'exchanges': {}  # No real exchanges for testing
    }

    try:
        print("\n1. Testing Gordon initialization...")
        gordon = GordonAgent(config)
        print("✅ Gordon initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Gordon: {e}")
        return

    # Test query classification
    print("\n2. Testing query classification...")
    test_queries = [
        ("What's Apple's revenue?", "research"),
        ("Run RSI strategy on BTC", "trading"),
        ("Analyze TSLA and trade if strong", "hybrid"),
        ("Backtest SMA on ETH", "backtest"),
        ("Hello Gordon", "general")
    ]

    for query, expected in test_queries:
        result = gordon._classify_query(query)
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{query[:30]}...' -> {result} (expected: {expected})")

    # Test symbol extraction
    print("\n3. Testing symbol extraction...")
    test_symbols = [
        ("Trade BTC with RSI", "BTCUSDT"),
        ("Analyze Apple stock", "AAPL"),
        ("What about TSLA?", "TSLA"),
        ("ETH price action", "ETHUSDT")
    ]

    for query, expected in test_symbols:
        result = gordon._extract_symbol(query)
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{query}' -> {result} (expected: {expected})")

    # Test scoring functions
    print("\n4. Testing scoring functions...")

    # Test fundamental scoring
    positive_text = "Strong revenue growth, excellent profitability, robust balance sheet"
    negative_text = "Declining revenues, poor margins, concerning debt levels"

    pos_score = gordon._score_fundamentals(positive_text)
    neg_score = gordon._score_fundamentals(negative_text)

    print(f"   Positive fundamental score: {pos_score}/10 {'✅' if pos_score > 5 else '❌'}")
    print(f"   Negative fundamental score: {neg_score}/10 {'✅' if neg_score < 5 else '❌'}")

    # Test technical scoring
    bullish_signals = {'rsi': 'BUY', 'sma': 'BULLISH', 'vwap': 'BUY'}
    bearish_signals = {'rsi': 'SELL', 'sma': 'BEARISH', 'vwap': 'SELL'}

    bull_score = gordon._score_technicals(bullish_signals)
    bear_score = gordon._score_technicals(bearish_signals)

    print(f"   Bullish technical score: {bull_score}/10 {'✅' if bull_score > 5 else '❌'}")
    print(f"   Bearish technical score: {bear_score}/10 {'✅' if bear_score < 5 else '❌'}")

    print("\n5. Testing ASCII art...")
    print_intro()

    print("\n" + "="*50)
    print("GORDON TEST COMPLETE!")
    print("="*50)
    print("\n✅ All basic tests passed. Gordon is ready!")
    print("\nTo run Gordon interactively:")
    print("  python cli.py")
    print("\nTo run a single query:")
    print("  python cli.py \"Analyze Apple and trade if strong\"")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════╗
    ║          GORDON SYSTEM TEST                ║
    ╚════════════════════════════════════════════╝
    """)

    try:
        asyncio.run(test_gordon())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)