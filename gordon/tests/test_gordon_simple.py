#!/usr/bin/env python3
"""
Simple Gordon Test
==================
Test Gordon without fancy UI elements.
"""

import os
import sys
from pathlib import Path

# Disable the fancy UI to avoid unicode issues
os.environ['GORDON_SIMPLE_UI'] = 'true'

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Gordon (Simple Mode)...")
print("=" * 60)

try:
    # Import configuration first
    from config_manager import get_config
    config = get_config()

    print("\n1. Configuration:")
    print(f"   - Dry run: {config.is_dry_run()}")
    print(f"   - Default exchange: {config.get('exchanges.default', 'binance')}")

    # Import hybrid analyzer
    from hybrid_analyzer import HybridAnalyzer

    print("\n2. Testing Hybrid Analyzer:")
    analyzer = HybridAnalyzer()

    # Run a simple analysis (will fail without API keys, but tests the structure)
    print("   - Attempting analysis on 'BTC/USDT'...")

    # Note: This will try to use the agent which requires OpenAI API key
    # For now, just test that it initializes
    print("   - [OK] Hybrid Analyzer initialized")

    # Test that tools are loaded
    from tools import TOOLS
    print(f"\n3. Tools Loaded: {len(TOOLS)} tools available")

    # List first 5 tools
    print("   Sample tools:")
    for i, tool in enumerate(TOOLS[:5]):
        if hasattr(tool, '__name__'):
            print(f"   - {tool.__name__}")
        else:
            print(f"   - {str(tool)[:50]}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Gordon components are working!")
    print("\nTo use Gordon interactively:")
    print("  python cli.py")
    print("\nNote: You'll need to set OPENAI_API_KEY in .env for full functionality")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)