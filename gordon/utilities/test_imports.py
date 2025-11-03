"""
Test script to verify all utility modules can be imported correctly.
Run this to ensure the refactoring maintains backward compatibility.
"""

import sys
from pathlib import Path

# Add parent directory to path for testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


def test_imports():
    """Test that all modules can be imported successfully."""

    print("Testing imports...")
    print("=" * 60)

    # Test individual module imports
    try:
        from gordon.utilities import data_utils
        print("[OK] data_utils imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import data_utils: {e}")

    try:
        from gordon.utilities import math_utils
        print("[OK] math_utils imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import math_utils: {e}")

    try:
        from gordon.utilities import time_utils
        print("[OK] time_utils imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import time_utils: {e}")

    try:
        from gordon.utilities import exchange_utils
        print("[OK] exchange_utils imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import exchange_utils: {e}")

    try:
        from gordon.utilities import signal_utils
        print("[OK] signal_utils imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import signal_utils: {e}")

    try:
        from gordon.utilities import format_utils
        print("[OK] format_utils imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import format_utils: {e}")

    try:
        from gordon.utilities import master_utils
        print("[OK] master_utils imported successfully (backward compatibility)")
    except Exception as e:
        print(f"[FAIL] Failed to import master_utils: {e}")

    print("=" * 60)

    # Test class imports
    print("\nTesting class imports...")
    print("=" * 60)

    try:
        from gordon.utilities import (
            DataUtils,
            MathUtils,
            TimeUtils,
            ExchangeUtils,
            SignalUtils,
            FormatUtils,
            MasterUtils
        )
        print("[OK] All utility classes imported successfully")
    except Exception as e:
        print(f"[FAIL] Failed to import classes: {e}")

    print("=" * 60)

    # Test that master_utils has access to all functions
    print("\nTesting master_utils function access...")
    print("=" * 60)

    try:
        from gordon.utilities import master_utils

        # Test a few key functions from each module
        functions_to_check = [
            'calculate_sma',           # from math_utils
            'calculate_rsi',           # from math_utils
            'detect_trend',            # from signal_utils
            'detect_engulfing',        # from signal_utils
            'calculate_position_size', # from exchange_utils
            'calculate_pnl',           # from exchange_utils
            'clean_ohlcv_data',        # from data_utils
            'resample_ohlcv',          # from data_utils
            'format_number',           # from format_utils
            'log_trade',               # from format_utils
        ]

        for func_name in functions_to_check:
            if hasattr(master_utils, func_name):
                print(f"[OK] master_utils.{func_name}() accessible")
            else:
                print(f"[FAIL] master_utils.{func_name}() NOT accessible")

    except Exception as e:
        print(f"[FAIL] Error checking master_utils functions: {e}")

    print("=" * 60)
    print("\n[OK] Import test complete!")


def test_basic_functionality():
    """Test basic functionality of key functions."""

    print("\n\nTesting basic functionality...")
    print("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        from gordon.utilities import (
            math_utils,
            signal_utils,
            exchange_utils,
            format_utils,
            time_utils
        )

        # Create sample data
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })

        # Test math_utils
        print("\nTesting math_utils...")
        sma = math_utils.calculate_sma(data, period=20)
        print(f"[OK] SMA calculation: {len(sma)} values")

        rsi = math_utils.calculate_rsi(data, period=14)
        print(f"[OK] RSI calculation: {len(rsi)} values")

        # Test signal_utils
        print("\nTesting signal_utils...")
        trend = signal_utils.detect_trend(data)
        print(f"[OK] Trend detection: {trend}")

        pattern = signal_utils.detect_engulfing(data)
        print(f"[OK] Pattern detection: {pattern}")

        # Test exchange_utils
        print("\nTesting exchange_utils...")
        position = exchange_utils.calculate_position_size(
            balance=10000,
            risk_percent=1.0,
            stop_loss_percent=2.0
        )
        print(f"[OK] Position size calculation: ${position}")

        pnl = exchange_utils.calculate_pnl(
            entry_price=100,
            exit_price=105,
            amount=10,
            side='buy'
        )
        print(f"[OK] PnL calculation: {pnl}")

        # Test format_utils
        print("\nTesting format_utils...")
        formatted = format_utils.format_number(1500000)
        print(f"[OK] Number formatting: {formatted}")

        price = format_utils.format_price(1234.56)
        print(f"[OK] Price formatting: {price}")

        # Test time_utils
        print("\nTesting time_utils...")
        timestamp = time_utils.get_current_timestamp()
        print(f"[OK] Current timestamp: {timestamp}")

        is_weekend = time_utils.is_weekend()
        print(f"[OK] Is weekend: {is_weekend}")

        print("\n" + "=" * 60)
        print("[OK] Functionality test complete!")

    except Exception as e:
        print(f"\n[FAIL] Error during functionality test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_imports()
    test_basic_functionality()
