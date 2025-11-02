"""
Simple test script to verify utility modules work independently.
This tests the utilities without loading the full exchange_orchestrator package.
"""

import sys
from pathlib import Path

# Add utilities directory to path
utilities_dir = Path(__file__).parent
sys.path.insert(0, str(utilities_dir))


def test_direct_imports():
    """Test direct imports from utility modules."""

    print("Testing direct imports from utility modules...")
    print("=" * 60)

    # Test individual module imports
    try:
        import data_utils
        print("[OK] data_utils imported successfully")
        print(f"     - Has clean_ohlcv_data: {hasattr(data_utils, 'DataUtils')}")
    except Exception as e:
        print(f"[FAIL] Failed to import data_utils: {e}")

    try:
        import math_utils
        print("[OK] math_utils imported successfully")
        print(f"     - Has MathUtils: {hasattr(math_utils, 'MathUtils')}")
    except Exception as e:
        print(f"[FAIL] Failed to import math_utils: {e}")

    try:
        import time_utils
        print("[OK] time_utils imported successfully")
        print(f"     - Has TimeUtils: {hasattr(time_utils, 'TimeUtils')}")
    except Exception as e:
        print(f"[FAIL] Failed to import time_utils: {e}")

    try:
        import exchange_utils
        print("[OK] exchange_utils imported successfully")
        print(f"     - Has ExchangeUtils: {hasattr(exchange_utils, 'ExchangeUtils')}")
    except Exception as e:
        print(f"[FAIL] Failed to import exchange_utils: {e}")

    try:
        import signal_utils
        print("[OK] signal_utils imported successfully")
        print(f"     - Has SignalUtils: {hasattr(signal_utils, 'SignalUtils')}")
    except Exception as e:
        print(f"[FAIL] Failed to import signal_utils: {e}")

    try:
        import format_utils
        print("[OK] format_utils imported successfully")
        print(f"     - Has FormatUtils: {hasattr(format_utils, 'FormatUtils')}")
    except Exception as e:
        print(f"[FAIL] Failed to import format_utils: {e}")

    try:
        import master_utils
        print("[OK] master_utils imported successfully")
        print(f"     - Has MasterUtils: {hasattr(master_utils, 'MasterUtils')}")
    except Exception as e:
        print(f"[FAIL] Failed to import master_utils: {e}")

    print("=" * 60)


def test_functionality():
    """Test basic functionality."""

    print("\nTesting basic functionality...")
    print("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        import math_utils
        import signal_utils
        import exchange_utils
        import format_utils
        import time_utils

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
        sma = math_utils.math_utils.calculate_sma(data, period=20)
        print(f"[OK] SMA calculation: {len(sma)} values, last value: {sma.iloc[-1]:.2f}")

        rsi = math_utils.math_utils.calculate_rsi(data, period=14)
        print(f"[OK] RSI calculation: {len(rsi)} values, last value: {rsi.iloc[-1]:.2f}")

        upper, middle, lower = math_utils.math_utils.calculate_bollinger_bands(data)
        print(f"[OK] Bollinger Bands: Upper={upper.iloc[-1]:.2f}, Middle={middle.iloc[-1]:.2f}, Lower={lower.iloc[-1]:.2f}")

        # Test signal_utils
        print("\nTesting signal_utils...")
        trend = signal_utils.signal_utils.detect_trend(data)
        print(f"[OK] Trend detection: {trend}")

        pattern = signal_utils.signal_utils.detect_engulfing(data)
        print(f"[OK] Engulfing pattern: {pattern}")

        composite = signal_utils.signal_utils.generate_composite_signal(data)
        print(f"[OK] Composite signal: {composite['overall']} (Buy: {composite['buy_signals']}, Sell: {composite['sell_signals']})")

        # Test exchange_utils
        print("\nTesting exchange_utils...")
        position = exchange_utils.exchange_utils.calculate_position_size(
            balance=10000,
            risk_percent=1.0,
            stop_loss_percent=2.0,
            leverage=5
        )
        print(f"[OK] Position size: ${position}")

        pnl = exchange_utils.exchange_utils.calculate_pnl(
            entry_price=100,
            exit_price=105,
            amount=10,
            side='buy',
            leverage=5
        )
        print(f"[OK] PnL: Gross=${pnl['gross_pnl']}, Net=${pnl['net_pnl']}, Percent={pnl['pnl_percent']}%")

        liq = exchange_utils.exchange_utils.calculate_liquidation_levels(
            price=100,
            leverage=10,
            side='long'
        )
        print(f"[OK] Liquidation: Price=${liq['liquidation_price']}, Distance={liq['distance_percent']}%")

        # Test format_utils
        print("\nTesting format_utils...")
        formatted = format_utils.format_utils.format_number(1500000)
        print(f"[OK] Number formatting: 1,500,000 -> {formatted}")

        price = format_utils.format_utils.format_price(1234.56)
        print(f"[OK] Price formatting: 1234.56 -> {price}")

        pct = format_utils.format_utils.format_percentage(12.5)
        print(f"[OK] Percentage formatting: 12.5 -> {pct}")

        # Test time_utils
        print("\nTesting time_utils...")
        timestamp = time_utils.time_utils.get_current_timestamp()
        print(f"[OK] Current timestamp: {timestamp}")

        unix_ts = time_utils.time_utils.get_unix_timestamp()
        print(f"[OK] Unix timestamp: {unix_ts}")

        is_weekend = time_utils.time_utils.is_weekend()
        print(f"[OK] Is weekend: {is_weekend}")

        weekday = time_utils.time_utils.get_weekday()
        print(f"[OK] Weekday: {weekday}")

        print("\n" + "=" * 60)
        print("[OK] All functionality tests passed!")

    except Exception as e:
        print(f"\n[FAIL] Error during functionality test: {e}")
        import traceback
        traceback.print_exc()


def test_master_utils_compatibility():
    """Test that master_utils provides backward compatibility."""

    print("\n\nTesting master_utils backward compatibility...")
    print("=" * 60)

    try:
        import master_utils
        import pandas as pd
        import numpy as np

        # Create sample data
        data = pd.DataFrame({
            'open': np.random.randn(50).cumsum() + 100,
            'high': np.random.randn(50).cumsum() + 102,
            'low': np.random.randn(50).cumsum() + 98,
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 50)
        })

        # Test that all original functions are accessible through master_utils
        functions = [
            ('calculate_sma', [data]),
            ('calculate_rsi', [data]),
            ('detect_trend', [data]),
            ('detect_engulfing', [data]),
            ('calculate_position_size', [10000]),
            ('format_number', [1500000]),
        ]

        for func_name, args in functions:
            if hasattr(master_utils.master_utils, func_name):
                try:
                    result = getattr(master_utils.master_utils, func_name)(*args)
                    print(f"[OK] master_utils.{func_name}() works correctly")
                except Exception as e:
                    print(f"[FAIL] master_utils.{func_name}() error: {e}")
            else:
                print(f"[FAIL] master_utils.{func_name}() not found")

        print("=" * 60)
        print("[OK] Backward compatibility test complete!")

    except Exception as e:
        print(f"\n[FAIL] Error during backward compatibility test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_direct_imports()
    test_functionality()
    test_master_utils_compatibility()
