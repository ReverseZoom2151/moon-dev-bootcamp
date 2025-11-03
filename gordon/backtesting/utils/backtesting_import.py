"""
Helper module for importing the external backtesting package.
This avoids namespace conflicts with gordon.backtesting.
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

# Cache the external backtesting module
_external_backtesting = None
_external_backtesting_lib = None
_external_backtesting_test = None


def get_backtesting_module():
    """
    Import the external backtesting package, avoiding conflicts with gordon.backtesting.
    
    Returns:
        Tuple of (backtesting module, backtesting.lib module, backtesting.test module) or (None, None, None) if not available
    """
    global _external_backtesting, _external_backtesting_lib, _external_backtesting_test
    
    if _external_backtesting is not None:
        return _external_backtesting, _external_backtesting_lib, _external_backtesting_test
    
    # Temporarily remove gordon directory from path to import external backtesting
    import os
    current_file = Path(__file__).resolve()
    # Find the gordon directory (parent of backtesting/utils)
    gordon_dir = current_file.parent.parent.parent
    
    # Remove gordon directory from sys.path temporarily
    gordon_dir_str = str(gordon_dir)
    removed_from_path = False
    if gordon_dir_str in sys.path:
        sys.path.remove(gordon_dir_str)
        removed_from_path = True
    
    try:
        # Try to import the external backtesting package
        import backtesting as bt
        from backtesting import lib as bt_lib
        try:
            from backtesting import test as bt_test
        except ImportError:
            bt_test = None
        _external_backtesting = bt
        _external_backtesting_lib = bt_lib
        _external_backtesting_test = bt_test
        return bt, bt_lib, bt_test
    except ImportError:
        # External package not installed
        _external_backtesting = None
        _external_backtesting_lib = None
        _external_backtesting_test = None
        return None, None, None
    finally:
        # Restore gordon directory to sys.path if we removed it
        if removed_from_path and gordon_dir_str not in sys.path:
            sys.path.insert(0, gordon_dir_str)


def get_backtesting_strategy():
    """
    Get the Strategy class from the external backtesting package.
    
    Returns:
        Strategy class or None if not available
    """
    bt_module, _, _ = get_backtesting_module()
    if bt_module is None:
        return None
    return getattr(bt_module, 'Strategy', None)


def get_backtesting_crossover():
    """
    Get the crossover function from the external backtesting.lib package.
    
    Returns:
        crossover function or None if not available
    """
    _, bt_lib, _ = get_backtesting_module()
    if bt_lib is None:
        return None
    return getattr(bt_lib, 'crossover', None)


def get_backtesting_trailing_strategy():
    """
    Get the TrailingStrategy class from the external backtesting.lib package.
    
    Returns:
        TrailingStrategy class or None if not available
    """
    _, bt_lib, _ = get_backtesting_module()
    if bt_lib is None:
        return None
    return getattr(bt_lib, 'TrailingStrategy', None)


def get_backtesting_sma():
    """
    Get the SMA function from the external backtesting.test package.
    
    Returns:
        SMA function or None if not available
    """
    _, _, bt_test = get_backtesting_module()
    if bt_test is None:
        return None
    return getattr(bt_test, 'SMA', None)

