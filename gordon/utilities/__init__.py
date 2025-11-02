"""
Exchange Orchestrator Utilities Package
=======================================
Comprehensive utilities for trading operations, organized into focused modules.

Available Modules:
- data_utils: Data manipulation and formatting
- math_utils: Mathematical calculations and indicators
- time_utils: Time and date utilities
- exchange_utils: Exchange-specific utilities
- signal_utils: Trading signal and pattern detection
- format_utils: Formatting and display utilities
- master_utils: Consolidated utilities (backward compatible)

Quick Import Examples:
    from exchange_orchestrator.utilities import master_utils
    from exchange_orchestrator.utilities import data_utils, math_utils
    from exchange_orchestrator.utilities.signal_utils import signal_utils
"""

# Import all utility modules for easy access
from .data_utils import data_utils, DataUtils
from .math_utils import math_utils, MathUtils
from .time_utils import time_utils, TimeUtils
from .exchange_utils import exchange_utils, ExchangeUtils
from .signal_utils import signal_utils, SignalUtils
from .format_utils import format_utils, FormatUtils
from .master_utils import master_utils, MasterUtils

__all__ = [
    # Singleton instances
    'master_utils',
    'data_utils',
    'math_utils',
    'time_utils',
    'exchange_utils',
    'signal_utils',
    'format_utils',
    # Classes
    'MasterUtils',
    'DataUtils',
    'MathUtils',
    'TimeUtils',
    'ExchangeUtils',
    'SignalUtils',
    'FormatUtils',
]

__version__ = '1.0.0'
