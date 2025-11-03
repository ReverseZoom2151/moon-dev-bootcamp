"""
Gordon - Financial Research & Trading Agent
============================================
Combines fundamental analysis with technical trading.
"""

__version__ = "1.0.0"
__author__ = "Gordon Team"

# Make imports work properly
import sys
from pathlib import Path

# Add gordon directory to path for imports
gordon_path = Path(__file__).parent
if str(gordon_path) not in sys.path:
    sys.path.insert(0, str(gordon_path))