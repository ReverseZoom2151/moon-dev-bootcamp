"""
Gordon Package Entry Point
==========================
Allows running: python -m gordon
"""

import sys
from pathlib import Path

# Add parent directory to path so gordon package can be found
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import and run CLI
from gordon.entrypoints.cli import main

if __name__ == "__main__":
    main()

