#!/bin/bash
# Gordon CLI Launcher
# Sets PYTHONPATH and runs Gordon CLI
# Usage: bash run_gordon.sh [arguments]
# Example: bash run_gordon.sh --help
# Example: bash run_gordon.sh --dashboard

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up one level to get the parent directory (where gordon package lives)
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
# Set PYTHONPATH to the parent directory
export PYTHONPATH="$PARENT_DIR"
# Run Gordon CLI
python -m gordon.entrypoints.cli "$@"


