#!/usr/bin/env python
"""
Demo Mode Launcher for SMA Trading Bot
This script launches the trading bot in demo mode to overcome API IP restrictions
"""

import os, sys, subprocess

print("\n=== SMA Trading Bot Demo Mode Launcher ===\n")
print("This will run the trading bot with simulated data")
print("No real trading will occur, but you can test all functionality\n")

# Find the path to the trading bot script
script_dir = os.path.dirname(os.path.abspath(__file__))
bot_script_path = os.path.join(script_dir, "6_sma.py")

# Check if the script exists
if not os.path.exists(bot_script_path):
    print(f"Error: Could not find 6_sma.py in {script_dir}")
    sys.exit(1)

print("Starting SMA trading bot in demo mode...\n")

# Launch the trading bot script with the demo flag
try:
    subprocess.run([sys.executable, bot_script_path, "--demo"])
except KeyboardInterrupt:
    print("\nDemo mode terminated by user")
except Exception as e:
    print(f"\nError running demo mode: {e}") 