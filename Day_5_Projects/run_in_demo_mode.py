#!/usr/bin/env python
"""
Demo Mode Launcher for Phemex Risk Management Tool
This script launches the risk management tool in demo mode to overcome API IP restrictions
"""

import os, sys, subprocess

print("\n=== Phemex Risk Management Demo Mode Launcher ===\n")
print("This will run the risk management tool with simulated data")
print("No real trading will occur, but you can test all functionality\n")

# Find the path to the risk management script
script_dir = os.path.dirname(os.path.abspath(__file__))
risk_script_path = os.path.join(script_dir, "5_risk.py")

# Check if the script exists
if not os.path.exists(risk_script_path):
    print(f"Error: Could not find 5_risk.py in {script_dir}")
    sys.exit(1)

print("Starting risk management tool in demo mode...\n")

# Launch the risk management script with the demo flag
try:
    subprocess.run([sys.executable, risk_script_path, "--demo"])
except KeyboardInterrupt:
    print("\nDemo mode terminated by user")
except Exception as e:
    print(f"\nError running demo mode: {e}") 