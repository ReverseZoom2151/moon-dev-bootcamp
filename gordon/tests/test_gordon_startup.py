#!/usr/bin/env python3
"""
Test Gordon startup and basic initialization
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Gordon startup...")
print("=" * 50)

# Test basic imports
try:
    from agent import Agent
    print("[OK] Agent module imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import Agent: {e}")
    sys.exit(1)

# Check for OpenAI API key
if os.getenv('OPENAI_API_KEY'):
    print("[OK] OPENAI_API_KEY found")
else:
    print("[WARNING]  OPENAI_API_KEY not set - Gordon will need this to function")

# Try to initialize the agent
try:
    print("\nInitializing Gordon agent...")
    agent = Agent()
    print("[OK] Gordon agent initialized successfully!")

    # Show agent properties
    print(f"   - Max steps: {agent.max_steps}")
    print(f"   - Max steps per task: {agent.max_steps_per_task}")

except Exception as e:
    print(f"[ERROR] Failed to initialize agent: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("[OK] Gordon is ready to use!")
print("\nTo run Gordon interactively:")
print("  python cli.py")
print("\nTo run Gordon with a query:")
print("  python cli.py \"Analyze Apple's financial health\"")