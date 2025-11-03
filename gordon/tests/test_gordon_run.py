#!/usr/bin/env python3
"""
Test Gordon Run
===============
Quick test to verify Gordon starts and can process a simple query.
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Gordon Startup...")
print("=" * 60)

try:
    # Import and initialize the agent
    from agent import Agent
    agent = Agent()
    print("[OK] Gordon Agent initialized successfully")

    # Test a simple query
    print("\nTesting simple query processing...")
    print("Query: 'Hello Gordon'")

    # Note: Agent.run() doesn't return a value, it prints directly
    agent.run("Hello Gordon, what can you do?")

    print("\n" + "=" * 60)
    print("[SUCCESS] Gordon is working!")

except Exception as e:
    print(f"[ERROR] Failed to run Gordon: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)