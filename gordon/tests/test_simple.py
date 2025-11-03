#!/usr/bin/env python3
"""
Simple test to verify Gordon's basic functionality
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

print("Gordon System Test")
print("=" * 50)

# Test imports
print("\n1. Testing imports...")

try:
    from agent import Agent
    print("✅ Agent imported")
except ImportError as e:
    print(f"❌ Failed to import Agent: {e}")

try:
    from gordon.utilities.ui import print_intro
    print("✅ Utils imported")
except ImportError as e:
    print(f"❌ Failed to import utils: {e}")

try:
    from tools import TOOLS
    print(f"✅ Tools imported ({len(TOOLS)} tools available)")
except ImportError as e:
    print(f"❌ Failed to import tools: {e}")

# Check API keys
print("\n2. Checking API keys...")

if os.getenv('OPENAI_API_KEY'):
    print("✅ OPENAI_API_KEY found")
else:
    print("❌ OPENAI_API_KEY not found - required for AI features")

if os.getenv('FINANCIAL_DATASETS_API_KEY'):
    print("✅ FINANCIAL_DATASETS_API_KEY found")
else:
    print("⚠️ FINANCIAL_DATASETS_API_KEY not found - some features may be limited")

# Test Agent initialization
print("\n3. Testing Agent initialization...")

try:
    from agent import Agent
    agent = Agent()
    print("✅ Agent initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Agent: {e}")

# Show available tools
print("\n4. Available tools:")
try:
    from tools import TOOLS
    for i, tool in enumerate(TOOLS, 1):
        print(f"   {i}. {tool.__name__}")
except Exception as e:
    print(f"❌ Could not list tools: {e}")

print("\n" + "=" * 50)
print("Test complete!")
print("\nTo run Gordon, use:")
print("  python -m gordon.entrypoints.cli")
print("\nMake sure you have set OPENAI_API_KEY in .env file")