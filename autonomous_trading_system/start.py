#!/usr/bin/env python3
"""
🚀 Autonomous Trading System - Easy Startup Script
Run this script to start your trading system with guided setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🚀 AUTONOMOUS TRADING SYSTEM")
    print("=" * 60)
    print("Starting your comprehensive trading system...")
    print("Based on Day_2 through Day_51 projects")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_virtual_environment():
    """Check if virtual environment is active"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️  No virtual environment detected")
        print("   Recommendation: Use a virtual environment")
        response = input("   Continue anyway? (y/N): ").lower()
        return response == 'y'

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Installing dependencies...")
        return install_dependencies()

def install_dependencies():
    """Install required dependencies"""
    try:
        requirements_file = Path("requirements.txt")
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        print("📦 Installing dependencies...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_configuration():
    """Check if configuration file exists"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✅ Configuration file (.env) found")
        return True
    elif env_example.exists():
        print("⚠️  No .env file found, but .env.example exists")
        response = input("   Create .env from .env.example? (Y/n): ").lower()
        if response != 'n':
            try:
                import shutil
                shutil.copy(env_example, env_file)
                print("✅ Created .env file from template")
                print("📝 Please edit .env file with your API keys before starting")
                return True
            except Exception as e:
                print(f"❌ Failed to create .env file: {e}")
                return False
    else:
        print("❌ No configuration files found")
        print("   Please ensure .env.example exists")
        return False

def start_system():
    """Start the trading system"""
    try:
        backend_dir = Path("backend")
        if not backend_dir.exists():
            print("❌ Backend directory not found")
            print("   Please run this script from the autonomous_trading_system directory")
            return False
        
        print("🚀 Starting Autonomous Trading System...")
        print("   Backend will be available at: http://localhost:8080")
        print("   API Documentation: http://localhost:8080/docs")
        print("   Health Check: http://localhost:8080/health")
        print("")
        print("Press Ctrl+C to stop the system")
        print("=" * 60)
        
        # Change to backend directory and start
        os.chdir(backend_dir)
        
        # Start with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        return False

def main():
    """Main startup function"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_virtual_environment():
        sys.exit(1)
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_configuration():
        print("\n📝 Configuration Setup Required:")
        print("   1. Edit the .env file with your API keys")
        print("   2. At minimum, set BIRDEYE_KEY for basic functionality")
        print("   3. Run this script again to start the system")
        sys.exit(1)
    
    # Start the system
    print("\n🎯 All checks passed! Starting system...")
    start_system()

if __name__ == "__main__":
    main() 