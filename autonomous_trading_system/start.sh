#!/bin/bash

echo "============================================================"
echo "🚀 AUTONOMOUS TRADING SYSTEM - Linux/Mac Startup"
echo "============================================================"
echo "Starting your comprehensive trading system..."
echo "Based on Day_2 through Day_51 projects"
echo "============================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python $python_version - Compatible"

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Backend directory not found"
    echo "Please run this script from the autonomous_trading_system directory"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  No .env file found, creating from template..."
        cp .env.example .env
        echo "✅ Created .env file from template"
        echo "📝 Please edit .env file with your API keys before starting"
        echo "Opening .env file for editing..."
        
        # Try to open with common editors
        if command -v code &> /dev/null; then
            code .env
        elif command -v nano &> /dev/null; then
            nano .env
        elif command -v vim &> /dev/null; then
            vim .env
        else
            echo "Please edit .env file manually with your preferred editor"
        fi
        
        echo "Press Enter after editing .env file to continue..."
        read -r
    else
        echo "❌ No configuration files found"
        echo "Please ensure .env.example exists"
        exit 1
    fi
fi

# Check if virtual environment is recommended
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  No virtual environment detected"
    echo "Recommendation: Use a virtual environment"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please set up a virtual environment and try again:"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  ./start.sh"
        exit 1
    fi
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
if ! python3 -c "import fastapi, uvicorn, pandas, numpy" &> /dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ Core dependencies found"
fi

echo "✅ All checks passed! Starting system..."
echo ""
echo "🚀 Starting Autonomous Trading System..."
echo "   Backend will be available at: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the system"
echo "============================================================"

# Change to backend directory and start
cd backend

# Start with uvicorn
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

echo ""
echo "🛑 System stopped" 