@echo off
echo ============================================================
echo 🚀 AUTONOMOUS TRADING SYSTEM - Windows Startup
echo ============================================================
echo Starting your comprehensive trading system...
echo Based on Day_2 through Day_51 projects
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "backend\main.py" (
    echo ❌ Backend directory not found
    echo Please run this script from the autonomous_trading_system directory
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    if exist ".env.example" (
        echo ⚠️ No .env file found, creating from template...
        copy ".env.example" ".env"
        echo ✅ Created .env file from template
        echo 📝 Please edit .env file with your API keys before starting
        echo Opening .env file for editing...
        notepad .env
        echo Press any key after editing .env file to continue...
        pause
    ) else (
        echo ❌ No configuration files found
        echo Please ensure .env.example exists
        pause
        exit /b 1
    )
)

REM Install dependencies if needed
echo 📦 Checking dependencies...
python -c "import fastapi, uvicorn, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✅ All checks passed! Starting system...
echo.
echo 🚀 Starting Autonomous Trading System...
echo    Backend will be available at: http://localhost:8000
echo    API Documentation: http://localhost:8000/docs
echo    Health Check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the system
echo ============================================================

REM Change to backend directory and start
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

echo.
echo 🛑 System stopped
pause 