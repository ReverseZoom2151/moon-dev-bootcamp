@echo off
REM Gordon CLI Launcher
REM Sets PYTHONPATH and runs Gordon CLI
REM Usage: run_gordon.bat [arguments]
REM Example: run_gordon.bat --help
REM Example: run_gordon.bat --dashboard

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
REM Go up one level to get the parent directory (where gordon package lives)
set "PARENT_DIR=%SCRIPT_DIR%.."
REM Set PYTHONPATH to the parent directory
set "PYTHONPATH=%PARENT_DIR%"
REM Run Gordon CLI
python -m gordon.entrypoints.cli %*
