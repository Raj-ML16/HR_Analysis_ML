@echo off
REM HR Predictive Analytics Dashboard Launcher for Windows
REM This batch file starts the complete application

title HR Predictive Analytics Dashboard

echo.
echo ========================================
echo  HR Predictive Analytics Dashboard
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH!
    echo Please install Python 3.7+ and try again.
    pause
    exit /b 1
)

REM Check if main.py exists
if not exist "main.py" (
    echo main.py not found!
    echo Please ensure the FastAPI server file is in this directory.
    pause
    exit /b 1
)

REM Check if dashboard file exists - prioritize web_interface.html since that's what user has
if exist "web_interface.html" (
    set DASHBOARD_FILE=web_interface.html
) else if exist "dashboard.html" (
    set DASHBOARD_FILE=dashboard.html
) else (
    echo Dashboard HTML file not found!
    echo Please ensure web_interface.html or dashboard.html is in this directory.
    pause
    exit /b 1
)

echo Python found
echo FastAPI server found
echo Dashboard file found: %DASHBOARD_FILE%
echo.

echo Starting HR Analytics Dashboard...
echo Dashboard will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.
echo Keep this window open while using the dashboard
echo Press Ctrl+C to stop the server
echo.

REM Start the FastAPI server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

echo.
echo Dashboard stopped. Press any key to exit...
pause >nul