@echo off
TITLE USMA Monitor Application Launcher

:: ============================================================================
::  USMA - Robust Application Launcher
::  This script ensures the environment is set up correctly before running.
::  It will:
::  1. Verify Python is installed.
::  2. Create a virtual environment ('sm_venv') if it doesn't exist.
::  3. Install all required packages from 'requirements.txt'.
::  4. Launch the main application.
:: ============================================================================

REM --- Step 1: Navigate to the script's directory ---
echo Navigating to script directory...
cd /d "%~dp0"

REM --- Step 2: Check for Python installation ---
echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found in the system PATH.
    echo Please install Python 3 and ensure "Add Python to PATH" is checked.
    pause
    exit /b
)
echo Python found!

REM --- Step 3: Check for and create the virtual environment ---
IF NOT EXIST "sm_venv" (
    echo [SETUP] Virtual environment 'sm_venv' not found. Creating it now...
    python -m venv sm_venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b
    )
    echo [SETUP] Virtual environment created successfully.
) ELSE (
    echo Virtual environment found.
)

REM --- Step 4: Activate the virtual environment ---
echo Activating Python virtual environment...
call "sm_venv\Scripts\activate.bat"

REM --- Step 5: Install dependencies from requirements.txt ---
IF NOT EXIST "requirements.txt" (
    echo [WARNING] requirements.txt not found. Cannot verify dependencies.
) ELSE (
    echo Checking and installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install required packages. Please check your internet connection.
        pause
        exit /b
    )
    echo All packages are up to date.
)

REM --- Step 6: Launch the main application ---
echo Launching the monitor application...
python monitor_app.py

echo.
echo Application closed.
pause
