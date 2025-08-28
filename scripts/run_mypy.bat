@echo off
REM Mypy type checking script for the collectivecrossing project

echo 🔍 Running mypy type checking...

REM Get the project root directory
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set SRC_DIR=%PROJECT_ROOT%\src
set TEST_DIR=%PROJECT_ROOT%\tests

echo 📁 Project root: %PROJECT_ROOT%
echo 📁 Source directory: %SRC_DIR%
echo --------------------------------------------------

REM Check if mypy is installed
python -m mypy --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: mypy not found. Please install it with:
    echo    uv add --dev mypy
    exit /b 1
)

REM Build the mypy command
set MYPY_CMD=python -m mypy --config-file %PROJECT_ROOT%\pyproject.toml %SRC_DIR%

REM Add test directory if it exists
if exist "%TEST_DIR%" (
    set MYPY_CMD=%MYPY_CMD% %TEST_DIR%
)

echo 🚀 Command: %MYPY_CMD%
echo.

REM Run mypy
cd /d "%PROJECT_ROOT%"
%MYPY_CMD%
if errorlevel 1 (
    echo --------------------------------------------------
    echo ❌ Mypy type checking failed!
    exit /b 1
) else (
    echo --------------------------------------------------
    echo ✅ Mypy type checking passed!
    exit /b 0
)
