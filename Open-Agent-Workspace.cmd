@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Follow the installation instructions in README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\open_agent_workspace.py
if errorlevel 1 pause
