@echo off
setlocal
cd /d "%~dp0"
echo FreshLens - first-time setup
py -3.11 --version >nul 2>&1
if errorlevel 1 goto missing_python
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if not exist ".venv\Scripts\python.exe" goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" check_model.py
if errorlevel 1 goto failed
echo.
echo Setup complete. Open run_windows.bat to start FreshLens.
pause
exit /b 0
:missing_python
echo Install Python 3.11 64-bit from python.org, including the Python launcher.
echo Then run this file again. See README.md for the download link.
pause
exit /b 1
:failed
echo Setup did not complete. Copy the error above for troubleshooting.
pause
exit /b 1
