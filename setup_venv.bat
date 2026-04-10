@echo off
setlocal
cd /d "%~dp0"

echo [setup] Creating virtual environment at .venv
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto error
) else (
    echo [setup] Existing .venv found
)

echo [setup] Upgrading pip
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error

echo [setup] Installing requirements
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [setup] Done. Next, double-click one of these files:
echo [setup]   run_train_100k.bat
echo [setup]   run_train_300k.bat
echo [setup]   run_train_visual.bat
echo [setup]   run_evaluate.bat
echo [setup]   run_play_gui.bat
echo [setup]   run_play_gui_300k.bat
pause
exit /b 0

:error
echo.
echo [setup] Failed. If the error mentions Box2D, box2d-py, or SWIG, try:
echo [setup]   ".venv\Scripts\python.exe" -m pip install swig
echo [setup]   ".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
pause
exit /b 1
