@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [gui] .venv was not found. Run setup_venv.bat first.
    pause
    exit /b 1
)

if not exist "models\ppo_lunarlander.zip" (
    echo [gui] models\ppo_lunarlander.zip was not found. Run run_train_100k.bat first.
    pause
    exit /b 1
)

echo [gui] Opening LunarLander-v3 human renderer
".venv\Scripts\python.exe" play_gui.py --episodes 3
if errorlevel 1 goto error

echo.
echo [gui] Done.
pause
exit /b 0

:error
echo.
echo [gui] Failed. Check the console message above.
pause
exit /b 1
