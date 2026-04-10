@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [gui] .venv was not found. Run setup_venv.bat first.
    pause
    exit /b 1
)

if not exist "models\ppo_lunarlander_300k.zip" (
    echo [gui] models\ppo_lunarlander_300k.zip was not found. Run run_train_300k.bat first.
    pause
    exit /b 1
)

echo [gui] Opening LunarLander-v3 human renderer with the 300k model
".venv\Scripts\python.exe" play_gui.py --model-path models/ppo_lunarlander_300k.zip --episodes 3
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
