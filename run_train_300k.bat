@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [train] .venv was not found. Run setup_venv.bat first.
    pause
    exit /b 1
)

echo [train] Starting PPO training for 300000 timesteps
".venv\Scripts\python.exe" train.py --timesteps 300000 --model-path models/ppo_lunarlander_300k
if errorlevel 1 goto error

echo.
echo [train] Done. Model should be saved at models\ppo_lunarlander_300k.zip
pause
exit /b 0

:error
echo.
echo [train] Failed. Check the console message above.
pause
exit /b 1
