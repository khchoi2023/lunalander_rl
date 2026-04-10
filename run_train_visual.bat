@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [train_visual] .venv was not found. Run setup_venv.bat first.
    pause
    exit /b 1
)

echo [train_visual] Starting visual PPO training for 100000 timesteps
echo [train_visual] A GUI demo opens after each of the first 10 completed training episodes
".venv\Scripts\python.exe" train_visual.py --timesteps 100000 --render-first-episodes 10 --render-every-episodes 0
if errorlevel 1 goto error

echo.
echo [train_visual] Done. No model was saved.
pause
exit /b 0

:error
echo.
echo [train_visual] Failed. Check the console message above.
pause
exit /b 1
