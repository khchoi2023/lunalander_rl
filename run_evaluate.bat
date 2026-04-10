@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [eval] .venv was not found. Run setup_venv.bat first.
    pause
    exit /b 1
)

if not exist "models\ppo_lunarlander.zip" (
    echo [eval] models\ppo_lunarlander.zip was not found. Run run_train_100k.bat first.
    pause
    exit /b 1
)

echo [eval] Evaluating saved model for 10 episodes
".venv\Scripts\python.exe" evaluate.py --episodes 10
if errorlevel 1 goto error

echo.
echo [eval] Done.
pause
exit /b 0

:error
echo.
echo [eval] Failed. Check the console message above.
pause
exit /b 1
