@echo off
echo ==========================================
echo  CFB ALGO v2 -- Daily Run
echo ==========================================
cd /d "%~dp0"

set /p WEEK="Enter week number (or press Enter for all): "

echo.
echo [1/3] Fetching latest data...
python fetch_cfb.py
if errorlevel 1 goto error

echo.
echo [2/3] Building features...
python features.py
if errorlevel 1 goto error

echo.
echo [3/3] Running predictions...
if "%WEEK%"=="" (
    python predict.py
) else (
    python predict.py %WEEK%
)
if errorlevel 1 goto error

echo.
echo Done! Check cfb_predictions.xlsx
pause
exit /b 0
:error
echo !! Error above
pause
exit /b 1
