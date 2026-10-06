@echo off
echo ==========================================
echo  CFB ALGO v2 -- Retrain + Backtest
echo ==========================================
cd /d "%~dp0"
python features.py
python model.py
python backtest.py
echo Done!
pause
